#!/usr/bin/env bash
# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0
# ==============================================================================
# Turnkey Relauncher — Pass 3: SGLang-TorchTPU Day-0 (TP=32, EP=8) on TPU v7x
# ==============================================================================
# Launches the SGLang-TorchTPU (origin/kimi-k31-pro-tpu) Day-0 serving recipe
# or single-stream Decode MegaKernel + Speculative Decoding across a 4-host
# TPU v7x 2x2x4 slice (16 physical chips = 32 TensorCores):
#   1. TP=32, EP=8 (896 MXFP4 experts sharded 8-way EP x 4-way TP intermediate)
#   2. SparseCore DSA Indexer (pallas_tpu_mla_sparse, index_topk=2048, FP8 KV cache)
#   3. Ironwood SparseCore Gather Offload (--xla_tpu_offload_gather_to_sparsecore=true)
#   4. DSA Indexer Token-Split across TP32 (SGLANG_TPU_DSA_INDEXER_TOKEN_SPLIT=1)
#   5. Per-Op Pallas Decode Kernels (KDA, MLA, MoE) + One-Shot Logits Gather
#   6. Optional Single-Stream Decode MegaKernel + DFlash2/DSpark Speculative Decoding
#
# Usage:
#   # Mode 1: Launch TP=32, EP=8 SGLang OpenAI Server + Run Full Benchmark Matrix:
#   POD_SELECTOR="app=tpu-v7x-16" MODE="server" bash scripts/relaunch_pass3_sglang_day0.sh
#
#   # Mode 2: Run Single-Stream Decode MegaKernel + Speculative Decoding Benchmark:
#   POD_SELECTOR="app=tpu-v7x-16" MODE="megakernel" bash scripts/relaunch_pass3_sglang_day0.sh
# ==============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODE="${MODE:-server}"                             # server or megakernel
CONFIG_TAG="${CONFIG_TAG:-sglang-tp32-ep8}"
MODEL_PATH="${MODEL_PATH:-/models/models/moonshotai/Kimi-K3.1-Pro}"
SPEC_DRAFT_PATH="${SPEC_DRAFT_PATH:-/models/models/RadixArk/Kimi-K3-DSpark}"
PORT="${PORT:-30000}"
WAVE="${WAVE:-all}"
POD_SELECTOR="${POD_SELECTOR:-}"
PODS_OVERRIDE="${PODS_OVERRIDE:-}"
KUBECTL_CONTEXT="${KUBECTL_CONTEXT:-}"

kctl() {
  if [[ -n "${KUBECTL_CONTEXT}" ]]; then
    kubectl --context "${KUBECTL_CONTEXT}" "$@"
  else
    kubectl "$@"
  fi
}

if [[ -n "${PODS_OVERRIDE}" ]]; then
  read -r -a ORDERED_PODS <<< "${PODS_OVERRIDE}"
else
  if [[ -n "${POD_SELECTOR}" ]]; then
    RAW_PODS=($(kctl get pods -l "${POD_SELECTOR}" -o jsonpath='{.items[*].metadata.name}'))
  else
    RAW_PODS=($(kctl get pods -o jsonpath='{.items[*].metadata.name}' | tr ' ' '\n' | grep -E 'v7x16|kimi-k31' | sort))
  fi
  if [[ "${#RAW_PODS[@]}" -ne 4 ]]; then
    echo "ERROR: Expected 4 pods for a 2x2x4 TPU v7x slice, found ${#RAW_PODS[@]} (${RAW_PODS[*]:-none}). Set PODS_OVERRIDE='pod0 pod1 pod2 pod3'." >&2
    exit 1
  fi
  # Sort pods by physical TPU_WORKER_ID (0..3) so NODE_RANK matches physical torus order
  ORDERED_PODS=($(
    for p in "${RAW_PODS[@]}"; do
      wid=$(kctl exec "${p}" -- bash -c 'echo ${TPU_WORKER_ID:-0}' 2>/dev/null | tr -d '\r')
      echo "${wid}:${p}"
    done | sort -t: -k1,1n | cut -d: -f2
  ))
fi

HEAD_POD="${ORDERED_PODS[0]}"
POD_IPS=()
for p in "${ORDERED_PODS[@]}"; do
  POD_IPS+=("$(kctl get pod "${p}" -o jsonpath='{.status.podIP}')")
done
HEAD_IP="${POD_IPS[0]}"
TPU_HOSTNAMES_CSV=$(IFS=,; echo "${POD_IPS[*]}")

echo "=== [Pass 3] Physical 2x2x4 Torus Pod Order (NODE_RANK 0..3): ${ORDERED_PODS[*]} ==="
echo "=== [Pass 3] TPU_WORKER_HOSTNAMES=${TPU_HOSTNAMES_CSV}, DIST_INIT_ADDR=${HEAD_IP}:10011, MODE=${MODE} ==="

# 1. Ensure SGLang-TorchTPU (origin/kimi-k31-pro-tpu) and dependencies are staged on /dev/shm across all 4 pods
for pod in "${ORDERED_PODS[@]}"; do
  echo "[Pass 3] Preparing SGLang-TorchTPU environment on /dev/shm & syncing configs on ${pod}..."
  kctl exec "${pod}" -- bash -c '
    mkdir -p /dev/shm/sglang_tmp /dev/shm/tpu_tier3_cache /dev/shm/torchinductor /dev/shm/xdg_cache /dev/shm/hf_cache /dev/shm/workspace/kimi_k31/configs /dev/shm/workspace/kimi_k31/results /models/benchmark-results/kimi_k31
    rm -rf /workspace/kimi_k31 /workspace/sglang /workspace/sglang_deps
    mkdir -p /workspace
    ln -sfn /dev/shm/workspace/kimi_k31 /workspace/kimi_k31
    ln -sfn /dev/shm/workspace/sglang /workspace/sglang
    ln -sfn /dev/shm/workspace/sglang_deps /workspace/sglang_deps
  '
  kctl cp "${REPO_ROOT}/configs/sglang-torchtpu-kimi-k3.1-pro-tp32-ep8.py" "${pod}:/dev/shm/workspace/kimi_k31/configs/sglang-torchtpu-kimi-k3.1-pro-tp32-ep8.py"
  kctl cp "${REPO_ROOT}/configs/sglang-torchtpu-kimi-k3.1-pro-megakernel.sh" "${pod}:/dev/shm/workspace/kimi_k31/configs/sglang-torchtpu-kimi-k3.1-pro-megakernel.sh"
  kctl cp "${REPO_ROOT}/benchmarks/run_kimi_k31_matrix.py" "${pod}:/dev/shm/workspace/kimi_k31/run_kimi_k31_matrix.py"
  kctl exec "${pod}" -- bash -c '
    python3 -c "import os, signal, pathlib; me={1, os.getpid(), os.getppid()}; [os.kill(int(p.name), 9) for p in pathlib.Path(\"/proc\").iterdir() if p.name.isdigit() and int(p.name) not in me and any(k in p.joinpath(\"cmdline\").read_bytes().decode(\"utf-8\",\"ignore\") for k in (\"vllm serve\", \"vllm.entrypoints\", \"VLLM::Worker\", \"multiprocessing.spawn\", \"sglang\", \"EngineCore\", \"run_kimi_k31_matrix\"))]" 2>/dev/null || true
    ray stop --force 2>/dev/null || true
    sleep 2
    if [[ ! -d /dev/shm/workspace/sglang/python/sglang ]] && [[ -f /models/sglang_kimi_k31.tar.gz ]]; then
      mkdir -p /dev/shm/workspace/sglang
      tar -xzf /models/sglang_kimi_k31.tar.gz -C /dev/shm/workspace/sglang
    fi
    if [[ ! -d /dev/shm/workspace/sglang_deps/yaml ]] && [[ -f /models/sglang_extra_site_packages.tar.gz ]]; then
      mkdir -p /dev/shm/workspace/sglang_deps
      tar -xzf /models/sglang_extra_site_packages.tar.gz -C /dev/shm/workspace/sglang_deps
    fi
    if [[ -d /models/sglang_extra_deps_v2 ]]; then
      mkdir -p /dev/shm/workspace/sglang_deps
      cp -rf /models/sglang_extra_deps_v2/* /dev/shm/workspace/sglang_deps/
    fi
    python3 - << "PY"
import pathlib
subs = [
    ("pltpu.semaphore_signal(", "pl.semaphore_signal("),
    ("pltpu.semaphore_wait(", "pl.semaphore_wait("),
    ("pltpu.semaphore_read(", "pl.semaphore_read("),
    ("pltpu.DeviceIdType", "pl.DeviceIdType"),
    ("pltpu.semaphore(", "pl.semaphore("),
]
for fp in pathlib.Path("/dev/shm/workspace/sglang/sglang_tpu").rglob("*.py"):
    orig = fp.read_text(encoding="utf-8", errors="replace")
    updated = orig
    for old, new in subs:
        updated = updated.replace(old, new)
    if updated != orig:
        fp.write_text(updated, encoding="utf-8")
        print("Patched JAX 0.11 Pallas in:", fp)
PY
    PYTHONPATH=/workspace/sglang/python:/workspace/sglang:/workspace/sglang_deps python3 -c "import sglang.launch_server, sglang.srt.entrypoints.http_server, sglang_tpu; print(\"SGLANG_IMPORT_OK\")"
  '
done

if [[ "${MODE}" == "megakernel" ]]; then
  echo "[Pass 3] Launching Single-Stream Decode MegaKernel + Speculative Decoding across all 4 pods..."
  for rank in 0 1 2 3; do
    pod="${ORDERED_PODS[$rank]}"
    kctl exec "${pod}" -- python3 -c "
import os, subprocess
env = os.environ.copy()
env.pop('TPU_VISIBLE_CHIPS', None)
env['NODE_RANK'] = '${rank}'
env['NNODES'] = '4'
env['DIST_INIT_ADDR'] = '${HEAD_IP}:10011'
env['TPU_WORKER_ID'] = '${rank}'
env['TPU_WORKER_HOSTNAMES'] = '${TPU_HOSTNAMES_CSV}'
env['MODEL_PATH'] = '${MODEL_PATH}'
env['SPEC_DRAFT_PATH'] = '${SPEC_DRAFT_PATH}'
env['DRAFT_MODEL_PATH'] = '${SPEC_DRAFT_PATH}'
env['TMPDIR'] = '/dev/shm/sglang_tmp'
env['TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT'] = '/dev/shm/tpu_tier3_cache'
env['TORCHINDUCTOR_CACHE_DIR'] = '/dev/shm/torchinductor'
env['XDG_CACHE_HOME'] = '/dev/shm/xdg_cache'
env['HF_HOME'] = '/dev/shm/hf_cache'
env['PYTHONPATH'] = '/workspace/sglang/python:/workspace/sglang:/workspace/sglang_deps'
log = open('/workspace/kimi_k31/sglang_megakernel_rank${rank}.log', 'w')
p = subprocess.Popen(
    ['bash', '/workspace/kimi_k31/configs/sglang-torchtpu-kimi-k3.1-pro-megakernel.sh'],
    cwd='/workspace/sglang',
    env=env,
    stdin=subprocess.DEVNULL,
    stdout=log,
    stderr=subprocess.STDOUT,
    start_new_session=True,
)
print('MEGAKERNEL_RANK_${rank}_PID=', p.pid)
"
  done
  echo "=== MegaKernel launched on all 4 ranks! Monitor: kubectl exec ${HEAD_POD} -- tail -f /workspace/kimi_k31/sglang_megakernel_rank0.log ==="
  exit 0
fi

# 2. Launch SGLang-TorchTPU (TP=32, EP=8) across all 4 pods (NODE_RANK=0..3)
for rank in 0 1 2 3; do
  pod="${ORDERED_PODS[$rank]}"
  echo "[Pass 3] Launching SGLang-TorchTPU NODE_RANK=${rank} on ${pod}..."
  kctl exec "${pod}" -- python3 -c "
import importlib.util, json, os, subprocess
os.environ['MODEL_PATH'] = '${MODEL_PATH}'
os.environ['PORT'] = '${PORT}'
spec = importlib.util.spec_from_file_location('cfg', '/workspace/kimi_k31/configs/sglang-torchtpu-kimi-k3.1-pro-tp32-ep8.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
cfg = dict(mod.CONFIG)
cfg['server_args'] = {k: v for k, v in cfg['server_args'].items() if v is not None}
json_path = '/workspace/kimi_k31/configs/sglang-torchtpu-kimi-k3.1-pro-tp32-ep8.json'
open(json_path, 'w').write(json.dumps(cfg, indent=2))

env = os.environ.copy()
env.pop('TPU_VISIBLE_CHIPS', None)
env['NODE_RANK'] = '${rank}'
env['NNODES'] = '4'
env['DIST_INIT_ADDR'] = '${HEAD_IP}:10011'
env['TPU_WORKER_ID'] = '${rank}'
env['TPU_WORKER_HOSTNAMES'] = '${TPU_HOSTNAMES_CSV}'
env['MODEL_PATH'] = '${MODEL_PATH}'
env['PORT'] = '${PORT}'
env['HEALTH_TIMEOUT'] = '7200'
env['TMPDIR'] = '/dev/shm/sglang_tmp'
env['TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT'] = '/dev/shm/tpu_tier3_cache'
env['TORCHINDUCTOR_CACHE_DIR'] = '/dev/shm/torchinductor'
env['XDG_CACHE_HOME'] = '/dev/shm/xdg_cache'
env['HF_HOME'] = '/dev/shm/hf_cache'
env['PYTHONPATH'] = '/workspace/sglang/python:/workspace/sglang:/workspace/sglang_deps'
log = open('/workspace/kimi_k31/sglang_tp32_ep8_rank${rank}.log', 'w')
p = subprocess.Popen(
    ['python3', '-u', '-m', 'benchmark.tpu.launch_server', '--config', '@' + json_path],
    cwd='/workspace/sglang',
    env=env,
    stdin=subprocess.DEVNULL,
    stdout=log,
    stderr=subprocess.STDOUT,
    start_new_session=True,
)
print('SGLANG_RANK_${rank}_PID=', p.pid)
"
done

EXTRA_BENCH_ARGS="${EXTRA_BENCH_ARGS:-}"
DETACH_BENCH="${DETACH_BENCH:-0}"
if [[ "${DETACH_BENCH}" == "1" ]]; then
  echo "[Pass 3] Spawning detached in-pod benchmark matrix driver on ${HEAD_POD} (logs: /workspace/kimi_k31/pass3_${CONFIG_TAG}_bench.log, durable GCS output: /models/benchmark-results/kimi_k31/${CONFIG_TAG})..."
  kctl exec "${HEAD_POD}" -- python3 -c "
import os, shlex, subprocess
cmd = [
    'python3', '-u', '/workspace/kimi_k31/run_kimi_k31_matrix.py',
    '--port', '${PORT}',
    '--wave', '${WAVE}',
    '--config-tag', '${CONFIG_TAG}',
    '--tokenizer', '${MODEL_PATH}',
    '--output-dir', '/models/benchmark-results/kimi_k31',
] + shlex.split('${EXTRA_BENCH_ARGS}')
log = open('/workspace/kimi_k31/pass3_${CONFIG_TAG}_bench.log', 'w')
p = subprocess.Popen(cmd, cwd='/workspace/kimi_k31', stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
print('PASS3_BENCH_DRIVER_PID=', p.pid)
"
  echo "=== [Pass 3 Detached] Server + Benchmark Driver running in-cluster on ${HEAD_POD}! ==="
  exit 0
fi

# 3. Wait for SGLang HTTP server on NODE_RANK=0 and run benchmark matrix
echo "[Pass 3] Waiting for SGLang /v1/models on ${HEAD_POD}:${PORT} (logs: /workspace/kimi_k31/sglang_tp32_ep8_rank0.log)..."
kctl exec "${HEAD_POD}" -- bash -c "
  for i in \$(seq 1 720); do
    if curl -sf http://127.0.0.1:${PORT}/v1/models >/dev/null 2>&1; then
      echo '=== SGLang-TorchTPU Server Ready! ==='
      exit 0
    fi
    sleep 5
  done
  echo 'ERROR: Timed out waiting for SGLang server' >&2
  exit 1
"

echo "[Pass 3] Running benchmark matrix (--port ${PORT} --wave ${WAVE} --config-tag ${CONFIG_TAG} ${EXTRA_BENCH_ARGS})..."
kctl exec "${HEAD_POD}" -- python3 -u /workspace/kimi_k31/run_kimi_k31_matrix.py \
  --port "${PORT}" \
  --wave "${WAVE}" \
  --config-tag "${CONFIG_TAG}" \
  --tokenizer "${MODEL_PATH}" \
  --output-dir /workspace/kimi_k31/results \
  ${EXTRA_BENCH_ARGS}

mkdir -p "${REPO_ROOT}/results/${CONFIG_TAG}"
kctl cp "${HEAD_POD}:/workspace/kimi_k31/results/${CONFIG_TAG}/summary.json" "${REPO_ROOT}/results/${CONFIG_TAG}/summary.json"
kctl cp "${HEAD_POD}:/workspace/kimi_k31/results/${CONFIG_TAG}/REPORT.md" "${REPO_ROOT}/results/${CONFIG_TAG}/REPORT.md"
echo "=== [Pass 3 Complete] Results saved to ${REPO_ROOT}/results/${CONFIG_TAG}/ ==="
