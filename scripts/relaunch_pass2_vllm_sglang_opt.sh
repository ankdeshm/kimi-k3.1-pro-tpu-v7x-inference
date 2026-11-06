#!/usr/bin/env bash
# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0
# ==============================================================================
# Turnkey Relauncher — Pass 2: SGLang-Optimized vllm-torchtpu on TPU v7x (2x2x4)
# ==============================================================================
# Applies all 5 SGLang-inspired hardware & Pallas kernel optimizations to
# vllm-torchtpu on a 4-host TPU v7x 2x2x4 slice (16 physical chips = 32 TensorCores):
#   1. Ironwood SparseCore Gather Offload (--xla_tpu_offload_gather_to_sparsecore=true)
#   2. TP=32 Sequence-Parallel Prefill (TPU_K3_SP_PREFILL=1) + Tuned v2 MLA Pallas Table
#   3. Fused Pallas MoE Decode Kernel (FUSED_MOE_DECODE_KERNEL=1) + One-Shot Greedy Top-Tokens (TPU_GREEDY_TOP_TOKENS=1)
#   4. MegaKDA Pallas DMA Overlap (KDA_OVERLAP_H0_DMA=1, KDA_OVERLAP_HT_DMA=1) + Head-Inversion Packing (KDA_PACK_HEAD_INV=1)
#   5. Hybrid KDA + MLA Aligned Scheduler Token Budget (--max-num-batched-tokens=2720 = 2 * 1360)
#
# Usage:
#   # Launch TP=32, DP=1, EP=32 (Sequence-Parallel Prefill + Tuned v2 MLA Pallas Table):
#   POD_SELECTOR="app=tpu-v7x-16" TOPOLOGY="tp32-ep32" bash scripts/relaunch_pass2_vllm_sglang_opt.sh
#
#   # Or launch TP=8, DP=4, EP=32 with SparseCore + Fused MoE Decode + MegaKDA DMA Overlap:
#   POD_SELECTOR="app=tpu-v7x-16" TOPOLOGY="tp8-dp4-ep32" bash scripts/relaunch_pass2_vllm_sglang_opt.sh
# ==============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TOPOLOGY="${TOPOLOGY:-tp32-ep32}"                  # tp32-ep32 or tp8-dp4-ep32
CONFIG_TAG="${CONFIG_TAG:-vllm-${TOPOLOGY}-sglang-opt}"
MODEL_URI="${MODEL_URI:-/models/models/moonshotai/Kimi-K3.1-Pro}"
TOKENIZER_DIR="${TOKENIZER_DIR:-/models/models/moonshotai/Kimi-K3.1-Pro}"
WAVE="${WAVE:-all}"                                # all, regular, long_context, prefix_caching
POD_SELECTOR="${POD_SELECTOR:-}"
PODS_OVERRIDE="${PODS_OVERRIDE:-}"                 # Optional space-separated pod list in worker 0..3 order

KUBECTL_CONTEXT="${KUBECTL_CONTEXT:-}"
KCTL=(kubectl)
if [[ -n "${KUBECTL_CONTEXT}" ]]; then
  KCTL+=(--context="${KUBECTL_CONTEXT}")
fi

if [[ -n "${PODS_OVERRIDE}" ]]; then
  read -r -a ORDERED_PODS <<< "${PODS_OVERRIDE}"
else
  if [[ -n "${POD_SELECTOR}" ]]; then
    RAW_PODS=($("${KCTL[@]}" get pods -l "${POD_SELECTOR}" -o jsonpath='{.items[*].metadata.name}'))
  else
    RAW_PODS=($("${KCTL[@]}" get pods -o jsonpath='{.items[*].metadata.name}' | tr ' ' '\n' | grep -E 'v7x16|kimi-k31' | sort))
  fi
  if [[ "${#RAW_PODS[@]}" -ne 4 ]]; then
    echo "ERROR: Expected 4 pods for a 2x2x4 TPU v7x slice, found ${#RAW_PODS[@]} (${RAW_PODS[*]:-none}). Set PODS_OVERRIDE='pod0 pod1 pod2 pod3'." >&2
    exit 1
  fi
  # Sort pods by physical TPU_WORKER_ID (0..3) so Worker 0 is the Head
  ORDERED_PODS=($(
    for p in "${RAW_PODS[@]}"; do
      wid=$("${KCTL[@]}" exec "${p}" -- bash -c 'echo ${TPU_WORKER_ID:-0}' 2>/dev/null | tr -d '\r')
      echo "${wid}:${p}"
    done | sort -t: -k1,1n | cut -d: -f2
  ))
fi

HEAD_POD="${ORDERED_PODS[0]}"
HEAD_IP=$("${KCTL[@]}" get pod "${HEAD_POD}" -o jsonpath='{.status.podIP}')
echo "=== [Pass 2] Physical 2x2x4 Torus Pod Order (Worker 0..3): ${ORDERED_PODS[*]} ==="
echo "=== [Pass 2] Master/Head Pod: ${HEAD_POD} (${HEAD_IP}), Topology: ${TOPOLOGY}, Config Tag: ${CONFIG_TAG} ==="

# 1. Sync patch, config, and benchmark scripts to all 4 pods and clean stale workers
for pod in "${ORDERED_PODS[@]}"; do
  echo "[Pass 2] Syncing scripts & applying Kimi-K3.1-Pro patch on ${pod}..."
  "${KCTL[@]}" exec "${pod}" -- mkdir -p /workspace/kimi_k31/scripts/patches /workspace/kimi_k31/configs /workspace/kimi_k31/results
  "${KCTL[@]}" cp "${REPO_ROOT}/scripts/patches/patch_vllm_torchtpu_kimi_k31.py" "${pod}:/workspace/kimi_k31/scripts/patches/patch_vllm_torchtpu_kimi_k31.py"
  "${KCTL[@]}" cp "${REPO_ROOT}/configs/kimi-k3.1-pro-${TOPOLOGY}.sh" "${pod}:/workspace/kimi_k31/configs/kimi-k3.1-pro-${TOPOLOGY}.sh"
  "${KCTL[@]}" cp "${REPO_ROOT}/benchmarks/run_kimi_k31_matrix.py" "${pod}:/workspace/kimi_k31/run_kimi_k31_matrix.py"
  "${KCTL[@]}" exec "${pod}" -- bash -c '
    python3 -c "import os, signal, pathlib; me={1, os.getpid(), os.getppid()}; [os.kill(int(p.name), 9) for p in pathlib.Path(\"/proc\").iterdir() if p.name.isdigit() and int(p.name) not in me and any(k in p.joinpath(\"cmdline\").read_bytes().decode(\"utf-8\",\"ignore\") for k in (\"vllm serve\", \"vllm.entrypoints\", \"sglang\", \"EngineCore\", \"VLLM::EngineCore\", \"VLLM::Worker\", \"multiprocessing.spawn\", \"multiprocessing.resource_tracker\"))]" 2>/dev/null || true
    sleep 4
    pgrep -f raylet >/dev/null 2>&1 && ray stop --force 2>/dev/null || true
    rm -rf /tmp/ray /dev/shm/vllm_cache /dev/shm/torchinductor /dev/shm/k3mk_* /dev/shm/torch_tpu_pg_store_* /dev/shm/workspace 2>/dev/null || true
    mkdir -p /dev/shm/hf_cache /dev/shm/vllm_tmp /dev/shm/vllm_cache /dev/shm/torchinductor
    if [[ -f /models/patches/vllm_torchtpu_v0.32.0/kernels/sparse_core/core_map_helper.py ]]; then
      cp -f /models/patches/vllm_torchtpu_v0.32.0/kernels/sparse_core/core_map_helper.py /workspace/vllm-torchtpu/src/vllm_torchtpu/kernels/sparse_core/core_map_helper.py
    fi
    python3 /workspace/kimi_k31/scripts/patches/patch_vllm_torchtpu_kimi_k31.py
  '
done

# 2. Launch vllm serve (--distributed-executor-backend=mp) in detached session across all 4 hosts
for idx in "${!ORDERED_PODS[@]}"; do
  pod="${ORDERED_PODS[$idx]}"
  pod_ip=$("${KCTL[@]}" get pod "${pod}" -o jsonpath='{.status.podIP}')
  echo "[Pass 2] Launching SGLang-Optimized vllm-torchtpu (${TOPOLOGY}) on Node ${idx}/4: ${pod} (${pod_ip})..."
  "${KCTL[@]}" exec "${pod}" -- python3 -c "
import os, subprocess
env = os.environ.copy()
env['NODE_RANK'] = '${idx}'
env['NNODES'] = '4'
env['HEAD_IP'] = '${HEAD_IP}'
env['POD_IP'] = '${pod_ip}'
env['MODEL_URI'] = '${MODEL_URI}'
env['TOKENIZER_DIR'] = '${TOKENIZER_DIR}'
log = open('/workspace/kimi_k31/vllm_${CONFIG_TAG}_rank${idx}.log', 'w')
p = subprocess.Popen(
    ['bash', '/workspace/kimi_k31/configs/kimi-k3.1-pro-${TOPOLOGY}.sh'],
    env=env,
    stdin=subprocess.DEVNULL,
    stdout=log,
    stderr=subprocess.STDOUT,
    start_new_session=True,
)
log.close()
print('NODE_RANK=${idx} VLLM_PID=', p.pid)
"
done

EXTRA_BENCH_ARGS="${EXTRA_BENCH_ARGS:-}"
DETACH_BENCH="${DETACH_BENCH:-0}"
DURABLE_OUT_DIR="${DURABLE_OUT_DIR:-/models/benchmark-results/kimi_k31}"

if [[ "${DETACH_BENCH}" == "1" ]]; then
  echo "[Pass 2] Spawning detached benchmark driver on ${HEAD_POD} (logs: /workspace/kimi_k31/pass2_${CONFIG_TAG}_bench.log, durable output: ${DURABLE_OUT_DIR}/${CONFIG_TAG})..."
  "${KCTL[@]}" exec "${HEAD_POD}" -- python3 -c "
import shlex, subprocess
cmd = [
    'python3', '-u', '/workspace/kimi_k31/run_kimi_k31_matrix.py',
    '--host', '127.0.0.1',
    '--port', '8000',
    '--wave', '${WAVE}',
    '--config-tag', '${CONFIG_TAG}',
    '--tokenizer', '${TOKENIZER_DIR}',
    '--output-dir', '${DURABLE_OUT_DIR}',
] + shlex.split('''${EXTRA_BENCH_ARGS}''')
log = open('/workspace/kimi_k31/pass2_${CONFIG_TAG}_bench.log', 'w')
p = subprocess.Popen(
    cmd,
    stdin=subprocess.DEVNULL,
    stdout=log,
    stderr=subprocess.STDOUT,
    start_new_session=True,
)
log.close()
print('BENCH_DRIVER_PID=', p.pid)
"
  exit 0
fi

# 3. Wait for /health on Worker 0 (fail early if vllm serve exits!) and launch benchmark matrix
echo "[Pass 2] Waiting for vLLM /health on ${HEAD_POD}:8000 (logs: /workspace/kimi_k31/vllm_${CONFIG_TAG}_rank0.log)..."
"${KCTL[@]}" exec "${HEAD_POD}" -- bash -c "
  for i in \$(seq 1 900); do
    if curl -sf http://127.0.0.1:8000/health >/dev/null 2>&1; then
      echo '=== vLLM Server Ready! ==='
      exit 0
    fi
    if ! pgrep -f 'vllm' >/dev/null 2>&1; then
      echo 'ERROR: vllm serve exited prematurely on head pod! Last 60 log lines:' >&2
      tail -n 60 /workspace/kimi_k31/vllm_${CONFIG_TAG}_rank0.log >&2 || true
      exit 1
    fi
    sleep 5
  done
  echo 'ERROR: Timed out waiting for vLLM /health' >&2
  exit 1
"

echo "[Pass 2] Running benchmark matrix (--wave ${WAVE} --config-tag ${CONFIG_TAG} ${EXTRA_BENCH_ARGS})..."
"${KCTL[@]}" exec "${HEAD_POD}" -- python3 -u /workspace/kimi_k31/run_kimi_k31_matrix.py \
  --wave "${WAVE}" \
  --config-tag "${CONFIG_TAG}" \
  --tokenizer "${TOKENIZER_DIR}" \
  --output-dir "${DURABLE_OUT_DIR}" \
  ${EXTRA_BENCH_ARGS}

mkdir -p "${REPO_ROOT}/results/${CONFIG_TAG}"
"${KCTL[@]}" cp "${HEAD_POD}:${DURABLE_OUT_DIR}/${CONFIG_TAG}/summary.json" "${REPO_ROOT}/results/${CONFIG_TAG}/summary.json"
"${KCTL[@]}" cp "${HEAD_POD}:${DURABLE_OUT_DIR}/${CONFIG_TAG}/REPORT.md" "${REPO_ROOT}/results/${CONFIG_TAG}/REPORT.md"
echo "=== [Pass 2 Complete] Results saved to ${REPO_ROOT}/results/${CONFIG_TAG}/ ==="
