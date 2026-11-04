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

if [[ -n "${PODS_OVERRIDE}" ]]; then
  read -r -a ORDERED_PODS <<< "${PODS_OVERRIDE}"
else
  if [[ -n "${POD_SELECTOR}" ]]; then
    RAW_PODS=($(kubectl get pods -l "${POD_SELECTOR}" -o jsonpath='{.items[*].metadata.name}'))
  else
    RAW_PODS=($(kubectl get pods -o jsonpath='{.items[*].metadata.name}' | tr ' ' '\n' | grep -E 'v7x16|kimi-k31' | sort))
  fi
  if [[ "${#RAW_PODS[@]}" -ne 4 ]]; then
    echo "ERROR: Expected 4 pods for a 2x2x4 TPU v7x slice, found ${#RAW_PODS[@]} (${RAW_PODS[*]:-none}). Set PODS_OVERRIDE='pod0 pod1 pod2 pod3'." >&2
    exit 1
  fi
  # Sort pods by physical TPU_WORKER_ID (0..3) so NODE_RANK matches physical torus order
  ORDERED_PODS=($(
    for p in "${RAW_PODS[@]}"; do
      wid=$(kubectl exec "${p}" -- bash -c 'echo ${TPU_WORKER_ID:-0}' 2>/dev/null | tr -d '\r')
      echo "${wid}:${p}"
    done | sort -t: -k1,1n | cut -d: -f2
  ))
fi

HEAD_POD="${ORDERED_PODS[0]}"
POD_IPS=()
for p in "${ORDERED_PODS[@]}"; do
  POD_IPS+=("$(kubectl get pod "${p}" -o jsonpath='{.status.podIP}')")
done
HEAD_IP="${POD_IPS[0]}"
TPU_HOSTNAMES_CSV=$(IFS=,; echo "${POD_IPS[*]}")

echo "=== [Pass 3] Physical 2x2x4 Torus Pod Order (NODE_RANK 0..3): ${ORDERED_PODS[*]} ==="
echo "=== [Pass 3] TPU_WORKER_HOSTNAMES=${TPU_HOSTNAMES_CSV}, DIST_INIT_ADDR=${HEAD_IP}:10011, MODE=${MODE} ==="

# 1. Ensure SGLang-TorchTPU (origin/kimi-k31-pro-tpu) and dependencies are staged on all 4 pods
for pod in "${ORDERED_PODS[@]}"; do
  echo "[Pass 3] Preparing SGLang-TorchTPU environment & syncing configs on ${pod}..."
  kubectl exec "${pod}" -- mkdir -p /workspace/kimi_k31/configs /workspace/kimi_k31/results
  kubectl cp "${REPO_ROOT}/configs/sglang-torchtpu-kimi-k3.1-pro-tp32-ep8.py" "${pod}:/workspace/kimi_k31/configs/sglang-torchtpu-kimi-k3.1-pro-tp32-ep8.py"
  kubectl cp "${REPO_ROOT}/configs/sglang-torchtpu-kimi-k3.1-pro-megakernel.sh" "${pod}:/workspace/kimi_k31/configs/sglang-torchtpu-kimi-k3.1-pro-megakernel.sh"
  kubectl cp "${REPO_ROOT}/benchmarks/run_kimi_k31_matrix.py" "${pod}:/workspace/kimi_k31/run_kimi_k31_matrix.py"
  kubectl exec "${pod}" -- bash -c '
    pkill -9 -f "vllm|sglang|EngineCore" 2>/dev/null || true
    ray stop --force 2>/dev/null || true
    if [[ ! -d /workspace/sglang/python/sglang ]] && [[ -f /models/sglang_kimi_k31.tar.gz ]]; then
      tar -xzf /models/sglang_kimi_k31.tar.gz -C /workspace
    fi
    if [[ ! -d /workspace/sglang_deps ]] && [[ -f /models/sglang_extra_site_packages.tar.gz ]]; then
      mkdir -p /workspace/sglang_deps
      tar -xzf /models/sglang_extra_site_packages.tar.gz -C /workspace/sglang_deps
    fi
    PYTHONPATH=/workspace/sglang/python:/workspace/sglang:/workspace/sglang_deps python3 -c "import sglang.launch_server, sglang_tpu; print(\"SGLANG_IMPORT_OK\")"
  '
done

if [[ "${MODE}" == "megakernel" ]]; then
  echo "[Pass 3] Launching Single-Stream Decode MegaKernel + Speculative Decoding across all 4 pods..."
  for rank in 0 1 2 3; do
    pod="${ORDERED_PODS[$rank]}"
    kubectl exec "${pod}" -- python3 -c "
import os, subprocess
env = os.environ.copy()
env['NODE_RANK'] = '${rank}'
env['NNODES'] = '4'
env['DIST_INIT_ADDR'] = '${HEAD_IP}:10011'
env['TPU_WORKER_ID'] = '${rank}'
env['TPU_WORKER_HOSTNAMES'] = '${TPU_HOSTNAMES_CSV}'
env['MODEL_PATH'] = '${MODEL_PATH}'
env['SPEC_DRAFT_PATH'] = '${SPEC_DRAFT_PATH}'
env['PYTHONPATH'] = '/workspace/sglang/python:/workspace/sglang:/workspace/sglang_deps'
log = open('/workspace/kimi_k31/sglang_megakernel_rank${rank}.log', 'w')
p = subprocess.Popen(
    ['bash', '/workspace/kimi_k31/configs/sglang-torchtpu-kimi-k3.1-pro-megakernel.sh'],
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
  kubectl exec "${pod}" -- python3 -c "
import os, subprocess
env = os.environ.copy()
env['NODE_RANK'] = '${rank}'
env['NNODES'] = '4'
env['DIST_INIT_ADDR'] = '${HEAD_IP}:10011'
env['TPU_WORKER_ID'] = '${rank}'
env['TPU_WORKER_HOSTNAMES'] = '${TPU_HOSTNAMES_CSV}'
env['MODEL_PATH'] = '${MODEL_PATH}'
env['PORT'] = '${PORT}'
env['PYTHONPATH'] = '/workspace/sglang/python:/workspace/sglang:/workspace/sglang_deps'
log = open('/workspace/kimi_k31/sglang_tp32_ep8_rank${rank}.log', 'w')
p = subprocess.Popen(
    ['python3', '-u', '/workspace/kimi_k31/configs/sglang-torchtpu-kimi-k3.1-pro-tp32-ep8.py'],
    env=env,
    stdin=subprocess.DEVNULL,
    stdout=log,
    stderr=subprocess.STDOUT,
    start_new_session=True,
)
print('SGLANG_RANK_${rank}_PID=', p.pid)
"
done

# 3. Wait for SGLang HTTP server on NODE_RANK=0 and run benchmark matrix
echo "[Pass 3] Waiting for SGLang /v1/models on ${HEAD_POD}:${PORT} (logs: /workspace/kimi_k31/sglang_tp32_ep8_rank0.log)..."
kubectl exec "${HEAD_POD}" -- bash -c "
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

echo "[Pass 3] Running benchmark matrix (--port ${PORT} --wave ${WAVE} --config-tag ${CONFIG_TAG})..."
kubectl exec "${HEAD_POD}" -- python3 -u /workspace/kimi_k31/run_kimi_k31_matrix.py \
  --port "${PORT}" \
  --wave "${WAVE}" \
  --config-tag "${CONFIG_TAG}" \
  --output-dir /workspace/kimi_k31/results

mkdir -p "${REPO_ROOT}/results/${CONFIG_TAG}"
kubectl cp "${HEAD_POD}:/workspace/kimi_k31/results/${CONFIG_TAG}/summary.json" "${REPO_ROOT}/results/${CONFIG_TAG}/summary.json"
kubectl cp "${HEAD_POD}:/workspace/kimi_k31/results/${CONFIG_TAG}/REPORT.md" "${REPO_ROOT}/results/${CONFIG_TAG}/REPORT.md"
echo "=== [Pass 3 Complete] Results saved to ${REPO_ROOT}/results/${CONFIG_TAG}/ ==="
