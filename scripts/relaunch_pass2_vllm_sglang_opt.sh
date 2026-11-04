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
  # Sort pods by physical TPU_WORKER_ID (0..3) so Worker 0 is the Ray head
  ORDERED_PODS=($(
    for p in "${RAW_PODS[@]}"; do
      wid=$(kubectl exec "${p}" -- bash -c 'echo ${TPU_WORKER_ID:-0}' 2>/dev/null | tr -d '\r')
      echo "${wid}:${p}"
    done | sort -t: -k1,1n | cut -d: -f2
  ))
fi

HEAD_POD="${ORDERED_PODS[0]}"
HEAD_IP=$(kubectl get pod "${HEAD_POD}" -o jsonpath='{.status.podIP}')
echo "=== [Pass 2] Physical 2x2x4 Torus Pod Order (Worker 0..3): ${ORDERED_PODS[*]} ==="
echo "=== [Pass 2] Ray Head Pod: ${HEAD_POD} (${HEAD_IP}), Topology: ${TOPOLOGY}, Config Tag: ${CONFIG_TAG} ==="

# 1. Sync patch, config, and benchmark scripts to all 4 pods and apply runtime patch
for pod in "${ORDERED_PODS[@]}"; do
  echo "[Pass 2] Syncing scripts & applying Kimi-K3.1-Pro patch on ${pod}..."
  kubectl exec "${pod}" -- mkdir -p /workspace/kimi_k31/scripts/patches /workspace/kimi_k31/configs /workspace/kimi_k31/results
  kubectl cp "${REPO_ROOT}/scripts/patches/patch_vllm_torchtpu_kimi_k31.py" "${pod}:/workspace/kimi_k31/scripts/patches/patch_vllm_torchtpu_kimi_k31.py"
  kubectl cp "${REPO_ROOT}/configs/kimi-k3.1-pro-${TOPOLOGY}.sh" "${pod}:/workspace/kimi_k31/configs/kimi-k3.1-pro-${TOPOLOGY}.sh"
  kubectl cp "${REPO_ROOT}/benchmarks/run_kimi_k31_matrix.py" "${pod}:/workspace/kimi_k31/run_kimi_k31_matrix.py"
  kubectl exec "${pod}" -- bash -c '
    pkill -9 -f "vllm|sglang|EngineCore" 2>/dev/null || true
    ray stop --force 2>/dev/null || true
    rm -rf /tmp/ray /dev/shm/vllm_cache /dev/shm/torchinductor
    mkdir -p /dev/shm/hf_cache /dev/shm/vllm_tmp /dev/shm/vllm_cache /dev/shm/torchinductor
    python3 /workspace/kimi_k31/scripts/patches/patch_vllm_torchtpu_kimi_k31.py
  '
done

# 2. Start Ray Head on Worker 0 and Ray Workers on Workers 1..3
echo "[Pass 2] Starting Ray Head on ${HEAD_POD} (${HEAD_IP}:6379)..."
kubectl exec "${HEAD_POD}" -- bash -c "
  export RAY_EXPERIMENTAL_NOSET_TPU_VISIBLE_CHIPS=1
  export PJRT_DEVICE=TPU
  export TPU_SKIP_MDS_QUERY=1
  ray start --head --port=6379 --num-cpus=120 --resources='{\"TPU\": 4}' --disable-usage-stats
"
for pod in "${ORDERED_PODS[@]:1}"; do
  echo "[Pass 2] Joining Ray Worker ${pod} to ${HEAD_IP}:6379..."
  kubectl exec "${pod}" -- bash -c "
    export RAY_EXPERIMENTAL_NOSET_TPU_VISIBLE_CHIPS=1
    export PJRT_DEVICE=TPU
    export TPU_SKIP_MDS_QUERY=1
    ray start --address='${HEAD_IP}:6379' --num-cpus=120 --resources='{\"TPU\": 4}' --disable-usage-stats
  "
done

# 3. Launch vllm serve in a detached session on Worker 0
echo "[Pass 2] Launching SGLang-Optimized vllm-torchtpu (${TOPOLOGY}) on ${HEAD_POD}..."
kubectl exec "${HEAD_POD}" -- python3 -c "
import os, subprocess
env = os.environ.copy()
env['MODEL_URI'] = '${MODEL_URI}'
env['TOKENIZER_DIR'] = '${TOKENIZER_DIR}'
log = open('/workspace/kimi_k31/vllm_${CONFIG_TAG}.log', 'w')
p = subprocess.Popen(
    ['bash', '/workspace/kimi_k31/configs/kimi-k3.1-pro-${TOPOLOGY}.sh'],
    env=env,
    stdin=subprocess.DEVNULL,
    stdout=log,
    stderr=subprocess.STDOUT,
    start_new_session=True,
)
print('VLLM_PID=', p.pid)
"

# 4. Wait for /health and launch benchmark matrix
echo "[Pass 2] Waiting for vLLM /health on ${HEAD_POD}:8000 (logs: /workspace/kimi_k31/vllm_${CONFIG_TAG}.log)..."
kubectl exec "${HEAD_POD}" -- bash -c '
  for i in $(seq 1 720); do
    if curl -sf http://127.0.0.1:8000/health >/dev/null 2>&1; then
      echo "=== vLLM Server Ready! ==="
      exit 0
    fi
    sleep 5
  done
  echo "ERROR: Timed out waiting for vLLM /health" >&2
  exit 1
'

echo "[Pass 2] Running benchmark matrix (--wave ${WAVE} --config-tag ${CONFIG_TAG})..."
kubectl exec "${HEAD_POD}" -- python3 -u /workspace/kimi_k31/run_kimi_k31_matrix.py \
  --wave "${WAVE}" \
  --config-tag "${CONFIG_TAG}" \
  --output-dir /workspace/kimi_k31/results

mkdir -p "${REPO_ROOT}/results/${CONFIG_TAG}"
kubectl cp "${HEAD_POD}:/workspace/kimi_k31/results/${CONFIG_TAG}/summary.json" "${REPO_ROOT}/results/${CONFIG_TAG}/summary.json"
kubectl cp "${HEAD_POD}:/workspace/kimi_k31/results/${CONFIG_TAG}/REPORT.md" "${REPO_ROOT}/results/${CONFIG_TAG}/REPORT.md"
echo "=== [Pass 2 Complete] Results saved to ${REPO_ROOT}/results/${CONFIG_TAG}/ ==="
