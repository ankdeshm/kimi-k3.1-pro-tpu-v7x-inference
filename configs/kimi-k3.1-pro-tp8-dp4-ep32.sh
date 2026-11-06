#!/usr/bin/env bash
# ==============================================================================
# Moonshot AI Kimi-K3.1-Pro (2.8T Total / 104B Active MoE)
# Topology: TP=8, DP=4, EP=32 on TPU v7x 2x2x4 (16 Physical Chips = 32 TensorCores)
# ==============================================================================
# Architecture:
#   - 93 Hybrid Layers: 69 KDA (Kimi Delta Attention) + 24 Gated MLA / DSA Layers
#   - 896 Routed Experts (top-8 per token, native MXFP4) + 2 Shared Experts (BF16)
#   - 4 Data-Parallel Replicas (1 per host, TP=8 intra-host) + 32-way Expert Parallelism
#   - Unified KV + Mamba State Pool (block_size=1360, kv_cache_dtype=fp8)
# ==============================================================================
set -euo pipefail

export MODEL_URI="${MODEL_URI:-gs://<YOUR_GCS_MODEL_BUCKET>/models/moonshotai/Kimi-K3.1-Pro}"
export TOKENIZER_DIR="${TOKENIZER_DIR:-/models/models/moonshotai/Kimi-K3.1-Pro}"
export SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-moonshotai/Kimi-K3.1-Pro}"
export PORT="${PORT:-8000}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-131072}"
export MAX_NUM_SEQS="${MAX_NUM_SEQS:-64}"
# 2720 = 2 * 1360 (hybrid KDA+MLA block_size). Must be >= 1360 + max_num_seqs so
# --mamba-cache-mode=align can schedule 1360-token aligned prefill chunks alongside active decode sequences.
export MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-2720}"
export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.88}"
export ENABLE_PREFIX_CACHING="${ENABLE_PREFIX_CACHING:-1}"

export NODE_RANK="${NODE_RANK:-${TPU_WORKER_ID:-0}}"
export HEAD_IP="${HEAD_IP:?HEAD_IP must be set to Worker 0 pod IP}"
export POD_IP="${POD_IP:-$(hostname -i | awk '{print $1}')}"
export DP_RPC_PORT="${DP_RPC_PORT:-29500}"

# 1. Apply Kimi-K3.1-Pro runtime patch
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f /models/patches/vllm_torchtpu_v0.32.0/kernels/sparse_core/core_map_helper.py ]]; then
  cp -f /models/patches/vllm_torchtpu_v0.32.0/kernels/sparse_core/core_map_helper.py /workspace/vllm-torchtpu/src/vllm_torchtpu/kernels/sparse_core/core_map_helper.py
fi
python3 "${SCRIPT_DIR}/../scripts/patches/patch_vllm_torchtpu_kimi_k31.py"

# 2. TPU v7x Multi-Host MP Environment Setup (clear GKE's injected slice coordinator, keep TPU_VISIBLE_CHIPS)
ulimit -c 0 || true
mkdir -p /dev/shm/hf_cache /dev/shm/vllm_tmp /dev/shm/vllm_cache /dev/shm/torchinductor
unset TORCH_TPU_TOPOLOGY
export MEGASCALE_COORDINATOR_ADDRESS="" MEGASCALE_NUM_SLICES="" MEGASCALE_PORT=""
export MEGASCALE_SLICE_ID="" TPU_PROCESS_ADDRESSES="" TPU_PROCESS_PORT=""
export TPU_WORKER_HOSTNAMES=""

export TCMALLOC_AGGRESSIVE_DECOMMIT=true
export PJRT_DEVICE=TPU
export VLLM_TARGET_DEVICE=tpu
export VLLM_PLUGINS="torchtpu,torchtpu_layers"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_NO_USAGE_STATS=1
export VLLM_HOST_IP="${POD_IP}"
export TPU_MULTIHOST_BACKEND=mp
export TPU_CHIPS_PER_HOST_BOUNDS="2,2,1"
export TPU_HOST_BOUNDS="1,1,4"
export TPU_NODE_ID="${NODE_RANK}"
export TPU_SKIP_MDS_QUERY=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export TPU_UNIFIED_KV_CACHE_LAYOUT=1
export TPU_VLLM_ENABLE_UNIFIED_BLOCK_POOL=1
export TPU_MOE_HIERARCHICAL_EP=1
export TPU_LATENT_PROJ_INTRA_CHIP_TP=1
# TP=8, DP=4, EP=32 requires TPU_K3_SP_PREFILL=0 (sequence-parallel prefill is TP=32 only)
export TPU_K3_SP_PREFILL=0
export LIBTPU_INIT_ARGS="${LIBTPU_INIT_ARGS:---xla_tpu_buffer_assignment_enable_fast_merge_fallback=false --xla_tpu_offload_gather_to_sparsecore=true --xla_tpu_scoped_vmem_limit_kib=65536}"

# SGLang-Inspired Hardware & Pallas Kernel Optimizations for vllm-torchtpu:
export FUSED_MOE_DECODE_KERNEL="${FUSED_MOE_DECODE_KERNEL:-1}"
export FUSED_MOE_DECODE_KERNEL_MAX_TOKENS="${FUSED_MOE_DECODE_KERNEL_MAX_TOKENS:-256}"
export TPU_MOE_SKIP_PADDED_TOKENS="${TPU_MOE_SKIP_PADDED_TOKENS:-1}"
export TPU_GREEDY_TOP_TOKENS="${TPU_GREEDY_TOP_TOKENS:-0}"
# MegaKDA Pallas DMA Overlap & Head-Inversion Packing (69 KDA Linear Attention Layers)
export KDA_OVERLAP_H0_DMA="${KDA_OVERLAP_H0_DMA:-1}"
export KDA_OVERLAP_HT_DMA="${KDA_OVERLAP_HT_DMA:-1}"
export KDA_PACK_HEAD_INV="${KDA_PACK_HEAD_INV:-1}"
export KDA_PACKED_METADATA="${KDA_PACKED_METADATA:-1}"

# RunAI Streamer & Multi-Host Weight Loading Optimizations
export RUNAI_STREAMER_PROCESS_GROUP_SIZE=8
export RUNAI_STREAMER_CONCURRENCY=8
export RUNAI_STREAMER_MEMORY_LIMIT=8589934592
export VLLM_TPU_WEIGHT_LOAD_SYNC_INTERVAL=100
export HF_HOME=/dev/shm/hf_cache
export TMPDIR=/dev/shm/vllm_tmp
export VLLM_CACHE_ROOT=/dev/shm/vllm_cache
export VLLM_XLA_CACHE_PATH=/dev/shm/vllm_cache
export TORCHINDUCTOR_CACHE_DIR=/dev/shm/torchinductor
export TORCHINDUCTOR_AUTOGRAD_CACHE=0
export TORCH_DIST_TIMEOUT=3600
export VLLM_RPC_TIMEOUT=3600000
export VLLM_SHM_BROADCAST_TIMEOUT_S=3600
export VLLM_ENGINE_ITERATION_TIMEOUT_S=3600
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=10800
export VLLM_ENGINE_READY_TIMEOUT_S=14400

# 3. Warm HuggingFace Remote Code before worker spawn
if [[ -f /workspace/vllm-torchtpu/.buildkite/kubernetes/warm_remote_code.py ]]; then
  python3 /workspace/vllm-torchtpu/.buildkite/kubernetes/warm_remote_code.py "${MODEL_URI}"
fi

PREFIX_CACHE_ARGS=()
if [[ "${ENABLE_PREFIX_CACHING}" == "1" ]]; then
  PREFIX_CACHE_ARGS=(--enable-prefix-caching --mamba-cache-mode=align --enable-prompt-tokens-details)
else
  PREFIX_CACHE_ARGS=(--no-enable-prefix-caching)
fi

ROLE_NODE_ARGS=()
if [[ "${NODE_RANK}" == "0" ]]; then
  ROLE_NODE_ARGS=(--host=0.0.0.0 --port="${PORT}" --api-server-count=1)
else
  ROLE_NODE_ARGS=(--headless --data-parallel-start-rank="${NODE_RANK}")
fi

echo "=== Launching moonshotai/Kimi-K3.1-Pro (TP=8, DP=4, EP=32) on Node ${NODE_RANK}/4 (master=${HEAD_IP}, ip=${POD_IP}) ==="
exec vllm serve "${MODEL_URI}" \
  --tokenizer="${TOKENIZER_DIR}" \
  --served-model-name="${SERVED_MODEL_NAME}" \
  --distributed-executor-backend=mp \
  --tensor-parallel-size=8 \
  --data-parallel-size=4 \
  --data-parallel-size-local=1 \
  --data-parallel-address="${HEAD_IP}" \
  --data-parallel-rpc-port="${DP_RPC_PORT}" \
  --distributed-timeout-seconds=3600 \
  --attention-backend=CUSTOM \
  --enable-expert-parallel \
  --enable-ep-weight-filter \
  --language-model-only \
  --limit-mm-per-prompt='{"image":0,"video":0}' \
  --load-format=runai_streamer \
  --safetensors-load-strategy=prefetch \
  --model-loader-extra-config='{"memory_limit":8589934592,"concurrency":8}' \
  --kv-cache-dtype=fp8 \
  --gpu-memory-utilization="${GPU_MEMORY_UTILIZATION}" \
  --max-model-len="${MAX_MODEL_LEN}" \
  --max-num-seqs="${MAX_NUM_SEQS}" \
  --max-num-batched-tokens="${MAX_NUM_BATCHED_TOKENS}" \
  --block-size=256 \
  --compilation-config='{"compile_sizes":[1,8,16,32,64,1360,2720]}' \
  --reasoning-parser=kimi_k3 \
  --tool-call-parser=kimi_k3 \
  --enable-auto-tool-choice \
  --async-scheduling \
  --trust-remote-code \
  "${ROLE_NODE_ARGS[@]}" \
  "${PREFIX_CACHE_ARGS[@]}"
