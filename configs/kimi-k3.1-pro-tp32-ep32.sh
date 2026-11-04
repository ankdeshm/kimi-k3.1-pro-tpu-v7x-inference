#!/usr/bin/env bash
# ==============================================================================
# Moonshot AI Kimi-K3.1-Pro (2.8T Total / 104B Active MoE)
# Topology: TP=32, DP=1, EP=32 on TPU v7x 2x2x4 (16 Physical Chips = 32 TensorCores)
# ==============================================================================
# Architecture:
#   - 93 Hybrid Layers: 69 KDA (Kimi Delta Attention) + 24 Gated MLA / DSA Layers
#   - 896 Routed Experts (top-8 per token, native MXFP4) + 2 Shared Experts (BF16)
#   - Single 32-way Tensor Parallel Replica + 32-way Expert Parallelism + SP Prefill
#   - Single Unified KV + Mamba State Pool (>1.36M tokens capacity across slice)
# ==============================================================================
set -euo pipefail

export MODEL_URI="${MODEL_URI:-gs://<YOUR_GCS_MODEL_BUCKET>/models/moonshotai/Kimi-K3.1-Pro}"
export TOKENIZER_DIR="${TOKENIZER_DIR:-/models/models/moonshotai/Kimi-K3.1-Pro}"
export SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-moonshotai/Kimi-K3.1-Pro}"
export PORT="${PORT:-8000}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-131072}"
export MAX_NUM_SEQS="${MAX_NUM_SEQS:-64}"
export MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-2720}"
export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.88}"
export ENABLE_PREFIX_CACHING="${ENABLE_PREFIX_CACHING:-1}"

# 1. Apply Kimi-K3.1-Pro runtime patch (must also be run on all Ray worker hosts prior to ray start)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python3 "${SCRIPT_DIR}/../scripts/patches/patch_vllm_torchtpu_kimi_k31.py"

# 2. TPU v7x & vllm-torchtpu Environment Variables
ulimit -c 0 || true
mkdir -p /dev/shm/hf_cache /dev/shm/vllm_tmp /dev/shm/vllm_cache /dev/shm/torchinductor
unset TORCH_TPU_TOPOLOGY

export PJRT_DEVICE=TPU
export VLLM_TARGET_DEVICE=tpu
export VLLM_PLUGINS="torchtpu,torchtpu_layers"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_NO_USAGE_STATS=1
export TPU_MULTIHOST_BACKEND=ray
export TPU_PROCESS_ADDRESSES=""
export TPU_SKIP_MDS_QUERY=1
export RAY_EXPERIMENTAL_NOSET_TPU_VISIBLE_CHIPS=1
export TPU_UNIFIED_KV_CACHE_LAYOUT=1
export TPU_VLLM_ENABLE_UNIFIED_BLOCK_POOL=1
export TPU_MOE_HIERARCHICAL_EP=1
export TPU_LATENT_PROJ_INTRA_CHIP_TP=1
# TP=32, DP=1, EP=32 supports Sequence-Parallel Prefill across all 32 TensorCores
export TPU_K3_SP_PREFILL=1
export LIBTPU_INIT_ARGS="${LIBTPU_INIT_ARGS:---xla_tpu_buffer_assignment_enable_fast_merge_fallback=false --xla_tpu_offload_gather_to_sparsecore=true --xla_tpu_scoped_vmem_limit_kib=65536}"

# SGLang-Inspired Hardware & Pallas Kernel Optimizations for vllm-torchtpu:
# 1. Fused Pallas MoE Decode Kernel & One-Shot Greedy Top-Tokens Gather
export FUSED_MOE_DECODE_KERNEL="${FUSED_MOE_DECODE_KERNEL:-1}"
export FUSED_MOE_DECODE_KERNEL_MAX_TOKENS="${FUSED_MOE_DECODE_KERNEL_MAX_TOKENS:-256}"
export TPU_MOE_SKIP_PADDED_TOKENS="${TPU_MOE_SKIP_PADDED_TOKENS:-1}"
export TPU_GREEDY_TOP_TOKENS="${TPU_GREEDY_TOP_TOKENS:-1}"
# 2. MegaKDA Pallas DMA Overlap & Head-Inversion Packing (69 KDA Linear Attention Layers)
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
export TORCH_DIST_TIMEOUT=1800
export VLLM_RPC_TIMEOUT=1800000
export VLLM_SHM_BROADCAST_TIMEOUT_S=1800
export VLLM_ENGINE_ITERATION_TIMEOUT_S=1800
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800
export VLLM_ENGINE_READY_TIMEOUT_S=7200
export VLLM_RAY_EXTRA_ENV_VARS_TO_COPY="VLLM_PLUGINS,RUNAI_STREAMER_PROCESS_GROUP_SIZE,RUNAI_STREAMER_CONCURRENCY,RUNAI_STREAMER_MEMORY_LIMIT,VLLM_TPU_WEIGHT_LOAD_SYNC_INTERVAL,TORCH_DIST_TIMEOUT,TMPDIR,HF_HOME,TORCHINDUCTOR_CACHE_DIR,TORCHINDUCTOR_AUTOGRAD_CACHE,TPU_VLLM_ENABLE_UNIFIED_BLOCK_POOL,TPU_MOE_HIERARCHICAL_EP,TPU_LATENT_PROJ_INTRA_CHIP_TP,TPU_K3_SP_PREFILL,LIBTPU_INIT_ARGS,FUSED_MOE_DECODE_KERNEL,FUSED_MOE_DECODE_KERNEL_MAX_TOKENS,TPU_MOE_SKIP_PADDED_TOKENS,TPU_GREEDY_TOP_TOKENS,KDA_OVERLAP_H0_DMA,KDA_OVERLAP_HT_DMA,KDA_PACK_HEAD_INV,KDA_PACKED_METADATA"

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

echo "=== Launching moonshotai/Kimi-K3.1-Pro (TP=32, DP=1, EP=32) on TPU v7x (32 TensorCores) ==="
exec vllm serve "${MODEL_URI}" \
  --tokenizer="${TOKENIZER_DIR}" \
  --served-model-name="${SERVED_MODEL_NAME}" \
  --host=0.0.0.0 \
  --port="${PORT}" \
  --api-server-count=1 \
  --attention-backend=CUSTOM \
  --tensor-parallel-size=32 \
  --data-parallel-size=1 \
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
  --compilation-config='{"compile_sizes":[1,16,32,64,1360,2720]}' \
  --reasoning-parser=kimi_k3 \
  --tool-call-parser=kimi_k3 \
  --enable-auto-tool-choice \
  --async-scheduling \
  --trust-remote-code \
  "${PREFIX_CACHE_ARGS[@]}"
