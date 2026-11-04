#!/usr/bin/env bash
# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0
#
# Launch moonshotai/Kimi-K3.1-Pro (2.8T total / 104B active MoE) on Ironwood TPU v7x
# Topology: 4 Hosts (2x2x4 = 16 physical v7x chips = 32 TensorCores, 3,072 GB HBM3e)
# Parallelism: TP=8, DP=4, EP=32 (Intra-Host TP=8 across 4 Data-Parallel Attention/KDA Replicas + Global EP=32)
# Best for: Maximum multi-stream decode throughput (4x attention/KDA DP replicas, 1.67M+ token KV pool)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

export MODEL_PATH="${MODEL_PATH:-/models/models/moonshotai/Kimi-K3.1-Pro}"
export TOKENIZER_PATH="${TOKENIZER_PATH:-/models/models/moonshotai/Kimi-K3.1-Pro}"
export SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-moonshotai/Kimi-K3.1-Pro}"
export PORT="${PORT:-8000}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-131072}"
export MAX_NUM_SEQS="${MAX_NUM_SEQS:-32}"
export MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-1360}"
export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.88}"
export ENABLE_PREFIX_CACHING="${ENABLE_PREFIX_CACHING:-1}"

# TorchTPU Kimi-K3.1-Pro kernel & collective optimizations
export TPU_VLLM_ENABLE_UNIFIED_BLOCK_POOL=1
export TPU_MOE_HIERARCHICAL_EP=1
export TPU_LATENT_PROJ_INTRA_CHIP_TP=1
export TPU_K3_SP_PREFILL=1

# Run:AI EP-filtered streaming loader tuning (loads only 28/896 local experts per TensorCore)
export RUNAI_STREAMER_PROCESS_GROUP_SIZE="${RUNAI_STREAMER_PROCESS_GROUP_SIZE:-8}"
export RUNAI_STREAMER_CONCURRENCY="${RUNAI_STREAMER_CONCURRENCY:-8}"
export RUNAI_STREAMER_MEMORY_LIMIT="${RUNAI_STREAMER_MEMORY_LIMIT:-8589934592}"
export VLLM_TPU_WEIGHT_LOAD_SYNC_INTERVAL="${VLLM_TPU_WEIGHT_LOAD_SYNC_INTERVAL:-100}"

# Runtime & compilation cache directories in RAM disk (/dev/shm)
mkdir -p /dev/shm/hf_cache /dev/shm/vllm_tmp /dev/shm/vllm_cache /dev/shm/torchinductor
export HF_HOME="/dev/shm/hf_cache"
export TMPDIR="/dev/shm/vllm_tmp"
export VLLM_CACHE_ROOT="/dev/shm/vllm_cache"
export VLLM_XLA_CACHE_PATH="/dev/shm/vllm_cache"
export TORCHINDUCTOR_CACHE_DIR="/dev/shm/torchinductor"
export TORCHINDUCTOR_AUTOGRAD_CACHE=0

export TPU_MULTIHOST_BACKEND="ray"
export TPU_PROCESS_ADDRESSES=""
export TPU_SKIP_MDS_QUERY=1
export RAY_EXPERIMENTAL_NOSET_TPU_VISIBLE_CHIPS=1
export VLLM_PLUGINS="torchtpu,torchtpu_layers"
export VLLM_TARGET_DEVICE="tpu"
export PJRT_DEVICE="TPU"
export VLLM_WORKER_MULTIPROC_METHOD="spawn"
export VLLM_NO_USAGE_STATS=1
export TORCH_DIST_TIMEOUT=1800
export VLLM_RPC_TIMEOUT=1800000
export VLLM_SHM_BROADCAST_TIMEOUT_S=1800
export VLLM_ENGINE_ITERATION_TIMEOUT_S=1800
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800
export VLLM_ENGINE_READY_TIMEOUT_S=7200
export VLLM_RAY_EXTRA_ENV_VARS_TO_COPY="VLLM_PLUGINS,RUNAI_STREAMER_PROCESS_GROUP_SIZE,RUNAI_STREAMER_CONCURRENCY,RUNAI_STREAMER_MEMORY_LIMIT,VLLM_TPU_WEIGHT_LOAD_SYNC_INTERVAL,TORCH_DIST_TIMEOUT,TMPDIR,HF_HOME,TORCHINDUCTOR_CACHE_DIR,TORCHINDUCTOR_AUTOGRAD_CACHE,TPU_VLLM_ENABLE_UNIFIED_BLOCK_POOL,TPU_MOE_HIERARCHICAL_EP,TPU_LATENT_PROJ_INTRA_CHIP_TP,TPU_K3_SP_PREFILL"

# Apply Kimi-K3.1-Pro architecture & 120 DSA indexer tensor patch before launching
python3 "${REPO_ROOT}/scripts/patches/patch_vllm_torchtpu_kimi_k31.py"

PREFIX_CACHE_ARGS=(--enable-prefix-caching --mamba-cache-mode=align)
if [[ "${ENABLE_PREFIX_CACHING}" == "0" ]]; then
  PREFIX_CACHE_ARGS=(--no-enable-prefix-caching)
fi

exec vllm serve "${MODEL_PATH}" \
  --tokenizer="${TOKENIZER_PATH}" \
  --served-model-name="${SERVED_MODEL_NAME}" \
  --host=0.0.0.0 \
  --port="${PORT}" \
  --api-server-count=1 \
  --attention-backend=CUSTOM \
  --tensor-parallel-size=8 \
  --data-parallel-size=4 \
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
  --compilation-config='{"compile_sizes":[1,8,32,1360]}' \
  "${PREFIX_CACHE_ARGS[@]}" \
  --enable-prompt-tokens-details \
  --reasoning-parser=kimi_k3 \
  --tool-call-parser=kimi_k3 \
  --enable-auto-tool-choice \
  --async-scheduling \
  --trust-remote-code
