#!/usr/bin/env bash
# ==============================================================================
# Moonshot AI Kimi-K3.1-Pro (2.8T Total / 104B Active MoE)
# SGLang-TorchTPU Decode MegaKernel + Optional DSpark / DFlash2 Speculative Decoding
# Topology: TP=32, EP=8 on TPU v7x 2x2x4 (16 Physical Chips = 32 TensorCores)
# ==============================================================================
# The Decode MegaKernel (sglang_tpu/k3_megakernel) fuses all 93 hybrid decoder
# layers (69 KDA + 24 MLA) into a single Pallas kernel invocation per decode step
# and owns their weights in HBM/VMEM (serving batch=1 ultra-low-latency streams).
# ==============================================================================
set -euo pipefail

export MODEL_PATH="${MODEL_PATH:-/models/models/moonshotai/Kimi-K3.1-Pro}"
export DRAFT_MODEL_PATH="${DRAFT_MODEL_PATH:-}"
export NODE_RANK="${NODE_RANK:-${TPU_WORKER_ID:-0}}"
export DIST_INIT_ADDR="${DIST_INIT_ADDR:-127.0.0.1:10011}"
export PORT="${PORT:-30000}"
export PREFILL_TOKENS="${PREFILL_TOKENS:-1024}"
export CONTEXT_LENGTH="${CONTEXT_LENGTH:-9280}"

export SGLANG_PLATFORM="tpu"
export HF_HUB_OFFLINE="0"
export TRANSFORMERS_OFFLINE="0"
export TORCH_LOGS="+recompiles"
export TPU_ENABLE_DIAGNOSTIC_COORDINATOR="0"
export TPU_DISABLE_DIAGNOSTIC_COORDINATOR="1"
export TPU_DIAGNOSTIC_HARVESTER_INTERVAL_SEC="86400"
export SGLANG_TPU_PRELOAD_TELEMETRY_STUB="1"
export SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION="0"
export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN="1"
export SGLANG_TPU_OVERLAP_PROFILE="1"
export TORCH_TPU_INTERNAL_MATERIALIZE_COLLECTIVE_TENSORS="false"
export TPU_ENABLE_TELEMETRY_HARVESTING="0"
export TPU_ENABLE_TELEMETRY="0"
export ENABLE_TPU_TELEMETRY="0"
export TPU_TELEMETRY_INTERVAL_SECONDS="86400"
export TPU_TELEMETRY_INTERVAL_SEC="86400"

export LIBTPU_INIT_ARGS="--xla_tpu_enable_async_collective_fusion=true --xla_tpu_overlap_compute_collective_tc=true --xla_tpu_enable_async_collective_fusion_multiple_steps=true --xla_tpu_enable_latency_hiding_scheduler=true --xla_latency_hiding_scheduler_rerun=10 --xla_tpu_all_gather_collective_matmul_mode=post_spmd_conservative --xla_tpu_reduce_scatter_collective_matmul_mode=post_spmd_conservative --xla_tpu_enable_sparse_core_collective_offload_all_reduce=false --xla_tpu_offload_gather_to_sparsecore=false --xla_tpu_enable_sparse_core_collective_offload_reduce_scatter=false"

export SGLANG_TPU_EAGER_MODE="DEFER_AND_FUSE"
export SGLANG_TPU_ENABLE_K3_DECODE_MEGAKERNEL="1"
export SGLANG_TPU_OPT_K3_MEGAKERNEL_RELAX="dense_fp8,pair_oproj,bf16_transport"

SPEC_ARGS=()
if [[ -n "${DRAFT_MODEL_PATH}" ]]; then
  export SGLANG_TPU_ENABLE_EXPERIMENTAL_SPEC_WORKER="1"
  export SGLANG_RAGGED_VERIFY_MODE="static"
  SPEC_ARGS=(
    --speculative-algorithm DSPARK
    --speculative-draft-model-path "${DRAFT_MODEL_PATH}"
    --speculative-dspark-block-size 7
    --speculative-draft-attention-backend pallas_tpu_rpa_v3
    --max-mamba-cache-size 1
  )
fi

echo "=== Launching SGLang-TorchTPU Kimi-K3.1-Pro Decode MegaKernel (TP=32, EP=8, node_rank=${NODE_RANK}) ==="
exec python3 -u -m sglang.launch_server \
  --model-path "${MODEL_PATH}" \
  --device tpu \
  --dtype bfloat16 \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --nnodes 4 \
  --node-rank "${NODE_RANK}" \
  --dist-init-addr "${DIST_INIT_ADDR}" \
  --tp-size 32 \
  --ep-size 8 \
  --grammar-backend none \
  --attention-backend pallas_tpu_mla \
  --page-size 128 \
  --max-running-requests 1 \
  --chunked-prefill-size "${PREFILL_TOKENS}" \
  --max-prefill-tokens "${PREFILL_TOKENS}" \
  --context-length "${CONTEXT_LENGTH}" \
  --torch-compile-max-bs 1 \
  --enable-torch-compile \
  --tpu-compile-tokens "${PREFILL_TOKENS}" \
  --tpu-compile-bs 1 \
  --disable-radix-cache \
  --mem-fraction-static 0.9 \
  --trust-remote-code \
  --reasoning-parser kimi_k3 \
  --tool-call-parser kimi_k3 \
  --mamba-ssm-dtype bfloat16 \
  --scheduler-recv-interval 16 \
  --weight-loader-disable-mmap \
  --watchdog-timeout 3600 \
  "${SPEC_ARGS[@]}"
