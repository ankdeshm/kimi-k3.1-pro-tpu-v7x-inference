#!/usr/bin/env python3
# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0
"""SGLang-TorchTPU Production Recipe for moonshotai/Kimi-K3.1-Pro on TPU v7x (tp32/ep8).

Run on every host of a 4-host TPU v7x 2x2x4 slice (16 physical chips = 32 TensorCores):

    python3 -m benchmark.tpu.launch_server \\
        --config configs/sglang-torchtpu-kimi-k3.1-pro-tp32-ep8.py

Key TPU v7x (Ironwood) Optimizations:
1. Geometry (tp_size=32, ep_size=8): Each of the 896 routed experts is split across 4 TensorCore
   ranks, eliminating the slowest-rank straggler in the decode MoE all-reduce.
2. True Sparse MLA / DSA Pallas Backend (attention_backend="pallas_tpu_mla_sparse",
   kv_cache_dtype="fp8_e4m3", SGLANG_TPU_ENABLE_QBLOCK_SPARSE_MLA_PREFILL="1"): Executes the
   24-layer DeepSeek Sparse Attention (DSA) indexer (top-2,048 sparse MLA) in FP8.
3. Ironwood SparseCore Gather Offload (--xla_tpu_offload_gather_to_sparsecore=true,
   --xla_tpu_buffer_assignment_enable_fast_merge_fallback=false): Offloads the DSA top-2,048
   token remap gathers from TensorCore to the TPU v7x SparseCore (~100 ms/layer saved at long context).
4. DSA Indexer Token-Split Across TP32 (SGLANG_TPU_DSA_INDEXER_TOKEN_SPLIT="1"): Each TP rank
   scores 1/32 of the prefill chunk's rows in the DSA indexer and reassembles via all-reduce
   (cutting cold 64K TTFT from 17.6 s -> 13.5 s and 128K TTFT from 37.9 s -> 27.0 s).
5. Per-Op Decode Pallas Kernels & Two-Level All-Reduce:
   - SGLANG_TPU_K3_STATIC_FP8="moe_front,latent_up,kda_qkvg"
   - SGLANG_TPU_ENABLE_K3_HOST_SPLIT_GEMV="1"
   - SGLANG_TPU_ENABLE_TWOLEVEL_ALLREDUCE="1", SGLANG_TPU_TWOLEVEL_ALLREDUCE_SPLIT_BARRIER="1"
   - SGLANG_K3_MERGE_MOE_ALLREDUCE="1", SGLANG_TPU_ENABLE_K3_ATTN_RES_KERNEL="1"
   - SGLANG_TPU_K3_QKVG_PREFETCH="1", SGLANG_TPU_K3_LATENT_UP_ATTN_RES="1"
   - SGLANG_TPU_K3_OPROJ_ROW="1", SGLANG_TPU_LOGITS_ONESHOT_GATHER="1"
6. Lean Host Runtime Dispatch:
   - SGLANG_TPU_DECODE_SKIP_GUARD_EVAL="1", SGLANG_TPU_AOT_LEAN_RUNTIME_WRAPPER="1"
   - SGLANG_TPU_SPLIT_EXEC_DIRECT_CALL="1", SGLANG_TPU_PALLAS_VMEM_RIGHTSIZE="1"
"""

import os

CONCURRENCY = int(os.environ.get("CONCURRENCY", "64"))
PREFILL_TOKENS = int(os.environ.get("PREFILL_TOKENS", "4096"))
CONTEXT_LENGTH = int(os.environ.get("CONTEXT_LENGTH", str(131072 + 8192)))
MODEL_PATH = os.environ.get("MODEL_PATH", "/models/models/moonshotai/Kimi-K3.1-Pro")
DECODE_BS = [b for b in [1, 2, 4, 8, 16, 32, 64] if b <= CONCURRENCY]
if CONCURRENCY not in DECODE_BS:
    DECODE_BS.append(CONCURRENCY)

CONFIG = {
    "server_args": {
        "model_path": MODEL_PATH,
        "device": "tpu",
        "dtype": "bfloat16",
        "host": "0.0.0.0",
        "port": int(os.environ.get("PORT", "30000")),
        "nnodes": 4,
        "tp_size": 32,
        "ep_size": 8,
        "quantization": None,
        "grammar_backend": "none",
        "attention_backend": "pallas_tpu_mla_sparse",
        "kv_cache_dtype": "fp8_e4m3",
        "page_size": 128,
        "context_length": CONTEXT_LENGTH,
        "max_running_requests": CONCURRENCY,
        "torch_compile_max_bs": CONCURRENCY,
        "enable_torch_compile": True,
        "tpu_compile_bs": DECODE_BS,
        "chunked_prefill_size": PREFILL_TOKENS,
        "max_prefill_tokens": PREFILL_TOKENS,
        "tpu_compile_tokens": [PREFILL_TOKENS],
        "prefill_max_requests": 8,
        "mem_fraction_static": float(os.environ.get("MEM_FRACTION_STATIC", "0.85")),
        "max_total_tokens": 4000000,
        "disable_radix_cache": os.environ.get("DISABLE_RADIX_CACHE", "0") == "1",
        "enable_cache_report": True,
        "skip_server_warmup": True,
        "trust_remote_code": True,
        "reasoning_parser": "kimi_k3",
        "tool_call_parser": "kimi_k3",
        "mamba_ssm_dtype": "bfloat16",
        "scheduler_recv_interval": 16,
        "weight_loader_disable_mmap": True,
        "watchdog_timeout": 3600,
    },
    "runtime_env": {
        "SGLANG_PLATFORM": "tpu",
        "HF_HUB_OFFLINE": "0",
        "TRANSFORMERS_OFFLINE": "0",
        "TORCH_LOGS": "+recompiles",
        "TPU_ENABLE_DIAGNOSTIC_COORDINATOR": "0",
        "TPU_DISABLE_DIAGNOSTIC_COORDINATOR": "1",
        "TPU_DIAGNOSTIC_HARVESTER_INTERVAL_SEC": "86400",
        "SGLANG_TPU_PRELOAD_TELEMETRY_STUB": "1",
        "SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION": "0",
        "SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN": "1",
        "SGLANG_TPU_OVERLAP_PROFILE": "1",
        "TORCH_TPU_INTERNAL_MATERIALIZE_COLLECTIVE_TENSORS": "false",
        "TPU_ENABLE_TELEMETRY_HARVESTING": "0",
        "TPU_ENABLE_TELEMETRY": "0",
        "ENABLE_TPU_TELEMETRY": "0",
        "TPU_TELEMETRY_INTERVAL_SECONDS": "86400",
        "TPU_TELEMETRY_INTERVAL_SEC": "86400",
        "LIBTPU_INIT_ARGS": (
            "--xla_tpu_enable_async_collective_fusion=true "
            "--xla_tpu_overlap_compute_collective_tc=true "
            "--xla_tpu_enable_async_collective_fusion_multiple_steps=true "
            "--xla_tpu_enable_latency_hiding_scheduler=true "
            "--xla_latency_hiding_scheduler_rerun=10 "
            "--xla_tpu_all_gather_collective_matmul_mode=post_spmd_conservative "
            "--xla_tpu_reduce_scatter_collective_matmul_mode=post_spmd_conservative "
            "--xla_tpu_enable_sparse_core_collective_offload_all_reduce=false "
            "--xla_tpu_enable_sparse_core_collective_offload_reduce_scatter=false "
            "--xla_tpu_buffer_assignment_enable_fast_merge_fallback=false "
            "--xla_tpu_offload_gather_to_sparsecore=true"
        ),
        "SGLANG_TPU_EAGER_MODE": "DEFER_AND_FUSE",
        "SGLANG_TPU_FUSED_MOE_PERMUTE_DENSE_THRESHOLD": "512",
        "SGLANG_TPU_K3_STATIC_FP8": "moe_front,latent_up,kda_qkvg",
        "SGLANG_TPU_ENABLE_K3_HOST_SPLIT_GEMV": "1",
        "SGLANG_TPU_ENABLE_TWOLEVEL_ALLREDUCE": "1",
        "SGLANG_K3_MERGE_MOE_ALLREDUCE": "1",
        "SGLANG_TPU_ENABLE_K3_ATTN_RES_KERNEL": "1",
        "SGLANG_TPU_ENABLE_NN_MODULE_GUARD_FILTER": "1",
        "SGLANG_TPU_ENABLE_QBLOCK_SPARSE_MLA_PREFILL": "1",
        "SGLANG_TPU_DSA_INDEXER_TOKEN_SPLIT": "1",
        "SGLANG_TPU_VL_TEXT_ONLY_EXTEND": "1",
        "SGLANG_SET_CPU_AFFINITY": "1",
        "SGLANG_TPU_ASYNC_COPY_WORKER": "0",
        "SGLANG_TPU_TWOLEVEL_ALLREDUCE_SPLIT_BARRIER": "1",
        "SGLANG_TPU_K3_QKVG_PREFETCH": "1",
        "SGLANG_TPU_K3_LATENT_UP_ATTN_RES": "1",
        "SGLANG_TPU_K3_OPROJ_ROW": "1",
        "SGLANG_TPU_DYNAMO_PARAMS_AS_GRAPH_ATTRS": "2",
        "SGLANG_TPU_ALLOC_DECODE_FAST_PATH": "1",
        "SGLANG_TPU_LOGITS_ONESHOT_GATHER": "1",
        "SGLANG_TPU_DECODE_SKIP_GUARD_EVAL": "1",
        "SGLANG_TPU_AOT_LEAN_RUNTIME_WRAPPER": "1",
        "SGLANG_TPU_SPLIT_EXEC_DIRECT_CALL": "1",
        "SGLANG_TPU_PALLAS_VMEM_RIGHTSIZE": "1",
    },
}
