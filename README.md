# Moonshot AI `Kimi-K3.1-Pro` (`2.8T` / `104B` Active MoE) Inference & Benchmarking on Google Cloud TPU v7x (Ironwood)

Production serving recipes, runtime patches, Kubernetes LeaderWorkerSet (`LWS`) manifests, and comprehensive performance benchmarks for **`moonshotai/Kimi-K3.1-Pro`** (`2.8T` total parameters / `104B` active parameters per token) on **Google Cloud TPU v7x (Ironwood)** using **`vllm-torchtpu`**.

---

## 1. Model Architecture & TPU v7x (`2x2x4`) Hardware Mapping

### 1.1 `moonshotai/Kimi-K3.1-Pro` Architecture Breakdown

| Parameter / Subsystem | Specification |
| :--- | :--- |
| **Hugging Face Repository** | [`moonshotai/Kimi-K3.1-Pro`](https://huggingface.co/moonshotai/Kimi-K3.1-Pro) |
| **Model Class & Type** | `KimiK31ForConditionalGeneration` (`model_type: "kimi_k31"`, `text_config.model_type: "kimi_k3"`) |
| **Total / Active Parameters** | **`2.8T` Total** / **`104B` Active** per token |
| **Checkpoint Size on Disk** | **`1,561.36 GB` (`1.56 TB`)** across `96` safetensors shards (`111` total files) |
| **Decoder Layers (`93` Total)** | **`69` KDA (Kimi Delta Attention)** Linear Attention layers + **`24` Gated MLA / DSA** layers (`3:1` hybrid ratio; MLA at layers `4, 8, 12, ..., 92` and layer `93`) |
| **Hidden Dimension & Vocab** | `hidden_size = 7168`, `vocab_size = 163840`, `rms_norm_eps = 1e-5` |
| **KDA Linear Attention (`69` Layers)** | `num_heads = 64`, `head_dim = 128`, `short_conv_kernel_size = 4`, recurrent state `[64, 128, 128]` per sequence |
| **Gated MLA / DSA (`24` Layers)** | `num_attention_heads = 64`, `kv_lora_rank = 512`, `q_lora_rank = 1536`, `qk_nope_head_dim = 128`, `qk_rope_head_dim = 64`, `v_head_dim = 128`, `gated_attention = True`, YaRN (`factor = 64.0`, `max_position_embeddings = 262144`) |
| **MoE Architecture (`92` MoE Layers)** | `n_routed_experts = 896`, `num_experts_per_tok = 8`, `n_group = 8`, `topk_group = 4`, `routed_scaling_factor = 3.0`, `moe_intermediate_size = 2048`, `n_shared_experts = 2`, `first_k_dense_replace = 1` (Layer `0` dense `intermediate_size = 18432`) |
| **Native Checkpoint Precision** | **Hybrid MXFP4 + BF16**: All `896` routed experts (`gate_proj`, `up_proj`, `down_proj`) are natively quantized in **`compressed-tensors` `mxfp4-pack-quantized`** (`4-bit` group-32 E8M0 block scales); shared experts, embeddings, norms, KDA, and Gated MLA projections are **`BF16`** |

### 1.2 Why `Kimi-K3.1-Pro` Fits on a Single `2x2x4` (`16`-Chip / `32`-TensorCore) TPU v7x Slice

A TPU v7x `2x2x4` slice consists of `4` `tpu7x-standard-4t` nodes (`16` physical chips = `32` logical XLA TensorCores), providing **`3,072 GB` (`3,031.68 GiB` usable) of HBM3e** at **`117.96 TB/s` aggregate HBM3e bandwidth**:

| Parallelism Topology | Sharding Layout across `32` TensorCores (`2x2x4`) | Weight + XLA HBM per TensorCore | Available Unified KV + KDA State HBM per Core (`gpu_memory_utilization=0.88`) | Total Token Capacity Across Slice (`block_size=1360`, FP8 KV) | Primary Use Case |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Config A: `TP=8, DP=4, EP=32`** | `4` independent `TP=8` DP replicas (`1` per host) + `32`-way Expert Parallelism (`28` routed experts/core) across the `2x2x4` ICI torus | **`68.26 GiB` / `94.74 GiB`** | **`15.11 GiB` / core** (`~492` unified blocks/replica) | **`~669,000` tokens / DP replica** (`~2.67M` tokens across `4` DP replicas) | **High-Throughput Serving** (4× independent attention engines + 8-way intra-host attention All-Reduce) |
| **Config B: `TP=32, DP=1, EP=32`** | Single `32`-way Tensor Parallel replica + `32`-way Expert Parallelism (`28` routed experts/core) + `TPU_K3_SP_PREFILL=1` sequence-parallel prefill | **`~52.50 GiB` / `94.74 GiB`** | **`~30.87 GiB` / core** (`~1,005` unified blocks) | **`>1,366,000` tokens in a single unified pool** | **Lowest Single-Stream Latency & Ultra-Long Context (`64k–128k+`)** |

---

## 2. How `vllm-torchtpu` Executes `moonshotai/Kimi-K3.1-Pro`

### 2.1 Unified Hybrid KV + KDA Recurrent State Cache (`TPU_UNIFIED_KV_CACHE_LAYOUT=1`)

Because `Kimi-K3.1-Pro` interleaves `69` KDA linear-attention layers with `24` Gated MLA full-attention layers (`3:1` ratio), managing separate physical HBM pools for standard paged KV caches and fixed-size recurrent Mamba/KDA states wastes HBM fragmentation. `vllm-torchtpu` implements `KimiUnifiedAttention`:
- **Unified Physical Block (`block_size = 1360` tokens)**: Each physical block stores `2` MLA KV tokens alongside `1` KDA recurrent/conv state slot (`[64, 128, 128]` + conv state).
- **`24` Raw HBM Tensors per TensorCore**: One unified tensor per MLA layer (`24` layers total), dynamically serving both the `24` Gated MLA layers and `69` KDA layers without HBM waste.
- **Prefix Caching (`--enable-prefix-caching --mamba-cache-mode align`)**: Aligns KDA recurrent state checkpoints on block boundaries (`1360` tokens) so multi-turn and shared-prefix prompts reuse both Gated MLA KV blocks and KDA linear-attention recurrent states.

### 2.2 Runtime Patch (`scripts/patches/patch_vllm_torchtpu_kimi_k31.py`)

Applying [`scripts/patches/patch_vllm_torchtpu_kimi_k31.py`](scripts/patches/patch_vllm_torchtpu_kimi_k31.py) on all 4 TPU hosts before `ray start` and `vllm serve` performs five idempotent surgical updates:
1. **Registers `KimiK31ForConditionalGeneration`**: Maps `KimiK31ForConditionalGeneration -> ("kimi_k3", "KimiK3ForCausalLM")` in `vllm_torchtpu/models/__init__.py` and `vllm/model_executor/models/registry.py`.
2. **Registers `model_type: "kimi_k31"`**: Maps `"kimi_k31": KimiK3Config` in `vllm/transformers_utils/configs/__init__.py` and `"kimi_k31": "kimi_linear"` in `vllm/transformers_utils/config.py`.
3. **Enables Unified KV Layout**: Adds `"KimiK31ForConditionalGeneration"` to `UNIFIED_KV_LAYOUT_ARCHITECTURES` in `vllm_torchtpu/Selector.py`.
4. **Filters DSA Indexer Tensors (`120` Tensors)**: Maps `".self_attn.indexer.": None` in `KimiLinearForCausalLM.hf_to_vllm_mapper` and skips `.self_attn.indexer.` in `_ordinary_weights()` so TPU `KimiMLAAttention` executes exact dense Gated MLA (`indexer=None, is_sparse=False`) without unexpected-weight assertion errors.
5. **Guards `TPU_K3_SP_PREFILL` for `TP=8, DP=4, EP=32`**: Automatically sets `TPU_K3_SP_PREFILL=0` when `TP != 32` or `DP != 1` to prevent `ValueError("K3 SP prefill requires TP32/EP32 and PP1/DP1")`.

### 2.3 Critical Hybrid KDA + MLA Scheduler Discovery (`--max-num-batched-tokens=2720`)

In `vllm-torchtpu`, `Kimi-K3.1-Pro` aligns its hybrid Mamba/KDA recurrent state slot (`870,400` bytes) with the FP8 MLA KV cache block slot (`640` bytes/token), deriving a unified `block_size = 1,360` tokens.
- Under `--mamba-cache-mode=align`, the vLLM v1 scheduler requires any partial prefill chunk of a multi-block prompt (`> 1,360` tokens, such as `8k`, `64k`, `128k`) to be a multiple of `mamba_block_size = 1,360` tokens (`>= 1,360` tokens).
- If `--max-num-batched-tokens=1360`, having even **1 active decode request (`1` token)** reduces the step's remaining token budget to `1360 - 1 = 1359 < 1360`, which rounds down to `(1359 // 1360) * 1360 = 0` tokens. This causes severe head-of-line blocking where each DP engine can only run `1` request at a time for prompts `>= 1,360` tokens (`54.39` output tok/s plateau at `8k/1k`).
- **Fix**: Setting **`--max-num-batched-tokens=2720`** (`2 × 1,360`) with `--compilation-config='{"compile_sizes":[1,8,16,32,64,1360,2720]}'` guarantees `token_budget = 2720 - num_running_decode_reqs >= 2720 - 64 = 2656 >= 1360`, allowing every DP engine to schedule a full `1,360`-token prefill chunk alongside up to `64` active decode sequences and lifting `8k/1k` throughput from `54.39` to **`328.22` output tok/s (`2,979.53` total tok/s)**.

### 2.4 SGLang-Inspired Hardware & Pallas Kernel Optimizations Ported to `vllm-torchtpu`

By cross-auditing the `SGLang-TorchTPU` Day-0 cookbook (`origin/kimi-k31-pro-tpu`) against `vllm-torchtpu` (`src/vllm_torchtpu/`), we identified 5 hardware and Pallas kernel optimizations that can be directly enabled in `vllm-torchtpu` (configured in [`configs/kimi-k3.1-pro-tp32-ep32.sh`](configs/kimi-k3.1-pro-tp32-ep32.sh) and [`configs/kimi-k3.1-pro-tp8-dp4-ep32.sh`](configs/kimi-k3.1-pro-tp8-dp4-ep32.sh)):

| Optimization | `vllm-torchtpu` Flag / Setting | Hardware Mechanism on TPU v7x (Ironwood) |
| :--- | :--- | :--- |
| **A. Ironwood SparseCore Gather Offload** | `LIBTPU_INIT_ARGS="--xla_tpu_offload_gather_to_sparsecore=true --xla_tpu_buffer_assignment_enable_fast_merge_fallback=false --xla_tpu_scoped_vmem_limit_kib=65536"` | Offloads embedding and MoE routing gather operations from the main TensorCore MXU/VPU onto Ironwood's dedicated SparseCore engines |
| **B. `TP=32` Sequence-Parallel Prefill + Tuned v2 MLA Pallas Table** | `--tensor-parallel-size=32 --data-parallel-size=1 --block-size=256` + `TPU_K3_SP_PREFILL=1` | Activates `FusedPrefillCollectives` sequence parallelism across all 32 TensorCores and hits the tuned v2 MLA Pallas kernel table (`actual_num_q_heads = 96/32 = 3`, `kv_dtype = fp8`, `page_size_per_kv_packing = 64`) |
| **C. Fused Pallas MoE Decode Kernel & One-Shot Greedy Top-Tokens** | `FUSED_MOE_DECODE_KERNEL=1`, `FUSED_MOE_DECODE_KERNEL_MAX_TOKENS=256`, `TPU_MOE_SKIP_PADDED_TOKENS=1`, `TPU_GREEDY_TOP_TOKENS=1` | Uses dedicated small-batch Pallas MoE decode kernel (`<= 256` tokens), skips padded routing tokens, and gathers only top-1 greedy token IDs instead of full `163,840`-vocab logits |
| **D. MegaKDA Pallas DMA Overlap & Head-Inversion Packing** | `KDA_OVERLAP_H0_DMA=1`, `KDA_OVERLAP_HT_DMA=1`, `KDA_PACK_HEAD_INV=1`, `KDA_PACKED_METADATA=1` | Overlaps initial/final recurrent state (`h0`/`ht`) HBM DMAs with Pallas compute and packs head-inversion metadata across all `69` KDA layers |
| **E. Native TPU Sparse Attention Indexer (`VllmTPUSparseAttnIndexer`)** | `VLLM_TPU_INDEXER_KV_CACHE_MODE=sal`, `TPU_SPARSE_MLA_NOPE_LAYOUT=sparsecore`, `TPU_SPARSE_MLA_ROPE_LAYOUT=sparsecore`, `TPU_MLA_SPARSE_SPLIT_Q=1` | Executes `streamindex_topk` (`topk=2048`) with SparseCore strided KV layout and Q-split sparse MLA Pallas kernels |

### 2.5 `SGLang-TorchTPU` (`TP=32, EP=8`) Day-0 Recipe (`configs/sglang-torchtpu-kimi-k3.1-pro-tp32-ep8.py`)

We also include the `SGLang-TorchTPU` (`origin/kimi-k31-pro-tpu`) Day-0 serving recipe ([`configs/sglang-torchtpu-kimi-k3.1-pro-tp32-ep8.py`](configs/sglang-torchtpu-kimi-k3.1-pro-tp32-ep8.py) and [`configs/sglang-torchtpu-kimi-k3.1-pro-megakernel.sh`](configs/sglang-torchtpu-kimi-k3.1-pro-megakernel.sh)):
- **`tp_size=32, ep_size=8`**: Shards the `896` MXFP4 routed experts 8-way across the EP sub-mesh and shards expert intermediate dimension (`2048`) 4-way across the remaining TP axis (`8 × 4 = 32` TensorCores).
- **`attention_backend="pallas_tpu_mla_sparse"`**: Executes DeepSeek Sparse Attention (`index_topk=2048`) on the `24` Gated MLA layers with `SGLANG_TPU_DSA_INDEXER_TOKEN_SPLIT=1` and SparseCore gather offload.
- **Per-Op Pallas Decode Kernels & Decode MegaKernel**: Enables `SGLANG_TPU_KDA_DECODE_PALLAS=1`, `SGLANG_TPU_MLA_DECODE_PALLAS=1`, `SGLANG_TPU_MOE_DECODE_PALLAS=1`, and single-stream `sglang_tpu.k3_megakernel` with `RadixArk/Kimi-K3.1-Dflash2-Preview` / `RadixArk/Kimi-K3-DSpark` speculative decoding.

---

## 3. Repository Structure

```text
kimi-k3.1-pro-tpu-v7x-inference/
├── README.md                                               # Architecture, memory math, & benchmark results
├── INFRA_SETUP.md                                          # Step-by-step GKE TPU v7x 2x2x4 & GCS FUSE setup guide
├── configs/
│   ├── kimi-k3.1-pro-tp8-dp4-ep32.sh                       # vllm-torchtpu Config A: TP=8, DP=4, EP=32
│   ├── kimi-k3.1-pro-tp32-ep32.sh                          # vllm-torchtpu Config B: TP=32, DP=1, EP=32 + SP Prefill
│   ├── sglang-torchtpu-kimi-k3.1-pro-tp32-ep8.py           # SGLang-TorchTPU: TP=32, EP=8 + SparseCore DSA
│   └── sglang-torchtpu-kimi-k3.1-pro-megakernel.sh         # SGLang-TorchTPU: Decode MegaKernel + Speculative Decoding
├── scripts/
│   ├── stage_hf_to_gcs.py                                  # Parallel multi-pod HF -> GCS staging & byte verifier
│   └── patches/
│       └── patch_vllm_torchtpu_kimi_k31.py                 # Idempotent vllm + vllm-torchtpu runtime patch
├── k8s/
│   ├── provision_gke_dws_calendar.sh                       # GKE + TPU v7x 2x2x4 node pool provisioning script
│   └── kimi-k3.1-pro-lws-v7x-16.yaml                       # 4-Pod LeaderWorkerSet manifest for 2x2x4 TPU v7x
├── benchmarks/
│   └── run_kimi_k31_matrix.py                              # Async OpenAI streaming benchmark harness (Waves 1-3)
└── results/
    └── vllm-tp8-dp4-ep32/                                  # Raw JSON & Markdown benchmark reports
```

---

## 4. Quick Start

### 1. Apply the Runtime Patch & Start Ray Cluster (All 4 Hosts)
On all 4 hosts of the `2x2x4` TPU v7x slice (`LWS_WORKER_INDEX=0..3`):
```bash
python3 scripts/patches/patch_vllm_torchtpu_kimi_k31.py
```

### 2. Launch `vllm serve` (Leader Host 0)
- **Throughput Mode (`TP=8, DP=4, EP=32`)**:
  ```bash
  MODEL_URI=/models/models/moonshotai/Kimi-K3.1-Pro \
  bash configs/kimi-k3.1-pro-tp8-dp4-ep32.sh
  ```
- **Single-Pool / Sequence-Parallel Prefill Mode (`TP=32, DP=1, EP=32`)**:
  ```bash
  MODEL_URI=/models/models/moonshotai/Kimi-K3.1-Pro \
  bash configs/kimi-k3.1-pro-tp32-ep32.sh
  ```

### 3. Run the Benchmark Matrix
```bash
# Run all 3 waves (Regular Context, Long Context, and 64k Prefix Caching)
python3 benchmarks/run_kimi_k31_matrix.py \
  --wave all \
  --config-tag tp8-dp4-ep32 \
  --output-dir results
```

---

## 5. Benchmark Matrix & Live TPU v7x (`2x2x4`) Results

> **Hardware**: `1×` TPU v7x `2x2x4` slice (`4` hosts × `4` chips = `16` physical chips = `32` TensorCores, `3,072 GB` HBM3e)  
> **Runtime**: `vllm-torchtpu` (`TPU_UNIFIED_KV_CACHE_LAYOUT=1`, `TPU_MULTIHOST_BACKEND=ray`, `kv_cache_dtype=fp8`, `--max-num-batched-tokens=2720`, native `compressed-tensors` `mxfp4-pack-quantized` routed experts + `BF16` shared experts/attention)  
> **Raw Data**: [`results/vllm-tp8-dp4-ep32/summary.json`](results/vllm-tp8-dp4-ep32/summary.json) | [`results/vllm-tp8-dp4-ep32/REPORT.md`](results/vllm-tp8-dp4-ep32/REPORT.md)

### 5.1 Wave 1: Regular Context (`1k / 1k`, `8k / 1k`) — `vllm-torchtpu` (`TP=8, DP=4, EP=32`)

| Workload (`ISL / OSL`) | Concurrency | Output Throughput (`tok/s`) | Total Throughput (`tok/s`) | Stream `tok/s` (Mean) | TTFT P50 (`ms`) | TTFT P95 (`ms`) | TTFT P99 (`ms`) | TPOT P50 (`ms`) | TPOT P95 (`ms`) | ITL P50 (`ms`) | E2EL P50 (`s`) | OK / Fail |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`1024 / 1024`** | `1` | **`15.25`** | **`31.63`** | `15.38` | `619.8` | `619.8` | `619.8` | `65.03` | `65.03` | `65.03` | `67.15` | `1 / 0` |
| **`1024 / 1024`** | `4` | **`58.09`** | **`120.48`** | `14.73` | `1,193.3` | `1,193.7` | `1,193.7` | `67.76` | `68.27` | `67.79` | `70.52` | `4 / 0` |
| **`1024 / 1024`** | `8` | **`104.96`** | **`217.71`** | `13.44` | `1,777.7` | `2,944.3` | `2,944.3` | `74.41` | `75.41` | `73.46` | `77.89` | `8 / 0` |
| **`1024 / 1024`** | `16` | **`196.01`** | **`406.55`** | `12.71` | `2,969.2` | `3,561.4` | `4,143.7` | `78.69` | `79.70` | `77.31` | `82.95` | `16 / 0` |
| **`1024 / 1024`** | `32` | **`364.03`** | **`755.06`** | `11.89` | `3,587.9` | `5,969.8` | `5,972.4` | `84.19` | `86.19` | `82.42` | `89.71` | `32 / 0` |
| **`1024 / 1024`** | `64` | **`609.35`** | **`1,263.91`** | `10.06` | `5,539.2` | `9,706.0` | `9,709.7` | `99.69` | `103.10` | `96.21` | `107.00` | `64 / 0` |
| **`8192 / 1024`** | `1` | **`14.49`** | **`131.48`** | `15.37` | `4,104.6` | `4,104.6` | `4,104.6` | `65.07` | `65.07` | `65.08` | `70.67` | `1 / 0` |
| **`8192 / 1024`** | `4` | **`54.31`** | **`492.85`** | `14.59` | `5,466.3` | `6,015.1` | `6,015.1` | `68.81` | `69.32` | `67.83` | `75.34` | `4 / 0` |
| **`8192 / 1024`** | `8` | **`93.95`** | **`852.91`** | `13.10` | `11,140.2` | `11,692.3` | `11,692.3` | `78.76` | `79.74` | `73.54` | `86.93` | `8 / 0` |
| **`8192 / 1024`** | `16` | **`156.14`** | **`1,417.41`** | `11.64` | `17,643.5` | `25,718.7` | `25,719.3` | `88.53` | `95.39` | `77.45` | `104.19` | `16 / 0` |
| **`8192 / 1024`** | `32` | **`241.34`** | **`2,190.81`** | `9.87` | `30,619.6` | `51,109.0` | `51,661.3` | `105.10` | `123.63` | `82.44` | `134.14` | `32 / 0` |
| **`8192 / 1024`** | `64` | **`328.22`** | **`2,979.53`** | `7.46` | `56,357.9` | `101,580.1` | `101,583.5` | `141.62` | `181.84` | `96.29` | `196.36` | `64 / 0` |

### 5.2 Wave 2: Long Context (`64k / 1k`, `64k / 4k`, `128k / 1k`) — `vllm-torchtpu` (`TP=8, DP=4, EP=32`)

> **Key Architectural Observation**: Because `69` of the `93` layers are constant-state linear KDA layers (`O(1)` decode memory bandwidth per token regardless of sequence length), single-stream decode latency at `64k` context (`65.60 ms` TPOT = `15.24 tok/s`) degrades by **`< 1%`** compared to `1k` context (`65.03 ms` TPOT = `15.38 tok/s`). At `c=16` (`>1.05M` active context tokens across `4` DP replicas), aggregate total throughput reaches **`3,820.36 tok/s`** while using only **`33.3%`** of the unified KV+KDA cache pool.

| Workload (`ISL / OSL`) | Concurrency | Output Throughput (`tok/s`) | Total Throughput (`tok/s`) | Stream `tok/s` (Mean) | TTFT P50 (`ms`) | TTFT P95 (`ms`) | TTFT P99 (`ms`) | TPOT P50 (`ms`) | TPOT P95 (`ms`) | ITL P50 (`ms`) | E2EL P50 (`s`) | OK / Fail |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`65536 / 1024`** | `1` | **`10.51`** | **`684.14`** | `15.24` | `30,295.5` | `30,295.5` | `30,295.5` | `65.60` | `65.60` | `65.60` | `97.41` | `1 / 0` |
| **`65536 / 1024`** | `2` | **`20.44`** | **`1,330.16`** | `14.92` | `31,294.3` | `31,886.3` | `31,886.3` | `66.77` | `67.28` | `66.76` | `100.12` | `2 / 0` |
| **`65536 / 1024`** | `4` | **`38.95`** | **`2,534.81`** | `14.49` | `35,292.7` | `35,292.7` | `35,292.7` | `68.29` | `71.36` | `68.33` | `105.15` | `4 / 0` |
| **`65536 / 1024`** | `8` | **`49.39`** | **`3,214.15`** | `10.67` | `86,849.2` | `89,813.8` | `89,813.8` | `123.90` | `127.48` | `74.65` | `165.64` | `8 / 0` |
| **`65536 / 1024`** | `16` | **`58.70`** | **`3,820.36`** | `7.61` | `141,473.1` | `199,075.8` | `199,076.2` | `177.74` | `227.43` | `79.00` | `275.57` | `16 / 0` |
| **`65536 / 4096`** | `1` | **`13.71`** | **`233.39`** | `15.25` | `30,229.4` | `30,229.4` | `30,229.4` | `65.56` | `65.56` | `65.56` | `298.70` | `1 / 0` |
| **`65536 / 4096`** | `2` | **`26.59`** | **`452.56`** | `14.89` | `31,253.6` | `34,549.6` | `34,549.6` | `66.79` | `67.55` | `66.80` | `307.87` | `2 / 0` |
| **`65536 / 4096`** | `4` | **`51.92`** | **`883.74`** | `14.57` | `35,293.2` | `35,294.7` | `35,294.7` | `68.43` | `69.20` | `68.44` | `315.53` | `4 / 0` |
| **`130048 / 1024`** | `1` | **`7.73`** | **`974.66`** | `15.12` | `64,795.2` | `64,795.2` | `64,795.2` | `66.14` | `66.14` | `66.15` | `132.46` | `1 / 0` |
| **`130048 / 1024`** | `2` | **`14.75`** | **`1,859.95`** | `14.49` | `66,166.8` | `69,990.0` | `69,990.0` | `67.29` | `70.82` | `67.28` | `138.62` | `2 / 0` |
| **`130048 / 1024`** | `4` | **`28.87`** | **`3,640.33`** | `14.34` | `71,412.7` | `71,414.0` | `71,414.0` | `68.86` | `72.44` | `68.88` | `141.86` | `4 / 0` |

### 5.3 Wave 3: `64k` Prefix Caching (`0%` Miss vs. `80%` Hit vs. `95%` Hit) — `vllm-torchtpu` (`TP=8, DP=4, EP=32`)

> **Key Prefix Caching Observation**: With `--enable-prefix-caching --mamba-cache-mode align` (`block_size=1360` unified MLA+KDA state checkpoints), a `95%` shared `64k` prefix (`93.2%` block-aligned cache hit) reduces single-stream TTFT P50 by **`11.1×`** (`30,295.5 ms -> 2,733.4 ms`) and boosts `c=8` total throughput by **`1.94×`** (`3,214.15 tok/s -> 6,235.01 tok/s`, **`95.78` output tok/s**).

| Workload (`65536 / 1024`) | Concurrency | Target Cache % | Observed Cache % | Output Throughput (`tok/s`) | Total Throughput (`tok/s`) | Stream `tok/s` (Mean) | TTFT P50 (`ms`) | TTFT P95 (`ms`) | TPOT P50 (`ms`) | ITL P50 (`ms`) | E2EL P50 (`s`) | OK / Fail |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`64k_1k_0pct_miss`** | `1` | `0.0%` | `0.0%` | **`10.51`** | **`684.14`** | `15.24` | `30,295.5` | `30,295.5` | `65.60` | `65.60` | `97.41` | `1 / 0` |
| **`64k_1k_0pct_miss`** | `4` | `0.0%` | `0.0%` | **`38.95`** | **`2,534.81`** | `14.49` | `35,292.7` | `35,292.7` | `68.29` | `68.33` | `105.15` | `4 / 0` |
| **`64k_1k_0pct_miss`** | `8` | `0.0%` | `0.0%` | **`49.39`** | **`3,214.15`** | `10.67` | `86,849.2` | `89,813.8` | `123.90` | `74.65` | `165.64` | `8 / 0` |
| **`64k_1k_80pct_hit`** | `1` | `80.0%` | `78.7%` | **`13.75`** | **`895.26`** | `15.25` | `7,369.5` | `7,369.5` | `65.58` | `65.59` | `74.46` | `1 / 0` |
| **`64k_1k_80pct_hit`** | `4` | `80.0%` | `78.7%` | **`50.26`** | **`3,272.09`** | `14.48` | `11,573.8` | `11,573.9` | `68.34` | `68.35` | `81.48` | `4 / 0` |
| **`64k_1k_80pct_hit`** | `8` | `80.0%` | `78.7%` | **`82.20`** | **`5,350.97`** | `12.49` | `20,524.7` | `23,491.0` | `84.78` | `74.71` | `99.44` | `8 / 0` |
| **`64k_1k_95pct_hit`** | `1` | `95.0%` | `93.2%` | **`14.67`** | **`955.05`** | `15.25` | `2,733.4` | `2,733.4` | `65.55` | `65.56` | `69.80` | `1 / 0` |
| **`64k_1k_95pct_hit`** | `4` | `95.0%` | `93.2%` | **`54.37`** | **`3,539.62`** | `14.51` | `5,446.4` | `5,446.7` | `68.31` | `68.33` | `75.33` | `4 / 0` |
| **`64k_1k_95pct_hit`** | `8` | `95.0%` | `93.2%` | **`95.78`** | **`6,235.01`** | `13.14` | `7,369.1` | `9,153.8` | `76.26` | `74.72` | `85.38` | `8 / 0` |

---

## 6. Turnkey Relaunch Scripts for Pass 2 & Pass 3

All scripts and patches for **Pass 2 (`vllm-torchtpu` + SGLang-inspired hardware/kernel optimizations)** and **Pass 3 (`sglang-torchtpu` Day-0 `origin/kimi-k31-pro-tpu` + MegaKernel + DFlash2)** are saved in [`scripts/`](scripts/) and [`configs/`](configs/) for one-command relaunch whenever the `2x2x4` TPU v7x slice is available:

### 6.1 Relaunch Pass 2 (`vllm-torchtpu` with SGLang-Inspired Hardware & Kernel Optimizations)
```bash
# Run TP=32, EP=32 with TPU_K3_SP_PREFILL=1, tuned v2 MLA Pallas table (block_size=256),
# Ironwood SparseCore gather offload, Fused MoE Decode kernel, and MegaKDA DMA overlap:
TOPOLOGY_MODE=tp32 bash scripts/relaunch_pass2_vllm_sglang_opt.sh

# Or run TP=8, DP=4, EP=32 with the same SGLang-inspired hardware/kernel optimizations:
TOPOLOGY_MODE=tp8-dp4 bash scripts/relaunch_pass2_vllm_sglang_opt.sh
```

### 6.2 Relaunch Pass 3 (`sglang-torchtpu` Day-0 `origin/kimi-k31-pro-tpu` + MegaKernel / DFlash2)
```bash
# Run TP=32, EP=8 with SparseCore DSA (pallas_tpu_mla_sparse, index_topk=2048) & Per-Op Pallas Decode Kernels:
EXEC_MODE=server bash scripts/relaunch_pass3_sglang_day0.sh

# Or run the single-stream Decode MegaKernel + DFlash2 / DSpark speculative decoding benchmark:
EXEC_MODE=megakernel bash scripts/relaunch_pass3_sglang_day0.sh
```

