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

---

## 3. Repository Structure

```text
kimi-k3.1-pro-tpu-v7x-inference/
├── README.md                                         # Architecture, memory math, & benchmark results
├── INFRA_SETUP.md                                    # Step-by-step GKE TPU v7x 2x2x4 & GCS FUSE setup guide
├── configs/
│   ├── kimi-k3.1-pro-tp8-dp4-ep32.sh                 # Config A: TP=8, DP=4, EP=32 (High-throughput mode)
│   └── kimi-k3.1-pro-tp32-ep32.sh                    # Config B: TP=32, DP=1, EP=32 (Single-pool + SP prefill)
├── scripts/
│   ├── stage_hf_to_gcs.py                            # Parallel multi-pod HF -> GCS staging & byte verifier
│   └── patches/
│       └── patch_vllm_torchtpu_kimi_k31.py           # Idempotent vllm + vllm-torchtpu runtime patch
├── k8s/
│   ├── provision_gke_dws_calendar.sh                 # GKE + TPU v7x 2x2x4 node pool provisioning script
│   └── kimi-k3.1-pro-lws-v7x-16.yaml                 # 4-Pod LeaderWorkerSet manifest for 2x2x4 TPU v7x
├── benchmarks/
│   └── run_kimi_k31_matrix.py                        # Async OpenAI streaming benchmark harness (Waves 1-3)
└── results/                                          # Raw JSON benchmark outputs per wave and topology
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
  MODEL_DIR=/models/models/moonshotai/Kimi-K3.1-Pro \
  RAY_HEAD_IP=$(hostname -i) \
  bash configs/kimi-k3.1-pro-tp8-dp4-ep32.sh
  ```
- **Single-Pool / Sequence-Parallel Prefill Mode (`TP=32, DP=1, EP=32`)**:
  ```bash
  MODEL_DIR=/models/models/moonshotai/Kimi-K3.1-Pro \
  RAY_HEAD_IP=$(hostname -i) \
  bash configs/kimi-k3.1-pro-tp32-ep32.sh
  ```

### 3. Run the Benchmark Matrix
```bash
# Wave 1: Regular Context (1k/1k, 8k/1k across concurrencies 1, 4, 8, 16, 32, 64)
python3 benchmarks/run_kimi_k31_matrix.py \
  --base-url http://127.0.0.1:8000 \
  --model /models/models/moonshotai/Kimi-K3.1-Pro \
  --config-name tp8_dp4_ep32 \
  --wave wave1 \
  --output results/wave1_tp8_dp4_ep32.json

# Wave 2: Long Context (64k/1k, 64k/4k, 128k/1k across concurrencies 1, 2, 4, 8, 16)
python3 benchmarks/run_kimi_k31_matrix.py \
  --base-url http://127.0.0.1:8000 \
  --model /models/models/moonshotai/Kimi-K3.1-Pro \
  --config-name tp8_dp4_ep32 \
  --wave wave2 \
  --output results/wave2_tp8_dp4_ep32.json

# Wave 3: 64k Shared-Prefix Caching (--enable-prefix-caching --mamba-cache-mode align)
python3 benchmarks/run_kimi_k31_matrix.py \
  --base-url http://127.0.0.1:8000 \
  --model /models/models/moonshotai/Kimi-K3.1-Pro \
  --config-name tp8_dp4_ep32_prefix_cache \
  --wave wave3 \
  --output results/wave3_tp8_dp4_ep32_prefix_cache.json
```

---

## 5. Benchmark Matrix & Live TPU v7x (`2x2x4`) Results

> **Hardware**: `1×` TPU v7x `2x2x4` slice (`4` hosts × `4` chips = `16` physical chips = `32` TensorCores, `3,072 GB` HBM3e)  
> **Runtime**: `vllm-torchtpu` (`TPU_UNIFIED_KV_CACHE_LAYOUT=1`, `TPU_MULTIHOST_BACKEND=ray`, `kv_cache_dtype=fp8`, native `compressed-tensors` `mxfp4-pack-quantized` routed experts + `BF16` shared experts/attention)

### 5.1 Baseline Reference (`TP=8, DP=4, EP=32`, Initial Validation Sweep)

Prior to the full Wave 1–3 matrix sweep, initial hardware validation on a `2x2x4` TPU v7x slice (`TP=8, DP=4, EP=32`, `--gpu-memory-utilization=0.82`, `484` unified blocks/replica = `1,669,440` tokens total) established the following baseline:

| Workload (`ISL / OSL`) | Concurrency | Median TTFT (`ms`) | P99 TTFT (`ms`) | Median TPOT (`ms`) | P99 TPOT (`ms`) | Output Throughput (`tok/s`) | Total Throughput (`tok/s`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`1024 / 1024`** | `1` | `1,303.4` | `1,303.4` | `66.16` | `66.16` | `14.8` | `29.6` |
| **`1024 / 1024`** | `8` | `977.6` | `989.1` | `86.25` | `86.41` | `87.3` | `174.5` |
| **`1024 / 1024`** | `32` | `1,896.8` | `4,225.7` | `124.21` | `127.08` | `238.1` | `476.3` |
| **`4096 / 1024`** | `8` | `1,910.4` | `3,389.4` | `84.88` | `86.10` | `88.7` | `443.4` |
| **`8192 / 1024`** | `4` | `2,944.0` | `2,953.5` | `84.83` | `84.89` | `44.9` | `404.0` |

### 5.2 Wave 1: Regular Context (`1k / 1k`, `8k / 1k`) — `TP=8, DP=4, EP=32` vs `TP=32, EP=32`

*(Live Wave 1 benchmark tables across concurrencies `1, 4, 8, 16, 32, 64` are populated in `results/` as each wave completes.)*

### 5.3 Wave 2: Long Context (`64k / 1k`, `64k / 4k`, `128k / 1k`)

*(Live Wave 2 benchmark tables across concurrencies `1, 2, 4, 8, 16` are populated in `results/` as each wave completes.)*

### 5.4 Wave 3: Prefix Caching (`--enable-prefix-caching --mamba-cache-mode align` vs `--no-enable-prefix-caching`) on `64k` Context

*(Live Wave 3 prefix-caching ablation tables (`0%`, `80%`, `95%` shared prefix across multi-turn `64k` workloads) are populated in `results/` as each wave completes.)*
