# Infrastructure & Production Deployment Guide: `moonshotai/Kimi-K3.1-Pro` on Google Cloud TPU v7x (Ironwood)

This guide covers end-to-end infrastructure provisioning, high-speed model staging (`1.56 TB`), multi-host LeaderWorkerSet (`LWS`) deployment, and runtime patching for serving **`moonshotai/Kimi-K3.1-Pro`** (`2.8T` total parameters / `104B` active parameters per token) on a **Google Cloud TPU v7x (Ironwood)** `2x2x4` slice (`4` nodes × `4` chips/node = `16` physical chips = `32` TensorCores = `3,072 GB` HBM3e).

---

## 1. Hardware Topology & HBM3e Memory Architecture

### 1.1 TPU v7x (`tpu7x-standard-4t`) `2x2x4` Multi-Host Slice

| Hardware Dimension | Per Physical Chip | Per Node (`tpu7x-standard-4t`) | Full `2x2x4` Slice (`4` Nodes) |
| :--- | :--- | :--- | :--- |
| **Physical TPU v7x Chips** | `1` | `4` | **`16` Chips** |
| **Logical XLA TensorCores (`devices`)** | `2` | `8` | **`32` TensorCores** |
| **HBM3e Capacity** | `192 GB` | `768 GB` | **`3,072 GB` (`3.07 TB`)** |
| **Usable HBM3e per TensorCore** | `94.74 GiB` | `757.92 GiB` | **`3,031.68 GiB`** |
| **HBM3e Bandwidth** | `7.37 TB/s` | `29.48 TB/s` | **`117.96 TB/s`** |
| **BF16 / FP8 Peak Compute** | `4.61 PFLOPS` | `18.43 PFLOPS` | **`73.73 PFLOPS`** |
| **Inter-Chip Interconnect (ICI)** | 3D Torus (`2x2x4`) | Intra-host ICI | **Full 16-Chip 3D Torus** |

### 1.2 Why `Kimi-K3.1-Pro` Requires a `2x2x4` (`16`-Chip / `32`-TensorCore) Slice

`moonshotai/Kimi-K3.1-Pro` has **`96` safetensors shards totaling `1,561.36 GB` (`1.56 TB`)**:
- **`93` Hybrid Decoder Layers**:
  - **`69` KDA (Kimi Delta Attention) Linear Attention Layers** (`3:1` ratio) with `num_heads=64`, `head_dim=128`, `conv_kernel_size=4`, and recurrent state `[64, 128, 128]` per sequence.
  - **`24` Gated MLA (Multi-head Latent Attention) / DSA Layers** (every 4th layer: `4, 8, 12, ..., 92` plus layer `93`) with `kv_lora_rank=512`, `q_lora_rank=1536`, `qk_rope_head_dim=64`, `qk_nope_head_dim=128`, `v_head_dim=128`, `num_heads=64`, and `gated_attention=True`.
- **Ultra-Sparse MoE (`896` Routed Experts + `2` Shared Experts)**:
  - `n_routed_experts = 896`, `num_experts_per_tok = 8`, `n_group = 8`, `topk_group = 4`, `routed_scaling_factor = 3.0`, `moe_intermediate_size = 2048`.
  - All `896` routed experts are natively quantized in **`compressed-tensors` `mxfp4-pack-quantized`** (`4-bit` group-32 E8M0 block scales packed in `uint8`), while the `2` shared experts (`n_shared_experts=2`), embeddings, norms, KDA projections, and Gated MLA projections remain in **`BF16`**.
  - On `32` TensorCores (`EP=32`), each TensorCore holds **`896 / 32 = 28` routed experts** per layer (`92` MoE layers, `first_k_dense_replace=1`).

---

## 2. Step-by-Step Infrastructure Setup

### Step 1: Provision GKE Cluster & `2x2x4` TPU v7x Node Pool

Use [`k8s/provision_gke_dws_calendar.sh`](k8s/provision_gke_dws_calendar.sh) to provision the GKE cluster with the **GCS FUSE CSI Driver**, **Workload Identity**, **gVNIC**, and the **LeaderWorkerSet (LWS)** controller:

```bash
export PROJECT_ID="<YOUR_GCP_PROJECT_ID>"
export REGION="us-central1"
export ZONE="us-central1-c"
export CLUSTER_NAME="tpu-v7x-inference-cluster"
export NODE_POOL_NAME="tpu-v7x-16-pool"
export GCS_BUCKET="<YOUR_GCS_MODEL_BUCKET>"
export RESERVATION_NAME="<YOUR_TPU_V7X_RESERVATION_NAME>"

bash k8s/provision_gke_dws_calendar.sh
```

### Step 2: Configure GCS FUSE CSI Driver & IAM Permissions

Grant the Kubernetes ServiceAccount (`default` or a dedicated SA) read access to your GCS model bucket so all 4 TPU nodes can stream the `1.56 TB` checkpoint at `>25 GB/s` aggregate throughput:

```bash
gcloud storage buckets add-iam-policy-binding "gs://${GCS_BUCKET}" \
  --member="principal://iam.googleapis.com/projects/$(gcloud projects describe ${PROJECT_ID} --format='value(projectNumber)')/locations/global/workloadIdentityPools/${PROJECT_ID}.svc.id.goog/subject/ns/default/sa/default" \
  --role="roles/storage.objectViewer"
```

Recommended GCS FUSE mount options in your Pod/LWS spec (included in [`k8s/kimi-k3.1-pro-lws-v7x-16.yaml`](k8s/kimi-k3.1-pro-lws-v7x-16.yaml)):
```yaml
mountOptions: "implicit-dirs,metadata-cache:ttl-secs:-1,metadata-cache:stat-cache-max-size-mb:-1,metadata-cache:type-cache-max-size-mb:-1,file-system:kernel-list-cache-ttl-secs:-1"
```

### Step 3: Parallel Multi-Pod Staging from Hugging Face to GCS (`1.56 TB`)

Because `moonshotai/Kimi-K3.1-Pro` contains `96` safetensors shards (`~16.5 GB` each, `1,561.36 GB` total), downloading sequentially on a single host is slow. Use [`scripts/stage_hf_to_gcs.py`](scripts/stage_hf_to_gcs.py) across `2` or `4` pods in parallel using `--shard-start` and `--shard-end`:

```bash
# Host 0: Metadata + Shards 0..48 (files 1..48 of 96)
HF_TOKEN="<YOUR_HF_TOKEN>" python3 scripts/stage_hf_to_gcs.py \
  --repo-id moonshotai/Kimi-K3.1-Pro \
  --dest-dir /models/models/moonshotai/Kimi-K3.1-Pro \
  --shard-start 0 --shard-end 48 --workers 6

# Host 1: Shards 48..96 (files 49..96 of 96)
HF_TOKEN="<YOUR_HF_TOKEN>" python3 scripts/stage_hf_to_gcs.py \
  --repo-id moonshotai/Kimi-K3.1-Pro \
  --dest-dir /models/models/moonshotai/Kimi-K3.1-Pro \
  --shard-start 48 --shard-end 96 --skip-metadata --workers 6

# Final Verification (writes _STAGE_COMPLETE.txt after verifying all 111 files)
HF_TOKEN="<YOUR_HF_TOKEN>" python3 scripts/stage_hf_to_gcs.py \
  --repo-id moonshotai/Kimi-K3.1-Pro \
  --dest-dir /models/models/moonshotai/Kimi-K3.1-Pro \
  --verify-only
```

---

## 3. Runtime Patching (`patch_vllm_torchtpu_kimi_k31.py`)

### Why the Patch Is Needed on `vllm-torchtpu`

`vllm-torchtpu` includes native support for `KimiK3ForCausalLM` (`model_type: "kimi_k3"`), including:
1. Custom Pallas/XLA kernels for **KDA (Kimi Delta Attention)** recurrent state updates and **Gated MLA** (`gated_attention=True`).
2. Unified KV + Mamba state pool (`KimiUnifiedAttention` + `TPU_UNIFIED_KV_CACHE_LAYOUT=1`), packing `2` MLA KV tokens + `1` KDA conv/recurrent state into a single `1360`-token unified block.
3. Native TPU MoE execution for `compressed-tensors` `mxfp4-pack-quantized` (`VllmCompressedTensorsW4ANMxfp4MoEMethod`).

When loading **`moonshotai/Kimi-K3.1-Pro`**, three differences in its checkpoint and configuration must be bridged:
1. **Architecture & Config Registration**: `Kimi-K3.1-Pro`'s `config.json` declares `"architectures": ["KimiK31ForConditionalGeneration"]` and `"model_type": "kimi_k31"`.
2. **DSA Indexer Tensor Filtering (`120` Tensors)**: `Kimi-K3.1-Pro` includes `5` DeepSeek Sparse Attention (DSA) indexer tensors (`self_attn.indexer.{k_norm.bias, k_norm.weight, weights_proj.weight, wk.weight, wq_b.weight}`) in each of its `24` Gated MLA layers (`120` tensors total). Because `vllm-torchtpu`'s `KimiMLAAttention` executes exact dense Gated MLA (`indexer=None, is_sparse=False`), `KimiLinearForCausalLM.hf_to_vllm_mapper` and `_ordinary_weights()` must ignore `.self_attn.indexer.` tensors (matching the pattern used in `vllm_torchtpu/models/vllm/qwen4_exp/model.py`).
3. **`TPU_K3_SP_PREFILL` Topology Guard**: `KimiModel.__init__` enables `TPU_K3_SP_PREFILL=1` by default, which requires `TP=32, EP=32, DP=1` and raises `ValueError("K3 SP prefill requires TP32/EP32 and PP1/DP1")` under `TP=8, DP=4, EP=32`. The patch automatically disables `TPU_K3_SP_PREFILL` when `TP != 32` or `DP != 1`.

### Multi-Host Ray Requirement

> [!IMPORTANT]
> Because `vllm serve` on a `2x2x4` slice uses `RayDistributedExecutorV2` across all 4 pods (1 leader + 3 workers), [`scripts/patches/patch_vllm_torchtpu_kimi_k31.py`](scripts/patches/patch_vllm_torchtpu_kimi_k31.py) **must be executed on all 4 pods before `ray start` and `vllm serve`**, so that every Ray worker process imports the patched `vllm` and `vllm_torchtpu` modules.

---

## 4. Launching `Kimi-K3.1-Pro` on GKE LeaderWorkerSet

Deploy the 4-pod `2x2x4` LeaderWorkerSet manifest:

```bash
kubectl apply -f k8s/kimi-k3.1-pro-lws-v7x-16.yaml
```

To switch between the two parallelism modes:
- **Throughput Mode (`TP=8, DP=4, EP=32`)**: [`configs/kimi-k3.1-pro-tp8-dp4-ep32.sh`](configs/kimi-k3.1-pro-tp8-dp4-ep32.sh)
- **Long-Context / Single-Pool Mode (`TP=32, DP=1, EP=32`)**: [`configs/kimi-k3.1-pro-tp32-ep32.sh`](configs/kimi-k3.1-pro-tp32-ep32.sh)
