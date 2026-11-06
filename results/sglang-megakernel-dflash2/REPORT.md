# `moonshotai/Kimi-K3.1-Pro` Benchmark Report — `sglang-megakernel-dflash2`

- **Timestamp (UTC)**: `20261106T064201Z`
- **Model**: `/models/models/moonshotai/Kimi-K3.1-Pro` (`2.8T` total / `104B` active MoE, `69` KDA + `24` Gated MLA/DSA layers)
- **Hardware Topology**: `tpu-v7x-16-2x2x4-32cores` (`16` physical TPU v7x chips = `32` TensorCores, `3,072 GB` HBM3e)
- **Configuration Tag**: `sglang-megakernel-dflash2`

| Suite | Workload | Concurrency | Target Cache % | Observed Cache % | Output tok/s | Total tok/s | Stream tok/s (Mean) | TTFT P50 (ms) | TTFT P95 (ms) | TTFT P99 (ms) | TPOT P50 (ms) | TPOT P95 (ms) | TPOT P99 (ms) | ITL P50 (ms) | E2EL P50 (s) | OK/Fail |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `regular_context` | `1k_1k` | 1 | 0.0% | 0.0% | **264.03** | **547.66** | 350.76 | 961.2 | 961.2 | 961.2 | 2.85 | 2.85 | 2.85 | 8.66 | 3.88 | 1/0 |
| `regular_context` | `8k_1k` | 1 | 0.0% | 0.0% | **100.01** | **907.44** | 311.03 | 6949.2 | 6949.2 | 6949.2 | 3.22 | 3.22 | 3.22 | 9.34 | 10.24 | 1/0 |
| `long_context` | `64k_1k` | 1 | 0.0% | 0.0% | **13.62** | **886.28** | 189.68 | 69791.6 | 69791.6 | 69791.6 | 5.27 | 5.27 | 5.27 | 13.79 | 75.19 | 1/0 |
| `long_context` | `64k_4k` | 1 | 0.0% | 0.0% | **45.06** | **766.95** | 194.21 | 69804.6 | 69804.6 | 69804.6 | 5.15 | 5.15 | 5.15 | 13.91 | 90.89 | 1/0 |
