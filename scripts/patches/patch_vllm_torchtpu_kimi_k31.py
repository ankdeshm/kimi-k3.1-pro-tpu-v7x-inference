#!/usr/bin/env python3
# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0
"""Runtime & source patcher for moonshotai/Kimi-K3.1-Pro on vllm-torchtpu (Ironwood TPU v7x).

What this script does:
1. Registers `KimiK31ForConditionalGeneration` (aliasing/extending `KimiK3ForConditionalGeneration`)
   and `model_type = "kimi_k31"` -> `KimiK3Config` across `vllm_torchtpu` and `vllm`:
   - `vllm_torchtpu/models/vllm/kimi_k3/model.py` & `__init__.py`
   - `vllm_torchtpu/models/vllm/__init__.py` (`ModelRegistry` & `_TPUKimiK3Config` in `MODELS_CONFIG_MAP`
     so routed experts stay on `compressed-tensors` `VllmCompressedTensorsW4ANMxfp4MoEMethod`)
   - `vllm_torchtpu/gdn_pool_layout.py` (`POOLED_GDN_ARCHITECTURES` for unified KV + KDA Mamba state pool)
   - `vllm/transformers_utils/config.py` (`kimi_k31="KimiK3Config"`)
   - `vllm/config/model.py`, `vllm/config/vllm.py`, `vllm/renderers/`, `vllm/tokenizers/registry.py`
2. Handles the 120 new DeepSeek Sparse Attention (DSA) indexer tensors
   (`self_attn.indexer.{k_norm.bias, k_norm.weight, weights_proj.weight, wk.weight, wq_b.weight}`)
   present in the 24 Gated MLA layers of `moonshotai/Kimi-K3.1-Pro` by mapping
   `".self_attn.indexer.": None` in `KimiLinearForCausalLM.hf_to_vllm_mapper` and filtering
   them in `_ordinary_weights()` (since TPU `KimiMLAAttention` executes exact dense Gated MLA).
3. Guards `TPU_K3_SP_PREFILL=1` in `KimiModel.__init__` (before constructing decoder layers)
   so that if `TPU_K3_SP_PREFILL=1` is set in the environment under `TP=8, DP=4, EP=32`,
   it automatically sets `os.environ["TPU_K3_SP_PREFILL"] = "0"` and `self.sp_prefill = False`
   instead of raising `ValueError("K3 SP prefill requires TP32/EP32 and PP1/DP1")`.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def _patch_file(path: Path, replacements: list[tuple[str, str, str]]) -> int:
    if not path.exists():
        print(f"[kimi-k31-patch] SKIP (not found): {path}")
        return 0
    text = path.read_text(encoding="utf-8")
    orig = text
    applied = 0
    for desc, old, new in replacements:
        if new in text:
            print(f"[kimi-k31-patch] OK (already applied): {desc} in {path.name}")
            continue
        if old not in text:
            print(f"[kimi-k31-patch] WARN (target snippet not found): {desc} in {path}")
            continue
        text = text.replace(old, new, 1)
        applied += 1
        print(f"[kimi-k31-patch] APPLIED: {desc} in {path}")
    if text != orig:
        path.write_text(text, encoding="utf-8")
    return applied


def patch_vllm_torchtpu(torchtpu_root: Path) -> int:
    total = 0

    # 1. vllm_torchtpu/models/vllm/kimi_k3/model.py
    model_py = torchtpu_root / "src/vllm_torchtpu/models/vllm/kimi_k3/model.py"
    total += _patch_file(
        model_py,
        [
            (
                "Map .self_attn.indexer. -> None in KimiLinearForCausalLM.hf_to_vllm_mapper",
                '            ".output_attn_res_proj.": ".output_attn_res.proj.",\n        },',
                '            ".output_attn_res_proj.": ".output_attn_res.proj.",\n'
                '            ".self_attn.indexer.": None,\n        },',
            ),
            (
                "Skip .self_attn.indexer. tensors in _ordinary_weights()",
                '                if name.startswith(("vision_tower.", "mm_projector.")):\n                    continue',
                '                if name.startswith(("vision_tower.", "mm_projector.")) or ".self_attn.indexer." in name:\n                    continue',
            ),
            (
                "Auto-guard TPU_K3_SP_PREFILL when TP!=32 or DP!=1 in KimiModel.__init__",
                '                raise ValueError("K3 SP prefill requires TP32/EP32 and PP1/DP1")\n            self.sp_group = TokenShardCollectives(get_tp_group())',
                '                import os as _os\n'
                '                logger.warning("Disabling TPU_K3_SP_PREFILL because TP=%s DP=%s (requires TP=32, DP=1)", parallel.tensor_parallel_size, parallel.data_parallel_size)\n'
                '                _os.environ["TPU_K3_SP_PREFILL"] = "0"\n'
                '                self.sp_prefill = False\n'
                '            else:\n'
                '                self.sp_group = TokenShardCollectives(get_tp_group())',
            ),
            (
                "Alias KimiK31ForConditionalGeneration = KimiK3ForConditionalGeneration",
                "        loader = AutoWeightsLoader(self)\n        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)\n",
                "        loader = AutoWeightsLoader(self)\n        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)\n\n\nKimiK31ForConditionalGeneration = KimiK3ForConditionalGeneration\n",
            ),
        ],
    )

    # 2. vllm_torchtpu/models/vllm/kimi_k3/__init__.py
    init_py = torchtpu_root / "src/vllm_torchtpu/models/vllm/kimi_k3/__init__.py"
    total += _patch_file(
        init_py,
        [
            (
                "Export KimiK31ForConditionalGeneration in kimi_k3/__init__.py",
                "from .model import KimiK3ForConditionalGeneration, KimiLinearForCausalLM, KimiModel",
                "from .model import KimiK31ForConditionalGeneration, KimiK3ForConditionalGeneration, KimiLinearForCausalLM, KimiModel",
            ),
            (
                "Add KimiK31ForConditionalGeneration to __all__ in kimi_k3/__init__.py",
                '    "KimiK3ForConditionalGeneration",',
                '    "KimiK3ForConditionalGeneration",\n    "KimiK31ForConditionalGeneration",',
            ),
        ],
    )

    # 3. vllm_torchtpu/models/vllm/__init__.py
    models_init_py = torchtpu_root / "src/vllm_torchtpu/models/vllm/__init__.py"
    total += _patch_file(
        models_init_py,
        [
            (
                "Register KimiK31ForConditionalGeneration in vllm_torchtpu ModelRegistry",
                '    ModelRegistry.register_model("KimiK3ForConditionalGeneration", _KIMI_K3_MODEL)',
                '    ModelRegistry.register_model("KimiK3ForConditionalGeneration", _KIMI_K3_MODEL)\n'
                '    ModelRegistry.register_model("KimiK31ForConditionalGeneration", _KIMI_K3_MODEL)',
            ),
            (
                "Register KimiK31ForConditionalGeneration in vllm_torchtpu MODELS_CONFIG_MAP",
                '    MODELS_CONFIG_MAP["KimiK3ForConditionalGeneration"] = _TPUKimiK3Config',
                '    MODELS_CONFIG_MAP["KimiK3ForConditionalGeneration"] = _TPUKimiK3Config\n'
                '    MODELS_CONFIG_MAP["KimiK31ForConditionalGeneration"] = _TPUKimiK3Config',
            ),
        ],
    )

    # 4. vllm_torchtpu/gdn_pool_layout.py
    gdn_py = torchtpu_root / "src/vllm_torchtpu/gdn_pool_layout.py"
    total += _patch_file(
        gdn_py,
        [
            (
                "Add KimiK31ForConditionalGeneration to POOLED_GDN_ARCHITECTURES",
                '        "KimiLinearForCausalLM",\n        "KimiK3ForConditionalGeneration",',
                '        "KimiLinearForCausalLM",\n        "KimiK3ForConditionalGeneration",\n        "KimiK31ForConditionalGeneration",',
            ),
        ],
    )

    # 5. vllm_torchtpu/layers/adapter/fused_moe_decode.py
    fused_moe_py = torchtpu_root / "src/vllm_torchtpu/layers/adapter/fused_moe_decode.py"
    if fused_moe_py.exists():
        total += _patch_file(
            fused_moe_py,
            [
                (
                    "Guard _async_drafter property in arm_runner_models when non-speculative (drafter is None)",
                    '    for name in ("drafter", "_async_drafter"):\n        drafter = getattr(model_runner, name, None)',
                    '    for name in ("drafter", "_async_drafter"):\n        if name == "_async_drafter" and not getattr(model_runner, "_is_async_drafter", False):\n            continue\n        drafter = getattr(model_runner, name, None)',
                ),
            ],
        )

    return total


def patch_vllm_site_packages(vllm_root: Path) -> int:
    total = 0

    # 1. vllm/transformers_utils/config.py
    tf_cfg_py = vllm_root / "transformers_utils/config.py"
    total += _patch_file(
        tf_cfg_py,
        [
            (
                "Register kimi_k31='KimiK3Config' in vllm _CONFIG_REGISTRY",
                '    kimi_k3="KimiK3Config",',
                '    kimi_k3="KimiK3Config",\n    kimi_k31="KimiK3Config",',
            ),
        ],
    )

    # 2. vllm/config/model.py
    cfg_model_py = vllm_root / "config/model.py"
    total += _patch_file(
        cfg_model_py,
        [
            (
                "Map KimiK31ForConditionalGeneration to tokenizer_mode='kimi_k3'",
                '            elif arch == "KimiK3ForConditionalGeneration":\n                self.tokenizer_mode = "kimi_k3"',
                '            elif arch in ("KimiK3ForConditionalGeneration", "KimiK31ForConditionalGeneration"):\n                self.tokenizer_mode = "kimi_k3"',
            ),
        ],
    )

    # 3. vllm/config/vllm.py
    cfg_vllm_py = vllm_root / "config/vllm.py"
    total += _patch_file(
        cfg_vllm_py,
        [
            (
                "Add KimiK31ForConditionalGeneration to cudagraph architectures",
                '        "KimiK3ForConditionalGeneration",\n        "KimiK3MTPModel",',
                '        "KimiK3ForConditionalGeneration",\n        "KimiK31ForConditionalGeneration",\n        "KimiK3MTPModel",',
            ),
            (
                "Add KimiK31ForConditionalGeneration to kda_architectures",
                '        kda_architectures = (\n            "KimiLinearForCausalLM",\n            "KimiK3ForConditionalGeneration",\n        )',
                '        kda_architectures = (\n            "KimiLinearForCausalLM",\n            "KimiK3ForConditionalGeneration",\n            "KimiK31ForConditionalGeneration",\n        )',
            ),
        ],
    )

    # 4. vllm/renderers/chat_utils.py
    chat_utils_py = vllm_root / "renderers/chat_utils.py"
    total += _patch_file(
        chat_utils_py,
        [
            (
                "Add kimi_k31 to _KIMI_MODEL_TYPES",
                '_KIMI_MODEL_TYPES = ("kimi_k2", "kimi_k25", "kimi_k3")',
                '_KIMI_MODEL_TYPES = ("kimi_k2", "kimi_k25", "kimi_k3", "kimi_k31")',
            ),
        ],
    )

    # 5. vllm/renderers/registry.py
    ren_reg_py = vllm_root / "renderers/registry.py"
    total += _patch_file(
        ren_reg_py,
        [
            (
                "Register kimi_k31 in _VLLM_RENDERERS",
                '    "kimi_k3": ("kimi_k3", "KimiK3Renderer"),',
                '    "kimi_k3": ("kimi_k3", "KimiK3Renderer"),\n    "kimi_k31": ("kimi_k3", "KimiK3Renderer"),',
            ),
        ],
    )

    # 6. vllm/tokenizers/registry.py
    tok_reg_py = vllm_root / "tokenizers/registry.py"
    total += _patch_file(
        tok_reg_py,
        [
            (
                "Register kimi_k31 in tokenizer registry",
                '    "kimi_k3": ("hf", "CachedHfTokenizer"),',
                '    "kimi_k3": ("hf", "CachedHfTokenizer"),\n    "kimi_k31": ("hf", "CachedHfTokenizer"),',
            ),
        ],
    )

    return total


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply Kimi-K3.1-Pro architecture & DSA indexer weight-loader patches to vllm-torchtpu."
    )
    parser.add_argument(
        "--torchtpu-root",
        default="/workspace/vllm-torchtpu",
        help="Path to vllm-torchtpu repository root",
    )
    parser.add_argument(
        "--vllm-root",
        default="/usr/local/lib/python3.12/site-packages/vllm",
        help="Path to installed vllm package directory",
    )
    args = parser.parse_args()

    n_tpu = patch_vllm_torchtpu(Path(args.torchtpu_root))
    n_vllm = patch_vllm_site_packages(Path(args.vllm_root))
    print(
        f"[kimi-k31-patch] Completed successfully (applied {n_tpu} vllm_torchtpu edits, {n_vllm} vllm edits).",
        flush=True,
    )


if __name__ == "__main__":
    main()
