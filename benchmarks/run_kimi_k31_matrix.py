#!/usr/bin/env python3
# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0
"""Automated benchmark matrix runner for moonshotai/Kimi-K3.1-Pro on Ironwood TPU v7x.

Covers:
1. Wave 1 — Regular Context (`1k / 1k`, `8k / 1k`) across concurrency `1, 4, 8, 16, 32, 64` (0% prefix-cache hit).
2. Wave 2 — Long Context (`64k / 1k`, `64k / 4k`, `128k / 1k`) across concurrency `1, 2, 4, 8, 16` (0% prefix-cache hit).
3. Wave 3 — Prefix Caching (`64k / 1k` with 80% & 95% shared multi-turn prefix across concurrency `1, 4, 8, 16`),
   measuring exact `cached_tokens` reported by vLLM (`--enable-prompt-tokens-details` + `--mamba-cache-mode=align`)
   and comparing `--enable-prefix-caching` vs `--no-enable-prefix-caching`.
"""

from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import urllib.request
import uuid

import aiohttp


VOCAB_WORDS = [
    "alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel",
    "india", "juliet", "kilo", "lima", "mike", "november", "oscar", "papa",
    "quebec", "romeo", "sierra", "tango", "uniform", "victor", "whiskey", "xray",
    "yankee", "zulu", "quantum", "tensor", "matrix", "vector", "router", "expert",
    "sparse", "attention", "linear", "kernel", "pallas", "systolic", "bfloat", "microscale",
]


def wait_for_server(base_url: str, timeout_s: int = 7200) -> str:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"{base_url}/v1/models", timeout=10) as resp:
                if resp.status == 200:
                    data = json.loads(resp.read().decode("utf-8"))
                    models = data.get("data", [])
                    if models:
                        model_id = models[0]["id"]
                        print(f"[kimi-k31-bench] Server ready! model_id={model_id}", flush=True)
                        return model_id
        except Exception:
            pass
        time.sleep(5)
    raise TimeoutError(f"vLLM server at {base_url} did not become ready within {timeout_s}s")


class PromptBuilder:
    """Builds exact-token-count prompts using the Kimi-K3.1-Pro TikToken vocabulary."""

    def __init__(self, tokenizer_path: str | None = None) -> None:
        self.tokenizer = None
        if tokenizer_path:
            try:
                from transformers import AutoTokenizer
                self.tokenizer = AutoTokenizer.from_pretrained(
                    tokenizer_path, trust_remote_code=True
                )
                print(f"[kimi-k31-bench] Loaded tokenizer from {tokenizer_path}", flush=True)
            except Exception as exc:
                print(f"[kimi-k31-bench] Fallback word calibration ({exc})", flush=True)
        self._shared_block_cache: dict[tuple[int, str], str] = {}
        self._body_cache: dict[int, str] = {}

    def _make_token_calibrated_text(self, target_tokens: int, seed_tag: str) -> str:
        prefix = f"[{seed_tag}] "
        body_tokens = max(8, target_tokens - 8)
        if body_tokens not in self._body_cache:
            n_words = max(16, body_tokens)
            body = " ".join(VOCAB_WORDS[i % len(VOCAB_WORDS)] for i in range(n_words))
            if self.tokenizer is not None:
                ids = self.tokenizer.encode(body, add_special_tokens=False)
                if len(ids) > body_tokens:
                    ids = ids[:body_tokens]
                    body = self.tokenizer.decode(ids)
                elif len(ids) < body_tokens:
                    pad_ids = ids * ((body_tokens - len(ids)) // max(len(ids), 1) + 2)
                    ids = (ids + pad_ids)[:body_tokens]
                    body = self.tokenizer.decode(ids)
            self._body_cache[body_tokens] = body
        return prefix + self._body_cache[body_tokens]

    def get_shared_prefix(self, shared_tokens: int, session_id: str) -> str:
        key = (shared_tokens, session_id)
        if key not in self._shared_block_cache:
            self._shared_block_cache[key] = self._make_token_calibrated_text(
                shared_tokens, f"SHARED-PREFIX-{session_id}"
            )
        return self._shared_block_cache[key]

    def build_prompt(
        self,
        input_tokens: int,
        req_idx: int,
        prefix_hit_ratio: float = 0.0,
        shared_session_id: str = "global",
    ) -> str:
        # Reserve ~24 tokens for chat template framing & instruction tail
        usable_tokens = max(32, input_tokens - 24)
        shared_tokens = int(usable_tokens * prefix_hit_ratio)
        unique_tokens = max(16, usable_tokens - shared_tokens)
        unique_seed = f"REQ-{req_idx}-{uuid.uuid4().hex[:10]}"

        if shared_tokens > 0:
            prefix_str = self.get_shared_prefix(shared_tokens, shared_session_id)
            suffix_str = self._make_token_calibrated_text(unique_tokens, unique_seed)
            return (
                f"{prefix_str}\n\n"
                f"[Turn {req_idx} Context: {suffix_str}]\n"
                "Provide a detailed technical derivation and step-by-step analysis."
            )
        else:
            body = self._make_token_calibrated_text(usable_tokens, unique_seed)
            return (
                f"{body}\n\n"
                "Provide a detailed technical derivation and step-by-step analysis."
            )


def pct(vals: list[float], q: float) -> float:
    if not vals:
        return 0.0
    s = sorted(vals)
    idx = min(len(s) - 1, max(0, int(round((q / 100.0) * (len(s) - 1)))))
    return s[idx]


async def _one_request(
    session: aiohttp.ClientSession,
    base_url: str,
    model_id: str,
    prompt: str,
    output_tokens: int,
    sem: asyncio.Semaphore,
) -> dict:
    payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": output_tokens,
        "temperature": 0.0,
        "ignore_eos": True,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    async with sem:
        t0 = time.perf_counter()
        first_tok_t: float | None = None
        prev_tok_t: float | None = None
        last_tok_t: float | None = None
        itls_ms: list[float] = []
        chunks = 0
        usage_in = 0
        usage_out = 0
        cached_toks = 0
        try:
            timeout = aiohttp.ClientTimeout(total=3600)
            async with session.post(
                f"{base_url}/v1/chat/completions", json=payload, timeout=timeout
            ) as resp:
                if resp.status != 200:
                    err = await resp.text()
                    return {"ok": False, "error": f"HTTP {resp.status}: {err[:300]}"}
                async for raw_line in resp.content:
                    line = raw_line.decode("utf-8", errors="ignore").strip()
                    if not line.startswith("data: "):
                        continue
                    data_str = line[6:]
                    if data_str == "[DONE]":
                        break
                    try:
                        obj = json.loads(data_str)
                    except Exception:
                        continue
                    choices = obj.get("choices") or []
                    if choices:
                        delta = choices[0].get("delta") or {}
                        if delta.get("content") or delta.get("reasoning_content") or delta.get("reasoning"):
                            now = time.perf_counter()
                            if first_tok_t is None:
                                first_tok_t = now
                            elif prev_tok_t is not None:
                                itls_ms.append((now - prev_tok_t) * 1000.0)
                            prev_tok_t = now
                            last_tok_t = now
                            chunks += 1
                    usage = obj.get("usage")
                    if usage:
                        usage_in = usage.get("prompt_tokens", usage_in)
                        usage_out = usage.get("completion_tokens", usage_out)
                        details = usage.get("prompt_tokens_details") or {}
                        cached_toks = details.get("cached_tokens", cached_toks) or 0
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

        t_end = time.perf_counter()
        if first_tok_t is None:
            return {"ok": False, "error": "No token chunks received"}
        out_toks = usage_out if usage_out > 0 else max(1, chunks)
        in_toks = usage_in
        ttft_ms = (first_tok_t - t0) * 1000.0
        decode_s = max((last_tok_t or t_end) - first_tok_t, 1e-6)
        e2e_s = max(t_end - t0, 1e-6)
        tpot_ms = (decode_s * 1000.0) / max(1, out_toks - 1) if out_toks > 1 else 0.0
        itl_mean_ms = sum(itls_ms) / len(itls_ms) if itls_ms else tpot_ms
        stream_tps = (out_toks - 1) / decode_s if out_toks > 1 else out_toks / e2e_s
        return {
            "ok": True,
            "prompt_tokens": in_toks,
            "output_tokens": out_toks,
            "cached_tokens": cached_toks,
            "ttft_ms": ttft_ms,
            "tpot_ms": tpot_ms,
            "itl_mean_ms": itl_mean_ms,
            "itls_ms": itls_ms,
            "e2e_s": e2e_s,
            "stream_tps": stream_tps,
        }


async def run_sweep_point(
    base_url: str,
    model_id: str,
    builder: PromptBuilder,
    suite: str,
    workload_tag: str,
    input_tokens: int,
    output_tokens: int,
    concurrency: int,
    num_prompts: int,
    prefix_hit_ratio: float = 0.0,
) -> dict:
    sem = asyncio.Semaphore(concurrency)
    shared_session = f"sess-{workload_tag}-{uuid.uuid4().hex[:6]}"
    prompts = [
        builder.build_prompt(
            input_tokens=input_tokens,
            req_idx=i,
            prefix_hit_ratio=prefix_hit_ratio,
            shared_session_id=shared_session,
        )
        for i in range(num_prompts)
    ]

    conn = aiohttp.TCPConnector(limit=max(128, concurrency * 2))
    async with aiohttp.ClientSession(connector=conn) as session:
        # If testing prefix cache hit > 0, warm the shared prefix across all DP engines (12 requests).
        # Under --mamba-cache-mode=align, MambaCacheManager retains the KDA state snapshot at the
        # replay boundary (last full 1360-token block) of Turn 1, so Turn 1 must be the shared prefix itself.
        if prefix_hit_ratio > 0.0:
            warm_count = 12
            warm_sem = asyncio.Semaphore(4)
            usable_tokens = max(32, input_tokens - 24)
            shared_tokens = int(usable_tokens * prefix_hit_ratio)
            warm_prompt = builder.get_shared_prefix(shared_tokens, shared_session)
            print(
                f"[kimi-k31-bench] Warming shared prefix ({int(prefix_hit_ratio*100)}% = {shared_tokens} toks) across DP ranks ({warm_count} warmup reqs)...",
                flush=True,
            )
            warm_tasks = [
                _one_request(session, base_url, model_id, warm_prompt, 4, warm_sem)
                for _ in range(warm_count)
            ]
            await asyncio.gather(*warm_tasks)

        t0 = time.perf_counter()
        tasks = [
            _one_request(session, base_url, model_id, p, output_tokens, sem)
            for p in prompts
        ]
        results = await asyncio.gather(*tasks)
        wall_s = max(time.perf_counter() - t0, 1e-6)

    ok_res = [r for r in results if r.get("ok")]
    failed = len(results) - len(ok_res)
    if not ok_res:
        first_err = results[0].get("error") if results else "unknown"
        print(
            f"[kimi-k31-bench] ERROR {suite} {workload_tag} c={concurrency}: 0/{len(results)} succeeded ({first_err})",
            flush=True,
        )
        return {
            "suite": suite,
            "workload": workload_tag,
            "target_input_tokens": input_tokens,
            "target_output_tokens": output_tokens,
            "concurrency": concurrency,
            "num_prompts": num_prompts,
            "prefix_hit_target_pct": round(prefix_hit_ratio * 100.0, 1),
            "successful": 0,
            "failed": failed,
            "error": first_err,
        }

    total_in = sum(r["prompt_tokens"] for r in ok_res)
    total_out = sum(r["output_tokens"] for r in ok_res)
    total_cached = sum(r["cached_tokens"] for r in ok_res)
    ttfts = [r["ttft_ms"] for r in ok_res]
    tpots = [r["tpot_ms"] for r in ok_res]
    all_itls = [x for r in ok_res for x in r.get("itls_ms", [])] or tpots
    e2es = [r["e2e_s"] for r in ok_res]
    stpss = [r["stream_tps"] for r in ok_res]

    row = {
        "suite": suite,
        "workload": workload_tag,
        "target_input_tokens": input_tokens,
        "target_output_tokens": output_tokens,
        "actual_input_tokens_mean": round(total_in / len(ok_res), 1),
        "actual_output_tokens_mean": round(total_out / len(ok_res), 1),
        "cached_tokens_mean": round(total_cached / len(ok_res), 1),
        "observed_prefix_hit_pct": round((total_cached / max(total_in, 1)) * 100.0, 1),
        "prefix_hit_target_pct": round(prefix_hit_ratio * 100.0, 1),
        "concurrency": concurrency,
        "num_prompts": num_prompts,
        "successful": len(ok_res),
        "failed": failed,
        "wall_s": round(wall_s, 2),
        "output_tok_s": round(total_out / wall_s, 2),
        "total_tok_s": round((total_in + total_out) / wall_s, 2),
        "stream_tok_s_mean": round(sum(stpss) / len(stpss), 2),
        "stream_tok_s_p50": round(pct(stpss, 50), 2),
        "ttft_mean_ms": round(sum(ttfts) / len(ttfts), 1),
        "ttft_p50_ms": round(pct(ttfts, 50), 1),
        "ttft_p90_ms": round(pct(ttfts, 90), 1),
        "ttft_p95_ms": round(pct(ttfts, 95), 1),
        "ttft_p99_ms": round(pct(ttfts, 99), 1),
        "tpot_mean_ms": round(sum(tpots) / len(tpots), 2),
        "tpot_p50_ms": round(pct(tpots, 50), 2),
        "tpot_p90_ms": round(pct(tpots, 90), 2),
        "tpot_p95_ms": round(pct(tpots, 95), 2),
        "tpot_p99_ms": round(pct(tpots, 99), 2),
        "itl_mean_ms": round(sum(all_itls) / len(all_itls), 2),
        "itl_p50_ms": round(pct(all_itls, 50), 2),
        "itl_p95_ms": round(pct(all_itls, 95), 2),
        "itl_p99_ms": round(pct(all_itls, 99), 2),
        "e2el_mean_s": round(sum(e2es) / len(e2es), 2),
        "e2el_p50_s": round(pct(e2es, 50), 2),
        "e2el_p95_s": round(pct(e2es, 95), 2),
        "e2el_p99_s": round(pct(e2es, 99), 2),
    }
    print(f"[kimi-k31-bench] RESULT: {json.dumps(row)}", flush=True)
    return row


def render_markdown(meta: dict, rows: list[dict]) -> str:
    lines = [
        f"# `moonshotai/Kimi-K3.1-Pro` Benchmark Report — `{meta['config_tag']}`",
        "",
        f"- **Timestamp (UTC)**: `{meta['timestamp_utc']}`",
        f"- **Model**: `{meta['model_id']}` (`2.8T` total / `104B` active MoE, `69` KDA + `24` Gated MLA/DSA layers)",
        f"- **Hardware Topology**: `{meta['slice_name']}` (`16` physical TPU v7x chips = `32` TensorCores, `3,072 GB` HBM3e)",
        f"- **Configuration Tag**: `{meta['config_tag']}`",
        "",
        "| Suite | Workload | Concurrency | Target Cache % | Observed Cache % | Output tok/s | Total tok/s | Stream tok/s (Mean) | TTFT P50 (ms) | TTFT P95 (ms) | TTFT P99 (ms) | TPOT P50 (ms) | TPOT P95 (ms) | TPOT P99 (ms) | ITL P50 (ms) | E2EL P50 (s) | OK/Fail |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        if r.get("successful", 0) == 0:
            lines.append(
                f"| `{r['suite']}` | `{r['workload']}` | {r['concurrency']} | {r['prefix_hit_target_pct']}% | - | ERR | ERR | - | - | - | - | - | - | - | - | - | 0/{r.get('failed', 0)} |"
            )
        else:
            lines.append(
                f"| `{r['suite']}` | `{r['workload']}` | {r['concurrency']} | "
                f"{r['prefix_hit_target_pct']}% | {r['observed_prefix_hit_pct']}% | "
                f"**{r['output_tok_s']}** | **{r['total_tok_s']}** | {r['stream_tok_s_mean']} | "
                f"{r['ttft_p50_ms']} | {r.get('ttft_p95_ms', r['ttft_p90_ms'])} | {r['ttft_p99_ms']} | "
                f"{r['tpot_p50_ms']} | {r.get('tpot_p95_ms', r['tpot_p90_ms'])} | {r['tpot_p99_ms']} | "
                f"{r.get('itl_p50_ms', r['tpot_p50_ms'])} | {r['e2el_p50_s']} | {r['successful']}/{r['failed']} |"
            )
    lines.append("")
    return "\n".join(lines)


def save_incremental(out_dir: Path, meta: dict, rows: list[dict]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {"meta": meta, "results": rows}
    (out_dir / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (out_dir / "REPORT.md").write_text(render_markdown(meta, rows), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Automated benchmark matrix for moonshotai/Kimi-K3.1-Pro on TPU v7x."
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--tokenizer", default="/models/models/moonshotai/Kimi-K3.1-Pro")
    parser.add_argument("--slice-name", default="tpu-v7x-16-2x2x4-32cores")
    parser.add_argument("--config-tag", default="tp8-dp4-ep32")
    parser.add_argument(
        "--wave",
        choices=["all", "regular", "long_context", "prefix_caching"],
        default="all",
        help="Which benchmark wave(s) to execute",
    )
    parser.add_argument(
        "--regular-concurrencies",
        default="1,4,8,16,32,64",
        help="Comma-separated concurrencies for Regular Context (1k/1k, 8k/1k)",
    )
    parser.add_argument(
        "--long-concurrencies",
        default="1,2,4,8,16",
        help="Comma-separated concurrencies for Long Context (64k/1k, 64k/4k, 128k/1k)",
    )
    parser.add_argument(
        "--prefix-concurrencies",
        default="1,4,8,16",
        help="Comma-separated concurrencies for 64k Prefix Caching",
    )
    parser.add_argument(
        "--output-dir",
        default="results",
        help="Directory to write JSON and Markdown benchmark reports",
    )
    args = parser.parse_args()

    base_url = f"http://{args.host}:{args.port}"
    model_id = wait_for_server(base_url)
    builder = PromptBuilder(args.tokenizer)

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(args.output_dir) / args.config_tag
    existing_summary = out_dir / "summary.json"
    all_rows: list[dict] = []
    if existing_summary.exists():
        try:
            all_rows = json.loads(existing_summary.read_text(encoding="utf-8")).get("results", [])
        except Exception:
            all_rows = []

    meta = {
        "timestamp_utc": ts,
        "slice_name": args.slice_name,
        "config_tag": args.config_tag,
        "model_id": model_id,
        "wave": args.wave,
    }

    reg_concs = [int(x) for x in args.regular_concurrencies.split(",") if x.strip()]
    long_concs = [int(x) for x in args.long_concurrencies.split(",") if x.strip()]
    pref_concs = [int(x) for x in args.prefix_concurrencies.split(",") if x.strip()]

    existing_keys = {
        (r.get("suite"), r.get("workload"), r.get("concurrency"))
        for r in all_rows
        if r.get("successful", 0) > 0
    }

    # Wave 1: Regular Context (1k / 1k, 8k / 1k) across concurrency 1, 4, 8, 16, 32, 64
    if args.wave in ("all", "regular"):
        for workload_tag, in_toks, out_toks in [
            ("1k_1k", 1024, 1024),
            ("8k_1k", 8192, 1024),
        ]:
            for c in reg_concs:
                if ("regular_context", workload_tag, c) in existing_keys:
                    print(f"[kimi-k31-bench] Skipping completed regular_context {workload_tag} c={c}", flush=True)
                    continue
                n_prompts = c
                row = asyncio.run(
                    run_sweep_point(
                        base_url=base_url,
                        model_id=model_id,
                        builder=builder,
                        suite="regular_context",
                        workload_tag=workload_tag,
                        input_tokens=in_toks,
                        output_tokens=out_toks,
                        concurrency=c,
                        num_prompts=n_prompts,
                        prefix_hit_ratio=0.0,
                    )
                )
                all_rows.append(row)
                save_incremental(out_dir, meta, all_rows)

    # Wave 2: Long Context (64k / 1k, 64k / 4k, 128k / 1k) across concurrency 1, 2, 4, 8, 16
    if args.wave in ("all", "long_context"):
        for workload_tag, in_toks, out_toks in [
            ("64k_1k", 65536, 1024),
            ("64k_4k", 65536, 4096),
            ("128k_1k", 128000, 1024),
        ]:
            for c in long_concs:
                if ("long_context", workload_tag, c) in existing_keys:
                    print(f"[kimi-k31-bench] Skipping completed long_context {workload_tag} c={c}", flush=True)
                    continue
                n_prompts = c
                row = asyncio.run(
                    run_sweep_point(
                        base_url=base_url,
                        model_id=model_id,
                        builder=builder,
                        suite="long_context",
                        workload_tag=workload_tag,
                        input_tokens=in_toks,
                        output_tokens=out_toks,
                        concurrency=c,
                        num_prompts=n_prompts,
                        prefix_hit_ratio=0.0,
                    )
                )
                all_rows.append(row)
                save_incremental(out_dir, meta, all_rows)

    # Wave 3: Prefix Caching on 64k shared-prefix multi-turn workloads
    if args.wave in ("all", "prefix_caching"):
        long_64k_by_c = {
            r.get("concurrency"): r
            for r in all_rows
            if r.get("suite") == "long_context"
            and r.get("workload") == "64k_1k"
            and r.get("successful", 0) > 0
        }
        for hit_ratio, tag_suffix in [(0.0, "0pct_miss"), (0.8, "80pct_hit"), (0.95, "95pct_hit")]:
            for c in pref_concs:
                wtag = f"64k_1k_{tag_suffix}"
                if ("prefix_caching_64k", wtag, c) in existing_keys:
                    print(f"[kimi-k31-bench] Skipping completed prefix_caching_64k {wtag} c={c}", flush=True)
                    continue
                if hit_ratio == 0.0 and c in long_64k_by_c:
                    row = dict(long_64k_by_c[c])
                    row["suite"] = "prefix_caching_64k"
                    row["workload"] = wtag
                    print(f"[kimi-k31-bench] Reusing 0% miss baseline from long_context 64k_1k c={c}: {json.dumps(row)}", flush=True)
                    all_rows.append(row)
                    save_incremental(out_dir, meta, all_rows)
                    continue
                n_prompts = c
                row = asyncio.run(
                    run_sweep_point(
                        base_url=base_url,
                        model_id=model_id,
                        builder=builder,
                        suite="prefix_caching_64k",
                        workload_tag=wtag,
                        input_tokens=65536,
                        output_tokens=1024,
                        concurrency=c,
                        num_prompts=n_prompts,
                        prefix_hit_ratio=hit_ratio,
                    )
                )
                all_rows.append(row)
                save_incremental(out_dir, meta, all_rows)

    print("\n================ BENCHMARK MATRIX COMPLETE ================\n", flush=True)
    print(render_markdown(meta, all_rows), flush=True)
    print(f"[kimi-k31-bench] Saved results to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
