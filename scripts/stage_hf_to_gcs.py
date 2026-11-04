#!/usr/bin/env python3
# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0
"""Multi-host / multi-threaded HuggingFace -> GCS weight streamer for moonshotai/Kimi-K3.1-Pro."""

from __future__ import annotations

import argparse
import concurrent.futures
import os
import time

from google.cloud import storage
from huggingface_hub import HfApi, hf_hub_url
import requests


def transfer_one_file(
    repo_id: str,
    rfilename: str,
    expected_size: int,
    bucket_name: str,
    prefix: str,
    hf_token: str | None,
) -> tuple[str, str, int]:
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob_path = f"{prefix.rstrip('/')}/{repo_id}/{rfilename}"
    blob = bucket.blob(blob_path)
    if blob.exists():
        blob.reload()
        if expected_size > 0 and blob.size == expected_size:
            return (rfilename, "SKIPPED_EXISTS", blob.size)

    url = hf_hub_url(repo_id=repo_id, filename=rfilename)
    headers = {"Authorization": f"Bearer {hf_token}"} if hf_token else {}
    t0 = time.time()
    for attempt in range(1, 5):
        written = 0
        try:
            with requests.get(url, headers=headers, stream=True, timeout=180) as resp:
                resp.raise_for_status()
                with blob.open("wb", chunk_size=64 * 1024 * 1024) as gcs_f:
                    for chunk in resp.iter_content(chunk_size=16 * 1024 * 1024):
                        if chunk:
                            gcs_f.write(chunk)
                            written += len(chunk)
            dt = max(time.time() - t0, 1e-3)
            mbps = (written / (1024 * 1024)) / dt
            print(
                f"[stage-gcs] DONE {repo_id}/{rfilename} ({written/1e9:.2f} GB in {dt:.1f}s = {mbps:.1f} MiB/s)",
                flush=True,
            )
            return (rfilename, "UPLOADED", written)
        except Exception as exc:
            print(f"[stage-gcs] WARN attempt {attempt}/4 failed for {repo_id}/{rfilename}: {exc}", flush=True)
            if attempt == 4:
                raise
            time.sleep(4 * attempt)
    return (rfilename, "FAILED", 0)


def stage_repo(
    repo_id: str,
    bucket_name: str,
    prefix: str,
    workers: int,
    hf_token: str | None,
    shard_index: int = 0,
    num_shards: int = 1,
) -> None:
    api = HfApi(token=hf_token)
    info = api.repo_info(repo_id=repo_id, repo_type="model", files_metadata=True)
    siblings = sorted(
        [s for s in info.siblings if not s.rfilename.startswith(".git")],
        key=lambda x: x.rfilename,
    )
    my_siblings = [s for idx, s in enumerate(siblings) if idx % num_shards == shard_index]
    total_bytes = sum((s.size or 0) for s in my_siblings)
    print(
        f"[stage-gcs] === Starting {repo_id} [worker-shard {shard_index}/{num_shards}] -> "
        f"gs://{bucket_name}/{prefix}/{repo_id}/ ({len(my_siblings)} files, {total_bytes/1e9:.2f} GB, threads={workers}) ===",
        flush=True,
    )
    t_repo = time.time()
    uploaded_bytes = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {
            pool.submit(
                transfer_one_file,
                repo_id,
                s.rfilename,
                s.size or 0,
                bucket_name,
                prefix,
                hf_token,
            ): s.rfilename
            for s in my_siblings
        }
        done_cnt = 0
        for fut in concurrent.futures.as_completed(futs):
            _, _, nbytes = fut.result()
            done_cnt += 1
            uploaded_bytes += nbytes
            if done_cnt % 4 == 0 or done_cnt == len(my_siblings):
                elapsed = max(time.time() - t_repo, 1.0)
                print(
                    f"[stage-gcs] Progress {repo_id} [{shard_index}/{num_shards}]: {done_cnt}/{len(my_siblings)} files "
                    f"({uploaded_bytes/1e9:.1f}/{total_bytes/1e9:.1f} GB, elapsed={elapsed:.0f}s)",
                    flush=True,
                )


def main() -> None:
    parser = argparse.ArgumentParser(description="Stream moonshotai/Kimi-K3.1-Pro from HuggingFace to GCS.")
    parser.add_argument("--repo-id", default="moonshotai/Kimi-K3.1-Pro")
    parser.add_argument("--bucket", required=True, help="Target GCS bucket name")
    parser.add_argument("--prefix", default="models")
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--shard-index", type=int, default=0, help="Host shard index (0..num_shards-1)")
    parser.add_argument("--num-shards", type=int, default=1, help="Total number of hosts staging in parallel")
    args = parser.parse_args()

    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    stage_repo(
        repo_id=args.repo_id,
        bucket_name=args.bucket,
        prefix=args.prefix,
        workers=args.workers,
        hf_token=hf_token,
        shard_index=args.shard_index,
        num_shards=args.num_shards,
    )


if __name__ == "__main__":
    main()
