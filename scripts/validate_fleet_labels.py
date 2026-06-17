#!/usr/bin/env python3
"""Post-fleet validation: shard-authoritative completeness + per-worker samples."""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import boto3

from pokerbot.solver.label_completion import (
    analyze_shard_completion_s3,
    shard_completed_keys_s3,
    spot_key_hash,
    unique_manifest_keys,
)

ROOT = Path(__file__).resolve().parents[1]


def _load_manifest(s3, bucket: str, prefix: str) -> list[str]:
    body = s3.get_object(Bucket=bucket, Key=f"{prefix}/work_manifest.json")["Body"].read()
    return list(json.loads(body).get("spot_keys", []))


def _load_completed(s3, bucket: str, prefix: str) -> list[dict]:
    records: list[dict] = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=f"{prefix}/completed/"):
        for obj in page.get("Contents", []):
            body = s3.get_object(Bucket=bucket, Key=obj["Key"])["Body"].read()
            records.append(json.loads(body))
    return records


def _stratified_samples(
    completed: list[dict],
    *,
    samples_per_worker: int,
) -> dict[str, list[str]]:
    by_worker: dict[str, list[str]] = defaultdict(list)
    for rec in completed:
        wid = str(rec.get("worker_id", "unknown"))
        spot = str(rec.get("spot_key", ""))
        if spot:
            by_worker[wid].append(spot)

    out: dict[str, list[str]] = {}
    for wid, spots in by_worker.items():
        k = min(samples_per_worker, len(spots))
        out[wid] = random.sample(spots, k) if k else []
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Validate fleet label completeness and per-worker samples")
    p.add_argument("--bucket", default="video-processing-6789")
    p.add_argument("--prefix", default="pokerbot/prod6000")
    p.add_argument("--reference-shard-dir", type=Path, default=None, help="Homeserver reference shards")
    p.add_argument("--samples-per-worker", type=int, default=3)
    p.add_argument("--output", type=Path, default=Path("logs/fleet_validation_report.json"))
    p.add_argument("--region", default="us-east-1")
    args = p.parse_args()

    s3 = boto3.client("s3", region_name=args.region)
    manifest_keys = _load_manifest(s3, args.bucket, args.prefix)
    expected_unique = unique_manifest_keys(manifest_keys)
    shard_report = analyze_shard_completion_s3(
        s3, bucket=args.bucket, prefix=args.prefix, manifest_keys=manifest_keys
    )
    shard_completed = shard_completed_keys_s3(
        s3, bucket=args.bucket, prefix=args.prefix, manifest_keys=manifest_keys
    )
    missing = sorted(set(expected_unique) - shard_completed)

    completed = _load_completed(s3, args.bucket, args.prefix)
    stratified = _stratified_samples(completed, samples_per_worker=args.samples_per_worker)
    semantic_results: dict[str, dict] = {}

    if args.reference_shard_dir and args.reference_shard_dir.is_dir():
        shard_checks: dict[str, dict] = {}
        for wid, spots in stratified.items():
            ok = 0
            missing_local = []
            for spot in spots:
                digest = spot_key_hash(spot)
                cloud_key = f"{args.prefix}/shards/{digest}.parquet"
                ref_path = args.reference_shard_dir / f"{digest}.parquet"
                try:
                    s3.head_object(Bucket=args.bucket, Key=cloud_key)
                    if ref_path.is_file():
                        ok += 1
                    else:
                        missing_local.append(spot)
                except Exception:
                    missing_local.append(spot)
            shard_checks[wid] = {
                "spots_sampled": len(spots),
                "reference_shards_found": ok,
                "missing": missing_local[:10],
            }
        semantic_results = shard_checks

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "expected_unique": len(expected_unique),
        "expected_manifest_entries": len(manifest_keys),
        "completed_shards": len(shard_completed),
        "shard_completion": shard_report.to_dict(),
        "complete": len(shard_completed) >= len(expected_unique) and not missing,
        "missing_count": len(missing),
        "missing_spot_keys": missing[:50],
        "workers": sorted(stratified.keys()),
        "stratified_samples": {k: v for k, v in stratified.items()},
        "semantic_by_worker": semantic_results,
        "ok": len(missing) == 0,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["ok"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
