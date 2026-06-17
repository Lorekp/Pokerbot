#!/usr/bin/env python3
"""Final report for prod6000 cloud EC2 run."""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import boto3

from pokerbot.solver.export import merge_shards
from pokerbot.solver.label_completion import CompletionLedger, spot_key_hash

ROOT = Path(__file__).resolve().parents[1]


def _list_completed(s3, bucket: str, prefix: str) -> list[dict]:
    records = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=f"{prefix}/completed/"):
        for obj in page.get("Contents", []):
            body = s3.get_object(Bucket=bucket, Key=obj["Key"])["Body"].read()
            records.append(json.loads(body))
    return records


def _download_shards(s3, bucket: str, prefix: str, dest: Path) -> int:
    dest.mkdir(parents=True, exist_ok=True)
    n = 0
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=f"{prefix}/shards/"):
        for obj in page.get("Contents", []):
            name = obj["Key"].split("/")[-1]
            local = dest / name
            if not local.is_file():
                s3.download_file(bucket, obj["Key"], str(local))
                n += 1
    return n


def main() -> None:
    p = argparse.ArgumentParser(description="Report prod6000 cloud EC2 run")
    p.add_argument("--bucket", default="video-processing-6789")
    p.add_argument("--prefix", default="pokerbot/prod6000")
    p.add_argument("--homeserver-shard-dir", type=Path, default=None)
    p.add_argument("--homeserver-ledger", type=Path, default=None)
    p.add_argument("--output", type=Path, default=Path("logs/prod6000_cloud_ec2_report.json"))
    p.add_argument("--region", default="us-east-1")
    args = p.parse_args()

    s3 = boto3.client("s3", region_name=args.region)
    completed = _list_completed(s3, args.bucket, args.prefix)

    near_timeout = [r for r in completed if r.get("near_timeout")]
    cloud_sources = Counter(r.get("source", "?") for r in completed)

    metrics_agg = {"claims_attempted": 0, "claim_collisions": 0, "spots_solved": 0}
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=args.bucket, Prefix=f"{args.prefix}/metrics/workers/"):
        for obj in page.get("Contents", []):
            m = json.loads(s3.get_object(Bucket=args.bucket, Key=obj["Key"])["Body"].read())
            for k in metrics_agg:
                metrics_agg[k] += m.get(k, 0)

    collision_rate = (
        round(metrics_agg["claim_collisions"] / metrics_agg["claims_attempted"], 4)
        if metrics_agg["claims_attempted"]
        else 0.0
    )

    # Merge shards
    merged_dir = ROOT / "data/solver/train_prod6000_merged_shards"
    merged_dir.mkdir(parents=True, exist_ok=True)
    if args.homeserver_shard_dir and args.homeserver_shard_dir.is_dir():
        import shutil

        for p in args.homeserver_shard_dir.glob("*.parquet"):
            if not p.name.endswith(".tmp"):
                shutil.copy2(p, merged_dir / p.name)
    _download_shards(s3, args.bucket, args.prefix, merged_dir)

    out_parquet = ROOT / "data/solver/train_prod6000.parquet"
    row_count = merge_shards(merged_dir, out_parquet, spot_key_hash_fn=spot_key_hash)

    exploit_values = [r["exploitability_pct"] for r in completed if r.get("exploitability_pct") is not None]
    elapsed_values = [r["elapsed_sec"] for r in completed if r.get("elapsed_sec") is not None]

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "completed_total": len(completed),
        "merged_parquet_rows": row_count,
        "cloud_sources": dict(cloud_sources),
        "claim_collision_rate": collision_rate,
        "metrics": metrics_agg,
        "near_timeout_count": len(near_timeout),
        "max_exploitability_pct": max(exploit_values) if exploit_values else None,
        "median_elapsed_sec": statistics.median(elapsed_values) if elapsed_values else None,
        "target_complete": len(completed) >= 6000 and row_count >= 6000,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
