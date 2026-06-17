#!/usr/bin/env python3
"""Build deterministic interleaved spot_key manifest for cloud/local label workers."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import boto3

from pokerbot.solver.label_completion import (
    shard_hashes_present_s3,
    spot_key_hash,
)
from pokerbot.solver.spot_enumerator import enumerate_spot_keys, load_solver_config


def _load_exclude_shard_hashes(
    *,
    s3: Any | None,
    bucket: str,
    exclude_s3_prefixes: list[str],
    exclude_shard_dirs: list[Path],
    exclude_shard_hashes_files: list[Path],
) -> set[str]:
    done: set[str] = set()
    if s3 is not None:
        for prefix in exclude_s3_prefixes:
            done |= shard_hashes_present_s3(s3, bucket=bucket, prefix=prefix)
    for shard_dir in exclude_shard_dirs:
        if shard_dir.is_dir():
            done |= {
                p.stem
                for p in shard_dir.glob("*.parquet")
                if not p.name.endswith(".tmp")
            }
    for path in exclude_shard_hashes_files:
        if path.is_file():
            for line in path.read_text(encoding="utf-8").splitlines():
                h = line.strip()
                if h:
                    done.add(h)
    return done


def main() -> None:
    p = argparse.ArgumentParser(description="Plan label work manifest")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--max-spots", type=int, default=None)
    p.add_argument("--run-id", type=str, default=None)
    p.add_argument(
        "--s3-bucket",
        default=os.environ.get("LABELGEN_S3_BUCKET", "video-processing-6789"),
        help="Bucket for --exclude-s3-prefix shard scans",
    )
    p.add_argument(
        "--exclude-s3-prefix",
        action="append",
        default=[],
        help="Skip manifest keys whose shard exists under this S3 prefix (repeatable)",
    )
    p.add_argument(
        "--exclude-shard-dir",
        type=Path,
        action="append",
        default=[],
        help="Skip keys with a local Parquet shard in this directory (repeatable)",
    )
    p.add_argument(
        "--exclude-shard-hashes-file",
        type=Path,
        action="append",
        default=[],
        help="Skip keys whose shard hash appears in this file, one per line (repeatable)",
    )
    p.add_argument(
        "--timing-manifest",
        type=Path,
        default=None,
        help="Prior run manifest JSON; embed board_timing medians into work manifest",
    )
    args = p.parse_args()

    config = load_solver_config(args.config)
    keys = enumerate_spot_keys(config)
    exclude_hashes = _load_exclude_shard_hashes(
        s3=boto3.client("s3") if args.exclude_s3_prefix else None,
        bucket=args.s3_bucket,
        exclude_s3_prefixes=args.exclude_s3_prefix,
        exclude_shard_dirs=args.exclude_shard_dir,
        exclude_shard_hashes_files=args.exclude_shard_hashes_file,
    )
    if exclude_hashes:
        keys = [k for k in keys if spot_key_hash(k) not in exclude_hashes]
    if args.max_spots is not None:
        keys = keys[: args.max_spots]

    run_id = args.run_id or datetime.now(timezone.utc).strftime("labelgen_%Y%m%d_%H%M%S")
    cfg_bytes = args.config.read_bytes()
    manifest = {
        "run_id": run_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config_path": str(args.config),
        "config_hash": hashlib.sha256(cfg_bytes).hexdigest()[:16],
        "enumeration_mode": config.enumeration_mode,
        "disable_bunching": config.disable_bunching,
        "spot_keys": keys,
        "entries": {k: {"status": "pending"} for k in keys},
    }
    if args.timing_manifest is not None:
        timing_src = json.loads(args.timing_manifest.read_text(encoding="utf-8"))
        board_timing = timing_src.get("board_timing")
        if isinstance(board_timing, dict) and board_timing:
            manifest["board_timing"] = board_timing
            medians = sorted(
                float(v["median_sec"])
                for v in board_timing.values()
                if isinstance(v, dict) and "median_sec" in v
            )
            if medians:
                manifest["slow_board_threshold_sec"] = medians[len(medians) // 2]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "run_id": run_id,
                "count": len(keys),
                "excluded_shard_hashes": len(exclude_hashes),
                "output": str(args.output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
