#!/usr/bin/env python3
"""Sync homeserver prod6000 ledger + shards to S3 for distributed cloud workers."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import boto3
from botocore.exceptions import ClientError

from pokerbot.solver.label_completion import CompletionLedger, has_shard, spot_key_hash

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BUCKET = os.environ.get("LABELGEN_S3_BUCKET", "video-processing-6789")
DEFAULT_PREFIX = os.environ.get("LABELGEN_S3_PREFIX", "pokerbot/prod6000")


def _upload_ledger_entries(
    s3, bucket: str, prefix: str, ledger: CompletionLedger, shard_dir: Path
) -> int:
    uploaded = 0
    for key, rec in ledger.completed.items():
        if not has_shard(shard_dir, key):
            continue
        digest = spot_key_hash(key)
        s3_key = f"{prefix}/completed/{digest}.json"
        try:
            s3.head_object(Bucket=bucket, Key=s3_key)
            continue
        except ClientError:
            pass
        body = json.dumps(rec, sort_keys=True).encode()
        s3.put_object(Bucket=bucket, Key=s3_key, Body=body, ContentType="application/json")
        uploaded += 1
    return uploaded


def _upload_shards(s3, bucket: str, prefix: str, shard_dir: Path) -> int:
    uploaded = 0
    if not shard_dir.is_dir():
        return 0
    for path in shard_dir.glob("*.parquet"):
        if path.name.endswith(".tmp"):
            continue
        s3_key = f"{prefix}/shards/{path.name}"
        try:
            s3.head_object(Bucket=bucket, Key=s3_key)
            continue
        except ClientError:
            pass
        s3.upload_file(str(path), bucket, s3_key)
        uploaded += 1
    return uploaded


def main() -> None:
    p = argparse.ArgumentParser(description="Sync prod6000 state to S3")
    p.add_argument("--bucket", default=DEFAULT_BUCKET)
    p.add_argument("--prefix", default=DEFAULT_PREFIX)
    p.add_argument("--ledger", type=Path, default=ROOT / "data/solver/train_prod6000_completed.jsonl")
    p.add_argument("--shard-dir", type=Path, default=ROOT / "data/solver/train_prod6000_shards")
    p.add_argument("--config", type=Path, default=ROOT / "config/solver_labels_prod6000.yaml")
    p.add_argument("--plan-manifest", action="store_true", help="Generate and upload work manifest")
    p.add_argument("--upload-image", type=str, default=None, help="Deprecated: path to image.tar.gz")
    p.add_argument("--image-tag", default=None, help="Docker image tag for content-addressed S3 upload")
    p.add_argument("--region", default="us-east-1")
    p.add_argument("--output", type=Path, default=Path("logs/prod6000_s3_sync.json"))
    args = p.parse_args()

    s3 = boto3.client("s3", region_name=args.region)
    ledger = CompletionLedger.load(args.ledger)

    if args.plan_manifest:
        manifest_path = ROOT / "data/solver/work_manifest_prod6000.json"
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts/plan_label_work.py"),
                "--config",
                str(args.config),
                "--output",
                str(manifest_path),
                "--run-id",
                "prod6000",
            ],
            check=True,
            cwd=ROOT,
        )
        s3.upload_file(str(manifest_path), args.bucket, f"{args.prefix}/work_manifest.json")

    ledger_uploaded = _upload_ledger_entries(s3, args.bucket, args.prefix, ledger, args.shard_dir)
    shards_uploaded = _upload_shards(s3, args.bucket, args.prefix, args.shard_dir)

    image_upload: dict | None = None
    if args.image_tag:
        sys.path.insert(0, str(ROOT / "scripts"))
        from labelgen_image import upload_labelgen_image

        image_upload = upload_labelgen_image(
            s3,
            bucket=args.bucket,
            prefix=args.prefix,
            image_tag=args.image_tag,
            skip_if_exists=True,
        )
    elif args.upload_image:
        sys.path.insert(0, str(ROOT / "scripts"))
        from labelgen_image import image_content_hash, image_s3_key, write_image_manifest

        content_hash = image_content_hash()
        s3_key = image_s3_key(args.prefix, content_hash)
        s3.upload_file(args.upload_image, args.bucket, s3_key)
        image_upload = write_image_manifest(
            bucket=args.bucket,
            prefix=args.prefix,
            content_hash=content_hash,
            image_tag="unknown",
            uploaded_new=True,
        )
        s3.put_object(
            Bucket=args.bucket,
            Key=f"{args.prefix}/image_manifest.json",
            Body=json.dumps(image_upload, indent=2).encode(),
            ContentType="application/json",
        )

    report = {
        "synced_at": datetime.now(timezone.utc).isoformat(),
        "bucket": args.bucket,
        "prefix": args.prefix,
        "ledger_total": len(ledger.completed),
        "ledger_new_uploads": ledger_uploaded,
        "shards_new_uploads": shards_uploaded,
        "image_upload": image_upload,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    s3.put_object(
        Bucket=args.bucket,
        Key=f"{args.prefix}/sync_report.json",
        Body=json.dumps(report, indent=2).encode(),
        ContentType="application/json",
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
