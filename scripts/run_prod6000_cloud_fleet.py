#!/usr/bin/env python3
"""Orchestrate prod6000 EC2 spot fleet: IAM check, sync, build, launch, monitor."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import boto3

from pokerbot.solver.label_completion import (
    shard_completed_keys_s3,
    unique_manifest_keys,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BUCKET = os.environ.get("LABELGEN_S3_BUCKET", "video-processing-6789")
DEFAULT_PREFIX = os.environ.get("LABELGEN_S3_PREFIX", "pokerbot/prod6000")
DEFAULT_INSTANCE_PROFILE = os.environ.get("LABELGEN_INSTANCE_PROFILE", "")
IMAGE_TAG = "pokerbot-labelgen:prod6000-slim"
STUCK_PLATEAU_POLLS = 5
STUCK_CLAIM_SEC = 300


def _run(cmd: list[str], *, check: bool = True) -> subprocess.CompletedProcess:
    print("+", " ".join(cmd), flush=True)
    return subprocess.run(cmd, cwd=ROOT, check=check)


def _verify_iam() -> bool:
    cmd = [sys.executable, "scripts/verify_labelgen_iam.py", "--bucket", DEFAULT_BUCKET, "--strict"]
    if DEFAULT_INSTANCE_PROFILE:
        cmd.extend(["--instance-profile", DEFAULT_INSTANCE_PROFILE])
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    print(r.stdout)
    if r.returncode != 0:
        print(r.stderr, file=sys.stderr)
    return r.returncode == 0


def _docker_build(*, platform: str = "linux/arm64") -> None:
    _run(
        [
            "docker",
            "build",
            "--platform",
            platform,
            "-f",
            "infra/labelgen/Dockerfile",
            "-t",
            IMAGE_TAG,
            ".",
        ]
    )


def _load_manifest_keys(s3, bucket: str, prefix: str) -> list[str]:
    try:
        body = s3.get_object(Bucket=bucket, Key=f"{prefix}/work_manifest.json")["Body"].read()
        manifest = json.loads(body)
        return list(manifest.get("spot_keys", []))
    except Exception:
        return []


def _count_shard_completed(s3, bucket: str, prefix: str, manifest_keys: list[str]) -> int:
    if not manifest_keys:
        return 0
    return len(shard_completed_keys_s3(s3, bucket=bucket, prefix=prefix, manifest_keys=manifest_keys))


def _list_stuck_claims(
    s3,
    bucket: str,
    prefix: str,
    *,
    manifest_keys: list[str],
) -> list[dict]:
    """Claims older than STUCK_CLAIM_SEC without a shard-backed completion."""
    shard_completed = shard_completed_keys_s3(
        s3, bucket=bucket, prefix=prefix, manifest_keys=manifest_keys
    )
    now = datetime.now(timezone.utc)
    stuck: list[dict] = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=f"{prefix}/claims/"):
        for obj in page.get("Contents", []):
            try:
                body = s3.get_object(Bucket=bucket, Key=obj["Key"])["Body"].read()
                rec = json.loads(body)
                claimed_at = rec.get("claimed_at")
                if not claimed_at:
                    stuck.append(rec)
                    continue
                ts = datetime.fromisoformat(str(claimed_at).replace("Z", "+00:00"))
                age = (now - ts).total_seconds()
                if age >= STUCK_CLAIM_SEC:
                    stuck.append(rec)
            except Exception:
                continue
    stuck = [c for c in stuck if str(c.get("spot_key", "")) not in shard_completed]
    return stuck


def _aggregate_metrics(s3, bucket: str, prefix: str) -> dict:
    collisions = 0
    attempts = 0
    solved = 0
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=f"{prefix}/metrics/workers/"):
        for obj in page.get("Contents", []):
            try:
                body = s3.get_object(Bucket=bucket, Key=obj["Key"])["Body"].read()
                m = json.loads(body)
                collisions += m.get("claim_collisions", 0)
                attempts += m.get("claims_attempted", 0)
                solved += m.get("spots_solved", 0)
            except Exception:
                continue
    rate = round(collisions / attempts, 4) if attempts else 0.0
    return {
        "claims_attempted": attempts,
        "claim_collisions": collisions,
        "claim_collision_rate": rate,
        "spots_solved_cloud": solved,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Run prod6000 EC2 spot fleet")
    p.add_argument("--bucket", default=DEFAULT_BUCKET)
    p.add_argument("--prefix", default=DEFAULT_PREFIX)
    p.add_argument("--budget-usd", type=float, default=20.0)
    p.add_argument("--max-instances", type=int, default=2)
    p.add_argument("--instance-types", nargs="+", default=["c6g.2xlarge", "c7g.2xlarge"])
    p.add_argument("--skip-iam", action="store_true")
    p.add_argument("--skip-build", action="store_true")
    p.add_argument("--skip-launch", action="store_true")
    p.add_argument("--monitor-only", action="store_true")
    p.add_argument("--target", type=int, default=0, help="0 = use work_manifest spot count")
    p.add_argument("--poll-sec", type=int, default=60)
    p.add_argument("--region", default=os.environ.get("AWS_DEFAULT_REGION", "us-east-1"))
    p.add_argument(
        "--confirm-homeserver-paused",
        action="store_true",
        help="Required before launch — homeserver must not solve concurrently with cloud",
    )
    args = p.parse_args()

    if not args.monitor_only and not args.skip_launch and not args.confirm_homeserver_paused:
        print(
            "Refusing to launch: pass --confirm-homeserver-paused after pausing homeserver build_solver_labels",
            file=sys.stderr,
        )
        sys.exit(2)

    if not args.skip_iam and not args.monitor_only:
        if not _verify_iam():
            print("IAM verification failed", file=sys.stderr)
            sys.exit(1)

    if not args.monitor_only:
        if not args.skip_build:
            try:
                _docker_build(platform="linux/arm64")
            except subprocess.CalledProcessError:
                print("arm64 build failed — falling back to linux/amd64", flush=True)
                _docker_build(platform="linux/amd64")
                args.instance_types = ["c6i.2xlarge"]

        _run(
            [
                sys.executable,
                "scripts/sync_prod6000_to_s3.py",
                "--bucket",
                args.bucket,
                "--prefix",
                args.prefix,
                "--plan-manifest",
                "--image-tag",
                IMAGE_TAG,
            ]
        )

        if not args.skip_launch:
            launch_cmd = [
                sys.executable,
                "scripts/launch_labelgen_spot.py",
                "--bucket",
                args.bucket,
                "--prefix",
                args.prefix,
                "--region",
                args.region,
                "--budget-usd",
                str(args.budget_usd),
                "--max-instances",
                str(args.max_instances),
                "--max-spots-per-worker",
                "0",
                *["--instance-types", *args.instance_types],
            ]
            if DEFAULT_INSTANCE_PROFILE:
                launch_cmd.extend(["--instance-profile", DEFAULT_INSTANCE_PROFILE])
            _run(launch_cmd)

    s3 = boto3.client("s3", region_name=args.region)
    ec2 = boto3.client("ec2", region_name=args.region)
    sys.path.insert(0, str(ROOT / "scripts"))
    from labelgen_ec2_diagnostics import list_running_labelgen_instances, report_instance_failure

    manifest_keys = _load_manifest_keys(s3, args.bucket, args.prefix)
    target = args.target if args.target > 0 else len(unique_manifest_keys(manifest_keys)) or 3000

    launch_path = ROOT / "logs/prod6000_cloud_launch.json"
    instance_ids: list[str] = []
    if launch_path.is_file():
        launch_info = json.loads(launch_path.read_text(encoding="utf-8"))
        instance_ids = [i["instance_id"] for i in launch_info.get("instances", []) if "instance_id" in i]

    print(f"Monitoring completion (target={target})...", flush=True)
    plateau_polls = 0
    last_completed = -1

    while True:
        completed = _count_shard_completed(s3, args.bucket, args.prefix, manifest_keys)
        metrics = _aggregate_metrics(s3, args.bucket, args.prefix)
        stuck = _list_stuck_claims(
            s3, args.bucket, args.prefix, manifest_keys=manifest_keys
        )
        running = list_running_labelgen_instances(ec2, run_tag="prod6000")

        if completed == last_completed:
            plateau_polls += 1
        else:
            plateau_polls = 0
            last_completed = completed

        status = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "completed_shards": completed,
            "target_unique": target,
            "manifest_entries": len(manifest_keys),
            "remaining": max(0, target - completed),
            "running_instances": running,
            "stuck_claims": len(stuck),
            "plateau_polls": plateau_polls,
            **metrics,
        }
        print(json.dumps(status), flush=True)
        (ROOT / "logs/prod6000_cloud_monitor.json").write_text(json.dumps(status, indent=2), encoding="utf-8")

        if completed >= target:
            break

        if plateau_polls >= STUCK_PLATEAU_POLLS and not running and completed < target:
            failure = {
                "status": "stuck_incomplete",
                "completed": completed,
                "target": target,
                "stuck_claims": stuck[:20],
                "stuck_count": len(stuck),
            }
            (ROOT / "logs/prod6000_cloud_stuck.json").write_text(json.dumps(failure, indent=2), encoding="utf-8")
            for iid in instance_ids:
                report_instance_failure(
                    ec2=ec2,
                    s3=s3,
                    instance_id=iid,
                    bucket=args.bucket,
                    prefix=args.prefix,
                    reason="completion_plateau_workers_gone",
                    output_path=ROOT / "logs" / f"prod6000_fleet_failure_{iid}.json",
                    log_prefix="prod6000_fleet",
                )
            print(json.dumps(failure, indent=2), file=sys.stderr)
            sys.exit(1)

        time.sleep(args.poll_sec)


if __name__ == "__main__":
    main()
