#!/usr/bin/env python3
"""100-spot coordination gate — tests S3 claim model under Spot worker contention."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import boto3
from botocore.exceptions import ClientError
from pokerbot.solver.label_completion import (
    analyze_shard_completion_s3,
    shard_completed_keys_s3,
    unique_manifest_keys,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BUCKET = os.environ.get("LABELGEN_S3_BUCKET", "video-processing-6789")
DEFAULT_PREFIX = os.environ.get("LABELGEN_COORD_PREFIX", "pokerbot/prod6000/coord_gate_100")
DEFAULT_INSTANCE_PROFILE = os.environ.get("LABELGEN_INSTANCE_PROFILE", "")
IMAGE_TAG = os.environ.get("LABELGEN_IMAGE_TAG", "pokerbot-labelgen:prod6000-slim")
GATE_SPOTS = 100
MAX_COLLISION_RATE = 0.15
STUCK_CLAIM_SEC = 300
DEFAULT_TIMEOUT_SEC = 21600
PROD6000_PREFIX = "pokerbot/prod6000"


def _ensure_image_on_prefix(
    s3,
    *,
    bucket: str,
    prefix: str,
    image_tag: str,
    skip_build: bool,
) -> None:
    manifest_key = f"{prefix}/image_manifest.json"
    try:
        s3.head_object(Bucket=bucket, Key=manifest_key)
        print(json.dumps({"image_manifest": "already_present", "prefix": prefix}), flush=True)
        return
    except ClientError:
        pass

    if skip_build:
        for source_prefix in (
            f"{PROD6000_PREFIX}/coord_gate_100",
            PROD6000_PREFIX,
            f"{PROD6000_PREFIX}/validate50_ec2",
        ):
            try:
                src_key = f"{source_prefix}/image_manifest.json"
                body = s3.get_object(Bucket=bucket, Key=src_key)["Body"].read()
                s3.put_object(
                    Bucket=bucket,
                    Key=manifest_key,
                    Body=body,
                    ContentType="application/json",
                )
                print(
                    json.dumps({"image_manifest": "copied", "from": source_prefix, "to": prefix}),
                    flush=True,
                )
                return
            except ClientError:
                continue

    sys.path.insert(0, str(ROOT / "scripts"))
    from labelgen_image import upload_labelgen_image

    upload_labelgen_image(
        s3,
        bucket=bucket,
        prefix=prefix,
        image_tag=image_tag,
        skip_if_exists=skip_build,
    )


def _run(cmd: list[str], *, check: bool = True) -> subprocess.CompletedProcess:
    print("+", " ".join(cmd), flush=True)
    return subprocess.run(cmd, cwd=ROOT, check=check)


def _load_manifest_expected(s3, bucket: str, prefix: str) -> list[str]:
    body = s3.get_object(Bucket=bucket, Key=f"{prefix}/work_manifest.json")["Body"].read()
    manifest = json.loads(body)
    return list(manifest.get("spot_keys", []))


def _list_completed_records(s3, bucket: str, prefix: str) -> list[dict]:
    records: list[dict] = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=f"{prefix}/completed/"):
        for obj in page.get("Contents", []):
            body = s3.get_object(Bucket=bucket, Key=obj["Key"])["Body"].read()
            records.append(json.loads(body))
    return records


def _list_active_claims(s3, bucket: str, prefix: str) -> list[dict]:
    claims: list[dict] = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=f"{prefix}/claims/"):
        for obj in page.get("Contents", []):
            body = s3.get_object(Bucket=bucket, Key=obj["Key"])["Body"].read()
            rec = json.loads(body)
            rec["_key"] = obj["Key"]
            claims.append(rec)
    return claims


def _aggregate_worker_metrics(s3, bucket: str, prefix: str) -> dict:
    totals = {
        "claims_attempted": 0,
        "claim_collisions": 0,
        "stale_claims_reclaimed": 0,
        "spots_solved": 0,
    }
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=f"{prefix}/metrics/workers/"):
        for obj in page.get("Contents", []):
            body = s3.get_object(Bucket=bucket, Key=obj["Key"])["Body"].read()
            m = json.loads(body)
            for k in totals:
                totals[k] += m.get(k, 0)
    rate = round(totals["claim_collisions"] / totals["claims_attempted"], 4) if totals["claims_attempted"] else 0.0
    totals["claim_collision_rate"] = rate
    return totals


def _stuck_claims(claims: list[dict], completed_keys: set[str]) -> list[dict]:
    now = datetime.now(timezone.utc)
    stuck: list[dict] = []
    for c in claims:
        spot = str(c.get("spot_key", ""))
        if spot in completed_keys:
            continue
        claimed_at = c.get("claimed_at")
        if not claimed_at:
            stuck.append(c)
            continue
        ts = datetime.fromisoformat(str(claimed_at).replace("Z", "+00:00"))
        age = (now - ts).total_seconds()
        if age >= STUCK_CLAIM_SEC:
            stuck.append(c)
    return stuck


def _evaluate_gate(
    *,
    expected_spots: list[str],
    shard_completed_keys: set[str],
    completed_records: list[dict],
    metrics: dict,
    stuck: list[dict],
) -> dict:
    expected_unique = unique_manifest_keys(expected_spots)
    expected_set = set(expected_unique)
    missing = sorted(expected_set - shard_completed_keys)

    spot_keys = [str(r.get("spot_key", "")) for r in completed_records]
    marker_duplicates = [k for k, n in Counter(spot_keys).items() if n > 1 and k]
    solved_sum = metrics.get("spots_solved", 0)

    checks = {
        "shard_count_exact": len(shard_completed_keys) == len(expected_set),
        "no_duplicate_completion_markers": len(marker_duplicates) == 0,
        "collision_rate_ok": metrics.get("claim_collision_rate", 0) <= MAX_COLLISION_RATE,
        "no_stuck_claims": len(stuck) == 0,
        "spots_solved_sum_ok": solved_sum >= len(expected_set),
    }
    return {
        "ok": all(checks.values()),
        "expected": len(expected_set),
        "expected_manifest_entries": len(expected_spots),
        "completed_unique_shards": len(shard_completed_keys),
        "completion_markers": len(set(spot_keys) - {""}),
        "marker_duplicates": marker_duplicates,
        "missing_spot_keys": missing[:20],
        "missing_count": len(missing),
        "metrics": metrics,
        "stuck_claims": len(stuck),
        "checks": checks,
    }


def _fetch_homeshard_hashes_file(host_path: str, *, output: Path) -> Path | None:
    """Fetch remote shard basenames via ssh and write one hash per line."""
    if not host_path or ":" not in host_path:
        return None
    host, remote_dir = host_path.split(":", 1)
    cmd = ["ssh", host, f"ls {remote_dir}"]
    print("+", " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        print(proc.stderr, file=sys.stderr)
        return None
    hashes: list[str] = []
    for line in proc.stdout.splitlines():
        name = line.strip()
        if name.endswith(".parquet"):
            hashes.append(name[: -len(".parquet")])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(hashes) + ("\n" if hashes else ""), encoding="utf-8")
    print(json.dumps({"homeshard_hashes": len(hashes), "file": str(output)}), flush=True)
    return output


def main() -> None:
    p = argparse.ArgumentParser(description="Run 100-spot coordination gate")
    p.add_argument("--bucket", default=DEFAULT_BUCKET)
    p.add_argument("--prefix", default=DEFAULT_PREFIX)
    p.add_argument("--run-id", default="coord_gate_100")
    p.add_argument("--region", default=os.environ.get("AWS_DEFAULT_REGION", "eu-central-1"))
    p.add_argument("--spots", type=int, default=GATE_SPOTS)
    p.add_argument("--max-instances", type=int, default=2)
    p.add_argument("--workers-per-instance", type=int, default=1)
    p.add_argument(
        "--instance-type",
        default="c6i.xlarge",
        help="EC2 instance type for each worker host (default c6i.xlarge)",
    )
    p.add_argument("--poll-sec", type=int, default=60)
    p.add_argument("--timeout-sec", type=int, default=DEFAULT_TIMEOUT_SEC)
    p.add_argument("--skip-build", action="store_true")
    p.add_argument("--skip-launch", action="store_true")
    p.add_argument("--monitor-only", action="store_true")
    p.add_argument(
        "--exclude-s3-prefix",
        action="append",
        default=[],
        help="Skip already-solved keys from these S3 prefixes when planning manifest",
    )
    p.add_argument(
        "--exclude-homeshard-ssh",
        default="",
        help="host:/path/to/shards — skip keys solved on homeserver (ssh ls *.parquet)",
    )
    p.add_argument(
        "--terminate-prior-run-tag",
        action="append",
        default=[],
        help="Terminate running labelgen instances for this Run tag before launch (repeatable)",
    )
    p.add_argument(
        "--confirm-homeserver-paused",
        action="store_true",
        help="Homeserver build_solver_labels is paused (required for prod6000 fleet on shared prefix)",
    )
    p.add_argument(
        "--allow-homeserver-concurrent",
        action="store_true",
        help="Allow launch while homeserver solves locally — safe for isolated coord_gate prefix only",
    )
    p.add_argument(
        "--allow-on-demand",
        action="store_true",
        help="Use On-Demand instead of Spot spare-capacity instances (not recommended)",
    )
    p.add_argument("--output", type=Path, default=Path("logs/coord_gate_report.json"))
    args = p.parse_args()

    if (
        not args.monitor_only
        and not args.skip_launch
        and not args.confirm_homeserver_paused
        and not args.allow_homeserver_concurrent
    ):
        print(
            "Refusing to launch: pass --confirm-homeserver-paused or --allow-homeserver-concurrent "
            "(isolated coord_gate prefix only)",
            file=sys.stderr,
        )
        sys.exit(2)

    if not DEFAULT_INSTANCE_PROFILE and not args.monitor_only:
        print("Set LABELGEN_INSTANCE_PROFILE", file=sys.stderr)
        sys.exit(2)

    manifest_path = ROOT / f"data/solver/work_manifest_{args.run_id}.json"
    launch_path = ROOT / f"logs/coord_gate_launch_{args.run_id}.json"
    monitor_path = ROOT / f"logs/coord_gate_monitor_{args.run_id}.json"
    homeshard_cache = ROOT / f"logs/homeshard_hashes_{args.run_id}.txt"

    if not args.monitor_only:
        plan_cmd = [
            sys.executable,
            "scripts/plan_label_work.py",
            "--config",
            "config/solver_labels_prod6000.yaml",
            "--output",
            str(manifest_path),
            "--max-spots",
            str(args.spots),
            "--run-id",
            args.run_id,
            "--s3-bucket",
            args.bucket,
        ]
        for exclude_prefix in args.exclude_s3_prefix:
            plan_cmd.extend(["--exclude-s3-prefix", exclude_prefix])
        timing_manifest = ROOT / "data/solver/manifest_prod600.json"
        if timing_manifest.is_file():
            plan_cmd.extend(["--timing-manifest", str(timing_manifest)])
        if args.exclude_homeshard_ssh:
            fetched = _fetch_homeshard_hashes_file(
                args.exclude_homeshard_ssh, output=homeshard_cache
            )
            if fetched is not None:
                plan_cmd.extend(["--exclude-shard-hashes-file", str(fetched)])
        _run(plan_cmd)

        if not args.skip_build:
            _run(
                [
                    "docker",
                    "build",
                    "--platform",
                    "linux/amd64",
                    "-f",
                    "infra/labelgen/Dockerfile",
                    "-t",
                    IMAGE_TAG,
                    ".",
                ]
            )

        s3 = boto3.client("s3", region_name=args.region)
        s3.upload_file(str(manifest_path), args.bucket, f"{args.prefix}/work_manifest.json")

        _ensure_image_on_prefix(
            s3,
            bucket=args.bucket,
            prefix=args.prefix,
            image_tag=IMAGE_TAG,
            skip_build=args.skip_build,
        )

        if not args.skip_launch:
            ec2 = boto3.client("ec2", region_name=args.region)
            sys.path.insert(0, str(ROOT / "scripts"))
            from launch_labelgen_spot import terminate_prior_run_instances

            prior_tags = list(dict.fromkeys([args.run_id, *args.terminate_prior_run_tag]))
            terminated: list[str] = []
            for tag in prior_tags:
                terminated.extend(
                    terminate_prior_run_instances(ec2, run_tag=tag, region=args.region)
                )
            if terminated:
                print(json.dumps({"terminated_prior_instances": terminated}), flush=True)

            launch_cmd = [
                sys.executable,
                "scripts/launch_labelgen_spot.py",
                "--bucket",
                args.bucket,
                "--prefix",
                args.prefix,
                "--region",
                args.region,
                "--max-instances",
                str(args.max_instances),
                "--max-spots-per-worker",
                "0",
                "--workers-per-instance",
                str(args.workers_per_instance),
                "--run-tag",
                args.run_id,
                "--terminate-prior",
                "--claim-ttl-sec",
                "600",
                "--instance-types",
                *([args.instance_type] * args.max_instances),
                "--output",
                str(launch_path.relative_to(ROOT)),
            ]
            if args.allow_on_demand:
                launch_cmd.append("--allow-on-demand")
            if DEFAULT_INSTANCE_PROFILE:
                launch_cmd.extend(["--instance-profile", DEFAULT_INSTANCE_PROFILE])
            r = subprocess.run(launch_cmd, cwd=ROOT, check=False)
            if launch_path.is_file():
                info = json.loads(launch_path.read_text(encoding="utf-8"))
                launched = [i for i in info.get("instances", []) if "instance_id" in i]
                if len(launched) < args.max_instances:
                    print(
                        json.dumps(
                            {
                                "warning": "partial_launch",
                                "launched": len(launched),
                                "requested": args.max_instances,
                                "hint": "Free vCPU capacity (pause other EC2) or use smaller instance types",
                            }
                        ),
                        flush=True,
                    )
            if r.returncode != 0:
                if launch_path.is_file():
                    info = json.loads(launch_path.read_text(encoding="utf-8"))
                    if any("instance_id" in i for i in info.get("instances", [])):
                        print("Launch returned non-zero but instances exist — continuing", flush=True)
                    else:
                        sys.exit(r.returncode)
                else:
                    sys.exit(r.returncode)

    s3 = boto3.client("s3", region_name=args.region)
    ec2 = boto3.client("ec2", region_name=args.region)
    expected = _load_manifest_expected(s3, args.bucket, args.prefix)
    expected_unique_count = len(unique_manifest_keys(expected))
    deadline = time.time() + args.timeout_sec
    instance_ids: list[str] = []
    if launch_path.is_file():
        launch_info = json.loads(launch_path.read_text(encoding="utf-8"))
        instance_ids = [i["instance_id"] for i in launch_info.get("instances", []) if "instance_id" in i]

    sys.path.insert(0, str(ROOT / "scripts"))
    from labelgen_ec2_diagnostics import list_running_labelgen_instances, report_instance_failure

    print("Monitoring coordination gate...", flush=True)
    while time.time() < deadline:
        completed_records = _list_completed_records(s3, args.bucket, args.prefix)
        metrics = _aggregate_worker_metrics(s3, args.bucket, args.prefix)
        shard_completed = shard_completed_keys_s3(
            s3, bucket=args.bucket, prefix=args.prefix, manifest_keys=expected
        )
        shard_report = analyze_shard_completion_s3(
            s3, bucket=args.bucket, prefix=args.prefix, manifest_keys=expected
        )
        claims = _list_active_claims(s3, args.bucket, args.prefix)
        stuck = _stuck_claims(claims, shard_completed)
        running = list_running_labelgen_instances(ec2, run_tag=args.run_id)

        status = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "run_id": args.run_id,
            "completed_shards": len(shard_completed),
            "expected_unique": expected_unique_count,
            "expected_manifest_entries": len(expected),
            "running_instances": running,
            "shard_completion": shard_report.to_dict(),
            **metrics,
            "stuck_claims": len(stuck),
        }
        print(json.dumps(status), flush=True)
        (monitor_path).write_text(json.dumps(status, indent=2), encoding="utf-8")

        if len(shard_completed) >= expected_unique_count:
            break

        if not running and len(shard_completed) < expected_unique_count:
            for iid in instance_ids:
                report_instance_failure(
                    ec2=ec2,
                    s3=s3,
                    instance_id=iid,
                    bucket=args.bucket,
                    prefix=args.prefix,
                    reason="workers_terminated_before_completion",
                    output_path=ROOT / "logs" / f"coord_gate_failure_{iid}.json",
                    log_prefix="coord_gate",
                    bootstrap_rel_key="logs/bootstrap_failure.log",
                )
            break

        time.sleep(args.poll_sec)

    completed_records = _list_completed_records(s3, args.bucket, args.prefix)
    metrics = _aggregate_worker_metrics(s3, args.bucket, args.prefix)
    shard_completed = shard_completed_keys_s3(
        s3, bucket=args.bucket, prefix=args.prefix, manifest_keys=expected
    )
    claims = _list_active_claims(s3, args.bucket, args.prefix)
    stuck = _stuck_claims(claims, shard_completed)
    report = _evaluate_gate(
        expected_spots=expected,
        shard_completed_keys=shard_completed,
        completed_records=completed_records,
        metrics=metrics,
        stuck=stuck,
    )
    report["generated_at"] = datetime.now(timezone.utc).isoformat()
    report["run_id"] = args.run_id
    report["prefix"] = args.prefix
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["ok"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
