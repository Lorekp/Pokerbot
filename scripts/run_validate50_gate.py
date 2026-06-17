#!/usr/bin/env python3
"""Run the 50-label cloud-vs-local validation gate end-to-end."""

from __future__ import annotations

import argparse
import gzip
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import boto3

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BUCKET = os.environ.get("LABELGEN_S3_BUCKET", "video-processing-6789")
DEFAULT_PREFIX = "pokerbot/validate50"
IMAGE_TAG = "pokerbot-labelgen:validate50"


def _run(cmd: list[str], *, cwd: Path | None = None, check: bool = True) -> subprocess.CompletedProcess:
    print("+", " ".join(cmd), flush=True)
    return subprocess.run(cmd, cwd=cwd or ROOT, check=check)


def _plan_manifest(out_dir: Path) -> Path:
    manifest = out_dir / "work_manifest.json"
    _run(
        [
            sys.executable,
            "scripts/plan_label_work.py",
            "--config",
            "config/solver_labels_validate50.yaml",
            "--output",
            str(manifest),
            "--max-spots",
            "50",
            "--run-id",
            "validate50_gate",
        ]
    )
    return manifest


def _docker_build() -> None:
    _run(
        [
            "docker",
            "build",
            "-f",
            "infra/labelgen/Dockerfile",
            "-t",
            IMAGE_TAG,
            ".",
        ]
    )


def _docker_run_labels(*, manifest: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    shards = output_dir / "shards"
    shards.mkdir(exist_ok=True)
    _run(
        [
            "docker",
            "run",
            "--rm",
            "-v",
            f"{manifest.resolve()}:/manifest.json:ro",
            "-v",
            f"{output_dir.resolve()}:/output",
            "--entrypoint",
            "python",
            IMAGE_TAG,
            "scripts/build_solver_labels.py",
            "--config",
            "config/solver_labels_validate50.yaml",
            "--work-manifest",
            "/manifest.json",
            "--output",
            "/output/train.parquet",
            "--manifest",
            "/output/manifest.json",
            "--report",
            "/output/run.md",
            "--weva-index",
            "data/weva",
            "--solver-backend",
            "subprocess",
            "--num-workers",
            "1",
            "--shard-dir",
            "/output/shards",
        ]
    )


def _upload_gate_artifacts(*, bucket: str, prefix: str, manifest: Path) -> None:
    s3 = boto3.client("s3")
    with tempfile.NamedTemporaryFile(suffix=".tar", delete=False) as tmp:
        tar_path = Path(tmp.name)
    gz_path = tar_path.with_suffix(".tar.gz")
    try:
        subprocess.run(["docker", "save", IMAGE_TAG, "-o", str(tar_path)], check=True)
        with open(tar_path, "rb") as f_in, gzip.open(gz_path, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
        s3.upload_file(str(gz_path), bucket, f"{prefix}/image.tar.gz")
        s3.upload_file(str(manifest), bucket, f"{prefix}/work_manifest.json")
    finally:
        tar_path.unlink(missing_ok=True)
        gz_path.unlink(missing_ok=True)


def _download_cloud_output(*, bucket: str, prefix: str, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    s3 = boto3.client("s3")
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=f"{prefix}/cloud/"):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            rel = key[len(f"{prefix}/cloud/") :]
            if not rel:
                continue
            local = dest / rel
            local.parent.mkdir(parents=True, exist_ok=True)
            s3.download_file(bucket, key, str(local))


def _ratio_check(manifest_path: Path) -> dict:
    m = json.loads(manifest_path.read_text(encoding="utf-8"))
    counts: Counter[str] = Counter()
    for row in m.get("per_spot_results", []):
        spot = str(row.get("spot", ""))
        if "|" in spot:
            counts[spot.split("|", 1)[0]] += 1
    vanilla = counts.get("sb_vs_bb_flop", 0) + counts.get("btn_vs_bb_flop", 0)
    bunching = counts.get("bunching_postflop", 0)
    return {
        "sb_vs_bb_flop": counts.get("sb_vs_bb_flop", 0),
        "btn_vs_bb_flop": counts.get("btn_vs_bb_flop", 0),
        "bunching_postflop": bunching,
        "vanilla": vanilla,
        "bunching": bunching,
        "ratio_vanilla_bunching": round(vanilla / max(bunching, 1), 3),
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Validate50 cloud-vs-local gate")
    p.add_argument("--out-dir", type=Path, default=Path("data/validate50"))
    p.add_argument("--bucket", default=DEFAULT_BUCKET)
    p.add_argument("--prefix", default=DEFAULT_PREFIX)
    p.add_argument("--skip-docker-build", action="store_true")
    p.add_argument("--skip-local", action="store_true")
    p.add_argument("--skip-cloud", action="store_true")
    p.add_argument("--skip-ec2-launch", action="store_true", help="Only download existing cloud output")
    p.add_argument("--workers", type=int, default=1, choices=[1, 2])
    args = p.parse_args()

    out_dir = args.out_dir
    local_dir = out_dir / "local_docker"
    cloud_dir = out_dir / "cloud"
    report_path = Path("logs/cloud_label_validation_50.json")

    manifest = _plan_manifest(out_dir)

    if not args.skip_docker_build:
        _docker_build()

    if not args.skip_local:
        if local_dir.exists():
            shutil.rmtree(local_dir)
        _docker_run_labels(manifest=manifest, output_dir=local_dir)

    if not args.skip_cloud:
        _upload_gate_artifacts(bucket=args.bucket, prefix=args.prefix, manifest=manifest)
        if not args.skip_ec2_launch:
            if args.workers == 1:
                _run(
                    [
                        sys.executable,
                        "scripts/launch_validate50_ec2.py",
                        "--bucket",
                        args.bucket,
                        "--prefix",
                        args.prefix,
                        "--wait",
                    ]
                )
            else:
                _run(
                    [
                        sys.executable,
                        "scripts/launch_validate50_ec2.py",
                        "--bucket",
                        args.bucket,
                        "--prefix",
                        args.prefix,
                        "--manifest-slice",
                        "0:25",
                        "--wait",
                    ]
                )
                _run(
                    [
                        sys.executable,
                        "scripts/launch_validate50_ec2.py",
                        "--bucket",
                        args.bucket,
                        "--prefix",
                        args.prefix,
                        "--manifest-slice",
                        "25:50",
                        "--wait",
                    ]
                )
        if cloud_dir.exists():
            shutil.rmtree(cloud_dir)
        _download_cloud_output(bucket=args.bucket, prefix=args.prefix, dest=cloud_dir)

    # Normalize manifest paths for validate_cloud_labels (expects manifest.json + train.parquet)
    for d in (local_dir, cloud_dir):
        mf = d / "manifest.json"
        if not mf.is_file():
            raise SystemExit(f"missing {mf}")

    _run(
        [
            sys.executable,
            "scripts/validate_cloud_labels.py",
            "--local-dir",
            str(local_dir),
            "--cloud-dir",
            str(cloud_dir),
            "--output",
            str(report_path),
        ]
    )

    base = json.loads(report_path.read_text(encoding="utf-8"))
    local_m = json.loads((local_dir / "manifest.json").read_text(encoding="utf-8"))
    cloud_m = json.loads((cloud_dir / "manifest.json").read_text(encoding="utf-8"))

    max_exploit = float(local_m.get("convergence", {}).get("max_exploitability_pct", 999))
    unconverged = sum(
        1 for r in local_m.get("per_spot_results", []) if not r.get("converged", False)
    )
    ratio = _ratio_check(local_dir / "manifest.json")

    per_spot_iters = []
    local_iters = {
        str(r.get("spot")): int(r["iterations_used"])
        for r in local_m.get("per_spot_results", [])
        if r.get("iterations_used") is not None
    }
    cloud_iters = {
        str(r.get("spot")): int(r["iterations_used"])
        for r in cloud_m.get("per_spot_results", [])
        if r.get("iterations_used") is not None
    }
    for spot in sorted(set(local_iters) & set(cloud_iters)):
        per_spot_iters.append(
            {
                "spot": spot,
                "local_iterations": local_iters[spot],
                "cloud_iterations": cloud_iters[spot],
                "delta": abs(local_iters[spot] - cloud_iters[spot]),
            }
        )

    base.update(
        {
            "gate_id": "validate50",
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "spot_count": 50,
            "max_exploitability_pct": max_exploit,
            "unconverged_count": unconverged,
            "ratio_check": ratio,
            "ratio_ok": 1.5 <= ratio["ratio_vanilla_bunching"] <= 2.5,
            "exploitability_ok": max_exploit <= 0.5,
            "unconverged_ok": unconverged == 0,
            "per_spot_iterations": per_spot_iters,
            "docker_image": IMAGE_TAG,
            "s3_prefix": f"s3://{args.bucket}/{args.prefix}",
        }
    )
    base["ok"] = (
        base.get("semantic_diff_passed", False)
        and base.get("exploitability_ok", False)
        and base.get("unconverged_ok", False)
        and base.get("ratio_ok", False)
        and base.get("iterations_used", {}).get("divergence_count", 1) == 0
    )
    report_path.write_text(json.dumps(base, indent=2), encoding="utf-8")
    print(json.dumps(base, indent=2))
    if not base["ok"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
