#!/usr/bin/env python3
"""Run the 50-label cloud-vs-local validation gate (modernized EC2 + instance profile)."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import boto3
from botocore.exceptions import ClientError

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BUCKET = os.environ.get("LABELGEN_S3_BUCKET", "video-processing-6789")
DEFAULT_PREFIX = os.environ.get("LABELGEN_VALIDATE50_PREFIX", "pokerbot/prod6000/validate50_ec2")
def _default_region() -> str:
    session = boto3.session.Session()
    return session.region_name or os.environ.get("AWS_DEFAULT_REGION", "eu-central-1")


DEFAULT_REGION = _default_region()
DEFAULT_INSTANCE_PROFILE = os.environ.get("LABELGEN_INSTANCE_PROFILE", "")
DEFAULT_INSTANCE_TYPE = os.environ.get("LABELGEN_INSTANCE_TYPE", "c6i.2xlarge")
IMAGE_TAG = os.environ.get("LABELGEN_IMAGE_TAG", "pokerbot-labelgen:validate50-slim")
CONTAINER_GATE_MAX_SPOTS = int(os.environ.get("LABELGEN_CONTAINER_GATE_SPOTS", "3"))


def _run(cmd: list[str], *, cwd: Path | None = None, check: bool = True) -> subprocess.CompletedProcess:
    print("+", " ".join(cmd), flush=True)
    return subprocess.run(cmd, cwd=cwd or ROOT, check=check)


def _verify_iam(*, bucket: str, prefix: str, instance_profile: str, strict: bool = True) -> dict:
    cmd = [
        sys.executable,
        "scripts/verify_labelgen_iam.py",
        "--bucket",
        bucket,
        "--prefix",
        f"{prefix}/_iam_probe",
        "--region",
        DEFAULT_REGION,
    ]
    if strict:
        cmd.append("--strict")
    if instance_profile:
        cmd.extend(["--instance-profile", instance_profile])
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    print(r.stdout)
    if r.stderr:
        print(r.stderr, file=sys.stderr)
    try:
        return json.loads(r.stdout)
    except json.JSONDecodeError:
        return {"all_ok": False, "parse_error": True}


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
            "validate50_ec2_gate",
        ]
    )
    return manifest


def _docker_build(*, platform: str | None = None) -> None:
    cmd = ["docker", "build", "-f", "infra/labelgen/Dockerfile", "-t", IMAGE_TAG]
    if platform:
        cmd.extend(["--platform", platform])
    cmd.append(".")
    _run(cmd)


def _container_gate(*, manifest: Path, output_dir: Path, max_spots: int = CONTAINER_GATE_MAX_SPOTS) -> dict:
    """Smoke proof: label path in torch-free slim container (2-3 spots, not full validate50)."""
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "shards").mkdir(exist_ok=True)

    size_proc = subprocess.run(
        ["docker", "images", IMAGE_TAG, "--format", "{{.Size}}"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    image_size = size_proc.stdout.strip()

    torch_proc = subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            "--entrypoint",
            "python",
            IMAGE_TAG,
            "-c",
            "import importlib.util; import sys; sys.exit(0 if importlib.util.find_spec('torch') is None else 1)",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    if torch_proc.returncode != 0:
        raise SystemExit(f"container has torch installed: {torch_proc.stderr}")

    _docker_run_labels(manifest=manifest, output_dir=output_dir, max_spots=max_spots)

    local_m = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    spot_count = len(local_m.get("per_spot_results", []))
    unconverged = sum(1 for r in local_m.get("per_spot_results", []) if not r.get("converged", False))
    result = {
        "image_size": image_size,
        "torch_absent": True,
        "max_spots": max_spots,
        "spots_solved": spot_count,
        "unconverged": unconverged,
        "ok": spot_count >= max_spots and unconverged == 0,
    }
    gate_log = Path("logs/labelgen_container_gate.json")
    gate_log.parent.mkdir(parents=True, exist_ok=True)
    gate_log.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not result["ok"]:
        raise SystemExit("container gate failed")
    return result


def _docker_run_labels(*, manifest: Path, output_dir: Path, max_spots: int | None = None) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "shards").mkdir(exist_ok=True)
    cmd = [
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
    if max_spots is not None:
        cmd.extend(["--max-spots", str(max_spots)])
    _run(cmd)


def _upload_gate_artifacts(*, bucket: str, prefix: str, manifest: Path) -> dict:
    sys.path.insert(0, str(ROOT / "scripts"))
    from labelgen_image import upload_labelgen_image

    s3 = boto3.client("s3", region_name=DEFAULT_REGION)
    upload_info = upload_labelgen_image(
        s3,
        bucket=bucket,
        prefix=prefix,
        image_tag=IMAGE_TAG,
        skip_if_exists=True,
    )
    s3.upload_file(str(manifest), bucket, f"{prefix}/work_manifest.json")
    print(
        json.dumps(
            {
                "uploaded": [upload_info["s3_uri"], f"{prefix}/work_manifest.json"],
                **upload_info,
            }
        )
    )
    return upload_info


def _download_cloud_output(*, bucket: str, prefix: str, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    s3 = boto3.client("s3", region_name=DEFAULT_REGION)
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
    p = argparse.ArgumentParser(description="Validate50 cloud-vs-local gate (real EC2)")
    p.add_argument("--out-dir", type=Path, default=Path("data/validate50_ec2"))
    p.add_argument("--bucket", default=DEFAULT_BUCKET)
    p.add_argument("--prefix", default=DEFAULT_PREFIX)
    p.add_argument("--instance-profile", default=DEFAULT_INSTANCE_PROFILE)
    p.add_argument("--instance-type", default=DEFAULT_INSTANCE_TYPE)
    p.add_argument("--skip-iam-check", action="store_true")
    p.add_argument("--skip-docker-build", action="store_true")
    p.add_argument("--skip-container-gate", action="store_true")
    p.add_argument(
        "--container-gate-spots",
        type=int,
        default=CONTAINER_GATE_MAX_SPOTS,
        help="Spots for torch-free container smoke gate (default 3)",
    )
    p.add_argument("--skip-local", action="store_true")
    p.add_argument("--skip-cloud", action="store_true")
    p.add_argument("--skip-ec2-launch", action="store_true", help="Only download existing cloud output")
    p.add_argument("--platform", default=os.environ.get("LABELGEN_DOCKER_PLATFORM", "linux/amd64"))
    p.add_argument("--local-dir", type=Path, default=Path("data/validate50/local_docker"))
    args = p.parse_args()

    out_dir = args.out_dir
    local_dir = args.local_dir
    cloud_dir = out_dir / "cloud_ec2"
    report_path = Path("logs/cloud_label_validation_50_ec2.json")

    if not args.skip_iam_check and not args.skip_cloud:
        if not args.instance_profile:
            print(
                json.dumps(
                    {
                        "error": "missing_instance_profile",
                        "message": "Set LABELGEN_INSTANCE_PROFILE before EC2 gate retry",
                    }
                ),
                file=sys.stderr,
            )
            sys.exit(2)
        iam = _verify_iam(
            bucket=args.bucket,
            prefix=args.prefix,
            instance_profile=args.instance_profile,
            strict=True,
        )
        if not iam.get("all_ok"):
            print(
                "IAM strict verification failed — fix permissions before EC2 retry",
                file=sys.stderr,
            )
            sys.exit(1)

    manifest = _plan_manifest(out_dir)

    if not args.skip_docker_build:
        try:
            _docker_build(platform=args.platform)
        except subprocess.CalledProcessError:
            print("arm64 build failed — falling back to native platform", flush=True)
            _docker_build(platform=None)

    container_dir = out_dir / "container_gate"
    if not args.skip_container_gate:
        _container_gate(manifest=manifest, output_dir=container_dir, max_spots=args.container_gate_spots)

    if not args.skip_local:
        if local_dir.exists():
            shutil.rmtree(local_dir)
        _docker_run_labels(manifest=manifest, output_dir=local_dir)

    if not args.skip_cloud:
        _upload_gate_artifacts(bucket=args.bucket, prefix=args.prefix, manifest=manifest)
        if not args.skip_ec2_launch:
            launch_cmd = [
                sys.executable,
                "scripts/launch_validate50_ec2.py",
                "--bucket",
                args.bucket,
                "--prefix",
                args.prefix,
                "--region",
                DEFAULT_REGION,
                "--instance-type",
                args.instance_type,
                "--instance-profile",
                args.instance_profile,
                "--require-instance-profile",
                "--on-demand",
            ]
            launch_cmd.extend(["--wait", "--timeout-sec", "7200"])
            _run(launch_cmd)
        if cloud_dir.exists():
            shutil.rmtree(cloud_dir)
        _download_cloud_output(bucket=args.bucket, prefix=args.prefix, dest=cloud_dir)

    for d, label in ((local_dir, "local"), (cloud_dir, "cloud")):
        mf = d / "manifest.json"
        if not mf.is_file():
            raise SystemExit(f"missing {label} manifest: {mf}")

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
    per_spot_iters = [
        {
            "spot": spot,
            "local_iterations": local_iters[spot],
            "cloud_iterations": cloud_iters[spot],
            "delta": abs(local_iters[spot] - cloud_iters[spot]),
        }
        for spot in sorted(set(local_iters) & set(cloud_iters))
    ]

    near_timeout = [
        r
        for r in cloud_m.get("per_spot_results", [])
        if r.get("converged") and float(r.get("elapsed_sec") or 0) >= 0.95 * 3600
    ]

    base.update(
        {
            "gate_id": "validate50_ec2",
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "spot_count": 50,
            "max_exploitability_pct": max_exploit,
            "unconverged_count": unconverged,
            "ratio_check": ratio,
            "ratio_ok": 1.5 <= ratio["ratio_vanilla_bunching"] <= 2.5,
            "exploitability_ok": max_exploit <= 0.5,
            "unconverged_ok": unconverged == 0,
            "per_spot_iterations": per_spot_iters,
            "near_timeout_cloud": len(near_timeout),
            "docker_image": IMAGE_TAG,
            "instance_profile": args.instance_profile,
            "instance_type": args.instance_type,
            "cloud_arm": "real_ec2_on_demand",
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
