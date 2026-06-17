#!/usr/bin/env python3
"""Launch EC2 spot workers for prod6000 label generation."""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import boto3
from botocore.exceptions import ClientError

DEFAULT_BUCKET = os.environ.get("LABELGEN_S3_BUCKET", "video-processing-6789")
DEFAULT_PREFIX = os.environ.get("LABELGEN_S3_PREFIX", "pokerbot/prod6000")
DEFAULT_REGION = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
IMAGE_TAG = os.environ.get("LABELGEN_IMAGE_TAG", "pokerbot-labelgen:prod6000-slim")
# EC2 instance profile name (required at launch). Workers get S3 access via this role — no keys on instances.
DEFAULT_INSTANCE_PROFILE = os.environ.get("LABELGEN_INSTANCE_PROFILE", "")

# Graviton Ubuntu 22.04 arm64 (us-east-1) — resolved at launch if unset
DEFAULT_AMI_ARM64 = os.environ.get("LABELGEN_AMI_ARM64", "")
DEFAULT_AMI_X86 = os.environ.get("LABELGEN_AMI_X86", "ami-0faab6bdbac9486fb")

# Spot price per vCPU-hr estimates for budget tracking
VCPU_HOURLY_EST = {
    "c6g.2xlarge": 0.0128 * 8,
    "c7g.2xlarge": 0.0138 * 8,
    "c6i.2xlarge": 0.017 * 8,
    "c7i.2xlarge": 0.018 * 8,
}


def _resolve_ami(ec2: Any, *, arch: str) -> str:
    pinned = DEFAULT_AMI_ARM64 if arch == "arm64" else DEFAULT_AMI_X86
    if pinned:
        try:
            resp = ec2.describe_images(ImageIds=[pinned])
            if resp.get("Images"):
                return pinned
        except ClientError:
            pass
    name_filter = (
        "ubuntu/images/hvm-ssd/ubuntu-jammy-22.04-arm64-server-*"
        if arch == "arm64"
        else "ubuntu/images/hvm-ssd/ubuntu-jammy-22.04-amd64-server-*"
    )
    resp = ec2.describe_images(
        Owners=["099720109477"],
        Filters=[
            {"Name": "name", "Values": [name_filter]},
            {"Name": "state", "Values": ["available"]},
            {"Name": "architecture", "Values": [arch]},
        ],
    )
    images = sorted(resp.get("Images", []), key=lambda i: i["CreationDate"], reverse=True)
    if not images:
        raise RuntimeError(f"No Ubuntu 22.04 AMI found for {arch}")
    return images[0]["ImageId"]


def _user_data(
    *,
    bucket: str,
    prefix: str,
    instance_type: str,
    worker_tag: str,
    max_spots: int,
    arch: str,
    workers_per_instance: int,
    idle_exit_sec: int = 180,
    claim_ttl_sec: int = 7200,
) -> str:
    workers_line = (
        "WORKERS_PER_INSTANCE=$(nproc)"
        if workers_per_instance <= 0
        else f"WORKERS_PER_INSTANCE={workers_per_instance}"
    )
    return f"""#!/bin/bash
set -euxo pipefail
export DEBIAN_FRONTEND=noninteractive
export AWS_DEFAULT_REGION={DEFAULT_REGION}
export LABELGEN_S3_BUCKET={bucket}
export LABELGEN_S3_PREFIX={prefix}
export LABELGEN_INSTANCE_TYPE={instance_type}
export LABELGEN_ARCH={arch}

{workers_line}
export WORKERS_PER_INSTANCE

cat > /usr/local/bin/stop-all-workers.sh <<'STOPEOF'
#!/bin/bash
for i in $(seq 0 $((WORKERS_PER_INSTANCE - 1))); do
  docker stop -t 30 labelgen-worker-$i 2>/dev/null || true
done
STOPEOF
chmod +x /usr/local/bin/stop-all-workers.sh

# Spot interruption handler — graceful stop so workers release S3 claims
cat > /usr/local/bin/spot-interrupt.sh << 'SPOTEOF'
#!/bin/bash
while sleep 5; do
  if curl -sf http://169.254.169.254/latest/meta-data/spot/termination-time; then
    echo "Spot termination notice received" >> /var/log/labelgen.log
    /usr/local/bin/stop-all-workers.sh
    shutdown -h now
    exit 0
  fi
done
SPOTEOF
chmod +x /usr/local/bin/spot-interrupt.sh
/usr/local/bin/spot-interrupt.sh &

apt-get update -qq
apt-get install -y -qq docker.io awscli gzip curl python3
systemctl start docker
until docker info >/dev/null 2>&1; do sleep 1; done

mkdir -p /opt/labelgen
aws s3 cp s3://{bucket}/{prefix}/image_manifest.json /tmp/image_manifest.json
IMAGE_S3_KEY=$(python3 -c "import json; print(json.load(open('/tmp/image_manifest.json'))['s3_key'])")
aws s3 cp s3://{bucket}/$IMAGE_S3_KEY /tmp/image.tar.gz
gunzip -c /tmp/image.tar.gz | docker load
docker image inspect {IMAGE_TAG}

metadata_get() {{
  local path="$1"
  local token val
  for _ in $(seq 1 30); do
    token=$(curl -sf -X PUT "http://169.254.169.254/latest/api/token" \\
      -H "X-aws-ec2-metadata-token-ttl-seconds: 21600" 2>/dev/null || true)
    if [[ -n "$token" ]]; then
      val=$(curl -sf -H "X-aws-ec2-metadata-token: $token" \\
        "http://169.254.169.254/latest${{path}}" 2>/dev/null || true)
    else
      val=$(curl -sf "http://169.254.169.254/latest${{path}}" 2>/dev/null || true)
    fi
    if [[ -n "$val" ]]; then
      echo "$val"
      return 0
    fi
    sleep 2
  done
  return 1
}}

INSTANCE_ID=$(metadata_get /meta-data/instance-id || echo "unknown")

WORKER_ID="{worker_tag}-$INSTANCE_ID"

for i in $(seq 0 $((WORKERS_PER_INSTANCE - 1))); do
  docker run -d --name labelgen-worker-$i \\
    -e LABELGEN_S3_BUCKET={bucket} \\
    -e LABELGEN_S3_PREFIX={prefix} \\
    -e AWS_DEFAULT_REGION={DEFAULT_REGION} \\
    -e RAYON_NUM_THREADS=1 \\
    -e AWS_EC2_METADATA_DISABLED=false \\
    -e "HOSTNAME=$WORKER_ID-cpu$i" \\
    {IMAGE_TAG} \\
    --bucket {bucket} \\
    --prefix {prefix} \\
    --worker-id "$WORKER_ID-cpu$i" \\
    --max-spots {max_spots} \\
    --idle-exit-sec {idle_exit_sec} \\
    --claim-ttl-sec {claim_ttl_sec}
done

: > /var/log/labelgen.log
for i in $(seq 0 $((WORKERS_PER_INSTANCE - 1))); do
  docker wait labelgen-worker-$i || true
  docker logs labelgen-worker-$i >> /var/log/labelgen.log 2>&1 || true
  aws s3 cp /var/log/labelgen.log s3://{bucket}/{prefix}/logs/$WORKER_ID-cpu$i.log || true
done

shutdown -h now
"""


def _instance_arch(instance_type: str) -> str:
    if instance_type.startswith(("c6g", "c7g", "m6g", "m7g", "r6g", "r7g", "t4g")):
        return "arm64"
    return "x86_64"


def _estimate_hourly(instance_type: str) -> float:
    return VCPU_HOURLY_EST.get(instance_type, 0.15)


def terminate_prior_run_instances(
    ec2: Any,
    *,
    run_tag: str,
    region: str = DEFAULT_REGION,
) -> list[str]:
    """Terminate running labelgen instances for run_tag before a new launch."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from labelgen_ec2_diagnostics import list_running_labelgen_instances

    ids = list_running_labelgen_instances(ec2, run_tag=run_tag)
    if not ids:
        return []
    ec2.terminate_instances(InstanceIds=ids)
    return ids


def _launch_worker(
    ec2: Any,
    *,
    bucket: str,
    prefix: str,
    instance_type: str,
    ami: str,
    worker_tag: str,
    max_spots: int,
    arch: str,
    spot_only: bool,
    instance_profile: str,
    workers_per_instance: int,
    run_tag: str = "prod6000",
    idle_exit_sec: int = 180,
    claim_ttl_sec: int = 7200,
) -> dict[str, Any]:
    ud = base64.b64encode(
        _user_data(
            bucket=bucket,
            prefix=prefix,
            instance_type=instance_type,
            worker_tag=worker_tag,
            max_spots=max_spots,
            arch=arch,
            workers_per_instance=workers_per_instance,
            idle_exit_sec=idle_exit_sec,
            claim_ttl_sec=claim_ttl_sec,
        ).encode()
    ).decode()

    params: dict[str, Any] = {
        "ImageId": ami,
        "InstanceType": instance_type,
        "MinCount": 1,
        "MaxCount": 1,
        "UserData": ud,
        "IamInstanceProfile": {"Name": instance_profile},
        "MetadataOptions": {
            "HttpTokens": "required",
            "HttpPutResponseHopLimit": 2,
            "InstanceMetadataTags": "enabled",
        },
        "TagSpecifications": [
            {
                "ResourceType": "instance",
                "Tags": [
                    {"Key": "Name", "Value": f"pokerbot-labelgen-{worker_tag}"},
                    {"Key": "Project", "Value": "pokerbot-labelgen"},
                    {"Key": "Run", "Value": run_tag},
                    {"Key": "InstanceType", "Value": instance_type},
                ],
            }
        ],
    }

    if spot_only:
        params["InstanceMarketOptions"] = {
            "MarketType": "spot",
            "SpotOptions": {
                "SpotInstanceType": "one-time",
                "InstanceInterruptionBehavior": "terminate",
            },
        }

    resp = ec2.run_instances(**params)
    inst = resp["Instances"][0]
    return {
        "instance_id": inst["InstanceId"],
        "instance_type": instance_type,
        "ami": ami,
        "arch": arch,
        "spot": spot_only,
        "worker_tag": worker_tag,
        "instance_profile": instance_profile,
        "launched_at": datetime.now(timezone.utc).isoformat(),
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Launch labelgen EC2 spot fleet")
    p.add_argument("--bucket", default=DEFAULT_BUCKET)
    p.add_argument("--prefix", default=DEFAULT_PREFIX)
    p.add_argument("--region", default=DEFAULT_REGION)
    p.add_argument("--budget-usd", type=float, default=20.0)
    p.add_argument(
        "--allow-on-demand",
        action="store_true",
        help="Use On-Demand instead of Spot spare-capacity (default: Spot)",
    )
    p.add_argument("--max-instances", type=int, default=2)
    p.add_argument("--instance-types", nargs="+", default=["c6g.2xlarge", "c7g.2xlarge"])
    p.add_argument("--max-spots-per-worker", type=int, default=0, help="0 = unlimited per worker process")
    p.add_argument(
        "--workers-per-instance",
        type=int,
        default=0,
        help="Docker worker processes per EC2 instance (0 = nproc)",
    )
    p.add_argument("--run-tag", default="prod6000", help="EC2 tag:Run value for filtering")
    p.add_argument(
        "--terminate-prior",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Terminate existing running instances with the same --run-tag before launch",
    )
    p.add_argument("--idle-exit-sec", type=int, default=180)
    p.add_argument("--claim-ttl-sec", type=int, default=7200)
    p.add_argument("--arch", choices=["arm64", "x86_64", "auto"], default="auto")
    p.add_argument(
        "--instance-profile",
        default=DEFAULT_INSTANCE_PROFILE,
        help="EC2 instance profile name (env: LABELGEN_INSTANCE_PROFILE). Required.",
    )
    p.add_argument("--output", type=Path, default=Path("logs/prod6000_cloud_launch.json"))
    args = p.parse_args()

    if not args.instance_profile:
        print(
            json.dumps(
                {
                    "error": "missing_instance_profile",
                    "message": "Set LABELGEN_INSTANCE_PROFILE or pass --instance-profile",
                    "hint": "Create IAM role with iam_policy_labelgen_worker_role.json, attach instance profile, pass name here",
                },
                indent=2,
            ),
            file=sys.stderr,
        )
        sys.exit(2)

    spot_only = not args.allow_on_demand
    ec2 = boto3.client("ec2", region_name=args.region)

    terminated_prior: list[str] = []
    if args.terminate_prior:
        terminated_prior = terminate_prior_run_instances(
            ec2, run_tag=args.run_tag, region=args.region
        )
        if terminated_prior:
            print(
                json.dumps(
                    {
                        "terminated_prior_instances": terminated_prior,
                        "run_tag": args.run_tag,
                    }
                ),
                flush=True,
            )
            time.sleep(15)

    arch = args.arch
    if arch == "auto":
        arch = _instance_arch(args.instance_types[0])

    try:
        ami = _resolve_ami(ec2, arch=arch)
    except ClientError as e:
        if arch == "arm64":
            print(json.dumps({"error": "arm64_ami_failed", "detail": str(e), "fallback": "x86_64"}))
            arch = "x86_64"
            ami = _resolve_ami(ec2, arch=arch)
            args.instance_types = ["c6i.2xlarge", "c7i.2xlarge"]
        else:
            raise

    hourly_est = sum(_estimate_hourly(t) for t in args.instance_types[: args.max_instances])
    budget_hours = args.budget_usd / max(hourly_est, 0.01)

    launched: list[dict[str, Any]] = []
    for i, itype in enumerate(args.instance_types[: args.max_instances]):
        tag = f"{itype.replace('.', '-')}-{i}"
        try:
            rec = _launch_worker(
                ec2,
                bucket=args.bucket,
                prefix=args.prefix,
                instance_type=itype,
                ami=ami,
                worker_tag=tag,
                max_spots=args.max_spots_per_worker,
                arch=arch,
                spot_only=spot_only,
                instance_profile=args.instance_profile,
                workers_per_instance=args.workers_per_instance,
                run_tag=args.run_tag,
                idle_exit_sec=args.idle_exit_sec,
                claim_ttl_sec=args.claim_ttl_sec,
            )
            launched.append(rec)
        except ClientError as e:
            launched.append({"instance_type": itype, "error": e.response.get("Error", {})})

    report = {
        "launched_at": datetime.now(timezone.utc).isoformat(),
        "bucket": args.bucket,
        "prefix": args.prefix,
        "arch": arch,
        "ami": ami,
        "spot_only": spot_only,
        "instance_profile": args.instance_profile,
        "budget_usd": args.budget_usd,
        "estimated_hourly_usd": round(hourly_est, 4),
        "budget_hours_estimate": round(budget_hours, 2),
        "terminated_prior_instances": terminated_prior,
        "idle_exit_sec": args.idle_exit_sec,
        "claim_ttl_sec": args.claim_ttl_sec,
        "instances": launched,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))

    if not any("instance_id" in r for r in launched):
        sys.exit(1)


if __name__ == "__main__":
    main()
