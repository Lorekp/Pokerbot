#!/usr/bin/env python3
"""Launch EC2 spot instance for validate50 label gate (boto3)."""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import time
from pathlib import Path

import boto3
from botocore.exceptions import ClientError

ROOT = Path(__file__).resolve().parents[1]

DEFAULT_BUCKET = os.environ.get("LABELGEN_S3_BUCKET", "video-processing-6789")
DEFAULT_PREFIX = os.environ.get("LABELGEN_VALIDATE50_PREFIX", "pokerbot/prod6000/validate50_ec2")
DEFAULT_INSTANCE = os.environ.get("LABELGEN_INSTANCE_TYPE", "c6i.2xlarge")
DEFAULT_INSTANCE_PROFILE = os.environ.get("LABELGEN_INSTANCE_PROFILE", "")
DEFAULT_AMI_ARM64 = os.environ.get("LABELGEN_AMI_ARM64", "")
# eu-central-1 Ubuntu 22.04 amd64; override via LABELGEN_AMI_X86 per region
DEFAULT_AMI_X86 = os.environ.get("LABELGEN_AMI_X86", "ami-0faab6bdbac9486fb")
IMAGE_TAG = os.environ.get("LABELGEN_IMAGE_TAG", "pokerbot-labelgen:validate50-slim")


def _default_region() -> str:
    return boto3.session.Session().region_name or os.environ.get("AWS_DEFAULT_REGION", "eu-central-1")


DEFAULT_REGION = _default_region()


def _launcher_creds() -> dict[str, str]:
    creds = boto3.session.Session().get_credentials()
    if creds is None:
        raise RuntimeError("no AWS credentials for launcher")
    frozen = creds.get_frozen_credentials()
    out = {"access_key": frozen.access_key, "secret_key": frozen.secret_key}
    if frozen.token:
        out["token"] = frozen.token
    return out


def _resolve_ami(ec2, *, instance_type: str, region: str) -> str:
    arch = "arm64" if instance_type.startswith(("c6g", "c7g", "m6g", "m7g")) else "x86_64"
    pinned = DEFAULT_AMI_ARM64 if arch == "arm64" else DEFAULT_AMI_X86
    if pinned:
        try:
            ec2.describe_images(ImageIds=[pinned])
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
        ],
    )
    images = sorted(resp.get("Images", []), key=lambda i: i["CreationDate"], reverse=True)
    for img in images:
        if img.get("Architecture") == arch:
            return img["ImageId"]
    raise RuntimeError(f"No Ubuntu 22.04 AMI for {arch} in {region}")


def _user_data(
    *,
    bucket: str,
    prefix: str,
    slice_spec: str | None,
    region: str,
    image_tag: str,
    creds: dict[str, str] | None,
) -> str:
    slice_arg = f"--manifest-slice {slice_spec}" if slice_spec else ""
    cred_block = ""
    docker_cred_env = ""
    if creds:
        cred_block = f"""export AWS_ACCESS_KEY_ID='{creds["access_key"]}'
export AWS_SECRET_ACCESS_KEY='{creds["secret_key"]}'
"""
        if creds.get("token"):
            cred_block += f"export AWS_SESSION_TOKEN='{creds['token']}'\n"
        docker_cred_env = (
            "  -e AWS_ACCESS_KEY_ID \\\n  -e AWS_SECRET_ACCESS_KEY \\\n"
            + ("  -e AWS_SESSION_TOKEN \\\n" if creds.get("token") else "")
        )
    script = f"""#!/bin/bash
set -euxo pipefail
LOG=/var/log/validate50-bootstrap.log
exec > >(tee -a "$LOG") 2>&1
fail() {{
  echo "bootstrap_failed $(date -Is)" >&2
  aws s3 cp "$LOG" s3://{bucket}/{prefix}/cloud/bootstrap_failure.log || true
  shutdown -h now || true
}}
trap fail ERR
export DEBIAN_FRONTEND=noninteractive
export AWS_DEFAULT_REGION={region}
{cred_block}apt-get update -qq
apt-get install -y -qq docker.io awscli gzip curl python3
systemctl start docker
until docker info >/dev/null 2>&1; do sleep 1; done
mkdir -p /opt/validate50/output
aws s3 cp s3://{bucket}/{prefix}/image_manifest.json /tmp/image_manifest.json
IMAGE_S3_KEY=$(python3 -c "import json; print(json.load(open('/tmp/image_manifest.json'))['s3_key'])")
aws s3 cp s3://{bucket}/$IMAGE_S3_KEY /tmp/image.tar.gz
gunzip -c /tmp/image.tar.gz | docker load
docker image inspect {image_tag}
aws s3 cp s3://{bucket}/{prefix}/work_manifest.json /opt/validate50/work_manifest.json
docker run --rm \\
  -e AWS_DEFAULT_REGION={region} \\
{docker_cred_env}  -v /opt/validate50/output:/output \\
  -v /opt/validate50/work_manifest.json:/manifest.json:ro \\
  --entrypoint python \\
  {image_tag} \\
  scripts/build_solver_labels.py \\
  --config config/solver_labels_validate50.yaml \\
  --work-manifest /manifest.json \\
  {slice_arg} \\
  --output /output/train.parquet \\
  --manifest /output/manifest.json \\
  --report /output/run.md \\
  --weva-index data/weva \\
  --solver-backend subprocess \\
  --num-workers 1 \\
  --shard-dir /output/shards
trap - ERR
aws s3 sync /opt/validate50/output s3://{bucket}/{prefix}/cloud/
aws s3 cp "$LOG" s3://{bucket}/{prefix}/cloud/bootstrap.log || true
shutdown -h now
"""
    return script


def _profile_valid(ec2, name: str, *, ami: str | None = None) -> bool:
    image_id = ami or DEFAULT_AMI_X86
    try:
        ec2.run_instances(
            ImageId=image_id,
            InstanceType="c6i.large",
            MinCount=1,
            MaxCount=1,
            DryRun=True,
            IamInstanceProfile={"Name": name},
        )
        return True
    except ClientError as e:
        code = e.response.get("Error", {}).get("Code", "")
        if code == "DryRunOperation":
            return True
        return False


def _fetch_console_output(ec2, instance_id: str, *, timeout_sec: int = 180) -> str:
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        try:
            resp = ec2.get_console_output(InstanceId=instance_id, Latest=True)
            output = resp.get("Output") or ""
            if output:
                return output
        except ClientError:
            pass
        time.sleep(5)
    return ""


def _save_console_output(instance_id: str, output: str) -> Path:
    log_dir = ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / f"validate50_ec2_console_{instance_id}.log"
    path.write_text(output or "(no console output available)", encoding="utf-8")
    return path


def _fetch_bootstrap_failure_log(s3, *, bucket: str, prefix: str, dest: Path) -> bool:
    key = f"{prefix}/cloud/bootstrap_failure.log"
    try:
        s3.download_file(bucket, key, str(dest))
        return True
    except ClientError:
        return False


def _instance_status_summary(ec2, instance_id: str) -> dict:
    try:
        resp = ec2.describe_instance_status(InstanceIds=[instance_id], IncludeAllInstances=True)
        statuses = resp.get("InstanceStatuses", [])
        if not statuses:
            return {"instance_id": instance_id, "status": "unknown"}
        st = statuses[0]
        return {
            "instance_id": instance_id,
            "instance_state": st.get("InstanceState", {}).get("Name"),
            "system_status": st.get("SystemStatus", {}).get("Status"),
            "instance_status": st.get("InstanceStatus", {}).get("Status"),
        }
    except ClientError as e:
        return {
            "instance_id": instance_id,
            "status": "error",
            "error": e.response.get("Error", {}).get("Code", str(e)),
        }


def _report_failure(
    *,
    ec2,
    s3,
    instance_id: str,
    bucket: str,
    prefix: str,
    reason: str,
) -> None:
    console = _fetch_console_output(ec2, instance_id)
    console_path = _save_console_output(instance_id, console)
    bootstrap_path = ROOT / "logs" / f"validate50_ec2_bootstrap_{instance_id}.log"
    bootstrap_fetched = _fetch_bootstrap_failure_log(
        s3, bucket=bucket, prefix=prefix, dest=bootstrap_path
    )
    status = _instance_status_summary(ec2, instance_id)
    excerpt = console[-8000:] if console else ""
    if not excerpt and bootstrap_fetched:
        excerpt = bootstrap_path.read_text(encoding="utf-8", errors="replace")[-8000:]

    failure = {
        "status": "failed",
        "reason": reason,
        "instance_id": instance_id,
        "instance_status": status,
        "console_log": str(console_path),
        "bootstrap_failure_log": str(bootstrap_path) if bootstrap_fetched else None,
        "console_excerpt": excerpt,
    }
    failure_path = ROOT / "logs" / "validate50_ec2_failure.json"
    failure_path.parent.mkdir(parents=True, exist_ok=True)
    failure_path.write_text(json.dumps(failure, indent=2), encoding="utf-8")
    print(json.dumps(failure, indent=2), file=sys.stderr)
    sys.exit(1)


def main() -> None:
    p = argparse.ArgumentParser(description="Launch validate50 EC2 spot worker")
    p.add_argument("--bucket", default=DEFAULT_BUCKET)
    p.add_argument("--prefix", default=DEFAULT_PREFIX)
    p.add_argument("--region", default=DEFAULT_REGION)
    p.add_argument("--instance-type", default=DEFAULT_INSTANCE)
    p.add_argument("--ami", default=None, help="Override AMI")
    p.add_argument("--max-price", type=str, default=None)
    p.add_argument("--manifest-slice", type=str, default=None)
    p.add_argument("--wait", action="store_true")
    p.add_argument("--timeout-sec", type=int, default=7200)
    p.add_argument("--instance-profile", default=DEFAULT_INSTANCE_PROFILE)
    p.add_argument(
        "--require-instance-profile",
        action="store_true",
        help="Fail if instance profile missing or invalid (EC2 gate default)",
    )
    p.add_argument(
        "--use-launcher-credentials",
        action="store_true",
        help="Embed launcher STS creds in user-data when no instance profile (local dev only)",
    )
    p.add_argument(
        "--on-demand",
        action="store_true",
        help="Use on-demand instead of spot (fallback when spot SLR cannot be created)",
    )
    args = p.parse_args()

    ec2 = boto3.client("ec2", region_name=args.region)
    s3 = boto3.client("s3", region_name=args.region)

    ami = args.ami or _resolve_ami(ec2, instance_type=args.instance_type, region=args.region)

    profile = ""
    if args.instance_profile:
        if _profile_valid(ec2, args.instance_profile, ami=ami):
            profile = args.instance_profile
        elif args.require_instance_profile:
            print(
                json.dumps(
                    {
                        "error": "invalid_instance_profile",
                        "instance_profile": args.instance_profile,
                        "message": "PassRole or profile DryRun failed — verify profile exists and launcher has iam:PassRole",
                    }
                ),
                file=sys.stderr,
            )
            sys.exit(2)

    creds: dict[str, str] | None = None
    if not profile:
        if args.require_instance_profile:
            print(
                json.dumps(
                    {
                        "error": "missing_instance_profile",
                        "message": "Set LABELGEN_INSTANCE_PROFILE or pass --instance-profile",
                    }
                ),
                file=sys.stderr,
            )
            sys.exit(2)
        if not args.use_launcher_credentials:
            print(
                json.dumps(
                    {
                        "error": "missing_instance_profile",
                        "message": "Set LABELGEN_INSTANCE_PROFILE or pass --use-launcher-credentials",
                    }
                ),
                file=sys.stderr,
            )
            sys.exit(2)
        creds = _launcher_creds()
        print(json.dumps({"warning": "using_launcher_credentials_in_user_data", "validate50_bootstrap_only": True}))

    ud = base64.b64encode(
        _user_data(
            bucket=args.bucket,
            prefix=args.prefix,
            slice_spec=args.manifest_slice,
            region=args.region,
            image_tag=IMAGE_TAG,
            creds=creds,
        ).encode()
    ).decode()

    spot_opts: dict = {"SpotInstanceType": "one-time", "InstanceInterruptionBehavior": "terminate"}
    if args.max_price:
        spot_opts["MaxPrice"] = args.max_price

    params: dict = {
        "ImageId": ami,
        "InstanceType": args.instance_type,
        "MinCount": 1,
        "MaxCount": 1,
        "UserData": ud,
        "MetadataOptions": {"HttpTokens": "required", "HttpPutResponseHopLimit": 2},
        "TagSpecifications": [
            {
                "ResourceType": "instance",
                "Tags": [
                    {"Key": "Name", "Value": "pokerbot-validate50-ec2-gate"},
                    {"Key": "Project", "Value": "pokerbot-labelgen"},
                ],
            }
        ],
    }
    if not args.on_demand:
        params["InstanceMarketOptions"] = {"MarketType": "spot", "SpotOptions": spot_opts}
    if profile:
        params["IamInstanceProfile"] = {"Name": profile}

    resp = ec2.run_instances(**params)
    instance_id = resp["Instances"][0]["InstanceId"]
    print(
        json.dumps(
            {
                "instance_id": instance_id,
                "ami": ami,
                "region": args.region,
                "instance_type": args.instance_type,
                "instance_profile": profile or None,
                "auth_mode": "instance_profile" if profile else "launcher_credentials_user_data",
                "bucket": args.bucket,
                "prefix": args.prefix,
            },
            indent=2,
        )
    )

    if not args.wait:
        return

    manifest_key = f"{args.prefix}/cloud/manifest.json"
    deadline = time.time() + args.timeout_sec
    while time.time() < deadline:
        try:
            s3.head_object(Bucket=args.bucket, Key=manifest_key)
            print(json.dumps({"status": "complete", "manifest": f"s3://{args.bucket}/{manifest_key}"}))
            return
        except ClientError:
            time.sleep(30)
    _report_failure(
        ec2=ec2,
        s3=s3,
        instance_id=instance_id,
        bucket=args.bucket,
        prefix=args.prefix,
        reason="timeout_waiting_for_cloud_manifest",
    )


if __name__ == "__main__":
    main()
