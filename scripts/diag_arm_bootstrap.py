#!/usr/bin/env python3
"""Diagnostic: launch a tiny arm64 instance, run the EXACT failing bootstrap steps, and capture
the real error via the EC2 SERIAL CONSOLE (get-console-output) — which does NOT depend on aws-cli
or S3 working, the channel we were blind to for 13 attempts.

The box does NOT terminate on finish (shutdown behavior=stop) so console output survives and we can
inspect. It writes clear ===MARKER=== lines so we can see exactly which step crashes.

Run:  python scripts/diag_arm_bootstrap.py launch    # prints instance id
      python scripts/diag_arm_bootstrap.py console <iid>   # dump serial console
      python scripts/diag_arm_bootstrap.py kill <iid>
"""
from __future__ import annotations

import base64
import sys

import boto3

REGION = "eu-central-1"
BUCKET = "video-processing-6789"
PREFIX = "pokerbot/prod6000/sd_cfr_hu"
PROFILE = "pokerbot-labelgen-worker"

UD = f"""#!/bin/bash
exec > /var/log/diag.log 2>&1
set -x
echo "===DIAG START==="
echo "===STEP cloud-init-version==="
cloud-init --version || true
echo "===STEP whoami/curl-imds-token==="
TOKEN=$(curl -sS -X PUT "http://169.254.169.254/latest/api/token" -H "X-aws-ec2-metadata-token-ttl-seconds: 120")
echo "token-len=${{#TOKEN}}"
echo "===STEP imds-role==="
curl -sS -H "X-aws-ec2-metadata-token: $TOKEN" http://169.254.169.254/latest/meta-data/iam/security-credentials/ ; echo
echo "===STEP apt-update==="
export DEBIAN_FRONTEND=noninteractive
apt-get update -y
echo "apt-update-rc=$?"
echo "===STEP apt-install-awscli==="
apt-get install -y awscli
echo "apt-awscli-rc=$?"
echo "===STEP aws-version==="
aws --version
echo "aws-version-rc=$?"
echo "===STEP aws-s3-ls==="
aws s3 ls s3://{BUCKET}/{PREFIX}/wheels_py310_arm64/ --region {REGION}
echo "aws-s3-ls-rc=$?"
echo "===STEP curl-presigned-test (no aws-cli) -- skipped, ls is the real test==="
echo "===DIAG END==="
# Mirror the diag log to the serial console so get-console-output captures it.
cat /var/log/diag.log > /dev/console
"""


def resolve_ami(ec2, arch="arm64"):
    name_arch = "arm64" if arch == "arm64" else "amd64"
    resp = ec2.describe_images(
        Owners=["099720109477"],
        Filters=[
            {"Name": "name", "Values": [f"ubuntu/images/hvm-ssd/ubuntu-jammy-22.04-{name_arch}-server-*"]},
            {"Name": "state", "Values": ["available"]},
            {"Name": "architecture", "Values": [arch]},
        ],
    )
    imgs = sorted(resp["Images"], key=lambda i: i["CreationDate"], reverse=True)
    return imgs[0]["ImageId"]


def launch():
    ec2 = boto3.client("ec2", region_name=REGION)
    ami = resolve_ami(ec2, "arm64")
    print(f"AMI {ami}", flush=True)
    resp = ec2.run_instances(
        ImageId=ami,
        InstanceType="t4g.medium",
        MinCount=1,
        MaxCount=1,
        UserData=base64.b64encode(UD.encode()).decode(),
        IamInstanceProfile={"Name": PROFILE},
        MetadataOptions={"HttpTokens": "required", "HttpPutResponseHopLimit": 2},
        InstanceInitiatedShutdownBehavior="stop",  # keep it alive for inspection
        TagSpecifications=[{"ResourceType": "instance",
                            "Tags": [{"Key": "Name", "Value": "diag-arm-bootstrap"}]}],
    )
    iid = resp["Instances"][0]["InstanceId"]
    print(f"LAUNCHED {iid}", flush=True)


def console(iid):
    ec2 = boto3.client("ec2", region_name=REGION)
    out = ec2.get_console_output(InstanceId=iid, Latest=True)
    print(out.get("Output", "<no console output yet>"))


def kill(iid):
    ec2 = boto3.client("ec2", region_name=REGION)
    ec2.terminate_instances(InstanceIds=[iid])
    print(f"terminating {iid}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "launch"
    if cmd == "launch":
        launch()
    elif cmd == "console":
        console(sys.argv[2])
    elif cmd == "kill":
        kill(sys.argv[2])
