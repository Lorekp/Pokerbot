#!/usr/bin/env python3
"""Verify IAM permissions required for prod6000 EC2 spot label fleet."""

from __future__ import annotations

import argparse
import json
import os
import sys

import boto3
from botocore.exceptions import ClientError

DEFAULT_INSTANCE_PROFILE = os.environ.get("LABELGEN_INSTANCE_PROFILE", "")

# Acceptable errors when probing permissions with fake instance IDs.
_CONSOLE_OUTPUT_OK_ERRORS = {
    "InvalidInstanceID.NotFound",
    "InvalidInstanceID.Malformed",
    "IncorrectInstanceState",
}
_SPOT_DRYRUN_OK = {"DryRunOperation"}


def _check(name: str, fn) -> dict:
    try:
        fn()
        return {"name": name, "ok": True}
    except ClientError as e:
        code = e.response.get("Error", {}).get("Code", "ClientError")
        msg = e.response.get("Error", {}).get("Message", str(e))
        return {"name": name, "ok": False, "error": code, "message": msg}
    except Exception as e:
        return {"name": name, "ok": False, "error": type(e).__name__, "message": str(e)}


def _mark_dryrun_ok(check: dict, *, ok_codes: set[str]) -> dict:
    if not check["ok"] and check.get("error") in ok_codes:
        check = {**check, "ok": True, "note": f"{check['error']} — permission granted"}
    return check


def main() -> None:
    p = argparse.ArgumentParser(description="Verify labelgen IAM permissions")
    p.add_argument("--bucket", default="video-processing-6789")
    p.add_argument("--prefix", default="pokerbot/prod6000/_iam_probe")
    p.add_argument("--region", default=boto3.session.Session().region_name or "eu-central-1")
    p.add_argument(
        "--instance-profile",
        default=DEFAULT_INSTANCE_PROFILE,
        help="Worker instance profile to verify exists (env: LABELGEN_INSTANCE_PROFILE)",
    )
    p.add_argument(
        "--strict",
        action="store_true",
        help="Require all checks including diagnostics and instance profile (EC2 gate prerequisite)",
    )
    args = p.parse_args()

    ec2 = boto3.client("ec2", region_name=args.region)
    s3 = boto3.client("s3", region_name=args.region)
    sts = boto3.client("sts")
    iam = boto3.client("iam")

    identity = sts.get_caller_identity()
    probe_key = f"{args.prefix}/probe.txt"
    probe_ami = os.environ.get("LABELGEN_AMI_X86", "ami-0faab6bdbac9486fb")

    checks = [
        _check(
            "ec2:DescribeSpotPriceHistory",
            lambda: ec2.describe_spot_price_history(
                InstanceTypes=["c6i.2xlarge"],
                ProductDescriptions=["Linux/UNIX"],
                MaxResults=5,
            ),
        ),
        _check(
            "ec2:DescribeImages",
            lambda: ec2.describe_images(ImageIds=[probe_ami]),
        ),
        _check("ec2:DescribeInstances", lambda: ec2.describe_instances(MaxResults=5)),
        _check(
            "ec2:DescribeInstanceStatus",
            lambda: ec2.describe_instance_status(MaxResults=5),
        ),
        _check(
            "ec2:GetConsoleOutput",
            lambda: ec2.get_console_output(InstanceId="i-00000000000000000", Latest=True),
        ),
        _check(
            "ec2:RunInstances",
            lambda: ec2.run_instances(
                ImageId=probe_ami,
                InstanceType="c6i.2xlarge",
                MinCount=1,
                MaxCount=1,
                DryRun=True,
            ),
        ),
        _check(
            "ec2:RunInstances(spot)",
            lambda: ec2.run_instances(
                ImageId=probe_ami,
                InstanceType="c6i.2xlarge",
                MinCount=1,
                MaxCount=1,
                DryRun=True,
                InstanceMarketOptions={
                    "MarketType": "spot",
                    "SpotOptions": {
                        "SpotInstanceType": "one-time",
                        "InstanceInterruptionBehavior": "terminate",
                    },
                },
            ),
        ),
        _check(
            "s3:PutObject",
            lambda: s3.put_object(Bucket=args.bucket, Key=probe_key, Body=b"probe"),
        ),
        _check(
            "s3:GetObject",
            lambda: s3.get_object(Bucket=args.bucket, Key=probe_key),
        ),
        _check("s3:DeleteObject", lambda: s3.delete_object(Bucket=args.bucket, Key=probe_key)),
        _check(
            "s3:ListBucket",
            lambda: s3.list_objects_v2(Bucket=args.bucket, Prefix=args.prefix.rsplit("/", 1)[0], MaxKeys=1),
        ),
    ]

    if args.instance_profile:
        checks.append(
            _check(
                "iam:GetInstanceProfile",
                lambda: iam.get_instance_profile(InstanceProfileName=args.instance_profile),
            )
        )
        checks.append(
            _check(
                "ec2:RunInstances(with_profile)",
                lambda: ec2.run_instances(
                    ImageId=probe_ami,
                    InstanceType="c6i.2xlarge",
                    MinCount=1,
                    MaxCount=1,
                    DryRun=True,
                    IamInstanceProfile={"Name": args.instance_profile},
                ),
            )
        )
    elif args.strict:
        checks.append(
            {
                "name": "iam:GetInstanceProfile",
                "ok": False,
                "error": "MissingInstanceProfile",
                "message": "Set LABELGEN_INSTANCE_PROFILE or pass --instance-profile",
            }
        )

    for i, c in enumerate(checks):
        if c["name"] == "ec2:GetConsoleOutput" and not c["ok"]:
            if c.get("error") in _CONSOLE_OUTPUT_OK_ERRORS:
                checks[i] = {**c, "ok": True, "note": f"{c['error']} — permission granted"}
        elif c["name"] in {"ec2:RunInstances", "ec2:RunInstances(spot)", "ec2:RunInstances(with_profile)"}:
            checks[i] = _mark_dryrun_ok(c, ok_codes=_SPOT_DRYRUN_OK)

    core_names = {"s3:PutObject", "ec2:RunInstances", "s3:ListBucket"}
    diagnostic_names = {
        "ec2:GetConsoleOutput",
        "ec2:DescribeInstanceStatus",
        "ec2:RunInstances(spot)",
        "iam:GetInstanceProfile",
        "ec2:RunInstances(with_profile)",
    }
    core_ok = all(c["ok"] for c in checks if c["name"] in core_names)
    diagnostics_ok = all(c["ok"] for c in checks if c["name"] in diagnostic_names)
    all_ok = all(c["ok"] for c in checks)

    report = {
        "identity": identity.get("Arn"),
        "account": identity.get("Account"),
        "region": args.region,
        "bucket": args.bucket,
        "instance_profile": args.instance_profile or None,
        "principal": "launcher",
        "strict": args.strict,
        "worker_auth": "instance_role_via_IamInstanceProfile (required for EC2 gate)",
        "all_ok": all_ok,
        "core_ok": core_ok,
        "diagnostics_ok": diagnostics_ok,
        "checks": checks,
    }
    print(json.dumps(report, indent=2))

    if args.strict:
        sys.exit(0 if all_ok else 1)
    sys.exit(0 if core_ok else 1)


if __name__ == "__main__":
    main()
