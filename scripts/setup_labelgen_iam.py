#!/usr/bin/env python3
"""One-time labelgen IAM setup — run with admin/root AWS credentials."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import boto3
from botocore.exceptions import ClientError

ROOT = Path(__file__).resolve().parents[1]
LAUNCHER_USER = "labelgen-worker"
WORKER_ROLE = "pokerbot-labelgen-worker"
WORKER_PROFILE = "pokerbot-labelgen-worker"


def _session(*, profile: str | None) -> boto3.session.Session:
    if profile:
        return boto3.Session(profile_name=profile)
    return boto3.Session()


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def setup(
    *,
    session: boto3.session.Session,
    launcher_user: str,
    worker_role: str,
    worker_profile: str,
) -> dict:
    iam = session.client("iam")
    results: dict[str, str] = {}

    launcher_policy = _load_json(ROOT / "infra/labelgen/iam_policy_labelgen_launcher.json")
    worker_policy = _load_json(ROOT / "infra/labelgen/iam_policy_labelgen_worker_role.json")
    assume = json.dumps(
        {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Principal": {"Service": "ec2.amazonaws.com"},
                    "Action": "sts:AssumeRole",
                }
            ],
        }
    )

    try:
        iam.put_user_policy(
            UserName=launcher_user,
            PolicyName="labelgen-launcher",
            PolicyDocument=json.dumps(launcher_policy),
        )
        results["launcher_policy"] = "updated"
    except ClientError as e:
        results["launcher_policy"] = e.response["Error"]["Code"]

    try:
        iam.create_role(
            RoleName=worker_role,
            AssumeRolePolicyDocument=assume,
            Description="Pokerbot labelgen EC2 worker",
        )
        results["create_role"] = "created"
    except ClientError as e:
        results["create_role"] = (
            "exists" if e.response["Error"]["Code"] == "EntityAlreadyExists" else e.response["Error"]["Code"]
        )

    try:
        iam.put_role_policy(
            RoleName=worker_role,
            PolicyName="labelgen-worker-s3",
            PolicyDocument=json.dumps(worker_policy),
        )
        results["worker_policy"] = "updated"
    except ClientError as e:
        results["worker_policy"] = e.response["Error"]["Code"]

    try:
        iam.create_instance_profile(InstanceProfileName=worker_profile)
        results["create_profile"] = "created"
    except ClientError as e:
        results["create_profile"] = (
            "exists" if e.response["Error"]["Code"] == "EntityAlreadyExists" else e.response["Error"]["Code"]
        )

    try:
        iam.add_role_to_instance_profile(InstanceProfileName=worker_profile, RoleName=worker_role)
        results["add_role_to_profile"] = "ok"
    except ClientError as e:
        code = e.response["Error"]["Code"]
        results["add_role_to_profile"] = "already_attached" if code == "LimitExceeded" else code

    try:
        iam.create_service_linked_role(AWSServiceName="spot.amazonaws.com")
        results["spot_slr"] = "created"
    except ClientError as e:
        code = e.response["Error"]["Code"]
        results["spot_slr"] = "exists" if code in ("InvalidInput", "EntityAlreadyExists") else code

    return results


def _print_access_denied_help(*, caller_arn: str) -> None:
    print(
        f"""
ERROR: AccessDenied — caller is {caller_arn}

setup_labelgen_iam.py must run with ADMIN credentials (root user or IAM admin),
NOT labelgen-worker. labelgen-worker cannot modify its own IAM policy.

Option A — separate admin profile in ~/.aws/credentials:
  [admin]
  aws_access_key_id = ...
  aws_secret_access_key = ...

  python scripts/setup_labelgen_iam.py --profile admin --verify

Option B — temporary env vars (CMD):
  set AWS_ACCESS_KEY_ID=<admin-key>
  set AWS_SECRET_ACCESS_KEY=<admin-secret>
  set AWS_DEFAULT_REGION=eu-central-1
  python scripts/setup_labelgen_iam.py --verify

Option C — AWS Console (see infra/labelgen/IAM.md section "Console setup")

After admin setup, switch BACK to labelgen-worker and verify:
  CMD:  set LABELGEN_INSTANCE_PROFILE=pokerbot-labelgen-worker
  CMD:  python scripts\\verify_labelgen_iam.py --strict --instance-profile pokerbot-labelgen-worker
""",
        file=sys.stderr,
    )


def main() -> None:
    p = argparse.ArgumentParser(description="One-time labelgen IAM setup (admin credentials required)")
    p.add_argument("--profile", default=None, help="AWS profile name with admin IAM permissions")
    p.add_argument("--launcher-user", default=LAUNCHER_USER)
    p.add_argument("--worker-role", default=WORKER_ROLE)
    p.add_argument("--worker-profile", default=WORKER_PROFILE)
    p.add_argument("--verify", action="store_true", help="Run verify_labelgen_iam.py --strict after setup")
    p.add_argument("--region", default="eu-central-1")
    args = p.parse_args()

    session = _session(profile=args.profile)
    caller = session.client("sts").get_caller_identity()
    caller_arn = caller.get("Arn", "")
    print(json.dumps({"caller": caller_arn, "profile": args.profile}, indent=2))

    if "user/labelgen-worker" in caller_arn:
        _print_access_denied_help(caller_arn=caller_arn)
        sys.exit(2)

    results = setup(
        session=session,
        launcher_user=args.launcher_user,
        worker_role=args.worker_role,
        worker_profile=args.worker_profile,
    )
    print(json.dumps(results, indent=2))

    failed = [k for k, v in results.items() if v not in ("updated", "created", "exists", "ok", "already_attached")]
    if failed:
        if "AccessDenied" in results.values():
            _print_access_denied_help(caller_arn=caller_arn)
        else:
            print(f"Setup incomplete — failed: {failed}", file=sys.stderr)
        sys.exit(1)

    print("Setup complete. Re-run verify as labelgen-worker (default credentials):", file=sys.stderr)
    print(
        "  set LABELGEN_INSTANCE_PROFILE=pokerbot-labelgen-worker && "
        "python scripts\\verify_labelgen_iam.py --strict --instance-profile pokerbot-labelgen-worker",
        file=sys.stderr,
    )

    if args.verify:
        import os

        # Post-setup verify uses launcher creds (default profile), not admin.
        os.environ.pop("AWS_ACCESS_KEY_ID", None)
        os.environ.pop("AWS_SECRET_ACCESS_KEY", None)
        os.environ.pop("AWS_SESSION_TOKEN", None)
        os.environ["LABELGEN_INSTANCE_PROFILE"] = args.worker_profile
        os.environ["AWS_DEFAULT_REGION"] = args.region
        r = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts/verify_labelgen_iam.py"),
                "--strict",
                "--region",
                args.region,
                "--instance-profile",
                args.worker_profile,
            ],
            cwd=ROOT,
        )
        sys.exit(r.returncode)


if __name__ == "__main__":
    main()
