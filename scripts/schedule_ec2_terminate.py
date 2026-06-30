#!/usr/bin/env python3
"""PC-INDEPENDENT hard-terminate backstop for a launched spot box: an EventBridge Scheduler one-time
schedule that calls ec2:TerminateInstances at now+N hours, executing IN AWS -- so it fires even with the
local PC powered off (unlike _probe_watch.py / poll_*.py, which die when the PC sleeps). This is belt-and-
suspenders ON TOP of the on-box logind `shutdown +floor` + systemd wallguard; use it for nights you turn
the PC off and don't want any chance of an overnight idle-bill.

Auto-creates a minimal scheduler role (trust scheduler.amazonaws.com; ec2:TerminateInstances scoped to the
one instance), then a one-time `at(...)` schedule with ActionAfterCompletion=DELETE so it self-cleans after
firing. Idempotent: reuses the role and updates the schedule if re-run.

  python scripts/schedule_ec2_terminate.py i-0a520f9c0c5aa1c60 --hours 4 --region eu-central-1
"""
from __future__ import annotations

import argparse
import datetime
import json
import time

import boto3
from botocore.exceptions import ClientError


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("instance_id")
    ap.add_argument("--hours", type=float, default=4.0)
    ap.add_argument("--region", default="eu-central-1")
    ap.add_argument("--role-name", default="pokerbot-sched-terminate")
    args = ap.parse_args()

    sess = boto3.Session(region_name=args.region)
    iam = sess.client("iam")
    sched = sess.client("scheduler")
    acct = sess.client("sts").get_caller_identity()["Account"]
    inst_arn = f"arn:aws:ec2:{args.region}:{acct}:instance/{args.instance_id}"

    # 1) minimal role assumable by EventBridge Scheduler, allowed to terminate THIS instance only
    trust = {"Version": "2012-10-17", "Statement": [{"Effect": "Allow",
             "Principal": {"Service": "scheduler.amazonaws.com"}, "Action": "sts:AssumeRole"}]}
    try:
        role_arn = iam.create_role(
            RoleName=args.role_name, AssumeRolePolicyDocument=json.dumps(trust),
            Description="pokerbot one-time EC2 terminate backstop (EventBridge Scheduler)")["Role"]["Arn"]
        log(f"created role {role_arn}")
    except ClientError as e:
        if e.response["Error"]["Code"] == "EntityAlreadyExists":
            role_arn = iam.get_role(RoleName=args.role_name)["Role"]["Arn"]
            log(f"reusing role {role_arn}")
        else:
            raise
    pol = {"Version": "2012-10-17", "Statement": [{"Effect": "Allow",
           "Action": "ec2:TerminateInstances", "Resource": inst_arn}]}
    iam.put_role_policy(RoleName=args.role_name, PolicyName="ec2-terminate-this", PolicyDocument=json.dumps(pol))
    log(f"scoped inline policy: ec2:TerminateInstances -> {inst_arn}")
    time.sleep(12)  # let the new role/policy propagate before the schedule references it

    # 2) one-time schedule at now + hours (UTC)
    fire = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(hours=args.hours)
    expr = "at(" + fire.strftime("%Y-%m-%dT%H:%M:%S") + ")"
    name = f"pokerbot-term-{args.instance_id}"
    target = {"Arn": "arn:aws:scheduler:::aws-sdk:ec2:terminateInstances", "RoleArn": role_arn,
              "Input": json.dumps({"InstanceIds": [args.instance_id]})}
    kw = dict(Name=name, ScheduleExpression=expr, ScheduleExpressionTimezone="UTC",
              FlexibleTimeWindow={"Mode": "OFF"}, Target=target,
              ActionAfterCompletion="DELETE", State="ENABLED")
    try:
        sched.create_schedule(**kw)
        log(f"created schedule {name}")
    except ClientError as e:
        if e.response["Error"]["Code"] == "ConflictException":
            sched.update_schedule(**kw)
            log(f"updated existing schedule {name}")
        else:
            raise

    local = fire.astimezone()
    log(f"DONE -- AWS-side backstop ARMED. {args.instance_id} will be TERMINATED at {expr} UTC "
        f"= {local.strftime('%Y-%m-%d %H:%M %Z')} local, regardless of PC state.")
    log("(If the on-box guards self-terminate first, this schedule hits an already-gone instance = harmless "
        "no-op, then DELETE-after-completion cleans it up.)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
