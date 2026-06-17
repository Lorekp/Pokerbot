#!/usr/bin/env python3
"""Shared EC2 diagnostics for labelgen launchers (console output, failure reports)."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from botocore.exceptions import ClientError

ROOT = Path(__file__).resolve().parents[1]


def fetch_console_output(ec2: Any, instance_id: str, *, timeout_sec: int = 180) -> str:
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


def save_console_output(instance_id: str, output: str, *, prefix: str = "fleet") -> Path:
    log_dir = ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / f"{prefix}_console_{instance_id}.log"
    path.write_text(output or "(no console output available)", encoding="utf-8")
    return path


def instance_status_summary(ec2: Any, instance_id: str) -> dict[str, Any]:
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


def fetch_bootstrap_failure_log(
    s3: Any, *, bucket: str, prefix: str, dest: Path, rel_key: str = "cloud/bootstrap_failure.log"
) -> bool:
    key = f"{prefix}/{rel_key}"
    try:
        s3.download_file(bucket, key, str(dest))
        return True
    except ClientError:
        return False


def report_instance_failure(
    *,
    ec2: Any,
    s3: Any | None,
    instance_id: str,
    bucket: str,
    prefix: str,
    reason: str,
    output_path: Path,
    log_prefix: str = "fleet",
    bootstrap_rel_key: str = "cloud/bootstrap_failure.log",
) -> dict[str, Any]:
    console = fetch_console_output(ec2, instance_id)
    console_path = save_console_output(instance_id, console, prefix=log_prefix)
    bootstrap_path = ROOT / "logs" / f"{log_prefix}_bootstrap_{instance_id}.log"
    bootstrap_fetched = False
    if s3 is not None:
        bootstrap_fetched = fetch_bootstrap_failure_log(
            s3, bucket=bucket, prefix=prefix, dest=bootstrap_path, rel_key=bootstrap_rel_key
        )
    status = instance_status_summary(ec2, instance_id)
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
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(failure, indent=2), encoding="utf-8")
    return failure


def list_running_labelgen_instances(ec2: Any, *, run_tag: str | None = None) -> list[str]:
    filters: list[dict[str, Any]] = [
        {"Name": "tag:Project", "Values": ["pokerbot-labelgen"]},
        {"Name": "instance-state-name", "Values": ["pending", "running", "stopping"]},
    ]
    if run_tag:
        filters.append({"Name": "tag:Run", "Values": [run_tag]})
    resp = ec2.describe_instances(Filters=filters)
    ids: list[str] = []
    for reservation in resp.get("Reservations", []):
        for inst in reservation.get("Instances", []):
            ids.append(inst["InstanceId"])
    return ids
