#!/usr/bin/env python3
"""Cost-safe watchdog/poller for the cpu128 profiling probe (or any one-shot EC2 batch box).

WHY THIS EXISTS: a probe launch on 2026-06-28 idle-billed a c6i.32xlarge for 3h20m because (1) a broken
apt awscli v1 (pip boto3 bumped botocore -> KeyError opsworkscm) killed the bootstrap at t=73s, and (2)
the on-box `nohup &` wall-guard was in cloud-init's systemd cgroup and got reaped at t=80s, so nothing
self-terminated. The launcher now fixes both (AWS CLI v2 + systemd-run transient-unit watchdog). THIS is
the external belt-and-suspenders: it polls 5 min after startup, then every 15 min, greps the serial
console for the known failure signatures, and TERMINATES the instance itself the instant it sees a busted
bootstrap — so a dead box can never bleed money again even if every on-box guard fails.

Exit codes: 0 = PROBE_DONE (success, box left keep-alive for manual pull+terminate),
            2 = failure detected (instance TERMINATED by this poller),
            3 = instance already gone (self-terminated / reclaimed / wall-guard fired),
            4 = hard timeout reached (instance TERMINATED by this poller).

Usage:
  python scripts/_probe_watch.py --instance-id i-xxxx --region eu-central-1 \
      --s3-prefix s3://bucket/pokerbot/.../probe_<ts>/probe/ \
      [--first-delay 300] [--interval 900] [--max-minutes 115] [--no-auto-terminate]
"""
import argparse
import re
import sys
import time
from datetime import datetime, timezone

import boto3

# --- console-output substrings that PROVE a failed/aborted run (terminate immediately) ---
FAIL_SIGS = [
    "opsworkscm",                 # the broken-awscli signature
    "FROZEN_DOWNLOAD_FAIL",
    "BUILD_FAIL",
    "CODE_DOWNLOAD_FAIL",
    "Traceback (most recent call last)",
]
# `onbox exit N` with N != 0 is the on-box script aborting; matched separately so exit 0 is NOT a fail.
ONBOX_EXIT_RE = re.compile(r"onbox exit ([0-9]+)")
KEEPALIVE_SIG = "BOOTSTRAP_DONE_KEEP_ALIVE"
# S3 keys that mean success / whole-run abort. NOTE: deliberately NOT per-shape or perf markers —
# PERF_BROKEN, ${SHAPE}_PERF_DEGRADED, ${SHAPE}_FAILED are NON-FATAL (the on-box script degrades and
# continues, and may still reach PROBE_DONE), so treating them as fail would kill a healthy run.
DONE_KEYS = ("PROBE_DONE",)
SWEEP_KEYS = ("WALLGUARD_SWEEP", "/SWEEP/")  # box already swept+shut itself => only informational


def now() -> str:
    return datetime.now(timezone.utc).strftime("%H:%M:%SZ")


def get_console(ec2, iid: str) -> str:
    try:
        r = ec2.get_console_output(InstanceId=iid, Latest=True)
        return r.get("Output", "") or ""
    except Exception as e:  # noqa: BLE001
        return f"(console unavailable: {e!r})"


def instance_state(ec2, iid: str) -> str:
    try:
        r = ec2.describe_instances(InstanceIds=[iid])
        return r["Reservations"][0]["Instances"][0]["State"]["Name"]
    except Exception as e:  # noqa: BLE001
        # NotFound => already terminated long ago
        if "NotFound" in repr(e) or "InvalidInstanceID" in repr(e):
            return "gone"
        return f"err:{e!r}"


def s3_list(s3, bucket: str, prefix: str):
    keys = []
    tok = None
    while True:
        kw = dict(Bucket=bucket, Prefix=prefix)
        if tok:
            kw["ContinuationToken"] = tok
        r = s3.list_objects_v2(**kw)
        for o in r.get("Contents", []):
            keys.append(o["Key"])
        tok = r.get("NextContinuationToken")
        if not (r.get("IsTruncated") and tok):  # break if not truncated OR token missing (no infinite loop)
            break
    return keys


def terminate(ec2, iid: str, why: str):
    try:
        ec2.terminate_instances(InstanceIds=[iid])
        print(f"[{now()}] >>> TERMINATED {iid} ({why})", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"[{now()}] !!! terminate FAILED for {iid}: {e!r} — TERMINATE MANUALLY", flush=True)


def classify(console: str, keys: list[str]):
    """Return (verdict, detail) where verdict in {done, fail, running}."""
    base = [k.rsplit("/", 1)[-1] for k in keys]
    if any(any(d in b for d in DONE_KEYS) for b in base):
        return "done", "PROBE_DONE present in S3"
    # hard console failure signatures
    for sig in FAIL_SIGS:
        if sig in console:
            return "fail", f"console signature: {sig!r}"
    m = ONBOX_EXIT_RE.search(console)
    if m and m.group(1) != "0":
        return "fail", f"onbox exit {m.group(1)}"
    # NOTE: we deliberately do NOT treat 'BOOTSTRAP_DONE_KEEP_ALIVE without PROBE_DONE' as a fail. A genuine
    # failure always shows a FAIL_SIG above or a non-zero `onbox exit` (caught above); `onbox exit 0` means
    # the run completed, and if its final PROBE_DONE upload merely had a transient blip the box still
    # self-terminates at the wall-guard floor — terminating it early here would mis-kill a real success.
    # partial sweep markers => the box already swept+shut (failure/hang handled on-box)
    if any(any(s in k for s in SWEEP_KEYS) for k in keys):
        return "fail", "sweep/fail marker in S3 (on-box guard already fired)"
    return "running", ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance-id", required=True)
    ap.add_argument("--region", required=True)
    ap.add_argument("--s3-prefix", required=True, help="s3://bucket/.../probe/ (artifacts + markers land here)")
    ap.add_argument("--first-delay", type=int, default=300, help="seconds before the FIRST poll (default 5 min)")
    ap.add_argument("--interval", type=int, default=900, help="seconds between polls after the first (default 15 min)")
    ap.add_argument("--max-minutes", type=int, default=115, help="hard ceiling; terminate the box if exceeded")
    ap.add_argument("--no-auto-terminate", action="store_true",
                    help="report failures but do NOT terminate (default: terminate on any failure/timeout)")
    a = ap.parse_args()

    assert a.s3_prefix.startswith("s3://")
    bucket, prefix = a.s3_prefix[5:].split("/", 1)
    ec2 = boto3.client("ec2", region_name=a.region)
    s3 = boto3.client("s3", region_name=a.region)
    auto = not a.no_auto_terminate
    t0 = time.time()
    deadline = t0 + a.max_minutes * 60

    print(f"[{now()}] WATCH {a.instance_id} region={a.region}", flush=True)
    print(f"[{now()}] prefix={a.s3_prefix}", flush=True)
    print(f"[{now()}] first poll in {a.first_delay}s, then every {a.interval}s, hard-stop {a.max_minutes}min, "
          f"auto-terminate={'ON' if auto else 'OFF'}", flush=True)

    # first delay (let the box boot+build); break it into 30s chunks so a very early self-terminate is caught
    waited = 0
    while waited < a.first_delay:
        time.sleep(min(30, a.first_delay - waited))
        waited += 30
        try:
            st = instance_state(ec2, a.instance_id)
            if st in ("terminated", "shutting-down", "stopped", "stopping", "gone"):
                keys = s3_list(s3, bucket, prefix)
                v, d = classify("", keys)
                print(f"[{now()}] instance {st} during warmup — verdict={v} ({d})", flush=True)
                sys.exit(0 if v == "done" else 3)
        except Exception as e:  # noqa: BLE001
            print(f"[{now()}] warmup transient error: {e!r} — continue", flush=True)

    poll = 0
    while True:
        poll += 1
        # The whole poll body is wrapped: a transient AWS error (throttle/5xx/creds-refresh) must degrade to
        # "retry next interval", NEVER crash the watcher (a dead watcher = an unwatched box). sys.exit() on a
        # terminal verdict raises SystemExit (a BaseException), so it propagates PAST `except Exception`.
        try:
            st = instance_state(ec2, a.instance_id)
            keys = s3_list(s3, bucket, prefix)
            console = get_console(ec2, a.instance_id) if st not in ("gone", "terminated") else ""
            v, d = classify(console, keys)
            mins = (time.time() - t0) / 60
            print(f"[{now()}] poll {poll} +{mins:.0f}min state={st} artifacts={len(keys)} verdict={v} {d}", flush=True)

            if v == "done":
                print(f"[{now()}] SUCCESS: PROBE_DONE. Box left keep-alive — pull artifacts then terminate manually.", flush=True)
                sys.exit(0)
            if st in ("terminated", "stopped", "gone"):
                print(f"[{now()}] instance {st} — verdict={v} ({d}). Nothing to terminate.", flush=True)
                sys.exit(0 if v == "done" else 3)
            if v == "fail":
                print(f"[{now()}] FAILURE DETECTED: {d}", flush=True)
                if auto:
                    terminate(ec2, a.instance_id, d)
                else:
                    print(f"[{now()}] auto-terminate OFF — TERMINATE {a.instance_id} MANUALLY.", flush=True)
                sys.exit(2)
        except Exception as e:  # noqa: BLE001
            print(f"[{now()}] poll {poll} transient error: {e!r} — retry next interval", flush=True)

        # Deadline runs even if every poll errors, so a persistently-failing API still caps cost.
        if time.time() > deadline:
            print(f"[{now()}] HARD TIMEOUT {a.max_minutes}min — terminating to stop the bleed.", flush=True)
            if auto:
                terminate(ec2, a.instance_id, f"hard timeout {a.max_minutes}min")
            sys.exit(4)

        # sleep the interval in 30s chunks so a self-terminate mid-interval is caught fast
        slept = 0
        while slept < a.interval and time.time() < deadline:
            time.sleep(min(30, a.interval - slept))
            slept += 30


if __name__ == "__main__":
    main()
