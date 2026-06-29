#!/usr/bin/env python3
"""Launch ONE spot box that measures the DCFR+pruning -vs- CFR+ EXPLOITABILITY-vs-COMPUTE curve (the
algorithmic convergence-speed lever) + the long-owed work_steal 32/64/128-core scaling number.

WHY a big box + WHY cloud: the curve needs ~50k CFR+ iterations per arm at a core-saturating n_deals to
reach the plateau where DCFR's early-convergence gap (and any final-quality regression) is visible; and
the work_steal load-imbalance payoff only appears at 32-64 cores (local 12t was +9-15%). Training is on
the box; LBR is scored LOCALLY for free (native made-hand Lbr6Harness, JAX root-dealer) by
scripts/score_dcfr_curve_lbr.py over every streamed checkpoint.

Made-hand abstraction only (no 152MB EMD blob): the blueprint must be abstraction-matched to the made-hand
native LBR. ONE binary, plain --features python -- work_steal AND dcfr are RUNTIME toggles, so this is the
production-buildable path, not a special feature build.

Cost-safety is a faithful clone of the proven launch_hogwild_throughput_spot.py / launch_cpu128_probe_spot.py
template: a logind `shutdown -h +N` ABSOLUTE floor (immune to cloud-init cgroup teardown) + a
`pokerbot-dcfrguard` systemd-run transient unit (NOT oneshot) + InstanceInitiatedShutdownBehavior=terminate
+ SELF-TERMINATE on finish (--no-keep-alive default) + an auto-wired _probe_watch.py poller. Checkpoints
STREAM to S3 per-milestone, so a hang/timeout/spot-kill loses at most the in-flight milestone, never the curve.
"""
from __future__ import annotations

import argparse
import base64
import io
import os
import subprocess
import sys
import tarfile
import time
from pathlib import Path

import boto3

from launch_native_cfr_training_spot import (  # noqa: E402
    DEFAULT_BUCKET, DEFAULT_PREFIX, DEFAULT_PROFILE, DEFAULT_REGION, ROOT,
    _launch_with_fallback, _spotpick_chain, _tar_filter, resolve_ubuntu_ami,
)

# build(~600s) + work_steal probe(~1200s) + curve train(~7200s) = ~9000s work; a 10800s wall-guard nests
# outside all of it (the box self-terminates on finish well before this).
WALL_GUARD_SECS = 10800
TRAIN_TIMEOUT = 7200

_INCLUDES = [
    ("tools/battery_native", "tools/battery_native"),
    ("scripts/dcfr_curve_onbox.sh", "scripts/dcfr_curve_onbox.sh"),
    ("scripts/drive_dcfr_curve_train.py", "scripts/drive_dcfr_curve_train.py"),
    ("scripts/dcfr_ws_probe.py", "scripts/dcfr_ws_probe.py"),
]


def tar_include_list() -> list[str]:
    return [arc for _s, arc in _INCLUDES]


def build_and_upload_code(s3, bucket: str, prefix: str) -> str:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for src, arc in _INCLUDES:
            p = ROOT / src
            if p.exists():
                tar.add(p, arcname=arc, filter=_tar_filter)
            else:
                print(f"WARN: tar include missing, skipped: {src}", flush=True)
    buf.seek(0)
    key = f"{prefix}/code/dcfr_curve_code.tar.gz"
    s3.put_object(Bucket=bucket, Key=key, Body=buf.read())
    print(f"Uploaded code tarball -> s3://{bucket}/{key} ({buf.tell()} bytes)", flush=True)
    return key


def user_data(*, bucket, prefix, region, code_key, run_subdir, num_buckets, n_deals, milestones,
              prune_threshold, explore_frac, probe_threads_list, probe_n_deals, warm_iters, window_iters,
              train_rayon, train_timeout, keep_alive, wall_guard_secs,
              awscli_zip="awscli-exe-linux-x86_64.zip") -> str:
    run_prefix = f"{prefix}/{run_subdir}"
    result_prefix = f"{run_prefix}/dcfrcurve"
    final = "echo BOOTSTRAP_DONE_KEEP_ALIVE" if keep_alive else "shutdown -h now"
    floor_min = wall_guard_secs // 60 + 15
    return f"""#!/bin/bash
set -uxo pipefail
export HOME="${{HOME:-/root}}"
export DEBIAN_FRONTEND=noninteractive AWS_DEFAULT_REGION={region}
mkdir -p /opt/pokerbot /var/log/dcfrcurve

# ===== STEP 0 -- DEAD-MAN COST CAP, ARMED FIRST (before apt/aws) =====
/sbin/shutdown -h +{floor_min} "pokerbot dcfr-curve wall-guard absolute cost cap" || true
systemd-run --unit=pokerbot-dcfrguard --collect -p TimeoutStartSec=infinity --no-block /bin/bash -c 'sleep {wall_guard_secs} ; sync ; timeout 180 aws s3 cp --recursive /var/log/dcfrcurve "s3://{bucket}/{result_prefix}/WALLGUARD/" --region {region} 2>/dev/null || true ; /sbin/shutdown -h now' || true
echo "WALL_GUARD armed: shutdown +{floor_min}min floor + pokerbot-dcfrguard unit @ {wall_guard_secs}s"

apt-get update -qq
apt-get install -y -qq python3-pip python3-dev gzip curl unzip build-essential

curl -fsSL "https://awscli.amazonaws.com/{awscli_zip}" -o /tmp/awscliv2.zip
unzip -q /tmp/awscliv2.zip -d /tmp
/tmp/aws/install --bin-dir /usr/local/bin --install-dir /usr/local/aws-cli --update
hash -r
aws --version

curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal
export PATH="$HOME/.cargo/bin:$PATH"
rustc --version

aws s3 cp s3://{bucket}/{code_key} /tmp/code.tar.gz --region {region}
tar -xzf /tmp/code.tar.gz -C /opt/pokerbot
cd /opt/pokerbot

export BUCKET="{bucket}"
export REGION="{region}"
export RUN_PREFIX="{run_prefix}"
export RESULT_PREFIX="{result_prefix}"
export DCFR_NUM_BUCKETS="{num_buckets}"
export DCFR_N_DEALS="{n_deals}"
export DCFR_MILESTONES="{milestones}"
export DCFR_PRUNE_THRESHOLD="{prune_threshold}"
export DCFR_EXPLORE_FRAC="{explore_frac}"
export PROBE_THREADS_LIST="{probe_threads_list}"
export PROBE_N_DEALS="{probe_n_deals}"
export WARM_ITERS="{warm_iters}"
export WINDOW_ITERS="{window_iters}"
export TRAIN_RAYON="{train_rayon}"
export TRAIN_TIMEOUT="{train_timeout}"
export PYTHONPATH=/opt/pokerbot
export RUST_BACKTRACE=1

bash /opt/pokerbot/scripts/dcfr_curve_onbox.sh
echo "onbox exit $?"
{final}
"""


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--instance-type", default="c6i.32xlarge",
                   help="128-vCPU Ice Lake (matches the production training box).")
    p.add_argument("--arch", default="x86_64", choices=["x86_64", "arm64"])
    p.add_argument("--num-buckets", type=int, default=200,
                   help="made-hand bucket count (must match the local LBR's POKERBOT_POSTFLOP_NUM_BUCKETS)")
    p.add_argument("--n-deals", type=int, default=1024, help="core-saturating MC batch (same for both arms)")
    p.add_argument("--milestones", default="1000,2000,4000,8000,16000,32000",
                   help="iteration checkpoints (x n_deals deals) where each arm's blueprint is snapshotted "
                        "(made-hand K=200 CFR+ plateaus ~12k iters, so 32k shows the plateau w/ margin)")
    p.add_argument("--prune-threshold", type=float, default=-300000.0,
                   help="DCFR Regret-Based-Pruning floor (bb regret units); ignored by the cfrplus arm")
    p.add_argument("--explore-frac", type=float, default=0.05,
                   help="DCFR fraction of iters that explore all actions (no pruning); Pluribus ~0.05")
    p.add_argument("--probe-threads-list", default="16,32,64,128",
                   help="RAYON_NUM_THREADS values for the work_steal scaling probe")
    p.add_argument("--probe-n-deals", type=int, default=1024)
    p.add_argument("--warm-iters", type=int, default=2000, help="probe fixed warm state (same node-growth)")
    p.add_argument("--window-iters", type=int, default=1500, help="probe timed window (fixed iters)")
    p.add_argument("--train-rayon", type=int, default=64,
                   help="RAYON_NUM_THREADS for the curve training (64 = physical cores; SMT-negative above)")
    p.add_argument("--train-timeout", type=int, default=TRAIN_TIMEOUT)
    p.add_argument("--run-subdir", default=None)
    p.add_argument("--bucket", default=DEFAULT_BUCKET)
    p.add_argument("--prefix", default=DEFAULT_PREFIX)
    p.add_argument("--region", default=DEFAULT_REGION)
    p.add_argument("--instance-profile", default=DEFAULT_PROFILE)
    p.add_argument("--on-demand", dest="spot", action="store_false", default=True,
                   help="use on-demand (default is SPOT -- the account's 128-vCPU quota lives on spot)")
    p.add_argument("--keep-alive", dest="keep_alive", action="store_true", default=False,
                   help="opt-in: stay up after DONE (idles to wall-guard floor). Default self-terminates.")
    p.add_argument("--wall-guard-secs", type=int, default=WALL_GUARD_SECS)
    p.add_argument("--spotpick", action="store_true",
                   help="on a spot-capacity error retry down a capacity-diverse SAME-uarch chain")
    p.add_argument("--spotpick-n", type=int, default=5)
    p.add_argument("--dry-run", action="store_true",
                   help="print the full plan + GO/NO-GO; ZERO mutating boto3 calls")
    args = p.parse_args()

    run_subdir = args.run_subdir or f"dcfrcurve_{int(time.time())}"
    run_prefix = f"{args.prefix}/{run_subdir}"

    print("=" * 78)
    print("DCFR-vs-CFR+ EXPLOITABILITY-vs-COMPUTE CURVE -- pre-launch plan")
    print("=" * 78)
    print(f"  run_subdir     : {run_subdir}")
    print(f"  instance_type  : {args.instance_type}  arch={args.arch}  "
          f"market={'spot' if args.spot else 'on-demand'}  keep_alive={args.keep_alive}")
    print(f"  abstraction    : made-hand K={args.num_buckets}  n_deals={args.n_deals}  "
          f"train_rayon={args.train_rayon}")
    print(f"  milestones     : {args.milestones}  (x{args.n_deals} deals)")
    print(f"  dcfr arm       : prune_threshold={args.prune_threshold}  explore_frac={args.explore_frac}")
    print(f"  ws probe       : threads={args.probe_threads_list}  warm={args.warm_iters} "
          f"window={args.window_iters} iters  n_deals={args.probe_n_deals}")
    print(f"  S3 result      : s3://{args.bucket}/{run_prefix}/dcfrcurve/")
    print(f"  wall-guard     : {args.wall_guard_secs}s  train_timeout={args.train_timeout}s")

    ami = None
    try:
        ec2 = boto3.client("ec2", region_name=args.region)
        ami = resolve_ubuntu_ami(ec2, args.arch)
        print(f"  resolved AMI   : {ami}")
    except Exception as e:  # noqa: BLE001
        if args.dry_run:
            ami = "ami-DRYRUN-STUB"
            print(f"  resolved AMI   : {ami}  (describe_images failed: {e!r}) [dry-run stub]")
        else:
            raise

    code_key = f"{args.prefix}/code/dcfr_curve_code.tar.gz"
    awscli_zip = "awscli-exe-linux-aarch64.zip" if args.arch == "arm64" else "awscli-exe-linux-x86_64.zip"
    ud = user_data(bucket=args.bucket, prefix=args.prefix, region=args.region, code_key=code_key,
                   run_subdir=run_subdir, num_buckets=args.num_buckets, n_deals=args.n_deals,
                   milestones=args.milestones, prune_threshold=args.prune_threshold,
                   explore_frac=args.explore_frac, probe_threads_list=args.probe_threads_list,
                   probe_n_deals=args.probe_n_deals, warm_iters=args.warm_iters,
                   window_iters=args.window_iters, train_rayon=args.train_rayon,
                   train_timeout=args.train_timeout, keep_alive=args.keep_alive,
                   wall_guard_secs=args.wall_guard_secs, awscli_zip=awscli_zip)
    ud_bytes = len(ud.encode())
    ud_b64 = len(base64.b64encode(ud.encode()))
    ud_size_ok = (ud_bytes < 15000 and ud_b64 < 16384)

    run_kwargs = dict(
        ImageId=ami, MinCount=1, MaxCount=1,
        UserData=base64.b64encode(ud.encode()).decode(),
        IamInstanceProfile={"Name": args.instance_profile},
        MetadataOptions={"HttpTokens": "required", "HttpPutResponseHopLimit": 2},
        BlockDeviceMappings=[{"DeviceName": "/dev/sda1",
                              "Ebs": {"VolumeSize": 80, "VolumeType": "gp3", "DeleteOnTermination": True}}],
        TagSpecifications=[{"ResourceType": "instance", "Tags": [
            {"Key": "Name", "Value": f"dcfrcurve-{run_subdir}"},
            {"Key": "Project", "Value": "pokerbot-native-cfr"},
            {"Key": "Run", "Value": run_subdir}]}],
        InstanceInitiatedShutdownBehavior="terminate",
    )
    if args.spot:
        run_kwargs["InstanceMarketOptions"] = {
            "MarketType": "spot",
            "SpotOptions": {"SpotInstanceType": "one-time", "InstanceInterruptionBehavior": "terminate"}}

    onbox_txt = (ROOT / "scripts/dcfr_curve_onbox.sh").read_text()
    onbox_cmds = "\n".join(ln for ln in onbox_txt.splitlines() if not ln.lstrip().startswith("#"))

    # ---- GO / NO-GO checklist ----
    checks = []

    def _chk(ok, label):
        checks.append((ok, label))

    _chk(ud_size_ok, "#1 user_data < 15000 plain AND < 16384 base64")
    _chk(set(tar_include_list()) >= {"scripts/dcfr_curve_onbox.sh",
                                     "scripts/drive_dcfr_curve_train.py",
                                     "scripts/dcfr_ws_probe.py", "tools/battery_native"},
         "#2 onbox + curve-train driver + ws-probe + crate tarred")
    _chk(any(b["Ebs"]["VolumeType"] == "gp3" and b["Ebs"]["VolumeSize"] >= 40
             for b in run_kwargs["BlockDeviceMappings"]), "#3 >=40GB gp3 root")
    _chk(("InstanceMarketOptions" not in run_kwargs) or args.spot, "#4 on-demand OR explicit spot")
    _chk(("pokerbot-dcfrguard" in ud) and ("/sbin/shutdown -h +" in ud)
         and (str(args.wall_guard_secs) in ud) and args.wall_guard_secs == WALL_GUARD_SECS,
         f"#5 dead-man cost cap in user_data: shutdown floor + dcfrguard unit @ {args.wall_guard_secs}s")
    _chk(run_kwargs["InstanceInitiatedShutdownBehavior"] == "terminate" and not args.keep_alive,
         "#6 shutdown=terminate AND self-terminate on finish (--no-keep-alive default)")
    _chk(("--features python " in onbox_cmds) and ("hogwild" not in onbox_cmds.split("cargo build")[1].split("\n")[0]
                                                   if "cargo build" in onbox_cmds else False)
         and ("target-cpu=x86-64-v3" in onbox_cmds),
         "#7 onbox builds plain --features python (NO hogwild feature) with -C target-cpu=x86-64-v3")
    _chk(("True, False," in onbox_cmds) and ("True, True," in onbox_cmds) and ("is_hogwild() is False" in onbox_cmds),
         "#8 onbox smokes BOTH arms (work_steal+dcfr-off, work_steal+dcfr-on) and asserts hogwild OFF")
    _chk(("S3_DEST=" in onbox_cmds) and ("drive_dcfr_curve_train.py" in onbox_cmds),
         "#9 curve driver gets S3_DEST -> per-milestone streaming upload (no end-only summary)")
    _chk(("--manifest-path tools/battery_native/Cargo.toml" in onbox_cmds)
         and not any(("sed" in ln or ">>" in ln) and "Cargo.toml" in ln for ln in onbox_cmds.splitlines()),
         "#10 cargo via --manifest-path only; Cargo.toml not edited on box")
    _chk(args.train_timeout + 1800 <= args.wall_guard_secs,
         "#11 train_timeout + build/probe headroom (1800s) fits inside wall-guard")
    _chk(ami is not None and ami != "ami-DRYRUN-STUB",
         f"#12 AWS creds + AMI resolve ({args.instance_type} {'spot' if args.spot else 'on-demand'})")

    print("\n" + "-" * 78)
    print(f"  user_data      : {ud_bytes} bytes / {ud_b64} base64  "
          f"=> {'OK' if ud_size_ok else 'OVER LIMIT'}")
    print(f"  tar includes   : {tar_include_list()}")
    print("-" * 78)
    print("GO / NO-GO CHECKLIST")
    print("-" * 78)
    for ok, label in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}")
    n_fail = sum(1 for ok, _ in checks if not ok)
    print(f"  ---- {len(checks) - n_fail}/{len(checks)} PASS ----")

    if args.dry_run:
        print("\nDRY-RUN: ZERO mutating boto3 calls. Drop --dry-run to launch.")
        return
    if n_fail:
        raise SystemExit(f"{n_fail} go/no-go check(s) FAILED -- refusing to launch.")

    s3 = boto3.client("s3", region_name=args.region)
    code_key = build_and_upload_code(s3, args.bucket, args.prefix)

    run_kwargs.pop("InstanceType", None)
    instance_types = (_spotpick_chain(args.instance_type, args.arch, args.spotpick_n)
                      if (args.spot and args.spotpick) else [args.instance_type])
    iid, used_type = _launch_with_fallback(ec2, run_kwargs, instance_types=instance_types, spot=args.spot)
    print(f"Launched {iid} as {used_type} ({'spot' if args.spot else 'on-demand'})", flush=True)
    print(f"Result S3: s3://{args.bucket}/{run_prefix}/dcfrcurve/", flush=True)

    watch_log = ROOT / "artifacts" / f"dcfrcurve_watch_{run_subdir}.log"
    watch_cmd = [sys.executable, str(ROOT / "scripts" / "_probe_watch.py"),
                 "--instance-id", iid, "--region", args.region,
                 "--s3-prefix", f"s3://{args.bucket}/{run_prefix}/dcfrcurve/",
                 "--max-minutes", str(args.wall_guard_secs // 60 + 5)]
    try:
        watch_log.parent.mkdir(parents=True, exist_ok=True)
        _wlog = open(watch_log, "a", buffering=1)
        if os.name == "nt":
            flags = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
            wp = subprocess.Popen(watch_cmd, stdout=_wlog, stderr=subprocess.STDOUT,
                                  creationflags=flags, close_fds=True)
        else:
            wp = subprocess.Popen(watch_cmd, stdout=_wlog, stderr=subprocess.STDOUT,
                                  start_new_session=True, close_fds=True)
        print(f"Cost-safety poller auto-started (pid {wp.pid}) -> {watch_log}", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"WARN: could not auto-start poller ({e!r}); run manually:\n  "
              f"python scripts/_probe_watch.py --instance-id {iid} --region {args.region} "
              f"--s3-prefix s3://{args.bucket}/{run_prefix}/dcfrcurve/", flush=True)

    print(f"Monitor: aws ec2 get-console-output --instance-id {iid} --region {args.region} --latest")
    print(f"Results: aws s3 ls s3://{args.bucket}/{run_prefix}/dcfrcurve/ --recursive")
    print(f"Score:   python scripts/poll_dcfr_curve.py {run_subdir} {iid}")
    print(f"Terminate: aws ec2 terminate-instances --instance-ids {iid} --region {args.region}")


if __name__ == "__main__":
    main()
