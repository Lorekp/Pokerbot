#!/usr/bin/env python3
"""Launch ONE spot box that measures the Hogwild-vs-fork-merge THROUGHPUT CROSSOVER on the production
EMD-K160 native CFR+ path (deals/s for each arm x RAYON_NUM_THREADS in {16,32,64,128}).

WHY a big box: local 12-core was a wash (the per-iteration fork-merge barrier idle is small there); the
idle GROWS with core count, so the crossover where lock-free Hogwild overtakes fork-merge can only be
measured at 32/64/128 cores. This is THROUGHPUT only — the QUALITY (LBR) A/B runs locally for free
(native Lbr6Harness is made-hand-only, can't score an EMD blueprint), so this box never trains to
convergence or runs LBR; it just times deals/s. Short (~20-30 min) and cheap (~$1).

Cost-safety is a faithful clone of launch_cpu128_probe_spot.py (the proven template): a logind
`shutdown -h +N` ABSOLUTE floor (immune to cloud-init cgroup teardown) + a `pokerbot-wallguard`
systemd-run transient unit (NOT oneshot) + InstanceInitiatedShutdownBehavior=terminate + SELF-TERMINATE
on finish (--no-keep-alive default) + an auto-wired _probe_watch.py poller. ONE binary built with
--features python,hogwild serves BOTH arms via the runtime toggle (same-binary-fair A/B).
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
from launch_cpu128_probe_spot import (  # noqa: E402
    EXPECTED_FROZEN_BYTES, EXPECTED_FROZEN_SHA256, EXPECTED_TABLE_ID, table_id,
)

# Box does quality-train(<=900s) THEN throughput-sweep(<=2700s) = 3600s work-cap; a 4200s wall-guard
# nests outside both (realistic total ~37 min; the box self-terminates on finish well before this).
WALL_GUARD_SECS = 4200

_INCLUDES = [
    ("tools/battery_native", "tools/battery_native"),
    ("scripts/hogwild_throughput_onbox.sh", "scripts/hogwild_throughput_onbox.sh"),
    ("scripts/drive_hogwild_throughput.py", "scripts/drive_hogwild_throughput.py"),
    ("scripts/drive_hogwild_quality_train.py", "scripts/drive_hogwild_quality_train.py"),
    ("data/abstraction/emd_frozen_flop_turn_160_r400.emdfrzn.sha256",
     "data/abstraction/emd_frozen_flop_turn_160_r400.emdfrzn.sha256"),
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
    key = f"{prefix}/code/hogwild_throughput_code.tar.gz"
    s3.put_object(Bucket=bucket, Key=key, Body=buf.read())
    print(f"Uploaded code tarball -> s3://{bucket}/{key} ({buf.tell()} bytes)", flush=True)
    return key


def user_data(*, bucket, prefix, region, code_key, run_subdir, frozen_key, threads_list, n_deals_list,
              prod_n_deals, window_secs, warm_secs, num_buckets, qual_num_buckets, qual_n_deals,
              qual_iters, keep_alive, wall_guard_secs, awscli_zip="awscli-exe-linux-x86_64.zip") -> str:
    run_prefix = f"{prefix}/{run_subdir}"
    result_prefix = f"{run_prefix}/hogwild"
    final = "echo BOOTSTRAP_DONE_KEEP_ALIVE" if keep_alive else "shutdown -h now"
    floor_min = wall_guard_secs // 60 + 5
    return f"""#!/bin/bash
set -uxo pipefail
export HOME="${{HOME:-/root}}"
export DEBIAN_FRONTEND=noninteractive AWS_DEFAULT_REGION={region}
mkdir -p /opt/pokerbot /var/log/hogwild

# ===== STEP 0 — DEAD-MAN COST CAP, ARMED FIRST (before apt/aws) =====
/sbin/shutdown -h +{floor_min} "pokerbot hogwild wall-guard absolute cost cap" || true
systemd-run --unit=pokerbot-wallguard --collect -p TimeoutStartSec=infinity --no-block /bin/bash -c 'sleep {wall_guard_secs} ; sync ; timeout 120 aws s3 cp --recursive /var/log/hogwild "s3://{bucket}/{result_prefix}/WALLGUARD_SWEEP/" --region {region} 2>/dev/null || true ; /sbin/shutdown -h now' || true
echo "WALL_GUARD armed: shutdown +{floor_min}min floor + pokerbot-wallguard unit @ {wall_guard_secs}s"

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
export FROZEN_KEY="{frozen_key}"
export EXPECTED_FROZEN_SHA256="{EXPECTED_FROZEN_SHA256}"
export THREADS_LIST="{threads_list}"
export N_DEALS_LIST="{n_deals_list}"
export PROD_N_DEALS="{prod_n_deals}"
export WINDOW_SECS="{window_secs}"
export WARM_SECS="{warm_secs}"
export NUM_BUCKETS="{num_buckets}"
export QUAL_NUM_BUCKETS="{qual_num_buckets}"
export QUAL_N_DEALS="{qual_n_deals}"
export QUAL_ITERS="{qual_iters}"
export PYTHONPATH=/opt/pokerbot
export RUST_BACKTRACE=1

bash /opt/pokerbot/scripts/hogwild_throughput_onbox.sh
echo "onbox exit $?"
{final}
"""


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--instance-type", default="c6i.32xlarge",
                   help="128-vCPU Ice Lake (matches the production training box). No metal/PMU needed — "
                        "this is wall-clock throughput only.")
    p.add_argument("--arch", default="x86_64", choices=["x86_64", "arm64"])
    p.add_argument("--threads-list", default="16,32,64,128",
                   help="RAYON_NUM_THREADS values to sweep per arm (the crossover x-axis)")
    p.add_argument("--window-secs", type=float, default=30.0, help="timed deals/s window per config")
    p.add_argument("--warm-secs", type=float, default=15.0,
                   help="warmup-to-steady per config (V3: long enough to saturate infoset discovery)")
    p.add_argument("--num-buckets", type=int, default=160, help="EMD bucket count K (== frozen table)")
    p.add_argument("--n-deals-list", default="200,1600,12800",
                   help="per-iteration batch sizes to sweep per (arm,threads) — the envelope x-axis. "
                        "fork-merge is squeezed (small=starved threads, large=cache thrash); hogwild has "
                        "no per-task map so it can use large batches to saturate cores penalty-free.")
    p.add_argument("--prod-n-deals", type=int, default=200,
                   help="the production-shape batch reported as its own crossover row")
    p.add_argument("--qual-num-buckets", type=int, default=200,
                   help="made-hand bucket count for the QUALITY blueprints (must match the local LBR's "
                        "POKERBOT_POSTFLOP_NUM_BUCKETS so the blueprint is abstraction-matched)")
    p.add_argument("--qual-n-deals", type=int, default=1024,
                   help="core-saturating MC batch for quality training (same for both arms -> fair LBR)")
    p.add_argument("--qual-iters", type=int, default=12000,
                   help="quality training iters (x qual_n_deals deals; 12000x1024 ~= 12.3M ~= converged)")
    p.add_argument("--centroids", default="data/abstraction/emd_buckets_160_r400.json")
    p.add_argument("--emd-frozen-rel", default="data/abstraction/emd_frozen_flop_turn_160_r400.emdfrzn")
    p.add_argument("--n-runouts", type=int, default=400)
    p.add_argument("--n-bins", type=int, default=20)
    p.add_argument("--run-subdir", default=None)
    p.add_argument("--bucket", default=DEFAULT_BUCKET)
    p.add_argument("--prefix", default=DEFAULT_PREFIX)
    p.add_argument("--region", default=DEFAULT_REGION)
    p.add_argument("--instance-profile", default=DEFAULT_PROFILE)
    p.add_argument("--on-demand", dest="spot", action="store_false", default=True,
                   help="use on-demand (default is SPOT — the account's 128-vCPU quota lives on spot)")
    p.add_argument("--keep-alive", dest="keep_alive", action="store_true", default=False,
                   help="opt-in: stay up after DONE (idles to wall-guard floor). Default self-terminates.")
    p.add_argument("--wall-guard-secs", type=int, default=WALL_GUARD_SECS)
    p.add_argument("--spotpick", action="store_true",
                   help="on a spot-capacity error retry down a capacity-diverse SAME-uarch chain")
    p.add_argument("--spotpick-n", type=int, default=5)
    p.add_argument("--dry-run", action="store_true",
                   help="print the full plan + GO/NO-GO; ZERO mutating boto3 calls")
    args = p.parse_args()

    threads = [int(x) for x in args.threads_list.split(",") if x.strip()]
    centroids_path = ROOT / args.centroids
    frozen_path = ROOT / args.emd_frozen_rel
    frozen_sha_path = Path(str(frozen_path) + ".sha256")

    if not centroids_path.exists():
        p.error(f"--centroids {args.centroids} not found (needed to compute TABLE_ID)")
    TID = table_id(centroids_path, args.n_bins, args.n_runouts)
    table_id_ok = (TID == EXPECTED_TABLE_ID)

    run_subdir = args.run_subdir or f"hogthru_{int(time.time())}"
    run_prefix = f"{args.prefix}/{run_subdir}"
    frozen_key = f"{run_prefix}/frozen/{TID}.emdfrzn"

    frozen_bytes_ok = frozen_path.exists() and frozen_path.stat().st_size == EXPECTED_FROZEN_BYTES
    frozen_sha_ok = (frozen_sha_path.exists()
                     and frozen_sha_path.read_text().strip().split()[0] == EXPECTED_FROZEN_SHA256)

    print("=" * 78)
    print("HOGWILD THROUGHPUT CROSSOVER — pre-launch plan")
    print("=" * 78)
    print(f"  run_subdir     : {run_subdir}")
    print(f"  instance_type  : {args.instance_type}  arch={args.arch}  "
          f"market={'spot' if args.spot else 'on-demand'}  keep_alive={args.keep_alive}")
    print(f"  threads sweep  : {threads}  (per arm: fork-merge, hogwild)")
    print(f"  window/warm    : {args.window_secs}s / {args.warm_secs}s per config")
    print(f"  EMD K          : {args.num_buckets}  TABLE_ID={TID} (expected {EXPECTED_TABLE_ID}) "
          f"=> {'OK' if table_id_ok else 'MISMATCH — ABORT'}")
    print(f"  frozen blob    : {args.emd_frozen_rel}  "
          f"size={'OK' if frozen_bytes_ok else 'BAD'}  sha={'OK' if frozen_sha_ok else 'BAD'}")
    print(f"  S3 result      : s3://{args.bucket}/{run_prefix}/hogwild/")
    print(f"  wall-guard     : {args.wall_guard_secs}s")

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

    code_key = f"{args.prefix}/code/hogwild_throughput_code.tar.gz"
    awscli_zip = "awscli-exe-linux-aarch64.zip" if args.arch == "arm64" else "awscli-exe-linux-x86_64.zip"
    ud = user_data(bucket=args.bucket, prefix=args.prefix, region=args.region, code_key=code_key,
                   run_subdir=run_subdir, frozen_key=frozen_key, threads_list=args.threads_list,
                   n_deals_list=args.n_deals_list, prod_n_deals=args.prod_n_deals,
                   window_secs=args.window_secs, warm_secs=args.warm_secs, num_buckets=args.num_buckets,
                   qual_num_buckets=args.qual_num_buckets, qual_n_deals=args.qual_n_deals,
                   qual_iters=args.qual_iters, keep_alive=args.keep_alive,
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
            {"Key": "Name", "Value": f"hogthru-{run_subdir}"},
            {"Key": "Project", "Value": "pokerbot-native-cfr"},
            {"Key": "Run", "Value": run_subdir}]}],
        InstanceInitiatedShutdownBehavior="terminate",
    )
    if args.spot:
        run_kwargs["InstanceMarketOptions"] = {
            "MarketType": "spot",
            "SpotOptions": {"SpotInstanceType": "one-time", "InstanceInterruptionBehavior": "terminate"}}

    onbox_txt = (ROOT / "scripts/hogwild_throughput_onbox.sh").read_text()
    onbox_cmds = "\n".join(ln for ln in onbox_txt.splitlines() if not ln.lstrip().startswith("#"))

    # ---- GO / NO-GO checklist ----
    checks = []

    def _chk(ok, label):
        checks.append((ok, label))

    _chk(table_id_ok, "#1 TABLE_ID == 60232de66618c7d1 (centroids unchanged -> frozen blob valid)")
    _chk(ud_size_ok, "#2 user_data < 15000 plain AND < 16384 base64")
    _chk(set(tar_include_list()) >= {"scripts/hogwild_throughput_onbox.sh",
                                     "scripts/drive_hogwild_throughput.py",
                                     "scripts/drive_hogwild_quality_train.py", "tools/battery_native"}
         and args.emd_frozen_rel not in tar_include_list(),
         "#3 onbox + throughput + quality-train driver + crate tarred; 152MB .emdfrzn NOT tarred")
    _chk(any(b["Ebs"]["VolumeType"] == "gp3" and b["Ebs"]["VolumeSize"] >= 40
             for b in run_kwargs["BlockDeviceMappings"]), "#4 >=40GB gp3 root")
    _chk(("InstanceMarketOptions" not in run_kwargs) or args.spot,
         "#5 on-demand OR explicit spot")
    _chk(("pokerbot-wallguard" in ud) and ("/sbin/shutdown -h +" in ud)
         and (str(args.wall_guard_secs) in ud) and args.wall_guard_secs == WALL_GUARD_SECS,
         f"#6 dead-man cost cap in user_data: shutdown floor + wallguard unit @ {args.wall_guard_secs}s")
    _chk(run_kwargs["InstanceInitiatedShutdownBehavior"] == "terminate" and not args.keep_alive,
         "#7 shutdown=terminate AND self-terminate on finish (--no-keep-alive default)")
    _chk(frozen_bytes_ok and frozen_sha_ok,
         "#8 local frozen blob bytes + sha256 sidecar intact (pre-upload source is production blob)")
    _chk(("--features python,hogwild" in onbox_cmds) and ("target-cpu=x86-64-v3" in onbox_cmds),
         "#9 onbox builds --features python,hogwild with -C target-cpu=x86-64-v3")
    _chk(("sha256sum" in onbox_cmds) and ("EXPECTED_FROZEN_SHA256" in onbox_cmds)
         and ("BUILD_FAIL" in onbox_cmds) and ("is_hogwild" in onbox_cmds),
         "#10 onbox verifies frozen sha + build-fail marker + both-arms toggle smoke (fail-fast)")
    _chk(("--manifest-path tools/battery_native/Cargo.toml" in onbox_cmds)
         and ("Cargo.toml" not in "".join(ln for ln in onbox_cmds.splitlines()
                                          if "sed" in ln or ">>" in ln)),
         "#11 cargo via --manifest-path only; Cargo.toml not edited on box")
    _chk(ami is not None and ami != "ami-DRYRUN-STUB",
         f"#12 AWS creds + AMI resolve ({args.instance_type} {'spot' if args.spot else 'on-demand'})")

    print("\n" + "-" * 78)
    print(f"  user_data      : {ud_bytes} bytes / {ud_b64} base64  "
          f"=> {'OK' if ud_size_ok else 'OVER LIMIT'}")
    print(f"  tar includes   : {tar_include_list()}")
    print(f"  NOT tarred     : {args.emd_frozen_rel} (152MB S3 pre-upload)")
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
        raise SystemExit(f"{n_fail} go/no-go check(s) FAILED — refusing to launch.")

    s3 = boto3.client("s3", region_name=args.region)
    print(f"Pre-uploading frozen blob (152MB) -> s3://{args.bucket}/{frozen_key} ...", flush=True)
    s3.upload_file(str(frozen_path), args.bucket, frozen_key)
    code_key = build_and_upload_code(s3, args.bucket, args.prefix)

    run_kwargs.pop("InstanceType", None)
    instance_types = (_spotpick_chain(args.instance_type, args.arch, args.spotpick_n)
                      if (args.spot and args.spotpick) else [args.instance_type])
    iid, used_type = _launch_with_fallback(ec2, run_kwargs, instance_types=instance_types, spot=args.spot)
    print(f"Launched {iid} as {used_type} ({'spot' if args.spot else 'on-demand'})", flush=True)
    print(f"Result S3: s3://{args.bucket}/{run_prefix}/hogwild/", flush=True)

    watch_log = ROOT / "artifacts" / f"hogthru_watch_{run_subdir}.log"
    watch_cmd = [sys.executable, str(ROOT / "scripts" / "_probe_watch.py"),
                 "--instance-id", iid, "--region", args.region,
                 "--s3-prefix", f"s3://{args.bucket}/{run_prefix}/hogwild/",
                 # C1: cap the external poller just above the on-box +75min floor so it's a real backstop
                 # (default 115min is looser than the on-box wall-guard); the onbox also emits PROBE_DONE.
                 "--max-minutes", "80"]
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
              f"--s3-prefix s3://{args.bucket}/{run_prefix}/hogwild/", flush=True)

    print(f"Monitor: aws ec2 get-console-output --instance-id {iid} --region {args.region} --latest")
    print(f"Results: aws s3 ls s3://{args.bucket}/{run_prefix}/hogwild/ --recursive")
    print(f"Terminate: aws ec2 terminate-instances --instance-ids {iid} --region {args.region}")


if __name__ == "__main__":
    main()
