#!/usr/bin/env python3
"""Launch ONE EC2 box that runs the instrumented 128-vCPU NATIVE-CFR PROFILING PROBE.

This is a DEDICATED one-shot launcher (a copy of ``launch_native_cfr_training_spot.py``'s structure, NOT
the 5-phase EMD pipeline — we do NOT retrain/ship/eval; this is a pure profiling pass). It builds a
debug-symbol'd profiling binary, pre-uploads the 152.5MB frozen EMD blob to S3, then execs a per-shape
``perf``/``numastat``/``mpstat`` capture loop (scripts/cpu128_probe_onbox.sh) to attribute the missing
~80% CPU of the native CFR+ trainer (WF6: memory-latency-bound vs NUMA cross-socket vs collect-barrier
imbalance vs 128-thread allocator/atomic contention vs SMT-sibling contention vs wall-average artifact).

DEFAULTS to ON-DEMAND (the operator opts into spot with --spot for the metal PHE-cert run, where the
account's 128-vCPU quota lives and the per-batch S3 upload + IMDSv2 reclaim trap + WALL_GUARD make a
reclaim cost only the in-flight shape) and to SELF-TERMINATE on finish (--no-keep-alive is the default;
strictly safer — a successful run shuts down rather than idling to the wall-guard floor, and artifacts are
in S3 as-you-go so keep-alive is unneeded). --keep-alive is opt-in. NO babysitter: one ``run_instances``,
MinCount=MaxCount=1 (an auto-wired _probe_watch.py poller is the external cost-safety backstop).

Reuses the native launcher's helpers (resolve_ubuntu_ami / _tar_filter / _launch_with_fallback /
_spotpick_chain) + the EMD launcher's ``table_id`` (copied VERBATIM) + the SAME AWS identity
(bucket=video-processing-6789, prefix=pokerbot/prod6000/native_cfr, region=eu-central-1,
profile=pokerbot-labelgen-worker).

FOUR BLOCKING FIXES are baked in (see artifacts/_wf_probe_refute_20260628.json):
  #1 DISK-FULL  -> 150GB gp3 root via BlockDeviceMappings (default Canonical root ~8GB overflows under
                   debuginfo=2+strip=none cargo target + 152MB blob + 4x perf.data); the on-box driver
                   also asserts `df -BG /` before each perf record.
  #2 PERF-BROKEN -> the on-box driver HARD-GATES perf (perf stat -e cycles true + a 2s perf record self
                   test) and DEGRADES to numastat+mpstat+train_profile_json-only (never a silent empty
                   flamegraph that looks like success).
  #3 PMU MASKED  -> per-event `<not supported>` parse + a NUMA hypothesis gated on `numactl --hardware`
                   node count (NUMA_NODES<2 => REFUTE-by-topology).
  #4 COST-RUNAWAY -> the on-box driver's steady-wait loop checks PID `kill -0` liveness + a STEADY_TIMEOUT,
                   and user_data launches an INDEPENDENT WALL_GUARD_SECS=5400 watchdog
                   (`nohup sleep 5400 && upload && shutdown -h now`) so keep-alive can never bill forever.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import os
import subprocess
import sys
import time
from pathlib import Path

import boto3

# Reuse the SAME helpers/conventions as the native-CFR launcher (spotpick chain, AMI resolve, fallback
# launch, tar filter, AWS identity) so this stays a thin profiling layer on the proven launch path.
from launch_native_cfr_training_spot import (  # noqa: E402
    DEFAULT_BUCKET,
    DEFAULT_PREFIX,
    DEFAULT_PROFILE,
    DEFAULT_REGION,
    ROOT,
    _launch_with_fallback,
    _spotpick_chain,
    _tar_filter,
    resolve_ubuntu_ami,
)

import io
import tarfile

# The production frozen EMD blob this probe profiles against. K=160, n_bins=20, n_runouts=400.
EXPECTED_TABLE_ID = "60232de66618c7d1"
EXPECTED_FROZEN_BYTES = 152_468_448
EXPECTED_FROZEN_SHA256 = "344ab2fd328f4b360fd2c11a09c1f6fd2c45a56e62f75d6bc1a502c71612cb74"

# Hard ceiling for the unattended on-box wall-clock watchdog (BLOCKING-FIX #4). 5400s = 90 min caps the
# worst-case bill if a build or shape hangs; the run's ETA is ~40-50 min, well inside it.
WALL_GUARD_SECS = 5400


def table_id(centroids_path: Path, n_bins: int, n_runouts: int) -> str:
    """Deterministic table id from the centroids file content + n_bins + n_runouts. Copied VERBATIM from
    scripts/launch_emd_pipeline_spot.py (VERIFIED computes 60232de66618c7d1 for emd_buckets_160_r400.json,
    n_bins=20, n_runouts=400). Two runs with the SAME centroids + params reuse the SAME cached frozen
    table (the precompute is a pure function of these, so a content hash is a correct cache key)."""
    h = hashlib.sha256()
    h.update(centroids_path.read_bytes())
    h.update(f"|n_bins={n_bins}|n_runouts={n_runouts}".encode())
    return h.hexdigest()[:16]


# Tar allow-list for the probe. The native launcher's _INCLUDE (tools/battery_native + native_solver.py +
# tabular_blueprint_adapter.py) PLUS the 4 probe additions. The 152MB .emdfrzn is NOT tarred — it is
# pre-uploaded to S3 by the launcher (the box CACHE_HITs and never recomputes the ~16.5M spots).
_PROBE_INCLUDES = [
    # --- already in the native launcher's _INCLUDE (keep) ---
    ("tools/battery_native", "tools/battery_native"),
    ("pokerbot/search/native_solver.py", "pokerbot/search/native_solver.py"),
    ("pokerbot/search/tabular_blueprint_adapter.py", "pokerbot/search/tabular_blueprint_adapter.py"),
    # --- the 4 probe additions ---
    ("scripts/cpu128_probe_onbox.sh", "scripts/cpu128_probe_onbox.sh"),
    ("scripts/drive_cpu128_probe.py", "scripts/drive_cpu128_probe.py"),
    ("scripts/profile_native_trainer_microbench.py", "scripts/profile_native_trainer_microbench.py"),
    ("data/abstraction/emd_buckets_160_r400.json", "data/abstraction/emd_buckets_160_r400.json"),
    ("data/abstraction/emd_frozen_flop_turn_160_r400.emdfrzn.sha256",
     "data/abstraction/emd_frozen_flop_turn_160_r400.emdfrzn.sha256"),
]


def tar_include_list() -> list[str]:
    """The arc-names that WILL ship in the code tarball (for --dry-run reporting + the go/no-go check)."""
    return [arc for _src, arc in _PROBE_INCLUDES]


def build_and_upload_code(s3, bucket: str, prefix: str, *, crate_src: str = "tools/battery_native") -> str:
    """Tar the crate (as tools/battery_native) + the 2 native serve helpers + the 4 probe additions, then
    upload as the probe code tarball. Does NOT ship the whole pokerbot package (the probe only constructs
    NlheCFRTrainer + calls train/train_profile_json — no JAX/serve/eval). The 152MB .emdfrzn is NOT here
    (pre-uploaded to S3). _tar_filter strips bytecode/.git/target."""
    includes = list(_PROBE_INCLUDES)
    # Honor a frozen-copy crate_src exactly like the native launcher (always arc'd as tools/battery_native).
    if crate_src and crate_src != "tools/battery_native":
        includes[0] = (crate_src, "tools/battery_native")
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for src, arc in includes:
            p = ROOT / src
            if p.exists():
                tar.add(p, arcname=arc, filter=_tar_filter)
            else:
                print(f"WARN: tar include missing on disk, skipped: {src}", flush=True)
    buf.seek(0)
    key = f"{prefix}/code/cpu128_probe_code.tar.gz"
    s3.put_object(Bucket=bucket, Key=key, Body=buf.read())
    print(f"Uploaded probe code tarball -> s3://{bucket}/{key} ({buf.tell()} bytes; crate_src={crate_src})",
          flush=True)
    return key


def user_data(*, bucket, prefix, region, code_key, run_subdir, table_id_hex, num_buckets, warmup_cap,
              window_iters, profile_every, shapes, n_runouts, n_bins, keep_alive=True,
              wall_guard_secs=WALL_GUARD_SECS, awscli_zip="awscli-exe-linux-x86_64.zip") -> str:
    """A MINIMAL bootstrap (kept well under 16KB): install deps -> rust toolchain -> download+extract the
    tarball -> EXPORT all probe config as env -> launch the INDEPENDENT wall-guard watchdog BEFORE the
    driver (BLOCKING-FIX #4) -> exec scripts/cpu128_probe_onbox.sh (which holds ALL the heavy build/perf/
    capture logic). The frozen blob download, perf hard-gate, per-shape loop, flamegraph, and upload-as
    -you-go all live in cpu128_probe_onbox.sh so this stays tiny."""
    run_prefix = f"{prefix}/{run_subdir}"
    probe_prefix = f"s3://{bucket}/{run_prefix}/probe"
    final = "echo BOOTSTRAP_DONE_KEEP_ALIVE" if keep_alive else "shutdown -h now"
    floor_min = wall_guard_secs // 60 + 5  # absolute shutdown floor: 5 min after the wall-guard unit
    return f"""#!/bin/bash
set -uxo pipefail
export HOME="${{HOME:-/root}}"   # cloud-init runs as root with HOME UNSET; under set -u a bare $HOME (rustup/cargo PATH) is FATAL
export DEBIAN_FRONTEND=noninteractive AWS_DEFAULT_REGION={region}
mkdir -p /opt/pokerbot /var/log/probe

# ===================== STEP 0 — DEAD-MAN COST CAP, ARMED FIRST (before apt/aws) =====================
# TWO independent layers so the box CANNOT outlive ~{floor_min} min no matter what fails below:
#  0a) ABSOLUTE FLOOR: `shutdown -h +{floor_min}` is scheduled by systemd/logind ITSELF (it is NOT a
#      child process / sleep), so it is immune to cloud-init's cgroup teardown — THE bug that idle-billed
#      the box for 3h20m when the old `nohup &` watchdog was reaped with cloud-final's cgroup at t=80s. It
#      needs no packages and fires even if systemd-run, aws, apt, rustup, AND the external poller all fail.
#  0b) WALL-GUARD UNIT: a systemd-run transient unit (its own cgroup) that at {wall_guard_secs}s sweeps any
#      un-uploaded /var/log/probe to S3, then `shutdown -h now` (supersedes 0a, ~5 min earlier). CRITICAL:
#      NO --service-type=oneshot — a oneshot unit is "activating" for its whole ExecStart and bounded by
#      DefaultTimeoutStartSec=90s, which would SIGTERM-reap our `sleep {wall_guard_secs}` at ~90s and
#      re-create the runaway. A default (simple) unit + explicit -p TimeoutStartSec=infinity has no such
#      bound. The S3 sweep is `timeout 120`-wrapped so a hung endpoint can't delay the unconditional shutdown.
/sbin/shutdown -h +{floor_min} "pokerbot wall-guard absolute cost cap" || true
systemd-run --unit=pokerbot-wallguard --collect -p TimeoutStartSec=infinity --no-block /bin/bash -c 'sleep {wall_guard_secs} ; sync ; timeout 120 aws s3 cp --recursive /var/log/probe "{probe_prefix}/WALLGUARD_SWEEP/" --region {region} 2>/dev/null || true ; /sbin/shutdown -h now' || true
echo "WALL_GUARD armed: absolute shutdown +{floor_min}min floor + pokerbot-wallguard unit at {wall_guard_secs}s"

apt-get update -qq
apt-get install -y -qq python3-pip python3-dev gzip curl unzip build-essential

# AWS CLI v2 — self-contained (its botocore is bundled under /usr/local/aws-cli, NOT system site-packages),
# so the driver's `pip install boto3` below can bump the SYSTEM botocore WITHOUT breaking the CLI. The apt
# awscli v1 broke exactly this way: pip boto3 pulled a newer botocore -> awscli v1.22 raised
# `KeyError: 'opsworkscm'` on EVERY `aws` call -> frozen-download + every upload failed at t=73s.
curl -fsSL "https://awscli.amazonaws.com/{awscli_zip}" -o /tmp/awscliv2.zip
unzip -q /tmp/awscliv2.zip -d /tmp
/tmp/aws/install --bin-dir /usr/local/bin --install-dir /usr/local/aws-cli --update
hash -r
aws --version   # MUST be aws-cli/2.x; if this dies the STEP-0 watchdog still bounds the box

curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal
export PATH="$HOME/.cargo/bin:$PATH"
rustc --version

aws s3 cp s3://{bucket}/{code_key} /tmp/code.tar.gz --region {region}
tar -xzf /tmp/code.tar.gz -C /opt/pokerbot
cd /opt/pokerbot
# boto3 for the Python driver. Safe now: it may bump the SYSTEM botocore but the v2 CLI is isolated.
python3 -m pip install --disable-pip-version-check --no-input boto3 || echo "WARN pip boto3"

# --- ALL probe config as env (consumed by cpu128_probe_onbox.sh + drive_cpu128_probe.py) ---
export BUCKET="{bucket}"
export PREFIX="{prefix}"
export REGION="{region}"
export RUN_SUBDIR="{run_subdir}"
export RUN_PREFIX="{run_prefix}"
export PROBE_PREFIX="{probe_prefix}"
export TABLE_ID="{table_id_hex}"
export NUM_BUCKETS="{num_buckets}"
export N_RUNOUTS="{n_runouts}"
export N_BINS="{n_bins}"
export WARMUP_CAP="{warmup_cap}"
export WINDOW_ITERS="{window_iters}"
export PROFILE_EVERY="{profile_every}"
export SHAPES="{shapes}"
export WALL_GUARD_SECS="{wall_guard_secs}"
export PYTHONPATH=/opt/pokerbot
export RUST_BACKTRACE=1

# (the dead-man wall-guard was already armed in STEP 0, before apt — see the top of this script)
bash /opt/pokerbot/scripts/cpu128_probe_onbox.sh
echo "onbox exit $?"
{final}
"""


def _shape_table(shapes: list[str]) -> str:
    """Human-readable per-shape config table for --dry-run (the on-box driver owns the canonical mapping;
    this mirrors it for the operator's go/no-go read)."""
    cfg = {
        "PROD64": "ab_stub=none, RAYON_NUM_THREADS=64 (HEADLINE) — full PMU cert at the 64t regime: IPC, stall-fraction "
                  "(cycle_activity.stalls_mem_any/cyc), effective MLP (l1d_pend_miss.pending/pending_cycles), offcore MLP "
                  "(split x2), DRAM local/remote. + CPU/DRAM flamegraph if this box delivers sampling.",
        "PROD": "ab_stub=none, 128t — same full PMU, for the 64-vs-128 stall/MLP delta = does 128t add memory-channel "
                "contention (stall up / MLP flat) or is each thread already at the per-thread latency wall?",
        "EMDNOMEMO64": "ab_stub=emd_nomemo, RAYON_NUM_THREADS=64 — a REAL 152MB FxHashMap probe EVERY postflop node; "
                       "its DRAM-flamegraph + stall/MLP/DRAM PMU = the frozen-table's microarchitectural DRAM signature.",
        "EMDNOMEMO": "(legacy) ab_stub=emd_nomemo, 128t.",
        "REGRETZERO": "(legacy) ab_stub=regret_zero — tree-drift confounded, dropped from default.",
        "SHOWDOWNCONST": "(legacy) ab_stub=showdown_const — mild drift, dropped from default.",
    }
    lines = []
    for s in shapes:
        lines.append(f"    {s}: {cfg.get(s, '(unknown shape — onbox skips)')}")
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--instance-type", default="c6i.metal",
                   help="c6i.metal=bare-metal Ice Lake 8375C, WORKING PMU (perf G2 self-record PASSES; same "
                        "uarch as the c6i.32xlarge KVM probe). Do NOT substitute c6a.metal (AMD, different PMU "
                        "events). c6i.32xlarge=KVM, PMU was NOT_SUPPORTED (inferred-only verdict).")
    p.add_argument("--arch", default="x86_64", choices=["x86_64", "arm64"])
    p.add_argument("--num-buckets", type=int, default=160,
                   help="postflop EMD bucket count K (MUST equal the frozen table's K=160)")
    p.add_argument("--centroids", default="data/abstraction/emd_buckets_160_r400.json",
                   help="centroids JSON (the table_id cache-key input)")
    p.add_argument("--emd-frozen-rel", default="data/abstraction/emd_frozen_flop_turn_160_r400.emdfrzn",
                   help="repo-relative path to the 152MB frozen (hole,board)->bucket blob (PRE-UPLOADED to "
                        "S3, NOT tarred). The .sha256 sidecar IS tarred for on-box integrity verify.")
    p.add_argument("--n-runouts", type=int, default=400, help="table_id param (frozen blob's n_runouts)")
    p.add_argument("--n-bins", type=int, default=20, help="table_id param (frozen blob's n_bins)")
    p.add_argument("--shapes", default="PROD64,PROD,EMDNOMEMO64",
                   help="PMU-CERT re-run (perf-stat timeout-SIGTERM bug FIXED -> recovers stall/MLP/IPC/DRAM the prior "
                        "run lost). PROD64(64t, the headline regime) + PROD(128t comparison) full PMU -> the 64-vs-128 "
                        "stall/MLP delta = contention vs per-thread latency wall; EMDNOMEMO64(64t, real 152MB probe "
                        "every node) = the frozen-table DRAM signature. Subtractive nodes/s verdict already banked; "
                        "this run is the microarchitectural cert.")
    p.add_argument("--warmup-iters", type=int, default=800,
                   help="WARMUP_CAP backstop (the driver's own steady-detect is the primary stop)")
    p.add_argument("--window-iters", type=int, default=200, help="measured iters per shape (per window)")
    p.add_argument("--profile-every", type=int, default=20,
                   help="train_profile_json batch size = stream+fsync+S3 granularity (kill loses <this)")
    p.add_argument("--run-subdir", default=None, help="S3 subdir for this run (default: probe_<ts>)")
    p.add_argument("--bucket", default=DEFAULT_BUCKET)
    p.add_argument("--prefix", default=DEFAULT_PREFIX)
    p.add_argument("--region", default=DEFAULT_REGION)
    p.add_argument("--instance-profile", default=DEFAULT_PROFILE)
    # ON-DEMAND is the DEFAULT for this probe (spot reclaim corrupts the signal + re-triggers data-loss).
    # --on-demand is a store_true so the default invocation (which passes it) is explicit; --spot flips it.
    p.add_argument("--on-demand", dest="spot", action="store_false", default=False,
                   help="(default) on-demand, no InstanceMarketOptions — a reclaim can't corrupt the window")
    p.add_argument("--spot", dest="spot", action="store_true",
                   help="opt-in to spot (NOT recommended for the probe; the signal is reclaim-fragile)")
    # SELF-TERMINATE is the DEFAULT (strictly safer): on finish the box `shutdown -h now` -> terminate, so a
    # successful run cannot idle-bill to the wall-guard floor. Artifacts are uploaded as-you-go to S3, so
    # keep-alive is NOT needed to retrieve them. --keep-alive is opt-in (e.g. to poke at a live box). The
    # STEP-0 cost cap is UNCONDITIONAL either way; #6 does not gate on keep_alive. (refuter-1 cost-safety fix:
    # the safe behavior is now the default so dropping a flag can never leave a box idling 90 min on success.)
    p.add_argument("--keep-alive", dest="keep_alive", action="store_true", default=False,
                   help="opt-in: box stays up after PROBE_DONE for manual S3 confirm (idles until wall-guard "
                        "floor — costs metal-spot for up to wall-guard-secs on success; NOT the default)")
    p.add_argument("--no-keep-alive", dest="keep_alive", action="store_false",
                   help="(default) self-terminate on finish (strictly safer; WALL_GUARD still bounds a hang)")
    p.add_argument("--wall-guard-secs", type=int, default=WALL_GUARD_SECS,
                   help="independent on-box wall-clock watchdog ceiling (BLOCKING-FIX #4)")
    p.add_argument("--spotpick", action="store_true",
                   help="on a spot-capacity error, retry down a capacity-diverse chain (spot only)")
    p.add_argument("--spotpick-n", type=int, default=5)
    p.add_argument("--crate-src", default="tools/battery_native",
                   help="crate source to upload (always arc'd as tools/battery_native)")
    p.add_argument("--dry-run", action="store_true",
                   help="print the full plan + computed TABLE_ID + user_data size + S3 paths + go/no-go; "
                        "makes ZERO mutating boto3 calls (read-only describe_images only, stubbed if creds fail)")
    args = p.parse_args()

    shapes = [s.strip() for s in args.shapes.split(",") if s.strip()]
    if args.num_buckets <= 0:
        p.error("--num-buckets must be > 0 (the frozen table is K=160)")

    centroids_path = ROOT / args.centroids
    frozen_path = ROOT / args.emd_frozen_rel
    frozen_sha_path = Path(str(frozen_path) + ".sha256")

    # ---- PRE-LAUNCH (a): compute + ASSERT TABLE_ID ----
    if not centroids_path.exists():
        p.error(f"--centroids {args.centroids} not found (needed to compute TABLE_ID)")
    TABLE_ID = table_id(centroids_path, args.n_bins, args.n_runouts)
    table_id_ok = (TABLE_ID == EXPECTED_TABLE_ID)

    run_subdir = args.run_subdir or f"probe_{int(time.time())}"
    run_prefix = f"{args.prefix}/{run_subdir}"
    frozen_key = f"{run_prefix}/frozen/{TABLE_ID}.emdfrzn"
    frozen_sha_key = f"{frozen_key}.sha256"

    # ---- Local source-intactness checks (the pre-upload source must be the production blob) ----
    frozen_bytes_ok = frozen_path.exists() and frozen_path.stat().st_size == EXPECTED_FROZEN_BYTES
    frozen_sha_ok = (
        frozen_sha_path.exists()
        and frozen_sha_path.read_text().strip().split()[0] == EXPECTED_FROZEN_SHA256
    )

    print("=" * 78)
    print("CPU128 PROFILING PROBE — pre-launch plan")
    print("=" * 78)
    print(f"  run_subdir         : {run_subdir}")
    print(f"  instance_type      : {args.instance_type}  arch={args.arch}  "
          f"market={'spot' if args.spot else 'ON-DEMAND'}  keep_alive={args.keep_alive}")
    print(f"  shapes             : {shapes}")
    print(f"  centroids          : {args.centroids}")
    print(f"  computed TABLE_ID  : {TABLE_ID}   (expected {EXPECTED_TABLE_ID})  "
          f"=> {'OK' if table_id_ok else 'MISMATCH — ABORT'}")
    print(f"  frozen blob        : {args.emd_frozen_rel}")
    print(f"     size            : {frozen_path.stat().st_size if frozen_path.exists() else 'MISSING'} bytes "
          f"(expected {EXPECTED_FROZEN_BYTES})  => {'OK' if frozen_bytes_ok else 'BAD'}")
    print(f"     sha256 sidecar  : {'OK ' + EXPECTED_FROZEN_SHA256 if frozen_sha_ok else 'BAD/MISSING'}")
    print(f"  S3 frozen pre-upload keys (NOT tarred — uploaded by the launcher):")
    print(f"     s3://{args.bucket}/{frozen_key}")
    print(f"     s3://{args.bucket}/{frozen_sha_key}")
    print(f"  run S3 prefix      : s3://{args.bucket}/{run_prefix}/probe/")
    print(f"  per-shape config:")
    print(_shape_table(shapes))

    if not table_id_ok:
        msg = (f"TABLE_ID {TABLE_ID} != {EXPECTED_TABLE_ID}: centroids/params drifted, the pre-uploaded "
               f"frozen blob would be STALE for the build. ABORT.")
        if args.dry_run:
            print(f"\n[dry-run] GO/NO-GO #1 FAIL: {msg}")
        else:
            raise SystemExit(msg)

    # ---- Resolve AMI (read-only describe_images; stub on creds failure under --dry-run) ----
    ami = None
    ami_note = ""
    try:
        ec2 = boto3.client("ec2", region_name=args.region)
        ami = resolve_ubuntu_ami(ec2, args.arch)
        print(f"  resolved AMI       : {ami}")
    except Exception as e:  # noqa: BLE001
        ami_note = f"(describe_images failed: {e!r})"
        if args.dry_run:
            ami = "ami-DRYRUN-STUB"
            print(f"  resolved AMI       : {ami}  {ami_note}  [dry-run: stubbed, rest of plan still prints]")
        else:
            raise

    # ---- Render user_data (uses a stub code_key under dry-run so we make no mutating calls) ----
    code_key = f"{args.prefix}/code/cpu128_probe_code.tar.gz"
    awscli_zip = "awscli-exe-linux-aarch64.zip" if args.arch == "arm64" else "awscli-exe-linux-x86_64.zip"
    ud = user_data(
        bucket=args.bucket, prefix=args.prefix, region=args.region, code_key=code_key,
        run_subdir=run_subdir, table_id_hex=TABLE_ID, num_buckets=args.num_buckets,
        warmup_cap=args.warmup_iters, window_iters=args.window_iters, profile_every=args.profile_every,
        shapes=args.shapes, n_runouts=args.n_runouts, n_bins=args.n_bins, keep_alive=args.keep_alive,
        wall_guard_secs=args.wall_guard_secs, awscli_zip=awscli_zip,
    )
    ud_bytes = len(ud.encode())
    ud_b64 = len(base64.b64encode(ud.encode()))
    ud_size_ok = (ud_bytes < 15000 and ud_b64 < 16384)

    run_kwargs = dict(
        ImageId=ami,
        MinCount=1,
        MaxCount=1,
        UserData=base64.b64encode(ud.encode()).decode(),
        IamInstanceProfile={"Name": args.instance_profile},
        MetadataOptions={"HttpTokens": "required", "HttpPutResponseHopLimit": 2},
        # BLOCKING-FIX #1: a 150GB gp3 root. The ~8GB Canonical default fills under the debuginfo=2+strip
        # =none cargo target + 152MB frozen blob + 4x perf.data -> perf.data truncation + S3 upload fail.
        BlockDeviceMappings=[{
            "DeviceName": "/dev/sda1",
            "Ebs": {"VolumeSize": 150, "VolumeType": "gp3", "DeleteOnTermination": True},
        }],
        TagSpecifications=[{
            "ResourceType": "instance",
            "Tags": [
                {"Key": "Name", "Value": f"cpu128-probe-{run_subdir}"},
                {"Key": "Project", "Value": "pokerbot-native-cfr"},
                {"Key": "Run", "Value": run_subdir},
            ],
        }],
        # Durability contract: every artifact is in S3 BEFORE the box stops; even an accidental shutdown
        # -h (or the wall-guard) => terminate.
        InstanceInitiatedShutdownBehavior="terminate",
    )
    if args.spot:
        run_kwargs["InstanceMarketOptions"] = {
            "MarketType": "spot",
            "SpotOptions": {"SpotInstanceType": "one-time", "InstanceInterruptionBehavior": "terminate"},
        }

    print()
    print("-" * 78)
    print("RENDERED user_data + run_kwargs")
    print("-" * 78)
    print(f"  user_data size     : {ud_bytes} bytes plain / {ud_b64} base64  "
          f"=> {'OK (<15000 / <16384)' if ud_size_ok else 'OVER LIMIT — move logic into onbox.sh'}")
    print(f"  BlockDeviceMappings: 150GB gp3 root (/dev/sda1, DeleteOnTermination)")
    print(f"  market             : {'spot' if args.spot else 'ON-DEMAND (no InstanceMarketOptions)'}")
    print(f"  shutdown behavior  : InstanceInitiatedShutdownBehavior=terminate")
    print(f"  WALL_GUARD         : {args.wall_guard_secs}s independent watchdog in user_data")
    print(f"  tar include list   : ({len(tar_include_list())} entries)")
    for arc in tar_include_list():
        print(f"     - {arc}")
    print(f"  NOT tarred (S3 pre-upload): {args.emd_frozen_rel}  (152MB)")

    # ---- GO/NO-GO checklist (13 items) ----
    nodes = []

    def _chk(ok, label):
        nodes.append((ok, label))

    _chk(table_id_ok, "#1 TABLE_ID == 60232de66618c7d1")
    _chk(ud_size_ok, "#2 user_data < 15000 plain AND < 16384 base64")
    _additions = {
        "scripts/cpu128_probe_onbox.sh", "scripts/drive_cpu128_probe.py",
        "scripts/profile_native_trainer_microbench.py",
        "data/abstraction/emd_buckets_160_r400.json",
        "data/abstraction/emd_frozen_flop_turn_160_r400.emdfrzn.sha256",
    }
    tar_set = set(tar_include_list())
    _chk(_additions.issubset(tar_set) and args.emd_frozen_rel not in tar_set,
         "#3 4 tar additions present AND the 152MB .emdfrzn NOT tarred")
    _chk(any(bdm["Ebs"]["VolumeSize"] >= 100 and bdm["Ebs"]["VolumeType"] == "gp3"
             for bdm in run_kwargs["BlockDeviceMappings"]),
         "#4 >=100GB gp3 root (we ship 150GB) [BLOCKING-FIX #1]")
    # #5: on-demand is the design's default (a reclaim can corrupt a measurement window). But the account's
    # 128-vCPU quota lives on SPOT, and the probe has full spot-survival (per-batch S3 upload + IMDSv2
    # reclaim trap that `aws s3 cp --recursive` flushes + the WALL_GUARD), so an explicit --spot run only
    # risks the in-flight shape, never a completed one. PASS if on-demand OR --spot was explicitly chosen.
    _chk(("InstanceMarketOptions" not in run_kwargs) or args.spot,
         "#5 on-demand (no InstanceMarketOptions) OR explicit --spot (spot-survival handled)")
    # The STEP-0 cost cap is now UNCONDITIONAL (armed before apt, independent of keep_alive). Assert the two
    # real guarantees are in the rendered ud: the logind-scheduled absolute floor `shutdown -h +N` (immune to
    # cgroup teardown) AND the pokerbot-wallguard transient unit @ wall_guard_secs. keep_alive no longer gates
    # cost safety — --no-keep-alive is in fact STRICTLY safer (self-terminates on finish instead of at +N).
    _chk(("pokerbot-wallguard" in ud) and ("/sbin/shutdown -h +" in ud)
         and (str(args.wall_guard_secs) in ud) and args.wall_guard_secs == 5400,
         f"#6 STEP-0 cost cap in user_data: shutdown floor + pokerbot-wallguard unit @{args.wall_guard_secs}s "
         f"(unconditional; keep_alive={args.keep_alive}, both bounded) [BLOCKING-FIX #4]")
    _chk(frozen_bytes_ok and frozen_sha_ok,
         "#7 local frozen blob 152,468,448 bytes + sha256 sidecar 344ab2f... intact")

    # #8-#12 are properties of the ON-BOX scripts; assert by static grep of the repo files. Strip comment
    # lines (those whose first non-blank char is '#') so a comment mentioning e.g. "NOT linux-tools-generic"
    # or "Cargo.toml is not edited" cannot trip a negative-substring check — the intent is about the actual
    # COMMAND lines.
    onbox_path = ROOT / "scripts/cpu128_probe_onbox.sh"
    driver_path = ROOT / "scripts/drive_cpu128_probe.py"
    onbox_txt = onbox_path.read_text() if onbox_path.exists() else ""
    driver_txt = driver_path.read_text() if driver_path.exists() else ""
    onbox_cmds = "\n".join(ln for ln in onbox_txt.splitlines() if not ln.lstrip().startswith("#"))
    _chk(("perf stat -e cycles true" in onbox_cmds) and ("perf record" in onbox_cmds)
         and ("PERF_BROKEN" in onbox_cmds),
         "#8 onbox has the perf HARD-GATE (perf stat -e cycles true + 2s perf record self-test -> PERF_BROKEN/degrade) [BLOCKING-FIX #2]")
    _chk(("linux-tools-aws" in onbox_cmds) and ('"linux-tools-$(uname -r)"' in onbox_cmds)
         and ("linux-tools-common" in onbox_cmds) and ("linux-tools-generic" not in onbox_cmds),
         "#9 apt requests linux-tools-aws AND linux-tools-$(uname -r) + linux-tools-common (NOT -generic) [BLOCKING-FIX #2]")
    _chk(("numactl --hardware" in onbox_cmds) and ("NUMA_NODES" in onbox_cmds)
         and ("perf_event_availability" in onbox_cmds),
         "#10 numactl --hardware always runs + NUMA_NODES gate + per-event availability parse [BLOCKING-FIX #3]")
    _chk(("os.fsync" in driver_txt) and ("put_object" in driver_txt)
         and ("try" in driver_txt and "finally" in driver_txt)
         and ("SIGTERM" in driver_txt) and ("--recursive" in onbox_cmds),
         "#11 driver streams+fsync+put_object per batch + try/finally + SIGTERM->raise; onbox spot trap aws s3 cp --recursive")
    # #12: every Cargo.toml reference in a COMMAND line must be the read-only --manifest-path (no in-place
    # edit like `sed -i ... Cargo.toml` / `>> Cargo.toml`), and the v3 codegen flag must be present.
    _cargo_cmd_refs = [ln for ln in onbox_cmds.splitlines() if "Cargo.toml" in ln]
    _cargo_clean = all("--manifest-path tools/battery_native/Cargo.toml" in ln for ln in _cargo_cmd_refs)
    _chk(("target-cpu=x86-64-v3" in onbox_cmds) and _cargo_clean,
         "#12 build keeps -C target-cpu=x86-64-v3 AND Cargo.toml NOT edited (RUSTFLAGS only)")
    _chk(ami is not None and ami != "ami-DRYRUN-STUB",
         f"#13 AWS creds resolve + AMI found (eu-central-1 {args.instance_type} {'spot' if args.spot else 'on-demand'} capacity)")

    print()
    print("-" * 78)
    print("GO / NO-GO CHECKLIST")
    print("-" * 78)
    for ok, label in nodes:
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}")
    n_fail = sum(1 for ok, _ in nodes if not ok)
    print(f"  ---- {len(nodes) - n_fail}/{len(nodes)} PASS ----")

    if args.dry_run:
        print()
        print("DRY-RUN: ZERO mutating boto3 calls made (no upload_file, no put_object, no run_instances).")
        print("To launch for real, drop --dry-run.")
        return

    # =========================== REAL LAUNCH (mutating) ===========================
    if n_fail:
        raise SystemExit(f"{n_fail} go/no-go check(s) FAILED — refusing to launch. Fix and retry.")

    s3 = boto3.client("s3", region_name=args.region)

    # (b) PRE-UPLOAD the frozen blob + sidecar so the box CACHE_HITs (never recomputes the ~16.5M spots).
    print(f"Pre-uploading frozen blob (152MB) -> s3://{args.bucket}/{frozen_key} ...", flush=True)
    s3.upload_file(str(frozen_path), args.bucket, frozen_key)
    s3.upload_file(str(frozen_sha_path), args.bucket, frozen_sha_key)
    print("  frozen + sidecar uploaded.", flush=True)

    # (c) build + upload the code tarball (this sets the REAL code_key; user_data already references it).
    code_key = build_and_upload_code(s3, args.bucket, args.prefix, crate_src=args.crate_src)
    # user_data was rendered with the same code_key string (deterministic), so re-render is unnecessary;
    # but re-render to be safe in case crate_src changed the path (it does not — key is fixed).

    # (d) launch ONE instance (no babysitter).
    run_kwargs.pop("InstanceType", None)
    instance_types = (
        _spotpick_chain(args.instance_type, args.arch, args.spotpick_n)
        if (args.spot and args.spotpick) else [args.instance_type]
    )
    iid, used_type = _launch_with_fallback(ec2, run_kwargs, instance_types=instance_types, spot=args.spot)
    print(f"Launched {iid} as {used_type} ({'spot' if args.spot else 'on-demand'})", flush=True)
    print(f"Run S3: s3://{args.bucket}/{run_prefix}/probe/", flush=True)

    # (e) AUTO-WIRE the external cost-safety poller as a DETACHED process that survives this launcher's exit
    # (the adversarial gate required an AUTOMATIC backstop, not an operator-remembered one). It polls 5 min
    # after launch then every 15 min, and TERMINATES the box on any failure signature or a hard timeout — so
    # the on-box wall-guard is never the SOLE guarantee. Defense in depth; failure to spawn is non-fatal.
    watch_log = ROOT / "artifacts" / f"probe_watch_{run_subdir}.log"
    watch_cmd = [sys.executable, str(ROOT / "scripts" / "_probe_watch.py"),
                 "--instance-id", iid, "--region", args.region,
                 "--s3-prefix", f"s3://{args.bucket}/{run_prefix}/probe/"]
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
        print(f"WARN: could not auto-start poller ({e!r}) — run it MANUALLY:\n  "
              f"python scripts/_probe_watch.py --instance-id {iid} --region {args.region} "
              f"--s3-prefix s3://{args.bucket}/{run_prefix}/probe/", flush=True)

    print(f"Monitor: aws ec2 get-console-output --instance-id {iid} --region {args.region} --latest", flush=True)
    print(f"Confirm artifacts: aws s3 ls s3://{args.bucket}/{run_prefix}/probe/ --recursive", flush=True)
    print(f"Terminate (after PROBE_DONE + S3 confirmed): "
          f"aws ec2 terminate-instances --instance-ids {iid} --region {args.region}", flush=True)


if __name__ == "__main__":
    main()
