#!/usr/bin/env python3
"""
Launch a SINGLE EC2 spot instance to train the NATIVE 6-max NLHE CFR+ blueprint (Workstream F).

Mirrors scripts/launch_sd_cfr_training_spot.py (spot launch + babysitter/resume pattern), but the
workload is the Rust ``NlheCFRTrainer`` (tools/battery_native/src/trainer.rs) rather than the JAX SD-CFR
stack. The box:
  1. installs Rust + a Python with PyO3 headers,
  2. pulls the repo tarball from S3 and ``cargo build --release --features python --lib`` the crate
     (the .so is the importable ``battery_native`` extension exposing ``NlheCFRTrainer``),
  3. runs a tiny embedded driver that constructs ``NlheCFRTrainer(num_buckets, depth_edges,
     n_deals_per_iter)``, RESUMES from the latest S3 checkpoint if one exists, then trains to
     ``--iters`` total, calling ``save_bytes()`` -> boto3 upload every ``--checkpoint-every`` iters,
  4. on spot preemption (or normal exit) the latest checkpoint is on S3, so a replacement box resumes.

Default is CPU-heavy (c6i.32xlarge, 128 vCPU) — the trainer is rayon-parallel across deals. Cost target
(plan): 5000 iters x ~200 deals ~ $12-40 spot, well under the Pluribus blueprint's <$150.

Reuses the same AWS identity + S3 prefix conventions as the SD-CFR launcher.
"""

from __future__ import annotations

import argparse
import base64
import io
import sys
import tarfile
import time
from pathlib import Path

import boto3
from botocore.exceptions import ClientError

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BUCKET = "video-processing-6789"
DEFAULT_PREFIX = "pokerbot/prod6000/native_cfr"  # within the IAM-allowed prod6000/* prefix
DEFAULT_REGION = "eu-central-1"
DEFAULT_PROFILE = "pokerbot-labelgen-worker"

# Ship the crate (its only build inputs) + the python serve/eval chain needed for offline validation.
# The trainer builds ON core.rs's public API; no JAX needed to TRAIN (only to LBR-validate, optional).
_INCLUDE = [
    "tools/battery_native",
    "pokerbot/search/native_solver.py",
    "pokerbot/search/tabular_blueprint_adapter.py",
]
_EXCLUDE_DIRS = ("__pycache__", ".git", "target")  # never ship the local Rust build dir (huge + stale)


def _tar_filter(ti: tarfile.TarInfo):
    parts = ti.name.split("/")
    if any(p in _EXCLUDE_DIRS for p in parts) or ti.name.endswith(".pyc"):
        return None
    return ti


def build_and_upload_code(s3, bucket: str, prefix: str, crate_src: str = "tools/battery_native",
                          emd_frozen: str = "", river_ochs: str = "") -> str:
    # (src, arcname) pairs. The crate may come from a FROZEN COPY (``crate_src``) but is ALWAYS uploaded
    # under "tools/battery_native" so the box builds it at the expected manifest path — this isolates a
    # running blueprint job from concurrent edits to the live crate (e.g. a parallel search optimization).
    includes = [
        (crate_src, "tools/battery_native"),
        ("pokerbot/search/native_solver.py", "pokerbot/search/native_solver.py"),
        ("pokerbot/search/tabular_blueprint_adapter.py", "pokerbot/search/tabular_blueprint_adapter.py"),
    ]
    # Ship the shared frozen EMD lookup so the box's trainer keys postflop buckets from the SAME bytes the
    # serve encoder will read (bit-identical train/serve). Uploaded under its given relative path.
    if emd_frozen:
        includes.append((emd_frozen, emd_frozen))
    # Blueprint-v2 river OCHS table (~10 KB): shipped so the box's trainer keys the RIVER with the
    # SAME nearest-centroid table the serve encoder will read (POKERBOT_RIVER_OCHS).
    if river_ochs:
        includes.append((river_ochs, river_ochs))
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for src, arc in includes:
            p = ROOT / src
            if p.exists():
                tar.add(p, arcname=arc, filter=_tar_filter)
    buf.seek(0)
    key = f"{prefix}/code/native_cfr_code.tar.gz"
    s3.put_object(Bucket=bucket, Key=key, Body=buf.read())
    print(f"Uploaded code tarball -> s3://{bucket}/{key} ({buf.tell()} bytes; crate_src={crate_src})", flush=True)
    return key


def resolve_ubuntu_ami(ec2, arch: str) -> str:
    name_arch = "amd64" if arch == "x86_64" else "arm64"
    resp = ec2.describe_images(
        Owners=["099720109477"],  # Canonical
        Filters=[
            {"Name": "name", "Values": [f"ubuntu/images/hvm-ssd/ubuntu-jammy-22.04-{name_arch}-server-*"]},
            {"Name": "state", "Values": ["available"]},
            {"Name": "architecture", "Values": [arch]},
        ],
    )
    imgs = sorted(resp["Images"], key=lambda i: i["CreationDate"], reverse=True)
    if not imgs:
        raise SystemExit(f"no {arch} Ubuntu 22.04 AMI found in region")
    return imgs[0]["ImageId"]


# Spot-capacity errors worth trying the NEXT instance type for (vs a hard config/quota error, which
# every type shares, so retrying down the chain is pointless).
_CAPACITY_ERROR_CODES = frozenset({
    "InsufficientInstanceCapacity",
    "SpotMaxPriceTooLow",
    "InstanceLimitExceeded",
    "MaxSpotInstanceCountExceeded",
    "Unsupported",  # instance type not offered in this AZ/region
})


def _spotpick_chain(instance_type: str, arch: str, n: int) -> list[str]:
    """Build a capacity-diverse fallback chain (one type per family) starting at the requested type's
    vCPU count, via the shared spotpick lib (.). Returns [instance_type] alone if
    spotpick or the catalog is unavailable, so --spotpick degrades gracefully to single-type behavior.
    The head of the chain is forced to be the explicitly-requested ``instance_type`` (so the default
    c6i.32xlarge is still tried FIRST), with the spotpick alternatives appended de-duplicated."""
    try:
        sys.path.insert(0, r".")
        import spotpick  # noqa: E402
    except Exception as e:  # noqa: BLE001
        print(f"[spotpick] unavailable ({e!r}) — falling back to single type {instance_type}", flush=True)
        return [instance_type]
    # Derive the vCPU target from the requested type via the catalog (so the chain matches its size).
    try:
        cat = {it.name: it for it in spotpick.load()}
        vcpus = cat[instance_type].vcpus if instance_type in cat else None
        if vcpus is None:
            print(f"[spotpick] {instance_type} not in catalog — single type only", flush=True)
            return [instance_type]
        alts = spotpick.chain(vcpus=vcpus, arch=arch, n=max(n, 1))
    except Exception as e:  # noqa: BLE001
        print(f"[spotpick] chain() failed ({e!r}) — single type {instance_type}", flush=True)
        return [instance_type]
    chain = [instance_type] + [a for a in alts if a != instance_type]
    return chain[:max(n, 1)]


def _launch_with_fallback(ec2, run_kwargs: dict, *, instance_types: list[str], spot: bool):
    """Try ``run_instances`` down ``instance_types`` until one succeeds (capacity errors advance to the
    next type; any other error is fatal). Returns (instance_id, instance_type_used). The caller bakes
    UserData/tags BEFORE this — only ``InstanceType`` changes per attempt, so a fallback box is otherwise
    byte-identical (same run-subdir => resumes the same S3 checkpoint)."""
    last_err = None
    for it in instance_types:
        kw = dict(run_kwargs, InstanceType=it)
        try:
            resp = ec2.run_instances(**kw)
            return resp["Instances"][0]["InstanceId"], it
        except ClientError as e:  # noqa: PERF203
            code = e.response.get("Error", {}).get("Code", "")
            last_err = e
            if code in _CAPACITY_ERROR_CODES and len(instance_types) > 1:
                print(f"[spotpick] {it}: {code} — trying next type", flush=True)
                continue
            raise
    raise SystemExit(f"all instance types exhausted ({instance_types}); last error: {last_err}")


def user_data_stub(*, bucket, region, run_prefix, bootstrap_key, wall_guard_secs,
                   awscli_zip="awscli-exe-linux-x86_64.zip") -> str:
    """Tiny cloud-init user-data (EC2 hard-caps user-data at 16384 bytes; the full bootstrap
    outgrew it once river-OCHS + resume logic landed). Contains ONLY: (1) STEP-0 dead-man cost
    cap — inline VERBATIM so the box is cost-bounded even if the S3 fetch below fails; (2) AWS
    CLI v2 install + fetch of the real bootstrap from S3 + exec. Bootstrap fetch failure =>
    immediate shutdown (never an idle box)."""
    floor_min = wall_guard_secs // 60 + 5   # absolute logind floor, 5 min after the wall-guard unit
    return f"""#!/bin/bash
set -uxo pipefail
export HOME="${{HOME:-/root}}"   # cloud-init runs as root with HOME UNSET; under set -u a bare $HOME is FATAL
export DEBIAN_FRONTEND=noninteractive AWS_DEFAULT_REGION={region}
mkdir -p /opt/pokerbot

# ===================== STEP 0 -- DEAD-MAN COST CAP, ARMED FIRST (before apt/fetch/anything) =====================
# TWO independent on-box terminators so the box CANNOT outlive ~{floor_min} min no matter what fails below
# (the documented 3h20m idle-bill fix; ported from launch_cpu128_probe_spot.py). These MUST be the first real
# commands so they survive a crash anywhere downstream (incl. a failed bootstrap fetch) AND a dead
# babysitter / powered-off laptop:
#  0a) ABSOLUTE FLOOR: `shutdown -h +{floor_min}` is scheduled by systemd/logind ITSELF (not a child/sleep),
#      immune to cloud-init cgroup teardown, needs no packages.
#  0b) WALL-GUARD UNIT: a systemd-run transient unit (own cgroup; NO --oneshot, whose 90s
#      DefaultTimeoutStartSec would reap the sleep) that at {wall_guard_secs}s sweeps the train log to S3 then
#      shuts down (~5 min before 0a).
/sbin/shutdown -h +{floor_min} "pokerbot native-cfr wall-guard absolute cost cap" || true
systemd-run --unit=pokerbot-wallguard --collect -p TimeoutStartSec=infinity --no-block /bin/bash -c 'sleep {wall_guard_secs} ; sync ; timeout 120 aws s3 cp /var/log/native-cfr-train.log s3://{bucket}/{run_prefix}/logs/train.WALLGUARD.log --region {region} 2>/dev/null || true ; /sbin/shutdown -h now' || true
echo "WALL_GUARD armed: absolute shutdown +{floor_min}min floor + pokerbot-wallguard unit at {wall_guard_secs}s"

# ===================== STEP 1 -- fetch + exec the real bootstrap from S3 =====================
# AWS CLI v2 (NOT apt awscli v1 — the documented botocore-breakage lesson lives in the bootstrap).
apt-get update -qq
apt-get install -y -qq curl unzip
curl -fsSL "https://awscli.amazonaws.com/{awscli_zip}" -o /tmp/awscliv2.zip
unzip -q /tmp/awscliv2.zip -d /tmp
/tmp/aws/install --bin-dir /usr/local/bin --install-dir /usr/local/aws-cli --update
hash -r
aws s3 cp s3://{bucket}/{bootstrap_key} /tmp/bootstrap.sh --region {region} || {{ echo "FATAL: bootstrap fetch failed" ; sync ; /sbin/shutdown -h now ; exit 1 ; }}
bash /tmp/bootstrap.sh
"""


def bootstrap_body(*, bucket, prefix, region, code_key, run_subdir, iters, n_deals, num_buckets,
                   depth_edges, checkpoint_every, keep_alive=False, emd_frozen_rel="", emd_sha="",
                   river_ochs_rel="", river_ochs_sha="", preflop_hist_key=False,
                   work_steal=False, dcfr=False, dcfr_prune_threshold=-300000.0, dcfr_explore_frac=0.05,
                   wall_guard_secs=21600, awscli_zip="awscli-exe-linux-x86_64.zip") -> str:
    """The full on-box bootstrap (was inline user-data until it exceeded EC2's 16KB cap).
    Uploaded to S3 by main() and exec'd by user_data_stub(). Content is otherwise the proven
    launcher script, byte-equivalent from `apt-get update` onward — STEP 0 (cost cap) lives in
    the stub and is ALREADY ARMED when this runs. The AWS CLI v2 re-install below is idempotent
    (--update) and kept so this script stays self-sufficient if ever run standalone."""
    run_prefix = f"{prefix}/{run_subdir}"
    final = "echo BOOTSTRAP_DONE_KEEP_ALIVE" if keep_alive else "shutdown -h now"
    depth_csv = ",".join(str(float(x)) for x in (depth_edges or []))
    hist_flag = "1" if preflop_hist_key else "0"
    # No output redirect (cloud-init captures stdout to /var/log/cloud-init-output.log, visible via
    # get-console-output) — same lesson the SD-CFR launcher documents.
    return f"""#!/bin/bash
set -uxo pipefail
export HOME="${{HOME:-/root}}"   # may run detached from the stub's shell; same set -u guard
export DEBIAN_FRONTEND=noninteractive AWS_DEFAULT_REGION={region}
mkdir -p /opt/pokerbot

apt-get update -qq
apt-get install -y -qq python3-pip python3-dev gzip curl unzip build-essential
# AWS CLI v2 (NOT apt awscli v1): self-contained (bundled botocore under /usr/local/aws-cli), so the driver's
# `pip install boto3` below can bump the SYSTEM botocore WITHOUT breaking the CLI. The apt awscli v1 broke
# exactly this way: pip boto3 -> newer botocore -> awscli v1 `KeyError: 'opsworkscm'` on every call.
curl -fsSL "https://awscli.amazonaws.com/{awscli_zip}" -o /tmp/awscliv2.zip
unzip -q /tmp/awscliv2.zip -d /tmp
/tmp/aws/install --bin-dir /usr/local/bin --install-dir /usr/local/aws-cli --update
hash -r
aws --version   # MUST be aws-cli/2.x; if this dies, STEP-0 still bounds the box
# Rust toolchain (rustup, stable) for the native build.
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal
export PATH="$HOME/.cargo/bin:$PATH"
rustc --version

mkdir -p /opt/pokerbot
aws s3 cp s3://{bucket}/{code_key} /tmp/code.tar.gz --region {region}
tar -xzf /tmp/code.tar.gz -C /opt/pokerbot
cd /opt/pokerbot

# boto3 for S3 checkpoint up/download from the driver.
python3 -m pip install --disable-pip-version-check --no-input boto3 numpy
# Build the native extension (.so) exposing NlheCFRTrainer. PyO3 abi3 -> one .so for the box's python.
export PYO3_PYTHON=$(python3 -c 'import sys;print(sys.executable)')
# target-cpu=x86-64-v3 (AVX2 + FMA, NO AVX512): autovectorizes the CFR+ [f64;11]/[bool;11] hot-loop
# kernels (regret_match, EV dot-product, strat_sum AXPY, CFR+ floor, side-pot award) to 4-wide and lets
# the hand-written SIMD emit FMA instead of SSE2-baseline scalar. It is NOT -ffast-math (no float
# reassociation), so the exact-int byte-identity goldens stay green. We deliberately do NOT use
# target-cpu=native: on an AVX512 box `native` enables AVX512, whose license-based downclock is a NET
# REGRESSION for this pointer-chasing/hashmap-bound hot path (~-30% it/s measured). v3 = AVX2+FMA with
# no AVX512 => no downclock, measured small net gain. The .so is built AND run on this same box and
# never ships to a different-CPU host, so AVX2/FMA is always present (no SIGILL). Set via RUSTFLAGS on
# the build line so it applies even though the tarball include-list omits tools/battery_native/.cargo/.
RUSTFLAGS="-C target-cpu=x86-64-v3" cargo build --release --features python --lib --manifest-path tools/battery_native/Cargo.toml
# The cdylib is target/release/libbattery_native.so; import it as `battery_native`.
SO=$(ls tools/battery_native/target/release/libbattery_native.so 2>/dev/null || ls tools/battery_native/target/release/battery_native.so 2>/dev/null)
cp "$SO" /opt/pokerbot/battery_native.so
ls -la /opt/pokerbot/battery_native.so

# RAYON FANOUT: pin to PHYSICAL cores, NOT logical (SMT). The cpu128 probe (2026-06-28,
# artifacts/cpu128_probe_verdict_20260628.md) measured the trainer is memory-latency-bound — cores sit ~69%
# idle inside the parallel traverse — so the 2nd hyperthread per core CONTENDS for shared L1/L2/load-store
# ports rather than adding compute. At identical fanout, 64 threads (1/phys core) beat 128 logical by
# +12.6% deals/s / +18.5% nodes/s at LOWER CPU. PHYS_CORES = unique (socket,core) pairs (correct on 2-socket
# boxes where core ids repeat per socket). Override by exporting RAYON_NUM_THREADS before launch.
PHYS_CORES=$(lscpu -p=SOCKET,CORE | grep -v '^#' | sort -u | wc -l)
if [ -z "${{RAYON_NUM_THREADS:-}}" ]; then export RAYON_NUM_THREADS=$PHYS_CORES; fi
echo "RAYON_NUM_THREADS=$RAYON_NUM_THREADS (physical cores=$PHYS_CORES, nproc=$(nproc); SMT-off per cpu128 probe verdict)"
export PYTHONPATH=/opt/pokerbot RUST_BACKTRACE=1
# Blueprint-v3 preflop HISTORY key. When 1, fingerprint.rs (env-gated, read once per process)
# appends b"H"+[n_limps,n_raises,opener_pos+1] to every PREFLOP infoset key, so strategically
# distinct preflop histories stop aliasing (artifacts/preflop_mispricing_diagnosis.md). The
# checkpoint MUST then be served with POKERBOT_PREFLOP_HIST_KEY=1 (ship_blueprint --preflop-hist-key
# stamps + guards it). "0" is byte-identical to the shipped key space.
export POKERBOT_PREFLOP_HIST_KEY={hist_flag}
echo "PREFLOP_HIST_KEY=$POKERBOT_PREFLOP_HIST_KEY"

# S2: verify the shipped EMD frozen blob matches its known sha256 BEFORE training. A same-K CORRUPTED blob
# (truncated S3 round-trip) -> wrong potential-aware buckets -> a broken blueprint that the convergence-stop
# (abstraction-agnostic) cannot catch and only the final ranking gate would; the ctor's num_buckets==K check
# does NOT detect content corruption. Empty EMD_SHA (made-hand path) -> skipped. A mismatch terminates the box.
EMD_SHA="{emd_sha}"
EMD_REL="{emd_frozen_rel}"
if [ -n "$EMD_SHA" ] && [ -n "$EMD_REL" ]; then
  ACT=$(sha256sum "/opt/pokerbot/$EMD_REL" | awk '{{print $1}}')
  if [ "$ACT" != "$EMD_SHA" ]; then
    echo "FATAL: EMD blob sha mismatch expect=$EMD_SHA actual=$ACT -- aborting + terminating"; sync; /sbin/shutdown -h now; exit 1
  fi
  echo "EMD blob sha256 OK: $ACT"
fi

# Blueprint-v2: verify the shipped river-OCHS table sha256 (same corruption rationale as the EMD blob;
# a corrupted table re-maps every river bucket -> a silently broken blueprint). Empty => skipped.
ROCHS_SHA="{river_ochs_sha}"
ROCHS_REL="{river_ochs_rel}"
if [ -n "$ROCHS_SHA" ] && [ -n "$ROCHS_REL" ]; then
  ACT=$(sha256sum "/opt/pokerbot/$ROCHS_REL" | awk '{{print $1}}')
  if [ "$ACT" != "$ROCHS_SHA" ]; then
    echo "FATAL: river-OCHS table sha mismatch expect=$ROCHS_SHA actual=$ACT -- aborting + terminating"; sync; /sbin/shutdown -h now; exit 1
  fi
  echo "river-OCHS table sha256 OK: $ACT"
fi

cat > /opt/pokerbot/drive_native_cfr.py <<'PYEOF'
import os, sys, time, traceback
import boto3
import battery_native as bn

BUCKET = "{bucket}"
RUN_PREFIX = "{run_prefix}"
REGION = "{region}"
ITERS = {iters}
N_DEALS = {n_deals}
NUM_BUCKETS = {num_buckets}
CKPT_EVERY = {checkpoint_every}
DEPTH_EDGES = [float(x) for x in "{depth_csv}".split(",") if x]
# Optional shared frozen EMD (hole,board)->bucket lookup (relative to /opt/pokerbot). When set, the
# trainer keys postflop flop/turn buckets from this table (river/preflop stay made-hand) — the SAME blob
# the serve encoder reads, so train==serve is bit-identical. Empty => the deployed made-hand path.
EMD_FROZEN_REL = "{emd_frozen_rel}"
EMD_FROZEN_PATH = ("/opt/pokerbot/" + EMD_FROZEN_REL) if EMD_FROZEN_REL else None
# Blueprint-v2 river OCHS table: when set, the trainer keys the RIVER with the OCHS nearest-centroid
# bucket (flop/turn stay on the EMD frozen blob; preflop unchanged). The blueprint MUST then be served
# with POKERBOT_RIVER_OCHS pointing at this same table (stamp-guarded). Empty => made-hand mirror river.
RIVER_OCHS_REL = "{river_ochs_rel}"
RIVER_OCHS_PATH = ("/opt/pokerbot/" + RIVER_OCHS_REL) if RIVER_OCHS_REL else None
# Throughput/convergence levers (gated default-OFF; both verified). work_steal: atomic deal-counter +
# rayon::broadcast over per-thread tables -> closes the deep-deal load imbalance (+33% deals/s @64c,
# measured). dcfr: signed-regret Discounted-CFR + active Regret-Based Pruning (~2x fewer iters to
# convergence on Leduc exact exploitability).
WORK_STEAL = {work_steal}
DCFR = {dcfr}
DCFR_PRUNE_THRESHOLD = {dcfr_prune_threshold}
DCFR_EXPLORE_FRAC = {dcfr_explore_frac}

s3 = boto3.client("s3", region_name=REGION)
LATEST_KEY = RUN_PREFIX + "/checkpoints/latest.nlcfr"

def _download_latest():
    try:
        obj = s3.get_object(Bucket=BUCKET, Key=LATEST_KEY)
        return obj["Body"].read()
    except Exception:
        return None

def _upload(blob, iters_done):
    # versioned per-iter checkpoint + a `latest` pointer (atomic on S3: a single PutObject).
    key = RUN_PREFIX + f"/checkpoints/iter_{{iters_done:06d}}.nlcfr"
    s3.put_object(Bucket=BUCKET, Key=key, Body=blob)
    s3.put_object(Bucket=BUCKET, Key=LATEST_KEY, Body=blob)
    print(f"CKPT iters_done={{iters_done}} -> s3://{{BUCKET}}/{{key}} ({{len(blob)}} bytes)", flush=True)

def main():
    # full ctor: (num_buckets, depth_edges, n_deals, n_seats, prune_threshold, fast_merge, emd_frozen_path,
    #             batch_deals, hogwild, hogwild_capacity, work_steal, dcfr, dcfr_prune_threshold, dcfr_explore_frac)
    if EMD_FROZEN_PATH:
        t = bn.NlheCFRTrainer(NUM_BUCKETS, DEPTH_EDGES, N_DEALS, 6, float("-inf"), False, EMD_FROZEN_PATH,
                              0, False, 1 << 21, WORK_STEAL, DCFR, DCFR_PRUNE_THRESHOLD, DCFR_EXPLORE_FRAC,
                              river_ochs_path=RIVER_OCHS_PATH)
        print(f"EMD frozen lookup ACTIVE: {{EMD_FROZEN_PATH}} stamp={{t.emd_stamp()}}", flush=True)
    else:
        t = bn.NlheCFRTrainer(NUM_BUCKETS, DEPTH_EDGES, N_DEALS, 6, float("-inf"), False, None,
                              0, False, 1 << 21, WORK_STEAL, DCFR, DCFR_PRUNE_THRESHOLD, DCFR_EXPLORE_FRAC,
                              river_ochs_path=RIVER_OCHS_PATH)
    if RIVER_OCHS_PATH:
        print(f"river-OCHS ACTIVE: {{RIVER_OCHS_PATH}} stamp={{t.river_ochs_stamp()}}", flush=True)
    print(f"LEVERS work_steal={{WORK_STEAL}} dcfr={{DCFR}} "
          f"(prune={{DCFR_PRUNE_THRESHOLD}} explore={{DCFR_EXPLORE_FRAC}}) hogwild={{t.is_hogwild()}}", flush=True)
    blob = _download_latest()
    if blob is not None and t.load_bytes(blob):
        print(f"RESUMED from latest checkpoint at iters_done={{t.iters_done()}}", flush=True)
    else:
        print("FRESH start (no resumable checkpoint)", flush=True)
    start = t.iters_done()
    print(f"TRAIN target={{ITERS}} from={{start}} n_deals={{N_DEALS}} num_buckets={{NUM_BUCKETS}} "
          f"depth_edges={{DEPTH_EDGES}}", flush=True)
    t0 = time.time()
    while t.iters_done() < ITERS:
        nxt = min(t.iters_done() + CKPT_EVERY, ITERS)
        step = nxt - t.iters_done()
        t.train(step)
        _upload(t.save_bytes(), t.iters_done())
        el = time.time() - t0
        done = t.iters_done() - start
        rate = done / el if el > 0 else 0.0
        print(f"PROGRESS iters_done={{t.iters_done()}}/{{ITERS}} infosets={{t.n_infosets()}} "
              f"elapsed={{el:.0f}}s rate={{rate:.3f}} it/s", flush=True)
    print(f"DONE iters_done={{t.iters_done()}} infosets={{t.n_infosets()}}", flush=True)

if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
PYEOF

# Background SPOT-INTERRUPT handler: poll IMDSv2 (token required) for the reclaim notice; on notice send
# SIGTERM so the driver's last checkpoint is already on S3 (it checkpoints every {checkpoint_every} iters).
cat > /tmp/spot_watch.sh <<'SWEOF'
TOK=$(curl -s -X PUT "http://169.254.169.254/latest/api/token" -H "X-aws-ec2-metadata-token-ttl-seconds: 21600")
while true; do
  C=$(curl -s -o /dev/null -w "%{{http_code}}" -H "X-aws-ec2-metadata-token: $TOK" http://169.254.169.254/latest/meta-data/spot/instance-action)
  if [ "$C" = "200" ]; then echo "SPOT_RECLAIM_NOTICE"; pkill -TERM -f drive_native_cfr.py; break; fi
  sleep 5
done
SWEOF
nohup bash /tmp/spot_watch.sh >/var/log/spot_watch.log 2>&1 &

python3 /opt/pokerbot/drive_native_cfr.py 2>&1 | tee /var/log/native-cfr-train.log
echo "train exit ${{PIPESTATUS[0]}}"

# Upload logs via boto3.
python3 - <<'LOGEOF'
import boto3
s3 = boto3.client("s3", region_name="{region}")
for src, key in [("/var/log/cloud-init-output.log", "{run_prefix}/logs/cloud-init-output.log"),
                 ("/var/log/native-cfr-train.log", "{run_prefix}/logs/train.log")]:
    try:
        s3.upload_file(src, "{bucket}", key); print("uploaded", key)
    except Exception as e:
        print("log upload fail", src, e)
LOGEOF
{final}
"""


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--instance-type", default="c6i.32xlarge",
                   help="c6i.32xlarge=full (128 vCPU); c6i.4xlarge=cheap validate")
    p.add_argument("--arch", default="x86_64", choices=["x86_64", "arm64"])
    p.add_argument("--iters", type=int, default=5000, help="total CFR+ iterations (target)")
    p.add_argument("--n-deals", type=int, default=200, help="deals fanned out per iteration (rayon)")
    p.add_argument("--num-buckets", type=int, default=200,
                   help="postflop card-abstraction buckets (MUST equal serve's num_buckets)")
    p.add_argument("--depth-edges", default="",
                   help="comma-sep depth-bucket edges for a depth-aware preflop key (e.g. '50,90,140'); "
                        "empty = non-depth blueprint")
    p.add_argument("--checkpoint-every", type=int, default=500,
                   help="checkpoint to S3 every N iters (preemption-bounded loss)")
    p.add_argument("--max-hours", type=float, default=6.0,
                   help="ON-BOX absolute dead-man cost cap (hours): the box self-terminates after this no "
                        "matter what (logind floor + systemd wall-guard), independent of the babysitter/PC. "
                        "Set comfortably above the expected run (~2h) but below the babysitter's 12h cap.")
    p.add_argument("--run-subdir", default=None, help="S3 subdir for this run (default: run_<ts>)")
    p.add_argument("--bucket", default=DEFAULT_BUCKET)
    p.add_argument("--prefix", default=DEFAULT_PREFIX)
    p.add_argument("--region", default=DEFAULT_REGION)
    p.add_argument("--instance-profile", default=DEFAULT_PROFILE)
    p.add_argument("--on-demand", action="store_true", help="use on-demand instead of spot")
    p.add_argument("--spotpick", action="store_true",
                   help="on a spot-capacity error, retry down a capacity-diverse fallback chain "
                        "(spotpick.chain at the same vCPU count) instead of hard-failing. Default OFF = "
                        "single --instance-type only (unchanged behavior).")
    p.add_argument("--spotpick-n", type=int, default=5,
                   help="max instance types in the --spotpick fallback chain (incl. the primary)")
    p.add_argument("--keep-alive", action="store_true",
                   help="don't shut down on finish (keep box for get-console-output inspection)")
    p.add_argument("--reuse-code", action="store_true",
                   help="skip the code upload; REUSE the existing S3 code tarball. The babysitter passes "
                        "this on RELAUNCHES so a locally-changing working tree (e.g. a concurrent crate "
                        "optimization) never re-uploads mid-edit and breaks a resumed box.")
    p.add_argument("--crate-src", default="tools/battery_native",
                   help="path to the crate source to upload (always as tools/battery_native). Point at a "
                        "FROZEN COPY to isolate a running job from concurrent edits to the live crate; "
                        "falls back to tools/battery_native if the path is missing.")
    p.add_argument("--emd-frozen", default="",
                   help="path (repo-relative) to a SHARED frozen EMD (hole,board)->bucket lookup "
                        "(precompute_emd_frozen_lookup.py output). When set, the trainer keys postflop "
                        "flop/turn buckets from it (river/preflop stay made-hand) — the SAME blob the serve "
                        "encoder reads (POKERBOT_POSTFLOP_EMD_FROZEN), so train==serve is bit-identical. "
                        "Empty = the deployed made-hand path. When set, --num-buckets MUST equal the blob's K "
                        "(the ctor enforces num_buckets==lookup.K; e.g. 160 for the K160 blob); the EMD "
                        "blueprint is a self-consistent K (serve must use the same K + EMD). Made-hand path "
                        "(empty) uses --num-buckets=200 to match serve's POSTFLOP_NUM_BUCKETS.")
    p.add_argument("--river-ochs", default="",
                   help="path (repo-relative) to the blueprint-v2 river-OCHS table (.rochs, from "
                        "scripts/blueprint2/fit_river_ochs.py). When set, the trainer keys the RIVER with "
                        "the OCHS nearest-centroid bucket (flop/turn stay on --emd-frozen; preflop "
                        "unchanged) and the checkpoint MUST be served with POKERBOT_RIVER_OCHS pointing at "
                        "the SAME table (stamp-guarded; a skew HITS wrong-semantics river rows). "
                        "Empty = the deployed made-hand mirror river.")
    p.add_argument("--preflop-hist-key", action="store_true",
                   help="blueprint-v3 (bp3_prefhist_1): export POKERBOT_PREFLOP_HIST_KEY=1 on the box so "
                        "the trainer keys PREFLOP infosets with the extra b'H'+[n_limps,n_raises,"
                        "opener_pos] bytes (fingerprint.rs, env-gated) — the fix for the preflop "
                        "betting-state ALIASING root cause (artifacts/preflop_mispricing_diagnosis.md). "
                        "The checkpoint MUST then be served with the same env: ship it via "
                        "ship_blueprint.py --preflop-hist-key (stamp + two-sided serve guard).")
    p.add_argument("--work-steal", action="store_true",
                   help="atomic deal-counter + rayon::broadcast scheduler (closes the deep-deal load "
                        "imbalance; +33%% deals/s @64c measured). Statistically-equivalent, default OFF.")
    p.add_argument("--dcfr", action="store_true",
                   help="signed-regret Discounted-CFR + active Regret-Based Pruning (~2x fewer iters to "
                        "convergence on Leduc). 6-max convergence being certified; default OFF.")
    p.add_argument("--dcfr-prune-threshold", type=float, default=-300000.0,
                   help="RBP floor (bb regret units) used when --dcfr is on")
    p.add_argument("--dcfr-explore-frac", type=float, default=0.05,
                   help="fraction of iters that explore all actions (no pruning) under --dcfr; Pluribus ~0.05")
    args = p.parse_args()

    if args.num_buckets <= 0:
        p.error("--num-buckets must be > 0 (serve uses 200)")
    if args.emd_frozen and not (ROOT / args.emd_frozen).exists():
        p.error(f"--emd-frozen {args.emd_frozen} not found (run precompute_emd_frozen_lookup.py first)")
    if args.river_ochs and not (ROOT / args.river_ochs).exists():
        p.error(f"--river-ochs {args.river_ochs} not found (run scripts/blueprint2/fit_river_ochs.py first)")
    depth_edges = [float(x) for x in args.depth_edges.split(",") if x.strip()]

    s3 = boto3.client("s3", region_name=args.region)
    ec2 = boto3.client("ec2", region_name=args.region)

    run_subdir = args.run_subdir or f"run_{int(time.time())}"
    crate_src = args.crate_src if (ROOT / args.crate_src).exists() else "tools/battery_native"
    if crate_src != args.crate_src:
        print(f"WARN: --crate-src {args.crate_src} not found; falling back to {crate_src}", flush=True)
    if args.reuse_code:
        code_key = f"{args.prefix}/code/native_cfr_code.tar.gz"
        print(f"Reusing existing code tarball -> s3://{args.bucket}/{code_key} (--reuse-code)", flush=True)
    else:
        code_key = build_and_upload_code(s3, args.bucket, args.prefix, crate_src=crate_src,
                                         emd_frozen=args.emd_frozen, river_ochs=args.river_ochs)
    ami = resolve_ubuntu_ami(ec2, args.arch)
    print(f"AMI: {ami}  instance: {args.instance_type}  arch: {args.arch}  region: {args.region}",
          flush=True)

    # S2: pass the EMD blob's expected sha256 (from its sidecar .sha256) so the box verifies it before training.
    emd_sha = ""
    if args.emd_frozen:
        sha_path = ROOT / (args.emd_frozen + ".sha256")
        if sha_path.exists():
            emd_sha = sha_path.read_text(encoding="utf-8").split()[0].strip()
            print(f"EMD blob sha256 (on-box verify armed): {emd_sha}", flush=True)
        else:
            print(f"WARN: {sha_path} missing -> on-box EMD sha check SKIPPED (blob unverified)", flush=True)
    river_ochs_sha = ""
    if args.river_ochs:
        sha_path = ROOT / (args.river_ochs + ".sha256")
        if sha_path.exists():
            river_ochs_sha = sha_path.read_text(encoding="utf-8").split()[0].strip()
            print(f"river-OCHS table sha256 (on-box verify armed): {river_ochs_sha}", flush=True)
        else:
            print(f"WARN: {sha_path} missing -> on-box river-OCHS sha check SKIPPED (table unverified)",
                  flush=True)
    # B2: the on-box dead-man cost cap floor = --max-hours (absolute, independent of the babysitter/PC).
    wall_guard_secs = max(1, int(args.max_hours * 3600))
    awscli_zip = "awscli-exe-linux-aarch64.zip" if args.arch == "arm64" else "awscli-exe-linux-x86_64.zip"

    # EC2 caps user-data at 16384 bytes and the full bootstrap outgrew it (river-OCHS launch,
    # 2026-07-03). Split: full bootstrap -> S3 (uploaded on EVERY launch, incl. --reuse-code
    # relaunches — content is deterministic from the args, so a babysitter relaunch re-uploads
    # the identical script); user-data = STEP-0 cost cap + fetch-and-exec stub.
    bootstrap = bootstrap_body(
        bucket=args.bucket, prefix=args.prefix, region=args.region, code_key=code_key,
        run_subdir=run_subdir, iters=args.iters, n_deals=args.n_deals,
        num_buckets=args.num_buckets, depth_edges=depth_edges,
        checkpoint_every=args.checkpoint_every, keep_alive=args.keep_alive,
        emd_frozen_rel=args.emd_frozen, emd_sha=emd_sha,
        river_ochs_rel=args.river_ochs, river_ochs_sha=river_ochs_sha,
        preflop_hist_key=args.preflop_hist_key,
        work_steal=args.work_steal, dcfr=args.dcfr,
        dcfr_prune_threshold=args.dcfr_prune_threshold, dcfr_explore_frac=args.dcfr_explore_frac,
        wall_guard_secs=wall_guard_secs, awscli_zip=awscli_zip,
    )
    bootstrap_key = f"{args.prefix}/{run_subdir}/code/bootstrap.sh"
    s3.put_object(Bucket=args.bucket, Key=bootstrap_key, Body=bootstrap.encode("utf-8"))
    print(f"Uploaded bootstrap ({len(bootstrap)} bytes) -> s3://{args.bucket}/{bootstrap_key}", flush=True)
    ud = user_data_stub(
        bucket=args.bucket, region=args.region, run_prefix=f"{args.prefix}/{run_subdir}",
        bootstrap_key=bootstrap_key, wall_guard_secs=wall_guard_secs, awscli_zip=awscli_zip,
    )
    # EC2's 16384-byte cap applies to the BASE64-ENCODED user-data (that's how the original
    # inline script tripped it at ~13.5KB raw = ~18KB encoded).
    ud_b64_len = len(base64.b64encode(ud.encode("utf-8")))
    if ud_b64_len > 16000:
        raise SystemExit(f"user-data stub too large (base64 {ud_b64_len} bytes) — must stay under 16384")
    run_kwargs = dict(
        ImageId=ami,
        InstanceType=args.instance_type,
        MinCount=1,
        MaxCount=1,
        UserData=base64.b64encode(ud.encode()).decode(),
        IamInstanceProfile={"Name": args.instance_profile},
        MetadataOptions={"HttpTokens": "required", "HttpPutResponseHopLimit": 2},
        TagSpecifications=[{
            "ResourceType": "instance",
            "Tags": [
                {"Key": "Name", "Value": f"native-cfr-train-{run_subdir}"},
                {"Key": "Project", "Value": "pokerbot-native-cfr"},
                {"Key": "Run", "Value": run_subdir},
            ],
        }],
        InstanceInitiatedShutdownBehavior="terminate",
    )
    if not args.on_demand:
        run_kwargs["InstanceMarketOptions"] = {
            "MarketType": "spot",
            "SpotOptions": {"SpotInstanceType": "one-time", "InstanceInterruptionBehavior": "terminate"},
        }

    # InstanceType is set per-attempt by _launch_with_fallback; drop the static key so the helper owns it.
    run_kwargs.pop("InstanceType", None)
    instance_types = (
        _spotpick_chain(args.instance_type, args.arch, args.spotpick_n)
        if args.spotpick else [args.instance_type]
    )
    if len(instance_types) > 1:
        print(f"spotpick fallback chain: {instance_types}", flush=True)
    iid, used_type = _launch_with_fallback(
        ec2, run_kwargs, instance_types=instance_types, spot=not args.on_demand
    )
    print(f"Launched {iid} as {used_type} ({'spot' if not args.on_demand else 'on-demand'})", flush=True)
    print(f"Run S3: s3://{args.bucket}/{args.prefix}/{run_subdir}/", flush=True)
    print(f"  checkpoints: .../{run_subdir}/checkpoints/  | logs: .../{run_subdir}/logs/", flush=True)
    print(f"Monitor console: aws ec2 get-console-output --instance-id {iid} --region {args.region} --latest",
          flush=True)
    print(f"Terminate: aws ec2 terminate-instances --instance-ids {iid} --region {args.region}", flush=True)


if __name__ == "__main__":
    main()
