#!/usr/bin/env python3
"""cert7: launch ONE cheap m-class spot box that EVALUATES bpv4_tier1_faithful_1 iter_126000.

Reuses the cert6 proven mechanics (AMI resolve, capacity-fallback chain, dead-man wall-guard
user-data stub, snap_avg -> mmap-index serve object, serve-miss gate) but:
  * serve object = iter_126000.nlcfr (preflop, verbatim) + snap_avg over the CONVERGED snapshot
    window {100k,105k,110k,115k,120k,125k} (postflop); the run trained with SNAPSHOT_AVG so the
    postflop average lives in the NLSNAP01 snapshots and MUST be snap_avg-merged (never filter the
    raw ckpt -- cert5 bug);
  * priority-ordered EVAL sections, each NON-FATAL and time-boxed, re-assembling VERDICT_FINAL.json
    after every section so a cap-stop / spot-reclaim always leaves the freshest partial verdict:
      1  serve object + miss gate (HARD gate: <0.5% self-play miss before any money)
      2  RAW strength: engine_ab_h2h new-vs-incumbent (CRN-paired, seat-rotated) + pool + verdict
      3  LBR6 exploitability: new-raw, incumbent (leaf-settle ON, menu fc)  [cheap, high value]
      4  preflop RAW shape dump + BTN-AA-limp
      5  SEARCH strength: engine_ab_h2h new-SEARCH(K=2000)-vs-incumbent + search pool  [expensive]
      6  LBR6 new-SEARCH + per-street attribution                                       [expensive]
      7  search-K convergence sweep {250,500,1000,2000,4000} (L1 vs K=4000)
      8  preflop SEARCH-resolved shape dump (does search neutralize the limp?)          [expensive]
      9  ladder plateau proxy (strategy movement 75k/100k/125k retained blobs)          [best-effort]

Cost: m-class 64-vCPU/256GB spot ~$1.12-1.73/h; 8h wall-guard bounds worst-case compute < ~$14
(well under the $18 dead-man cap). Local babysitter enforces the wall + stale-heartbeat kill.
"""
from __future__ import annotations

import argparse
import base64
import gzip
import hashlib
import io
import json
import sys
import tarfile
from pathlib import Path

import boto3
from botocore.exceptions import ClientError

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

from launch_native_cfr_training_spot import (  # noqa: E402
    _launch_with_fallback, resolve_ubuntu_ami, user_data_stub,
)

BUCKET = "video-processing-6789"
PREFIX = "pokerbot/prod6000/cert_pluribus"
REGION = "eu-central-1"
PROFILE = "pokerbot-labelgen-worker"
RUN_S3 = "s3://video-processing-6789/pokerbot/prod6000/native_cfr/bpv4_tier1_faithful_1"
NLCFR_S3 = f"{RUN_S3}/checkpoints/iter_126000.nlcfr"
# Converged post-limp-discard window (train.log: on-policy postflop miss ~0.0-0.2% here). Preflop
# comes from iter_126000 verbatim; these supply the postflop snapshot average.
SNAP_ITERS = (100000, 105000, 110000, 115000, 120000, 125000)
SNAPS_S3 = [f"{RUN_S3}/snapshots/snap_{i:06d}.nlsnap" for i in SNAP_ITERS]

EMD_FROZEN = "/opt/pokerbot/data/abstraction/emd_frozen_flop_turn_160_r400.emdfrzn"
RIVER_OCHS = "/opt/pokerbot/data/abstraction/river_ochs_160.rochs"
IDX = "/opt/pokerbot/checkpoints/idx_new_full"

# NEW-arm keying (import-latched native gates) -- the trained serve regime.
NEW_KEYS = {
    "POKERBOT_FULLHIST_KEY": "1", "POKERBOT_PRESUM_KEY": "1", "POKERBOT_MRR": "1",
    "POKERBOT_CLOSE_CALLED_STREET": "1", "POKERBOT_PRE_KEEP": "2,3,4,5,6",
    "POKERBOT_PRE_RR_KEEP": "4,5,6", "POKERBOT_ACTION_TRANSLATE": "1",
    "POKERBOT_POSTFLOP_SB_FIRST": "1", "POKERBOT_BP_COVERAGE": "1",
    "POKERBOT_RIVER_OCHS": RIVER_OCHS,
}
# Shared EMD-K160 postflop abstraction (both blueprints). engine_ab_h2h setdefaults these; the
# self-contained scripts (lbr6, search_divergence, preflop) do NOT -- so we export them explicitly.
EMD_ENV = {
    "POKERBOT_POSTFLOP_MODE": "emd", "POKERBOT_POSTFLOP_NUM_BUCKETS": "160",
    "POKERBOT_POSTFLOP_EMD_FROZEN": EMD_FROZEN, "POKERBOT_EMD_FROZEN_MIRROR_FALLBACK": "1",
}
INC_KEYS = {"POKERBOT_CLOSE_CALLED_STREET": "1", "POKERBOT_POSTFLOP_SB_FIRST": "1",
            "POKERBOT_BP_COVERAGE": "1"}


def _sh(d: dict) -> str:
    return " ".join(f"{k}={v}" for k, v in d.items())


ARM_NEW = json.dumps({"worker_env": NEW_KEYS, "settle_showdown_leaf": True, "search": False,
                      "blueprint_path": IDX})
ARM_INC = json.dumps({"worker_env": INC_KEYS, "settle_showdown_leaf": True, "search": False})
ARM_NEW_SEARCH = json.dumps({"worker_env": NEW_KEYS, "settle_showdown_leaf": True, "search": True,
                             "iterations": 2000, "time_budget_ms": 2000, "ensemble_streams": 1,
                             "leaf_cache_worlds": 4, "search_preflop": False, "blueprint_path": IDX})

_EXCLUDE_DIRS = ("__pycache__", ".git", "target", "artifacts", "runs")
_CODE_INCLUDE = [
    "pokerbot",
    "tools/battery_native",  # Rust source; target/ excluded by _EXCLUDE_DIRS (box builds on-box)
    "scripts/engine_ab_h2h.py", "scripts/h2h_arm_worker.py", "scripts/pool_absolute_one_arm.py",
    "scripts/track2_s3c_pool.py", "scripts/nlcfr_stream_filter.py", "scripts/nlcfr_key_index.py",
    "scripts/serve_miss_probe.py", "scripts/cert_verdict_merge.py",
    "scripts/probe_fullhist_key_parity.py", "scripts/probe_native_emd_parity.py",
    "scripts/f32_regret_strategy_delta.py",
    # cert7 sections
    "scripts/lbr6_audit.py", "scripts/lbr6_attribute.py", "scripts/pool_lbr6.py",
    "scripts/native_exploit_equiv.py", "scripts/search_divergence_selfplay.py",
    "scripts/audit_preflop_blueprint.py", "scripts/blueprint_convergence_stop.py",
    "scripts/assemble_cert7_verdict.py",
    "data/weva", "data/player_profiles",
]
_DATA_UPLOADS = [
    ("data/abstraction/emd_frozen_flop_turn_160_r400.emdfrzn", "emd_frozen_flop_turn_160_r400.emdfrzn"),
    ("data/abstraction/river_ochs_160.rochs", "river_ochs_160.rochs"),
    ("checkpoints/serve_blueprint_6max.pkl", "serve_blueprint_6max.pkl"),
]


def _tar_filter(ti: tarfile.TarInfo):
    parts = ti.name.split("/")
    if any(p in _EXCLUDE_DIRS for p in parts) or ti.name.endswith((".pyc", ".pkl.bak")):
        return None
    return ti


def _upload_code(s3) -> str:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for rel in _CODE_INCLUDE:
            p = ROOT / rel
            if not p.exists():
                raise SystemExit(f"code include missing: {p}")
            tar.add(p, arcname=rel, filter=_tar_filter)
    buf.seek(0)
    h = hashlib.sha256()
    with gzip.GzipFile(fileobj=buf, mode="rb") as gz:
        for chunk in iter(lambda: gz.read(1 << 24), b""):
            h.update(chunk)
    sha = h.hexdigest()
    buf.seek(0)
    key = f"{PREFIX}/cert7/code/cert7_code.tar.gz"
    try:
        head = s3.head_object(Bucket=BUCKET, Key=key)
        if head.get("Metadata", {}).get("code-sha") == sha:
            print(f"code tarball unchanged (sha={sha}) — skipping upload", flush=True)
            return key
    except ClientError:
        pass
    s3.put_object(Bucket=BUCKET, Key=key, Body=buf.read(), Metadata={"code-sha": sha})
    print(f"uploaded code tarball -> s3://{BUCKET}/{key} (sha={sha})", flush=True)
    return key


def _upload_data(s3) -> None:
    for local, name in _DATA_UPLOADS:
        p = ROOT / local
        if not p.exists():
            raise SystemExit(f"data upload missing: {p}")
        key = f"{PREFIX}/data/{name}"
        want = hashlib.sha256(p.read_bytes()).hexdigest()
        try:
            head = s3.head_object(Bucket=BUCKET, Key=key)
            if head.get("Metadata", {}).get("sha256") == want:
                print(f"  data unchanged: {name}", flush=True)
                continue
        except ClientError:
            pass
        s3.upload_file(str(p), BUCKET, key, ExtraArgs={"Metadata": {"sha256": want}})
        print(f"  uploaded {name} ({p.stat().st_size/1e6:.1f} MB)", flush=True)


def _render_bootstrap(*, raw_decks: int, search_decks: int, pool_decks: int,
                      search_pool_decks: int, lbr_hands: int, lbr_search_hands: int) -> str:
    res = f"s3://{BUCKET}/{PREFIX}/cert7/results"
    snap_urls = " ".join(SNAPS_S3)
    newenv = _sh(NEW_KEYS)
    emdenv = _sh(EMD_ENV)
    incenv = _sh(INC_KEYS)
    T = _TEMPLATE
    subs = {
        "@@REGION@@": REGION, "@@BUCKET@@": BUCKET, "@@PREFIX@@": PREFIX, "@@RES@@": res,
        "@@NLCFR_S3@@": NLCFR_S3, "@@SNAP_URLS@@": snap_urls,
        "@@EMD_FROZEN@@": EMD_FROZEN, "@@RIVER_OCHS@@": RIVER_OCHS, "@@IDX@@": IDX,
        "@@NEWENV@@": newenv, "@@EMDENV@@": emdenv, "@@INCENV@@": incenv,
        "@@ARM_NEW@@": ARM_NEW, "@@ARM_INC@@": ARM_INC, "@@ARM_NEW_SEARCH@@": ARM_NEW_SEARCH,
        "@@RAW_DECKS@@": str(raw_decks), "@@SEARCH_DECKS@@": str(search_decks),
        "@@POOL_DECKS@@": str(pool_decks), "@@SEARCH_POOL_DECKS@@": str(search_pool_decks),
        "@@LBR_HANDS@@": str(lbr_hands), "@@LBR_SEARCH_HANDS@@": str(lbr_search_hands),
    }
    for k, v in subs.items():
        T = T.replace(k, v)
    return T


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-decks", type=int, default=6000)
    ap.add_argument("--search-decks", type=int, default=600)
    ap.add_argument("--pool-decks", type=int, default=600)
    ap.add_argument("--search-pool-decks", type=int, default=150)
    ap.add_argument("--lbr-hands", type=int, default=800)
    ap.add_argument("--lbr-search-hands", type=int, default=200)
    ap.add_argument("--wall-hours", type=float, default=7.5)
    # cheap m-class 64-vCPU/256GB chain (max spot ~$1.73/h => 7.5h+EBS < ~$15, under the $18 cap)
    ap.add_argument("--instance-types", default="m6i.16xlarge,m7i.16xlarge,m5.16xlarge,m6a.16xlarge")
    ap.add_argument("--volume-gb", type=int, default=800)
    ap.add_argument("--on-demand", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    bootstrap = _render_bootstrap(
        raw_decks=args.raw_decks, search_decks=args.search_decks, pool_decks=args.pool_decks,
        search_pool_decks=args.search_pool_decks, lbr_hands=args.lbr_hands,
        lbr_search_hands=args.lbr_search_hands)
    if args.dry_run:
        Path(ROOT / "artifacts").mkdir(exist_ok=True)
        (ROOT / "artifacts" / "_cert7_bootstrap_render.sh").write_text(bootstrap, encoding="utf-8")
        print(f"--dry-run: wrote artifacts/_cert7_bootstrap_render.sh ({len(bootstrap)} bytes), "
              "not launching, no uploads")
        return

    session = boto3.Session(region_name=REGION)
    s3 = session.client("s3")
    ec2 = session.client("ec2")

    _upload_data(s3)
    _upload_code(s3)

    bkey = f"{PREFIX}/cert7/code/bootstrap.sh"
    s3.put_object(Bucket=BUCKET, Key=bkey, Body=bootstrap.encode())
    print(f"uploaded bootstrap ({len(bootstrap)} bytes) -> s3://{BUCKET}/{bkey}", flush=True)

    wall = int(args.wall_hours * 3600)
    ud = user_data_stub(bucket=BUCKET, region=REGION, run_prefix=f"{PREFIX}/cert7",
                        bootstrap_key=bkey, wall_guard_secs=wall)
    ami, root_dev = resolve_ubuntu_ami(ec2, "x86_64")
    run_kwargs = dict(
        ImageId=ami, MinCount=1, MaxCount=1,
        UserData=base64.b64encode(ud.encode()).decode(),
        IamInstanceProfile={"Name": PROFILE},
        MetadataOptions={"HttpTokens": "required", "HttpPutResponseHopLimit": 2},
        TagSpecifications=[{"ResourceType": "instance", "Tags": [
            {"Key": "Name", "Value": "cert7-tier1-126k"},
            {"Key": "Project", "Value": "pokerbot-cert"},
            {"Key": "Run", "Value": "cert7"}]}],
        InstanceInitiatedShutdownBehavior="terminate",
        BlockDeviceMappings=[{"DeviceName": root_dev,
                              "Ebs": {"VolumeSize": args.volume_gb, "VolumeType": "gp3",
                                      "Iops": 12000, "Throughput": 750,
                                      "DeleteOnTermination": True}}],
    )
    if not args.on_demand:
        run_kwargs["InstanceMarketOptions"] = {
            "MarketType": "spot",
            "SpotOptions": {"SpotInstanceType": "one-time",
                            "InstanceInterruptionBehavior": "terminate"}}
    chain = [t.strip() for t in args.instance_types.split(",") if t.strip()]
    iid, used = _launch_with_fallback(ec2, run_kwargs, instance_types=chain,
                                      spot=not args.on_demand)
    print(f"LAUNCHED {iid} as {used} ({'spot' if not args.on_demand else 'on-demand'})")
    print(f"IID={iid}")
    print(f"results: s3://{BUCKET}/{PREFIX}/cert7/results/")
    print(f"terminate: aws ec2 terminate-instances --instance-ids {iid} --region {REGION}")


# --------------------------------------------------------------------------- bootstrap template
# Plain string with @@TOKEN@@ substitution (NOT an f-string) so bash ${..} / {..} stay literal.
_TEMPLATE = r"""#!/bin/bash
set -uxo pipefail
export HOME="${HOME:-/root}"
export DEBIAN_FRONTEND=noninteractive AWS_DEFAULT_REGION=@@REGION@@
OUT=/opt/pokerbot/cert_out; mkdir -p $OUT /opt/pokerbot
fail() { echo "$1" > $OUT/FAILED.txt; aws s3 sync $OUT @@RES@@/ --region @@REGION@@ || true; sync; poweroff; exit 1; }
stage() { echo "$(date -u +%s) $1" >> $OUT/stage.txt; }
assemble() { PYTHONPATH=/opt/pokerbot python3 /opt/pokerbot/scripts/assemble_cert7_verdict.py --out-dir $OUT --run-id cert7 >> $OUT/assemble.log 2>&1 || true; cp $OUT/VERDICT_FINAL.json $OUT/VERDICT.partial.json 2>/dev/null || true; aws s3 sync $OUT @@RES@@/ --region @@REGION@@ --quiet || true; }
# section wrapper: NON-FATAL + time-boxed. $1=stage-name $2=timeout-secs $3=logfile ; rest=cmd
section() { local nm="$1"; local tmo="$2"; local lg="$3"; shift 3; stage "$nm"; echo "=== $nm (timeout ${tmo}s) $(date -u +%FT%TZ)" | tee -a $OUT/$lg; ( timeout "$tmo" "$@" ) >> $OUT/$lg 2>&1 || echo "SECTION $nm rc=$? (non-fatal)" | tee -a $OUT/$lg $OUT/section_warnings.txt; assemble; }
stage boot

# heartbeat + result sync (entire run)
( while true; do date -u +%s > $OUT/heartbeat.txt; aws s3 sync $OUT @@RES@@/ --region @@REGION@@ --quiet || true; sleep 120; done ) &

stage deps
apt-get update -qq && apt-get install -y -qq python3-pip python3-dev build-essential curl unzip || fail deps
curl -fsSL "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o /tmp/aws.zip && unzip -q /tmp/aws.zip -d /tmp && /tmp/aws/install --update && hash -r
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal
export PATH="$HOME/.cargo/bin:$PATH"

aws s3 cp s3://@@BUCKET@@/@@PREFIX@@/cert7/code/cert7_code.tar.gz /tmp/code.tar.gz --region @@REGION@@ || fail codefetch
tar -xzf /tmp/code.tar.gz -C /opt/pokerbot
cd /opt/pokerbot
python3 -m pip install --disable-pip-version-check --no-input --break-system-packages boto3 numpy pandas "jax[cpu]" phevaluator flax optax cloudpickle \
  || python3 -m pip install --disable-pip-version-check --no-input boto3 numpy pandas "jax[cpu]" phevaluator flax optax cloudpickle || fail pip
python3 -m pip install --disable-pip-version-check --no-input --break-system-packages torch --index-url https://download.pytorch.org/whl/cpu \
  || python3 -m pip install --disable-pip-version-check --no-input torch --index-url https://download.pytorch.org/whl/cpu || fail torch

mkdir -p data/abstraction checkpoints runs
aws s3 cp s3://@@BUCKET@@/@@PREFIX@@/data/emd_frozen_flop_turn_160_r400.emdfrzn data/abstraction/ --region @@REGION@@ || fail blob1
aws s3 cp s3://@@BUCKET@@/@@PREFIX@@/data/river_ochs_160.rochs data/abstraction/ --region @@REGION@@ || fail blob2
aws s3 cp s3://@@BUCKET@@/@@PREFIX@@/data/serve_blueprint_6max.pkl checkpoints/ --region @@REGION@@ || fail incpkl
echo "344ab2fd328f4b360fd2c11a09c1f6fd2c45a56e62f75d6bc1a502c71612cb74  data/abstraction/emd_frozen_flop_turn_160_r400.emdfrzn" | sha256sum -c - || fail blob1sha
echo "25c860d2f35a08fcb38499d436d29547233b771e5a6b4fc057da8a72d7d53d44  data/abstraction/river_ochs_160.rochs" | sha256sum -c - || fail blob2sha

# native build: snap_avg FIRST (no python feature), then the python-feature lib (feature sets do not
# unify across cargo invocations). NO target-cpu override (native_solver f32 bit-identity self-check).
stage build
cd tools/battery_native && cargo build --release --example snap_avg 2>&1 | tail -2 || fail build_snapavg
cargo build --release --features python 2>&1 | tail -3 || fail build
cp target/release/libbattery_native.so /opt/pokerbot/battery_native.so
mkdir -p target/release && cp target/release/libbattery_native.so target/release/battery_native.so || true
cd /opt/pokerbot
python3 -c "import battery_native; print('caps', sorted(battery_native.native_caps()))" || fail natimport
python3 -c "import battery_native,sys; sys.exit(0 if {'river_ochs_serve','settle_showdown_leaf','presum_key','full_hist_key','action_translate'} <= set(battery_native.native_caps()) else 1)" || fail natcaps
POKERBOT_SELFCHECK_VERBOSE=1 python3 -c "
from pokerbot.search import native_solver as ns
import sys
print('available', ns.available(), 'selfcheck', ns._SELFCHECK_OK, flush=True)
sys.exit(0 if ns.available() else 1)" 2>&1 | tee $OUT/selfcheck_diag.log || fail selfcheck

# ---------------- SERVE OBJECT ----------------
# iter_126000 (preflop, verbatim) + snap_avg over the converged snapshot window (postflop). NEVER
# filter the raw ckpt (postflop-EMPTY under SNAPSHOT_AVG -- cert5 bug). Full mmap index (no row cap).
stage nlcfr_download
aws s3 cp @@NLCFR_S3@@ /opt/pokerbot/iter_126000.nlcfr --region @@REGION@@ || fail nlcfr
echo "abcb00730f4e8671cc8ae5a27db1ac256a4f3d83437005f6cbfb30b372b1ae2f  /opt/pokerbot/iter_126000.nlcfr" | sha256sum -c - || fail nlcfr_sha
SNAPS=""
for SURL in @@SNAP_URLS@@; do
  SBASE=$(basename $SURL)
  aws s3 cp $SURL /opt/pokerbot/$SBASE --region @@REGION@@ || fail snapdl
  SNAPS="$SNAPS /opt/pokerbot/$SBASE"
done
stage snap_merge
tools/battery_native/target/release/examples/snap_avg /opt/pokerbot/merged_126k.nlcfr \
  /opt/pokerbot/iter_126000.nlcfr $SNAPS 2>&1 | tee $OUT/snap_merge.log || fail snap_merge
rm -f /opt/pokerbot/iter_126000.nlcfr /opt/pokerbot/snap_*.nlsnap
stage index
python3 scripts/nlcfr_key_index.py /opt/pokerbot/merged_126k.nlcfr @@IDX@@ --with-rows 2>&1 | tee $OUT/index.log || fail index
rm -f /opt/pokerbot/merged_126k.nlcfr
cp @@IDX@@/meta.json $OUT/serve_index_meta.json || true
export PYTHONPATH=/opt/pokerbot

# ---------------- SERVE-MISS GATE (HARD, before any money) ----------------
stage serve_gate
@@NEWENV@@ python3 scripts/serve_miss_probe.py --blueprint @@IDX@@ --full-index @@IDX@@ \
  --mode selfplay --hands 2000 --seed 11 --miss-log $OUT/serve_gate_miss.jsonl \
  --out $OUT/serve_gate_selfplay.json 2>&1 | tail -8 | tee $OUT/serve_gate.log
python3 - <<'PY' || fail serve_gate
import json
d = json.load(open('/opt/pokerbot/cert_out/serve_gate_selfplay.json'))
mr = d["total"]["miss_rate"]
print("serve gate: self-play miss_rate=%.4f (threshold 0.005)" % mr)
assert mr < 0.005, "SERVE-MISS GATE FAIL: %.4f >= 0.5%% -- NOT spending on the money cert" % mr
PY
assemble

# ---------------- non-fatal parity/knob gates (recorded, do not abort) ----------------
section gate_parity_fullhist 900 gate_parity_fullhist.log env @@NEWENV@@ python3 scripts/probe_fullhist_key_parity.py --fullhist 1 --presum 1 --min-states 1024 --min-postflop 1024 --out $OUT/gate_parity_fullhist.json
section gate_parity_emd 900 gate_parity_emd.log env @@NEWENV@@ python3 scripts/probe_native_emd_parity.py --min-postflop 600 --min-per-street 80 --out $OUT/gate_parity_emd_ochs.json
section gate_parity_translate 900 gate_parity_translate.log env @@NEWENV@@ python3 scripts/probe_fullhist_key_parity.py --fullhist 1 --presum 1 --translate 1 --out $OUT/gate_parity_translate.json

MEMGB=$(awk '/MemTotal/ {printf "%d", $2/1048576}' /proc/meminfo)
SHARDS=3; [ "$MEMGB" -gt 400 ] && SHARDS=6
DPS=$(( @@RAW_DECKS@@ / SHARDS ))
echo "MEMGB=$MEMGB SHARDS=$SHARDS raw_decks/shard=$DPS" | tee $OUT/plan.txt

# ================= SECTION 2: RAW STRENGTH (headline, cheap) =================
stage sec2_raw_strength
for SEED in 101 202; do
  pids=()
  for SH in $(seq 0 $((SHARDS-1))); do
    env POKERBOT_CLOSE_CALLED_STREET=1 POKERBOT_POSTFLOP_SB_FIRST=1 python3 scripts/engine_ab_h2h.py \
      --arm-a '@@ARM_NEW@@' --arm-b '@@ARM_INC@@' \
      --decks $DPS --deck-start $SH --deck-stride $SHARDS --seed $SEED \
      --label raw_new_vs_inc_s${SEED}_sh${SH} \
      --out $OUT/h2h_seed${SEED}_shard${SH}.json > $OUT/h2h_seed${SEED}_shard${SH}.log 2>&1 &
    pids+=($!)
  done
  for p in "${pids[@]}"; do wait $p || echo "raw shard rc!=0 (seed $SEED)" >> $OUT/section_warnings.txt; done
  assemble
done
# pool absolutes (one process per arm-env)
@@NEWENV@@ python3 scripts/pool_absolute_one_arm.py --blueprint @@IDX@@ \
  --spec '{"search": false, "settle_showdown_leaf": true}' --decks @@POOL_DECKS@@ --seed 0 --workers 1 \
  --label pool_new --out $OUT/pool_new.json > $OUT/pool_new.log 2>&1 &
PNEW=$!
env @@INCENV@@ python3 scripts/pool_absolute_one_arm.py --blueprint checkpoints/serve_blueprint_6max.pkl \
  --spec '{"search": false, "settle_showdown_leaf": true}' --decks @@POOL_DECKS@@ --seed 0 --workers 1 \
  --label pool_inc --out $OUT/pool_inc.json > $OUT/pool_inc.log 2>&1 &
PINC=$!
wait $PNEW || echo "pool_new rc!=0" >> $OUT/section_warnings.txt
wait $PINC || echo "pool_inc rc!=0" >> $OUT/section_warnings.txt
python3 scripts/cert_verdict_merge.py --out-dir $OUT --run-id cert7 2>&1 | tee $OUT/verdict_merge.log || echo "verdict merge rc!=0" >> $OUT/section_warnings.txt
assemble

# ================= SECTION 3: LBR6 exploitability new-raw + incumbent (cheap) =================
# leaf-settle ON (env) all arms; EMD-K160 postflop env explicit; menu fc (model-error-free floor).
section lbr_new_raw 5400 lbr_new_raw.log env POKERBOT_SETTLE_SHOWDOWN_LEAF=1 @@EMDENV@@ @@NEWENV@@ \
  python3 scripts/lbr6_audit.py --target blueprint --menu fc --hands @@LBR_HANDS@@ --seed 0 \
  --blueprint-snapshot @@IDX@@ --out $OUT/lbr_new_raw.json
section lbr_incumbent 5400 lbr_incumbent.log env POKERBOT_SETTLE_SHOWDOWN_LEAF=1 @@EMDENV@@ @@INCENV@@ \
  python3 scripts/lbr6_audit.py --target blueprint --menu fc --hands @@LBR_HANDS@@ --seed 0 \
  --blueprint-snapshot checkpoints/serve_blueprint_6max.pkl --out $OUT/lbr_incumbent.json

# ================= SECTION 4: preflop RAW shape dump (cheap) =================
section preflop_raw 2400 preflop_raw.log env AUDIT_PF_CHECKPOINT=@@IDX@@ AUDIT_PF_OUT=$OUT/preflop_new_raw.json \
  @@NEWENV@@ python3 scripts/audit_preflop_blueprint.py 3000

# ================= SECTION 5: SEARCH strength (expensive) =================
stage sec5_search_strength
SDPS=$(( @@SEARCH_DECKS@@ / SHARDS ))
pids=()
for SH in $(seq 0 $((SHARDS-1))); do
  env POKERBOT_CLOSE_CALLED_STREET=1 POKERBOT_POSTFLOP_SB_FIRST=1 POKERBOT_EMD_FROZEN_MIRROR_FALLBACK=1 timeout 5400 python3 scripts/engine_ab_h2h.py \
    --arm-a '@@ARM_NEW_SEARCH@@' --arm-b '@@ARM_INC@@' \
    --decks $SDPS --deck-start $SH --deck-stride $SHARDS --seed 303 \
    --label search_new_vs_inc_sh${SH} \
    --out $OUT/search_h2h_seed303_shard${SH}.json > $OUT/search_h2h_shard${SH}.log 2>&1 &
  pids+=($!)
done
for p in "${pids[@]}"; do wait $p || echo "search shard rc!=0" >> $OUT/section_warnings.txt; done
assemble
# search vs pool (small)
section search_pool_new 9000 search_pool_new.log env @@NEWENV@@ python3 scripts/pool_absolute_one_arm.py \
  --blueprint @@IDX@@ --spec '{"search": true, "iterations": 2000, "settle_showdown_leaf": true, "leaf_cache_worlds": 4}' \
  --decks @@SEARCH_POOL_DECKS@@ --seed 0 --workers 1 --label pool_new_search --out $OUT/search_pool_new.json

# ================= SECTION 6: LBR6 new-SEARCH + per-street attribution (expensive) =================
section lbr_new_search 7200 lbr_new_search.log env POKERBOT_SETTLE_SHOWDOWN_LEAF=1 POKERBOT_NATIVE_SUBGAME=1 @@EMDENV@@ @@NEWENV@@ \
  python3 scripts/lbr6_audit.py --target blueprint-search --menu fc --search-iters 2000 --search-leaf uniform \
  --hands @@LBR_SEARCH_HANDS@@ --seed 0 --blueprint-snapshot @@IDX@@ --out $OUT/lbr_new_search.json
section lbr_attribute_raw 5400 lbr_attribute_raw.log env POKERBOT_SETTLE_SHOWDOWN_LEAF=1 @@EMDENV@@ @@NEWENV@@ \
  python3 scripts/lbr6_attribute.py --raw --menu fc --hands 3000 --seed 0 --blueprint-snapshot @@IDX@@

# ================= SECTION 7: search-K convergence sweep =================
section ksweep 5400 ksweep.log env SD_KS=250,500,1000,2000,4000 SD_DECKS=30 SD_SEED=777 \
  BLUEPRINT_PATH=@@IDX@@ SD_OUT=$OUT/ksweep_selfplay.json POKERBOT_NATIVE_SUBGAME=1 @@EMDENV@@ @@NEWENV@@ \
  python3 scripts/search_divergence_selfplay.py

# ================= SECTION 8: preflop SEARCH-resolved shape (expensive) =================
section preflop_search 9000 preflop_search.log env AUDIT_PF_CHECKPOINT=@@IDX@@ AUDIT_PF_OUT=$OUT/preflop_new_search.json \
  AUDIT_PF_SEARCH_PREFLOP=1 AUDIT_PF_ITERS=2000 POKERBOT_NATIVE_SUBGAME=1 @@EMDENV@@ @@NEWENV@@ \
  python3 scripts/audit_preflop_blueprint.py 300

# ================= SECTION 9: ladder plateau signal (cheap; the run's own per-iter gate shape) =================
# The training run wrote gates/verdict_iter_*.json at every 10k with the SHAPE trajectory
# (BTN-open %, AA-BTN-limp %, per-street gate verdicts). Summarizing 70k->110k shows whether the
# blueprint is still moving (climbing) or plateaued -- the "is a relaunch to 200k worth it?" signal.
# Cheap (tiny JSONs, no serve build, no blob load).
section ladder_plateau 900 ladder_plateau.log env PYTHONPATH=/opt/pokerbot python3 - <<'PY'
import json, os, subprocess
OUT = "/opt/pokerbot/cert_out"
G = "s3://@@BUCKET@@/pokerbot/prod6000/native_cfr/bpv4_tier1_faithful_1/gates"
res = {"note": "run's own per-iter shape trajectory (from training gate verdicts); flat => plateaued",
       "trajectory": {}}
for it in (20000, 40000, 60000, 70000, 80000, 90000, 100000, 110000):
    try:
        raw = subprocess.check_output(
            ["aws", "s3", "cp", "%s/verdict_iter_%06d.json" % (G, it), "-",
             "--region", "@@REGION@@"], stderr=subprocess.DEVNULL)
        gd = json.loads(raw).get("gate", {}).get("gate_detail", {})
        row = {"rfi_pct": gd.get("opens_pct"),
               "aa_btn_limp": (gd.get("G3") or {}).get("aa_limp"),
               "btn_open_pct": (gd.get("G2") or {}).get("btn_open_pct"),
               "G1": gd.get("G1"),
               "G2_pass": (gd.get("G2") or {}).get("pass"),
               "G3_pass": (gd.get("G3") or {}).get("pass")}
        res["trajectory"]["iter_%d" % it] = row
    except Exception as e:  # noqa: BLE001
        res["trajectory"]["iter_%d" % it] = {"error": repr(e)}
# plateau read: AA-limp + BTN-open movement over the last ~40k
try:
    xs = [(k, v) for k, v in sorted(res["trajectory"].items()) if isinstance(v, dict) and v.get("aa_btn_limp") is not None]
    if len(xs) >= 2:
        res["aa_limp_last_minus_first"] = xs[-1][1]["aa_btn_limp"] - xs[0][1]["aa_btn_limp"]
        res["btn_open_last_minus_first"] = (xs[-1][1].get("btn_open_pct") or 0) - (xs[0][1].get("btn_open_pct") or 0)
except Exception:  # noqa: BLE001
    pass
json.dump(res, open(os.path.join(OUT, "ladder_plateau.json"), "w"), indent=1)
print(json.dumps(res, indent=1))
PY

stage verdict_final
assemble
cp $OUT/VERDICT_FINAL.json $OUT/VERDICT_FINAL.json 2>/dev/null || true
aws s3 cp $OUT/VERDICT_FINAL.json @@RES@@/VERDICT_FINAL.json --region @@REGION@@ || true

stage done
date -u +"%Y-%m-%dT%H:%M:%SZ" > $OUT/RESULT_DONE.txt
aws s3 sync $OUT @@RES@@/ --region @@REGION@@
sync
poweroff
"""


if __name__ == "__main__":
    main()
