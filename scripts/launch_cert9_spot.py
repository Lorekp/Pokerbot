#!/usr/bin/env python3
"""CERT9: honest re-certification of the SEARCH-minus-RAW money delta for the tier-1 126k
blueprint, with MANDATORY fail-closed proof that native K=2000 search is genuinely running.

WHY: cert7/cert8's "with search" arms never searched — the pre-033a172 native SubgameSolver only
took a dict blueprint, the mmap-index build crashed, and make_search_policy_fn's except silently
served the RAW blueprint (cert7 search shard: 200 decks in 53 s). 033a172 fixed the engine
(SubgameSolver.from_index_dir, capability "subgame_index_dir"); this cert measures the true
paired search-minus-raw delta under strict, fail-closed instrumentation:

  * arm A (SEARCH): tier-1 serve index + {"search": true, iterations: 2000, leaf_cache_worlds: 4,
    settle_showdown_leaf: true} with POKERBOT_SEARCH_STRICT=1 (any fallback RAISES) AND the
    engine_ab_h2h --require-search-native a per-deck gate (any non-native postflop serve aborts
    the shard). Every shard JSON carries decisions_total / decisions_search_native /
    decisions_fallback / per-street wall p50/p95.
  * arm B (RAW): same index, search off. PRIMARY = CRN-paired A-B delta (chips/deck).
  * arm C (secondary, remaining wall only): incumbent K160.

Flow on the box (each step time-boxed; results synced to S3 every 120 s, poweroff at the end):
  deps -> native build (caps must include subgame_index_dir) -> serve-index S3 sync + sha-verify
  -> serve-miss HARD gate -> PRE-SPEND PROBE (12 decks; cert9_gate must PASS: 100% native postflop,
  0 fallbacks, mean postflop wall >= 0.1 s) -> A-B money shards (deadline-bound, per-deck ckpts)
  -> optional incumbent shards -> assemble VERDICT_FINAL.json -> sync -> poweroff.

Cost bound: 64-vCPU spot ~$1.2-1.7/h, 11.5 h wall guard => ~$16-20 compute + ~$1.5 EBS, well under
the $40 cap. The user-data stub arms TWO on-box dead-man terminators before anything else.
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
SERVE_INDEX_S3 = f"{RUN_S3}/serve_index"   # ALREADY BUILT (538,848,758 rows) — sync + sha, no snap_avg

EMD_FROZEN = "/opt/pokerbot/data/abstraction/emd_frozen_flop_turn_160_r400.emdfrzn"
RIVER_OCHS = "/opt/pokerbot/data/abstraction/river_ochs_160.rochs"
IDX = "/opt/pokerbot/checkpoints/idx_new_full"

# Tier-1 certified serve keying (import-latched; set in the arm's worker_env so the isolated
# worker process configures exactly like the production serve).
NEW_KEYS = {
    "POKERBOT_FULLHIST_KEY": "1", "POKERBOT_PRESUM_KEY": "1", "POKERBOT_MRR": "1",
    "POKERBOT_CLOSE_CALLED_STREET": "1", "POKERBOT_PRE_KEEP": "2,3,4,5,6",
    "POKERBOT_PRE_RR_KEEP": "4,5,6", "POKERBOT_ACTION_TRANSLATE": "1",
    "POKERBOT_POSTFLOP_SB_FIRST": "1", "POKERBOT_BP_COVERAGE": "1",
    "POKERBOT_RIVER_OCHS": RIVER_OCHS,
}
EMD_ENV = {
    "POKERBOT_POSTFLOP_MODE": "emd", "POKERBOT_POSTFLOP_NUM_BUCKETS": "160",
    "POKERBOT_POSTFLOP_EMD_FROZEN": EMD_FROZEN, "POKERBOT_EMD_FROZEN_MIRROR_FALLBACK": "1",
}
INC_KEYS = {"POKERBOT_CLOSE_CALLED_STREET": "1", "POKERBOT_POSTFLOP_SB_FIRST": "1",
            "POKERBOT_BP_COVERAGE": "1"}


def _sh(d: dict) -> str:
    return " ".join(f"{k}={v}" for k, v in d.items())


# Arm specs (cert7's ARM_NEW_SEARCH / ARM_NEW / ARM_INC, plus strict mode on the search arm).
ARM_SEARCH = json.dumps({
    "worker_env": {**NEW_KEYS, "POKERBOT_SEARCH_STRICT": "1"},
    "settle_showdown_leaf": True, "search": True, "iterations": 2000, "time_budget_ms": 2000,
    "ensemble_streams": 1, "leaf_cache_worlds": 4, "search_preflop": False,
    "blueprint_path": IDX})
ARM_RAW = json.dumps({"worker_env": NEW_KEYS, "settle_showdown_leaf": True, "search": False,
                      "blueprint_path": IDX})
ARM_INC = json.dumps({"worker_env": INC_KEYS, "settle_showdown_leaf": True, "search": False})

_EXCLUDE_DIRS = ("__pycache__", ".git", "target", "artifacts", "runs")
_CODE_INCLUDE = [
    "pokerbot",
    "tools/battery_native",
    "scripts/engine_ab_h2h.py", "scripts/h2h_arm_worker.py",
    "scripts/serve_miss_probe.py",
    "scripts/probe_fullhist_key_parity.py", "scripts/probe_native_emd_parity.py",
    "scripts/cert9_gate.py", "scripts/assemble_cert9_verdict.py",
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
    key = f"{PREFIX}/cert9_search/code/cert9_code.tar.gz"
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
        key = f"{PREFIX}/data/{name}"
        if not p.exists():
            # Worktree launches: the big blobs live only in the main checkout, but cert7 already
            # uploaded sha-verified copies under this exact key (and the bootstrap re-verifies the
            # abstraction blobs against pinned shas on-box). Accept the existing S3 object.
            try:
                s3.head_object(Bucket=BUCKET, Key=key)
                print(f"  data not local, using existing S3 object: {name}", flush=True)
                continue
            except ClientError:
                raise SystemExit(f"data upload missing locally AND in S3: {p} / s3://{BUCKET}/{key}")
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


def _render_bootstrap(*, wall_secs: int, ab_seed: int, inc_seed: int, decks_per_shard: int,
                      inc_decks_per_shard: int, probe_decks: int) -> str:
    res = f"s3://{BUCKET}/{PREFIX}/cert9_search"
    T = _TEMPLATE
    subs = {
        "@@REGION@@": REGION, "@@BUCKET@@": BUCKET, "@@PREFIX@@": PREFIX, "@@RES@@": res,
        "@@SERVE_INDEX_S3@@": SERVE_INDEX_S3, "@@IDX@@": IDX,
        "@@NEWENV@@": _sh(NEW_KEYS), "@@EMDENV@@": _sh(EMD_ENV), "@@INCENV@@": _sh(INC_KEYS),
        "@@ARM_SEARCH@@": ARM_SEARCH, "@@ARM_RAW@@": ARM_RAW, "@@ARM_INC@@": ARM_INC,
        "@@WALL_SECS@@": str(wall_secs),
        "@@AB_SEED@@": str(ab_seed), "@@INC_SEED@@": str(inc_seed),
        "@@DPS@@": str(decks_per_shard), "@@INC_DPS@@": str(inc_decks_per_shard),
        "@@PROBE_DECKS@@": str(probe_decks),
    }
    for k, v in subs.items():
        T = T.replace(k, v)
    return T


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wall-hours", type=float, default=11.5)
    ap.add_argument("--ab-seed", type=int, default=909)
    ap.add_argument("--inc-seed", type=int, default=707)
    ap.add_argument("--decks-per-shard", type=int, default=1400,
                    help="A-B deck TARGET per shard; the section deadline usually binds first")
    ap.add_argument("--inc-decks-per-shard", type=int, default=400)
    ap.add_argument("--probe-decks", type=int, default=16)
    ap.add_argument("--instance-types",
                    default="m6i.16xlarge,m7i.16xlarge,m6a.16xlarge,m5.16xlarge,c6a.16xlarge")
    ap.add_argument("--volume-gb", type=int, default=150)
    ap.add_argument("--on-demand", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    wall = int(args.wall_hours * 3600)
    bootstrap = _render_bootstrap(
        wall_secs=wall, ab_seed=args.ab_seed, inc_seed=args.inc_seed,
        decks_per_shard=args.decks_per_shard, inc_decks_per_shard=args.inc_decks_per_shard,
        probe_decks=args.probe_decks)
    if args.dry_run:
        Path(ROOT / "artifacts").mkdir(exist_ok=True)
        (ROOT / "artifacts" / "_cert9_bootstrap_render.sh").write_text(bootstrap, encoding="utf-8")
        print(f"--dry-run: wrote artifacts/_cert9_bootstrap_render.sh ({len(bootstrap)} bytes)")
        return

    session = boto3.Session(region_name=REGION)
    s3 = session.client("s3")
    ec2 = session.client("ec2")

    _upload_data(s3)
    _upload_code(s3)

    bkey = f"{PREFIX}/cert9_search/code/bootstrap.sh"
    s3.put_object(Bucket=BUCKET, Key=bkey, Body=bootstrap.encode())
    print(f"uploaded bootstrap ({len(bootstrap)} bytes) -> s3://{BUCKET}/{bkey}", flush=True)

    ud = user_data_stub(bucket=BUCKET, region=REGION, run_prefix=f"{PREFIX}/cert9_search",
                        bootstrap_key=bkey, wall_guard_secs=wall)
    ami, root_dev = resolve_ubuntu_ami(ec2, "x86_64")
    run_kwargs = dict(
        ImageId=ami, MinCount=1, MaxCount=1,
        UserData=base64.b64encode(ud.encode()).decode(),
        IamInstanceProfile={"Name": PROFILE},
        MetadataOptions={"HttpTokens": "required", "HttpPutResponseHopLimit": 2},
        TagSpecifications=[{"ResourceType": "instance", "Tags": [
            {"Key": "Name", "Value": "cert9-search-vs-raw"},
            {"Key": "Project", "Value": "pokerbot-cert"},
            {"Key": "Run", "Value": "cert9"}]}],
        InstanceInitiatedShutdownBehavior="terminate",
        BlockDeviceMappings=[{"DeviceName": root_dev,
                              "Ebs": {"VolumeSize": args.volume_gb, "VolumeType": "gp3",
                                      "Iops": 6000, "Throughput": 500,
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
    print(f"results: s3://{BUCKET}/{PREFIX}/cert9_search/")
    print(f"terminate: aws ec2 terminate-instances --instance-ids {iid} --region {REGION}")


# --------------------------------------------------------------------------- bootstrap template
# Plain string with @@TOKEN@@ substitution (NOT an f-string) so bash ${..} stays literal.
_TEMPLATE = r"""#!/bin/bash
set -uxo pipefail
export HOME="${HOME:-/root}"
export DEBIAN_FRONTEND=noninteractive AWS_DEFAULT_REGION=@@REGION@@
OUT=/opt/pokerbot/cert_out; mkdir -p $OUT /opt/pokerbot
T_BOOT=$SECONDS
fail() { echo "$1" > $OUT/FAILED.txt; echo "$1" > $OUT/INFRA_FAIL.txt; aws s3 sync $OUT @@RES@@/ --region @@REGION@@ || true; sync; poweroff; exit 1; }
stage() { echo "$(date -u +%s) $1" >> $OUT/stage.txt; }
stage boot

# heartbeat + continuous result sync (entire run; spot reclaim never loses completed decks)
( while true; do date -u +%s > $OUT/heartbeat.txt; aws s3 sync $OUT @@RES@@/ --region @@REGION@@ --quiet || true; sleep 120; done ) &

stage deps
apt-get update -qq && apt-get install -y -qq python3-pip python3-dev build-essential curl unzip || fail deps
curl -fsSL "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o /tmp/aws.zip && unzip -q /tmp/aws.zip -d /tmp && /tmp/aws/install --update && hash -r
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal
export PATH="$HOME/.cargo/bin:$PATH"

aws s3 cp s3://@@BUCKET@@/@@PREFIX@@/cert9_search/code/cert9_code.tar.gz /tmp/code.tar.gz --region @@REGION@@ || fail codefetch
tar -xzf /tmp/code.tar.gz -C /opt/pokerbot
cd /opt/pokerbot
python3 -m pip install --disable-pip-version-check --no-input --break-system-packages boto3 numpy pandas "jax[cpu]" phevaluator flax optax cloudpickle \
  || python3 -m pip install --disable-pip-version-check --no-input boto3 numpy pandas "jax[cpu]" phevaluator flax optax cloudpickle || fail pip
python3 -m pip install --disable-pip-version-check --no-input --break-system-packages torch --index-url https://download.pytorch.org/whl/cpu \
  || python3 -m pip install --disable-pip-version-check --no-input torch --index-url https://download.pytorch.org/whl/cpu || fail torch

mkdir -p data/abstraction checkpoints runs artifacts
aws s3 cp s3://@@BUCKET@@/@@PREFIX@@/data/emd_frozen_flop_turn_160_r400.emdfrzn data/abstraction/ --region @@REGION@@ || fail blob1
aws s3 cp s3://@@BUCKET@@/@@PREFIX@@/data/river_ochs_160.rochs data/abstraction/ --region @@REGION@@ || fail blob2
aws s3 cp s3://@@BUCKET@@/@@PREFIX@@/data/serve_blueprint_6max.pkl checkpoints/ --region @@REGION@@ || fail incpkl
echo "344ab2fd328f4b360fd2c11a09c1f6fd2c45a56e62f75d6bc1a502c71612cb74  data/abstraction/emd_frozen_flop_turn_160_r400.emdfrzn" | sha256sum -c - || fail blob1sha
echo "25c860d2f35a08fcb38499d436d29547233b771e5a6b4fc057da8a72d7d53d44  data/abstraction/river_ochs_160.rochs" | sha256sum -c - || fail blob2sha

# native build (NO target-cpu override: native_solver f32 bit-identity self-check)
stage build
cd tools/battery_native && cargo build --release --features python 2>&1 | tail -3 || fail build
cp target/release/libbattery_native.so /opt/pokerbot/battery_native.so
mkdir -p target/release && cp target/release/libbattery_native.so target/release/battery_native.so || true
cd /opt/pokerbot
python3 -c "import battery_native; print('caps', sorted(battery_native.native_caps()))" | tee $OUT/native_caps.txt || fail natimport
python3 -c "import battery_native,sys; sys.exit(0 if {'river_ochs_serve','settle_showdown_leaf','presum_key','full_hist_key','action_translate','subgame_index_dir'} <= set(battery_native.native_caps()) else 1)" || fail natcaps_subgame_index_dir
POKERBOT_SELFCHECK_VERBOSE=1 python3 -c "
from pokerbot.search import native_solver as ns
import sys
print('available', ns.available(), 'selfcheck', ns._SELFCHECK_OK, flush=True)
sys.exit(0 if ns.available() else 1)" 2>&1 | tee $OUT/selfcheck_diag.log || fail selfcheck

# ---------------- SERVE INDEX: sync the ALREADY-BUILT index + sha-verify (NO snap_avg rebuild) ----------------
stage index_sync
mkdir -p @@IDX@@
aws s3 sync @@SERVE_INDEX_S3@@/ @@IDX@@/ --region @@REGION@@ --quiet || fail index_sync
( cd @@IDX@@ && sha256sum -c sha256sums.txt ) || fail index_sha
cp @@IDX@@/meta.json $OUT/serve_index_meta.json || true
export PYTHONPATH=/opt/pokerbot

# ---------------- SERVE-MISS GATE (HARD, before any money) ----------------
stage serve_gate
env @@NEWENV@@ python3 scripts/serve_miss_probe.py --blueprint @@IDX@@ --full-index @@IDX@@ \
  --mode selfplay --hands 2000 --seed 11 --miss-log $OUT/serve_gate_miss.jsonl \
  --out $OUT/serve_gate_selfplay.json 2>&1 | tail -8 | tee $OUT/serve_gate.log
python3 - <<'PY' || fail serve_gate
import json
d = json.load(open('/opt/pokerbot/cert_out/serve_gate_selfplay.json'))
mr = d["total"]["miss_rate"]
print("serve gate: self-play miss_rate=%.4f (threshold 0.005)" % mr)
assert mr < 0.005, "SERVE-MISS GATE FAIL: %.4f" % mr
PY

# ---------------- non-fatal parity gates (recorded) ----------------
stage gates_parity
( timeout 900 env @@NEWENV@@ python3 scripts/probe_fullhist_key_parity.py --fullhist 1 --presum 1 --min-states 1024 --min-postflop 1024 --out $OUT/gate_parity_fullhist.json ) > $OUT/gate_parity_fullhist.log 2>&1 || echo "parity_fullhist rc!=0" >> $OUT/section_warnings.txt
( timeout 900 env @@NEWENV@@ python3 scripts/probe_native_emd_parity.py --min-postflop 600 --min-per-street 80 --out $OUT/gate_parity_emd_ochs.json ) > $OUT/gate_parity_emd.log 2>&1 || echo "parity_emd rc!=0" >> $OUT/section_warnings.txt
aws s3 sync $OUT @@RES@@/ --region @@REGION@@ --quiet || true

# ================= PRE-SPEND PROBE (HARD GATE: proof-of-search before any money) =================
stage probe
env POKERBOT_CLOSE_CALLED_STREET=1 POKERBOT_POSTFLOP_SB_FIRST=1 POKERBOT_EMD_FROZEN_MIRROR_FALLBACK=1 \
  timeout 3600 python3 scripts/engine_ab_h2h.py \
  --arm-a '@@ARM_SEARCH@@' --arm-b '@@ARM_RAW@@' \
  --decks @@PROBE_DECKS@@ --seed 555 --label cert9_probe --require-search-native a \
  --out $OUT/probe_ab.json > $OUT/probe_ab.log 2>&1 || fail probe_run
python3 scripts/cert9_gate.py $OUT/probe_ab.json --arm a --min-postflop 16 --min-mean-wall 0.1 \
  --out $OUT/probe_gate.json 2>&1 | tee $OUT/probe_gate.log || fail probe_gate
aws s3 sync $OUT @@RES@@/ --region @@REGION@@ --quiet || true

# ================= SIZING =================
stage plan
MEMGB=$(awk '/MemTotal/ {printf "%d", $2/1048576}' /proc/meminfo)
SHARDS=8; [ "$MEMGB" -lt 180 ] && SHARDS=5
WALL_END=$(( @@WALL_SECS@@ ))
AB_END=$(( WALL_END - 7200 ))      # leave 2h: incumbent section + assemble/sync buffer
INC_END=$(( WALL_END - 1800 ))     # leave 30min: assemble + final sync
SEC_PER_DECK=$(python3 -c "import json; d=json.load(open('$OUT/probe_gate.json')); print(round(d.get('sec_per_deck') or 60.0, 1))")
echo "MEMGB=$MEMGB SHARDS=$SHARDS sec_per_deck_probe=$SEC_PER_DECK t_now=$SECONDS ab_end=$AB_END inc_end=$INC_END" | tee $OUT/plan.txt

# ================= PRIMARY: SEARCH (A) vs RAW (B), CRN-paired, deadline-bound =================
# Deck target deliberately generous; the section timeout binds. engine_ab_h2h checkpoints the out
# JSON after EVERY deck (atomic), and the heartbeat sync ships it every 120 s — a timeout/reclaim
# keeps all completed decks. --require-search-native a aborts a shard within one deck of any
# non-native/fallback serve (INFRA_FAIL, not a money number).
stage sec_ab_money
AB_TMO=$(( AB_END - SECONDS )); [ "$AB_TMO" -lt 900 ] && fail ab_no_time
pids=()
for SH in $(seq 0 $((SHARDS-1))); do
  env POKERBOT_CLOSE_CALLED_STREET=1 POKERBOT_POSTFLOP_SB_FIRST=1 POKERBOT_EMD_FROZEN_MIRROR_FALLBACK=1 \
    timeout $AB_TMO python3 scripts/engine_ab_h2h.py \
    --arm-a '@@ARM_SEARCH@@' --arm-b '@@ARM_RAW@@' \
    --decks @@DPS@@ --deck-start $SH --deck-stride $SHARDS --seed @@AB_SEED@@ \
    --label cert9_ab_s@@AB_SEED@@_sh${SH} --require-search-native a \
    --out $OUT/cert9_ab_seed@@AB_SEED@@_shard${SH}.json > $OUT/cert9_ab_shard${SH}.log 2>&1 &
  pids+=($!)
done
for p in "${pids[@]}"; do wait $p || echo "ab shard rc!=0 (timeout-bound is expected)" >> $OUT/section_warnings.txt; done
python3 scripts/assemble_cert9_verdict.py --out-dir $OUT --run-id cert9 2>&1 | tee -a $OUT/assemble.log || true
aws s3 sync $OUT @@RES@@/ --region @@REGION@@ --quiet || true

# ================= SECONDARY: SEARCH (A) vs INCUMBENT K160 (B), remaining wall =================
stage sec_inc_money
INC_TMO=$(( INC_END - SECONDS ))
if [ "$INC_TMO" -gt 1200 ]; then
  pids=()
  for SH in $(seq 0 $((SHARDS-1))); do
    env POKERBOT_CLOSE_CALLED_STREET=1 POKERBOT_POSTFLOP_SB_FIRST=1 POKERBOT_EMD_FROZEN_MIRROR_FALLBACK=1 \
      timeout $INC_TMO python3 scripts/engine_ab_h2h.py \
      --arm-a '@@ARM_SEARCH@@' --arm-b '@@ARM_INC@@' \
      --decks @@INC_DPS@@ --deck-start $SH --deck-stride $SHARDS --seed @@INC_SEED@@ \
      --label cert9_inc_s@@INC_SEED@@_sh${SH} --require-search-native a \
      --out $OUT/cert9_inc_seed@@INC_SEED@@_shard${SH}.json > $OUT/cert9_inc_shard${SH}.log 2>&1 &
    pids+=($!)
  done
  for p in "${pids[@]}"; do wait $p || echo "inc shard rc!=0 (timeout-bound is expected)" >> $OUT/section_warnings.txt; done
else
  echo "incumbent section skipped: only ${INC_TMO}s left" >> $OUT/section_warnings.txt
fi

stage verdict_final
python3 scripts/assemble_cert9_verdict.py --out-dir $OUT --run-id cert9 2>&1 | tee -a $OUT/assemble.log || echo "assemble rc!=0" >> $OUT/section_warnings.txt
aws s3 cp $OUT/VERDICT_FINAL.json @@RES@@/VERDICT_FINAL.json --region @@REGION@@ || true

stage done
date -u +"%Y-%m-%dT%H:%M:%SZ" > $OUT/RESULT_DONE.txt
aws s3 sync $OUT @@RES@@/ --region @@REGION@@
sync
poweroff
"""


if __name__ == "__main__":
    main()
