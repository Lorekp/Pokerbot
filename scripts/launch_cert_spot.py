#!/usr/bin/env python3
"""Launch ONE x86 spot box for the >=Pluribus STRENGTH CERT (NEW fullhist blueprint vs incumbent).

Reuses the PROVEN spot mechanics from launch_native_cfr_training_spot (AMI resolve, capacity
fallback chain, dead-man-cost-cap user-data stub) but renders a CERT bootstrap:

  1. deps (python + rust) + build battery_native --features python (fast default-release build);
  2. download iter_050000.nlcfr + EMD/OCHS blobs + incumbent pkl from S3;
  3. convert the FULL 148.5M-row strat table -> arrays_v1 f32 serve pkl (nlcfr_stream_filter);
  4. GATES (abort-on-fail => marker + poweroff): fullhist+presum+closure key parity,
     EMD+river-OCHS native serve parity, worker exact-zero, knob-effectiveness nonzero;
  5. BATTERY: CRN-paired engine_ab_h2h NEW-vs-INCUMBENT, both arms RAW BLUEPRINT
     (search:false — isolates the BLUEPRINT delta; search additivity is Track-2-certified
     separately), settle-showdown-leaf ON, closure ON, 2 seeds x 4000 decks, deck-sharded;
  6. POOL: pool_absolute_one_arm per arm (own process env = own serve regime), CRN deck seeds;
  7. results synced to S3 every 120 s (heartbeat) + RESULT_DONE marker; poweroff on completion.

Cost bounds: spot one-time, InstanceInitiatedShutdownBehavior=terminate, STEP-0 dead-man shutdown
floor via user_data_stub (wall-guard), plus the driver's own poweroff. Run a local poller on top.
"""
from __future__ import annotations

import argparse
import base64
import gzip
import hashlib
import io
import json
import os
import sys
import tarfile
from pathlib import Path

import boto3
from botocore.exceptions import ClientError

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

from launch_native_cfr_training_spot import (  # noqa: E402  (proven helpers, same dir)
    _launch_with_fallback, resolve_ubuntu_ami, user_data_stub,
)

BUCKET = "video-processing-6789"
PREFIX = "pokerbot/prod6000/cert_pluribus"
REGION = "eu-central-1"
PROFILE = "pokerbot-labelgen-worker"
NLCFR_S3 = ("s3://video-processing-6789/pokerbot/prod6000/native_cfr/"
            "bpv4_pluribus_finemenu_1/checkpoints/iter_050000.nlcfr")

_EXCLUDE_DIRS = ("__pycache__", ".git", "target", "artifacts", "runs")

_CODE_INCLUDE = [
    "pokerbot",
    "tools/battery_native",
    "scripts/engine_ab_h2h.py",
    "scripts/h2h_arm_worker.py",
    "scripts/pool_absolute_one_arm.py",
    "scripts/track2_s3c_pool.py",
    "scripts/nlcfr_stream_filter.py",
    "scripts/cert_verdict_merge.py",
    "scripts/probe_fullhist_key_parity.py",
    "scripts/probe_native_emd_parity.py",
    "scripts/f32_regret_strategy_delta.py",
    "data/weva",
    "data/player_profiles",
]

_DATA_UPLOADS = [  # (local, s3 key under PREFIX/data/) — big/stable blobs shipped OUTSIDE the tarball
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
    key = f"{PREFIX}/code/cert_code.tar.gz"
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


def _render_bootstrap(*, run_id: str, decks_per_seed: int, seeds: str, pool_decks: int,
                      pool_only: bool = False) -> str:
    res = f"s3://{BUCKET}/{PREFIX}/{run_id}/results"
    newenv = ("POKERBOT_FULLHIST_KEY=1 POKERBOT_PRESUM_KEY=1 POKERBOT_MRR=1 "
              "POKERBOT_CLOSE_CALLED_STREET=1 POKERBOT_PRE_KEEP=2,3,4,5,6 "
              "POKERBOT_PRE_RR_KEEP=4,5,6 "
              "POKERBOT_RIVER_OCHS=/opt/pokerbot/data/abstraction/river_ochs_160.rochs")
    newenv_json = json.dumps({
        "POKERBOT_FULLHIST_KEY": "1", "POKERBOT_PRESUM_KEY": "1", "POKERBOT_MRR": "1",
        "POKERBOT_CLOSE_CALLED_STREET": "1", "POKERBOT_PRE_KEEP": "2,3,4,5,6",
        "POKERBOT_PRE_RR_KEEP": "4,5,6",
        "POKERBOT_RIVER_OCHS": "/opt/pokerbot/data/abstraction/river_ochs_160.rochs",
    })
    incenv_json = json.dumps({"POKERBOT_CLOSE_CALLED_STREET": "1"})
    # RAW arms: search:false serves the blueprint root policy directly (no subgame solve) —
    # the cert isolates the BLUEPRINT object; both arms still settle showdown money (harness
    # money fix default-ON) and play under closure (parent env).
    arm_new = json.dumps({"worker_env": json.loads(newenv_json), "settle_showdown_leaf": True,
                          "search": False,
                          "blueprint_path": "/opt/pokerbot/checkpoints/bp_new_full.pkl"})
    arm_inc = json.dumps({"worker_env": json.loads(incenv_json), "settle_showdown_leaf": True,
                          "search": False})
    head = f"""#!/bin/bash
set -uxo pipefail
export HOME="${{HOME:-/root}}"
export DEBIAN_FRONTEND=noninteractive AWS_DEFAULT_REGION={REGION}
OUT=/opt/pokerbot/cert_out; mkdir -p $OUT /opt/pokerbot
fail() {{ echo "$1" > $OUT/FAILED.txt; aws s3 sync $OUT {res}/ --region {REGION} || true; sync; poweroff; exit 1; }}
stage() {{ echo "$(date -u +%s) $1" >> $OUT/stage.txt; }}
stage boot

# heartbeat + result sync loop (background, entire run)
( while true; do date -u +%s > $OUT/heartbeat.txt; aws s3 sync $OUT {res}/ --region {REGION} --quiet || true; sleep 120; done ) &

stage deps
apt-get update -qq && apt-get install -y -qq python3-pip python3-dev build-essential curl unzip || fail deps
curl -fsSL "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o /tmp/aws.zip && unzip -q /tmp/aws.zip -d /tmp && /tmp/aws/install --update && hash -r
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal
export PATH="$HOME/.cargo/bin:$PATH"

aws s3 cp s3://{BUCKET}/{PREFIX}/code/cert_code.tar.gz /tmp/code.tar.gz --region {REGION} || fail codefetch
tar -xzf /tmp/code.tar.gz -C /opt/pokerbot
cd /opt/pokerbot
python3 -m pip install --disable-pip-version-check --no-input --break-system-packages boto3 numpy "jax[cpu]" phevaluator flax optax cloudpickle \
  || python3 -m pip install --disable-pip-version-check --no-input boto3 numpy "jax[cpu]" phevaluator flax optax cloudpickle || fail pip
# CPU torch: pokerbot.search.state_bridge -> pbs_tracker -> sequence_encoder imports torch
# (cert2 failure; same fallback the eval-suite launcher ships)
python3 -m pip install --disable-pip-version-check --no-input --break-system-packages torch --index-url https://download.pytorch.org/whl/cpu \
  || python3 -m pip install --disable-pip-version-check --no-input torch --index-url https://download.pytorch.org/whl/cpu || fail torch

mkdir -p data/abstraction checkpoints
aws s3 cp s3://{BUCKET}/{PREFIX}/data/emd_frozen_flop_turn_160_r400.emdfrzn data/abstraction/ --region {REGION} || fail blob1
aws s3 cp s3://{BUCKET}/{PREFIX}/data/river_ochs_160.rochs data/abstraction/ --region {REGION} || fail blob2
aws s3 cp s3://{BUCKET}/{PREFIX}/data/serve_blueprint_6max.pkl checkpoints/ --region {REGION} || fail incpkl
echo "344ab2fd328f4b360fd2c11a09c1f6fd2c45a56e62f75d6bc1a502c71612cb74  data/abstraction/emd_frozen_flop_turn_160_r400.emdfrzn" | sha256sum -c - || fail blob1sha
echo "25c860d2f35a08fcb38499d436d29547233b771e5a6b4fc057da8a72d7d53d44  data/abstraction/river_ochs_160.rochs" | sha256sum -c - || fail blob2sha

# native build (default release profile — minutes, not the trainer's lto=fat hour)
# NO target-cpu override: the native_solver startup self-check requires BIT-IDENTICAL
# Rust-vs-Python f32 bet-amount arithmetic; x86-64-v3 FMA contraction broke it on cert1
# (locally-validated builds use the default target). Baseline codegen is plenty for serve.
stage build
cd tools/battery_native && cargo build --release --features python 2>&1 | tail -3 || fail build
cp target/release/libbattery_native.so /opt/pokerbot/battery_native.so
mkdir -p target/release && cp target/release/libbattery_native.so target/release/battery_native.so || true
cd /opt/pokerbot
python3 -c "import battery_native; print('caps', sorted(battery_native.native_caps()))" || fail natimport
python3 -c "import battery_native,sys; sys.exit(0 if {{'river_ochs_serve','settle_showdown_leaf','presum_key','full_hist_key'}} <= set(battery_native.native_caps()) else 1)" || fail natcaps
# explicit native_solver self-check with verbose diagnostics (the cert1 failure surface)
POKERBOT_SELFCHECK_VERBOSE=1 python3 -c "
from pokerbot.search import native_solver as ns
import sys
print('available', ns.available(), 'selfcheck', ns._SELFCHECK_OK, flush=True)
sys.exit(0 if ns.available() else 1)" 2>&1 | tee $OUT/selfcheck_diag.log || fail selfcheck

# blueprint download + FULL-table conversion (arrays_v1 f32)
stage nlcfr_download
aws s3 cp {NLCFR_S3} /opt/pokerbot/iter_050000.nlcfr --region {REGION} || fail nlcfr
stage convert
python3 scripts/nlcfr_stream_filter.py /opt/pokerbot/iter_050000.nlcfr checkpoints/bp_new_full.pkl \\
  --mass 1.0 --max-rows 200000000 --arrays 2>&1 | tee $OUT/convert.log || fail convert
rm -f /opt/pokerbot/iter_050000.nlcfr

export PYTHONPATH=/opt/pokerbot
"""
    mid = "" if pool_only else f"""# ---------------- GATES (abort-on-fail) ----------------
stage gates
{newenv} python3 scripts/probe_fullhist_key_parity.py --fullhist 1 --presum 1 \\
  --min-states 1024 --min-postflop 1024 --out $OUT/gate_parity_fullhist.json 2>&1 | tail -3 | tee $OUT/gate_parity_fullhist.log || fail gate_parity_fullhist
{newenv} python3 scripts/probe_native_emd_parity.py --min-postflop 600 --min-per-street 80 \\
  --out $OUT/gate_parity_emd_ochs.json 2>&1 | tail -4 | tee $OUT/gate_parity_emd_ochs.log || fail gate_parity_emd_ochs
POKERBOT_CLOSE_CALLED_STREET=1 python3 scripts/engine_ab_h2h.py \\
  --arm-a '{arm_new}' --arm-b '{arm_new}' \\
  --decks 6 --seed 4242 --label box_zero_gate --out $OUT/gate_zero.json 2>&1 | tail -6 | tee $OUT/gate_zero.log
python3 - <<'PY' || fail gate_zero
import json; d=json.load(open('/opt/pokerbot/cert_out/gate_zero.json'))
# cert4 postmortem: ci95 is a LIST [lo, hi] — abs(list) raised TypeError and spuriously failed
# a PERFECT-zero gate. Check mean + both CI bounds + every per-deck net element-wise.
assert abs(d["a_minus_b_per_deck"]) < 1e-9, d["a_minus_b_per_deck"]
assert all(abs(x) < 1e-9 for x in d["ci95"]), d["ci95"]
assert all(abs(x) < 1e-9 for x in d["per_deck_nets"]), d["per_deck_nets"]
PY
POKERBOT_CLOSE_CALLED_STREET=1 python3 scripts/engine_ab_h2h.py \\
  --arm-a '{arm_new}' --arm-b '{arm_inc}' \\
  --decks 40 --seed 4242 --label box_knob_screen --out $OUT/gate_screen.json 2>&1 | tail -6 | tee $OUT/gate_screen.log
python3 - <<'PY' || fail gate_screen
import json, numpy as np; d=json.load(open('/opt/pokerbot/cert_out/gate_screen.json'))
nz = sum(1 for x in d["per_deck_nets"] if abs(x) > 1e-9)
assert nz >= 5, f"only {{nz}} nonzero per-deck diffs — knobs not flowing"
PY

# ---------------- BATTERY: NEW vs INCUMBENT, sharded ----------------
MEMGB=$(awk '/MemTotal/ {{printf "%d", $2/1048576}}' /proc/meminfo)
# RAW arms: each shard = parent + 2 workers; the NEW-arm worker holds the 8.91GB arrays pkl
# (~20-30GB resident). Shard counts sized so worst-case worker RSS stays < half of RAM.
SHARDS=2; [ "$MEMGB" -gt 400 ] && SHARDS=6; [ "$MEMGB" -gt 200 ] && [ "$MEMGB" -le 400 ] && SHARDS=3
DPS=$(( {decks_per_seed} / SHARDS ))
echo "MEMGB=$MEMGB SHARDS=$SHARDS decks/shard=$DPS" | tee $OUT/plan.txt
for SEED in {seeds}; do
  stage battery_seed$SEED
  pids=()
  for SH in $(seq 0 $((SHARDS-1))); do
    POKERBOT_CLOSE_CALLED_STREET=1 python3 scripts/engine_ab_h2h.py \\
      --arm-a '{arm_new}' --arm-b '{arm_inc}' \\
      --decks $DPS --deck-start $SH --deck-stride $SHARDS --seed $SEED \\
      --label cert_new_vs_inc_s${{SEED}}_sh${{SH}} \\
      --out $OUT/h2h_seed${{SEED}}_shard${{SH}}.json > $OUT/h2h_seed${{SEED}}_shard${{SH}}.log 2>&1 &
    pids+=($!)
  done
  for p in "${{pids[@]}}"; do wait $p || echo "shard rc!=0 (seed $SEED)" >> $OUT/battery_warnings.txt; done
done
"""
    tail = f"""# ---------------- POOL: absolute vs roster, one process per arm-env ----------------
stage pool
{newenv} python3 scripts/pool_absolute_one_arm.py \\
  --blueprint checkpoints/bp_new_full.pkl --spec '{{"search": false, "settle_showdown_leaf": true}}' \\
  --decks {pool_decks} --seed 0 --workers 1 --label pool_new --out $OUT/pool_new.json > $OUT/pool_new.log 2>&1 &
PNEW=$!
POKERBOT_CLOSE_CALLED_STREET=1 python3 scripts/pool_absolute_one_arm.py \\
  --blueprint checkpoints/serve_blueprint_6max.pkl --spec '{{"search": false, "settle_showdown_leaf": true}}' \\
  --decks {pool_decks} --seed 0 --workers 1 --label pool_inc --out $OUT/pool_inc.json > $OUT/pool_inc.log 2>&1 &
PINC=$!
wait $PNEW || echo "pool_new rc!=0" >> $OUT/battery_warnings.txt
wait $PINC || echo "pool_inc rc!=0" >> $OUT/battery_warnings.txt

stage verdict
python3 scripts/cert_verdict_merge.py --out-dir $OUT --run-id {run_id} 2>&1 | tee $OUT/verdict_merge.log \\
  || echo "verdict merge rc!=0" >> $OUT/battery_warnings.txt

stage done
date -u +"%Y-%m-%dT%H:%M:%SZ" > $OUT/RESULT_DONE.txt
aws s3 sync $OUT {res}/ --region {REGION}
sync
poweroff
"""
    return head + mid + tail


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default="cert1")
    ap.add_argument("--decks-per-seed", type=int, default=4000)
    ap.add_argument("--seeds", default="101 202")
    ap.add_argument("--pool-decks", type=int, default=600)
    ap.add_argument("--pool-only", action="store_true",
                    help="skip gates+battery; run ONLY the pool absolutes (e.g. after a battery-"
                         "complete run whose pool leg failed)")
    ap.add_argument("--wall-hours", type=float, default=12.0)
    ap.add_argument("--instance-types", default="r8i.32xlarge,r6i.32xlarge,c6i.32xlarge")
    ap.add_argument("--on-demand", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    session = boto3.Session(region_name=REGION)
    s3 = session.client("s3")
    ec2 = session.client("ec2")

    _upload_data(s3)
    _upload_code(s3)

    bootstrap = _render_bootstrap(run_id=args.run_id, decks_per_seed=args.decks_per_seed,
                                  seeds=args.seeds, pool_decks=args.pool_decks,
                                  pool_only=args.pool_only)
    bkey = f"{PREFIX}/{args.run_id}/code/bootstrap.sh"
    s3.put_object(Bucket=BUCKET, Key=bkey, Body=bootstrap.encode())
    print(f"uploaded bootstrap ({len(bootstrap)} bytes) -> s3://{BUCKET}/{bkey}", flush=True)
    if args.dry_run:
        print("--dry-run: not launching"); return

    wall = int(args.wall_hours * 3600)
    ud = user_data_stub(bucket=BUCKET, region=REGION, run_prefix=f"{PREFIX}/{args.run_id}",
                        bootstrap_key=bkey, wall_guard_secs=wall)
    ami, root_dev = resolve_ubuntu_ami(ec2, "x86_64")
    run_kwargs = dict(
        ImageId=ami, MinCount=1, MaxCount=1,
        UserData=base64.b64encode(ud.encode()).decode(),
        IamInstanceProfile={"Name": PROFILE},
        MetadataOptions={"HttpTokens": "required", "HttpPutResponseHopLimit": 2},
        TagSpecifications=[{"ResourceType": "instance", "Tags": [
            {"Key": "Name", "Value": f"cert-pluribus-{args.run_id}"},
            {"Key": "Project", "Value": "pokerbot-cert"},
            {"Key": "Run", "Value": args.run_id},
        ]}],
        InstanceInitiatedShutdownBehavior="terminate",
        BlockDeviceMappings=[{"DeviceName": root_dev,
                              "Ebs": {"VolumeSize": 300, "VolumeType": "gp3", "Iops": 3000,
                                      "Throughput": 500, "DeleteOnTermination": True}}],
    )
    if not args.on_demand:
        run_kwargs["InstanceMarketOptions"] = {
            "MarketType": "spot",
            "SpotOptions": {"SpotInstanceType": "one-time",
                            "InstanceInterruptionBehavior": "terminate"},
        }
    chain = [t.strip() for t in args.instance_types.split(",") if t.strip()]
    iid, used = _launch_with_fallback(ec2, run_kwargs, instance_types=chain,
                                      spot=not args.on_demand)
    print(f"LAUNCHED {iid} as {used} ({'spot' if not args.on_demand else 'on-demand'})")
    print(f"results: s3://{BUCKET}/{PREFIX}/{args.run_id}/results/")
    print(f"console: aws ec2 get-console-output --instance-id {iid} --region {REGION} --latest")
    print(f"terminate: aws ec2 terminate-instances --instance-ids {iid} --region {REGION}")


if __name__ == "__main__":
    main()
