#!/bin/bash
# On-box logic for the DCFR-vs-CFR+ LBR-vs-compute CURVE cert + work_steal scaling probe. The dead-man
# cost cap (logind shutdown floor + pokerbot-dcfrguard unit) is armed by user_data STEP-0 BEFORE this
# runs, so any hang/build-fail here can never idle-bill. No EMD blob: made-hand abstraction only (so the
# native made-hand LBR can score the blueprints locally), built with a plain --features python (work_steal
# and dcfr are RUNTIME toggles, not cargo features) -- the production-buildable path.
set -uxo pipefail
export HOME="${HOME:-/root}"
cd /opt/pokerbot
mkdir -p /var/log/dcfrcurve

S3="s3://${BUCKET}/${RESULT_PREFIX}"
say() { echo "[$(date +%H:%M:%S)] $*"; }
upload() { timeout 180 aws s3 cp "$1" "${S3}/$2" --region "${REGION}" 2>/dev/null || echo "WARN upload $2"; }
uploaddir() { timeout 180 aws s3 cp --recursive /var/log/dcfrcurve "${S3}/" --region "${REGION}" 2>/dev/null || true; }
fail() { echo "$1" | tee /var/log/dcfrcurve/STATUS; upload /var/log/dcfrcurve/STATUS STATUS; uploaddir; }

# ---- build the native module (ONE binary; work_steal/dcfr are runtime toggles) ----
export PATH="$HOME/.cargo/bin:$PATH"
export PYO3_PYTHON="$(command -v python3)"
say "cargo build --release --features python (PYO3_PYTHON=${PYO3_PYTHON})"
if ! RUSTFLAGS='-C target-cpu=x86-64-v3' cargo build --release --features python --lib \
       --manifest-path tools/battery_native/Cargo.toml 2>&1 | tee /var/log/dcfrcurve/build.log; then
  fail "BUILD_FAIL"; upload /var/log/dcfrcurve/build.log build.log; shutdown -h now; exit 1
fi
SO=tools/battery_native/target/release/libbattery_native.so
if [ ! -f "${SO}" ]; then fail "BUILD_NO_SO"; shutdown -h now; exit 1; fi
mkdir -p /opt/pokerbot/pyd && cp "${SO}" /opt/pokerbot/pyd/battery_native.so
say "built + staged battery_native.so"

# ---- toggle smoke: both arms constructible, work_steal/dcfr accepted, hogwild OFF (fail fast) ----
export PYD_DIR=/opt/pokerbot/pyd
if ! python3 - <<'PYSMOKE' 2>&1 | tee /var/log/dcfrcurve/smoke.log
import os, sys
sys.path.insert(0, "/opt/pokerbot/pyd")
import battery_native as bn
# cfrplus arm: work_steal ON, dcfr OFF
a = bn.NlheCFRTrainer(200, [], 256, 6, float('-inf'), False, None, 0, False, 1<<21, True, False, -300000.0, 0.05)
# dcfr arm: work_steal ON, dcfr ON
b = bn.NlheCFRTrainer(200, [], 256, 6, float('-inf'), False, None, 0, False, 1<<21, True, True, -300000.0, 0.05)
assert a.is_hogwild() is False and b.is_hogwild() is False
a.train(2); b.train(2)
print("SMOKE_OK cfrplus_infosets=%d dcfr_infosets=%d" % (a.n_infosets(), b.n_infosets()))
PYSMOKE
then
  fail "SMOKE_FAIL"; upload /var/log/dcfrcurve/smoke.log smoke.log; shutdown -h now; exit 1
fi

# ---- work_steal SCALING probe FIRST (short, ~5min): the long-owed 32/64/128c load-imbalance number.
#      Static (WS=0) vs work_steal (WS=1) x threads; RAYON pool fixed per subprocess from the env. ----
say "work_steal scaling probe (static vs work_steal x ${PROBE_THREADS_LIST})"
export DCFR_NUM_BUCKETS="${DCFR_NUM_BUCKETS}" PROBE_N_DEALS="${PROBE_N_DEALS}"
export WARM_ITERS="${WARM_ITERS}" WINDOW_ITERS="${WINDOW_ITERS}"
: > /var/log/dcfrcurve/ws_probe.jsonl
for ws in 0 1; do
  for th in ${PROBE_THREADS_LIST//,/ }; do
    say "  probe ws=${ws} threads=${th}"
    RAYON_NUM_THREADS=${th} WS=${ws} timeout 240 python3 scripts/dcfr_ws_probe.py \
      >> /var/log/dcfrcurve/ws_probe.jsonl 2>>/var/log/dcfrcurve/ws_probe.err || echo "WARN probe ws=${ws} th=${th}"
  done
done
upload /var/log/dcfrcurve/ws_probe.jsonl ws_probe.jsonl
upload /var/log/dcfrcurve/ws_probe.err ws_probe.err
say "probe done"

# ---- CURVE training: both arms, checkpoints STREAMED to S3 per-milestone (no end-only upload). Train
#      threads capped per RAYON_NUM_THREADS (64 = physical cores; SMT-negative above that). ----
say "CURVE training (cfrplus, dcfr) RAYON_NUM_THREADS=${TRAIN_RAYON}"
export DCFR_N_DEALS="${DCFR_N_DEALS}" DCFR_MILESTONES="${DCFR_MILESTONES}"
export DCFR_PRUNE_THRESHOLD="${DCFR_PRUNE_THRESHOLD}" DCFR_EXPLORE_FRAC="${DCFR_EXPLORE_FRAC}"
export OUT_DIR=/var/log/dcfrcurve S3_DEST="${S3}" REGION="${REGION}"
RAYON_NUM_THREADS=${TRAIN_RAYON} timeout "${TRAIN_TIMEOUT}" python3 scripts/drive_dcfr_curve_train.py 2>&1 \
  | tee /var/log/dcfrcurve/train.log
RC=${PIPESTATUS[0]}
say "curve train exit ${RC}"

echo "DCFR_CURVE_DONE rc=${RC}" | tee /var/log/dcfrcurve/STATUS
# emit a PROBE_DONE-named marker so _probe_watch.py (DONE_KEYS=('PROBE_DONE',)) classifies this completed
echo "rc=${RC}" > /var/log/dcfrcurve/PROBE_DONE
uploaddir
upload /var/log/dcfrcurve/PROBE_DONE PROBE_DONE
say "uploaded results to ${S3}/"
