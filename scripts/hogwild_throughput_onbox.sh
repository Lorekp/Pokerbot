#!/bin/bash
# On-box heavy logic for the Hogwild throughput-crossover run (download EMD -> build hogwild binary ->
# deals/s sweep -> upload). The dead-man cost cap (logind shutdown floor + pokerbot-wallguard unit) is
# armed by user_data STEP-0 BEFORE this script runs, so a hang/build-fail here can never idle-bill.
set -uxo pipefail
export HOME="${HOME:-/root}"
cd /opt/pokerbot
mkdir -p /var/log/hogwild data/abstraction

S3="s3://${BUCKET}/${RESULT_PREFIX}"
say() { echo "[$(date +%H:%M:%S)] $*"; }
upload() { timeout 120 aws s3 cp "$1" "${S3}/$2" --region "${REGION}" 2>/dev/null || echo "WARN upload $2"; }
fail() { echo "$1" | tee /var/log/hogwild/STATUS; upload /var/log/hogwild/STATUS STATUS; \
         timeout 120 aws s3 cp --recursive /var/log/hogwild "${S3}/" --region "${REGION}" 2>/dev/null || true; }

# ---- frozen EMD blob: download from S3 (pre-uploaded by the launcher) + verify sha256 ----
FROZEN_LOCAL="data/abstraction/emd_frozen_flop_turn_160_r400.emdfrzn"
say "downloading frozen EMD blob s3://${BUCKET}/${FROZEN_KEY}"
if ! timeout 300 aws s3 cp "s3://${BUCKET}/${FROZEN_KEY}" "${FROZEN_LOCAL}" --region "${REGION}"; then
  fail "FROZEN_DOWNLOAD_FAIL"; shutdown -h now; exit 1
fi
GOT_SHA=$(sha256sum "${FROZEN_LOCAL}" | awk '{print $1}')
say "frozen sha256=${GOT_SHA} expected=${EXPECTED_FROZEN_SHA256}"
if [ "${GOT_SHA}" != "${EXPECTED_FROZEN_SHA256}" ]; then
  fail "FROZEN_SHA_MISMATCH ${GOT_SHA}"; shutdown -h now; exit 1
fi

# ---- build the hogwild-enabled native module (ONE binary; runtime toggle serves both arms) ----
export PATH="$HOME/.cargo/bin:$PATH"
export PYO3_PYTHON="$(command -v python3)"
say "cargo build --release --features python,hogwild (PYO3_PYTHON=${PYO3_PYTHON})"
if ! RUSTFLAGS='-C target-cpu=x86-64-v3' cargo build --release --features python,hogwild --lib \
       --manifest-path tools/battery_native/Cargo.toml 2>&1 | tee /var/log/hogwild/build.log; then
  fail "BUILD_FAIL"; upload /var/log/hogwild/build.log build.log; shutdown -h now; exit 1
fi
SO=tools/battery_native/target/release/libbattery_native.so
if [ ! -f "${SO}" ]; then fail "BUILD_NO_SO"; shutdown -h now; exit 1; fi
mkdir -p /opt/pokerbot/pyd && cp "${SO}" /opt/pokerbot/pyd/battery_native.so
say "built + staged battery_native.so"

# quick import + both-arms toggle smoke (fail fast before the sweep)
if ! PYD_DIR=/opt/pokerbot/pyd python3 - <<'PYSMOKE' 2>&1 | tee /var/log/hogwild/smoke.log
import os, sys
sys.path.insert(0, "/opt/pokerbot/pyd")
import battery_native as bn
emd = "/opt/pokerbot/data/abstraction/emd_frozen_flop_turn_160_r400.emdfrzn"
f = bn.NlheCFRTrainer(160, [], 200, 6, float('-inf'), False, emd, 0, False, 1<<21)
h = bn.NlheCFRTrainer(160, [], 200, 6, float('-inf'), False, emd, 0, True, 1<<21)
assert f.is_hogwild() is False and h.is_hogwild() is True
print("SMOKE_OK is_hogwild fork=%s hog=%s emd_stamp=%s" % (f.is_hogwild(), h.is_hogwild(), h.emd_stamp()))
PYSMOKE
then
  fail "SMOKE_FAIL"; upload /var/log/hogwild/smoke.log smoke.log; shutdown -h now; exit 1
fi

# ---- QUALITY training (made-hand, both arms) FIRST — so a later throughput hang never loses the
#      correctness blueprints. Trained here (where hogwild is fast at high core count); LBR scored LOCALLY
#      (the native root-dealer needs JAX, kept off the box). Blueprints uploaded immediately. ----
say "QUALITY training (made-hand, both arms; LBR scored locally)"
export PYD_DIR=/opt/pokerbot/pyd OUT_DIR=/var/log/hogwild
export QUAL_NUM_BUCKETS="${QUAL_NUM_BUCKETS}" QUAL_N_DEALS="${QUAL_N_DEALS}" QUAL_ITERS="${QUAL_ITERS}"
timeout 900 env PYTHONPATH=/opt/pokerbot python3 scripts/drive_hogwild_quality_train.py 2>&1 \
  | tee /var/log/hogwild/qual.log
QRC=${PIPESTATUS[0]}
say "quality train exit ${QRC}"
timeout 180 aws s3 cp --recursive /var/log/hogwild "${S3}/" --region "${REGION}" 2>/dev/null \
  || echo "WARN qual upload"

# ---- run the crossover sweep ----
say "running throughput crossover sweep"
export PYD_DIR=/opt/pokerbot/pyd
export EMD_FROZEN_PATH=/opt/pokerbot/data/abstraction/emd_frozen_flop_turn_160_r400.emdfrzn
export OUT_DIR=/var/log/hogwild
export NUM_BUCKETS="${NUM_BUCKETS}"
export THREADS_LIST="${THREADS_LIST}" N_DEALS_LIST="${N_DEALS_LIST}" PROD_N_DEALS="${PROD_N_DEALS}"
export WINDOW_SECS="${WINDOW_SECS}" WARM_SECS="${WARM_SECS}"
# C2: bound the whole sweep so a wedged config self-aborts to shutdown well before the wall-guard floor.
# 2700s (45min) covers the 24-config envelope sweep (incl. per-subprocess 152MB EMD reloads) with margin.
# quality(900) + throughput(2700) = 3600s worst case, nested INSIDE the 4200s wall-guard + +75min floor.
timeout 2700 env PYTHONPATH=/opt/pokerbot python3 scripts/drive_hogwild_throughput.py 2>&1 \
  | tee /var/log/hogwild/sweep.log
RC=${PIPESTATUS[0]}
say "sweep exit ${RC}"

echo "HOGWILD_THROUGHPUT_DONE rc=${RC}" | tee /var/log/hogwild/STATUS
# C1: emit a PROBE_DONE-named marker so the external _probe_watch.py poller (DONE_KEYS=('PROBE_DONE',))
# classifies this run as completed rather than only ever seeing the box self-terminate.
echo "rc=${RC} $(cat /var/log/hogwild/DRIVER_STATUS 2>/dev/null || echo '?')" > /var/log/hogwild/PROBE_DONE
timeout 180 aws s3 cp --recursive /var/log/hogwild "${S3}/" --region "${REGION}" 2>/dev/null || echo "WARN final upload"
say "uploaded results to ${S3}/"
