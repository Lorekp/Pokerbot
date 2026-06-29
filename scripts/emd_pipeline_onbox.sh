#!/bin/bash
# ON-BOX driver for the 5-PHASE EMD BLUEPRINT PIPELINE. Shipped INSIDE the code tarball and exec'd by the
# MINIMAL cloud-init bootstrap (scripts/launch_emd_pipeline_spot.py:user_data) so the inline EC2 user_data
# stays well under the 16 KB limit. Everything heavy lives here.
#
# PHASES (run on EVERY boot; each writes an S3 completion MARKER so a relaunch RESUMES at the first
# incomplete phase and NEVER re-does finished work -- a kill during ship/eval must NOT re-train):
#   0 BUILD      : cargo build --release --features python of the EMD crate -> battery_native.so
#   1 PRECOMPUTE : ensure the frozen (hole,board)->bucket table (S3 cache by TABLE_ID; download or build+upload)
#                  -> marker table_cached
#   2 TRAIN      : native CFR+ to the iters target, resuming from the S3 latest checkpoint; on a clean DONE
#                  -> marker CONVERGED (the babysitter also writes CONVERGED on its convergence-stop)
#   4 SHIP       : convert the converged .nlcfr -> serve snapshot, pass the serve-skew gate
#                  (ship_blueprint.py --emd-frozen), STAGE it (no --activate) -> marker SHIPPED
#   5 EVAL       : LBR the new EMD blueprint AND iter62000 on the SAME CRN seeds + the always-fold anchor,
#                  emit the per-seat blind verdict JSON to S3 -> marker EVALED
#
# CONFIG comes entirely from EXPORTED ENV (set by the bootstrap). All paths/IDs are passed in, so this file
# is abstraction-agnostic and never needs editing per run. ASCII-only, POSIX-bash.
set -uxo pipefail
export HOME="${HOME:-/root}"   # cloud-init runs as root with HOME UNSET; under set -u a bare $HOME is FATAL

# ---- required config (exported by the bootstrap) ----
: "${BUCKET:?}" "${PREFIX:?}" "${REGION:?}" "${RUN_SUBDIR:?}" "${CODE_DIR:?}"
: "${ITERS:?}" "${N_DEALS:?}" "${NUM_BUCKETS:?}" "${CKPT_EVERY:?}" "${DEPTH_CSV=}"
: "${CENTROIDS_REL:?}" "${N_BINS:?}" "${N_RUNOUTS:?}" "${TABLE_ID:?}"
: "${SHIP_NAME:?}" "${ITER62000_SNAPSHOT_S3:?}" "${EVAL_SEEDS:?}" "${EVAL_HANDS:?}"
: "${STOP_AFTER_SHIP:=0}"

CENTROIDS_REL="$(printf '%s' "$CENTROIDS_REL" | tr '\\' '/')"
RUN_PREFIX="${PREFIX}/${RUN_SUBDIR}"
export AWS_DEFAULT_REGION="${REGION}"
cd "${CODE_DIR}"
export PYTHONPATH="${CODE_DIR}" RUST_BACKTRACE=1
export PYO3_PYTHON="$(python3 -c 'import sys;print(sys.executable)')"
export RAYON_NUM_THREADS="${RAYON_NUM_THREADS:-$(nproc)}"
echo "RAYON_NUM_THREADS=${RAYON_NUM_THREADS}"

# ===== S3 PHASE MARKERS =====
MARK_PREFIX="s3://${BUCKET}/${RUN_PREFIX}/markers"
M_TABLE="${MARK_PREFIX}/table_cached"
M_CONVERGED="${MARK_PREFIX}/CONVERGED"
M_SHIPPED="${MARK_PREFIX}/SHIPPED"
M_EVALED="${MARK_PREFIX}/EVALED"

# After user_data installs boto3, apt's awscli v1 can become botocore-incompatible. Keep all on-box S3
# phase I/O on boto3 so marker/frozen/ship persistence cannot silently diverge from checkpoint uploads.
_s3_py() {
  python3 - "$REGION" "$@" <<'PYEOF'
import pathlib
import sys
from urllib.parse import urlparse

import boto3

region, op, uri = sys.argv[1], sys.argv[2], sys.argv[3]
p = urlparse(uri)
if p.scheme != "s3" or not p.netloc or not p.path:
    raise SystemExit(f"bad s3 uri: {uri}")
bucket, key = p.netloc, p.path.lstrip("/")
s3 = boto3.client("s3", region_name=region)

try:
    if op == "exists":
        s3.head_object(Bucket=bucket, Key=key)
    elif op == "download":
        dst = pathlib.Path(sys.argv[4])
        dst.parent.mkdir(parents=True, exist_ok=True)
        s3.download_file(bucket, key, str(dst))
    elif op == "upload":
        s3.upload_file(sys.argv[4], bucket, key)
    elif op == "put-text":
        s3.put_object(Bucket=bucket, Key=key, Body=sys.argv[4].encode("utf-8"))
    else:
        raise SystemExit(f"bad op: {op}")
except Exception as e:
    print(f"S3_{op.upper()}_FAIL {uri}: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
    raise SystemExit(1)
PYEOF
}
s3_download() { _s3_py download "$1" "$2"; }
s3_upload() { _s3_py upload "$2" "$1"; }
has_marker() { _s3_py exists "$1" >/dev/null 2>&1; }
put_marker() { _s3_py put-text "$1" "$(date -u +%FT%TZ) $2"; }
upload_logs() {
  set +e
  RUN_PREFIX_FOR_LOGS="${RUN_PREFIX}" BUCKET_FOR_LOGS="${BUCKET}" REGION_FOR_LOGS="${REGION}" python3 - <<'LOGEOF'
import os, boto3
s3 = boto3.client("s3", region_name=os.environ["REGION_FOR_LOGS"])
bucket = os.environ["BUCKET_FOR_LOGS"]
rp = os.environ["RUN_PREFIX_FOR_LOGS"]
for src, key in [("/var/log/cloud-init-output.log", rp + "/logs/cloud-init-output.log"),
                 ("/var/log/emd-precompute.log", rp + "/logs/precompute.log"),
                 ("/var/log/emd-train.log", rp + "/logs/train.log"),
                 ("/var/log/emd-ship.log", rp + "/logs/ship.log"),
                 ("/var/log/emd-eval-emd.log", rp + "/logs/eval_emd.log"),
                 ("/var/log/emd-eval-base.log", rp + "/logs/eval_iter62000.log"),
                 ("/var/log/emd-eval-combine.log", rp + "/logs/eval_combine.log")]:
    try:
        s3.upload_file(src, bucket, key); print("uploaded", key)
    except Exception as e:
        print("log upload fail", src, e)
LOGEOF
}
start_log_sync() {
  ( while true; do
      echo "PHASE=eval LOG_SYNC $(date -u +%FT%TZ)"
      upload_logs >/dev/null 2>&1
      sleep 60
    done ) &
  LOG_SYNC_PID=$!
}
stop_log_sync() {
  if [ -n "${LOG_SYNC_PID:-}" ]; then
    kill "$LOG_SYNC_PID" 2>/dev/null || true
    wait "$LOG_SYNC_PID" 2>/dev/null || true
    LOG_SYNC_PID=""
  fi
}
trap upload_logs EXIT

# ===== PHASE 0: BUILD the EMD crate (release, python feature) =====
echo "PHASE=build START $(date -u +%FT%TZ)"
RUSTFLAGS="-C target-cpu=x86-64-v3" cargo build --release --features python --lib --manifest-path tools/battery_native/Cargo.toml
SO=$(ls tools/battery_native/target/release/libbattery_native.so 2>/dev/null || ls tools/battery_native/target/release/battery_native.so 2>/dev/null)
cp "$SO" "${CODE_DIR}/battery_native.so"
ls -la "${CODE_DIR}/battery_native.so"
python3 - <<'BUILD_SMOKE'
import battery_native as bn
required = ("NlheCFRTrainer", "EmdFrozenLookup", "emd_frozen_build")
missing = [name for name in required if not hasattr(bn, name)]
if missing:
    raise SystemExit(f"PHASE=build FATAL battery_native missing symbols: {missing}")
print("PHASE=build IMPORT_SMOKE_OK", getattr(bn, "__file__", "battery_native"),
      "rayon_threads", bn.rayon_num_threads(), flush=True)
BUILD_SMOKE
echo "PHASE=build DONE $(date -u +%FT%TZ)"

# ===== PHASE 1: ENSURE the frozen (hole,board)->bucket table (cache by TABLE_ID in S3) =====
FROZEN_LOCAL="${CODE_DIR}/frozen_${TABLE_ID}.emdfrzn"
FROZEN_S3="s3://${BUCKET}/${RUN_PREFIX}/frozen/${TABLE_ID}.emdfrzn"
FROZEN_SHA_LOCAL="${FROZEN_LOCAL}.sha256"
FROZEN_SHA_S3="${FROZEN_S3}.sha256"
write_frozen_sha_sidecar() {
  python3 - "$FROZEN_LOCAL" "$FROZEN_SHA_LOCAL" <<'SHA_EOF'
import hashlib
import sys
blob, out = sys.argv[1], sys.argv[2]
with open(blob, "rb") as f:
    sha = hashlib.sha256(f.read()).hexdigest()
with open(out, "w", encoding="ascii") as f:
    f.write(sha + "\n")
print(f"PHASE=precompute SHA256 sidecar -> {out} sha256={sha}", flush=True)
SHA_EOF
}
echo "PHASE=precompute TABLE_ID=${TABLE_ID}  s3=${FROZEN_S3}"
if s3_download "$FROZEN_S3" "$FROZEN_LOCAL"; then
  echo "PHASE=precompute CACHE_HIT downloaded cached frozen table ($(stat -c%s "$FROZEN_LOCAL") bytes) -> SKIP recompute"
  if s3_download "$FROZEN_SHA_S3" "$FROZEN_SHA_LOCAL"; then
    echo "PHASE=precompute CACHE_HIT downloaded frozen SHA256 sidecar -> $FROZEN_SHA_LOCAL"
  else
    echo "PHASE=precompute CACHE_HIT missing SHA256 sidecar in S3 -> writing migration sidecar from downloaded blob"
    write_frozen_sha_sidecar
    s3_upload "$FROZEN_SHA_LOCAL" "$FROZEN_SHA_S3" || { echo "PHASE=precompute FATAL frozen-table sidecar upload failed"; exit 13; }
  fi
else
  echo "PHASE=precompute CACHE_MISS no cached table -> running production precompute (n_runouts=${N_RUNOUTS})"
  # Heartbeat: emd_frozen_build is one long rayon call (no per-spot Python callback), so a background
  # ticker logs PRECOMPUTE_HEARTBEAT + the growing output-file size every 60s ("stuck vs slow" visible).
  ( while true; do
      SZ=$(stat -c%s "$FROZEN_LOCAL" 2>/dev/null || echo 0)
      echo "PRECOMPUTE_HEARTBEAT $(date -u +%FT%TZ) out_bytes=$SZ"
      sleep 60
    done ) &
  HB_PID=$!
  EMD_OUT="$FROZEN_LOCAL" EMD_CENT="${CODE_DIR}/${CENTROIDS_REL}" EMD_NBINS="${N_BINS}" EMD_NRUNOUTS="${N_RUNOUTS}" \
    python3 - <<'PRECOMP_EOF' 2>&1 | tee /var/log/emd-precompute.log
import os, time, json
import battery_native as bn
CENTROIDS = os.environ["EMD_CENT"]
OUT = os.environ["EMD_OUT"]
d = json.load(open(CENTROIDS))
cent = d["centroids"]
n_bins = int(os.environ["EMD_NBINS"]) or int(d.get("n_bins", 20))
n_runouts = int(os.environ["EMD_NRUNOUTS"]) or int(d.get("n_runouts", 200))
print(f"PRECOMPUTE start K={len(cent)} n_bins={n_bins} n_runouts={n_runouts}  "
      f"(~16.5M canonical flop+turn spots, rayon-parallel)", flush=True)
t0 = time.time()
count, stamp = bn.emd_frozen_build(cent, OUT, n_bins, n_runouts)
dt = time.time() - t0
rate = count / max(dt, 1.0)
print(f"PRECOMPUTE done spots={count} stamp={stamp} elapsed={dt:.0f}s rate={rate:.0f} spot/s", flush=True)
PRECOMP_EOF
  kill $HB_PID 2>/dev/null || true
  test -s "$FROZEN_LOCAL" || { echo "PRECOMPUTE FAILED (no table written)"; exit 11; }
  write_frozen_sha_sidecar
  s3_upload "$FROZEN_LOCAL" "$FROZEN_S3" || { echo "PHASE=precompute FATAL frozen-table upload failed"; exit 12; }
  s3_upload "$FROZEN_SHA_LOCAL" "$FROZEN_SHA_S3" || { echo "PHASE=precompute FATAL frozen-table sidecar upload failed"; exit 13; }
  echo "PHASE=precompute UPLOADED cached frozen table -> $FROZEN_S3"
fi
test -s "$FROZEN_LOCAL" || { echo "PHASE=precompute FATAL no frozen table after phase 1"; exit 11; }
test -s "$FROZEN_SHA_LOCAL" || { echo "PHASE=precompute FATAL no frozen SHA256 sidecar after phase 1"; exit 13; }
put_marker "$M_TABLE" "tid=${TABLE_ID}"   # idempotent: a resumed box CACHE_HITs this in seconds
echo "PHASE=precompute DONE $(date -u +%FT%TZ)"

# ===== PHASE 2: TRAIN to convergence (resume from S3 checkpoint if present) =====
cat > "${CODE_DIR}/drive_emd_train.py" <<PYEOF
import os, sys, time, traceback
import boto3
import battery_native as bn

BUCKET = "${BUCKET}"
RUN_PREFIX = "${RUN_PREFIX}"
REGION = "${REGION}"
ITERS = ${ITERS}
N_DEALS = ${N_DEALS}
NUM_BUCKETS = ${NUM_BUCKETS}
CKPT_EVERY = ${CKPT_EVERY}
TRAIN_LOG_EVERY = max(1, int(os.environ.get("TRAIN_LOG_EVERY", "50")))
TRAIN_PROFILE_EVERY = max(0, int(os.environ.get("TRAIN_PROFILE_EVERY", "0")))
TRAIN_FAST_MERGE = os.environ.get("TRAIN_FAST_MERGE", "0").strip().lower() in ("1", "true", "yes", "on")
DEPTH_EDGES = [float(x) for x in "${DEPTH_CSV}".split(",") if x]
EMD_FROZEN_PATH = os.environ["EMD_FROZEN_PATH"]  # exported below before invoking this driver

s3 = boto3.client("s3", region_name=REGION)
LATEST_KEY = RUN_PREFIX + "/checkpoints/latest.nlcfr"
TRAIN_STATUS_KEY = RUN_PREFIX + "/progress/train_status.txt"

def _download_latest():
    try:
        return s3.get_object(Bucket=BUCKET, Key=LATEST_KEY)["Body"].read()
    except Exception:
        return None

def _upload(blob, iters_done):
    key = RUN_PREFIX + f"/checkpoints/iter_{iters_done:06d}.nlcfr"
    s3.put_object(Bucket=BUCKET, Key=key, Body=blob)
    s3.put_object(Bucket=BUCKET, Key=LATEST_KEY, Body=blob)
    print(f"CKPT iters_done={iters_done} -> s3://{BUCKET}/{key} ({len(blob)} bytes)", flush=True)

def _emit_progress(kind, t, start, t0):
    el = time.time() - t0
    done = t.iters_done() - start
    rate = done / el if el > 0 else 0.0
    rem = max(ITERS - t.iters_done(), 0)
    eta = (rem / rate) if rate > 1e-9 else -1
    line = (f"{kind} iters_done={t.iters_done()}/{ITERS} infosets={t.n_infosets()} "
            f"elapsed={el:.0f}s rate={rate:.3f} it/s eta={eta:.0f}s")
    print(line, flush=True)
    try:
        s3.put_object(Bucket=BUCKET, Key=TRAIN_STATUS_KEY, Body=(line + "\n").encode("utf-8"))
    except Exception as e:
        print(f"TRAIN_STATUS_UPLOAD_FAIL {type(e).__name__}: {e}", flush=True)

def _train_step(t, iterations):
    if TRAIN_PROFILE_EVERY <= 0:
        t.train(iterations)
        return
    remaining = iterations
    while remaining > 0:
        chunk = min(TRAIN_PROFILE_EVERY, remaining)
        print("TRAIN_PROFILE_JSON " + t.train_profile_json(chunk), flush=True)
        remaining -= chunk

def main():
    # signature: (num_buckets, depth_edges, n_deals, n_seats=6, prune=-inf, fast_merge=False, emd_frozen_path)
    t = bn.NlheCFRTrainer(NUM_BUCKETS, DEPTH_EDGES, N_DEALS, 6, float("-inf"), TRAIN_FAST_MERGE, EMD_FROZEN_PATH)
    print(f"EMD frozen lookup ACTIVE: {EMD_FROZEN_PATH} stamp={t.emd_stamp()}", flush=True)
    blob = _download_latest()
    if blob is not None and t.load_bytes(blob):
        print(f"RESUMED from latest checkpoint at iters_done={t.iters_done()}", flush=True)
    else:
        print("FRESH start (no resumable checkpoint)", flush=True)
    start = t.iters_done()
    print(f"PHASE=train target={ITERS} from={start} n_deals={N_DEALS} num_buckets={NUM_BUCKETS} "
          f"depth_edges={DEPTH_EDGES} ckpt_every={CKPT_EVERY} log_every={TRAIN_LOG_EVERY} "
          f"profile_every={TRAIN_PROFILE_EVERY} fast_merge={TRAIN_FAST_MERGE}", flush=True)
    t0 = time.time()
    _emit_progress("TRAIN_HEARTBEAT", t, start, t0)
    while t.iters_done() < ITERS:
        ckpt_target = min(t.iters_done() + CKPT_EVERY, ITERS)
        while t.iters_done() < ckpt_target:
            step_target = min(t.iters_done() + TRAIN_LOG_EVERY, ckpt_target)
            _train_step(t, step_target - t.iters_done())
            _emit_progress("TRAIN_HEARTBEAT", t, start, t0)
        _upload(t.save_bytes(), t.iters_done())
        _emit_progress("PROGRESS", t, start, t0)
    print(f"DONE iters_done={t.iters_done()} infosets={t.n_infosets()}", flush=True)

if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc(); sys.exit(1)
PYEOF

# Background SPOT-INTERRUPT handler: on the reclaim notice SIGTERM the driver so its last S3 checkpoint
# (every CKPT_EVERY iters) is the resume point.
cat > /tmp/spot_watch.sh <<'SWEOF'
TOK=$(curl -s -X PUT "http://169.254.169.254/latest/api/token" -H "X-aws-ec2-metadata-token-ttl-seconds: 21600")
while true; do
  C=$(curl -s -o /dev/null -w "%{http_code}" -H "X-aws-ec2-metadata-token: $TOK" http://169.254.169.254/latest/meta-data/spot/instance-action)
  if [ "$C" = "200" ]; then echo "SPOT_RECLAIM_NOTICE"; pkill -TERM -f drive_emd_train.py; break; fi
  sleep 5
done
SWEOF
nohup bash /tmp/spot_watch.sh >/var/log/spot_watch.log 2>&1 &

# PHASE 2 TRAIN -- SKIPPED on resume if CONVERGED is already present (a kill during ship/eval must NOT
# re-train). The babysitter writes CONVERGED when its convergence stop fires; the driver writes it too when
# it reaches the iters target. Either way a relaunched box jumps to PHASE 4.
if has_marker "$M_CONVERGED"; then
  echo "PHASE=train SKIP (CONVERGED marker present -> resume at ship/eval; no re-train)"
else
  echo "PHASE=train START $(date -u +%FT%TZ)"
  EMD_FROZEN_PATH="$FROZEN_LOCAL" python3 "${CODE_DIR}/drive_emd_train.py" 2>&1 | tee /var/log/emd-train.log
  TRAIN_RC=${PIPESTATUS[0]}
  echo "train exit $TRAIN_RC"
  # Only mark CONVERGED when the driver exited 0 having reached the target (its last line is DONE ...).
  # A spot SIGTERM kill exits non-zero -> NO marker -> the babysitter relaunch RESUMES training from latest.
  if [ "$TRAIN_RC" = "0" ] && grep -q "^DONE iters_done=" /var/log/emd-train.log; then
    put_marker "$M_CONVERGED" "iters_target_reached"
    echo "PHASE=train DONE -> CONVERGED marker written"
  else
    echo "PHASE=train did NOT complete (rc=$TRAIN_RC, preempted or error) -> no CONVERGED marker; babysitter will resume"
  fi
fi

# ===== PHASE 4: SHIP -- convert the converged .nlcfr to a serve snapshot, run the serve-skew gate
# (ship_blueprint.py --emd-frozen) and STAGE it (versioned snapshot, NO --activate here). SKIPPED on resume
# if SHIPPED is present. Only runs once CONVERGED exists.
if has_marker "$M_CONVERGED" && ! has_marker "$M_SHIPPED"; then
  echo "PHASE=ship START $(date -u +%FT%TZ)"
  FINAL_CKPT="${CODE_DIR}/final_converged.nlcfr"
  echo "PHASE=ship DOWNLOAD_CKPT s3://${BUCKET}/${RUN_PREFIX}/checkpoints/latest.nlcfr -> $FINAL_CKPT"
  s3_download "s3://${BUCKET}/${RUN_PREFIX}/checkpoints/latest.nlcfr" "$FINAL_CKPT"
  test -s "$FINAL_CKPT" || { echo "PHASE=ship FATAL no converged checkpoint to ship"; exit 21; }
  # The ship gate (check_blueprint_serve_env): EMD flag + the frozen-lookup blake2b16 stamp on BOTH sides
  # MUST match. STAGE only (versioned snapshot -> checkpoints/serve_blueprints/${SHIP_NAME}.pkl); never flip
  # the live serve pointer here. Progress: ship_blueprint prints [1/3]CONVERT [2/3]VALIDATE [3/3]INSTALL.
  echo "PHASE=ship RUN_SHIP_BLUEPRINT name=${SHIP_NAME} num_buckets=${NUM_BUCKETS}"
  POKERBOT_POSTFLOP_NUM_BUCKETS=${NUM_BUCKETS} \
  POKERBOT_POSTFLOP_EMD_FROZEN_S3="${FROZEN_S3}" \
  POKERBOT_POSTFLOP_EMD_TABLE_ID="${TABLE_ID}" \
  POKERBOT_FINGERPRINT_NATIVE=1 \
    python3 scripts/ship_blueprint.py \
      --checkpoint "$FINAL_CKPT" --emd-frozen "$FROZEN_LOCAL" \
      --num-buckets ${NUM_BUCKETS} --name "${SHIP_NAME}" 2>&1 | tee /var/log/emd-ship.log
  SHIP_RC=${PIPESTATUS[0]}
  echo "ship exit $SHIP_RC"
  if [ "$SHIP_RC" != "0" ]; then echo "PHASE=ship FAILED (serve-skew gate) rc=$SHIP_RC"; exit 22; fi
  STAGED_LOCAL="${CODE_DIR}/checkpoints/serve_blueprints/${SHIP_NAME}.pkl"
  echo "PHASE=ship UPLOAD_STAGED $STAGED_LOCAL -> s3://${BUCKET}/${RUN_PREFIX}/serve/${SHIP_NAME}.pkl"
  s3_upload "$STAGED_LOCAL" "s3://${BUCKET}/${RUN_PREFIX}/serve/${SHIP_NAME}.pkl"
  put_marker "$M_SHIPPED" "staged=${SHIP_NAME}"
  echo "PHASE=ship DONE -> staged ${SHIP_NAME} (NOT activated) $(date -u +%FT%TZ)"
else
  if has_marker "$M_SHIPPED"; then echo "PHASE=ship SKIP (SHIPPED marker present)"; fi
fi

if [ "${STOP_AFTER_SHIP}" = "1" ] && has_marker "$M_SHIPPED"; then
  echo "PHASE=eval SKIP (STOP_AFTER_SHIP=1; Step 3 stops after staged serve artifact)"
  upload_logs
  echo "ONBOX_PIPELINE_DONE_STOP_AFTER_SHIP $(date -u +%FT%TZ)"
  exit 0
fi

# ===== PHASE 5: EVAL -- the verified-better gate. LBR-evaluate the NEW EMD blueprint AND iter62000 on the
# SAME seeds (CRN), with the always-fold anchor calibration (must reproduce ~+125 or abort), and emit the
# per-seat blind verdict JSON to S3. SKIPPED on resume if EVALED is present.
if has_marker "$M_SHIPPED" && ! has_marker "$M_EVALED"; then
  echo "PHASE=eval START $(date -u +%FT%TZ)"
  LOG_SYNC_PID=""
  start_log_sync
  EMD_PARTIAL_S3="s3://${BUCKET}/${RUN_PREFIX}/eval/emd_partial.json"
  BASE_PARTIAL_S3="s3://${BUCKET}/${RUN_PREFIX}/eval/iter62000_partial.json"
  partial_is_native() {
    local json_path="$1"
    python3 - "$json_path" <<'PY'
import json
import sys

path = sys.argv[1]
try:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
except Exception as e:
    print(f"partial_native_check failed to read {path}: {e}", flush=True)
    raise SystemExit(1)

result_method = str(payload.get("result", {}).get("method", ""))
anchor_method = str(payload.get("anchor", {}).get("method", ""))
ok = result_method.startswith("native_lbr6") and anchor_method.startswith("native_lbr6")
if not ok:
    print(
        f"partial_native_check reject {path}: result.method={result_method!r} "
        f"anchor.method={anchor_method!r}",
        flush=True,
    )
    raise SystemExit(1)
raise SystemExit(0)
PY
  }
  # The staged serve snapshot (SHIPPED marker guarantees it is in S3); fall back to the raw converged .nlcfr
  # only if the staged .pkl somehow vanished. EVAL_CKPT is whichever actually landed locally.
  STAGED_LOCAL="${CODE_DIR}/checkpoints/serve_blueprints/${SHIP_NAME}.pkl"
  EVAL_CKPT="$STAGED_LOCAL"
  s3_download "s3://${BUCKET}/${RUN_PREFIX}/serve/${SHIP_NAME}.pkl" "$STAGED_LOCAL" || true
  if [ ! -s "$STAGED_LOCAL" ]; then
    EVAL_CKPT="${CODE_DIR}/final_converged.nlcfr"
    s3_download "s3://${BUCKET}/${RUN_PREFIX}/checkpoints/latest.nlcfr" "$EVAL_CKPT"
  fi
  # EMD blueprint LBR (mode=emd env, EMD-correct via native frozen lookup when available). Anchor runs inside.
  EMD_PARTIAL_OK=0
  if has_marker "$EMD_PARTIAL_S3" && s3_download "$EMD_PARTIAL_S3" "${CODE_DIR}/eval_emd.json"; then
    if partial_is_native "${CODE_DIR}/eval_emd.json"; then
      echo "PHASE=eval EMD_SKIP cached native partial -> $EMD_PARTIAL_S3" | tee /var/log/emd-eval-emd.log
      EMD_PARTIAL_OK=1
    else
      echo "PHASE=eval EMD_PARTIAL_REJECT non-native/stale cached partial -> recompute" | tee /var/log/emd-eval-emd.log
      rm -f "${CODE_DIR}/eval_emd.json"
    fi
  fi
  if [ "$EMD_PARTIAL_OK" = "1" ]; then
    EVAL_EMD_RC=0
  else
    echo "PHASE=eval EMD_START seeds=${EVAL_SEEDS} hands=${EVAL_HANDS} checkpoint=$EVAL_CKPT"
    POKERBOT_FINGERPRINT_NATIVE=1 \
    POKERBOT_NATIVE_DEBUG=1 \
      python3 scripts/eval/eval_emd_vs_iter62000.py emd \
      --checkpoint "$EVAL_CKPT" --emd-frozen "$FROZEN_LOCAL" \
      --seeds ${EVAL_SEEDS} --hands ${EVAL_HANDS} --num-buckets ${NUM_BUCKETS} --value-only \
      --native --require-native \
      --out "${CODE_DIR}/eval_emd.json" 2>&1 | tee /var/log/emd-eval-emd.log
    EVAL_EMD_RC=${PIPESTATUS[0]}
    if [ "$EVAL_EMD_RC" = "0" ]; then
      s3_upload "${CODE_DIR}/eval_emd.json" "$EMD_PARTIAL_S3" || { echo "PHASE=eval FATAL EMD partial upload failed"; exit 31; }
      echo "PHASE=eval EMD_PARTIAL_UPLOADED -> $EMD_PARTIAL_S3"
    fi
  fi
  # iter62000 LBR (made-hand mode=fast env) on the SAME seeds -> CRN per-seat delta.
  BASE_PARTIAL_OK=0
  if has_marker "$BASE_PARTIAL_S3" && s3_download "$BASE_PARTIAL_S3" "${CODE_DIR}/eval_iter62000.json"; then
    if partial_is_native "${CODE_DIR}/eval_iter62000.json"; then
      echo "PHASE=eval ITER62000_SKIP cached native partial -> $BASE_PARTIAL_S3" | tee /var/log/emd-eval-base.log
      BASE_PARTIAL_OK=1
    else
      echo "PHASE=eval ITER62000_PARTIAL_REJECT non-native/stale cached partial -> recompute" | tee /var/log/emd-eval-base.log
      rm -f "${CODE_DIR}/eval_iter62000.json"
    fi
  fi
  if [ "$BASE_PARTIAL_OK" = "1" ]; then
    EVAL_BASE_RC=0
  else
    echo "PHASE=eval ITER62000_START seeds=${EVAL_SEEDS} hands=${EVAL_HANDS} checkpoint=${ITER62000_SNAPSHOT_S3}"
    POKERBOT_NATIVE_DEBUG=1 \
      python3 scripts/eval/eval_emd_vs_iter62000.py fast \
      --checkpoint "${ITER62000_SNAPSHOT_S3}" \
      --seeds ${EVAL_SEEDS} --hands ${EVAL_HANDS} --num-buckets 200 --value-only \
      --native --require-native \
      --out "${CODE_DIR}/eval_iter62000.json" 2>&1 | tee /var/log/emd-eval-base.log
    EVAL_BASE_RC=${PIPESTATUS[0]}
    if [ "$EVAL_BASE_RC" = "0" ]; then
      s3_upload "${CODE_DIR}/eval_iter62000.json" "$BASE_PARTIAL_S3" || { echo "PHASE=eval FATAL iter62000 partial upload failed"; exit 32; }
      echo "PHASE=eval ITER62000_PARTIAL_UPLOADED -> $BASE_PARTIAL_S3"
    fi
  fi
  if [ "$EVAL_EMD_RC" = "0" ] && [ "$EVAL_BASE_RC" = "0" ]; then
    echo "PHASE=eval COMBINE_START"
    python3 scripts/eval/eval_emd_vs_iter62000.py combine \
      --emd-json "${CODE_DIR}/eval_emd.json" --iter62000-json "${CODE_DIR}/eval_iter62000.json" \
      --out "${CODE_DIR}/eval_verdict.json" \
      --num-buckets "${NUM_BUCKETS}" --emd-frozen "$FROZEN_LOCAL" \
      --emd-frozen-s3 "$FROZEN_S3" --emd-table-id "$TABLE_ID" \
      --serve-blueprint "$STAGED_LOCAL" --serve-blueprint-s3 "s3://${BUCKET}/${RUN_PREFIX}/serve/${SHIP_NAME}.pkl" \
      --s3-out "s3://${BUCKET}/${RUN_PREFIX}/eval/verdict.json" 2>&1 | tee /var/log/emd-eval-combine.log
    put_marker "$M_EVALED" "verdict_written"
    echo "PHASE=eval DONE -> verdict at s3://${BUCKET}/${RUN_PREFIX}/eval/verdict.json $(date -u +%FT%TZ)"
  else
    echo "PHASE=eval FAILED (emd_rc=$EVAL_EMD_RC base_rc=$EVAL_BASE_RC) -> no EVALED marker; babysitter retries"
  fi
  stop_log_sync
  upload_logs
else
  if has_marker "$M_EVALED"; then echo "PHASE=eval SKIP (EVALED marker present)"; fi
fi

# Upload logs now; the EXIT trap also covers early failures.
trap - EXIT
upload_logs
echo "ONBOX_PIPELINE_DONE $(date -u +%FT%TZ)"
