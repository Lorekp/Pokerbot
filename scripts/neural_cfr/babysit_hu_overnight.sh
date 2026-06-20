#!/usr/bin/env bash
# AWS-style SUPERVISOR for the overnight heads-up NLHE neural-CFR GPU run. ASCII only.
# Keeps the training alive across crashes / OOM / spot-style death: on any non-zero exit it
# RESUMES from the last atomic checkpoint and relaunches, with a restart-storm guard so a
# hard-broken config aborts instead of hammering the GPU forever.
#
# Start (one command; it self-detaches and survives the terminal closing):
#   wsl -d Debian -- bash -lc './scripts/neural_cfr/babysit_hu_overnight.sh'
# Watch:   scripts/neural_cfr/watch_hu_overnight.sh
# Stop:    kill $(cat runs/hu_neural/babysit.pid)   (then kill the train pid if still up)
set -uo pipefail

REPO=.
VENV="$HOME/jaxgpu-venv"
RUNDIR="$REPO/runs/hu_neural"
mkdir -p "$RUNDIR"

# --- self-detach (re-exec under setsid nohup so it outlives the launching shell) ------------------
if [ -z "${_HU_BABYSAT:-}" ]; then
  export _HU_BABYSAT=1
  setsid nohup "$0" "$@" >"$RUNDIR/babysit_boot.out" 2>&1 &
  echo "[babysitter] detached PID $!. watch: scripts/neural_cfr/watch_hu_overnight.sh   stop: kill $!"
  exit 0
fi

echo $$ > "$RUNDIR/babysit.pid"
BLOG="$RUNDIR/babysit.log"

# --- config (env-overridable; mirrors run_hu_overnight.sh) ----------------------------------------
GAME=${GAME:-nlhe-hu}
ITERS=${ITERS:-100000}            # effective "run until killed"; checkpoints preserve progress
TRAVERSALS=${TRAVERSALS:-128}
BATCH=${BATCH:-1024}
HIDDEN=${HIDDEN:-256}
TRAIN_STEPS=${TRAIN_STEPS:-16}
AVG_STEPS=${AVG_STEPS:-128}
HIST_STEPS=${HIST_STEPS:-32}
LR=${LR:-1e-4}
SEED=${SEED:-0}
CKPT_EVERY=${CKPT_EVERY:-25}
BUF_CKPT_EVERY=${BUF_CKPT_EVERY:-500}   # heavy reservoir snapshot cadence (cheap core ckpt stays at CKPT_EVERY)
LOG_EVERY=${LOG_EVERY:-25}              # progress-line cadence (default would be iters/50 = far too sparse)
CKPT=${CKPT:-checkpoints/sd_cfr_hu_neural/hu_vrpdcfr.pkl}
# restart-storm guard: if more than MAX_FAST restarts happen within FAST_WINDOW seconds, the failure
# is structural (not transient OOM) -> abort instead of looping.
MAX_FAST=${MAX_FAST:-5}
FAST_WINDOW=${FAST_WINDOW:-900}
BACKOFF=${BACKOFF:-20}

# --- GPU / memory safety (same as launcher) -------------------------------------------------------
export PYTHONPATH=.
export POKERBOT_INFO_SUITS=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.6
export JAX_PLATFORMS=cuda
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4

cd "$REPO"
# shellcheck disable=SC1090
source "$VENV/bin/activate"
mkdir -p "$REPO/$(dirname "$CKPT")"

log() { echo "[$(date +%Y-%m-%dT%H:%M:%S)] $*" | tee -a "$BLOG"; }

log "babysitter up (PID $$). game=$GAME iters=$ITERS trav=$TRAVERSALS batch=$BATCH hidden=$HIDDEN steps=$TRAIN_STEPS/$AVG_STEPS/$HIST_STEPS ckpt_every=$CKPT_EVERY -> $CKPT"
echo "0" > "$RUNDIR/babysit.restarts"

attempt=0
declare -a recent=()
while true; do
  attempt=$((attempt + 1))
  STAMP=$(date +%Y%m%d_%H%M%S)
  TLOG="$RUNDIR/hu_train_${STAMP}.log"
  echo "$((attempt - 1))" > "$RUNDIR/babysit.restarts"

  RESUME_ARG=()
  if [ -f "$REPO/${CKPT%.pkl}_full.pkl" ]; then
    RESUME_ARG=(--resume "$CKPT")
    log "attempt #$attempt: RESUMING from ${CKPT%.pkl}_full.pkl -> $TLOG"
  else
    log "attempt #$attempt: fresh start (no checkpoint) -> $TLOG"
  fi
  echo "$TLOG" > "$RUNDIR/hu_train.curlog"

  # foreground so we capture the real exit code; train writes its own progress to TLOG.
  python -u scripts/train_neural_cfr.py \
    --game "$GAME" --iterations "$ITERS" --traversals "$TRAVERSALS" --batch-size "$BATCH" \
    --hidden-dim "$HIDDEN" --train-steps "$TRAIN_STEPS" --avg-steps "$AVG_STEPS" \
    --hist-steps "$HIST_STEPS" --lr "$LR" --seed "$SEED" \
    --checkpoint "$CKPT" --checkpoint-every "$CKPT_EVERY" \
    --buffer-checkpoint-every "$BUF_CKPT_EVERY" --log-every "$LOG_EVERY" --traversal-workers 0 \
    "${RESUME_ARG[@]}" >>"$TLOG" 2>&1 &
  TPID=$!
  echo "$TPID" > "$RUNDIR/hu_train.pid"
  log "training PID $TPID launched"
  wait "$TPID"
  rc=$?

  if [ "$rc" -eq 0 ]; then
    log "training exited 0 (reached iteration cap $ITERS). babysitter done."
    break
  fi

  # crash / OOM / external kill. record + storm-guard.
  now=$(date +%s)
  recent+=("$now")
  recent=($(for t in "${recent[@]}"; do [ $((now - t)) -le "$FAST_WINDOW" ] && echo "$t"; done))
  log "training exited rc=$rc (crash/OOM/kill). fast-restarts in last ${FAST_WINDOW}s: ${#recent[@]}/$MAX_FAST. tail:"
  tail -n 3 "$TLOG" | sed 's/^/    /' | tee -a "$BLOG"

  if [ "${#recent[@]}" -ge "$MAX_FAST" ]; then
    log "ABORT: $MAX_FAST failures within ${FAST_WINDOW}s -> structural failure, not transient. Fix the cause; last log: $TLOG"
    echo "ABORTED" > "$RUNDIR/babysit.status"
    exit 1
  fi
  log "backing off ${BACKOFF}s then resuming..."
  sleep "$BACKOFF"
done
echo "DONE" > "$RUNDIR/babysit.status"
log "supervisor loop ended."
