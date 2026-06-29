#!/usr/bin/env bash
# AWS-style SUPERVISOR for the overnight 7-MAX (tournament) NLHE neural-CFR GPU run. ASCII only.
# Mirrors babysit_hu_overnight.sh: keeps training alive across crashes / OOM / spot-style death by
# RESUMING from the last atomic checkpoint on any non-zero exit, with a restart-storm guard so a
# hard-broken config aborts instead of hammering the GPU forever. Adds a STALL WATCHDOG that surfaces
# a hung training loop (no new progress line within STALL_SECS) in the babysitter log + status file.
#
# Workstream C (7-player MTT blueprint). The VR persp_sign estimator is biased at >2 players (same as
# the deployed 6-max nlhe path); the 7-max net is a behavioral blueprint, calibrated against the
# always-fold anchor (pokerbot/tournament/anchor.py: +203.6 bb/100 at 7-max ante=0.125).
#
# Start (one command; it self-detaches and survives the terminal closing):
#   wsl -d Debian -- bash -lc './scripts/neural_cfr/babysit_7max_overnight.sh'
# Watch:   scripts/neural_cfr/watch_7max_overnight.sh
# Stop:    kill $(cat runs/sevenmax_neural/babysit.pid)   (then kill the train pid if still up)
set -uo pipefail

REPO=.
VENV="$HOME/jaxgpu-venv"
RUNDIR="$REPO/runs/sevenmax_neural"
mkdir -p "$RUNDIR"

# --- self-detach (re-exec under setsid nohup so it outlives the launching shell) ------------------
if [ -z "${_7MAX_BABYSAT:-}" ]; then
  export _7MAX_BABYSAT=1
  setsid nohup "$0" "$@" >"$RUNDIR/babysit_boot.out" 2>&1 &
  echo "[babysitter] detached PID $!. watch: scripts/neural_cfr/watch_7max_overnight.sh   stop: kill $!"
  exit 0
fi

echo $$ > "$RUNDIR/babysit.pid"
BLOG="$RUNDIR/babysit.log"

# --- config (env-overridable) ---------------------------------------------------------------------
GAME=${GAME:-nlhe-7max}
ITERS=${ITERS:-100000}            # effective "run until killed"; checkpoints preserve progress
TRAVERSALS=${TRAVERSALS:-128}     # keep a multiple of any device count for clean saturation
BATCH=${BATCH:-1024}
HIDDEN=${HIDDEN:-256}
TRAIN_STEPS=${TRAIN_STEPS:-16}
# LEVER 1 (free convergence win): exploitability is measured ONLY on the served average-strategy net,
# which trains solely on the strategy reservoir. Traversal/data-gen is the bottleneck (~711 ms/iter),
# so extra SGD on the tiny 256-wide served MLP is nearly free -- fit it HARDER to lower $-LBR with no
# change to the estimator/sampling/env/serve-skew. 128 -> 512 (toward the reference ~5000/750 served:adv
# ratio); HV baseline 32 -> 64 so the VR bootstrap the served net reads is also tighter. Override via env.
AVG_STEPS=${AVG_STEPS:-512}
HIST_STEPS=${HIST_STEPS:-64}
LR=${LR:-1e-4}
SEED=${SEED:-0}
CKPT_EVERY=${CKPT_EVERY:-25}
BUF_CKPT_EVERY=${BUF_CKPT_EVERY:-500}   # heavy reservoir snapshot cadence (cheap core ckpt stays at CKPT_EVERY)
LOG_EVERY=${LOG_EVERY:-25}              # progress-line cadence (default would be iters/50 = far too sparse)
CKPT=${CKPT:-checkpoints/sd_cfr_7max_neural/seven_v1.pkl}
# tournament structure: per-player ante (round(BB/8) ~= 0.125 bb) is ON by default in the 7-max env
# spec; the depth band trains the stack-agnostic shallow/deep range. Override via env if desired.
ANTE_BB=${POKERBOT_7MAX_ANTE_BB:-0.125}
# LEVER 3 (ICM-leak targeting, opt-in): the grade is +chip-EV but -$-EV at the bubble. Chip-EV CFR is
# ICM-blind, so MORE chip-EV iters alone will not close the bubble leak -- but skewing the trained DEPTH
# band toward SHALLOW makes the net spend capacity where $-EV diverges most from chip-EV (short stacks /
# near-money push-fold). To run a bubble-weighted annealing leg, launch with e.g.
#   POKERBOT_7MAX_DEPTH_BB=8,40 AVG_STEPS=512 ...
# (the serve-time ICM overlay in tournament.icm_overlay remains the PRIMARY fix; this only re-weights
# data, no estimator change). Default keeps the full 8,120 band so a warm resume stays distribution-
# compatible with the existing iter-100000 checkpoint.
DEPTH_BB=${POKERBOT_7MAX_DEPTH_BB:-8,120}
# restart-storm guard: if more than MAX_FAST restarts happen within FAST_WINDOW seconds, the failure
# is structural (not transient OOM) -> abort instead of looping.
MAX_FAST=${MAX_FAST:-5}
FAST_WINDOW=${FAST_WINDOW:-900}
BACKOFF=${BACKOFF:-20}
# stall watchdog: if the current training log gets no new "iter N/" progress line for this many
# seconds, flag a stall (the loop is hung, not crashed -> no non-zero exit to trigger resume).
STALL_SECS=${STALL_SECS:-1800}

# --- GPU / memory safety (same as the HU launcher) ------------------------------------------------
export PYTHONPATH=.
export POKERBOT_INFO_SUITS=1
export POKERBOT_7MAX_ANTE_BB="$ANTE_BB"
export POKERBOT_7MAX_DEPTH_BB="$DEPTH_BB"
# In-loop convergence proxy: append the served avg-strategy reservoir loss to each progress line so the
# log shows whether the served policy (the one LBR scores) is converging, not just throughput. Cheap
# (one forward pass per LOG_EVERY tick). Override POKERBOT_CFR_LOSS_PROBE=0 to silence it.
export POKERBOT_CFR_LOSS_PROBE=${POKERBOT_CFR_LOSS_PROBE:-1}
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

log "babysitter up (PID $$). game=$GAME iters=$ITERS trav=$TRAVERSALS batch=$BATCH hidden=$HIDDEN steps=$TRAIN_STEPS/$AVG_STEPS/$HIST_STEPS ante=$ANTE_BB depth=$DEPTH_BB ckpt_every=$CKPT_EVERY -> $CKPT"
echo "0" > "$RUNDIR/babysit.restarts"
echo "RUNNING" > "$RUNDIR/babysit.status"

# --- stall watchdog (background): watches the current train log's last progress timestamp ----------
stall_watch() {
  while true; do
    sleep 120
    CUR=$(cat "$RUNDIR/seven_train.curlog" 2>/dev/null || echo "")
    [ -n "$CUR" ] && [ -f "$CUR" ] || continue
    LASTPROG=$(grep -nE "iter [0-9]+/|Saved|Resuming|Training game" "$CUR" 2>/dev/null | tail -1)
    # mtime of the log file is a cheap proxy for "did anything happen recently".
    MOD=$(stat -c %Y "$CUR" 2>/dev/null || echo 0)
    NOW=$(date +%s)
    if [ "$((NOW - MOD))" -ge "$STALL_SECS" ]; then
      log "STALL WARNING: no log activity on $(basename "$CUR") for $((NOW - MOD))s (>= ${STALL_SECS}s). last: ${LASTPROG:-<none>}"
      echo "STALLED" > "$RUNDIR/babysit.status"
    fi
  done
}
stall_watch &
WATCH_PID=$!
# ensure the watchdog dies with the babysitter
trap 'kill "$WATCH_PID" 2>/dev/null || true' EXIT

attempt=0
declare -a recent=()
while true; do
  attempt=$((attempt + 1))
  STAMP=$(date +%Y%m%d_%H%M%S)
  TLOG="$RUNDIR/seven_train_${STAMP}.log"
  echo "$((attempt - 1))" > "$RUNDIR/babysit.restarts"

  RESUME_ARG=()
  if [ -f "$REPO/${CKPT%.pkl}_full.pkl" ]; then
    RESUME_ARG=(--resume "$CKPT")
    log "attempt #$attempt: RESUMING from ${CKPT%.pkl}_full.pkl -> $TLOG"
  else
    log "attempt #$attempt: fresh start (no checkpoint) -> $TLOG"
  fi
  echo "$TLOG" > "$RUNDIR/seven_train.curlog"

  # foreground so we capture the real exit code; train writes its own progress to TLOG.
  # --traversal-workers 0: the mp pool is 6-max-only, so 7-max uses the in-process GPU path (the
  # single-GPU saturation lever is irrelevant anyway -- the train script enforces this).
  python -u scripts/train_neural_cfr.py \
    --game "$GAME" --iterations "$ITERS" --traversals "$TRAVERSALS" --batch-size "$BATCH" \
    --hidden-dim "$HIDDEN" --train-steps "$TRAIN_STEPS" --avg-steps "$AVG_STEPS" \
    --hist-steps "$HIST_STEPS" --lr "$LR" --seed "$SEED" \
    --checkpoint "$CKPT" --checkpoint-every "$CKPT_EVERY" \
    --buffer-checkpoint-every "$BUF_CKPT_EVERY" --log-every "$LOG_EVERY" --traversal-workers 0 \
    "${RESUME_ARG[@]}" >>"$TLOG" 2>&1 &
  TPID=$!
  echo "$TPID" > "$RUNDIR/seven_train.pid"
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
