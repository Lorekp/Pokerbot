#!/usr/bin/env bash
# Read-only dashboard for the overnight 7-MAX neural-CFR run + its babysitter. ASCII only.
#   wsl -d Debian -- bash -lc './scripts/neural_cfr/watch_7max_overnight.sh'
# Pass an interval (seconds) to loop: '... watch_7max_overnight.sh 30'
set -uo pipefail
REPO=.
RUNDIR="$REPO/runs/sevenmax_neural"
EVERY="${1:-0}"

show() {
  echo "=== 7-max neural-CFR overnight watch  [$(date +%H:%M:%S)] ==="

  # babysitter
  BPID=$(cat "$RUNDIR/babysit.pid" 2>/dev/null || echo "")
  BSTATE="not started"
  if [ -n "$BPID" ] && kill -0 "$BPID" 2>/dev/null; then BSTATE="ALIVE (PID $BPID)";
  elif [ -f "$RUNDIR/babysit.status" ]; then BSTATE="$(cat "$RUNDIR/babysit.status") (exited)"; fi
  RST=$(cat "$RUNDIR/babysit.restarts" 2>/dev/null || echo "0")
  STATUS=$(cat "$RUNDIR/babysit.status" 2>/dev/null || echo "?")
  echo "babysitter: $BSTATE   restarts: $RST   status: $STATUS"

  # training process
  TPID=$(cat "$RUNDIR/seven_train.pid" 2>/dev/null || echo "")
  if [ -n "$TPID" ] && kill -0 "$TPID" 2>/dev/null; then
    ET=$(ps -o etimes= -p "$TPID" 2>/dev/null | tr -d ' ')
    echo "training:   ALIVE (PID $TPID, up ${ET:-?}s)"
  else
    echo "training:   not running"
  fi

  # GPU
  echo "--- GPU ---"
  nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu \
    --format=csv,noheader 2>/dev/null || echo "(nvidia-smi unavailable)"

  # latest training progress (incl. iters/sec + ETA lines from the ProgressLogger)
  LOG=$(cat "$RUNDIR/seven_train.curlog" 2>/dev/null || ls -t "$RUNDIR"/seven_train_*.log 2>/dev/null | head -1)
  if [ -n "${LOG:-}" ] && [ -f "$LOG" ]; then
    echo "--- progress ($(basename "$LOG"), last 8 lines) ---"
    grep -E "iter [0-9]+/|Saved|Resuming|Training game|num_players|Traceback|Error|Killed|STALL" "$LOG" 2>/dev/null | tail -8 \
      || tail -8 "$LOG"
  else
    echo "--- no training log yet ---"
  fi

  # checkpoints
  echo "--- checkpoints ---"
  ls -t "$REPO"/checkpoints/sd_cfr_7max_neural/*_full.pkl 2>/dev/null | head -2 | while read -r f; do
    echo "  $(stat -c '%y' "$f" | cut -d. -f1)  $(basename "$f")"
  done || echo "  (none yet)"

  # gate poller (run-to-gate): convergence decision from the periodic net ICM-LBR eval
  GPID=$(cat "$RUNDIR/gate.pid" 2>/dev/null || echo "")
  GSTATE="not started"
  if [ -n "$GPID" ] && kill -0 "$GPID" 2>/dev/null; then GSTATE="ALIVE (PID $GPID)";
  elif [ -f "$RUNDIR/gate.status" ]; then GSTATE="$(cat "$RUNDIR/gate.status") (exited)"; fi
  echo "--- gate poller ---"
  echo "gate: $GSTATE"
  if [ -f "$RUNDIR/gate.log" ]; then
    # surface the last gate decision block (per-cell chip-LBR vs floor + ALL_PASS/NOT_YET) + heartbeat.
    grep -E "GATE: |below_anchor=|poll #|GATE MET|STALL\?" "$RUNDIR/gate.log" 2>/dev/null | tail -6 \
      || echo "  (no gate decision yet)"
  fi

  # recent babysitter events
  if [ -f "$RUNDIR/babysit.log" ]; then
    echo "--- babysitter log (last 4) ---"
    tail -4 "$RUNDIR/babysit.log"
  fi
}

if [ "$EVERY" -gt 0 ] 2>/dev/null; then
  echo "(refresh ${EVERY}s, Ctrl+C to stop)"
  while true; do show; echo; sleep "$EVERY"; done
else
  show
fi
