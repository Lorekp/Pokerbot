#!/usr/bin/env bash
# Read-only dashboard for the overnight HU neural-CFR run + its babysitter. ASCII only.
#   wsl -d Debian -- bash -lc './scripts/neural_cfr/watch_hu_overnight.sh'
# Pass an interval (seconds) to loop: '... watch_hu_overnight.sh 30'
set -uo pipefail
REPO=.
RUNDIR="$REPO/runs/hu_neural"
EVERY="${1:-0}"

show() {
  echo "=== HU neural-CFR overnight watch  [$(date +%H:%M:%S)] ==="

  # babysitter
  BPID=$(cat "$RUNDIR/babysit.pid" 2>/dev/null || echo "")
  BSTATE="not started"
  if [ -n "$BPID" ] && kill -0 "$BPID" 2>/dev/null; then BSTATE="ALIVE (PID $BPID)";
  elif [ -f "$RUNDIR/babysit.status" ]; then BSTATE="$(cat "$RUNDIR/babysit.status") (exited)"; fi
  RST=$(cat "$RUNDIR/babysit.restarts" 2>/dev/null || echo "0")
  echo "babysitter: $BSTATE   restarts: $RST"

  # training process
  TPID=$(cat "$RUNDIR/hu_train.pid" 2>/dev/null || echo "")
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

  # latest training progress
  LOG=$(cat "$RUNDIR/hu_train.curlog" 2>/dev/null || ls -t "$RUNDIR"/hu_train_*.log 2>/dev/null | head -1)
  if [ -n "${LOG:-}" ] && [ -f "$LOG" ]; then
    echo "--- progress ($(basename "$LOG"), last 8 lines) ---"
    grep -E "iter [0-9]+/|Saved|Resuming|Training game|Traceback|Error|Killed" "$LOG" 2>/dev/null | tail -8 \
      || tail -8 "$LOG"
  else
    echo "--- no training log yet ---"
  fi

  # checkpoints
  echo "--- checkpoints ---"
  ls -t "$REPO"/checkpoints/sd_cfr_hu_neural/*_full.pkl 2>/dev/null | head -2 | while read -r f; do
    echo "  $(stat -c '%y' "$f" | cut -d. -f1)  $(basename "$f")"
  done || echo "  (none yet)"

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
