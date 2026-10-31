#!/usr/bin/env bash
# Poll SD-CFR Leduc training status (run on homeserver; invoke via ssh monitor).
set -euo pipefail
ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PID_FILE="$ROOT/logs/sd_cfr_leduc_train.pid"
LOG="$ROOT/logs/sd_cfr_leduc_train.log"
CKPT_DIR="$ROOT/checkpoints/sd_cfr_leduc_run"

status="unknown"
pid=""
running="false"
iter=""
total=""
loss=""
latest_ckpt=""
log_tail=""

if [[ -f "$PID_FILE" ]]; then
  pid="$(tr -d '\r\n' < "$PID_FILE")"
  if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
    running="true"
    status="running"
  else
    status="stopped"
  fi
else
  status="no_pid_file"
fi

if [[ -f "$LOG" ]]; then
  last_line="$(grep 'SD-CFR iter' "$LOG" | tail -1 || true)"
  if [[ -n "$last_line" ]]; then
    iter="$(echo "$last_line" | sed -n 's/.*SD-CFR iter \([0-9]*\)\/\([0-9]*\).*/\1/p')"
    total="$(echo "$last_line" | sed -n 's/.*SD-CFR iter \([0-9]*\)\/\([0-9]*\).*/\2/p')"
    loss="$(echo "$last_line" | sed -n 's/.*loss=\([0-9.]*\).*/\1/p')"
  fi
  if grep -q "Infoset BR:" "$LOG" 2>/dev/null; then
    status="complete"
    running="false"
  fi
  log_tail="$(tail -3 "$LOG" | tr '\n' ' ' | sed 's/"/\\"/g')"
fi

if [[ -d "$CKPT_DIR" ]]; then
  latest_ckpt="$(ls -1t "$CKPT_DIR"/sd_cfr_leduc_iter_*.pkl 2>/dev/null | head -1 || true)"
  if [[ -z "$latest_ckpt" && -f "$CKPT_DIR/sd_cfr_leduc_latest.pkl" ]]; then
    latest_ckpt="$CKPT_DIR/sd_cfr_leduc_latest.pkl"
  fi
fi

printf '{"status":"%s","running":%s,"pid":"%s","iter":"%s","total":"%s","loss":"%s","latest_checkpoint":"%s","log_tail":"%s"}\n' \
  "$status" "$running" "$pid" "$iter" "$total" "$loss" "$latest_ckpt" "$log_tail"
