#!/usr/bin/env bash
# One-shot status for homeserver exploitability AB shard build (--skip-train).
# Prints JSON on stdout; exit 0=running, 1=done, 2=failed/stale.
set -euo pipefail
ROOT="${ROOT:-.}"
LOG="$ROOT/logs/exploitability_kl_ab_shards.log"
PIDFILE="$ROOT/logs/exploitability_kl_ab_shards.pid"
TARGET_SOLVES=40
P05="$ROOT/data/solver/exploitability_ab/shards_0.5/labels.parquet"
P10="$ROOT/data/solver/exploitability_ab/shards_1.0/labels.parquet"

ok_converged=0
label_solve=0
if [[ -f "$LOG" ]]; then
  ok_converged=$(grep -c 'OK converged' "$LOG" 2>/dev/null || true)
  label_solve=$(grep -c 'Label solve' "$LOG" 2>/dev/null || true)
fi

pid=""
running=0
if [[ -f "$PIDFILE" ]]; then
  pid=$(cat "$PIDFILE" 2>/dev/null || true)
  if [[ -n "$pid" ]] && ps -p "$pid" >/dev/null 2>&1; then
    running=1
  fi
fi

p05=0
p10=0
[[ -f "$P05" ]] && p05=1
[[ -f "$P10" ]] && p10=1

last_line=""
if [[ -f "$LOG" ]]; then
  last_line=$(tail -1 "$LOG" 2>/dev/null | tr -d '\r' || true)
fi

status="running"
exit_code=0
if [[ "$p05" -eq 1 && "$p10" -eq 1 && "$running" -eq 0 ]]; then
  status="done"
  exit_code=1
elif [[ "$running" -eq 0 && "$ok_converged" -lt "$TARGET_SOLVES" ]]; then
  if echo "$last_line" | grep -qE 'Traceback|Error|SystemExit|NameError'; then
    status="failed"
    exit_code=2
  elif [[ "$ok_converged" -gt 0 ]]; then
    status="stale"
    exit_code=2
  else
    status="not_started"
    exit_code=2
  fi
fi

pct=0
if [[ "$TARGET_SOLVES" -gt 0 ]]; then
  pct=$(( ok_converged * 100 / TARGET_SOLVES ))
fi

printf '{"ts":"%s","status":"%s","ok_converged":%d,"label_solve":%d,"target":%d,"pct":%d,"running":%d,"pid":"%s","p05":%d,"p10":%d,"last":"%s"}\n' \
  "$(date -Iseconds)" "$status" "$ok_converged" "$label_solve" "$TARGET_SOLVES" "$pct" \
  "$running" "$pid" "$p05" "$p10" "${last_line//\"/\\\"}"

exit "$exit_code"
