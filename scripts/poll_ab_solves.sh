#!/usr/bin/env bash
# Poll homeserver AB solve-only job until 20/20 complete.
set -euo pipefail
ROOT="."
LOG="$ROOT/logs/exploitability_kl_ab_solves.log"
TARGET=20

while true; do
  count=$(grep -c 'Solving' "$LOG" 2>/dev/null || echo 0)
  running=$(ps aux | grep -c '[r]un_exploitability_kl_ab.py' || true)
  echo "$(date -Iseconds) solves_started=$count running=$running"
  if [[ "$count" -ge "$TARGET" ]] && [[ "$running" -eq 0 ]]; then
    echo "DONE: $count solves, process exited"
    tail -3 "$LOG"
    exit 0
  fi
  sleep 120
done
