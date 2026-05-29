#!/usr/bin/env bash
set -euo pipefail
ROOT="."
mkdir -p "$ROOT/logs"
if pgrep -f 'monitor_solver_expansion_hourly_loop' >/dev/null 2>&1; then
  echo "hourly loop already running"
  exit 0
fi
nohup bash -c '
  while true; do
    sleep 3600
    echo "=== ETA $(date -Iseconds) ===" >> '"$ROOT"'/logs/eta_hourly.log
    bash '"$ROOT"'/scripts/monitor_solver_expansion_eta_remote.sh >> '"$ROOT"'/logs/eta_hourly.log
  done
' >> "$ROOT/logs/eta_hourly_loop.nohup" 2>&1 &
echo "started hourly ETA loop pid=$!"
