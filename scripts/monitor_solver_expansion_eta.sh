#!/usr/bin/env bash
# Report prod600 / prod6000 solver label expansion progress + ETA (run via ssh monitor).
set -euo pipefail

REMOTE="${REMOTE:-monitor}"
ROOT="${REMOTE_ROOT:-.}"
LOG="$ROOT/logs/solver_expansion_chain.log"

ssh "$REMOTE" "bash -s" <<EOF
set -euo pipefail
LOG="$LOG"
if [[ ! -f "\$LOG" ]]; then
  echo "STATUS=missing_log"
  exit 0
fi

started=\$(grep -m1 '^=== solver expansion chain started' "\$LOG" | sed 's/.*started //' | sed 's/ ===.*//' || true)
now_epoch=\$(date +%s)
if [[ -n "\$started" ]]; then
  start_epoch=\$(date -d "\$started" +%s 2>/dev/null || echo 0)
else
  start_epoch=0
fi

completed=\$(grep -c '^  OK converged=True' "\$LOG" || true)
started_spots=\$(grep -c '^\[[0-9]*\] solving' "\$LOG" || true)
last_line=\$(tail -1 "\$LOG")

phase="prod600"
target=600
if grep -q '^=== prod6000 starting' "\$LOG"; then
  phase="prod6000"
  target=6000
  # spots in prod6000 phase only
  completed=\$(awk '/^=== prod6000 starting/{p=1} p && /^  OK converged=True/{c++} END{print c+0}' "\$LOG")
  started_spots=\$(awk '/^=== prod6000 starting/{p=1} p && /^\[[0-9]*\] solving/{s++} END{print s+0}' "\$LOG")
fi

if grep -q '^=== solver expansion chain complete' "\$LOG"; then
  echo "STATUS=complete"
  echo "PHASE=done"
  echo "COMPLETED=\$completed"
  echo "LAST=\$last_line"
  exit 0
fi

running="no"
if pgrep -f 'build_solver_labels.py.*solver_labels_prod' >/dev/null 2>&1; then
  running="yes"
fi

elapsed=0
if [[ "\$start_epoch" -gt 0 ]]; then
  elapsed=\$((now_epoch - start_epoch))
fi

rate="0"
eta_sec=0
if [[ "\$completed" -gt 0 && "\$elapsed" -gt 0 ]]; then
  rate=\$(awk -v c="\$completed" -v e="\$elapsed" 'BEGIN{printf "%.4f", c/e}')
  remaining=\$((target - completed))
  if awk -v r="\$rate" 'BEGIN{exit (r>0)?0:1}'; then
    eta_sec=\$(awk -v r="\$rate" -v rem="\$remaining" 'BEGIN{printf "%.0f", rem/r}')
  fi
fi

# Combined ETA if still in prod600
eta6000_sec=0
if [[ "\$phase" == "prod600" && "\$rate" != "0" ]]; then
  rem600=\$((600 - completed))
  rem6000=6000
  eta6000_sec=\$(awk -v r="\$rate" -v a="\$rem600" -v b="\$rem6000" 'BEGIN{printf "%.0f", (a+b)/r}')
fi

fmt_eta() {
  local s=\$1
  if [[ "\$s" -le 0 ]]; then echo "unknown"; return; fi
  local d=\$((s/86400)) h=\$(((s%86400)/3600)) m=\$(((s%3600)/60))
  if [[ "\$d" -gt 0 ]]; then echo "\${d}d \${h}h \${m}m"; else echo "\${h}h \${m}m"; fi
}

echo "STATUS=running"
echo "RUNNING=\$running"
echo "PHASE=\$phase"
echo "TARGET=\$target"
echo "COMPLETED=\$completed"
echo "STARTED_SPOTS=\$started_spots"
echo "ELAPSED_SEC=\$elapsed"
echo "RATE_PER_SEC=\$rate"
echo "ETA_PHASE=\$(fmt_eta \$eta_sec)"
echo "ETA_COMBINED6600_SEC=\$eta6000_sec"
echo "ETA_COMBINED6600=\$(fmt_eta \$eta6000_sec)"
echo "CHAIN_STARTED=\$started"
echo "NOW=\$(date -Iseconds)"
echo "LAST=\$last_line"
EOF
