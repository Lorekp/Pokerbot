#!/bin/bash
# Wait for the ladder driver to finish, then run improvement battery round 1.
cd "$(dirname "$0")/.."
LADDER_LOG="artifacts/task-output.txt"
for i in $(seq 1 240); do
  grep -q "LADDER COMPLETE" "$LADDER_LOG" 2>/dev/null && break
  sleep 60
done
echo "=== BATTERY R1 START $(date) ==="
export POKERBOT_BETTING_ABSTRACTION=1 POKERBOT_POSTFLOP_BUCKETS=1 POKERBOT_PREFLOP_LOSSLESS=1 POKERBOT_POSTFLOP_MODE=fast POKERBOT_POSTFLOP_NUM_BUCKETS=200
python -u scripts/engine_ab_battery.py \
  --baseline '{}' \
  --candidates artifacts/ab_candidates_r1.json \
  --decks 300 --seed 7331 \
  --outdir artifacts/battery_r1 2>&1 | grep -E "CANDIDATE|deck (50|100|150|200|250|300)/|VERDICT|FINAL|Error|Traceback"
echo "=== BATTERY R1 COMPLETE $(date) ==="
