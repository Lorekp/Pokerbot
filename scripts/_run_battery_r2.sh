#!/bin/bash
cd "$(dirname "$0")/.."
export POKERBOT_BETTING_ABSTRACTION=1 POKERBOT_POSTFLOP_BUCKETS=1 POKERBOT_PREFLOP_LOSSLESS=1 POKERBOT_POSTFLOP_MODE=fast POKERBOT_POSTFLOP_NUM_BUCKETS=200
export POKERBOT_NATIVE_SUBGAME=1
export POKERBOT_H2H_SHARDS=4
echo "=== BATTERY R2 START $(date) ==="
python -u scripts/engine_ab_battery.py \
  --baseline '{}' \
  --candidates artifacts/ab_candidates_r2.json \
  --decks 300 --seed 9182 \
  --outdir artifacts/battery_r2 2>&1 | grep -E "RUNNING|MERGED|VERDICT|WARNING|Error|Traceback"
echo "=== BATTERY R2 COMPLETE $(date) ==="
