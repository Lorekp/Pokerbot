#!/bin/bash
cd "$(dirname "$0")/.."
export POKERBOT_BETTING_ABSTRACTION=1 POKERBOT_POSTFLOP_BUCKETS=1 POKERBOT_PREFLOP_LOSSLESS=1 POKERBOT_POSTFLOP_MODE=fast POKERBOT_POSTFLOP_NUM_BUCKETS=200
export POKERBOT_NATIVE_SUBGAME=1
export POKERBOT_H2H_SHARDS=4
echo "=== BATTERY R3 START $(date) ==="
python -u scripts/engine_ab_battery.py \
  --baseline '{}' \
  --candidates artifacts/ab_candidates_r3.json \
  --decks 400 --seed 24601 \
  --outdir artifacts/battery_r3 2>&1 | grep -E "RUNNING|MERGED|VERDICT|WARNING|Error|Traceback"
echo "=== BATTERY R3 COMPLETE $(date) ==="
