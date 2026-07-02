#!/bin/bash
# Battery R1b: rerun of all 7 candidates on the RECONCILED engine (knobs verified live).
cd "$(dirname "$0")/.."
export POKERBOT_BETTING_ABSTRACTION=1 POKERBOT_POSTFLOP_BUCKETS=1 POKERBOT_PREFLOP_LOSSLESS=1 POKERBOT_POSTFLOP_MODE=fast POKERBOT_POSTFLOP_NUM_BUCKETS=200
export POKERBOT_NATIVE_SUBGAME=1
export POKERBOT_H2H_SHARDS=4
echo "=== BATTERY R1b START $(date) ==="
python -u scripts/engine_ab_battery.py \
  --baseline '{}' \
  --candidates artifacts/ab_candidates_r1.json \
  --decks 300 --seed 7331 \
  --outdir artifacts/battery_r1b 2>&1 | grep -E "RUNNING|progress|MERGED|VERDICT|ADOPT|REJECT|INCONCLUSIVE|WARNING|Error|Traceback"
echo "=== BATTERY R1b COMPLETE $(date) ==="
