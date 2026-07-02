#!/bin/bash
# Restarted pipeline: remaining ladder rungs + improvement battery R1,
# native subgame ON, 4-way CRN-exact deck sharding (rayon capped per shard).
cd "$(dirname "$0")/.."
export POKERBOT_BETTING_ABSTRACTION=1 POKERBOT_POSTFLOP_BUCKETS=1 POKERBOT_PREFLOP_LOSSLESS=1 POKERBOT_POSTFLOP_MODE=fast POKERBOT_POSTFLOP_NUM_BUCKETS=200
export POKERBOT_NATIVE_SUBGAME=1
export POKERBOT_H2H_SHARDS=4

for opp in honest deepbot; do
  echo "=== LADDER RUNG (native, sharded): $opp $(date) ==="
  python -u scripts/h2h_sharded.py --shards 4 --rayon-per-shard 3 --decks 300 \
    --out "artifacts/ladder_${opp}.json" -- \
    scripts/external_ladder_h2h.py --opponent "$opp" --iters 2000 --seed 4242 2>&1 | tail -30
done

echo "=== BATTERY R1 (native, sharded) START $(date) ==="
python -u scripts/engine_ab_battery.py \
  --baseline '{}' \
  --candidates artifacts/ab_candidates_r1.json \
  --decks 300 --seed 7331 \
  --outdir artifacts/battery_r1 2>&1 | grep -E "RUNNING|progress|MERGED|VERDICT|ADOPT|REJECT|INCONCLUSIVE|WARNING|Error|Traceback"
echo "=== ALL COMPLETE $(date) ==="
