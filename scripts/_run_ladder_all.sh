#!/bin/bash
# Sequential ladder driver: our deployed engine vs each external rung.
cd "$(dirname "$0")/.."
for opp in fold_man random fish honest deepbot; do
  echo "=== LADDER RUNG: $opp $(date) ==="
  python -u scripts/external_ladder_h2h.py --opponent "$opp" --decks 300 --iters 2000 --seed 4242 \
    --out "artifacts/ladder_${opp}.json" 2>&1 | tee "artifacts/_ladder_${opp}.log" | grep -E "deck |FINAL|VERDICT|Error|Traceback" | tail -400
done
echo "=== LADDER COMPLETE $(date) ==="
