#!/usr/bin/env bash
# Interleaved head-to-head throughput orchestration.
# Runs: for each round (1..3), for each config, run NEW then OLD as separate
# fresh python processes, swapping the repo-root pyd before each.
# Interleaving (new,old,new,old,...) cancels thermal drift.
set -euo pipefail
cd .

export RAYON_NUM_THREADS=12
THRU=artifacts/_thru
HARNESS=scripts/_throughput_test.py

run_one () {
  local variant="$1" pyd="$2" nb="$3" nd="$4" iters="$5" tag="$6"
  cp "$pyd" battery_native.pyd
  echo ">>> [$(date +%H:%M:%S)] variant=$variant config=nb${nb}_nd${nd}_it${iters} tag=$tag"
  RAYON_NUM_THREADS=12 python "$HARNESS" "$variant" "$nb" "$nd" "$iters" \
    "$THRU/${tag}.bin" "$THRU/${tag}.json" \
    | grep -E "wall_sec|iters_per_sec|deals_per_sec|n_infosets|blob_sha256"
}

echo "=========== PRIMARY config: nb=200 nd=200 iters=5000 ==========="
for r in 1 2 3; do
  run_one "new" battery_native_new.pyd 200 200 5000 "new_d200_r${r}"
  run_one "old" battery_native_old.pyd 200 200 5000 "old_d200_r${r}"
done

echo "=========== CLOUD-SHAPE config: nb=200 nd=1024 iters=500 ==========="
for r in 1 2 3; do
  run_one "new" battery_native_new.pyd 200 1024 500 "new_d1024_r${r}"
  run_one "old" battery_native_old.pyd 200 1024 500 "old_d1024_r${r}"
done

echo "=========== DONE ==========="
