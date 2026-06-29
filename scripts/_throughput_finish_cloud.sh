#!/usr/bin/env bash
# Complete the remaining cloud-shape runs: old r2, new r3, old r3.
# Same interleaving discipline (alternate variants).
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

echo "=========== CLOUD-SHAPE completion: old r2, new r3, old r3 ==========="
run_one "old" battery_native_old.pyd 200 1024 500 "old_d1024_r2"
run_one "new" battery_native_new.pyd 200 1024 500 "new_d1024_r3"
run_one "old" battery_native_old.pyd 200 1024 500 "old_d1024_r3"
echo "=========== CLOUD-SHAPE DONE ==========="
