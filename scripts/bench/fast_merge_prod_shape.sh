#!/usr/bin/env bash
# Runs the 4 fast_merge configs 5x each, captures wall time from stderr
set -euo pipefail
OUT=./artifacts/fast_merge_prod_shape
BIN=./tools/battery_native/target/release/examples/bench_trainer.exe
cd ./tools/battery_native

# (label) iters n_deals fast_merge threads out_path
run_one() {
  local label=$1 iters=$2 n_deals=$3 fast=$4 threads=$5 idx=$6
  local out_path="$OUT/${label}_${idx}.bin"
  local log="$OUT/${label}_${idx}.log"
  echo "=== $label run $idx ==="
  RAYON_NUM_THREADS=$threads FAST_MERGE=$fast "$BIN" "$iters" "$n_deals" "$out_path" 2> "$log"
  cat "$log"
  echo
}

for i in 1 2 3 4 5; do
  # (a) serial, n_deals=200, iters=500, threads=12 (baseline reproduction)
  run_one serial_200 500 200 0 12 $i
  # (b) serial, n_deals=1024, iters=100, threads=12 (prod shape, serial)
  run_one serial_1024 100 1024 0 12 $i
  # (c) fast_merge, n_deals=1024, iters=100, threads=12 (prod shape, fast_merge)
  run_one fast_1024 100 1024 1 12 $i
  # (d) fast_merge, n_deals=1024, iters=100, threads=8 (subsaturated)
  run_one fast_1024_8t 100 1024 1 8 $i
done

echo "=== DONE ==="