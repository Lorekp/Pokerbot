#!/usr/bin/env bash
set -euo pipefail
ROOT="."
cd "$ROOT"
source .venv/bin/activate
export PYTHONPATH="$ROOT"
export JAX_PLATFORMS=cpu

mkdir -p logs checkpoints/sd_cfr_leduc_short
rm -f checkpoints/sd_cfr_leduc_short/*

if [[ -f logs/sd_cfr_leduc_short.pid ]]; then
  old_pid="$(cat logs/sd_cfr_leduc_short.pid)"
  kill "$old_pid" 2>/dev/null || true
fi

nohup python3 scripts/train_sd_cfr_leduc.py \
  --iterations 40 \
  --traversals 32 \
  --seed 2 \
  --checkpoint-dir checkpoints/sd_cfr_leduc_short \
  --checkpoint-every 20 \
  --output checkpoints/sd_cfr_leduc_short.pkl \
  > logs/sd_cfr_leduc_short.log 2>&1 &

echo $! > logs/sd_cfr_leduc_short.pid
echo "Started short SD-CFR pid=$(cat logs/sd_cfr_leduc_short.pid)"
