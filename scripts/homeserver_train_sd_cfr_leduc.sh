#!/usr/bin/env bash
# Launch SD-CFR Leduc training on homeserver (CPU; tree traversal bound).
set -euo pipefail
ROOT="."
cd "$ROOT"
source .venv/bin/activate
export PYTHONPATH="$ROOT"
export JAX_PLATFORMS=cpu

mkdir -p logs checkpoints/sd_cfr_leduc_run

if [[ -f logs/sd_cfr_leduc_train.pid ]]; then
  old_pid="$(cat logs/sd_cfr_leduc_train.pid)"
  if kill -0 "$old_pid" 2>/dev/null; then
    echo "SD-CFR Leduc already running pid=$old_pid"
    exit 1
  fi
fi

nohup python3 scripts/train_sd_cfr_leduc.py \
  --iterations 200 \
  --traversals 32 \
  --seed 2 \
  --checkpoint-dir checkpoints/sd_cfr_leduc_run \
  --checkpoint-every 20 \
  --output checkpoints/sd_cfr_leduc.pkl \
  > logs/sd_cfr_leduc_train.log 2>&1 &

echo $! > logs/sd_cfr_leduc_train.pid
echo "Started SD-CFR Leduc pid=$(cat logs/sd_cfr_leduc_train.pid)"
