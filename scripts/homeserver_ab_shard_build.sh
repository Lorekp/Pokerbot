#!/usr/bin/env bash
set -euo pipefail
ROOT="."
cd "$ROOT"
source .venv/bin/activate
export PYTHONPATH="$ROOT"
mkdir -p logs data/solver/exploitability_ab
nohup python3 scripts/run_exploitability_kl_ab.py \
  --spot-keys-file logs/ab_spot_keys_20.txt \
  --skip-train \
  --output logs/exploitability_kl_ab_shards.json \
  > logs/exploitability_kl_ab_shards.log 2>&1 &
echo $! > logs/exploitability_kl_ab_shards.pid
echo "Started shard build pid=$(cat logs/exploitability_kl_ab_shards.pid)"
