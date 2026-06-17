#!/usr/bin/env bash
# Launch long-running KL jobs on homeserver (CPU solves; training may run locally).
set -euo pipefail
ROOT="."
cd "$ROOT"
source .venv/bin/activate
export PYTHONPATH="$ROOT"

mkdir -p logs data/solver/exploitability_ab

# 1) Validate merge + 3000 coverage (no train)
python3 scripts/run_prod6000_definitive_kl_trend.py --merge-only 2>&1 | tee logs/prod6000_definitive_merge.log
python3 scripts/run_prod6000_definitive_kl_trend.py --skip-train 2>&1 | tee logs/prod6000_definitive_readiness.log

# 2) Exploitability A/B: solve 1.0% only (reuse manifest_prod600 for 0.5% timing)
nohup python3 scripts/run_exploitability_kl_ab.py \
  --spot-keys-file logs/ab_spot_keys_20.txt \
  --baseline-timing-manifest data/solver/manifest_prod600.json \
  --solve-only \
  --output logs/exploitability_kl_ab_solves.json \
  > logs/exploitability_kl_ab_solves.log 2>&1 &
echo $! > logs/exploitability_kl_ab_solves.pid
echo "Started AB 1.0% solves pid=$(cat logs/exploitability_kl_ab_solves.pid)"
