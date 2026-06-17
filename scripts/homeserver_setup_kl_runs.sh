#!/usr/bin/env bash
set -euo pipefail
ROOT="."
cd "$ROOT"
mkdir -p data/solver logs checkpoints config tmp_sync
for f in heldout_prod600_manifest.json heldout_prod600_vanilla.parquet heldout_prod600_bunching.parquet; do
  if [[ -f "tmp_sync/$f" ]]; then mv -f "tmp_sync/$f" data/solver/; fi
done
for pair in "prod600_pre_sft_diagnostic.json:logs" "ab_spot_keys_20.txt:logs" "solver_labels_exploitability_ab.yaml:config" "manifest_prod600.json:data/solver"; do
  f="${pair%%:*}"
  d="${pair##*:}"
  if [[ -f "tmp_sync/$f" ]]; then mv -f "tmp_sync/$f" "$d/"; fi
done
source .venv/bin/activate
export PYTHONPATH="$ROOT"
python3 -c "import torch; print('cuda_available', torch.cuda.is_available()); print('device', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"
echo "shards=$(ls data/solver/train_prod6000_shards/*.parquet 2>/dev/null | wc -l)"
ls -la data/solver/heldout_prod600_vanilla.parquet logs/ab_spot_keys_20.txt config/solver_labels_exploitability_ab.yaml 2>&1
