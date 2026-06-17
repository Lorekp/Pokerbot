#!/usr/bin/env bash
# Periodic sync of homeserver prod6000 completions to S3 (run alongside local solver).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ -f "$ROOT/.venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "$ROOT/.venv/bin/activate"
fi

export PYTHONPATH="$ROOT"
export LABELGEN_S3_BUCKET="${LABELGEN_S3_BUCKET:-video-processing-6789}"
export LABELGEN_S3_PREFIX="${LABELGEN_S3_PREFIX:-pokerbot/prod6000}"

python3 scripts/sync_prod6000_to_s3.py \
  --bucket "$LABELGEN_S3_BUCKET" \
  --prefix "$LABELGEN_S3_PREFIX" \
  --ledger data/solver/train_prod6000_completed.jsonl \
  --shard-dir data/solver/train_prod6000_shards
