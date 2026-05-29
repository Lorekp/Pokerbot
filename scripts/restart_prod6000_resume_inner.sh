#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ -f "$ROOT/.venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "$ROOT/.venv/bin/activate"
fi

export PYTHONPATH="$ROOT"

PID_FILE="logs/solver_expansion_chain.pid"
CHAIN_LOG="logs/solver_expansion_chain.log"
CLI="${POSTFLOP_SOLVER_CLI:-${ROOT}/external/postflop_solver_cli/target/release/postflop_solver_cli}"

echo $$ > "$PID_FILE"
exec > >(tee -a "$CHAIN_LOG") 2>&1

echo "=== prod6000 cutover restart interleaved $(date -Iseconds) pid=$$ ==="

python3 scripts/build_solver_labels.py \
  --config config/solver_labels_prod6000.yaml \
  --output data/solver/train_prod6000.parquet \
  --manifest data/solver/manifest_prod6000.json \
  --report logs/solver_label_run_prod6000.md \
  --weva-index data/weva \
  --cli-path "$CLI" \
  --solver-backend subprocess \
  --num-workers 1 \
  --resume \
  --bootstrap-log logs/solver_expansion_chain.log \
  --completion-ledger data/solver/train_prod6000_completed.jsonl \
  --shard-dir data/solver/train_prod6000_shards

echo "=== prod6000 finished $(date -Iseconds) exit=$? ==="
rm -f "$PID_FILE"
