#!/usr/bin/env bash
# Chained solver label expansion: prod600 then prod6000 (no gap between batches).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ -f "$ROOT/.venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "$ROOT/.venv/bin/activate"
fi

mkdir -p logs data/solver
CHAIN_LOG="logs/solver_expansion_chain.log"
PID_FILE="logs/solver_expansion_chain.pid"

echo $$ > "$PID_FILE"
exec > >(tee -a "$CHAIN_LOG") 2>&1

echo "=== solver expansion chain started $(date -Iseconds) pid=$$ ==="
echo "ROOT=$ROOT"

SOLVER_BACKEND="${SOLVER_BACKEND:-subprocess}"
NUM_WORKERS="${NUM_WORKERS:-1}"
# Omit SOLVER_THREADS_PER_WORKER on saturated hosts — let solver use all cores.

# Optional C++ TexasSolver adapter (after parity sign-off):
# export POSTFLOP_SOLVER_CLI="${ROOT}/tools/texas_solver_cli/build/texas_solver_cli"
# export TEXAS_SOLVER_ROOT=/path/to/TexasSolver/install

CLI="${POSTFLOP_SOLVER_CLI:-${ROOT}/external/postflop_solver_cli/target/release/postflop_solver_cli}"
if [[ ! -x "$CLI" ]]; then
  echo "ERROR: missing solver CLI at $CLI" >&2
  exit 1
fi

run_batch() {
  local name="$1"
  local config="$2"
  local output="$3"
  local manifest="$4"
  local report="$5"
  shift 5
  echo ""
  echo "=== ${name} starting $(date -Iseconds) ==="
  echo "config=$config output=$output backend=$SOLVER_BACKEND workers=$NUM_WORKERS extra_args=$*"
  python3 scripts/build_solver_labels.py \
    --config "$config" \
    --output "$output" \
    --manifest "$manifest" \
    --report "$report" \
    --weva-index data/weva \
    --cli-path "$CLI" \
    --solver-backend "$SOLVER_BACKEND" \
    --num-workers "$NUM_WORKERS" \
    "$@"
  echo "=== ${name} finished $(date -Iseconds) exit=$? ==="
}

run_batch "prod600" \
  config/solver_labels_prod600.yaml \
  data/solver/train_prod600.parquet \
  data/solver/manifest_prod600.json \
  logs/solver_label_run_prod600.md

run_batch "prod6000" \
  config/solver_labels_prod6000.yaml \
  data/solver/train_prod6000.parquet \
  data/solver/manifest_prod6000.json \
  logs/solver_label_run_prod6000.md \
  --resume \
  --bootstrap-log logs/solver_expansion_chain.log \
  --completion-ledger data/solver/train_prod6000_completed.jsonl \
  --shard-dir data/solver/train_prod6000_shards

echo "=== solver expansion chain complete $(date -Iseconds) ==="
rm -f "$PID_FILE"
