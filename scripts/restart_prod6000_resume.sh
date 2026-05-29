#!/usr/bin/env bash
# Restart prod6000 only with interleaved resume (post-cutover).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ -f "$ROOT/.venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "$ROOT/.venv/bin/activate"
fi

export PYTHONPATH="$ROOT"

mkdir -p logs data/solver data/solver/train_prod6000_shards

PID_FILE="logs/solver_expansion_chain.pid"
CHAIN_LOG="logs/solver_expansion_chain.log"
CLI="${POSTFLOP_SOLVER_CLI:-${ROOT}/external/postflop_solver_cli/target/release/postflop_solver_cli}"

if [[ ! -x "$CLI" ]]; then
  echo "ERROR: missing solver CLI at $CLI" >&2
  exit 1
fi

if [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
  echo "ERROR: chain already running pid=$(cat "$PID_FILE")" >&2
  exit 1
fi

nohup bash "$ROOT/scripts/restart_prod6000_resume_inner.sh" >> logs/solver_expansion_chain.nohup.log 2>&1 &
echo "Started prod6000 resume pid=$!"
