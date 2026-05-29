#!/usr/bin/env bash
# One-time homeserver setup for solver label expansion, then launch chained run.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "=== setup $(date -Iseconds) ==="

if ! command -v cargo >/dev/null 2>&1; then
  echo "Installing Rust toolchain..."
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
  # shellcheck disable=SC1091
  source "$HOME/.cargo/env"
fi

export PATH="$HOME/.cargo/bin:$PATH"
cargo --version
python3 --version

VENV="$ROOT/.venv"
if [[ ! -d "$VENV" ]]; then
  echo "Creating venv at $VENV..."
  python3 -m venv "$VENV"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"

echo "Installing Python package..."
pip install -q -e .

echo "Building postflop_solver_cli (release)..."
cd external/postflop_solver_cli
cargo build --release
cd "$ROOT"

CLI="$ROOT/external/postflop_solver_cli/target/release/postflop_solver_cli"
if [[ ! -x "$CLI" ]]; then
  echo "ERROR: CLI build failed" >&2
  exit 1
fi
echo "CLI ready: $CLI"

mkdir -p logs data/solver

if [[ -f logs/solver_expansion_chain.pid ]] && kill -0 "$(cat logs/solver_expansion_chain.pid)" 2>/dev/null; then
  echo "Chain already running pid=$(cat logs/solver_expansion_chain.pid)" >&2
  exit 1
fi

nohup bash scripts/run_solver_expansion_chain.sh >> logs/solver_expansion_chain.nohup.log 2>&1 &
echo "Started solver expansion chain pid=$!"
