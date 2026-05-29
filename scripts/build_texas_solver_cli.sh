#!/usr/bin/env bash
# Build tools/texas_solver_cli (requires cmake + C++17 compiler).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD="${ROOT}/tools/texas_solver_cli/build"
cmake -S "${ROOT}/tools/texas_solver_cli" -B "$BUILD"
cmake --build "$BUILD" --config Release
echo "Built: ${BUILD}/texas_solver_cli"
