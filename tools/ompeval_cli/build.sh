#!/usr/bin/env bash
# Build enumerate_hu_equity + hu_table_native + hu_table_build + bench_native (OMPEval required)
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$here"

if [[ ! -f vendor/OMPEval/omp/HandEvaluator.h ]]; then
  echo "OMPEval vendor missing. Run: git submodule update --init vendor/OMPEval" >&2
  exit 1
fi

cmake -B build -DUSE_OMPEVAL=ON -DBUILD_BENCH_NATIVE=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j"$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)"

echo "Built:"
echo "  build/hu_table_native*.so"
echo "  build/hu_table_build"
echo "  build/enumerate_hu_equity"
echo "  build/bench_native"
