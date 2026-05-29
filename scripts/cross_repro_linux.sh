#!/usr/bin/env bash
# Linux cross-machine score2 repro (CI-faithful). Exit 1 on manifest mismatch.
set -euo pipefail

cd "$(dirname "$0")/.."
ROOT="$(pwd)"
sed -i 's/\r$//' scripts/cross_repro_linux.sh tools/ompeval_cli/build.sh 2>/dev/null || true

export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq cmake g++ libomp-dev python3 python3-pip python3-dev git

pip3 install --break-system-packages -q \
  numpy numba "phevaluator>=0.5" pybind11 "pokerkit>=0.5" pyyaml

git submodule update --init tools/ompeval_cli/vendor/OMPEval

cd tools/ompeval_cli
rm -rf build
sed -i 's/\r$//' ./build.sh
bash ./build.sh
cd "$ROOT"

export PYTHONPATH=.
python3 -c "from tools.ompeval_cli.native import native_backend_available; assert native_backend_available(), 'native self-test failed'"

OUT=/tmp/repro_out
rm -rf "$OUT"
python3 scripts/export_hu_showdown_score2.py --out-dir "$OUT" --threads 4

LINUX_SHA=$(python3 -c "import json; print(json.load(open('$OUT/score2.meta.json'))['score2_sha256'])")
MANIFEST_SHA=$(python3 -c "import json; print(json.load(open('data/aof/hu_showdown_equity.manifest.json'))['score2_sha256'])")

echo "linux score2_sha256=$LINUX_SHA"
echo "manifest score2_sha256=$MANIFEST_SHA"
echo "--- toolchain sidecar ---"
cat "$OUT/score2.meta.json"

if [[ "$LINUX_SHA" != "$MANIFEST_SHA" ]]; then
  echo "Cross-machine score2 MISMATCH" >&2
  exit 1
fi
echo "Cross-repro OK"
