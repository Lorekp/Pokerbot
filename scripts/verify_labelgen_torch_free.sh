#!/usr/bin/env bash
# Fast pre-check: labelgen runs with torch genuinely absent (not mocked).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
python -m pytest tests/test_labelgen_torch_free.py -m slow -q "$@"
