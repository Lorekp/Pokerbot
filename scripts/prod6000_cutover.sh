#!/usr/bin/env bash
# prod6000 cutover: bootstrap-and-validate BEFORE pause (never kill before ledger is valid).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

EXPECTED="${BOOTSTRAP_EXPECTED_COUNT:-1331}"
LOG="${CHAIN_LOG:-logs/solver_expansion_chain.log}"
REPORT="${BOOTSTRAP_REPORT:-logs/prod6000_bootstrap_validation.md}"

echo "=== Step 1: validate bootstrap WHILE prod6000 may still be running ==="
python3 scripts/validate_bootstrap_log.py \
  --log "$LOG" \
  --expected-count "$EXPECTED" \
  --report "$REPORT"

echo ""
echo "=== Bootstrap validation PASSED ($EXPECTED keys) ==="
echo "Review: $REPORT"
echo ""
echo "=== Step 2: NOW safe to pause prod6000 ==="
echo "  kill \$(cat logs/solver_expansion_chain.pid)"
echo ""
echo "=== Step 3: deploy code, then restart interleaved resume ==="
echo "  bash scripts/run_solver_expansion_chain.sh   # prod6000 section only if prod600 done"
echo ""
echo "Optional: disable bunching mid-run via config/solver_labels_prod6000.yaml:"
echo "  disable_bunching: true"
echo ""
echo "=== Step 4: confirm btn+bunching accumulating ==="
echo "  python3 scripts/report_enumeration_order.py --ledger data/solver/train_prod6000_completed.jsonl"
