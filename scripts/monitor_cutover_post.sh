#!/usr/bin/env bash
set -euo pipefail

ROOT="."
LOG="$ROOT/logs/solver_expansion_chain.log"
SHARD_DIR="$ROOT/data/solver/train_prod6000_shards"
LEDGER="$ROOT/data/solver/train_prod6000_completed.jsonl"
MARK="prod6000 cutover restart interleaved 2026-05-29T13:14:28"

for _ in $(seq 1 60); do
  n=$(awk -v m="$MARK" '$0 ~ m {p=1} p && /^\[[0-9]+\] solving/{c++} END{print c+0}' "$LOG")
  echo "solves_started=$n"
  if [ "$n" -ge 20 ]; then
    break
  fi
  sleep 30
done

echo "=== FIRST 20 POST-CUTOVER SOLVES ==="
awk -v m="$MARK" '$0 ~ m {p=1} p && /^\[[0-9]+\] solving/{print; if(++c>=20) exit}' "$LOG"

echo "=== SB RE-SOLVES (expect 0) ==="
awk -v m="$MARK" '$0 ~ m {p=1} p && /^\[[0-9]+\] solving sb_vs_bb/{c++} END{print c+0}' "$LOG"

echo "=== LEDGER COUNT ==="
wc -l "$LEDGER"

echo "=== SHARD FILES (newest 20) ==="
ls -lt "$SHARD_DIR"/*.parquet 2>/dev/null | head -20

echo "=== SPOT TYPE COUNTS IN POST-CUTOVER SOLVES ==="
awk -v m="$MARK" '$0 ~ m {p=1} p && /^\[[0-9]+\] solving/{print}' "$LOG" | sed 's/.*solving //;s/|.*//' | sort | uniq -c
