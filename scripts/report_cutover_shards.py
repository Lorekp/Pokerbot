#!/usr/bin/env python3
from pathlib import Path
from pokerbot.solver.label_completion import CompletionLedger, spot_key_hash

ledger = CompletionLedger.load(Path("data/solver/train_prod6000_completed.jsonl"))
shard_dir = Path("data/solver/train_prod6000_shards")
bootstrap = sum(1 for e in ledger.completed.values() if e.get("source") == "bootstrap_log")
new = [e for e in ledger.completed.values() if e.get("source") == "solve"]
print(f"bootstrap_entries={bootstrap}")
print(f"new_solve_entries={len(new)}")
print(f"total_ledger={len(ledger.completed)}")
print("=== NEW SOLVES WITH SHARDS ===")
for e in new[:20]:
    sk = e["spot_key"]
    h = spot_key_hash(sk)
    shard = shard_dir / f"{h}.parquet"
    print(f"{sk} -> {h}.parquet exists={shard.is_file()}")
