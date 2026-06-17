#!/usr/bin/env python3
"""Audit prod6000 ledger vs shards; report sb gap for definitive SFT mix."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from pokerbot.solver.label_completion import CompletionLedger, spot_key_hash


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ledger", default="data/solver/train_prod6000_completed.jsonl")
    p.add_argument("--shard-dir", default="data/solver/train_prod6000_shards")
    p.add_argument("--batch-parquet", default="data/solver/train_prod6000.parquet")
    p.add_argument("--output", default="logs/prod6000_sb_shard_gap_audit.json")
    args = p.parse_args()

    ledger = CompletionLedger.load(Path(args.ledger))
    shard_dir = Path(args.shard_dir)
    batch = Path(args.batch_parquet)

    missing_by_type: Counter[str] = Counter()
    present_by_type: Counter[str] = Counter()
    missing_keys: list[str] = []

    for key in sorted(ledger.completed):
        t = key.split("|", 1)[0]
        if (shard_dir / f"{spot_key_hash(key)}.parquet").is_file():
            present_by_type[t] += 1
        else:
            missing_by_type[t] += 1
            missing_keys.append(key)

    batch_rows = 0
    batch_sb = 0
    if batch.is_file():
        import pandas as pd

        df = pd.read_parquet(batch)
        batch_rows = len(df)
        if "spot_class" in df.columns and batch_rows:
            # infer sb from raw if needed
            batch_sb = batch_rows  # refined below

    report = {
        "ledger_total": len(ledger.completed),
        "shards_present": sum(present_by_type.values()),
        "shards_missing": sum(missing_by_type.values()),
        "present_by_type": dict(present_by_type),
        "missing_by_type": dict(missing_by_type),
        "batch_parquet_exists": batch.is_file(),
        "batch_parquet_rows": batch_rows,
        "sb_missing_count": missing_by_type.get("sb_vs_bb_flop", 0),
        "sb_backfill_estimate_hours": round(
            missing_by_type.get("sb_vs_bb_flop", 0) * 13 / 3600, 1
        ),
        "missing_sb_keys_sample": [
            k for k in missing_keys if k.startswith("sb_vs_bb")][:5],
        "action": (
            "Import from batch parquet if rows exist; else re-solve missing sb "
            "spots with --resume (ledger skip for non-sb; force re-solve sb only) "
            "before definitive KL trend read."
        ),
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
