#!/usr/bin/env python3
"""Audit homeserver prod6000 completion ledger vs shard directory."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from pokerbot.solver.label_completion import CompletionLedger, spot_key_hash
from pokerbot.solver.spot_enumerator import enumerate_spot_keys, load_solver_config


def main() -> None:
    p = argparse.ArgumentParser(description="Audit prod6000 ledger vs shards")
    p.add_argument(
        "--ledger",
        type=Path,
        default=Path("data/solver/train_prod6000_completed.jsonl"),
    )
    p.add_argument(
        "--shard-dir",
        type=Path,
        default=Path("data/solver/train_prod6000_shards"),
    )
    p.add_argument(
        "--config",
        type=Path,
        default=Path("config/solver_labels_prod6000.yaml"),
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path("logs/homeserver_ledger_shard_audit.json"),
    )
    args = p.parse_args()

    ledger = CompletionLedger.load(args.ledger)
    raw_lines_list = [
        ln.strip() for ln in args.ledger.read_text(encoding="utf-8").splitlines() if ln.strip()
    ]
    ledger_lines = len(raw_lines_list)

    line_key_counts: Counter[str] = Counter()
    for line in raw_lines_list:
        rec = json.loads(line)
        key = str(rec.get("spot_key", ""))
        if key:
            line_key_counts[key] += 1
    duplicate_line_keys = {k: v for k, v in line_key_counts.items() if v > 1}
    duplicate_ledger_lines = sum(v - 1 for v in line_key_counts.values() if v > 1)

    completed_set = set(ledger.completed.keys())

    shard_hashes = {
        f.stem
        for f in args.shard_dir.glob("*.parquet")
        if not f.name.endswith(".tmp")
    }
    manifest_keys = enumerate_spot_keys(load_solver_config(args.config))
    unique_keys = sorted(set(manifest_keys))
    stack_dupes = len(manifest_keys) - len(unique_keys)

    keys_without_shard = sorted(k for k in completed_set if spot_key_hash(k) not in shard_hashes)
    shards_without_ledger = sorted(
        h
        for h in shard_hashes
        if h not in {spot_key_hash(k) for k in completed_set}
    )
    ledger_no_shard_count = len(keys_without_shard)

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "ledger_path": str(args.ledger),
        "shard_dir": str(args.shard_dir),
        "config": str(args.config),
        "enumeration": {
            "total_manifest_keys": len(manifest_keys),
            "unique_spot_keys": len(unique_keys),
            "stack_depth_duplicates_in_enum": stack_dupes,
            "note": "spot_key omits stack_depth_bb; 40bb and 100bb share one key",
        },
        "ledger": {
            "raw_lines": ledger_lines,
            "unique_keys_in_ledger": len(completed_set),
            "duplicate_ledger_lines": duplicate_ledger_lines,
            "keys_with_multiple_lines": len(duplicate_line_keys),
            "top_duplicate_keys": sorted(
                [{"spot_key": k, "line_count": v} for k, v in duplicate_line_keys.items()],
                key=lambda x: -x["line_count"],
            )[:15],
        },
        "shards": {
            "parquet_count": len(shard_hashes),
        },
        "reconciliation": {
            "ledger_minus_shard_unique": len(completed_set) - len(shard_hashes),
            "raw_lines_minus_shards": ledger_lines - len(shard_hashes),
            "completed_without_shard": ledger_no_shard_count,
            "shards_without_ledger_entry": len(shards_without_ledger),
            "unique_keys_missing_shard": sorted(
                k for k in unique_keys if spot_key_hash(k) not in shard_hashes
            ),
            "unique_keys_missing_shard_count": sum(
                1 for k in unique_keys if spot_key_hash(k) not in shard_hashes
            ),
        },
        "by_spot_type": {
            "ledger": dict(Counter(k.split("|", 1)[0] for k in completed_set)),
            "unique_manifest": dict(Counter(k.split("|", 1)[0] for k in unique_keys)),
        },
        "sample_completed_without_shard": keys_without_shard[:10],
        "diagnosis": [],
    }

    if stack_dupes > 0:
        report["diagnosis"].append(
            f"Enumeration emits {stack_dupes} duplicate spot_keys (two stack depths); "
            "resume may log completion twice per unique key."
        )
    if duplicate_ledger_lines > 0:
        report["diagnosis"].append(
            f"{duplicate_ledger_lines} duplicate ledger lines across "
            f"{len(duplicate_line_keys)} spot_keys (stack-depth re-solves or resume re-append)."
        )
    if ledger_no_shard_count == 0 and len(shard_hashes) == len(unique_keys):
        report["diagnosis"].append(
            "All unique spot_keys have shards; ledger line excess is duplicate completions only."
        )
    if report["reconciliation"]["unique_keys_missing_shard_count"] == 0:
        report["diagnosis"].append("prod6000 unique spot coverage: COMPLETE (3000/3000).")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
