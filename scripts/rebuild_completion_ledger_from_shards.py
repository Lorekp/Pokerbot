#!/usr/bin/env python3
"""Rebuild completion ledger from shard files (ground truth)."""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from pokerbot.solver.label_completion import CompletionLedger, analyze_shard_completion, unique_manifest_keys
from pokerbot.solver.spot_enumerator import enumerate_spot_keys, load_solver_config


def main() -> None:
    p = argparse.ArgumentParser(description="Rebuild completion ledger from Parquet shards")
    p.add_argument("--config", type=Path, default=Path("config/solver_labels_prod6000.yaml"))
    p.add_argument("--work-manifest", type=Path, default=None)
    p.add_argument("--ledger", type=Path, default=Path("data/solver/train_prod6000_completed.jsonl"))
    p.add_argument("--shard-dir", type=Path, default=Path("data/solver/train_prod6000_shards"))
    p.add_argument("--output", type=Path, default=None, help="Ledger path to write (default: --ledger)")
    p.add_argument("--backup", action="store_true", help="Backup existing ledger before overwrite")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--report", type=Path, default=Path("logs/shard_completion_report.json"))
    args = p.parse_args()

    if args.work_manifest is not None:
        wm = json.loads(args.work_manifest.read_text(encoding="utf-8"))
        manifest_keys = list(wm.get("spot_keys", []))
    else:
        config = load_solver_config(args.config)
        manifest_keys = enumerate_spot_keys(config)

    unique_keys = unique_manifest_keys(manifest_keys)

    before = analyze_shard_completion(
        manifest_keys=unique_keys,
        shard_dir=args.shard_dir,
        ledger_path=args.ledger if args.ledger.is_file() else None,
    )

    out_path = args.output or args.ledger
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "before": before.to_dict(),
        "ledger_path": str(out_path),
        "shard_dir": str(args.shard_dir),
        "manifest_count": len(manifest_keys),
        "unique_manifest_count": len(unique_keys),
        "dry_run": args.dry_run,
    }

    if not args.dry_run:
        if args.backup and out_path.is_file():
            ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            backup_path = out_path.with_suffix(out_path.suffix + f".bak.{ts}")
            shutil.copy2(out_path, backup_path)
            report["backup_path"] = str(backup_path)

        ledger = CompletionLedger.rebuild_from_shards(
            out_path,
            args.shard_dir,
            unique_keys,
            source="shard_scan",
        )
        report["ledger_written"] = len(ledger.completed)

    after = analyze_shard_completion(
        manifest_keys=unique_keys,
        shard_dir=args.shard_dir,
        ledger_path=out_path if not args.dry_run else args.ledger,
    )
    report["after"] = after.to_dict()
    report["todo_count"] = after.todo_count

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
