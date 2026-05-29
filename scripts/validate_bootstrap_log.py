#!/usr/bin/env python3
"""Validate chain-log bootstrap for prod6000 cutover (safe to run while process is live)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from pokerbot.solver.label_completion import (
    bootstrap_and_validate,
    write_completion_report,
)


def main() -> None:
    p = argparse.ArgumentParser(description="Validate solver chain log bootstrap")
    p.add_argument(
        "--log",
        type=Path,
        default=Path("logs/solver_expansion_chain.log"),
        help="Chain log path",
    )
    p.add_argument(
        "--expected-count",
        type=int,
        default=None,
        help="Require exactly this many unique valid paired completions (default: auto-detect)",
    )
    p.add_argument(
        "--section-marker",
        type=str,
        default="prod6000 starting",
        help="Only parse lines after this marker",
    )
    p.add_argument(
        "--shard-dir",
        type=Path,
        default=None,
        help="Optional shard dir for cross-check",
    )
    p.add_argument(
        "--parquet",
        type=Path,
        default=None,
        help="Optional partial Parquet for cross-check",
    )
    p.add_argument(
        "--ledger",
        type=Path,
        default=Path("data/solver/train_prod6000_completed.jsonl"),
        help="Ledger path (written only with --write-ledger)",
    )
    p.add_argument(
        "--report",
        type=Path,
        default=Path("logs/prod6000_bootstrap_validation.md"),
        help="Markdown report path",
    )
    p.add_argument(
        "--write-ledger",
        action="store_true",
        help="Write validated keys to completion ledger",
    )
    p.add_argument(
        "--allow-in-flight",
        action="store_true",
        default=True,
        help="Treat one trailing unpaired solving line as in-flight (default: true)",
    )
    p.add_argument(
        "--no-allow-in-flight",
        action="store_false",
        dest="allow_in_flight",
        help="Fail if any unpaired solving line remains",
    )
    args = p.parse_args()

    if not args.log.is_file():
        print(f"ERROR: log not found: {args.log}", file=sys.stderr)
        sys.exit(1)

    # Auto-detect count when not specified (live validation).
    probe = bootstrap_and_validate(
        args.log,
        ledger_path=args.ledger,
        shard_dir=args.shard_dir,
        parquet_path=args.parquet,
        expected_count=None,
        section_marker=args.section_marker,
        write_ledger=False,
        allow_in_flight=args.allow_in_flight,
    )
    expected = args.expected_count if args.expected_count is not None else probe.valid_count

    result = bootstrap_and_validate(
        args.log,
        ledger_path=args.ledger,
        shard_dir=args.shard_dir,
        parquet_path=args.parquet,
        expected_count=expected,
        section_marker=args.section_marker,
        write_ledger=args.write_ledger,
        allow_in_flight=args.allow_in_flight,
    )
    write_completion_report(result, args.report)

    summary = {
        "ok": result.ok,
        "expected_count": expected,
        "valid_count": result.valid_count,
        "invalid_keys": len(result.invalid_keys),
        "unpaired_solving": len(result.unpaired_solving),
        "duplicate_keys": len(result.duplicate_keys),
        "in_flight_solving": result.in_flight_solving,
        "cross_check_missing": len(result.cross_check_missing),
        "report": str(args.report),
    }
    print(json.dumps(summary, indent=2))

    if not result.ok:
        if result.invalid_keys:
            print("Invalid keys:", result.invalid_keys[:10], file=sys.stderr)
        if result.unpaired_solving:
            print("Unpaired solving:", result.unpaired_solving[:10], file=sys.stderr)
        if result.duplicate_keys:
            print("Duplicates:", result.duplicate_keys[:10], file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
