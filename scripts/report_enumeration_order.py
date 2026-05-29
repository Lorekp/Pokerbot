#!/usr/bin/env python3
"""Dry-run enumeration order report for interleaved label generation."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from pokerbot.solver.spot_enumerator import enumerate_spot_keys, load_solver_config
from pokerbot.solver.label_completion import CompletionLedger, validate_spot_key


def _spot_type(key: str) -> str:
    return key.split("|", 1)[0]


def main() -> None:
    p = argparse.ArgumentParser(description="Report interleaved enumeration order")
    p.add_argument("--config", type=Path, default=Path("config/solver_labels_prod6000.yaml"))
    p.add_argument("--max-keys", type=int, default=600, help="How many keys to report")
    p.add_argument("--ledger", type=Path, default=None, help="Skip keys in completion ledger")
    p.add_argument(
        "--output",
        type=Path,
        default=Path("logs/prod6000_reorder_report.md"),
    )
    args = p.parse_args()

    config = load_solver_config(args.config)
    all_keys = enumerate_spot_keys(config)
    ledger = CompletionLedger.load(args.ledger) if args.ledger else None

    remaining = [k for k in all_keys if ledger is None or not ledger.contains(k)]
    head = remaining[: args.max_keys]
    type_counts = Counter(_spot_type(k) for k in head)
    vanilla = type_counts.get("sb_vs_bb_flop", 0) + type_counts.get("btn_vs_bb_flop", 0)
    bunching = type_counts.get("bunching_postflop", 0)

    lines = [
        "# Enumeration order report",
        "",
        f"Config: `{args.config}`",
        f"Mode: `{config.enumeration_mode}` | disable_bunching: `{config.disable_bunching}`",
        f"Total keys (all): {len(all_keys)}",
        f"Remaining (after ledger skip): {len(remaining)}",
        "",
        f"## First {len(head)} remaining keys — spot-type counts",
        "",
        f"- sb_vs_bb_flop: {type_counts.get('sb_vs_bb_flop', 0)}",
        f"- btn_vs_bb_flop: {type_counts.get('btn_vs_bb_flop', 0)}",
        f"- bunching_postflop: {type_counts.get('bunching_postflop', 0)}",
        f"- vanilla/bunching ratio: {vanilla}:{bunching} ({vanilla / max(bunching, 1):.2f}:1)",
        "",
        "## Spot types in first 12 keys",
        "",
    ]
    first12_types = {_spot_type(k) for k in head[:12]}
    lines.append(f"Distinct spot types in first 12: {sorted(first12_types)}")
    lines.extend(["", "## First 50 keys", ""])
    for i, key in enumerate(head[:50], 1):
        valid = validate_spot_key(key)
        lines.append(f"{i}. `{key}` {'(invalid!)' if not valid else ''}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines), encoding="utf-8")

    print(
        json.dumps(
            {
                "total_keys": len(all_keys),
                "remaining": len(remaining),
                "first_n": len(head),
                "type_counts": dict(type_counts),
                "vanilla_bunching_ratio": round(vanilla / max(bunching, 1), 3),
                "spot_types_in_first_12": sorted(first12_types),
                "report": str(args.output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
