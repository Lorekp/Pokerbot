#!/usr/bin/env python3
"""Gate 0: NLHE WEVA encoding injectivity + Parquet replay parity report."""

from __future__ import annotations

import argparse

from pokerbot.training.nlhe_collision_check import run_gate0_report


def main() -> None:
    p = argparse.ArgumentParser(description="NLHE Gate 0 encoding report")
    p.add_argument("--heldout-vanilla", default="data/solver/heldout_prod600_vanilla.parquet")
    p.add_argument("--heldout-bunching", default="data/solver/heldout_prod600_bunching.parquet")
    p.add_argument("--output", default="logs/nlhe_gate0_encode_report.json")
    args = p.parse_args()
    report = run_gate0_report(
        heldout_vanilla=args.heldout_vanilla,
        heldout_bunching=args.heldout_bunching,
        output=args.output,
    )
    print(f"gate0_passed={report['gate0_passed']}")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
