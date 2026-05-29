#!/usr/bin/env python3
"""CLI wrapper for SFT input collision precondition."""

from __future__ import annotations

import argparse
import json

from pokerbot.data.sft_collision_check import check_sft_input_collisions


def main() -> None:
    p = argparse.ArgumentParser(description="Check SFT Parquet for policy input collisions")
    p.add_argument("--parquet", default="data/solver/train_prod60.parquet")
    p.add_argument("--target-kl-threshold", type=float, default=0.05)
    args = p.parse_args()

    summary = check_sft_input_collisions(
        args.parquet, target_kl_threshold=args.target_kl_threshold
    )
    print(json.dumps(summary, indent=2))
    if not summary["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
