#!/usr/bin/env python3
"""Snapshot: exact vs MC multiway equity on small combo spaces."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from aof_equity import (
    equity_vs_three_ranges_adaptive,
    equity_vs_two_ranges,
    equity_vs_two_ranges_adaptive,
    warmup,
)

SPOTS: list[dict] = [
    {
        "name": "AA_vs_KK_QQ",
        "hero_cards": "AsAd",
        "villain_ranges": ["KK", "QQ"],
    },
    {
        "name": "AKs_vs_AA_KK",
        "hero_cards": "AsKs",
        "villain_ranges": ["AA", "KK"],
    },
    {
        "name": "99_vs_AA_KK_QQ",
        "hero_cards": "9h9d",
        "villain_ranges": ["AA", "KK", "QQ"],
    },
]

SAMPLES = 20_000
SEED = 42


def _row_two(hero: str, r0: str, r1: str) -> dict:
    exact_eq, meta = equity_vs_two_ranges_adaptive(
        hero, r0, r1, SAMPLES, SEED, return_meta=True
    )
    if meta.get("method") == "exact":
        mc_eq = equity_vs_two_ranges(hero, r0, r1, SAMPLES, SEED, max_combos=96)
    else:
        mc_eq = exact_eq
    return {
        "name": f"{hero}_vs_{r0}_{r1}",
        "hero_cards": hero,
        "villain_ranges": [r0, r1],
        "exact_equity": float(exact_eq),
        "mc_equity": float(mc_eq),
        "abs_delta": float(abs(exact_eq - mc_eq)),
        "method": "exact_vs_mc",
        "samples": SAMPLES,
        "seed": SEED,
        "equity_method": meta.get("method"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("artifacts/aof_multiway_equity_snapshot.json"),
    )
    args = parser.parse_args()
    warmup()

    rows: list[dict] = []
    for spec in SPOTS:
        hero = spec["hero_cards"]
        ranges = spec["villain_ranges"]
        if len(ranges) == 2:
            row = _row_two(hero, ranges[0], ranges[1])
            row["name"] = spec["name"]
            rows.append(row)
        elif len(ranges) == 3:
            exact_eq, meta = equity_vs_three_ranges_adaptive(
                hero,
                ranges,
                SAMPLES,
                SEED,
                return_meta=True,
            )
            mc_eq = exact_eq
            if meta.get("method") == "exact":
                mc_eq, _ = equity_vs_three_ranges_adaptive(
                    hero,
                    ranges,
                    SAMPLES,
                    SEED + 99,
                    max_combos=32,
                    return_meta=True,
                )
            rows.append(
                {
                    "name": spec["name"],
                    "hero_cards": hero,
                    "villain_ranges": ranges,
                    "exact_equity": float(exact_eq),
                    "mc_equity": float(mc_eq),
                    "abs_delta": float(abs(exact_eq - mc_eq)),
                    "method": "exact_vs_mc",
                    "samples": SAMPLES,
                    "seed": SEED,
                    "equity_method": meta.get("method"),
                }
            )

    payload = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config": {"samples": SAMPLES, "seed": SEED},
        "spots": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
