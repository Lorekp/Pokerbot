#!/usr/bin/env python3
"""Report CALL_PCT vs combo-share of CALL_RANGE_STR (pokerkit)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from pokerkit import parse_range

from aof_ranges_chart import CALL_PCT, CALL_RANGE_STR, STACK_DEPTHS


def combo_fraction(range_str: str) -> float:
    combos = list(parse_range(range_str))
    if not combos:
        return 0.0
    # 1326 total preflop combos
    return len(combos) / 1326.0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "-o",
        "--out",
        type=Path,
        default=Path("artifacts/aof_call_pct_sync_report.json"),
    )
    p.add_argument("--depth", type=int, default=8)
    args = p.parse_args()

    depth = args.depth
    if depth not in CALL_RANGE_STR:
        depth = min(STACK_DEPTHS, key=lambda d: abs(d - args.depth))

    rng_map = CALL_RANGE_STR[depth]
    pct_map = CALL_PCT.get(depth, CALL_PCT[8])
    rows: list[dict] = []
    for key, rng in sorted(rng_map.items()):
        caller, opener = key
        computed = combo_fraction(rng)
        current = pct_map.get(key)
        delta = None if current is None else computed - current
        rows.append(
            {
                "caller": caller,
                "opener": opener,
                "range_str": rng,
                "combo_fraction": round(computed, 4),
                "call_pct_current": current,
                "delta": round(delta, 4) if delta is not None else None,
            }
        )

    payload = {
        "depth_bb": depth,
        "rows": rows,
        "note": "Manual CALL_PCT overrides are OK for independent-call tuning.",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {args.out} ({len(rows)} pairs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
