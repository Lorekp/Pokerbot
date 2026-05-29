#!/usr/bin/env python3
"""Measure independent vs sequential open-shove EV on sample spots."""

from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from aof import shove_ev_open
from aof_equity import warmup
from aof_parse import parse_spot_line

SPOTS = [
    "AJo UTG 8bb",
    "77 BTN 3 8bb",
    "KQs BTN 8bb",
    "22 UTG 8bb",
]
SAMPLES = 12_000
SEED = 42
RHO = 0.35


def main() -> None:
    warmup()
    rows = []
    max_abs = 0.0
    for line in SPOTS:
        spot = parse_spot_line(line)
        assert not hasattr(spot, "shover")
        ev_i, _, _ = shove_ev_open(
            spot, samples=SAMPLES, seed=SEED, model="independent"
        )
        ev_s, _, _ = shove_ev_open(
            spot, samples=SAMPLES, seed=SEED + 1, model="sequential", rho=RHO
        )
        delta = ev_i - ev_s
        max_abs = max(max_abs, abs(delta))
        rows.append(
            {
                "spot": line,
                "ev_independent_bb": round(ev_i, 4),
                "ev_sequential_bb": round(ev_s, 4),
                "delta_bb": round(delta, 4),
            }
        )
    report = {
        "samples": SAMPLES,
        "seed": SEED,
        "rho": RHO,
        "max_abs_delta_bb": round(max_abs, 4),
        "spots": rows,
    }
    out = _ROOT / "data" / "aof" / "independence_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
