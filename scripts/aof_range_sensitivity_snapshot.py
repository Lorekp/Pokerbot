#!/usr/bin/env python3
"""Snapshot: chart-pool default vs tight archive on representative spots."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from aof import advise_static_profile
from aof_equity import warmup
from aof_parse import parse_spot_line

SPOTS = [
    "AKo UTG 8bb",
    "J8o BTN 8bb",
    "AA BB vs BTN 8bb",
    "99 SB vs BTN behind 1 8bb",
    "Ks4d BB vs UTG,BTN 8bb",
    "Ks4d BB vs UTG,BTN,SB 8bb",
]
SAMPLES = 8000
SEED = 42


def _apply_profile(name: str) -> None:
    import aof_ranges as base
    import aof_ranges_chart as chart
    import aof_ranges_tight as tight

    if name == "chart":
        base.CALL_PCT = copy.deepcopy(chart.CALL_PCT)
        base.CALL_RANGE_STR = copy.deepcopy(chart.CALL_RANGE_STR)
        base.SHOVE_RANGE_STR = copy.deepcopy(chart.SHOVE_RANGE_STR)
        base.SECOND_JAM_RANGE_STR = copy.deepcopy(chart.SECOND_JAM_RANGE_STR)
        base.THIRD_JAM_RANGE_STR = copy.deepcopy(chart.THIRD_JAM_RANGE_STR)
    elif name == "tight":
        base.CALL_PCT = copy.deepcopy(tight.CALL_PCT)
        base.CALL_RANGE_STR = copy.deepcopy(tight.CALL_RANGE_STR)
        base.SHOVE_RANGE_STR = copy.deepcopy(tight.SHOVE_RANGE_STR)
    else:
        raise ValueError(name)


def _ev_from_detail(action: str, detail: str) -> float:
    import re

    m = re.search(r"(?:callEV|shoveEV)=([+-]?\d+\.?\d*)", detail)
    if not m:
        return 0.0
    val = float(m.group(1))
    return val if "ALL IN" in action else 0.0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("artifacts/aof_range_sensitivity_snapshot.json"),
    )
    args = parser.parse_args()
    warmup()

    chart_rows: list[tuple[str, str, str, float]] = []
    _apply_profile("chart")
    for line in SPOTS:
        spot = parse_spot_line(line)
        act, det = advise_static_profile(spot, samples=SAMPLES, seed=SEED)
        chart_rows.append((line, act, det, _ev_from_detail(act, det)))

    rows = []
    _apply_profile("tight")
    for line, c_act, c_det, c_ev in chart_rows:
        spot = parse_spot_line(line)
        t_act, t_det = advise_static_profile(spot, samples=SAMPLES, seed=SEED)
        t_ev = _ev_from_detail(t_act, t_det)
        rows.append(
            {
                "spot": line,
                "chart_action": c_act,
                "tight_action": t_act,
                "chart_ev": float(c_ev),
                "tight_ev": float(t_ev),
                "delta_bb": float(t_ev - c_ev),
                "action_flip": c_act != t_act,
            }
        )

    payload = {
        "schema_version": 2,
        "profiles": ["chart_default", "tight_archive"],
        "spots": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
