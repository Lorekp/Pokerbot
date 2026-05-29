#!/usr/bin/env python3
"""Snapshot: blind-aware call-with-behind vs legacy hardcoded-pot model."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from aof import call_ev_static_profile
from aof_equity import equity_vs_range, warmup
from aof_parse import parse_spot_line

SPOTS = [
    "99 SB vs BTN behind 1 8bb",
    "AKs SB vs BTN behind 1 8bb",
    "TT BTN vs UTG behind 1 8bb",
    "KQo SB vs BTN behind 1 8bb",
]
SAMPLES = 6000
SEED = 42


def _legacy_hardcoded_pot_ev(spot, samples: int, seed: int) -> float:
    """Old approximation: pot_3 = 1.5 + 3*stack for 3-way call branch."""
    from aof import _call_pot_terms, _lookup_call, _hero_key

    shove_pct, shove_rng = _lookup_call(
        spot.position, spot.shover, spot.stack_bb, table_size=spot.table_size
    )
    _ = shove_pct
    posted, _cost, pot_hu, _req = _call_pot_terms(spot.position, spot.stack_bb)
    eq_hu = equity_vs_range(
        _hero_key(spot.hero_cards), shove_rng, samples, seed, 64
    )
    ev_hu = eq_hu * pot_hu - spot.stack_bb

    behind = spot.behind_seats or ()
    if not behind:
        return ev_hu

    call_p = []
    for seat in behind:
        pct, _rng = _lookup_call(
            spot.position, seat, spot.stack_bb, table_size=spot.table_size
        )
        call_p.append(min(max(pct or 0.0, 0.0), 0.99))

    p_all_fold = 1.0
    for p in call_p:
        p_all_fold *= 1.0 - p

    ev_call = p_all_fold * ev_hu
    pot_3 = 1.5 + 3 * spot.stack_bb
    invest = spot.stack_bb
    for mask in range(1, 1 << len(behind)):
        w = 1.0
        for i in range(len(behind)):
            w *= call_p[i] if mask & (1 << i) else 1.0 - call_p[i]
        if w <= 0:
            continue
        eq = equity_vs_range(
            _hero_key(spot.hero_cards), shove_rng, samples // 4, seed + mask, 64
        )
        ev_call += w * (eq * pot_3 - invest)
    return ev_call


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("artifacts/aof_call_behind_snapshot.json"),
    )
    args = parser.parse_args()
    warmup()

    rows = []
    for line in SPOTS:
        spot = parse_spot_line(line)
        new_ev, *_ = call_ev_static_profile(spot, samples=SAMPLES, seed=SEED)
        legacy = _legacy_hardcoded_pot_ev(spot, SAMPLES, SEED)
        rows.append(
            {
                "spot": line,
                "new_ev_call": float(new_ev),
                "legacy_ev_call": float(legacy),
                "delta_bb": float(new_ev - legacy),
            }
        )

    payload = {"schema_version": 1, "spots": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
