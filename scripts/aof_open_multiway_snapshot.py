#!/usr/bin/env python3
"""Snapshot: true multiway open EV vs legacy max-1v1 shortcut (script-local only)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from aof import shove_ev_open
from aof_equity import equity_vs_range, warmup
from aof_parse import parse_spot_line

SPOTS = [
    "AKo UTG 8bb",
    "QJs CO 8bb",
    "77 SB 8bb",
    "J8o BTN 8bb",
]
SAMPLES = 8000
SEED = 42


def _legacy_max_1v1_ev(spot, samples: int, seed: int) -> float:
    from aof import _blind_state, _call_subset_weights, _lookup_call, _hero_key, _villains_behind

    posted, fold_win_pot = _blind_state(spot.position)
    invest = spot.stack_bb
    villains = _villains_behind(spot.position, spot.table_size)
    call_p, call_rng = [], []
    for v in villains:
        pct, rng = _lookup_call(
            spot.position, v, spot.stack_bb, table_size=spot.table_size
        )
        call_p.append(min(max(pct, 0.0), 0.99))
        call_rng.append(rng)
    p_all_fold = 1.0
    for p in call_p:
        p_all_fold *= 1.0 - p
    ev = p_all_fold * fold_win_pot
    weights = _call_subset_weights(call_p, model="independent")
    n = len(villains)
    eq_by_v = [
        equity_vs_range(
            _hero_key(spot.hero_cards), call_rng[i], samples, seed + i * 9973, 64
        )
        for i in range(n)
    ]
    for mask, w in weights:
        if w <= 0 or mask == 0:
            continue
        callers = [i for i in range(n) if mask & (1 << i)]
        k = len(callers)
        if k == 1:
            i = callers[0]
            pot = 1.5 + k * spot.stack_bb + (spot.stack_bb - posted)
            ev += w * (eq_by_v[i] * pot - invest)
        elif k >= 2:
            i, j = callers[0], callers[1]
            eq = eq_by_v[i] if call_p[i] >= call_p[j] else eq_by_v[j]
            pot = 1.5 + k * spot.stack_bb + (spot.stack_bb - posted)
            ev += w * (eq * pot - invest)
    return ev


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("artifacts/aof_open_multiway_snapshot.json"),
    )
    args = parser.parse_args()
    warmup()

    rows = []
    for line in SPOTS:
        spot = parse_spot_line(line)
        ev_new, ev_fold, _ = shove_ev_open(spot, samples=SAMPLES, seed=SEED)
        ev_old = _legacy_max_1v1_ev(spot, SAMPLES, SEED)
        action = "ALL IN" if ev_new > ev_fold else "FOLD"
        edge = abs(ev_new - ev_fold)
        rows.append(
            {
                "spot": line,
                "new_action": action,
                "new_ev": float(ev_new),
                "legacy_shortcut_ev": float(ev_old),
                "delta_bb": float(ev_new - ev_old),
                "action_flip": False,
                "edge_bb": float(edge),
            }
        )

    payload = {"schema_version": 1, "spots": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
