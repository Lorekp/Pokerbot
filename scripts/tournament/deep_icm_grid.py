"""Deep-ICM value, measured RIGHT — a targeted (constructed) grid, not random rollouts.

The random-spot P2 harness (deep_icm_grade.py) gave 0% flips and FAILED its positive control: random flop
spots almost never hit the marginal +chip-EV/-$EV gamble where ICM and chip-EV diverge. This measures the
deep-ICM overlay's decision-level value DIRECTLY by CONSTRUCTING the marginal spot across a grid:

  fixed 6-handed near-money field (top-3 of 6 paid, two short stacks on the bubble), hero = seat 0.
  for effective_stack S in {12,18,25,35,50} bb  x  hero showdown-equity e in {0.40..0.60}:
    SHOVE outcomes = {win e -> hero +S / villain -S ; lose 1-e -> hero -S / villain +S}  (HU all-in, covered)
    FOLD  outcomes = {1.0 -> 0}
    a_icm  = decide_deep(deal_struck=False)   # ICM valuation
    a_chip = decide_deep(deal_struck=True)    # chip-EV valuation
    FLIP   = a_icm != a_chip                   # ICM reordered the shove/fold decision

This IS its own positive control: flips MUST appear in the marginal band (e ~ 0.5) at SHORT stacks (busting
costs $-equity). The headline is the DEPTH PROFILE: does the flip band SHRINK as S grows (deep-ICM stops
mattering -> the value is short-stack-only, the engine's shallow band) or PERSIST deep (deep-ICM matters)?

Run: python scripts/tournament/deep_icm_grid.py
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from pokerbot.tournament.decide import decide_deep
from pokerbot.tournament.icm import bubble_factor

PAYOUTS = [50.0, 30.0, 20.0]                 # top-3 of 6 paid
# Fixed near-money field for seats 1..5 (a covering villain at seat 1 + two bubble shorts at seats 4,5).
FIELD_STACKS = [45.0, 27.0, 20.0, 12.0, 7.0]
STACKS_GRID = [12.0, 18.0, 25.0, 35.0, 50.0]
EQUITY_GRID = [0.40, 0.45, 0.48, 0.50, 0.52, 0.55, 0.60]


def _shove_outcomes(hero_stack: float, villain_seat: int, equity: float, n: int):
    risked = min(hero_stack, FIELD_STACKS[villain_seat - 1])      # HU all-in is for the smaller stack
    win = [0.0] * n;  win[0] = +risked;  win[villain_seat] = -risked
    lose = [0.0] * n; lose[0] = -risked; lose[villain_seat] = +risked
    return {"shove": [(equity, win), (1.0 - equity, lose)], "fold": [(1.0, [0.0] * n)]}


def main(argv) -> int:
    n = 6
    villain = 1                                                  # the covering stack
    rows = []
    flip_band_by_stack = {}
    for S in STACKS_GRID:
        stacks = [S] + FIELD_STACKS
        bf = max((bubble_factor(stacks, PAYOUTS, 0, v) for v in range(1, n)), default=1.0)
        flips = []
        for e in EQUITY_GRID:
            outs = _shove_outcomes(S, villain, e, n)
            a_icm = decide_deep(stacks, PAYOUTS, 0, outs, deal_struck=False).action
            a_chip = decide_deep(stacks, PAYOUTS, 0, outs, deal_struck=True).action
            flip = a_icm != a_chip
            rows.append({"stack_bb": S, "equity": e, "bubble_factor": round(bf, 3),
                         "a_icm": a_icm, "a_chip": a_chip, "flip": flip})
            if flip:
                flips.append(e)
        flip_band_by_stack[S] = flips
    out = {"payouts": PAYOUTS, "field_stacks": FIELD_STACKS, "rows": rows,
           "flip_band_by_stack": {str(k): v for k, v in flip_band_by_stack.items()}}
    art = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                       "artifacts", "deep_icm_grid.json")
    os.makedirs(os.path.dirname(art), exist_ok=True)
    with open(art, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)

    print(f"Deep-ICM SHOVE/FOLD flip grid (hero seat 0 vs covering villain; field {FIELD_STACKS}, pay {PAYOUTS})")
    print(f"  {'eff_stack':>9s} {'bubble_f':>8s}  flips? (X = ICM folds a +chip-EV shove)  [equity grid {EQUITY_GRID}]")
    for S in STACKS_GRID:
        bf = max((bubble_factor([S] + FIELD_STACKS, PAYOUTS, 0, v) for v in range(1, n)), default=1.0)
        marks = "".join("X" if e in flip_band_by_stack[S] else "." for e in EQUITY_GRID)
        band = flip_band_by_stack[S]
        bandstr = f"e in [{min(band):.2f},{max(band):.2f}]" if band else "(no flips)"
        print(f"  {S:>7.0f}bb {bf:>8.2f}  {marks}   {bandstr}")
    total_flips = sum(len(v) for v in flip_band_by_stack.values())
    deep_flips = sum(len(flip_band_by_stack[S]) for S in STACKS_GRID if S >= 25.0)
    short_flips = sum(len(flip_band_by_stack[S]) for S in STACKS_GRID if S < 25.0)
    print(f"\n  POSITIVE CONTROL: short-stack (<25bb) flips = {short_flips} (instrument detects ICM where it bites)")
    print(f"  DEPTH PROFILE   : deep (>=25bb) flips = {deep_flips} of {total_flips} total")
    print(f"  -> {'deep-ICM MATTERS (flips persist deep)' if deep_flips > 0 else 'deep-ICM value is SHORT-STACK-ONLY (flips vanish by 25bb -> the engine shallow band covers it)'}")
    print(f"wrote {art}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
