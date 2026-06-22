"""Chip-conservation invariant at SCALE for the tournament SNG simulator (Workstream C, item 2).

Plays many full push/fold SNGs to completion and asserts the chip-conservation invariant on EVERY
hand (per-hand sum of stacks == constant table total), independently of the assert already inside
``simulate_sng``. Records ``artifacts/tourney_conservation.json`` with the realized counts.

The at-scale default uses the FIELD strategies (fixed range tables -> no per-decision push/fold
solve), so it reaches >=100k hands in seconds. The ICM engine path is identical mechanics (same
``play_hand`` / ``_settle_allin`` side-pot resolution + conservation assert); to run the full
ICM-engine conservation at scale (slower, per-decision solves are cached):

    python scripts/tournament/conservation_at_scale.py --hands 100000 --kinds icm field field

Run (default at-scale field-vs-field, >=100k hands):
    python scripts/tournament/conservation_at_scale.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np

import pokerbot.tournament.sim as sim
from pokerbot.tournament.pushfold_icm import _hu_grid

EQ, _LAB, W = _hu_grid()
STRENGTH = EQ.mean(axis=1)
PAY = [65.0, 35.0, 0.0]


def _top_frac(frac: float) -> np.ndarray:
    order = np.argsort(-STRENGTH)
    keep = np.cumsum(W[order]) <= frac * W.sum()
    r = np.zeros(169)
    r[order[keep]] = 1.0
    return r


FIELD = {"jam": _top_frac(0.6), "call": _top_frac(0.35)}


def run(target_hands: int, kinds: list[str], ante: float, start_bb: float, seed: int) -> dict:
    """Play SNGs until >= ``target_hands`` hands have been played, checking conservation on each
    hand via an instrumented ``play_hand`` wrapper. Returns the realized stats."""
    rng = np.random.default_rng(seed)
    cache: dict = {}

    hands = 0
    violations = 0
    worst_abs_drift = 0.0
    sngs = 0

    real_play_hand = sim.play_hand

    def counting_play_hand(stacks, order, kinds_, payouts, blinds_bb, ante_, rng_, cache_, field_):
        nonlocal hands, violations, worst_abs_drift
        total_before = float(sum(stacks))
        real_play_hand(stacks, order, kinds_, payouts, blinds_bb, ante_, rng_, cache_, field_)
        total_after = float(sum(stacks))
        drift = abs(total_after - total_before)
        if drift > worst_abs_drift:
            worst_abs_drift = drift
        if drift >= 1e-6:
            violations += 1
        hands += 1

    sim.play_hand = counting_play_hand
    t0 = time.perf_counter()
    try:
        while hands < target_hands:
            out = sim.simulate_sng(list(kinds), PAY, ante=ante, rng=rng, field=FIELD,
                                   cache=cache, start_bb=start_bb)
            sngs += 1
            # SNG-level conservation: the payout vector must conserve the prize pool exactly.
            assert abs(sum(out) - sum(PAY)) < 1e-9, "SNG payout pool not conserved"
    finally:
        sim.play_hand = real_play_hand
    elapsed = time.perf_counter() - t0

    return {
        "kinds": list(kinds),
        "ante_bb": ante,
        "start_bb": start_bb,
        "table_chip_total_bb": float(start_bb * len(kinds)),
        "sngs_played": sngs,
        "hands_played": hands,
        "per_hand_conservation_violations": violations,
        "worst_abs_chip_drift_bb": worst_abs_drift,
        "conserved": violations == 0,
        "hands_per_sec": round(hands / elapsed, 1) if elapsed > 0 else None,
        "elapsed_sec": round(elapsed, 2),
        "seed": seed,
    }


def main(argv) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hands", type=int, default=100_000, help="minimum hands to play")
    ap.add_argument("--kinds", nargs="+", default=["field", "field", "field"])
    ap.add_argument("--ante", type=float, default=0.1)
    ap.add_argument("--start-bb", type=float, default=12.0)
    ap.add_argument("--seed", type=int, default=20260622)
    ap.add_argument("--out", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "artifacts", "tourney_conservation.json"))
    args = ap.parse_args(argv[1:])

    res = run(args.hands, args.kinds, args.ante, args.start_bb, args.seed)
    res["full_icm_engine_command"] = (
        "python scripts/tournament/conservation_at_scale.py --hands 100000 --kinds icm field field")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)

    print(json.dumps(res, indent=2))
    status = "CONSERVED" if res["conserved"] else "VIOLATED"
    print(f"\n{status}: {res['hands_played']} hands, "
          f"{res['per_hand_conservation_violations']} violations, "
          f"worst drift {res['worst_abs_chip_drift_bb']:.2e} bb")
    return 0 if res["conserved"] else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
