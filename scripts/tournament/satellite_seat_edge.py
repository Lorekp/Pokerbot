"""Satellite seat-lock edge: with a FLAT seat-payoff (a satellite pays ``seats`` identical seats, rest
nothing), does the ICM-aware arm secure a seat MORE often than a naive over-shoving field seat?

A satellite is the extreme-ICM regime: above the bubble the payoff is flat, so marginal chips are worth
~nothing and a locked seat is max equity -> fold everything that risks the seat. This reuses the certified
push/fold SNG (``sim.simulate_sng``) with a SATELLITE payout vector ``[seat_value]*seats + [0]`` -- no new
engine needed; the ICM solver handles the flat payoff directly (it folds when locked). The metric is mean
$ = P(secure a seat) * seat_value, CI-guarded edge vs a field seat.

Run: ``python scripts/tournament/satellite_seat_edge.py [N]``  (POKERBOT_SIM_PUSHFOLD_ITERS for speed).
"""

from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np

import pokerbot.tournament.sim as sim
from pokerbot.tournament.pushfold_icm import _hu_grid

_EQ, _LAB, _W = _hu_grid()
_STRENGTH = _EQ.mean(axis=1)

# 3 players, 2 identical seats -> flat top-2 payoff (the satellite bubble where ICM is most extreme).
SEAT_VALUE = 1.0
PAY = [SEAT_VALUE, SEAT_VALUE, 0.0]
FAIR = sum(PAY) / 3.0
ANTE = 0.125


def _top_frac(frac: float) -> np.ndarray:
    order = np.argsort(-_STRENGTH)
    keep = np.cumsum(_W[order]) <= frac * _W.sum()
    r = np.zeros(169)
    r[order[keep]] = 1.0
    return r


FIELD = {"jam": _top_frac(0.42), "call": _top_frac(0.50)}     # IRC loose-passive over-caller


def run(seat0_kind: str, n: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    cache: dict = {}
    res = np.array([sim.simulate_sng([seat0_kind, "field", "field"], PAY, ante=ANTE, rng=rng,
                                     field=FIELD, cache=cache)[0] for _ in range(n)])
    return float(res.mean()), float(res.std() / np.sqrt(n))


def main(argv) -> int:
    n = int(argv[1]) if len(argv) > 1 else 2000
    t0 = time.time()
    icm_mean, icm_se = run("icm", n, seed=21)
    fld_mean, fld_se = run("field", n, seed=22)            # a field seat as the naive baseline
    edge = icm_mean - fld_mean
    edge_se = (icm_se ** 2 + fld_se ** 2) ** 0.5
    edge_lcb = edge - 1.96 * edge_se
    out = {
        "n": n, "pay": PAY, "fair_seat_share": FAIR, "ante": ANTE,
        "icm_seat_prob": icm_mean, "icm_se": icm_se,
        "field_seat_prob": fld_mean, "field_se": fld_se,
        "seat_edge_vs_field": edge, "seat_edge_se": edge_se, "seat_edge_lcb95": edge_lcb,
        "icm_beats_fair": bool(icm_mean - 1.96 * icm_se > FAIR),
        "icm_beats_field_ci": bool(edge_lcb > 0.0),
        "elapsed_s": time.time() - t0,
    }
    art = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                       "artifacts", "satellite_seat_edge.json")
    os.makedirs(os.path.dirname(art), exist_ok=True)
    with open(art, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print(f"Satellite 3-player / 2-seat SNG (flat {PAY}, fair seat share {FAIR:.3f}); {n} SNGs/arm vs IRC field")
    print(f"  ICM (seat-aware) P(seat) : {icm_mean:.3f} +/- {icm_se:.3f}  (beats fair {FAIR:.3f}: {out['icm_beats_fair']})")
    print(f"  field seat        P(seat) : {fld_mean:.3f} +/- {fld_se:.3f}")
    print(f"  SEAT EDGE vs field : {edge:+.3f} +/- {edge_se:.3f}  (95% LCB {edge_lcb:+.3f}) -> ICM wins seats: {out['icm_beats_field_ci']}")
    print(f"wrote {art}  ({out['elapsed_s']:.0f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
