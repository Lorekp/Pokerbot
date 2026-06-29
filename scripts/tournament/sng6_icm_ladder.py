"""6-handed multiway-ICM laddering edge: does the certified ICM open-shove arm ladder into the money MORE
than a loose-passive field seat at the 6-handed bubble (top-3 of 6 paid)?

This exercises the MULTIWAY ICM the 3-handed SNG (roi_vs_irc) cannot: a 6-player table with TWO pay jumps
(3rd->2nd->1st) where the ICM arm must fold chip-EV-profitable shoves that risk laddering past the
short stacks, and shove where the field cannot call. Reuses ``sim.simulate_sng_n`` (N-handed) + the
certified ``decide_pushfold`` ICM solve. CI-guarded edge = ICM-arm $ minus a field-seat's $.

Run: python scripts/tournament/sng6_icm_ladder.py [N]   (POKERBOT_SIM_PUSHFOLD_ITERS lowers solver iters).
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

PAY = [50.0, 30.0, 20.0]          # 6 players, top-3 paid -> two pay jumps (multiway ICM)
FAIR = sum(PAY) / 6.0             # a fair seat's expected $ (6-way split of the pool)
ANTE = 0.125
START_BB = 15.0                   # shallow multiway bubble (where ICM bites across 6 seats)


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
    res = np.array([sim.simulate_sng_n([seat0_kind] + ["field"] * 5, PAY, ante=ANTE, rng=rng,
                                       field=FIELD, cache=cache, n_players=6, start_bb=START_BB)[0]
                    for _ in range(n)])
    return float(res.mean()), float(res.std() / np.sqrt(n))


def main(argv) -> int:
    n = int(argv[1]) if len(argv) > 1 else 1500
    t0 = time.time()
    icm_mean, icm_se = run("icm", n, seed=31)
    fld_mean, fld_se = run("field", n, seed=32)
    edge = icm_mean - fld_mean
    edge_se = (icm_se ** 2 + fld_se ** 2) ** 0.5
    edge_lcb = edge - 1.96 * edge_se
    out = {
        "n": n, "pay": PAY, "fair_seat_share": FAIR, "start_bb": START_BB, "n_players": 6,
        "icm_mean": icm_mean, "icm_se": icm_se, "field_mean": fld_mean, "field_se": fld_se,
        "icm_ladder_edge": edge, "edge_se": edge_se, "edge_lcb95": edge_lcb,
        "icm_beats_fair_ci": bool(icm_mean - 1.96 * icm_se > FAIR),
        "icm_beats_field_ci": bool(edge_lcb > 0.0),
        "elapsed_s": time.time() - t0,
    }
    art = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                       "artifacts", "sng6_icm_ladder.json")
    os.makedirs(os.path.dirname(art), exist_ok=True)
    with open(art, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print(f"6-handed ICM SNG (pay {PAY}, fair seat {FAIR:.2f}, start {START_BB}bb); {n} SNGs/arm vs IRC field")
    print(f"  ICM (multiway open-shove) $ : {icm_mean:6.2f} +/- {icm_se:.2f}  (beats fair {FAIR:.2f}: {out['icm_beats_fair_ci']})")
    print(f"  field seat               $ : {fld_mean:6.2f} +/- {fld_se:.2f}")
    print(f"  ICM LADDER EDGE vs field : {edge:+.2f} +/- {edge_se:.2f}  (95% LCB {edge_lcb:+.2f}) -> {out['icm_beats_field_ci']}")
    print(f"wrote {art}  ({out['elapsed_s']:.0f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
