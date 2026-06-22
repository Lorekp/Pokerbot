"""End-to-end SNG ROI: the certified push/fold PRIOR engine vs the UNIFORM fallback it supersedes.

Closes the up-to-7 tournament re-validation: does the certified ante-aware ICM push/fold prior
(96d24e9 -- the lever that replaced the search root's uniform-over-legal fallback at N>6) improve
*end-to-end* tournament results in a full 3-handed final-table SNG, not just the isolated push/fold
cell? And, honestly: is the engine's raw chip edge enough to clear the MTT fee floor vs an AVERAGE
field, or only vs a SOFT one (game selection is the lever, already built)?

We reuse pokerbot.tournament.sim verbatim (real cards + side pots + ICM + chip-conservation). The
sim's "icm" kind IS the prior engine: it routes preflop through decide.decide_pushfold ->
open_shove_icm.solve_open_shove_icm, the identical certified solver pokerbot.search.pushfold_prior
wraps as the search root prior (no fork). The new "uniform" kind is the 50/50 jam/fold-call/fold
fallback the prior replaced. So "icm vs uniform" here is exactly "prior vs uniform fallback",
measured end-to-end in money.

HONEST chip-EV vs net-$:
  * CHIP-EV (raw edge): mean $ winnings vs the fair share (sum(PAY)/3). Pre-rake -- the engine's
    real skill edge, no fee.
  * NET-$ ROI: after the ~8% MTT fee (each entry buys in at pool/(3*(1-rake))) minus CoinRewards'
    15% fee rebate. This is what a player actually keeps. We report both, with 95% CIs, and never
    let the chip edge masquerade as net profit vs an average field if the fee floor dominates
    (the documented cash analogue: memory aof-bot-correct-but-game-unbeatable).

FIELDS:
  * AVERAGE  -- IRC-calibrated loose-passive (jam42/call50): the realistic CoinPoker field.
  * SOFT     -- NIT (jam30/call15) and STATION (jam85/call50): the exploitable bookends a
                game-selecting bot would seek out.

Run: python scripts/tournament/sng_prior_vs_uniform_roi.py [N] [N_bookend]
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

EQ, _LAB, W = _hu_grid()
STRENGTH = EQ.mean(axis=1)
PAY = [65.0, 35.0, 0.0]                       # top-2 paid; bubble at 3 (where ICM bites)
FAIR = sum(PAY) / 3
RAKE_FRAC = 0.08                              # CoinPoker tournament fee (median)
COINREWARDS_RAKEBACK = 0.15                   # 15% Daily Guaranteed Rewards baseline
ANTE = 0.125                                  # CoinPoker MTT per-player ante (round(BB/8))


def top_frac(frac: float) -> np.ndarray:
    order = np.argsort(-STRENGTH)
    keep = np.cumsum(W[order]) <= frac * W.sum()
    r = np.zeros(169)
    r[order[keep]] = 1.0
    return r


FIELDS = {
    "AVERAGE  IRC loose-passive (jam42/call50)": ("average", {"jam": top_frac(0.42), "call": top_frac(0.50)}),
    "SOFT     NIT (jam30/call15)":               ("soft",    {"jam": top_frac(0.30), "call": top_frac(0.15)}),
    "SOFT     STATION (jam85/call50)":           ("soft",    {"jam": top_frac(0.85), "call": top_frac(0.50)}),
}


def _buyin() -> float:
    return sum(PAY) / (3.0 * (1.0 - RAKE_FRAC))


def _roi(mean_win: float, se_win: float) -> dict:
    """gross/net ROI% (+CI) and chip-EV vs fair share (+CI) from a seat's mean winnings."""
    b = _buyin()
    fee = b * RAKE_FRAC
    gross = (mean_win - b) / b * 100.0
    net = (mean_win - (b - COINREWARDS_RAKEBACK * fee)) / b * 100.0
    roi_ci = 1.96 * se_win / b * 100.0
    chip_ev = mean_win - FAIR                  # raw edge vs fair share, pre-rake ($)
    chip_ci = 1.96 * se_win
    return {"gross_roi": gross, "net_roi": net, "roi_ci95": roi_ci,
            "chip_ev": chip_ev, "chip_ev_ci95": chip_ci}


def run(seat0_kind: str, field: dict, n: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    cache: dict = {}
    res = np.array([sim.simulate_sng([seat0_kind, "field", "field"], PAY, ante=ANTE, rng=rng,
                                     field=field, cache=cache) for _ in range(n)])
    return float(res[:, 0].mean()), float(res[:, 0].std(ddof=1) / np.sqrt(n))


def main(argv) -> int:
    n = int(argv[1]) if len(argv) > 1 else 8000
    n_book = int(argv[2]) if len(argv) > 2 else 3000
    t0 = time.time()
    out: dict = {"pay": PAY, "fair_share": FAIR, "rake_frac": RAKE_FRAC,
                 "coinrewards_rakeback": COINREWARDS_RAKEBACK, "ante": ANTE, "buyin": _buyin(),
                 "n_average": n, "n_soft": n_book, "cells": {}}
    print(f"3-handed final-table SNG {PAY} (fair {FAIR:.2f}, buy-in {_buyin():.2f}); "
          f"prior(icm) vs uniform fallback; AVERAGE {n} / SOFT {n_book} SNGs\n", flush=True)
    for fname, (cls, field) in FIELDS.items():
        nc = n if cls == "average" else n_book
        base, bse = run("field", field, nc, seed=3)
        unif, use = run("uniform", field, nc, seed=6)
        nash, nse = run("icm", field, nc, seed=4)
        expl, ese = run("exploit", field, nc, seed=5)
        cell = {
            "class": cls,
            "field_self_mean": base, "field_self_se": bse,
            "uniform_mean": unif, "uniform_se": use, "uniform": _roi(unif, use),
            "prior_mean": nash, "prior_se": nse, "prior": _roi(nash, nse),
            "exploit_mean": expl, "exploit_se": ese, "exploit": _roi(expl, ese),
            "prior_minus_uniform_chip_ev": nash - unif,
            "prior_minus_uniform_ci95": 1.96 * np.hypot(nse, use),
        }
        out["cells"][fname] = cell
        pm = cell["prior"]; um = cell["uniform"]; xm = cell["exploit"]
        print(f" vs {fname}  [{cls.upper()}]")
        print(f"   field self-baseline : {base:6.2f} +/- {bse:.2f}  (fair {FAIR:.2f})")
        print(f"   UNIFORM fallback : chip-EV {um['chip_ev']:+6.2f} +/-{um['chip_ev_ci95']:.2f} $ | "
              f"net ROI {um['net_roi']:+6.2f}% +/-{um['roi_ci95']:.2f}")
        print(f"   PRIOR (icm)      : chip-EV {pm['chip_ev']:+6.2f} +/-{pm['chip_ev_ci95']:.2f} $ | "
              f"gross {pm['gross_roi']:+6.2f}% | net ROI {pm['net_roi']:+6.2f}% +/-{pm['roi_ci95']:.2f}")
        print(f"   EXPLOIT          : chip-EV {xm['chip_ev']:+6.2f} +/-{xm['chip_ev_ci95']:.2f} $ | "
              f"gross {xm['gross_roi']:+6.2f}% | net ROI {xm['net_roi']:+6.2f}% +/-{xm['roi_ci95']:.2f}")
        d = cell["prior_minus_uniform_chip_ev"]; dci = cell["prior_minus_uniform_ci95"]
        print(f"   prior - uniform  : {d:+6.2f} $ +/-{dci:.2f}  "
              f"=> prior {'IMPROVES end-to-end (CI-excluded)' if d - dci > 0 else 'not separated'}\n")

    out["elapsed_s"] = time.time() - t0
    art = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                       "artifacts", "sng_prior_vs_uniform_roi.json")
    with open(art, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print(f"wrote {art}  ({out['elapsed_s']:.0f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
