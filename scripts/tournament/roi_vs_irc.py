"""ROI of the tournament engine vs an IRC-calibrated loose-passive field (Workstream C acceptance).

Closes the C acceptance line: "Positive ROI vs an IRC-calibrated tournament field in sim.py; rake/fee
accounting via CoinRewards (15% baseline) reflected in net ROI."

THE FIELD (IRC-calibrated, character-faithful). The IRC fit (7.4M hands, 25,235 players; see
``data/player_profiles/irc_population_prior_fitted.json`` + ``pokerbot/opponent/population_prior``)
is LOOSE-PASSIVE: median VPIP ~48% (preflop-unopened fold ~0.42 in the fitted prior), PFR ~10%,
aggression ~0.65, and the station signature -- UNDER-FOLD / OVER-CALL (preflop facing-a-bet fold only
~0.42). IRC hold'em is mostly LIMIT, so all-in sizings are under-represented (the fit's own caveat) --
a direct bin->push/fold map is unreliable. We therefore calibrate the push/fold archetype to the IRC
*character* with documented parameters rather than a naive bin map:

  * JAM  ~0.42  -- a VPIP-toned first-in range (loose, but passive players under-jam vs a GTO ~0.55).
  * CALL ~0.50  -- the station UNDER-FOLD signature (GTO call vs a 12bb jam is ~0.30; they call ~0.50).

The exploitable lever is the OVER-WIDE CALL: we jam a value-skewed range the station over-calls, the
thin-value-vs-passive thesis (see memory ``population-hud-analysis``). NIT and STATION bookends are run
alongside for context.

CoinRewards NET. The prize pool ``[65,35,0]`` is post-rake, so each buy-in = pool / (n * (1-rake)).
A fair player loses exactly the rake (the floor). CoinRewards' 15% Daily Guaranteed Rewards baseline
rebates 15% of the fee, lifting net ROI. We report GROSS and NET (CoinRewards-adjusted) ROI%.

Run: python scripts/tournament/roi_vs_irc.py [N_per_cell]   (writes artifacts/roi_vs_irc_result.json)
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
PAY = [65.0, 35.0, 0.0]                       # top-2 paid; the bubble is at 3 (where ICM bites)
FAIR = sum(PAY) / 3
RAKE_FRAC = 0.08                              # CoinPoker tournament fee (median, tournament_payout.txt)
COINREWARDS_RAKEBACK = 0.15                   # 15% Daily Guaranteed Rewards baseline (coinpokerinfo.txt)
ANTE = 0.125                                  # CoinPoker MTT per-player ante (round(BB/8))


def top_frac(frac: float) -> np.ndarray:
    order = np.argsort(-STRENGTH)
    keep = np.cumsum(W[order]) <= frac * W.sum()
    r = np.zeros(169)
    r[order[keep]] = 1.0
    return r


# IRC-calibrated loose-passive field + nit/station bookends.
FIELDS = {
    "IRC loose-passive (jam42/call50)": {"jam": top_frac(0.42), "call": top_frac(0.50)},
    "NIT (jam30/call15)": {"jam": top_frac(0.30), "call": top_frac(0.15)},
    "STATION (jam85/call50)": {"jam": top_frac(0.85), "call": top_frac(0.50)},
}


def _buyin() -> float:
    """Per-entry buy-in (incl. fee) implied by a post-rake pool of ``sum(PAY)`` over 3 entries."""
    return sum(PAY) / (3.0 * (1.0 - RAKE_FRAC))


def _roi(mean_win: float) -> tuple[float, float]:
    """(gross_roi%, coinrewards_net_roi%) from a seat's mean winnings."""
    b = _buyin()
    fee = b * RAKE_FRAC
    gross = (mean_win - b) / b * 100.0
    net = (mean_win - (b - COINREWARDS_RAKEBACK * fee)) / b * 100.0   # 15% fee rebate
    return gross, net


def run(seat0_kind: str, field: dict, n: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    cache: dict = {}
    res = np.array([sim.simulate_sng([seat0_kind, "field", "field"], PAY, ante=ANTE, rng=rng,
                                     field=field, cache=cache) for _ in range(n)])
    return float(res[:, 0].mean()), float(res[:, 0].std() / np.sqrt(n))


def main(argv) -> int:
    # argv[1] = N for the IRC acceptance cell (high); argv[2] = N for the NIT/STATION bookends (light).
    n = int(argv[1]) if len(argv) > 1 else 8000
    n_book = int(argv[2]) if len(argv) > 2 else 2000
    irc_name = "IRC loose-passive (jam42/call50)"
    t0 = time.time()
    out: dict = {"n_irc": n, "n_bookend": n_book, "pay": PAY, "fair_share": FAIR,
                 "rake_frac": RAKE_FRAC, "coinrewards_rakeback": COINREWARDS_RAKEBACK,
                 "ante": ANTE, "buyin": _buyin(), "cells": {}}
    print(f"3-handed final-table SNG {PAY} (fair {FAIR:.2f}, buy-in {_buyin():.2f}); "
          f"IRC cell {n} SNGs, bookends {n_book}\n", flush=True)
    for fname, field in FIELDS.items():
        nc = n if fname == irc_name else n_book
        base, bse = run("field", field, nc, seed=3)        # fairness baseline (a field seat)
        nash, nse = run("icm", field, nc, seed=4)
        expl, ese = run("exploit", field, nc, seed=5)
        ng, nn = _roi(nash)
        eg, en = _roi(expl)
        cell = {
            "field_self_mean": base, "field_self_se": bse,
            "nash_mean": nash, "nash_se": nse, "nash_gross_roi": ng, "nash_net_roi": nn,
            "exploit_mean": expl, "exploit_se": ese, "exploit_gross_roi": eg, "exploit_net_roi": en,
        }
        out["cells"][fname] = cell
        print(f" vs {fname}")
        print(f"   field self-baseline : {base:6.2f} +/- {bse:.2f}  (fair {FAIR:.2f})")
        print(f"   ICM-NASH    : win {nash:6.2f}  gross ROI {ng:+6.2f}%  net(CoinRewards) {nn:+6.2f}%")
        print(f"   ICM-EXPLOIT : win {expl:6.2f}  gross ROI {eg:+6.2f}%  net(CoinRewards) {en:+6.2f}%\n")

    irc = out["cells"]["IRC loose-passive (jam42/call50)"]
    out["verdict"] = {
        "exploit_beats_nash_vs_irc": irc["exploit_mean"] > irc["nash_mean"],
        "exploit_net_roi_positive_vs_irc": irc["exploit_net_roi"] > 0.0,
        "exploit_net_roi_vs_irc": irc["exploit_net_roi"],
    }
    out["elapsed_s"] = time.time() - t0
    art = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                       "artifacts", "roi_vs_irc_result.json")
    with open(art, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print(f"VERDICT vs IRC field: exploit net ROI {irc['exploit_net_roi']:+.2f}% "
          f"(exploit {'>' if irc['exploit_mean'] > irc['nash_mean'] else '<='} nash)")
    print(f"wrote {art}  ({out['elapsed_s']:.0f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
