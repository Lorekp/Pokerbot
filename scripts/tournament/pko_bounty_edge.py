"""PKO chip-EV != $-EV: does the bounty-AWARE arm collect more total $ than the prize-only ICM arm?

Plays 3-handed PKO push/fold SNGs (``pokerbot.tournament.sim_pko``). seat0 is either the bounty-aware
``pko`` arm (calls wider, valuing the jammer's live bounty via ``bounty.min_equity_to_call``) or the
prize-only ``icm`` arm (the certified ICM push/fold, bounty-blind); seats 1-2 are the IRC loose-passive
field. Total realized $ per seat = prize finish $ + bounty cash. The CI-guarded EDGE = mean(pko) -
mean(icm); a 95% lower-confidence-bound > 0 demonstrates that valuing the bounty makes wider calls +$EV
(chip-EV != $-EV) -- the ULTRACODE Workstream-C PKO acceptance, otherwise unmeasured.

Run: ``python scripts/tournament/pko_bounty_edge.py [N] [iters_via_env]``  (POKERBOT_SIM_PUSHFOLD_ITERS
lowers the ICM-arm solver iters for speed; the bounty-arm call is a closed form, already fast).
"""

from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np

from pokerbot.tournament.sim_pko import _EQ, _W, simulate_pko_sng

_STRENGTH = _EQ.mean(axis=1)

# PKO economics: prize pool == bounty pool (CoinPoker 50/50). Prize ladder pays top-2 of 3; the bubble at
# 3 is where ICM bites. entry_bounty chosen so the bounty pool == the prize pool.
PRIZE = [65.0, 35.0, 0.0]
ENTRY_BOUNTY = sum(PRIZE) / 3.0            # 3 * entry_bounty == prize pool -> 50/50 prize/bounty
ANTE = 0.125
RAKE_FRAC = 0.08
_BUYIN = (sum(PRIZE) + 3.0 * ENTRY_BOUNTY) / (3.0 * (1.0 - RAKE_FRAC))   # prize+bounty pool over 3 entries


def _top_frac(frac: float) -> np.ndarray:
    order = np.argsort(-_STRENGTH)
    keep = np.cumsum(_W[order]) <= frac * _W.sum()
    r = np.zeros(169)
    r[order[keep]] = 1.0
    return r


# IRC loose-passive field (the same character as roi_vs_irc): under-fold / over-call.
FIELD = {"jam": _top_frac(0.42), "call": _top_frac(0.50)}


def run(seat0_kind: str, n: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    cache: dict = {}
    res = np.array([simulate_pko_sng([seat0_kind, "field", "field"], PRIZE, ENTRY_BOUNTY, ante=ANTE,
                                     rng=rng, field=FIELD, cache=cache)[0] for _ in range(n)])
    return float(res.mean()), float(res.std() / np.sqrt(n))


def main(argv) -> int:
    n = int(argv[1]) if len(argv) > 1 else 2000
    t0 = time.time()
    pko_mean, pko_se = run("pko", n, seed=11)
    icm_mean, icm_se = run("icm", n, seed=12)
    edge = pko_mean - icm_mean
    edge_se = (pko_se ** 2 + icm_se ** 2) ** 0.5      # independent-seed arms
    edge_lcb = edge - 1.96 * edge_se
    out = {
        "n": n, "prize": PRIZE, "entry_bounty": ENTRY_BOUNTY, "buyin": _BUYIN,
        "pko_total_mean": pko_mean, "pko_total_se": pko_se,
        "icm_total_mean": icm_mean, "icm_total_se": icm_se,
        "bounty_edge": edge, "bounty_edge_se": edge_se, "bounty_edge_lcb95": edge_lcb,
        "edge_pct_buyin": edge / _BUYIN * 100.0,
        "chip_ev_neq_dollar_ev": bool(edge_lcb > 0.0),    # CI-guarded: the bounty lever is real
        "elapsed_s": time.time() - t0,
    }
    art = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                       "artifacts", "pko_bounty_edge.json")
    os.makedirs(os.path.dirname(art), exist_ok=True)
    with open(art, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print(f"PKO 3-handed SNG (prize {PRIZE}, entry bounty {ENTRY_BOUNTY:.2f}, buy-in {_BUYIN:.2f}); "
          f"{n} SNGs/arm vs IRC field")
    print(f"  bounty-aware (pko) total $ : {pko_mean:6.2f} +/- {pko_se:.2f}")
    print(f"  prize-only   (icm) total $ : {icm_mean:6.2f} +/- {icm_se:.2f}")
    print(f"  BOUNTY EDGE  : {edge:+.2f} +/- {edge_se:.2f}  (95% LCB {edge_lcb:+.2f}, "
          f"{out['edge_pct_buyin']:+.2f}% of buy-in)")
    print(f"  chip-EV != $-EV (CI-guarded): {out['chip_ev_neq_dollar_ev']}")
    print(f"wrote {art}  ({out['elapsed_s']:.0f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
