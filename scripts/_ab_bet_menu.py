"""Bet-menu A/B: does a FINER LBR adversary bet menu find materially MORE exploitability
in blueprint+search than a coarse one? If delta ~ 0, finer betting buys ~nothing.

Coarse arm = (half, pot, all-in) = indices (4,7,10).
Fine   arm = full 7-size lattice + all-in = indices (3,4,5,6,7,8,9,10).

Same seed per arm pair => deck-level CRN, so the per-seed DELTA is low-variance even at
moderate N. value_only_raises=True on both (the repo's model-robust graded protocol: the LBR
bets/jams only when it profits on pure showdown value vs the calling range). range_cap=80,
search-iters=10 -- matches scripts/lbr6_attribute.py's diagnostic config.

Underscore-prefixed = scratch, not committed.
"""

from __future__ import annotations

import json
import os
import pickle
import statistics
import time
from pathlib import Path

# Match the default (non-lossless) snapshot's abstraction stamp BEFORE importing pokerbot
# (the encoder bakes at import + the strict serve-env guard would raise on a skew).
os.environ.pop("POKERBOT_PREFLOP_LOSSLESS", None)

from pokerbot.eval import lbr6  # noqa: E402
from pokerbot.eval.lbr import make_blueprint_policy, check_blueprint_serve_env  # noqa: E402
from pokerbot.eval.solver_exploitability import make_search_policy_fn  # noqa: E402

SNAP = r"checkpoints/bp6max_dcfrvr_iter200.pkl"
HANDS = int(os.environ.get("AB_HANDS", "8000"))
SEEDS = tuple(int(s) for s in os.environ.get("AB_SEEDS", "0,1,2,3").split(","))
SEARCH_ITERS = int(os.environ.get("AB_ITERS", "10"))
ARMS = {
    "coarse_3": (lbr6._HALF_POT_INDEX, lbr6._POT_BET_INDEX, lbr6.ALL_IN),       # (4,7,10)
    "fine_8":   (3, 4, 5, 6, 7, 8, 9, 10),                                       # full lattice + all-in
}
OUT = Path(r"./runs/bet_menu_ab")


def run_arm(snap_strat, seed: int, bet_candidates) -> float:
    act = make_search_policy_fn(snap_strat, leaf_select="uniform", iterations=SEARCH_ITERS, seed=seed)
    model = make_blueprint_policy(snap_strat)
    cfg = lbr6.LBR6Config(hands=HANDS, seed=seed, paired=False, value_only_raises=True, range_cap=80)
    cfg.bet_candidates = tuple(bet_candidates)
    res = lbr6.lbr6_exploitability(act, cfg, model_fn=model)
    return float(res.bb_per_100)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with open(SNAP, "rb") as f:
        snap = pickle.load(f)
    stamp = snap.get("abstraction") if isinstance(snap, dict) else None
    check_blueprint_serve_env(strict=True, expected=stamp)
    strat = snap["strat_sum"] if isinstance(snap, dict) and "strat_sum" in snap else snap

    print(f"=== bet-menu A/B: coarse(4,7,10) vs fine(3..10) | {HANDS} hands x {len(SEEDS)} seeds | "
          f"base=dcfrvr_iter200 search-iters={SEARCH_ITERS} value_only ===", flush=True)
    rows = []
    for seed in SEEDS:
        rec = {"seed": seed}
        for name, menu in ARMS.items():
            t0 = time.time()
            bb = run_arm(strat, seed, menu)
            rec[name] = bb
            print(f"  seed {seed}  {name:9s} {bb:+8.1f} bb/100   ({(time.time()-t0)/60:.1f} min)", flush=True)
        rec["delta_fine_minus_coarse"] = rec["fine_8"] - rec["coarse_3"]
        print(f"  seed {seed}  DELTA(fine-coarse) {rec['delta_fine_minus_coarse']:+8.1f} bb/100", flush=True)
        rows.append(rec)
        (OUT / "partial.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")

    deltas = [r["delta_fine_minus_coarse"] for r in rows]
    coarse = [r["coarse_3"] for r in rows]
    fine = [r["fine_8"] for r in rows]
    mean_d = statistics.mean(deltas)
    sd_d = statistics.pstdev(deltas) if len(deltas) > 1 else 0.0
    se_d = sd_d / (len(deltas) ** 0.5) if len(deltas) > 1 else 0.0
    summary = {
        "hands": HANDS, "seeds": list(SEEDS), "search_iters": SEARCH_ITERS,
        "coarse_mean": statistics.mean(coarse), "fine_mean": statistics.mean(fine),
        "delta_mean": mean_d, "delta_se": se_d,
        "delta_ci95": [mean_d - 1.96 * se_d, mean_d + 1.96 * se_d],
        "per_seed": rows,
        "verdict": ("FINER MENU HELPS (CI excludes 0)" if (mean_d - 1.96 * se_d) > 0
                    else "FINER MENU ~0 (CI straddles 0) -> finer betting buys nothing"),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\n=== POOLED ===", flush=True)
    print(f"  coarse mean {summary['coarse_mean']:+.1f}   fine mean {summary['fine_mean']:+.1f}", flush=True)
    print(f"  DELTA(fine-coarse) {mean_d:+.1f} +/- {se_d:.1f} bb/100  "
          f"CI95 [{summary['delta_ci95'][0]:+.1f}, {summary['delta_ci95'][1]:+.1f}]", flush=True)
    print(f"  VERDICT: {summary['verdict']}", flush=True)
    print(f"  wrote {OUT / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
