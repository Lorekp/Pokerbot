"""Localize the blueprint+search leak: WHERE does the fc value-caller's +180 bb/100 come from?

Runs the fc-floor LBR vs blueprint+search and attributes each hand's LBR result to the STREET of its last
voluntary CALL (the decisive chips-in), plus a folded/saw-through split. Tells us whether the leak is
PREFLOP (base opens/3bets badly -> Stage 2 blueprint-range retrain) or POSTFLOP, and which street -- a
turn/river concentration implicates draw-equity mis-valuation -> Stage 4 potential-aware abstraction.

  python scripts/lbr6_attribute.py --hands 8000 --seed 0
"""

from __future__ import annotations

import argparse
import pickle
import time

from pokerbot.eval import lbr6
from pokerbot.eval.lbr import make_blueprint_policy
from pokerbot.eval.solver_exploitability import make_search_policy_fn

_STREET = {-1: "no-postflop-call", 0: "preflop", 1: "flop", 2: "turn", 3: "river", 4: "terminal"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hands", type=int, default=8000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--search-iters", type=int, default=10)
    ap.add_argument("--blueprint-snapshot",
                    default=r"checkpoints/bp6max_dcfrvr_iter200.pkl")
    args = ap.parse_args()

    with open(args.blueprint_snapshot, "rb") as f:
        snap = pickle.load(f)
    snap = snap["strat_sum"] if isinstance(snap, dict) and "strat_sum" in snap else snap
    act = make_search_policy_fn(snap, leaf_select="uniform", iterations=args.search_iters, seed=args.seed)
    model = make_blueprint_policy(snap)

    records: list[dict] = []
    cfg = lbr6.LBR6Config(hands=args.hands, seed=args.seed, paired=False, value_only_raises=True,
                          range_cap=80)   # match the diagnostic (lbr6_audit default)
    cfg.bet_candidates = ()   # fc floor: fold/call only (the clean, model-error-free adversary)

    t0 = time.time()
    res = lbr6.lbr6_exploitability(act, cfg, model_fn=model, hand_sink=records.append)
    dt = time.time() - t0

    n = len(records)
    total_bb = sum(r["reward"] for r in records)
    print(f"\n=== fc caller vs blueprint+search: profit attribution ({n} hands, {dt/60:.1f} min) ===")
    print(f"overall: {res.bb_per_100:+.1f} bb/100   (mean {total_bb/n:+.3f} bb/hand)\n")

    print("by STREET of the LBR's last voluntary call (where it commits the decisive chips):")
    print(f"  {'bucket':18s} {'n':>6s} {'freq':>6s} {'mean bb/hand':>13s} {'bb/100 contrib':>15s}")
    for s in (-1, 0, 1, 2, 3):
        rs = [r["reward"] for r in records if r["last_call_street"] == s]
        if not rs:
            continue
        contrib = sum(rs) / n * 100.0   # this bucket's share of the overall bb/100
        print(f"  {_STREET[s]:18s} {len(rs):6d} {len(rs)/n*100:5.1f}% {sum(rs)/len(rs):+13.3f} "
              f"{contrib:+15.1f}")

    print("\nby outcome:")
    for lbl, pred in (("LBR folded", lambda r: r["folded"]),
                      ("LBR saw it through", lambda r: not r["folded"])):
        rs = [r["reward"] for r in records if pred(r)]
        if rs:
            print(f"  {lbl:20s} {len(rs):6d} ({len(rs)/n*100:4.1f}%)  contrib {sum(rs)/n*100:+8.1f} bb/100")

    print("\nby DEEPEST street the hand reached (end_street):")
    for s in (0, 1, 2, 3):
        rs = [r["reward"] for r in records if r["end_street"] == s]
        if rs:
            print(f"  {_STREET[s]:10s} reached {len(rs):6d} ({len(rs)/n*100:4.1f}%)  contrib "
                  f"{sum(rs)/n*100:+8.1f} bb/100")

    print("\nINTERPRETATION: profit concentrated in preflop/no-postflop-call => base preflop ranges "
          "(Stage 2 retrain).\n  Profit concentrated on flop/turn/river calls => base mis-bets postflop; "
          "turn/river => draw mis-valuation (Stage 4 potential-aware abstraction).")


if __name__ == "__main__":
    main()
