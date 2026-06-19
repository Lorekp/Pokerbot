"""SCRATCH measurement harness (no production edits): fc-floor LBR6 vs blueprint+search.

Apples-to-apples preflop-quality probe for the lossless-preflop retrain vs the deployed WEVA blueprint.
Reuses pokerbot.eval.lbr6 + make_search_policy_fn + make_blueprint_policy unchanged. Reports the
fc-floor LBR edge (bb/100 + 95% CI), the PREFLOP-attributed component, and a blueprint-coverage report.

  python scripts/neural_cfr/lbr6_quality_probe.py --snapshot <pkl> [--search-preflop] --hands 5000 --seed 0

Serve env MUST be set BEFORE this imports pokerbot (the encoder bakes the abstraction flags at import):
  NEW lossless: POKERBOT_PREFLOP_LOSSLESS=1 POKERBOT_PREFLOP_LOSSLESS_BETTING=0
  OLD weva:     POKERBOT_PREFLOP_LOSSLESS=0 POKERBOT_PREFLOP_LOSSLESS_BETTING=0
plus POKERBOT_BETTING_ABSTRACTION=1 POKERBOT_POSTFLOP_BUCKETS=1 POKERBOT_POSTFLOP_MODE=fast
POKERBOT_POSTFLOP_NUM_BUCKETS=200 for both. The strict serve-env guard below rejects a mismatch.
"""

from __future__ import annotations

import argparse
import math
import os
import pickle
import time

os.environ["POKERBOT_BP_COVERAGE"] = "1"  # enable coverage counters (cheap; for the serve-skew sanity)

from pokerbot.eval import lbr6
from pokerbot.eval.lbr import (bp_coverage_report, bp_coverage_reset, check_blueprint_serve_env,
                               make_blueprint_policy)
from pokerbot.eval.solver_exploitability import make_search_policy_fn

_STREET = {-1: "no-postflop-call", 0: "preflop", 1: "flop", 2: "turn", 3: "river"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", required=True)
    ap.add_argument("--search-preflop", action="store_true")
    ap.add_argument("--hands", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--search-iters", type=int, default=10)
    args = ap.parse_args()

    with open(args.snapshot, "rb") as f:
        snap_full = pickle.load(f)
    stamp = snap_full.get("abstraction") if isinstance(snap_full, dict) else None
    check_blueprint_serve_env(strict=True, expected=stamp)   # rejects serve-skew
    snap = (snap_full["strat_sum"]
            if isinstance(snap_full, dict) and "strat_sum" in snap_full else snap_full)
    print(f"snapshot       : {args.snapshot}")
    print(f"  stamp        : {stamp}")
    print(f"  infosets     : {len(snap)}")
    print(f"  search_preflop: {args.search_preflop}   search_iters: {args.search_iters}")

    act = make_search_policy_fn(snap, leaf_select="uniform", iterations=args.search_iters,
                                seed=args.seed, search_preflop=args.search_preflop)
    model = make_blueprint_policy(snap)

    cfg = lbr6.LBR6Config(hands=args.hands, seed=args.seed, paired=False,
                          value_only_raises=True, range_cap=80)
    cfg.bet_candidates = ()   # fc floor: fold/call only (model-error-free, low-variance adversary)

    records: list[dict] = []
    bp_coverage_reset()
    t0 = time.time()
    res = lbr6.lbr6_exploitability(act, cfg, model_fn=model, hand_sink=records.append)
    dt = time.time() - t0

    n = len(records)
    lo = res.bb_per_100 - 1.96 * res.stderr_bb_per_hand * 100.0
    hi = res.bb_per_100 + 1.96 * res.stderr_bb_per_hand * 100.0
    print(f"\n=== fc-floor LBR6 vs blueprint+search ({n} hands, {dt/60:.1f} min, {dt/max(n,1):.3f}s/hand) ===")
    print(f"  fc edge    : {res.bb_per_100:+.2f} bb/100   95% CI [{lo:+.2f}, {hi:+.2f}]")
    print(f"  stderr     : {res.stderr_bb_per_hand*100.0:.2f} bb/100")
    print(f"  decisions  : {res.detail.get('lbr_decisions')}")

    # Preflop-attributed component (street of LBR's last voluntary call)
    print("\n  attribution by street of LBR's last voluntary call:")
    pre_contrib = 0.0
    for s in (-1, 0, 1, 2, 3):
        rs = [r["reward"] for r in records if r["last_call_street"] == s]
        if not rs:
            continue
        contrib = sum(rs) / n * 100.0
        if s == 0:
            pre_contrib = contrib
        print(f"    {_STREET[s]:18s} {len(rs):6d} {len(rs)/n*100:5.1f}%  contrib {contrib:+8.1f} bb/100")
    print(f"\n  PREFLOP-attributed: {pre_contrib:+.1f} bb/100")

    print("\n  blueprint coverage (hit = trained row used, miss = uniform fallback):")
    for street, d in bp_coverage_report().items():
        print(f"    {street:8s} hits={d['hits']:7d} misses={d['misses']:7d} hit_rate={d['hit_rate']*100:5.1f}%")


if __name__ == "__main__":
    main()
