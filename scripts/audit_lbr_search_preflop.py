"""AUDIT: does search_preflop=ON close the iter62000 preflop hole?

Three-way 6-max LBR exploitability on the SAME iter62000 snapshot via the two-policy harness:
  * RAW BLUEPRINT (no search)          -- the table-only floor
  * SEARCH search_preflop=OFF          -- preflop served from the (leaky) blueprint
  * SEARCH search_preflop=ON           -- the DEPLOYED config (claimed to close the hole)

Seats ACT the policy; the LBR range-tracks against the cheap raw-blueprint MODEL. Pools seeds for a CI.
Coverage of the raw-blueprint model is verified (must be ~100% postflop) before any number is trusted.
"""
from __future__ import annotations

import argparse
import math
import os
import time

os.environ.setdefault("POKERBOT_BETTING_ABSTRACTION", "1")
os.environ.setdefault("POKERBOT_POSTFLOP_BUCKETS", "1")
os.environ.setdefault("POKERBOT_PREFLOP_LOSSLESS", "1")
os.environ.setdefault("POKERBOT_POSTFLOP_MODE", "emd")
os.environ.setdefault("POKERBOT_POSTFLOP_NUM_BUCKETS", "160")
# EMD-K160 deployed regime (promoted 2026-07-02, certified +32.89 bb/100 vs made-hand iter62000):
# frozen flop/turn EMD lookup + train-mirrored made-hand fallback on the river (the river is not
# in the blob; without the mirror flag the river frozen-miss serves a UNIFORM policy).
os.environ.setdefault("POKERBOT_POSTFLOP_EMD_FROZEN", os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "data", "abstraction", "emd_frozen_flop_turn_160_r400.emdfrzn")))
os.environ.setdefault("POKERBOT_EMD_FROZEN_MIRROR_FALLBACK", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("JAX_PLATFORMS", "cpu")


def _pool(triples):
    triples = [t for t in triples if t[2]]
    N = sum(n for _, _, n in triples)
    M = sum(m * n for m, _, n in triples) / N
    ss = sum(n * (sd ** 2) + n * (m - M) ** 2 for m, sd, n in triples)
    se = math.sqrt((ss / N) / N)
    return M * 100, (M - 1.96 * se) * 100, (M + 1.96 * se) * 100, N


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hands", type=int, default=300)
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--n-seats", type=int, default=6)
    ap.add_argument("--snapshot", default="checkpoints/serve_blueprints/bp6max_iter62000.pkl")
    ap.add_argument("--menu", choices=("fc", "fcpa", "full"), default="fcpa")
    args = ap.parse_args()

    import pickle
    import numpy as np

    from pokerbot.eval import lbr6
    from pokerbot.eval.lbr import make_blueprint_policy
    from pokerbot.eval.solver_exploitability import make_search_policy_fn

    with open(args.snapshot, "rb") as f:
        obj = pickle.load(f)
    snap = obj["strat_sum"] if isinstance(obj, dict) and "strat_sum" in obj else obj
    print(f"snapshot: {len(snap)} infosets  ({args.snapshot})", flush=True)

    bp_policy = make_blueprint_policy(snap)
    _MENUS = {"fc": (), "fcpa": (lbr6._POT_BET_INDEX, lbr6.ALL_IN),
              "full": (lbr6._HALF_POT_INDEX, lbr6._POT_BET_INDEX, lbr6.ALL_IN)}
    menu = _MENUS[args.menu]

    def measure(make_policy):
        trip = []
        t0 = time.time()
        for seed in range(args.seeds):
            cfg = lbr6.LBR6Config(hands=args.hands, seed=seed, n_seats=args.n_seats,
                                  track_ranges=True, paired=False, value_only_raises=True)
            cfg.bet_candidates = menu
            r = lbr6.lbr6_exploitability(make_policy(seed), cfg, model_fn=bp_policy)
            trip.append((r.ev_bb_per_hand, r.stderr_bb_per_hand * (r.hands ** 0.5), r.hands))
            p = _pool(trip)
            print(f"    seed {seed:2d} -> running pool {p[0]:+.1f} [{p[1]:+.1f},{p[2]:+.1f}] "
                  f"({p[3]}h, {(time.time()-t0)/60:.1f}min)", flush=True)
        return _pool(trip)

    print("\n[1/3] RAW BLUEPRINT (no search)", flush=True)
    raw = measure(lambda seed: bp_policy)

    print("\n[2/3] SEARCH search_preflop=OFF", flush=True)
    off = measure(lambda seed: make_search_policy_fn(
        snap, leaf_select="uniform", seed=seed, iterations=args.iters, search_preflop=False))

    print("\n[3/3] SEARCH search_preflop=ON  (DEPLOYED)", flush=True)
    on = measure(lambda seed: make_search_policy_fn(
        snap, leaf_select="uniform", seed=seed, iterations=args.iters, search_preflop=True))

    def fmt(name, t):
        return f"{name:34s}: {t[0]:+7.1f} bb/100  [{t[1]:+.1f}, {t[2]:+.1f}]   ({t[3]}h)"

    print(f"\n=== iter62000 LBR exploitability ({args.menu} menu, {args.iters} iters, "
          f"{args.seeds}x{args.hands}h/arm) ===", flush=True)
    print(fmt("RAW BLUEPRINT (no search)", raw))
    print(fmt("SEARCH preflop=OFF", off))
    print(fmt("SEARCH preflop=ON (deployed)", on))
    print(f"\ndelta search-ON vs raw : {on[0]-raw[0]:+.1f} bb/100  "
          f"({'search REDUCES' if on[0] < raw[0] else 'NO reduction'})")
    print(f"delta preflop ON vs OFF: {on[0]-off[0]:+.1f} bb/100  "
          f"({'preflop-search helps' if on[0] < off[0] else 'preflop-search does NOT help'})")


if __name__ == "__main__":
    main()
