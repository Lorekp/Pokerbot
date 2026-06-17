#!/usr/bin/env python3
"""Profile the pluribus exploitation battery hot path. Usage:
   POKERBOT_BETTING_ABSTRACTION=1 python tools/battery_native/profile_battery.py --hands 40 --iterations 10
"""
from __future__ import annotations

import argparse
import cProfile
import pickle
import pstats
import random
import time

from pokerbot.eval.exploitation_battery import run_archetype


def build_factory(snap, leaf_select, iterations, allin_ev):
    from pokerbot.eval.production_hero import ProductionHero
    from pokerbot.eval.search_agent import SearchAgent

    def factory(*, exploit: bool):
        agent = SearchAgent(
            use_pluribus_search=True,
            blueprint_snapshot=snap,
            pluribus_leaf_select=leaf_select,
            time_budget_ms=10 ** 9,
            transformer_checkpoint=None,
            hero_seat=0,
        )
        agent.solver.config.max_iterations = iterations
        agent.solver.config.time_budget_ms = 10 ** 9
        return ProductionHero(agent, random.Random(0), exploit=exploit, hero_seat=0, n_seats=6)

    return factory


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hands", type=int, default=40)
    ap.add_argument("--iterations", type=int, default=10)
    ap.add_argument("--leaf-select", default="uniform")
    ap.add_argument("--archetype", default="calling_station")
    ap.add_argument("--allin-ev", action="store_true")
    ap.add_argument("--snapshot", default=r"checkpoints/bp6max_dcfrvr_iter200.pkl")
    ap.add_argument("--no-profile", action="store_true")
    args = ap.parse_args()

    with open(args.snapshot, "rb") as f:
        snap_full = pickle.load(f)
    snap = snap_full["strat_sum"] if isinstance(snap_full, dict) and "strat_sum" in snap_full else snap_full
    print(f"blueprint: {len(snap)} infosets")

    factory = build_factory(snap, args.leaf_select, args.iterations, args.allin_ev)

    def run():
        return run_archetype(args.archetype, hands=args.hands, seed=0, hero_factory=factory,
                             allin_ev=args.allin_ev, progress_every=0)

    if args.no_profile:
        t0 = time.perf_counter()
        r = run()
        dt = time.perf_counter() - t0
        print(f"{args.hands} hand-pairs in {dt:.2f}s = {dt/args.hands*1000:.1f} ms/hand-pair, "
              f"{args.hands/dt:.3f} hand-pairs/sec")
        print(f"uplift={r['uplift_bb_per_hand']:+.3f} zero_frac={r['uplift_zero_frac']:.3f}")
        return

    pr = cProfile.Profile()
    t0 = time.perf_counter()
    pr.enable()
    r = run()
    pr.disable()
    dt = time.perf_counter() - t0
    print(f"\n{args.hands} hand-pairs in {dt:.2f}s = {dt/args.hands*1000:.1f} ms/hand-pair, "
          f"{args.hands/dt:.3f} hand-pairs/sec")
    print(f"uplift={r['uplift_bb_per_hand']:+.3f} zero_frac={r['uplift_zero_frac']:.3f}\n")
    st = pstats.Stats(pr)
    st.sort_stats("tottime")
    st.print_stats(35)
    print("\n===== cumulative =====")
    st.sort_stats("cumulative")
    st.print_stats(35)


if __name__ == "__main__":
    main()
