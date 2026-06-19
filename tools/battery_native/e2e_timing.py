#!/usr/bin/env python3
"""Gate 3: end-to-end run_field_absolute_rotated, native rollout OFF vs ON, in ONE process (cancels
the ~28s warmup + reduces contention skew). Reports both arms' off_base mean + CI and seconds/deck.
"""
from __future__ import annotations

import os
import pickle
import random
import sys
import time

os.environ.setdefault("POKERBOT_PREFLOP_LOSSLESS", "1")
os.environ.setdefault("POKERBOT_SEARCH_PREFLOP", "1")
os.environ.setdefault("POKERBOT_BETTING_ABSTRACTION", "1")
os.environ.setdefault("POKERBOT_POSTFLOP_BUCKETS", "1")
os.environ.setdefault("POKERBOT_POSTFLOP_MODE", "fast")
os.environ.setdefault("POKERBOT_POSTFLOP_NUM_BUCKETS", "200")

import pokerbot.search.pluribus_rollout as pr
from pokerbot.eval.exploitation_battery import (
    build_calibrated_field_bot,
    field_is_calibrated,
    field_spec,
    run_field_absolute_rotated,
)
from pokerbot.eval.production_hero import ProductionHero
from pokerbot.eval.search_agent import SearchAgent

SNAP = sys.argv[1] if len(sys.argv) > 1 else r"checkpoints/bp6max_lossless2_iter200.pkl"
DECKS = int(sys.argv[2]) if len(sys.argv) > 2 else 8
SEED = 0


def make_factory(snap):
    def factory(*, exploit: bool):
        agent = SearchAgent(
            use_pluribus_search=True, blueprint_snapshot=snap, pluribus_leaf_select="uniform",
            time_budget_ms=None, transformer_checkpoint=None, hero_seat=0,
            root_exploit_mixer=True, root_mixer_unopened=True, aivat_leaf_q=False,
        )
        agent.solver.config.max_iterations = 20
        agent.solver.config.time_budget_ms = 10 ** 9
        return ProductionHero(agent, random.Random(SEED), exploit=exploit, hero_seat=0, n_seats=6)
    return factory


def run_arm(label, native_on, snap, field, decks):
    pr._NATIVE_ROLLOUT = bool(native_on)
    # reset the per-adapter engine cache so each arm builds its own (and to drop stale handles)
    import pokerbot.search.native_solver as ns
    ns._PLAYOUT_ENGINE_CACHE.clear()
    factory = make_factory(snap)
    builder = build_calibrated_field_bot if field_is_calibrated("real_recreational_table") else None
    t = time.perf_counter()
    res = run_field_absolute_rotated(
        field, decks=decks, seed=SEED, hero_factory=factory, allin_ev=True, learn=True,
        n_seats=6, button_seat=3, progress_every=10_000, bot_builder=builder,
    )
    dt = time.perf_counter() - t
    return res, dt


def main():
    with open(SNAP, "rb") as f:
        full = pickle.load(f)
    snap = full["strat_sum"] if isinstance(full, dict) and "strat_sum" in full else full
    print(f"blueprint: {len(snap)} infosets; decks={DECKS}")
    field = field_spec("real_recreational_table")

    # Warm both paths once (build native engine + JAX) on 1 deck, untimed.
    _ = run_arm("warm-off", False, snap, field, 1)
    _ = run_arm("warm-on", True, snap, field, 1)

    off_res, off_dt = run_arm("OFF", False, snap, field, DECKS)
    on_res, on_dt = run_arm("ON", True, snap, field, DECKS)

    def ob(res):
        b = res["off_base"]
        m, se = b["mean_bb"], b["se"]
        return (m * 100.0, (m - 1.96 * se) * 100.0, (m + 1.96 * se) * 100.0)  # bb/100 + 95% CI

    print("\n=== RESULTS (off_base = pure-base seat-mean, the arm the prompt compares) ===")
    om, olo, ohi = ob(off_res)
    nm, nlo, nhi = ob(on_res)
    print(f"OFF(python) off_base = {om:+.1f} bb/100  CI95 [{olo:+.1f}, {ohi:+.1f}]  decks={off_res['decks']}")
    print(f"ON (native) off_base = {nm:+.1f} bb/100  CI95 [{nlo:+.1f}, {nhi:+.1f}]  decks={on_res['decks']}")
    overlap = not (ohi < nlo or nhi < olo)
    print(f"CIs overlap: {overlap}")
    print(f"\nOFF total {off_dt:.1f}s -> {off_dt/DECKS:.2f} s/deck")
    print(f"ON  total {on_dt:.1f}s -> {on_dt/DECKS:.2f} s/deck")
    print(f"SPEEDUP (s/deck) x{off_dt/on_dt:.2f}")


if __name__ == "__main__":
    main()
