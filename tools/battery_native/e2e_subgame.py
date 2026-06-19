#!/usr/bin/env python3
"""End-to-end: run_field_absolute_rotated across THREE engine arms in one process (cancels warmup):
  python    -- pure Python _cfv loop (native ENGINE primitives still on; the baseline)
  leaf      -- native uniform continuation leaf only  (POKERBOT_NATIVE_ROLLOUT)
  subgame   -- native whole-CFR subgame solve         (POKERBOT_NATIVE_SUBGAME)

Reports each arm's off_base mean + 95% CI (must overlap -> statistically equivalent) and s/deck +
speedup over the python baseline.
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
import pokerbot.search.pluribus_search as psearch
import pokerbot.search.native_solver as ns
from pokerbot.eval.exploitation_battery import (
    build_calibrated_field_bot,
    field_is_calibrated,
    field_spec,
    run_field_absolute_rotated,
)
from pokerbot.eval.production_hero import ProductionHero
from pokerbot.eval.search_agent import SearchAgent

SNAP = sys.argv[1] if len(sys.argv) > 1 else r"checkpoints/bp6max_lossless2_iter200.pkl"
DECKS = int(sys.argv[2]) if len(sys.argv) > 2 else 24
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


def run_arm(label, snap, field, decks, *, native_leaf, native_subgame):
    pr._NATIVE_ROLLOUT = bool(native_leaf)
    psearch.set_native_subgame(bool(native_subgame))
    ns._PLAYOUT_ENGINE_CACHE.clear()
    ns._SUBGAME_ENGINE_CACHE.clear()
    factory = make_factory(snap)
    builder = build_calibrated_field_bot if field_is_calibrated("real_recreational_table") else None
    t = time.perf_counter()
    res = run_field_absolute_rotated(
        field, decks=decks, seed=SEED, hero_factory=factory, allin_ev=True, learn=True,
        n_seats=6, button_seat=3, progress_every=10_000, bot_builder=builder,
    )
    return res, time.perf_counter() - t


def ob(res):
    b = res["off_base"]
    m, se = b["mean_bb"], b["se"]
    return (m * 100.0, (m - 1.96 * se) * 100.0, (m + 1.96 * se) * 100.0)


def main():
    with open(SNAP, "rb") as f:
        full = pickle.load(f)
    snap = full["strat_sum"] if isinstance(full, dict) and "strat_sum" in full else full
    print(f"blueprint: {len(snap)} infosets; decks={DECKS}")
    field = field_spec("real_recreational_table")

    # Warm all paths once (build engines + JAX) on 1 deck, untimed.
    for nl, ns_ in ((False, False), (True, False), (False, True)):
        run_arm("warm", snap, field, 1, native_leaf=nl, native_subgame=ns_)

    arms = [("python", False, False), ("leaf", True, False), ("subgame", False, True)]
    results = {}
    for label, nl, ns_ in arms:
        res, dt = run_arm(label, snap, field, DECKS, native_leaf=nl, native_subgame=ns_)
        results[label] = (res, dt)

    print("\n=== off_base (pure-base seat-mean) per arm ===")
    base_dt = results["python"][1]
    for label, _, _ in arms:
        res, dt = results[label]
        m, lo, hi = ob(res)
        spd = base_dt / dt if dt > 0 else 0.0
        print(f"{label:8s} off_base={m:+8.1f} bb/100  CI95 [{lo:+8.1f},{hi:+8.1f}]  "
              f"{dt:6.1f}s  {dt / DECKS:5.2f} s/deck  x{spd:4.2f}")
    # CI overlap of subgame vs python (the equivalence gate).
    pm, plo, phi = ob(results["python"][0])
    sm, slo, shi = ob(results["subgame"][0])
    overlap = not (phi < slo or shi < plo)
    print(f"\nsubgame vs python CIs overlap: {overlap}")


if __name__ == "__main__":
    main()
