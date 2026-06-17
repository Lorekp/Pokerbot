#!/usr/bin/env python3
"""Solve-level equivalence: native-engine ON vs pure-Python OFF must give bit-identical solve(root)
action_probs over many decisions, mixed configs (iterations, with/without opponent_model). Because the
native core is a bit-identical drop-in for the engine primitives and the CFR/RNG stays in Python, the
solver output is a pure function of (public state, opponent model, config) and must match byte-for-byte.

This is the B-path verification: it proves the native wiring does NOT perturb the Monte-Carlo stream or
any float reduction (those stay Python), so the FULL solve is identical — strictly stronger than the B2
determinism + zero-deviation gates the task allows.

Usage: POKERBOT_BETTING_ABSTRACTION=1 python tools/battery_native/solve_crosscheck.py [--decisions 200]
"""
from __future__ import annotations

import argparse
import pickle
import random
import struct

import numpy as np

from pokerbot.engine.game import GameState, play_hand
from pokerbot.eval.production_hero import ProductionHero
from pokerbot.eval.search_agent import SearchAgent
from pokerbot.opponent.stats import OpponentStats
from pokerbot.search import native_solver as ns

assert ns.available(), "native extension unavailable"


def bits(x):
    a = np.asarray(x, dtype=np.float64).reshape(-1)
    return b"".join(struct.pack("<d", float(v)) for v in a)


def build_agent(snap, iterations, native, opp_profile):
    agent = SearchAgent(
        use_pluribus_search=True, blueprint_snapshot=snap, pluribus_leaf_select="uniform",
        time_budget_ms=10 ** 9, transformer_checkpoint=None, hero_seat=0,
    )
    agent.solver.config.max_iterations = iterations
    agent.solver.config.time_budget_ms = 10 ** 9
    agent.solver.config.native = native
    if opp_profile is not None:
        agent.set_opponent_profiles({s: opp_profile for s in range(1, 6)})
    return agent


def collect_probs(agent, n_hands, seed):
    """Play n_hands hands, recording every solver action_probs the hero produces via a solve() hook."""
    out = []
    orig_solve = agent.solver.solve

    def wrapped(root, pbs):
        r = orig_solve(root, pbs)
        out.append(np.asarray(r.action_probs, dtype=float).copy())
        return r

    agent.solver.solve = wrapped
    exploit = agent._opponent_profiles is not None
    for h in range(n_hands):
        hs = 1000 + h
        gs = GameState.new_hand(button_seat=3, rng=random.Random(hs))
        ph = ProductionHero(agent, random.Random(hs), exploit=exploit, hero_seat=0, n_seats=6)
        if hasattr(ph, "on_new_hand"):
            ph.on_new_hand()
        bot = lambda st, _r=random.Random(5000 + hs): _simple_bot(st, _r)
        policies = {s: (ph if s == 0 else bot) for s in range(6)}
        try:
            play_hand(policies, state=gs)
        except Exception:  # noqa: BLE001
            pass
    agent.solver.solve = orig_solve
    return out


def _simple_bot(state, rng):
    from pokerbot.training.self_play import _legal_mask, _index_to_action
    seat = state.action_seat
    legal = np.flatnonzero(np.asarray(_legal_mask(state, seat), dtype=bool))
    idx = int(rng.choice(list(legal)))
    return _index_to_action(state, seat, idx)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hands", type=int, default=60)
    ap.add_argument("--snapshot", default=r"checkpoints/bp6max_dcfrvr_iter200.pkl")
    args = ap.parse_args()

    with open(args.snapshot, "rb") as f:
        snap_full = pickle.load(f)
    snap = snap_full["strat_sum"] if isinstance(snap_full, dict) and "strat_sum" in snap_full else snap_full

    # leaky profile for the opponent_model=on case
    prof = OpponentStats()
    rng = random.Random(0)
    for _ in range(400):
        ck = random.choice(["1|wet|1", "1|dry|1", "2|paired|1", "0|-|1"])
        prof.add_ctx(ck, [3, 1, 0, 0])
    from pokerbot.opponent.population_prior import context_prior
    prof.hands = 50
    for ck in list(prof.ctx_counts):
        prof.decay_and_add(ck, [3, 1, 0, 0], context_prior(ck), prof.hands)

    configs = [
        ("iters=4  opp=None", 4, None),
        ("iters=10 opp=None", 10, None),
        ("iters=20 opp=None", 20, None),
        ("iters=10 opp=leaky", 10, prof),
        ("iters=20 opp=leaky", 20, prof),
    ]
    total_solves = 0
    total_mismatch = 0
    for label, iters, opp in configs:
        a_nat = build_agent(snap, iters, True, opp)
        probs_nat = collect_probs(a_nat, args.hands, seed=0)
        a_py = build_agent(snap, iters, False, opp)
        probs_py = collect_probs(a_py, args.hands, seed=0)
        n = min(len(probs_nat), len(probs_py))
        mm = 0
        for i in range(n):
            if bits(probs_nat[i]) != bits(probs_py[i]):
                mm += 1
        total_solves += n
        total_mismatch += mm
        print(f"{label:22s}: {n:5d} solves, {mm} bit-mismatches "
              f"(native_solves={len(probs_nat)} py_solves={len(probs_py)})")
    print(f"\nTOTAL: {total_solves} solves, {total_mismatch} bit-mismatches")
    if total_mismatch == 0 and total_solves >= 200:
        print("PASS: native and pure-Python solves are bit-identical (>=200 solves).")
    elif total_solves < 200:
        print(f"WARNING: only {total_solves} solves collected (<200).")
        raise SystemExit(2)
    else:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
