#!/usr/bin/env python3
"""Tier-A bit-identity cross-check: native (Rust) engine vs pure-Python np_engine.

Generates >=5000 golden cases by playing random hands end-to-end (all streets) plus targeted edge
cases (all-ins, multiway side pots, 2-6 live players, showdown vs fold-win) and asserts EXACT equality
of: legal masks, apply_action next-states (every EnvState field), sample_chance deals, terminal rewards,
and standalone evaluate_seven / compute_returns. Equality is np.array_equal (bit-exact for ints/bools).

Usage: POKERBOT_BETTING_ABSTRACTION=1 python tools/battery_native/crosscheck.py [--hands 1200]
"""
from __future__ import annotations

import argparse
import random

import numpy as np

import jax

from pokerbot.engine.jax_engine import init_hand
from pokerbot.engine.np_engine import (
    to_numpy_state,
    apply_action_np as apply_py,
    legal_action_mask_np as mask_py,
    sample_chance_np as chance_py,
    is_chance_node_np as is_chance_py,
    evaluate_seven as eval7_py,
    compute_returns_np as returns_py,
)
from pokerbot.search import native_solver as ns

assert ns.available(), "native extension not available — build it first"


def eq_state(a, b):
    for f in a._fields:
        if not np.array_equal(np.asarray(getattr(a, f)), np.asarray(getattr(b, f))):
            return f
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hands", type=int, default=1200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    n_cases = 0
    n_mask = 0
    n_chance = 0
    n_terminal = 0
    mismatches = []

    for hi in range(args.hands):
        key = jax.random.PRNGKey(rng.randint(0, 10**9))
        button = rng.randint(0, 5)
        jstate = init_hand(key, button=button)
        env = to_numpy_state(jstate)
        depth = 0
        while not bool(env.done) and depth < 300:
            if is_chance_py(env):
                # also cross-check is_chance flag
                if ns.is_chance_node_np(env) != True:
                    mismatches.append((hi, "is_chance flag"))
                e_n = ns.sample_chance_np(env)
                e_p = chance_py(env)
                f = eq_state(e_n, e_p)
                if f is not None:
                    mismatches.append((hi, f"sample_chance:{f}"))
                env = e_p
                n_chance += 1
                n_cases += 1
                continue
            seat = int(env.action_seat)
            if seat < 0:
                break
            m_n = ns.legal_action_mask_np(env, seat)
            m_p = mask_py(env, seat)
            if not np.array_equal(m_n, m_p):
                mismatches.append((hi, "legal_mask"))
            n_mask += 1
            legal = np.flatnonzero(m_p)
            if legal.size == 0:
                break
            # Bias toward call/check (idx 1,2) so hands reach the flop/turn/river deals and showdowns,
            # exercising sample_chance + the side-pot/showdown returns far more than uniform play (which
            # mostly folds out preflop). ~70% passive, 30% uniform-over-legal.
            passive = [a for a in legal.tolist() if a in (1, 2)]
            if passive and rng.random() < 0.70:
                a = int(rng.choice(passive))
            else:
                a = int(rng.choice(list(legal)))
            e_n = ns.apply_action_np(env, a)
            e_p = apply_py(env, a)
            f = eq_state(e_n, e_p)
            if f is not None:
                mismatches.append((hi, f"apply_action:{f}"))
            if bool(e_p.done):
                # terminal reward cross-check (already covered by eq_state rewards field, but count)
                n_terminal += 1
            env = e_p
            depth += 1
            n_cases += 1

    # Standalone evaluate_seven fuzz (random 7-card sets) — also pins the showdown scorer directly.
    n_eval = 0
    for _ in range(20000):
        cards = rng.sample(range(52), 7)
        arr = np.asarray(cards, dtype=np.int32)
        if int(ns.evaluate_seven(arr)) != int(eval7_py(arr)):
            mismatches.append(("eval7", tuple(cards)))
        n_eval += 1

    # Standalone compute_returns fuzz: random committed/status/board/holes (side-pot edge cases).
    n_ret = 0
    for _ in range(5000):
        cards = rng.sample(range(52), 17)  # 6*2 holes + 5 board
        holes = np.asarray(cards[:12], dtype=np.int32).reshape(6, 2)
        board = np.asarray(cards[12:17], dtype=np.int32)
        # random statuses with >=2 eligible some of the time
        status = np.asarray([rng.choice([1, 2, 3]) for _ in range(6)], dtype=np.int8)
        committed = np.asarray([rng.choice([0, 50, 100, 200, 300, 500]) for _ in range(6)], dtype=np.int32)
        button = rng.randint(0, 5)
        n_showdown = int(((status == 2) | (status == 3)).sum())
        r_p = returns_py(holes=holes, board=board, seat_status=status,
                         committed_total=committed, button=button, n_showdown=n_showdown)
        h_flat = [int(x) for x in holes.reshape(-1)]
        r_n = ns._EXT.compute_returns(h_flat, [int(x) for x in board],
                                      [int(x) for x in status], [int(x) for x in committed],
                                      button, n_showdown)
        if not np.array_equal(np.asarray(r_n, dtype=np.int32), r_p):
            mismatches.append(("compute_returns", "mismatch"))
        n_ret += 1

    print(f"hands={args.hands}")
    print(f"transition cases (apply+chance): {n_cases}")
    print(f"legal_mask cases:                {n_mask}")
    print(f"sample_chance cases:             {n_chance}")
    print(f"terminal (reward) cases:         {n_terminal}")
    print(f"evaluate_seven fuzz:             {n_eval}")
    print(f"compute_returns fuzz:            {n_ret}")
    total = n_cases + n_mask + n_eval + n_ret
    print(f"TOTAL golden cases:              {total}")
    print(f"MISMATCHES:                      {len(mismatches)}")
    if mismatches:
        for m in mismatches[:20]:
            print("  ", m)
        raise SystemExit(1)
    print("ALL BIT-IDENTICAL (Tier A pass)")


if __name__ == "__main__":
    main()
