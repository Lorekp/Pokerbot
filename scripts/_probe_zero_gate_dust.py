#!/usr/bin/env python3
"""Forensic probe: replicate engine_ab_h2h game loop for ONE deck with identical arms,
dump per-rotation per-seat rewards BITWISE (hex of float64) and the deck_a/deck_b
accumulation, to localize the zero-gate float dust (rot0 vs rot1).

Usage: python scripts/_probe_zero_gate_dust.py --seed 7 --deck 2
(deck index is 0-based; deck 3/8 in the run log = index 2)
"""
from __future__ import annotations
import argparse
import os
import struct
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import jax

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def fhex(x: float) -> str:
    return struct.pack(">d", float(x)).hex()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--deck", type=int, default=2)
    ap.add_argument("--settle-samples", type=int, default=200)
    args = ap.parse_args()

    # Mirror engine_ab_h2h imports / setup exactly.
    from pokerbot.engine.jax_engine import init_hand
    from pokerbot.engine.np_engine import (
        apply_action_np, current_player_np, is_chance_node_np, is_terminal_np,
        legal_action_mask_np, sample_chance_np, to_numpy_state,
    )
    from pokerbot.search.pluribus_search import set_native_subgame
    from pokerbot.search import native_solver as _ns
    import pickle

    if not _ns.native_subgame_available():
        _ns.enable()

    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "engine_ab_h2h", os.path.join(os.path.dirname(os.path.abspath(__file__)), "engine_ab_h2h.py"))
    mod = importlib.util.module_from_spec(spec)
    # engine_ab_h2h has a main() guard; safe to exec for _build_policy/_ev_settle/_nboard.
    spec.loader.exec_module(mod)

    base_seed = args.seed
    d = args.deck
    settle_samples = args.settle_samples
    repo_root = os.getcwd()

    bp_path = f"{repo_root}/checkpoints/serve_blueprint_6max.pkl"
    raw = pickle.load(open(bp_path, "rb"))
    snap = raw["strat_sum"] if isinstance(raw, dict) and "strat_sum" in raw else raw

    pf_a = mod._build_policy({}, snap, base_seed, repo_root)
    pf_b = mod._build_policy({}, snap, base_seed, repo_root)

    occ = jax.numpy.array([True] * 6)

    def act(pf, env, seat, salt):
        set_native_subgame(True)
        p = np.asarray(pf(env, seat), dtype=np.float64)
        pc = np.clip(p, 0, None)
        s = pc.sum()
        if s <= 0:
            m = legal_action_mask_np(env, seat).astype(np.float64)
            pc = m / m.sum()
        else:
            pc = pc / s
        return int(np.random.default_rng(salt).choice(len(pc), p=pc)), pc

    rot_rewards = {}
    deck_a = 0.0
    deck_b = 0.0
    for rot in (0, 1):
        is_a = {s: ((s % 2 == 0) == (rot == 0)) for s in range(6)}
        env = mod.to_numpy_state if False else None
        from pokerbot.engine.np_engine import to_numpy_state as _tns
        env = _tns(init_hand(jax.random.PRNGKey(base_seed * 100003 + d), occupancy=occ))
        prev_env, prev_action, prev_nboard = None, None, 0
        g = 0
        actions = []
        while not is_terminal_np(env) and g < 400:
            g += 1
            if is_chance_node_np(env):
                env = sample_chance_np(env)
                continue
            seat = current_player_np(env)
            if seat < 0:
                break
            mask = legal_action_mask_np(env, seat).astype(bool)
            if int(mask.sum()) <= 1:
                a = int(np.argmax(mask))
                pvec = None
            else:
                salt = base_seed * 7 + d * 131 + g
                pf = pf_a if is_a[int(seat)] else pf_b
                a, pvec = act(pf, env, int(seat), salt)
            actions.append((g, int(seat), a, None if pvec is None else [fhex(x) for x in pvec]))
            prev_env, prev_action, prev_nboard = env, a, mod._nboard(env)
            env = apply_action_np(env, a)

        settle_rng = np.random.default_rng(base_seed * 991 + d * 61 + g)
        settled = False
        if settle_samples > 0 and prev_env is not None and mod._nboard(env) > prev_nboard:
            settled = True
            rew = mod._ev_settle(
                prev_env, prev_action, settle_rng, settle_samples,
                apply_action_np, is_terminal_np, is_chance_node_np, sample_chance_np,
            )
        else:
            rew = np.asarray(env.rewards, dtype=np.float64)

        rot_rewards[rot] = np.asarray(rew, dtype=np.float64).copy()
        print(f"\n--- rot {rot}: steps g={g} settled={settled}")
        for (gg, seat, a, pvec) in actions:
            print(f"  step g={gg} seat={seat} action={a}" + (f" pvec={pvec}" if pvec else ""))
        for s in range(6):
            print(f"  rew[{s}] = {rew[s]!r}  hex={fhex(rew[s])}  arm={'A' if is_a[s] else 'B'}")
        for s in range(6):
            if is_a[s]:
                deck_a += float(rew[s])
            else:
                deck_b += float(rew[s])
        print(f"  running deck_a={deck_a!r} hex={fhex(deck_a)}")
        print(f"  running deck_b={deck_b!r} hex={fhex(deck_b)}")

    r0, r1 = rot_rewards[0], rot_rewards[1]
    print("\n=== rot0 vs rot1 reward bitwise comparison ===")
    identical = True
    for s in range(6):
        same = fhex(r0[s]) == fhex(r1[s])
        identical &= same
        print(f"  seat {s}: rot0={fhex(r0[s])} rot1={fhex(r1[s])} {'SAME' if same else 'DIFF'}")
    print(f"rewards bitwise identical across rotations: {identical}")
    net = deck_a - deck_b
    print(f"deck_a={deck_a!r} deck_b={deck_b!r} net={net!r}")

    # Reference: rotation-symmetric accumulation (per-rotation partial sums, seat order)
    sa0 = float(r0[0]) + float(r0[2]) + float(r0[4])
    sb0 = float(r0[1]) + float(r0[3]) + float(r0[5])
    sa1 = float(r1[1]) + float(r1[3]) + float(r1[5])
    sb1 = float(r1[0]) + float(r1[2]) + float(r1[4])
    da = sa0 + sa1
    db = sb0 + sb1
    print(f"symmetric-accum net = {da - db!r}")


if __name__ == "__main__":
    main()
