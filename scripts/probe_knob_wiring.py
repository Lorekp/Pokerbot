#!/usr/bin/env python3
"""Direct wiring probe for the leaf_cache_worlds / avg_skip_frac knobs (2026-07-02).

Reaches REAL flop decision states via game flow (init_hand + passive walk — never a hand-built
node), then calls the NATIVE subgame solve at the deployed baseline config with the knob at its
default vs its test value, same public seed. The knob is WIRED iff the served root distributions
differ (L1 > 0); the default arm must equal a second default call bit-for-bit (determinism).
"""
from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import jax

from pokerbot.engine.jax_engine import init_hand
from pokerbot.engine.np_engine import (
    apply_action_np, current_player_np, is_chance_node_np, is_terminal_np,
    legal_action_mask_np, sample_chance_np, to_numpy_state,
)


def reach_flop_state(seed: int):
    """Walk a real hand to the first postflop decision (call/check-down preflop)."""
    occ = jax.numpy.array([True] * 6)
    env = to_numpy_state(init_hand(jax.random.PRNGKey(seed), occupancy=occ))
    for _ in range(200):
        if is_terminal_np(env):
            return None
        if is_chance_node_np(env):
            env = sample_chance_np(env)
            continue
        if int(env.street) > 0:
            seat = current_player_np(env)
            if seat >= 0:
                return env, int(seat)
        seat = current_player_np(env)
        if seat < 0:
            return None
        mask = legal_action_mask_np(env, seat)
        # passive: check if legal else call else fold
        a = 2 if mask[2] else (1 if mask[1] else 0)
        env = apply_action_np(env, int(a))
    return None


def main():
    import pickle
    from pokerbot.search.pluribus_search import PluribusReBeLSolver, PluribusReBeLConfig, set_native_subgame
    from pokerbot.search import native_solver as _ns
    from pokerbot.engine.jax_weva_encode import _POSTFLOP_NUM_BUCKETS

    set_native_subgame(True)
    _ns.enable()
    snap = pickle.load(open("checkpoints/serve_blueprint_6max.pkl", "rb"))
    if isinstance(snap, dict) and "strat_sum" in snap:
        snap = snap["strat_sum"]
    cfg = PluribusReBeLConfig(
        max_iterations=2000, time_budget_ms=2000, two_sided=True, repair_deck=True,
        decouple_sampler=True, range_aware=True, range_uniform_floor=0.25,
        blueprint_mix=1.0,
    )
    sv = PluribusReBeLSolver(snap, cfg)

    def native_probs(env, hero, **kw):
        return _ns.native_subgame_solve(
            env, hero, sv.adapter,
            public_seed=sv._public_seed(env, hero),
            max_iterations=2000, max_depth=48, blueprint_mix=1.0,
            rollout_samples=4, resample_board=True,
            max_playout_depth=64, num_buckets=int(_POSTFLOP_NUM_BUCKETS),
            linear_averaging=True, repair_deck=True, two_sided=True,
            decouple_sampler=True, range_aware=True, range_uniform_floor=0.25,
            ensemble_streams=1, **kw,
        )

    n_states = 0
    l1_leaf, l1_skip = [], []
    for seed in range(100, 160):
        r = reach_flop_state(seed)
        if r is None:
            continue
        env, hero = r
        base = native_probs(env, hero)
        base2 = native_probs(env, hero)
        assert base is not None, "native path unavailable"
        assert np.array_equal(base, base2), "default arm not deterministic!"
        lw4 = native_probs(env, hero, leaf_cache_worlds=4)
        sk3 = native_probs(env, hero, avg_skip_frac=0.3)
        d_lw = float(np.abs(lw4 - base).sum())
        d_sk = float(np.abs(sk3 - base).sum())
        l1_leaf.append(d_lw)
        l1_skip.append(d_sk)
        n_states += 1
        print(f"seed={seed} street={int(env.street)} hero={hero} "
              f"L1(leaf_worlds4)={d_lw:.6f} L1(avg_skip0.3)={d_sk:.6f}", flush=True)
        if n_states >= 8:
            break

    print(f"\nstates={n_states}")
    print(f"leaf_cache_worlds=4: nonzero on {sum(1 for d in l1_leaf if d > 0)}/{n_states}, "
          f"mean L1 {np.mean(l1_leaf):.6f}")
    print(f"avg_skip_frac=0.3 : nonzero on {sum(1 for d in l1_skip if d > 0)}/{n_states}, "
          f"mean L1 {np.mean(l1_skip):.6f}")
    ok = all(d > 0 for d in l1_leaf) and all(d > 0 for d in l1_skip)
    print("WIRING VERDICT:", "BOTH KNOBS CHANGE THE SERVED ROOT DISTRIBUTION" if ok
          else "SOME STATES UNCHANGED (check per-state lines)")


if __name__ == "__main__":
    main()
