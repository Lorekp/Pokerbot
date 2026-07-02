#!/usr/bin/env python3
"""Wiring + latency probe for the R2 leaf_biased_continuations knob (2026-07-02).

Reaches REAL flop decision states via game flow (init_hand + passive walk — never a hand-built
node, which nondeterministically hangs the native solver), then calls the NATIVE subgame solve at
the deployed baseline config with the knob OFF vs ON, same public seed.

Checks:
  1. determinism: the default arm twice -> bit-identical; the ON arm twice -> bit-identical.
  2. wiring: OFF vs ON served root distributions differ (L1 > 0) on flop states.
  3. mult wiring: ON@5 vs ON@10 differ (leaf_bias_mult is live when the bool is on).
  4. latency: per-solve wall time OFF vs ON at K=2000 over ~10 flop solves -> ratio.
"""
from __future__ import annotations

import os
import time

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
    l1_on, l1_mult = [], []
    t_off, t_on = [], []
    for seed in range(100, 200):
        r = reach_flop_state(seed)
        if r is None:
            continue
        env, hero = r
        t0 = time.perf_counter()
        base = native_probs(env, hero)
        t_off.append(time.perf_counter() - t0)
        base2 = native_probs(env, hero)
        assert base is not None, "native path unavailable"
        assert np.array_equal(base, base2), "default arm not deterministic!"
        t0 = time.perf_counter()
        on5 = native_probs(env, hero, leaf_biased_continuations=True)
        t_on.append(time.perf_counter() - t0)
        on5b = native_probs(env, hero, leaf_biased_continuations=True)
        assert np.array_equal(on5, on5b), "biased arm not deterministic!"
        on10 = native_probs(env, hero, leaf_biased_continuations=True, leaf_bias_mult=10.0)
        d_on = float(np.abs(on5 - base).sum())
        d_mult = float(np.abs(on10 - on5).sum())
        l1_on.append(d_on)
        l1_mult.append(d_mult)
        n_states += 1
        print(f"seed={seed} street={int(env.street)} hero={hero} "
              f"L1(on5 vs off)={d_on:.6f} L1(on10 vs on5)={d_mult:.6f} "
              f"t_off={t_off[-1]:.2f}s t_on={t_on[-1]:.2f}s", flush=True)
        if n_states >= 10:
            break

    print(f"\nstates={n_states}")
    print(f"leaf_biased ON@5 vs OFF : nonzero on {sum(1 for d in l1_on if d > 0)}/{n_states}, "
          f"mean L1 {np.mean(l1_on):.6f}")
    print(f"leaf_bias_mult 10 vs 5  : nonzero on {sum(1 for d in l1_mult if d > 0)}/{n_states}, "
          f"mean L1 {np.mean(l1_mult):.6f}")
    ratio = float(np.sum(t_on) / max(np.sum(t_off), 1e-9))
    print(f"latency: OFF total {np.sum(t_off):.2f}s (mean {np.mean(t_off):.2f}s) | "
          f"ON total {np.sum(t_on):.2f}s (mean {np.mean(t_on):.2f}s) | ratio ON/OFF = {ratio:.2f}x")
    ok = all(d > 0 for d in l1_on) and all(d > 0 for d in l1_mult)
    print("WIRING VERDICT:", "KNOB + MULT BOTH CHANGE THE SERVED ROOT DISTRIBUTION" if ok
          else "SOME STATES UNCHANGED (check per-state lines)")


if __name__ == "__main__":
    main()
