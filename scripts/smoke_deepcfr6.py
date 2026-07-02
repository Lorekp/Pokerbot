#!/usr/bin/env python3
"""Smoke test for the DeepCFR-6max adapter (deepcfr6).

Checks:
  1. Forward-pass determinism + legality assertions over ~200 random game states.
  2. Correctness: output length=11, sums to 1.0, zero mass on illegal indices.
  3. Determinism: same (env, seat) -> identical vector.
  4. No safe_policy failures (exception counter stays 0).
  5. Action-frequency summary (random weights -> should be mixed).

Run:
    python -u scripts/smoke_deepcfr6.py 2>&1 | tee artifacts/smoke_deepcfr6.log

Expected output: PASS — plumbing verified (random-init weights, untrained output).
"""
from __future__ import annotations

import os
import time
from collections import Counter

# Must be set before any pokerbot import
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("POKERBOT_BETTING_ABSTRACTION", "1")
os.environ.setdefault("POKERBOT_POSTFLOP_BUCKETS", "1")
os.environ.setdefault("POKERBOT_PREFLOP_LOSSLESS", "1")
os.environ.setdefault("POKERBOT_POSTFLOP_MODE", "fast")
os.environ.setdefault("POKERBOT_POSTFLOP_NUM_BUCKETS", "200")

import jax
import jax.numpy as jnp
import numpy as np

from pokerbot.engine.jax_engine import init_hand
from pokerbot.engine.np_engine import (
    apply_action_np,
    current_player_np,
    is_chance_node_np,
    is_terminal_np,
    legal_action_mask_np,
    sample_chance_np,
    to_numpy_state,
)
import pokerbot.eval.external_bots.common as _common
from pokerbot.eval.external_bots.registry import get_opponent_policy

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
N_HANDS = 200
SEED_BASE = 9999
BOT_NAME = "deepcfr6"

_FOLD, _CHECK, _CALL, _ALLIN = 0, 1, 2, 10


def _classify(probs: np.ndarray) -> str:
    idx = int(np.argmax(probs))
    if idx == _FOLD:
        return "fold"
    if idx == _CHECK:
        return "check"
    if idx == _CALL:
        return "call"
    if idx == _ALLIN:
        return "allin"
    return "raise_bet"


def _run_hand(env, policy_fn, hero_seat: int, rng: np.random.Generator):
    """Run one hand. Returns list of (snap_env, probs, mask) for hero decisions."""
    decisions = []
    guard = 0
    while not is_terminal_np(env) and guard < 400:
        guard += 1
        if is_chance_node_np(env):
            env = sample_chance_np(env)
            continue
        seat = current_player_np(env)
        if seat < 0:
            break
        mask = legal_action_mask_np(env, seat)
        if int(mask.sum()) <= 1:
            env = apply_action_np(env, int(np.argmax(mask)))
            continue
        if seat == hero_seat:
            probs = policy_fn(env, seat)
            decisions.append((env, probs, mask))
            pc = np.clip(np.asarray(probs, dtype=np.float64), 0, None)
            s = pc.sum()
            if s <= 0:
                mf = mask.astype(np.float64)
                pc = mf / mf.sum()
            else:
                pc = pc / s
            a = int(rng.choice(len(pc), p=pc))
        else:
            mf = mask.astype(np.float64)
            a = int(rng.choice(len(mf), p=mf / mf.sum()))
        env = apply_action_np(env, a)
    return decisions


def main() -> None:
    print(f"=== smoke_deepcfr6  N_HANDS={N_HANDS}  bot={BOT_NAME} ===", flush=True)
    t0 = time.perf_counter()

    _common._failure_count = 0
    policy_fn = get_opponent_policy(BOT_NAME)
    print(f"Loaded: {BOT_NAME}", flush=True)

    occ = jnp.array([True] * 6)
    rng = np.random.default_rng(SEED_BASE)
    freq: Counter = Counter()
    n_decisions = 0
    n_det_mismatches = 0
    n_illegal_mass = 0
    n_wrong_sum = 0
    n_wrong_len = 0
    latencies = []

    for hand_idx in range(N_HANDS):
        hero_seat = hand_idx % 6
        env = to_numpy_state(
            init_hand(jax.random.PRNGKey(SEED_BASE + hand_idx * 997), occupancy=occ)
        )
        decisions = _run_hand(env, policy_fn, hero_seat, rng)

        for snap_env, probs, mask in decisions:
            n_decisions += 1

            # Length check
            if len(probs) != 11:
                n_wrong_len += 1
                print(f"  FAIL wrong length {len(probs)} (hand {hand_idx})", flush=True)

            # Sum check
            s = float(np.asarray(probs).sum())
            if abs(s - 1.0) > 1e-9:
                n_wrong_sum += 1
                print(f"  FAIL probs sum={s:.8f} (hand {hand_idx})", flush=True)

            # Illegal mass check
            illegal = float(np.asarray(probs)[~mask].sum())
            if illegal > 1e-9:
                n_illegal_mass += 1
                print(
                    f"  FAIL illegal mass={illegal:.6f}  "
                    f"mask={mask.astype(int).tolist()}  "
                    f"probs={np.round(probs, 3).tolist()}  (hand {hand_idx})",
                    flush=True,
                )

            # Determinism: call twice, compare
            t1 = time.perf_counter()
            probs2 = policy_fn(snap_env, hero_seat)
            latencies.append(time.perf_counter() - t1)

            p1 = np.asarray(probs, dtype=np.float64)
            p2 = np.asarray(probs2, dtype=np.float64)
            if not np.allclose(p1, p2, atol=1e-12):
                n_det_mismatches += 1
                print(
                    f"  FAIL non-deterministic (hand {hand_idx}) "
                    f"max_diff={np.abs(p1 - p2).max():.3e}",
                    flush=True,
                )

            freq[_classify(probs)] += 1

        if (hand_idx + 1) % 50 == 0:
            el = time.perf_counter() - t0
            eta = el / (hand_idx + 1) * (N_HANDS - hand_idx - 1)
            print(
                f"  hand {hand_idx + 1}/{N_HANDS}  decisions_so_far={n_decisions}  "
                f"elapsed={el:.1f}s  ETA={eta:.1f}s",
                flush=True,
            )

    # Summary
    el = time.perf_counter() - t0
    avg_lat = float(np.mean(latencies)) * 1000 if latencies else 0.0
    total = max(n_decisions, 1)

    print(f"\n--- Results ---", flush=True)
    print(f"  decisions       : {n_decisions}", flush=True)
    print(f"  safe_failures   : {_common._failure_count}", flush=True)
    print(f"  avg_latency     : {avg_lat:.2f}ms", flush=True)
    print(f"  det_mismatches  : {n_det_mismatches}", flush=True)
    print(f"  illegal_mass    : {n_illegal_mass}", flush=True)
    print(f"  wrong_sum       : {n_wrong_sum}", flush=True)
    print(f"  wrong_len       : {n_wrong_len}", flush=True)
    print(
        f"  action_freq     : "
        + "  ".join(
            f"{k}={v}/{total}={100*v/total:.1f}%"
            for k, v in sorted(freq.items(), key=lambda x: -x[1])
        ),
        flush=True,
    )
    print(f"  wall            : {el:.1f}s", flush=True)

    # Assertions
    assert _common._failure_count == 0, \
        f"safe_policy failures: {_common._failure_count}"
    assert n_wrong_len == 0, f"wrong-length outputs: {n_wrong_len}"
    assert n_wrong_sum == 0, f"probs-don't-sum-to-1: {n_wrong_sum}"
    assert n_illegal_mass == 0, f"illegal-mass violations: {n_illegal_mass}"
    assert n_det_mismatches == 0, f"determinism failures: {n_det_mismatches}"

    print(f"\n=== PASS  ({el:.1f}s total) ===", flush=True)


if __name__ == "__main__":
    main()
