"""Smoke test for external bot adapters.

For each of the 4 PyPokerEngine bots, runs ~200 random-legal-play hands,
asserting at every decision node:
  - output length == 11
  - output sums to ~1.0 (within 1e-9)
  - ZERO mass on illegal actions (per legal_action_mask_np)
  - no exceptions (caught by safe_policy, but we check _failure_count)
  - determinism: same env twice -> identical vector

Also prints per-bot decision latency and fold/call/raise frequency summary.

Run:
    python -m pokerbot.eval.external_bots.smoke_external
"""
from __future__ import annotations

import os
import time
from collections import Counter

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np

from pokerbot.engine.jax_engine import init_hand
from pokerbot.engine.np_engine import (
    SEAT_LIVE,
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

N_HANDS = 200         # hands per bot
SEED_BASE = 31337
BOT_NAMES = ["fold_man", "random", "fish", "honest"]

# Action index labels for frequency reporting
_ACTION_LABELS = ["fold", "check", "call"] + [f"bet{i}" for i in range(7)] + ["allin"]
_FOLD, _CHECK, _CALL, _ALLIN = 0, 1, 2, 10


def _run_random_hand(env, policy_fn, hero_seat: int, rng: np.random.Generator) -> list[np.ndarray]:
    """Run one hand: hero_seat uses policy_fn, other seats play random-legal.

    Returns list of (env_snapshot, probs) pairs for every decision the hero made.
    """
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
            a = int(np.argmax(mask))
            env = apply_action_np(env, a)
            continue
        if seat == hero_seat:
            probs = policy_fn(env, seat)
            decisions.append((env, probs, mask))
            # Sample action for game continuation
            pc = np.clip(np.asarray(probs, dtype=np.float64), 0, None)
            s = pc.sum()
            legal_f = mask.astype(np.float64)
            pc = pc / s if s > 0 else legal_f / legal_f.sum()
            a = int(rng.choice(len(pc), p=pc))
        else:
            # Random legal play for other seats
            legal_f = mask.astype(np.float64)
            pc = legal_f / legal_f.sum()
            a = int(rng.choice(len(pc), p=pc))
        env = apply_action_np(env, a)
    return decisions


def _classify_action(probs: np.ndarray) -> str:
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


def smoke_one_bot(name: str, policy_fn) -> None:
    occ = jnp.array([True] * 6)
    rng = np.random.default_rng(SEED_BASE)
    freq: Counter = Counter()
    n_decisions = 0
    n_det_checks = 0
    n_det_mismatches = 0
    n_illegal_mass = 0
    n_wrong_sum = 0
    n_wrong_len = 0
    latencies = []

    _common._failure_count = 0  # reset failure counter

    print(f"\n--- {name} ---")
    for hand_idx in range(N_HANDS):
        hero_seat = hand_idx % 6
        env = to_numpy_state(init_hand(jax.random.PRNGKey(SEED_BASE + hand_idx * 997), occupancy=occ))
        decisions = _run_random_hand(env, policy_fn, hero_seat, rng)

        for snap_env, probs, mask in decisions:
            n_decisions += 1
            t0 = time.perf_counter()
            probs2 = policy_fn(snap_env, hero_seat)
            latencies.append(time.perf_counter() - t0)

            # Length check
            if len(probs) != 11:
                n_wrong_len += 1
                print(f"  FAIL [{name}] wrong length {len(probs)} (hand {hand_idx})")

            # Sum check
            s = float(np.asarray(probs).sum())
            if abs(s - 1.0) > 1e-9:
                n_wrong_sum += 1
                print(f"  FAIL [{name}] probs sum={s:.8f} != 1 (hand {hand_idx})")

            # Illegal mass check
            illegal_mass = float(np.asarray(probs)[~mask].sum())
            if illegal_mass > 1e-9:
                n_illegal_mass += 1
                print(f"  FAIL [{name}] illegal mass={illegal_mass:.6f} "
                      f"mask={mask.astype(int).tolist()} probs={np.round(probs, 3).tolist()} "
                      f"(hand {hand_idx})")

            # Determinism check (second call with same env)
            p2 = np.asarray(probs2, dtype=np.float64)
            p1 = np.asarray(probs, dtype=np.float64)
            n_det_checks += 1
            if not np.allclose(p1, p2, atol=1e-12):
                n_det_mismatches += 1
                print(f"  FAIL [{name}] non-deterministic (hand {hand_idx}): "
                      f"diff={np.abs(p1 - p2).max():.3e}")

            freq[_classify_action(probs)] += 1

    # Summary
    total = max(n_decisions, 1)
    avg_lat = float(np.mean(latencies)) * 1000 if latencies else 0.0
    print(f"  decisions={n_decisions}  failures={_common._failure_count}  "
          f"avg_latency={avg_lat:.2f}ms")
    print(f"  det_mismatches={n_det_mismatches}/{n_det_checks}  "
          f"illegal_mass_violations={n_illegal_mass}  "
          f"wrong_sum={n_wrong_sum}  wrong_len={n_wrong_len}")
    print(f"  action freq: " + "  ".join(
        f"{k}={v}/{total}={100*v/total:.1f}%" for k, v in sorted(freq.items(), key=lambda x: -x[1])
    ))

    # Assertions
    assert _common._failure_count == 0, f"[{name}] {_common._failure_count} safe_policy failures"
    assert n_wrong_len == 0, f"[{name}] {n_wrong_len} wrong-length outputs"
    assert n_wrong_sum == 0, f"[{name}] {n_wrong_sum} probs-don't-sum-to-1 violations"
    assert n_illegal_mass == 0, f"[{name}] {n_illegal_mass} illegal-mass violations"
    assert n_det_mismatches == 0, f"[{name}] {n_det_mismatches} determinism failures"
    print(f"  PASS")


def main() -> None:
    print(f"=== external_bots smoke test  N_HANDS={N_HANDS} per bot ===")
    t0 = time.perf_counter()
    for name in BOT_NAMES:
        policy_fn = get_opponent_policy(name)
        smoke_one_bot(name, policy_fn)
    elapsed = time.perf_counter() - t0
    print(f"\n=== ALL PASS  ({elapsed:.1f}s total) ===")


if __name__ == "__main__":
    main()
