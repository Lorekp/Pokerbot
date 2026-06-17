"""Micro-benchmark: per-node legal_action_mask + apply_action, JAX vs pure NumPy.

The NumPy engine (``pokerbot.engine.np_engine``) exists to avoid per-node JAX/XLA eager
dispatch overhead during CPU game-tree traversal.  This script times ~5000 single-node
calls of ``legal_action_mask`` + ``apply_action`` on a fixed mid-hand state and prints the
per-call microseconds and the speedup.  It is a print script, not a test.

Run:  python scripts/bench_np_engine.py
"""

from __future__ import annotations

import time

import jax
import jax.numpy as jnp
import numpy as np

from pokerbot.engine import jax_engine as je
from pokerbot.engine import np_engine as ne

N = 5000


def _mid_hand_state():
    """Build a deterministic mid-hand (flop, one bet outstanding) state to call on.

    We advance a fresh hand a few preflop streets so the node is a realistic interior
    decision node (live actor, non-trivial legal set), then freeze it.
    """
    key = jax.random.PRNGKey(7)
    s = je.init_hand(key)  # 6-max, 100bb
    # Preflop: UTG calls around to BB, BB checks -> close preflop -> deal flop.
    # Drive with CALL/CHECK to reach a flop decision node with someone able to bet.
    for _ in range(40):
        if bool(np.asarray(s.done)):
            break
        if bool(np.asarray(je.is_chance_node(s))):
            s = je.sample_chance(s)
            continue
        if int(np.asarray(s.street)) >= je.STREET_FLOP:
            break
        mask = np.asarray(je.legal_action_mask(s))
        # prefer CALL then CHECK to keep players in for a multiway flop node
        if mask[je.CALL]:
            a = je.CALL
        elif mask[je.CHECK]:
            a = je.CHECK
        else:
            a = int(np.nonzero(mask)[0][0])
        s = je.apply_action(s, jnp.int32(a))
    return s


def bench_jax(state, action):
    # warm up / trigger jit compile (apply_action & legal_action_mask are @jax.jit)
    _ = je.legal_action_mask(state)
    _ = je.apply_action(state, jnp.int32(action))
    jax.block_until_ready(_)
    act = jnp.int32(action)
    t0 = time.perf_counter()
    for _ in range(N):
        m = je.legal_action_mask(state)
        ns = je.apply_action(state, act)
    jax.block_until_ready((m, ns))
    return (time.perf_counter() - t0) / N


def bench_np(state, action):
    # warm up
    _ = ne.legal_action_mask_np(state)
    _ = ne.apply_action_np(state, action)
    t0 = time.perf_counter()
    for _ in range(N):
        m = ne.legal_action_mask_np(state)
        ns = ne.apply_action_np(state, action)
    return (time.perf_counter() - t0) / N


def main() -> None:
    s_jax = _mid_hand_state()
    s_np = ne.to_numpy_state(s_jax)

    mask = np.asarray(je.legal_action_mask(s_jax))
    legal = np.nonzero(mask)[0]
    # pick a representative non-terminal action (a bet bucket if available, else check/call)
    action = int(legal[len(legal) // 2])

    # sanity: states + mask agree before timing
    assert np.array_equal(mask, np.asarray(ne.legal_action_mask_np(s_np)))

    jax_per = bench_jax(s_jax, action)
    np_per = bench_np(s_np, action)

    print(f"state: street={int(np.asarray(s_jax.street))} "
          f"action_seat={int(np.asarray(s_jax.action_seat))} action={action} "
          f"(legal={legal.tolist()})")
    print(f"calls per measurement: {N} (legal_action_mask + apply_action each)")
    print(f"JAX   : {jax_per * 1e6:10.2f} us/call")
    print(f"NumPy : {np_per * 1e6:10.2f} us/call")
    print(f"speedup (JAX/NumPy): {jax_per / np_per:.1f}x")


if __name__ == "__main__":
    main()
