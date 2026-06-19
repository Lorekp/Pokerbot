"""Stress the max hand length with a LENGTH-MAXIMIZING policy (NLHE), not uniform.

To set a safe scan bound we need the WORST-CASE betting depth, not the average. A length-maximizing
adversary keeps the hand alive and raises by the SMALLEST legal increment (most re-raises). We bias
the policy hard toward the smallest bet-bucket / call / check (never fold when it can continue), and
take the max over many hands and seeds. This bounds the realizable tree depth far above what a
uniform or trained policy reaches.
"""

from __future__ import annotations

import sys

import jax
import jax.numpy as jnp
import numpy as np

from pokerbot.engine import jax_engine as je
from pokerbot.training.cfr_env import nlhe_env_spec

PROBE_HORIZON = 600
FOLD = je.FOLD
CHECK = je.CHECK
CALL = je.CALL
BET_START = je.BET_BUCKET_START


def make_counter(env, horizon):
    def count_one(reset_key):
        state = env.reset(reset_key)
        init = (state, jnp.logical_not(env.done(state)))

        def body(carry, _):
            state, active = carry
            step_now = jnp.logical_and(active, jnp.logical_not(env.done(state)))

            def do_step(state):
                mask = env.legal_mask(state)
                # length-maximizing preference: smallest bet bucket > call > check > all-in > fold.
                # Build a priority score; pick the highest-priority LEGAL action.
                pri = jnp.full((env.num_actions,), -1.0)
                # smallest raise bucket (index BET_START) highest, then call, then check
                pri = pri.at[BET_START].set(10.0)        # min pot-fraction raise = most re-raises
                pri = pri.at[BET_START + 1].set(9.0)
                pri = pri.at[CALL].set(5.0)
                pri = pri.at[CHECK].set(4.0)
                pri = pri.at[FOLD].set(-5.0)             # never fold if anything else legal
                scored = jnp.where(mask, pri, -1e9)
                action = jnp.argmax(scored)
                s2 = jax.lax.cond(env.is_chance(state), env.sample_chance, lambda s: s, state)
                s2 = env.step(s2, action)
                return s2

            new_state = jax.lax.cond(step_now, do_step, lambda s: s, state)
            return (new_state, step_now), step_now.astype(jnp.int32)

        _, flags = jax.lax.scan(body, init, None, length=horizon)
        return jnp.sum(flags)

    return jax.jit(jax.vmap(count_one))


def measure(label, env, n_hands, seed, horizon=PROBE_HORIZON):
    counter = make_counter(env, horizon)
    rng = jax.random.PRNGKey(seed)
    keys = jax.random.split(rng, n_hands)
    counts = np.asarray(jax.block_until_ready(counter(keys)))
    print(f"--- {label}: {n_hands} hands (length-maximizing policy, horizon={horizon}) ---")
    print(f"  max={counts.max()}  mean={counts.mean():.2f}  p99={np.percentile(counts,99):.0f}")
    print(f"  hands hitting horizon: {int((counts>=horizon).sum())}")
    print()
    return int(counts.max())


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 20000
    mx = 0
    for sd in range(4):
        mx = max(mx, measure("NLHE 6-max", nlhe_env_spec(), n, seed=100 + sd))
    print(f"OVERALL MAX (length-maximizing, {4*n} hands): {mx}")
