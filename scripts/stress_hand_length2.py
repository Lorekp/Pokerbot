"""Worst-case hand-length stress: never-fold + RANDOM raise sizes (maximizes re-raise wars).

The deterministic min-bucket policy collapses to one line. The true worst case keeps everyone in
(never fold when another action is legal) but VARIES raise sizes randomly so re-raise wars of every
shape are explored. Max over millions of hands -> a safe empirical ceiling for the scan bound.
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


def make_counter(env, horizon):
    def count_one(reset_key, pol_key):
        state = env.reset(reset_key)
        init = (state, pol_key, jnp.logical_not(env.done(state)))

        def body(carry, _):
            state, key, active = carry
            step_now = jnp.logical_and(active, jnp.logical_not(env.done(state)))

            def do_step(args):
                state, key = args
                mask = env.legal_mask(state)
                # never fold if any non-fold action legal
                nonfold = mask.at[FOLD].set(False)
                eff = jnp.where(jnp.any(nonfold), nonfold, mask)
                n = jnp.maximum(jnp.sum(eff.astype(jnp.float32)), 1.0)
                probs = eff.astype(jnp.float32) / n
                key, ka = jax.random.split(key)
                action = jax.random.choice(ka, env.num_actions, p=probs)
                s2 = jax.lax.cond(env.is_chance(state), env.sample_chance, lambda s: s, state)
                s2 = env.step(s2, action)
                return (s2, key)

            new_state, new_key = jax.lax.cond(step_now, do_step, lambda a: a, (state, key))
            return (new_state, new_key, step_now), step_now.astype(jnp.int32)

        _, flags = jax.lax.scan(body, init, None, length=horizon)
        return jnp.sum(flags)

    return jax.jit(jax.vmap(count_one))


def measure(env, n_hands, seed, horizon=PROBE_HORIZON):
    counter = make_counter(env, horizon)
    rng = jax.random.PRNGKey(seed)
    keys = jax.random.split(rng, n_hands * 2)
    counts = np.asarray(jax.block_until_ready(counter(keys[:n_hands], keys[n_hands:])))
    return int(counts.max()), int((counts >= horizon).sum()), float(counts.mean())


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 50000
    rounds = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    env = nlhe_env_spec()
    overall = 0
    for sd in range(rounds):
        mx, hit, mean = measure(env, n, seed=2000 + sd)
        overall = max(overall, mx)
        print(f"  seed {sd}: max={mx} mean={mean:.2f} horizon-hits={hit}")
    print(f"OVERALL WORST-CASE MAX (never-fold + random sizes, {rounds*n} hands): {overall}")
