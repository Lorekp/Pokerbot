"""Measure the ACTUAL maximum number of decision/transition steps per hand (vectorized).

The traversal scan uses a fixed MAX_TRAVERSAL_STEPS=128 horizon and masks out steps past
terminal. If real hands are far shorter, the bound can be tightened (statistically safe iff
NO real hand is ever truncated). This counts steps the way _scan_one_traversal does: a body step
runs while `active AND not done(state)`; we sum that flag over a long jitted+vmapped scan, so the
count is exact and fast (no per-step host dispatch).
"""

from __future__ import annotations

import sys

import jax
import jax.numpy as jnp
import numpy as np

from pokerbot.training.cfr_env import leduc_env_spec, nlhe_env_spec

PROBE_HORIZON = 512  # well above 128 to detect truncation


def make_counter(env, horizon):
    def count_one(reset_key, trav_key):
        state = env.reset(reset_key)
        init = (state, trav_key, jnp.logical_not(env.done(state)))

        def body(carry, _):
            state, key, active = carry
            step_now = jnp.logical_and(active, jnp.logical_not(env.done(state)))

            def do_step(args):
                state, key = args
                mask = env.legal_mask(state)
                n_legal = jnp.maximum(jnp.sum(mask.astype(jnp.float32)), 1.0)
                probs = mask.astype(jnp.float32) / n_legal
                key, ka = jax.random.split(key)
                action = jax.random.choice(ka, env.num_actions, p=probs)
                s2 = jax.lax.cond(env.is_chance(state), env.sample_chance, lambda s: s, state)
                s2 = env.step(s2, action)
                return (s2, key)

            new_state, new_key = jax.lax.cond(
                step_now, do_step, lambda args: args, (state, key))
            return (new_state, new_key, step_now), step_now.astype(jnp.int32)

        _, flags = jax.lax.scan(body, init, None, length=horizon)
        return jnp.sum(flags)

    return jax.jit(jax.vmap(count_one))


def measure(label, env, n_hands, seed=0, horizon=PROBE_HORIZON):
    counter = make_counter(env, horizon)
    rng = jax.random.PRNGKey(seed)
    rng, kb = jax.random.split(rng)
    keys = jax.random.split(kb, n_hands * 2)
    counts = np.asarray(jax.block_until_ready(counter(keys[:n_hands], keys[n_hands:])))
    print(f"--- {label}: {n_hands} hands (probe horizon={horizon}) ---")
    print(f"  max={counts.max()}  mean={counts.mean():.2f}  p50={np.percentile(counts,50):.0f}  "
          f"p99={np.percentile(counts,99):.0f}  p999={np.percentile(counts,99.9):.0f}")
    n_at_horizon = int((counts >= horizon).sum())
    print(f"  hands hitting probe horizon ({horizon}): {n_at_horizon}")
    print(f"  current MAX_TRAVERSAL_STEPS=128  ->  headroom = {128 - int(counts.max())} steps")
    print()
    return int(counts.max())


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 20000
    measure("NLHE 6-max", nlhe_env_spec(), n, seed=1)
    measure("Leduc", leduc_env_spec(), n, seed=2)
