"""Sub-breakdown of the device-traversal scan body for NLHE.

Builds ablated variants of one full _batched_scans dispatch by zeroing out classes of work, so we
can attribute the ~5s device-traversal time to: net applies vs env step/transition vs encoding.

Each variant is a self-contained jitted vmap-over-traversals scan, timed warmup-excluded + synced.
We measure the FULL body, then bodies with (i) net applies removed, (ii) only env.step in the body,
to bracket how much is matmuls vs env. This is diagnostic only -- not wired into the trainer.
"""

from __future__ import annotations

import time

import jax
import jax.numpy as jnp

from pokerbot.training.cfr_env import nlhe_env_spec
from pokerbot.training.neural_cfr import (
    CumulativeAdvantageNet,
    HistoryValueNet,
    InstantaneousAdvantageNet,
    VRDeepPDCFRConfig,
    VRDeepPDCFRPlusTrainer,
    _stack_player_params,
    pdcfr_predicted_regrets,
    regret_matching,
)

ALPHA = 2.3
EPS = 0.6
HORIZON = 64
K = 64


def _sync(x):
    jax.block_until_ready(x)


def build(env, hidden, cum_p, inst_p, hv_p, variant):
    cum_mod = CumulativeAdvantageNet(hidden, env.num_actions)
    inst_mod = InstantaneousAdvantageNet(hidden, env.num_actions)
    hv_mod = HistoryValueNet(hidden, env.num_actions)
    cum_s = _stack_player_params(cum_p)
    inst_s = _stack_player_params(inst_p)

    def gather(stacked, seat):
        return jax.tree_util.tree_map(lambda x: x[seat], stacked)

    def one(reset_key, trav_key):
        state = env.reset(reset_key)
        init = (state, trav_key, jnp.logical_not(env.done(state)))

        def body(carry, _):
            state, key, active = carry

            def do_step(args):
                state, key = args
                seat = env.action_seat(state)
                mask = env.legal_mask(state)
                info = env.encode_info(state, seat)
                hist = env.encode_history(state)
                if variant in ("full", "nets_only"):
                    cum = cum_mod.apply(gather(cum_s, seat), info, mask)
                    inst = inst_mod.apply(gather(inst_s, seat), info, mask)
                    policy = regret_matching(pdcfr_predicted_regrets(cum, inst, jnp.int32(1), ALPHA), mask)
                else:
                    policy = mask.astype(jnp.float32) / jnp.maximum(jnp.sum(mask.astype(jnp.float32)), 1.0)
                n = jnp.maximum(jnp.sum(mask.astype(jnp.float32)), 1.0)
                xi = policy if variant == "nets_only" else mask.astype(jnp.float32) / n
                key, ka = jax.random.split(key)
                action = jax.random.choice(ka, env.num_actions, p=mask.astype(jnp.float32) / n)
                s2 = jax.lax.cond(env.is_chance(state), env.sample_chance, lambda s: s, state)
                s2 = env.step(s2, action)
                if variant == "full":
                    nseat = env.action_seat(s2)
                    nmask = env.legal_mask(s2)
                    ninfo = env.encode_info(s2, nseat)
                    nhist = env.encode_history(s2)
                    ncum = cum_mod.apply(gather(cum_s, nseat), ninfo, nmask)
                    ninst = inst_mod.apply(gather(inst_s, nseat), ninfo, nmask)
                    npol = regret_matching(pdcfr_predicted_regrets(ncum, ninst, jnp.int32(1), ALPHA), nmask)
                    qn = hv_mod.apply(hv_p, nhist)
                    q = hv_mod.apply(hv_p, hist)
                    _ = jnp.sum(npol * qn) + jnp.sum(q)
                return (s2, key)

            ns, nk = jax.lax.cond(active, do_step, lambda a: a, (state, key))
            step_now = jnp.logical_and(active, jnp.logical_not(env.done(state)))
            return (ns, nk, step_now), step_now.astype(jnp.int32)

        _, flags = jax.lax.scan(body, init, None, length=HORIZON)
        return jnp.sum(flags)

    return jax.jit(jax.vmap(one))


def time_variant(label, fn, keys):
    for _ in range(2):
        _sync(fn(keys[:K], keys[K:]))
    n = 3
    t0 = time.perf_counter()
    for _ in range(n):
        _sync(fn(keys[:K], keys[K:]))
    dt = (time.perf_counter() - t0) / n * 1e3
    # x6 traversers to compare against the per-iter device-traversal number
    print(f"  {label:24s}: {dt:8.2f} ms/dispatch   (x6 traversers ~ {dt*6:8.1f} ms/iter)")
    return dt


if __name__ == "__main__":
    env = nlhe_env_spec()
    cfg = VRDeepPDCFRConfig(hidden_dim=256, traversals_per_iter=K)
    tr = VRDeepPDCFRPlusTrainer(cfg)
    cum_p = tuple(s.params for s in tr.cum_states)
    inst_p = tuple(s.params for s in tr.inst_states)
    hv_p = tr.hv_state.params
    rng = jax.random.PRNGKey(0)
    keys = jax.random.split(rng, K * 2)
    print("NLHE scan-body ablation (hidden=256, K=64, horizon=64):")
    f_full = build(env, 256, cum_p, inst_p, hv_p, "full")
    f_nets = build(env, 256, cum_p, inst_p, hv_p, "nets_only")
    f_env = build(env, 256, cum_p, inst_p, hv_p, "env_only")
    time_variant("full body", f_full, keys)
    time_variant("nets(cur)+env, no next", f_nets, keys)
    time_variant("env step+encode only", f_env, keys)
