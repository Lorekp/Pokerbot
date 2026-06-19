"""Probe candidate #2: fuse the 6 per-traverser _batched_scans dispatches into ONE vmap over
(traverser x traversals). Same total FLOPs; the question is whether one 384-way dispatch beats six
64-way dispatches on this backend (CPU here; on an accelerator the fused form exposes more
parallelism). Diagnostic only -- compares wall time of 6 sequential dispatches vs 1 fused dispatch.
"""

from __future__ import annotations

import time

import jax
import jax.numpy as jnp

from pokerbot.training.cfr_env import nlhe_env_spec
from pokerbot.training.neural_cfr import (
    VRDeepPDCFRConfig,
    VRDeepPDCFRPlusTrainer,
    _batched_traversal_scans,
)


def _sync(x):
    jax.block_until_ready(x)


def main():
    env = nlhe_env_spec()
    cfg = VRDeepPDCFRConfig(hidden_dim=256, traversals_per_iter=64, batch_size=512)
    tr = VRDeepPDCFRPlusTrainer(cfg)
    cum_p = tuple(s.params for s in tr.cum_states)
    inst_p = tuple(s.params for s in tr.inst_states)
    hv_p = tr.hv_state.params
    k = 64
    NP = env.num_players
    rng = jax.random.PRNGKey(0)

    # --- current: 6 sequential dispatches ---
    seq = tr._batched_scans

    def run_seq():
        nonlocal rng
        outs = []
        for trav in range(NP):
            rng, kb = jax.random.split(rng)
            keys = jax.random.split(kb, k * 2)
            outs.append(seq(keys[:k], keys[k:], jnp.int32(trav), jnp.int32(1),
                            cum_p, inst_p, hv_p, cfg.alpha, cfg.epsilon,
                            cfg.reward_scale, cfg.hidden_dim, env))
        return outs

    # --- fused: one vmap over (traverser x traversals) ---
    def fused_fn(reset_keys, trav_keys, travs, iteration, cum, inst, hv,
                 alpha, epsilon, rs, hd, e):
        # reset_keys/trav_keys: (NP, k, 2); travs: (NP,)
        return jax.vmap(
            lambda rk, tk, tv: _batched_traversal_scans(
                rk, tk, tv, iteration, cum, inst, hv, alpha, epsilon, rs, hd, e)
        )(reset_keys, trav_keys, travs)

    fused = jax.jit(fused_fn, static_argnames=("alpha", "epsilon", "rs", "hd", "e"))

    def run_fused():
        nonlocal rng
        rng, kb = jax.random.split(rng)
        keys = jax.random.split(kb, NP * k * 2).reshape(NP, k * 2, 2)
        reset_keys = keys[:, :k, :]
        trav_keys = keys[:, k:, :]
        travs = jnp.arange(NP, dtype=jnp.int32)
        return fused(reset_keys, trav_keys, travs, jnp.int32(1),
                     cum_p, inst_p, hv_p, cfg.alpha, cfg.epsilon,
                     cfg.reward_scale, cfg.hidden_dim, env)

    for _ in range(2):
        _sync(run_seq()); _sync(run_fused())
    n = 3
    t0 = time.perf_counter()
    for _ in range(n):
        _sync(run_seq())
    seq_ms = (time.perf_counter() - t0) / n * 1e3
    t0 = time.perf_counter()
    for _ in range(n):
        _sync(run_fused())
    fused_ms = (time.perf_counter() - t0) / n * 1e3
    print(f"  6 sequential dispatches : {seq_ms:8.1f} ms/iter")
    print(f"  1 fused (NPxK) dispatch : {fused_ms:8.1f} ms/iter")
    print(f"  speedup                 : {seq_ms/fused_ms:6.2f}x")


if __name__ == "__main__":
    main()
