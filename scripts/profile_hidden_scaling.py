"""Attribute NLHE device-traversal time to nets vs env by scaling hidden_dim.

Net-apply FLOPs scale ~hidden_dim^2; env step + encoding are independent of hidden_dim. Timing the
REAL _batched_scans (whose outputs depend on the nets, so nothing is DCE'd) across hidden in
{16,64,256} brackets the split: the hidden-invariant floor ~ env+encode, the rest ~ nets.
"""

from __future__ import annotations

import time

import jax
import jax.numpy as jnp

from pokerbot.training.cfr_env import nlhe_env_spec
from pokerbot.training.neural_cfr import VRDeepPDCFRConfig, VRDeepPDCFRPlusTrainer


def _sync(x):
    jax.block_until_ready(x)


def measure(hidden, k=64):
    env = nlhe_env_spec()
    cfg = VRDeepPDCFRConfig(hidden_dim=hidden, traversals_per_iter=k, batch_size=512)
    tr = VRDeepPDCFRPlusTrainer(cfg)
    cum_p = tuple(s.params for s in tr.cum_states)
    inst_p = tuple(s.params for s in tr.inst_states)
    hv_p = tr.hv_state.params
    rng = jax.random.PRNGKey(0)

    def one_dispatch():
        nonlocal rng
        rng, kb = jax.random.split(rng)
        keys = jax.random.split(kb, k * 2)
        return tr._batched_scans(keys[:k], keys[k:], jnp.int32(0), jnp.int32(1),
                                 cum_p, inst_p, hv_p, cfg.alpha, cfg.epsilon,
                                 cfg.reward_scale, cfg.hidden_dim, env)

    for _ in range(2):
        _sync(one_dispatch())
    n = 4
    t0 = time.perf_counter()
    for _ in range(n):
        _sync(one_dispatch())
    dt = (time.perf_counter() - t0) / n * 1e3
    print(f"  hidden={hidden:4d}: {dt:8.2f} ms/dispatch  (x6 ~ {dt*6:8.1f} ms/iter)")
    return dt


if __name__ == "__main__":
    print("NLHE _batched_scans device-traversal vs hidden_dim (K=64, horizon=64):")
    measure(16)
    measure(64)
    measure(256)
