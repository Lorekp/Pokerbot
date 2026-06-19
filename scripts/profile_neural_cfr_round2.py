"""Round-2 profiling harness for VR-DeepPDCFR+ training.

Times one full training iteration for NLHE and/or Leduc, broken into:
  (a) device traversal  (run_traversals: _batched_scans, summed over traversers)
  (b) buffer ingestion  (_insert_traversal_batch)
  (c) grad steps        (all four network updates)

Run via:  python -m scripts.profile_neural_cfr_round2 [nlhe|leduc|both] [traversals]

Warmup-excluded, device-synced. Local execution is JAX-on-CPU.
"""

from __future__ import annotations

import sys
import time

import jax
import jax.numpy as jnp

from pokerbot.training.cfr_env import leduc_env_spec, nlhe_env_spec
from pokerbot.training.leduc_gate import LEDUC_REWARD_SCALE
from pokerbot.training.neural_cfr import VRDeepPDCFRConfig, VRDeepPDCFRPlusTrainer
from pokerbot.training.replay_buffers import clear_all_advantage_buffers


def _sync(x):
    jax.block_until_ready(x)


def _run_one_iter(trainer, iteration):
    """One iteration broken into (scan, insert, grad) wall-clock, device-synced."""
    cfg = trainer.config
    env = trainer.env
    trainer.adv_buffers = clear_all_advantage_buffers(trainer.adv_buffers)
    cum_params = tuple(s.params for s in trainer.cum_states)
    inst_params = tuple(s.params for s in trainer.inst_states)
    hv_params = trainer.hv_state.params
    iter_arr = jnp.int32(iteration)
    k = cfg.traversals_per_iter

    scan_t = insert_t = 0.0
    for traverser in range(env.num_players):
        trainer.key, key_block = jax.random.split(trainer.key)
        keys = jax.random.split(key_block, k * 2)
        t0 = time.perf_counter()
        out = trainer._batched_scans(
            keys[:k], keys[k:], jnp.int32(traverser), iter_arr,
            cum_params, inst_params, hv_params, cfg.alpha, cfg.epsilon,
            cfg.reward_scale, cfg.hidden_dim, env,
        )
        _sync(out)
        t1 = time.perf_counter()
        adv, strat, hist, is_trav, active = out
        trainer._insert_traversal_batch(traverser, adv, strat, hist, is_trav, active)
        _sync((trainer.adv_buffers, trainer.strat_buffer, trainer.hist_buffer))
        t2 = time.perf_counter()
        scan_t += t1 - t0
        insert_t += t2 - t1

    t0 = time.perf_counter()
    steps = max(1, cfg.train_steps_per_iter)
    for player in range(env.num_players):
        for _ in range(steps):
            trainer._train_player_cumulative(player, iteration)
            trainer._train_player_instantaneous(player)
    for _ in range(steps):
        trainer._train_history_value(iteration)
        trainer._train_average_strategy(iteration)
    _sync(tuple(s.params for s in trainer.cum_states))
    _sync(tuple(s.params for s in trainer.inst_states))
    _sync(trainer.hv_state.params)
    _sync(trainer.avg_state.params)
    trainer._prev_cum_params = tuple(s.params for s in trainer.cum_states)
    grad_t = time.perf_counter() - t0
    return scan_t, insert_t, grad_t


def measure(label, env, reward_scale, hidden, traversals, batch_size, warmup, measure_n,
            train_steps=1):
    cfg = VRDeepPDCFRConfig(
        total_iterations=100, traversals_per_iter=traversals, batch_size=batch_size,
        hidden_dim=hidden, buffer_capacity=100_000, seed=0, reward_scale=reward_scale,
        train_steps_per_iter=train_steps,
    )
    trainer = VRDeepPDCFRPlusTrainer(cfg, env=env)
    for it in range(1, warmup + 1):
        _run_one_iter(trainer, it)
    s = i = g = 0.0
    for it in range(warmup + 1, warmup + measure_n + 1):
        ds, di, dg = _run_one_iter(trainer, it)
        s += ds; i += di; g += dg
    n = measure_n
    scan_ms = s / n * 1e3; ins_ms = i / n * 1e3; grad_ms = g / n * 1e3
    total = scan_ms + ins_ms + grad_ms
    print(f"--- {label}  (traversals={traversals}, hidden={hidden}, bs={batch_size}) ---")
    print(f"  (a) device traversal   : {scan_ms:10.2f} ms/iter  ({scan_ms/total*100:5.1f}%)")
    print(f"  (b) buffer ingestion   : {ins_ms:10.2f} ms/iter  ({ins_ms/total*100:5.1f}%)")
    print(f"  (c) grad steps         : {grad_ms:10.2f} ms/iter  ({grad_ms/total*100:5.1f}%)")
    print(f"  TOTAL                  : {total:10.2f} ms/iter   ({1000.0/total:6.3f} iters/sec)")
    print()
    return total


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "both"
    trav = int(sys.argv[2]) if len(sys.argv) > 2 else 64
    if mode in ("nlhe", "both"):
        measure("NLHE", nlhe_env_spec(), float(__import__("pokerbot.engine.jax_engine",
                fromlist=["STARTING_STACK_CHIPS"]).STARTING_STACK_CHIPS),
                hidden=256, traversals=trav, batch_size=512, warmup=2, measure_n=3)
    if mode in ("leduc", "both"):
        measure("Leduc", leduc_env_spec(), LEDUC_REWARD_SCALE,
                hidden=64, traversals=trav, batch_size=256, warmup=2, measure_n=3)
