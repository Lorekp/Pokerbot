"""Profiling harness for VR-DeepPDCFR+ training throughput.

Times one NLHE training iteration broken down by phase:
  (a) run_traversals device scan (_batched_scans)
  (b) _insert_traversal_batch (host transfer + buffer inserts)
  (c) the grad-step methods

Usage: python scripts/profile_neural_cfr.py
"""

from __future__ import annotations

import time

import jax
import jax.numpy as jnp

from pokerbot.engine.jax_engine import NUM_PLAYERS
from pokerbot.training.neural_cfr import VRDeepPDCFRConfig, VRDeepPDCFRPlusTrainer


def _sync(x):
    jax.block_until_ready(x)


def profile_run_traversals(trainer: VRDeepPDCFRPlusTrainer, iteration: int) -> dict[str, float]:
    """Break run_traversals into device-scan vs host-insert per traverser."""
    cfg = trainer.config
    from pokerbot.training.replay_buffers import clear_all_advantage_buffers

    trainer.adv_buffers = clear_all_advantage_buffers(trainer.adv_buffers)
    cum_params = tuple(s.params for s in trainer.cum_states)
    inst_params = tuple(s.params for s in trainer.inst_states)
    hv_params = trainer.hv_state.params
    iter_arr = jnp.int32(iteration)
    k = cfg.traversals_per_iter

    scan_t = 0.0
    insert_t = 0.0
    for traverser in range(NUM_PLAYERS):
        trainer.key, key_block = jax.random.split(trainer.key)
        keys = jax.random.split(key_block, k * 2)
        reset_keys = keys[:k]
        trav_keys = keys[k:]

        t0 = time.perf_counter()
        adv, strat, hist, is_trav, active = trainer._batched_scans(
            reset_keys, trav_keys, jnp.int32(traverser), iter_arr,
            cum_params, inst_params, hv_params, cfg.alpha, cfg.epsilon,
        )
        _sync((adv, strat, hist, is_trav, active))
        t1 = time.perf_counter()
        trainer._insert_traversal_batch(traverser, adv, strat, hist, is_trav, active)
        t2 = time.perf_counter()
        scan_t += t1 - t0
        insert_t += t2 - t1
    return {"scan": scan_t, "insert": insert_t}


def profile_grad_steps(trainer: VRDeepPDCFRPlusTrainer, iteration: int) -> float:
    t0 = time.perf_counter()
    for player in range(NUM_PLAYERS):
        trainer._train_player_cumulative(player, iteration)
        trainer._train_player_instantaneous(player)
    trainer._train_history_value(iteration)
    trainer._train_average_strategy(iteration)
    # force completion
    _sync(tuple(s.params for s in trainer.cum_states))
    _sync(tuple(s.params for s in trainer.inst_states))
    _sync(trainer.hv_state.params)
    _sync(trainer.avg_state.params)
    trainer._prev_cum_params = tuple(s.params for s in trainer.cum_states)
    t1 = time.perf_counter()
    return t1 - t0


def run(traversals: int, n_warmup: int, n_measure: int) -> None:
    cfg = VRDeepPDCFRConfig(
        total_iterations=100,
        traversals_per_iter=traversals,
        batch_size=512,
        hidden_dim=256,
        buffer_capacity=100_000,
        seed=0,
    )
    trainer = VRDeepPDCFRPlusTrainer(cfg)

    print(f"=== NLHE  traversals_per_iter={traversals}  batch_size={cfg.batch_size} ===")
    # Warmup (jit compile)
    for it in range(1, n_warmup + 1):
        profile_run_traversals(trainer, it)
        profile_grad_steps(trainer, it)

    scan_sum = insert_sum = grad_sum = total_sum = 0.0
    for it in range(n_warmup + 1, n_warmup + n_measure + 1):
        t0 = time.perf_counter()
        ph = profile_run_traversals(trainer, it)
        g = profile_grad_steps(trainer, it)
        t1 = time.perf_counter()
        scan_sum += ph["scan"]
        insert_sum += ph["insert"]
        grad_sum += g
        total_sum += t1 - t0

    n = n_measure
    scan_ms = scan_sum / n * 1e3
    insert_ms = insert_sum / n * 1e3
    grad_ms = grad_sum / n * 1e3
    total_ms = total_sum / n * 1e3
    print(f"  (a) run_traversals device scan : {scan_ms:9.2f} ms/iter  ({scan_ms/total_ms*100:5.1f}%)")
    print(f"  (b) _insert_traversal_batch    : {insert_ms:9.2f} ms/iter  ({insert_ms/total_ms*100:5.1f}%)")
    print(f"  (c) grad-step methods          : {grad_ms:9.2f} ms/iter  ({grad_ms/total_ms*100:5.1f}%)")
    print(f"  TOTAL                          : {total_ms:9.2f} ms/iter   ({1000.0/total_ms:6.3f} iters/sec)")
    print()


if __name__ == "__main__":
    run(traversals=64, n_warmup=2, n_measure=5)
