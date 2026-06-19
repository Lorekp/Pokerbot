"""Before/after comparison harness for the buffer-ingestion optimization.

Measures one NLHE training iteration broken into:
  (a) run_traversals device scan (_batched_scans)
  (b) buffer ingestion  (OLD host double-loop  vs  NEW vectorized batch ops)
  (c) grad-step methods

Run via:  python -m scripts.profile_neural_cfr_compare
"""

from __future__ import annotations

import time

import jax
import jax.numpy as jnp

from pokerbot.engine.jax_engine import NUM_PLAYERS
from pokerbot.training.neural_cfr import VRDeepPDCFRConfig, VRDeepPDCFRPlusTrainer
from pokerbot.training.replay_buffers import (
    AdvantageBuffers,
    HistoryValueBuffer,
    StrategyBuffer,
    clear_all_advantage_buffers,
    insert_advantage_sample,
    insert_history_value_sample,
    insert_strategy_sample,
)


def _sync(x):
    jax.block_until_ready(x)


def _old_insert(trainer, traverser, adv, strat, hist, is_trav, active):
    """Verbatim copy of the original host double-loop ingestion."""
    adv_np = jax.device_get(adv)
    strat_np = jax.device_get(strat)
    hist_np = jax.device_get(hist)
    is_trav_np = jax.device_get(is_trav)
    active_np = jax.device_get(active)

    n_batch = int(adv_np["info_state"].shape[0])
    n_steps = int(adv_np["info_state"].shape[1])
    for b in range(n_batch):
        for t in range(n_steps):
            if not active_np[b, t]:
                continue
            if is_trav_np[b, t]:
                buf = trainer.adv_buffers.buffers[traverser]
                trainer.adv_buffers = AdvantageBuffers(
                    buffers=trainer.adv_buffers.buffers[:traverser]
                    + (
                        insert_advantage_sample(
                            buf,
                            info_state=jnp.asarray(adv_np["info_state"][b, t]),
                            advantages=jnp.asarray(adv_np["advantages"][b, t]),
                            legal_mask=jnp.asarray(adv_np["legal_mask"][b, t]),
                            iteration=jnp.asarray(adv_np["iteration"][b, t]),
                        ),
                    )
                    + trainer.adv_buffers.buffers[traverser + 1 :]
                )
            else:
                trainer.key, ks = jax.random.split(trainer.key)
                trainer.strat_buffer = StrategyBuffer(
                    buffer=insert_strategy_sample(
                        trainer.strat_buffer.buffer,
                        ks,
                        info_state=jnp.asarray(strat_np["info_state"][b, t]),
                        strategy=jnp.asarray(strat_np["strategy"][b, t]),
                        legal_mask=jnp.asarray(strat_np["legal_mask"][b, t]),
                        iteration=jnp.asarray(strat_np["iteration"][b, t]),
                    )
                )
            trainer.key, kh = jax.random.split(trainer.key)
            trainer.hist_buffer = HistoryValueBuffer(
                buffer=insert_history_value_sample(
                    trainer.hist_buffer.buffer,
                    kh,
                    history=jnp.asarray(hist_np["history"][b, t]),
                    next_history=jnp.asarray(hist_np["next_history"][b, t]),
                    action=jnp.asarray(hist_np["action"][b, t]),
                    utility=jnp.asarray(hist_np["utility"][b, t]),
                    player=jnp.asarray(hist_np["player"][b, t]),
                    iteration=jnp.asarray(hist_np["iteration"][b, t]),
                    next_info_state=jnp.asarray(hist_np["next_info_state"][b, t]),
                    next_legal_mask=jnp.asarray(hist_np["next_legal_mask"][b, t]),
                )
            )


def _run_one_iter(trainer, iteration, insert_fn):
    cfg = trainer.config
    trainer.adv_buffers = clear_all_advantage_buffers(trainer.adv_buffers)
    cum_params = tuple(s.params for s in trainer.cum_states)
    inst_params = tuple(s.params for s in trainer.inst_states)
    hv_params = trainer.hv_state.params
    iter_arr = jnp.int32(iteration)
    k = cfg.traversals_per_iter

    scan_t = insert_t = 0.0
    for traverser in range(NUM_PLAYERS):
        trainer.key, key_block = jax.random.split(trainer.key)
        keys = jax.random.split(key_block, k * 2)
        t0 = time.perf_counter()
        out = trainer._batched_scans(
            keys[:k], keys[k:], jnp.int32(traverser), iter_arr,
            cum_params, inst_params, hv_params, cfg.alpha, cfg.epsilon,
        )
        _sync(out)
        t1 = time.perf_counter()
        adv, strat, hist, is_trav, active = out
        insert_fn(trainer, traverser, adv, strat, hist, is_trav, active)
        _sync((trainer.adv_buffers, trainer.strat_buffer, trainer.hist_buffer))
        t2 = time.perf_counter()
        scan_t += t1 - t0
        insert_t += t2 - t1

    # grad steps
    t0 = time.perf_counter()
    for player in range(NUM_PLAYERS):
        trainer._train_player_cumulative(player, iteration)
        trainer._train_player_instantaneous(player)
    trainer._train_history_value(iteration)
    trainer._train_average_strategy(iteration)
    _sync(tuple(s.params for s in trainer.cum_states))
    _sync(trainer.hv_state.params)
    _sync(trainer.avg_state.params)
    trainer._prev_cum_params = tuple(s.params for s in trainer.cum_states)
    grad_t = time.perf_counter() - t0
    return scan_t, insert_t, grad_t


def _new_insert(trainer, traverser, adv, strat, hist, is_trav, active):
    trainer._insert_traversal_batch(traverser, adv, strat, hist, is_trav, active)


def measure(label, insert_fn, traversals, warmup, measure_n):
    cfg = VRDeepPDCFRConfig(
        total_iterations=100, traversals_per_iter=traversals,
        batch_size=512, hidden_dim=256, buffer_capacity=100_000, seed=0,
    )
    trainer = VRDeepPDCFRPlusTrainer(cfg)
    for it in range(1, warmup + 1):
        _run_one_iter(trainer, it, insert_fn)
    s = i = g = 0.0
    for it in range(warmup + 1, warmup + measure_n + 1):
        ds, di, dg = _run_one_iter(trainer, it, insert_fn)
        s += ds; i += di; g += dg
    n = measure_n
    scan_ms = s / n * 1e3; ins_ms = i / n * 1e3; grad_ms = g / n * 1e3
    total = scan_ms + ins_ms + grad_ms
    print(f"--- {label}  (traversals={traversals}) ---")
    print(f"  (a) device scan        : {scan_ms:10.2f} ms/iter  ({scan_ms/total*100:5.1f}%)")
    print(f"  (b) buffer ingestion   : {ins_ms:10.2f} ms/iter  ({ins_ms/total*100:5.1f}%)")
    print(f"  (c) grad steps         : {grad_ms:10.2f} ms/iter  ({grad_ms/total*100:5.1f}%)")
    print(f"  TOTAL                  : {total:10.2f} ms/iter   ({1000.0/total:6.3f} iters/sec)")
    print()
    return total


if __name__ == "__main__":
    import sys

    mode = sys.argv[1] if len(sys.argv) > 1 else "both"
    trav = int(sys.argv[2]) if len(sys.argv) > 2 else 64
    if mode in ("old", "both"):
        measure("OLD host double-loop", _old_insert, trav, warmup=1, measure_n=2)
    if mode in ("new", "both"):
        measure("NEW vectorized batch", _new_insert, trav, warmup=2, measure_n=3)
