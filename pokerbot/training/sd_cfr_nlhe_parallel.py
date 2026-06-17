"""
On-machine fork-parallel traversal for single-instance SD-CFR (Linux only).

CFR traversals within an iteration are independent, so the shards run in separate PROCESSES
(bypassing the GIL on the deep Python traversal recursion) and their deltas are merged in the
parent. Uses the ``fork`` start method so workers inherit the parent's read-only base tables via
copy-on-write — no pickling of the tables (and after the compact-key fix the returned deltas are
tiny). Reuses the bit-identical-validated ``run_traversal_shard`` / ``merge_traversal_shards``.

CAVEAT (must be validated on the target instance): JAX + fork is historically fragile (XLA thread
pools do not survive fork). Each worker pins itself to a single intra-op thread. If fork-after-JAX
deadlocks on the instance, the FALLBACK is the subprocess-via-ObjectStore path
(`run_iteration_via_store` with independent `python` worker processes, fresh JAX each) — same merge,
no fork. This module raises on non-fork platforms (Windows); the single-threaded path is the
reference and is what the test suite validates.
"""

from __future__ import annotations

import multiprocessing as mp
import os
from typing import Any

from pokerbot.training.sd_cfr_nlhe_distributed import (
    _snapshot_regret_tables,
    copy_trainer_for_shard,
    iteration_traversal_plan,
    merge_traversal_shards,
    run_traversal_shard,
    shard_traversal_plan,
)

# Set in the parent immediately before the pool forks; workers inherit these copy-on-write.
_PARENT_TRAINER: Any = None
_PARENT_BASE: Any = None


def _worker_init() -> None:
    # Pin each worker to one math thread so N workers don't oversubscribe the cores.
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false")


def _run_shard(args):
    shard_id, plan, num_shards, iter_num = args
    pairs = shard_traversal_plan(plan, num_shards, shard_id)
    if not pairs:
        return []
    worker = copy_trainer_for_shard(_PARENT_TRAINER, *_PARENT_BASE)
    worker._begin_iteration_regret()
    return run_traversal_shard(worker, plan, pairs, iter_num)


def run_iteration_parallel(
    trainer,
    iteration_weight: float,
    num_shards: int,
    num_workers: int | None = None,
) -> float:
    """One bulk-sync iteration with shards run in parallel fork processes; deltas merged in-parent.

    Bit-identical to single-threaded (same shards, same global-order merge) — the only difference
    from ``train_iteration_sharded`` is that the shards run concurrently in forked processes.
    """
    global _PARENT_TRAINER, _PARENT_BASE
    if num_workers is None:
        num_workers = min(num_shards, os.cpu_count() or 1)
    iter_num = int(iteration_weight)
    trainer._iter_samples = []
    plan = iteration_traversal_plan(trainer.config, iteration_weight)
    trainer._begin_iteration_regret()
    base_cum, base_strat = _snapshot_regret_tables(trainer)
    _PARENT_TRAINER = trainer
    _PARENT_BASE = (base_cum, base_strat)
    try:
        ctx = mp.get_context("fork")
    except ValueError as e:  # non-fork platform (Windows)
        _PARENT_TRAINER = _PARENT_BASE = None
        raise RuntimeError("fork-parallel traversal requires Linux (fork start method)") from e
    try:
        args = [(sid, plan, num_shards, iter_num) for sid in range(num_shards)]
        with ctx.Pool(processes=num_workers, initializer=_worker_init) as pool:
            shard_results = pool.map(_run_shard, args)
    finally:
        _PARENT_TRAINER = _PARENT_BASE = None
    unit_results = [r for shard in shard_results for r in shard]
    merge_traversal_shards(trainer, unit_results, plan)
    return trainer._finish_iteration_after_traversal()
