"""Persistent worker pool for single-instance SD-CFR — JAX-init-once + dynamic load balancing.

Motivation (measured 2026-06-05): the per-iteration ``run_iteration_subproc`` path is correct and
bit-identical but only ~2-3x because (a) it pays JAX startup PER worker PER iteration and (b) it
splits work into CONTIGUOUS static shards, so one heavy unit (per-unit traversal cost varies ~8x)
bottlenecks its shard while other workers idle.

This pool fixes both:
  * PERSISTENT workers (``spawn`` — fresh JAX after exec, so no JAX-after-fork deadlock) that init
    JAX exactly ONCE and live for the whole run.
  * A shared work QUEUE of individual (sweep, trav) units. Workers pull one unit at a time, so a
    heavy unit only blocks the single worker running it; the others keep draining the queue →
    dynamic load balancing, near-linear with enough units.

Bit-identical to single-threaded: within an iteration the traversal strategy is the FROZEN net (the
"single-deep" part of SD-CFR), so per-unit regret deltas are ORDER-INDEPENDENT; the merge sums each
pair's delta exactly once and concatenates pending in global plan order. Validated against the
single-threaded reference (see tests). ``spawn`` works on Windows too, so this is validated locally.

Per-iteration each worker reloads the (growing) base checkpoint once via an epoch tag carried on the
task — the net params + base tables change each iteration. (A future optimization is to ship only
the merged delta instead of reloading; not needed for correctness.)
"""

from __future__ import annotations

import multiprocessing as mp
import os
import pickle
import queue
import tempfile
import time
import traceback
from pathlib import Path

from pokerbot.training.sd_cfr_nlhe import (
    SDCFRNLHETrainer,
    load_nlhe_checkpoint,
    save_nlhe_checkpoint,
)
from pokerbot.training.sd_cfr_nlhe_cloud import deserialize_shard, serialize_shard
from pokerbot.training.sd_cfr_nlhe_distributed import (
    _snapshot_regret_tables,
    iteration_traversal_plan,
    merge_traversal_shards,
    run_traversal_unit,
)

_SENTINEL = None
_COMPILE_WARM_TIMEOUT = 300.0  # workers 1+ wait at most this long for worker 0's cold XLA compile


def _worker_loop(task_q, result_q, worker_index: int = 0, warm_event=None) -> None:
    """Pull (epoch, base_path, pair, iter_num) units; reload base on epoch change; return deltas.

    ONE trainer per worker, built once and reused (XLA traverse compiled once for the worker's life).
    Each epoch the new base checkpoint is loaded directly into it via ``load_nlhe_checkpoint`` — which
    does ``state.replace(params=...)`` (apply_fn/JIT preserved) and sets the tables in place — so the
    worker keeps a SINGLE copy of the (large, lossless-key) tables. The previous design held a base
    trainer + an accumulation trainer + two snapshot dicts = ~3x the table memory per worker, which
    OOM-killed the box at full depth. Python errors are reported back so the parent fails fast instead
    of blocking forever on a never-arriving result.

    THUNDERING-HERD FIX: N workers building+compiling XLA simultaneously contend badly (60 workers
    stalled >25 min on first iteration). Instead, worker 0 does the ONE cold compile and writes the
    JAX persistent compilation cache (dir set by the parent, inherited via spawn); workers 1+ wait on
    ``warm_event`` until worker 0 finishes its first unit, then build+compile WARM from the cache. This
    serializes the single expensive compile and is bit-identical (only startup ordering changes).
    """
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1")

    loaded_epoch = None
    plan = None
    worker = None
    released_herd = False
    while True:
        item = task_q.get()
        if item is _SENTINEL:
            return
        epoch, base_path, pair, iter_num = item
        try:
            if epoch != loaded_epoch:
                if worker is None:
                    # Gate the cold compile to worker 0; the rest wait for the warm cache.
                    if warm_event is not None and worker_index != 0:
                        warm_event.wait(timeout=_COMPILE_WARM_TIMEOUT)
                    with open(base_path, "rb") as f:
                        cfg = pickle.load(f)["config"]
                    worker = SDCFRNLHETrainer(cfg)   # built ONCE; JIT/apply_fn reused across epochs
                load_nlhe_checkpoint(worker, Path(base_path))  # params + base tables IN PLACE
                plan = iteration_traversal_plan(worker.config, float(iter_num))
                worker._begin_iteration_regret()
                loaded_epoch = epoch
            res = run_traversal_unit(worker, plan, pair[0], pair[1], iter_num)
            if not released_herd and warm_event is not None and worker_index == 0:
                warm_event.set()  # worker 0's cold compile is cached; release workers 1+
                released_herd = True
            result_q.put((pair, serialize_shard([res])))
        except Exception:  # noqa: BLE001 — surface to the parent instead of dying silently
            if warm_event is not None and worker_index == 0 and not warm_event.is_set():
                warm_event.set()  # don't deadlock the herd if worker 0 fails
            result_q.put(("__ERROR__", f"worker unit {pair}:\n{traceback.format_exc()}"))
            return


class PersistentPool:
    """Long-lived spawn workers + dynamic unit queue. Reuse across all iterations; ``close()`` at end."""

    def __init__(self, num_workers: int, *, workdir: str | None = None):
        self.num_workers = max(1, num_workers)
        self._workdir = workdir or tempfile.mkdtemp(prefix="sdcfr_pool_")
        # Enable JAX's persistent compilation cache in a FRESH per-pool dir BEFORE workers spawn (they
        # inherit os.environ). Worker 0 compiles cold + populates it; workers 1+ (gated on warm_event)
        # build warm from it → no N-way simultaneous XLA compile (the thundering-herd stall). Fresh dir
        # = only entries compiled by THIS box's CPU this run, so no cross-machine/stale feature mismatch
        # (which can warn "could lead to SIGILL"); homogeneous within a box.
        cache_dir = os.path.join(self._workdir, "jax_cache")
        os.makedirs(cache_dir, exist_ok=True)
        os.environ["JAX_COMPILATION_CACHE_DIR"] = cache_dir
        os.environ["JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS"] = "0"
        os.environ["JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES"] = "0"
        self.ctx = mp.get_context("spawn")
        self.task_q = self.ctx.Queue()
        self.result_q = self.ctx.Queue()
        self.warm_event = self.ctx.Event()  # set by worker 0 once its cold compile is cached
        self.workers = [
            self.ctx.Process(
                target=_worker_loop,
                args=(self.task_q, self.result_q, idx, self.warm_event),
                daemon=True,
            )
            for idx in range(self.num_workers)
        ]
        for w in self.workers:
            w.start()
        self.epoch = 0

    def run_iteration(self, trainer: SDCFRNLHETrainer, iteration_weight: float) -> float:
        """One bulk-sync iteration via the persistent pool; bit-identical to single-threaded."""
        iter_num = int(iteration_weight)
        trainer._iter_samples = []
        plan = iteration_traversal_plan(trainer.config, iteration_weight)
        trainer._begin_iteration_regret()
        _snapshot_regret_tables(trainer)  # parity with the reference phase (parent base untouched)

        base_path = str(Path(self._workdir) / f"base_iter{iter_num}.pkl")
        # SLIM base: workers only read cum_regret; strat_sum/reservoir are write-only and merged from
        # deltas, so omitting them keeps the merge bit-identical while shrinking the per-iteration
        # base each of the N workers must deserialize (~4x smaller at depth 16 — the dominant overhead).
        save_nlhe_checkpoint(trainer, Path(base_path), worker_base=True)

        self.epoch += 1
        n = len(plan.pairs)
        for pair in plan.pairs:
            self.task_q.put((self.epoch, base_path, pair, iter_num))

        # Collect n unit results, but never block forever: if a worker dies (OOM SIGKILL, which can't
        # be caught in-process) its result never arrives, so on a get() timeout we check liveness and
        # fail fast — the run exits, logs upload, and the coordinator respawns from the last checkpoint
        # (vs. the old behaviour: a silent ~50-minute hang on result_q.get()).
        unit_results = []
        got = 0
        while got < n:
            try:
                tag, blob = self.result_q.get(timeout=120)
            except queue.Empty:
                alive = sum(1 for w in self.workers if w.is_alive())
                if alive < self.num_workers:
                    raise RuntimeError(
                        f"pool worker(s) died mid-iteration ({alive}/{self.num_workers} alive) with "
                        f"{n - got}/{n} units outstanding — almost certainly OOM (check console for "
                        "'Out of memory: Killed'); reduce --workers or use a larger-RAM instance."
                    )
                continue  # all workers alive, just a slow unit — keep waiting
            if tag == "__ERROR__":
                raise RuntimeError(f"pool worker raised:\n{blob}")
            unit_results.extend(deserialize_shard(blob))
            got += 1

        merge_traversal_shards(trainer, unit_results, plan)
        loss = trainer._finish_iteration_after_traversal()
        # best-effort cleanup of the per-iter base file (workers have already loaded it)
        try:
            os.remove(base_path)
        except OSError:
            pass
        return loss

    def close(self) -> None:
        for _ in self.workers:
            self.task_q.put(_SENTINEL)
        for w in self.workers:
            w.join(timeout=30)
            if w.is_alive():
                w.terminate()


def run_convergence_pool(
    trainer: SDCFRNLHETrainer,
    iterations: int,
    num_workers: int,
    *,
    on_iter=None,
    start_iter: int = 0,
) -> list[float]:
    """Convenience: run ``iterations`` iterations through one persistent pool (closes at the end)."""
    import numpy as np

    pool = PersistentPool(num_workers)
    losses: list[float] = []
    try:
        for n in range(start_iter + 1, iterations + 1):
            np.random.seed(trainer.config.seed + n - 1)
            t0 = time.perf_counter()
            loss = pool.run_iteration(trainer, float(n))
            trainer._completed_iterations = n
            losses.append(loss)
            if on_iter is not None:
                on_iter(n, loss, time.perf_counter() - t0)
    finally:
        pool.close()
    return losses
