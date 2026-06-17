"""Subprocess-parallel traversal for single-instance SD-CFR (cross-platform; the fork replacement).

WHY THIS EXISTS: ``sd_cfr_nlhe_parallel.run_iteration_parallel`` uses ``fork`` so workers inherit the
parent's tables copy-on-write. That path DEADLOCKS on the EC2 arm64 instance — JAX/XLA initialises
its runtime threads in the parent (net construction), and those threads do not survive ``fork``, so
the first JAX op in a forked child hangs forever. Empirically confirmed (2026-06-05): a single-thread
run finished both iterations while a ``workers=2`` fork run hung indefinitely at the pool-creation
point, same depth/traversals.

THE FIX: each worker is an INDEPENDENT subprocess (``python -m pokerbot.training.sd_cfr_nlhe_subproc``)
that initialises JAX FRESH after the OS ``exec`` — no inherited XLA state, so no deadlock. The parent
writes the per-iteration base state once (the validated ``save_nlhe_checkpoint``); each worker loads
it, runs its assigned shards via the SAME ``copy_trainer_for_shard`` / ``run_traversal_shard`` /
``merge_traversal_shards`` primitives the single-threaded reference uses, and writes its serialized
shard results back. The parent gathers + merges in global plan order.

BIT-IDENTICAL to single-threaded: the per-unit PRNG keys come from
``traversal_prng_keys(trainer, iteration_num, sweep_i, trav_i)`` (process-independent), and the merge
is order-deterministic + idempotent. So this is a drop-in replacement for the fork path with the same
numerical result — only the executor differs. Works on Windows (local validation) and Linux (cloud).
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from pokerbot.training.sd_cfr_nlhe import (
    SDCFRNLHETrainer,
    load_nlhe_checkpoint,
    save_nlhe_checkpoint,
)
from pokerbot.training.sd_cfr_nlhe_cloud import deserialize_shard, serialize_shard
from pokerbot.training.sd_cfr_nlhe_distributed import (
    _snapshot_regret_tables,
    copy_trainer_for_shard,
    iteration_traversal_plan,
    merge_traversal_shards,
    run_traversal_shard,
    shard_traversal_plan,
)


def _worker_env() -> dict:
    env = dict(os.environ)
    # Each worker pins itself to one math thread so N workers don't oversubscribe the cores, and
    # forces CPU JAX. (We are the one launching them, so set rather than setdefault.)
    env["JAX_PLATFORMS"] = "cpu"
    env["OMP_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
    return env


def _run_worker(base_path: str, iteration_num: int, num_shards: int, shard_ids: list[int],
                out_path: str) -> None:
    """Worker body: load the base state, run assigned shards, write serialized results."""
    weight = float(iteration_num)
    # Build a trainer from the checkpoint's own config, then load the full base state into it.
    import pickle

    with open(base_path, "rb") as f:
        cfg = pickle.load(f)["config"]
    trainer = SDCFRNLHETrainer(cfg)
    load_nlhe_checkpoint(trainer, Path(base_path))

    plan = iteration_traversal_plan(trainer.config, weight)
    trainer._begin_iteration_regret()
    base_cum, base_strat = _snapshot_regret_tables(trainer)

    results = []
    for shard_id in shard_ids:
        pairs = shard_traversal_plan(plan, num_shards, shard_id)
        if not pairs:
            continue
        worker = copy_trainer_for_shard(trainer, base_cum, base_strat)
        worker._begin_iteration_regret()
        results.extend(run_traversal_shard(worker, plan, pairs, iteration_num))
    Path(out_path).write_bytes(serialize_shard(results))


def run_iteration_subproc(
    trainer: SDCFRNLHETrainer,
    iteration_weight: float,
    num_shards: int,
    num_workers: int | None = None,
    *,
    workdir: str | None = None,
    python_exe: str | None = None,
) -> float:
    """One bulk-sync iteration with shards run in independent subprocesses; merged in-parent.

    Drop-in, bit-identical replacement for ``run_iteration_parallel`` (the fork path) that avoids the
    JAX-after-fork deadlock. ``num_workers`` independent processes split the ``num_shards`` shards.
    """
    if num_workers is None:
        num_workers = min(num_shards, os.cpu_count() or 1)
    num_workers = max(1, min(num_workers, num_shards))
    iter_num = int(iteration_weight)

    trainer._iter_samples = []
    plan = iteration_traversal_plan(trainer.config, iteration_weight)
    trainer._begin_iteration_regret()
    # Parent keeps `trainer` as the base (workers don't touch it); merge applies deltas to it.
    _snapshot_regret_tables(trainer)  # parity with the reference phase (begin already done)

    tmp = workdir or tempfile.mkdtemp(prefix=f"sdcfr_iter{iter_num}_")
    base_path = str(Path(tmp) / f"base_iter{iter_num}.pkl")
    save_nlhe_checkpoint(trainer, Path(base_path))

    env = _worker_env()
    procs: list[tuple[subprocess.Popen, str, list[int]]] = []
    for wid in range(num_workers):
        shard_ids = [s for s in range(num_shards) if s % num_workers == wid]
        if not shard_ids:
            continue
        out_path = str(Path(tmp) / f"out_w{wid}.pkl")
        cmd = [
            python_exe or sys.executable,
            "-m", "pokerbot.training.sd_cfr_nlhe_subproc",
            "--worker",
            "--base", base_path,
            "--iter", str(iter_num),
            "--num-shards", str(num_shards),
            "--shards", ",".join(map(str, shard_ids)),
            "--out", out_path,
        ]
        procs.append((subprocess.Popen(cmd, env=env), out_path, shard_ids))

    failures = []
    for p, out_path, shard_ids in procs:
        rc = p.wait()
        if rc != 0:
            failures.append((shard_ids, rc))
    if failures:
        raise RuntimeError(f"subprocess traversal worker(s) failed: {failures}")

    unit_results = []
    for _p, out_path, _sids in procs:
        unit_results.extend(deserialize_shard(Path(out_path).read_bytes()))

    merge_traversal_shards(trainer, unit_results, plan)
    return trainer._finish_iteration_after_traversal()


def _worker_cli() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--base", required=True)
    ap.add_argument("--iter", type=int, required=True)
    ap.add_argument("--num-shards", type=int, required=True)
    ap.add_argument("--shards", required=True, help="comma-separated shard ids for this worker")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    shard_ids = [int(s) for s in args.shards.split(",") if s != ""]
    t0 = time.perf_counter()
    _run_worker(args.base, args.iter, args.num_shards, shard_ids, args.out)
    print(f"[subproc-worker shards={shard_ids}] done in {time.perf_counter()-t0:.1f}s",
          file=sys.stderr, flush=True)


if __name__ == "__main__":
    _worker_cli()
