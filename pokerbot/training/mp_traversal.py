"""Multiprocess data-parallel traversal generation for the VR-DeepPDCFR+ trainer.

WHY THIS EXISTS (measured): a single JAX-on-CPU process is ~1-2 cores on the NLHE traversal regardless
of the XLA Eigen intra-op thread count -- the traversal is a ``lax.scan`` of ~64 sequential steps each
doing tiny per-traversal MLP applies, so the work is latency/dispatch bound and Eigen cannot parallelize
the serial scan (benchmarked FLAT: 123 trav/s at 1 thread == 123 trav/s at 12 threads; see
scripts/neural_cfr/scaling_curve.py). pmap over ``--xla_force_host_platform_device_count`` host CPU
devices was benchmarked 6x SLOWER (each logical device single-threads and the per-device batch is too
small to amortize compile/dispatch). The ONLY lever that adds throughput is INDEPENDENT OS PROCESSES,
the same "many thin single-thread workers" pattern the SD-CFR launcher already uses.

CORRECTNESS: the traversals are outcome-sampled and INDEPENDENT, so splitting K traversals across W
workers and concatenating their samples is statistically identical to generating them serially -- it is
the SAME estimator, only the RNG substreams differ (the parent derives a disjoint key per (worker,
traverser, iteration) so streams never collide). The single canonical buffer + single grad path stay in
the parent; workers ONLY generate samples. So the trained strategy is statistically equivalent to the
single-process path (not bit-identical, by design -- different RNG streams).

THREADING: each worker pins itself to ONE compute thread (XLA Eigen off, OMP/BLAS=1) in the pool
initializer, BEFORE importing jax. With W workers that is W*1 == W compute threads; size W == vcpus for
full saturation with zero oversubscription. The parent's grad phase runs in its own process (also
single-thread) overlapping the next iteration's traversal dispatch.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np

# Worker-global trainer shell (built once per worker process, reused across tasks for jit-cache reuse).
_WORKER: dict[str, Any] = {}


def _worker_init(env_name: str, hidden_dim: int, num_actions: int, reward_scale: float,
                 alpha: float, epsilon: float) -> None:
    """Pool initializer: pin to one compute thread, import jax, build the jitted scan once."""
    # MUST set before importing jax: one Eigen/BLAS thread per worker so W workers == W threads.
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    os.environ["XLA_FLAGS"] = ("--xla_cpu_multi_thread_eigen=false "
                               "intra_op_parallelism_threads=1 inter_op_parallelism_threads=1")
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "NPY_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[v] = "1"

    import jax

    from pokerbot.training.cfr_env import leduc_env_spec, nlhe_env_spec
    from pokerbot.training.neural_cfr import _batched_traversal_scans

    env = nlhe_env_spec() if env_name == "nlhe" else leduc_env_spec()
    scans = jax.jit(
        _batched_traversal_scans,
        static_argnames=("alpha", "epsilon", "reward_scale", "hidden_dim", "env"),
    )
    _WORKER.update(
        jax=jax, env=env, scans=scans, hidden_dim=hidden_dim, reward_scale=reward_scale,
        alpha=alpha, epsilon=epsilon,
    )


def _worker_traverse(args: tuple) -> dict[str, np.ndarray]:
    """Generate one shard of traversals for ONE traverser; return host (numpy) flattened samples.

    args = (cum_params, inst_params, hv_params, traverser, iteration, reset_key_bytes, trav_key_bytes,
            shard_size). Params arrive as host numpy pytrees (picklable); keys as raw uint32 seeds the
    worker expands to a disjoint substream. Returns flattened (n_samples, ...) numpy arrays + masks so
    the parent can concat across workers and ingest in one batched pass."""
    import jax
    import jax.numpy as jnp

    (cum_params, inst_params, hv_params, traverser, iteration, reset_seed, trav_seed,
     shard) = args
    w = _WORKER
    env = w["env"]
    # Disjoint substreams: fold (worker-provided) seeds into PRNGKeys, then split per traversal.
    rk_root = jax.random.PRNGKey(int(reset_seed))
    tk_root = jax.random.PRNGKey(int(trav_seed))
    reset_keys = jax.random.split(rk_root, shard)
    trav_keys = jax.random.split(tk_root, shard)

    adv, strat, hist, is_trav, active = w["scans"](
        reset_keys, trav_keys, jnp.int32(traverser), jnp.int32(iteration),
        cum_params, inst_params, hv_params,
        w["alpha"], w["epsilon"], w["reward_scale"], w["hidden_dim"], env,
    )

    # flatten (shard, steps) -> (shard*steps) and move to host as numpy (picklable back to parent).
    def flat_np(d):
        return {k: np.asarray(v.reshape((-1,) + v.shape[2:])) for k, v in d.items()}

    return {
        "adv": flat_np(adv),
        "strat": flat_np(strat),
        "hist": flat_np(hist),
        "is_trav": np.asarray(is_trav.reshape((-1,))),
        "active": np.asarray(active.reshape((-1,))),
    }


class TraversalPool:
    """Persistent worker pool that generates traversal samples data-parallel across processes.

    Build once (workers pay the jit compile once on their first task, then reuse). Call
    :meth:`generate` per (traverser, iteration) with the CURRENT host params; it shards the traversals
    across workers, runs them concurrently, and returns concatenated host-numpy samples for the parent
    to ingest. Spawn/fork is chosen by the platform default; on Linux (the AWS target) fork inherits the
    pinned env. Use as a context manager so the pool is cleanly closed."""

    def __init__(self, *, num_workers: int, env_name: str, hidden_dim: int, num_actions: int,
                 reward_scale: float, alpha: float, epsilon: float):
        import multiprocessing as mp

        self.num_workers = num_workers
        self._ctx = mp.get_context("spawn")  # spawn: portable (Windows local + Linux cloud), clean env
        self._pool = self._ctx.Pool(
            processes=num_workers,
            initializer=_worker_init,
            initargs=(env_name, hidden_dim, num_actions, reward_scale, alpha, epsilon),
        )

    def generate(self, cum_params, inst_params, hv_params, traverser: int, iteration: int,
                 total_traversals: int, base_seed: int) -> dict[str, np.ndarray]:
        """Run ``total_traversals`` traversals for ``traverser`` split across the workers; concat."""
        w = self.num_workers
        per = max(1, total_traversals // w)
        shards = [per] * w
        shards[-1] += total_traversals - per * w  # remainder onto the last worker
        tasks = []
        for i, shard in enumerate(shards):
            if shard <= 0:
                continue
            # disjoint per-(worker,traverser,iteration) seeds (cheap, collision-free for our scale).
            rs = (base_seed * 2_654_435_761 + iteration * 1_000_003 + traverser * 9_176 + i * 2) & 0x7FFFFFFF
            ts = (base_seed * 2_246_822_519 + iteration * 1_000_033 + traverser * 7_919 + i * 2 + 1) & 0x7FFFFFFF
            tasks.append((cum_params, inst_params, hv_params, traverser, iteration, rs, ts, shard))
        results = self._pool.map(_worker_traverse, tasks)
        return _concat_results(results)

    def close(self) -> None:
        self._pool.close()
        self._pool.join()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()


def _concat_results(results: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    """Concatenate per-worker flattened sample dicts along the sample axis."""
    out = {"adv": {}, "strat": {}, "hist": {}}
    for grp in ("adv", "strat", "hist"):
        keys = results[0][grp].keys()
        for k in keys:
            out[grp][k] = np.concatenate([r[grp][k] for r in results], axis=0)
    out["is_trav"] = np.concatenate([r["is_trav"] for r in results], axis=0)
    out["active"] = np.concatenate([r["active"] for r in results], axis=0)
    return out
