"""
Cloud data-path layer for bulk-synchronous SD-CFR HU: shard serialization + object store.

The in-process distributed trainer (`sd_cfr_nlhe_distributed`) proves the merge MATH is correct.
This module proves the merge survives a real round-trip through an object store: a worker
serializes its `TraversalShardResult`s, PUTs them to a store, and the coordinator GETs them back,
deserializes, and merges. Fingerprint-tuple keys, numpy dtypes, and encode-bundle arrays must all
reconstruct bit-identically — a corruption here would silently poison the convergence run.

`ObjectStore` is the seam between local validation and the cloud:
  - `LocalDirObjectStore` (here) is a filesystem stand-in used by tests and local target-scale
    runs — NO AWS, fully deterministic.
  - The S3 backend (next) wraps `pokerbot.solver.s3_coordination.S3Coordinator.{upload,download}_file`
    with the same interface, so the orchestrator code below is unchanged between local and cloud.

This is the disciplined bridge: validate the exact data path locally (bit-identical vs
single-threaded) before any EC2 spend.
"""

from __future__ import annotations

import pickle
import random
from pathlib import Path
from typing import Protocol, runtime_checkable

import tempfile

import numpy as np

from pokerbot.training.sd_cfr_nlhe import (
    SDCFRNLHETrainer,
    load_nlhe_checkpoint,
    save_nlhe_checkpoint,
)
from pokerbot.training.sd_cfr_nlhe_distributed import (
    TraversalShardResult,
    _snapshot_regret_tables,
    copy_trainer_for_shard,
    iteration_traversal_plan,
    merge_traversal_shards,
    run_traversal_shard,
    shard_traversal_plan,
)

# Bump if the on-the-wire shard layout changes; deserialize refuses mismatched versions so a
# stale artifact from an older worker can never be silently merged.
SHARD_FORMAT_VERSION = 1


def serialize_shard(results: list[TraversalShardResult]) -> bytes:
    """Serialize one worker's per-pair unit results (the cloud shard artifact)."""
    return pickle.dumps((SHARD_FORMAT_VERSION, results), protocol=pickle.HIGHEST_PROTOCOL)


def deserialize_shard(blob: bytes) -> list[TraversalShardResult]:
    version, results = pickle.loads(blob)
    if version != SHARD_FORMAT_VERSION:
        raise ValueError(f"shard format version {version} != expected {SHARD_FORMAT_VERSION}")
    if not isinstance(results, list) or not all(
        isinstance(r, TraversalShardResult) for r in results
    ):
        raise TypeError("deserialized shard is not list[TraversalShardResult]")
    return results


def shard_key(iteration: int, shard_id: int) -> str:
    """Object-store key for a worker's traversal shard artifact (matches the design doc layout)."""
    return f"iter_{iteration:04d}/traversal_shard_{shard_id:04d}.pkl"


def checkpoint_key(iteration: int) -> str:
    return f"iter_{iteration:04d}/sd_cfr_hu_iter_{iteration:04d}.pkl"


@runtime_checkable
class ObjectStore(Protocol):
    """Minimal object store: the seam between LocalDir (validation) and S3 (cloud)."""

    def put(self, key: str, data: bytes) -> None: ...
    def get(self, key: str) -> bytes: ...
    def exists(self, key: str) -> bool: ...
    def list_keys(self, prefix: str) -> list[str]: ...


class LocalDirObjectStore:
    """Filesystem-backed ObjectStore — deterministic S3 stand-in for local validation."""

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        p = self.root / key
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def put(self, key: str, data: bytes) -> None:
        # Write-then-rename so a reader never sees a half-written artifact (mirrors S3 atomicity).
        tmp = self._path(key + ".partial")
        tmp.write_bytes(data)
        tmp.replace(self._path(key))

    def get(self, key: str) -> bytes:
        return self._path(key).read_bytes()

    def exists(self, key: str) -> bool:
        return (self.root / key).is_file()

    def list_keys(self, prefix: str) -> list[str]:
        keys: list[str] = []
        for p in self.root.rglob("*"):
            if not p.is_file():
                continue
            rel = str(p.relative_to(self.root)).replace("\\", "/")
            if rel.startswith(prefix) and not rel.endswith(".partial"):
                keys.append(rel)
        return sorted(keys)


class S3ObjectStore:
    """
    S3-backed ObjectStore for the cloud coordinator. Drop-in behind the ObjectStore Protocol so
    `run_iteration_via_store*` and `run_convergence_via_store` are unchanged from local.

    NOT YET LIVE-VALIDATED. The logic mirrors standard boto3 usage, but per project discipline a
    thing is verified only when its artifact exists — this needs a live-S3 round-trip smoke
    (put → get → exists → list_keys against a real bucket) before it is trusted in a run. The local
    `LocalDirObjectStore` is the validated reference; this must reproduce its semantics exactly.
    Keys are stored under ``s3://{bucket}/{prefix}/{key}``. boto3 is imported lazily so local use
    never requires it.
    """

    def __init__(self, bucket: str, prefix: str, *, region: str = "us-east-1") -> None:
        import boto3  # lazy: only the cloud path needs it

        self.bucket = bucket
        self.prefix = prefix.rstrip("/")
        self._s3 = boto3.client("s3", region_name=region)

    def _full(self, key: str) -> str:
        return f"{self.prefix}/{key}" if self.prefix else key

    def put(self, key: str, data: bytes) -> None:
        self._s3.put_object(Bucket=self.bucket, Key=self._full(key), Body=data)

    def get(self, key: str) -> bytes:
        resp = self._s3.get_object(Bucket=self.bucket, Key=self._full(key))
        return resp["Body"].read()

    def exists(self, key: str) -> bool:
        from botocore.exceptions import ClientError

        try:
            self._s3.head_object(Bucket=self.bucket, Key=self._full(key))
            return True
        except ClientError as e:
            if e.response.get("Error", {}).get("Code") in ("404", "NoSuchKey", "NotFound"):
                return False
            raise

    def list_keys(self, prefix: str) -> list[str]:
        full_prefix = self._full(prefix)
        base = f"{self.prefix}/" if self.prefix else ""
        paginator = self._s3.get_paginator("list_objects_v2")
        keys: list[str] = []
        for page in paginator.paginate(Bucket=self.bucket, Prefix=full_prefix):
            for obj in page.get("Contents", []):
                full_key = obj["Key"]
                rel = full_key[len(base) :] if base and full_key.startswith(base) else full_key
                keys.append(rel)
        return sorted(keys)


def run_iteration_via_store(
    trainer: SDCFRNLHETrainer,
    iteration_weight: float,
    num_shards: int,
    store: ObjectStore,
    *,
    iteration_num: int | None = None,
) -> float:
    """
    One bulk-sync iteration routed through an object store (the cloud data path, run locally).

    Fan-out: each shard runs on a fresh worker copy off the per-iteration base snapshot, then
    serializes its results and PUTs them under ``shard_key(iter, shard_id)``.
    Barrier+merge: the coordinator GETs every shard artifact back, deserializes, and merges in
    global plan order. Bit-identical to single-threaded because the store round-trip is lossless.

    This mirrors ``run_traversal_phase_sharded`` but with a serialize/store/deserialize hop between
    worker and coordinator — the part the cloud adds and the part that must not corrupt the merge.
    """
    if iteration_num is None:
        iteration_num = int(iteration_weight)
    trainer._iter_samples = []
    plan = iteration_traversal_plan(trainer.config, iteration_weight)
    trainer._begin_iteration_regret()
    base_cum, base_strat = _snapshot_regret_tables(trainer)

    # --- fan out: worker copies run shards, results go through the store ---
    for shard_id in range(num_shards):
        pairs = shard_traversal_plan(plan, num_shards, shard_id)
        if not pairs:
            continue
        worker = copy_trainer_for_shard(trainer, base_cum, base_strat)
        worker._begin_iteration_regret()
        results = run_traversal_shard(worker, plan, pairs, iteration_num)
        store.put(shard_key(iteration_num, shard_id), serialize_shard(results))

    # --- barrier: gather every shard artifact back, deserialize, merge in plan order ---
    unit_results: list[TraversalShardResult] = []
    for shard_id in range(num_shards):
        key = shard_key(iteration_num, shard_id)
        if not store.exists(key):
            continue
        unit_results.extend(deserialize_shard(store.get(key)))

    merge_traversal_shards(trainer, unit_results, plan)
    return trainer._finish_iteration_after_traversal()


def checkpoint_to_store(trainer: SDCFRNLHETrainer, store: ObjectStore, iteration: int) -> None:
    """Write the iteration barrier checkpoint to the object store (reuses the validated save path)."""
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "ckpt.pkl"
        save_nlhe_checkpoint(trainer, p)
        store.put(checkpoint_key(iteration), p.read_bytes())


def checkpoint_from_store(trainer: SDCFRNLHETrainer, store: ObjectStore, iteration: int) -> int:
    """Load a barrier checkpoint from the store; returns (and sets) completed_iterations."""
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "ckpt.pkl"
        p.write_bytes(store.get(checkpoint_key(iteration)))
        return load_nlhe_checkpoint(trainer, p)


def latest_checkpoint_iteration(store: ObjectStore) -> int:
    """Highest iteration N for which a barrier checkpoint exists in the store (0 if none)."""
    best = 0
    for key in store.list_keys("iter_"):
        name = key.rsplit("/", 1)[-1]
        if name.startswith("sd_cfr_hu_iter_") and name.endswith(".pkl"):
            try:
                best = max(best, int(name[len("sd_cfr_hu_iter_") : -len(".pkl")]))
            except ValueError:
                continue
    return best


def run_convergence_via_store(
    trainer: SDCFRNLHETrainer,
    num_iters: int,
    num_shards: int,
    store: ObjectStore,
    *,
    start_iter: int = 1,
    checkpoint_every_barrier: bool = True,
) -> int:
    """
    The bulk-sync coordinator loop (object-store backed, AWS-free): for each iteration, fan out
    shards through the store, barrier+merge+train, then checkpoint the barrier to the store.

    Per-iteration RNG is seeded as ``config.seed + n - 1`` so iteration N depends only on (seed, N):
    resuming at N reproduces an uninterrupted run bit-identically (validated by the resume gate).
    Resume by constructing a fresh trainer, ``checkpoint_from_store(..., latest)``, and calling this
    with ``start_iter = latest + 1``.
    """
    for n in range(start_iter, num_iters + 1):
        np.random.seed(trainer.config.seed + n - 1)
        run_iteration_via_store(trainer, float(n), num_shards, store, iteration_num=n)
        trainer._completed_iterations = n
        if checkpoint_every_barrier:
            checkpoint_to_store(trainer, store, n)
    return trainer._completed_iterations


def run_iteration_via_store_coordinated(
    trainer: SDCFRNLHETrainer,
    iteration_weight: float,
    num_shards: int,
    store: ObjectStore,
    *,
    iteration_num: int | None = None,
    killed_shard_ids: frozenset[int] = frozenset(),
    shuffle_gather: bool = False,
    duplicate_shard_ids: frozenset[int] = frozenset(),
) -> float:
    """
    Bulk-sync iteration through an object store, with induced CLOUD failure modes — the data-path
    analog of ``train_iteration_coordinated``, but exercising real serialization + the idempotent
    merge together. All modes must stay bit-identical to single-threaded:

    - ``killed_shard_ids``: skipped on the first pass (no artifact uploaded), then reclaimed (a
      fresh worker re-runs the whole shard from the per-iteration base and re-uploads). Partial
      work is never published, so reclaim cannot double-count.
    - ``shuffle_gather``: the coordinator reads shard artifacts back in arbitrary order (S3 list is
      unordered). The merge re-imposes global ``plan.pairs`` order, so this is benign.
    - ``duplicate_shard_ids``: a shard's artifact is delivered twice (slow-but-alive worker plus
      its reclaim). The merge dedupes by unit, so the duplicate cannot double-count regrets.
    """
    if iteration_num is None:
        iteration_num = int(iteration_weight)
    trainer._iter_samples = []
    plan = iteration_traversal_plan(trainer.config, iteration_weight)
    trainer._begin_iteration_regret()
    base_cum, base_strat = _snapshot_regret_tables(trainer)

    def _run_and_store(shard_id: int) -> None:
        pairs = shard_traversal_plan(plan, num_shards, shard_id)
        if not pairs:
            return
        worker = copy_trainer_for_shard(trainer, base_cum, base_strat)
        worker._begin_iteration_regret()
        results = run_traversal_shard(worker, plan, pairs, iteration_num)
        store.put(shard_key(iteration_num, shard_id), serialize_shard(results))

    # First pass: every shard except the induced-dead ones.
    for shard_id in range(num_shards):
        if shard_id in killed_shard_ids:
            continue
        _run_and_store(shard_id)
    # Reclaim pass: re-run the dead shards from base and re-upload (whole-shard, idempotent).
    for shard_id in sorted(killed_shard_ids):
        if not store.exists(shard_key(iteration_num, shard_id)):
            _run_and_store(shard_id)

    # Barrier: gather every shard artifact, optionally out-of-order and with a duplicate delivery.
    gather_ids = [s for s in range(num_shards) if store.exists(shard_key(iteration_num, s))]
    gather_ids += [s for s in sorted(duplicate_shard_ids) if s in gather_ids]
    if shuffle_gather:
        random.shuffle(gather_ids)
    missing = [s for s in range(num_shards) if not store.exists(shard_key(iteration_num, s))]
    if missing:
        raise RuntimeError(f"barrier incomplete: missing shard artifacts {missing}")

    unit_results: list[TraversalShardResult] = []
    for shard_id in gather_ids:
        unit_results.extend(deserialize_shard(store.get(shard_key(iteration_num, shard_id))))

    merge_traversal_shards(trainer, unit_results, plan)
    return trainer._finish_iteration_after_traversal()
