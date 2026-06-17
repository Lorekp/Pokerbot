"""Bulk-synchronous HU SD-CFR: traversal shards, merge, and coordinator helpers."""

from __future__ import annotations

import copy
import json
import random
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TextIO

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

# Leduc-gate-analog HU cloud convergence scale (1080 traverser-visits / iter).
HU_CLOUD_TARGET_TRAVERSALS_PER_ITER = 90
HU_CLOUD_TARGET_DEAL_SWEEPS = 6
HU_CLOUD_TARGET_WORKERS = 8
HU_CLOUD_TARGET_VISITS_PER_ITER = (
    HU_CLOUD_TARGET_TRAVERSALS_PER_ITER * HU_CLOUD_TARGET_DEAL_SWEEPS * 2
)

from pokerbot.engine.jax_engine import init_hand
from pokerbot.training.sd_cfr_nlhe import (
    NLHE_NUM_ACTIONS,
    SDCFRNLHEConfig,
    SDCFRNLHETrainer,
    _NLHERegretVisit,
)


@dataclass
class TraversalShardResult:
    """Partial iteration outputs from one disjoint (sweep, trav) subset."""

    sweep_trav_pairs: list[tuple[int, int]]
    cum_regret_delta: dict[tuple, np.ndarray] = field(default_factory=dict)
    strat_sum_delta: dict[tuple, np.ndarray] = field(default_factory=dict)
    pending_regret: dict[tuple, list[_NLHERegretVisit]] = field(default_factory=dict)
    collisions: int = 0  # tabular-key collisions metered while running this unit (should be 0)


@dataclass
class IterationTraversalPlan:
    """Ordered traversal units for one train_iteration (bulk-sync fan-out)."""

    iteration_weight: float
    pairs: list[tuple[int, int]]  # (sweep_i, trav_i) in global order


def iteration_traversal_plan(cfg: SDCFRNLHEConfig, iteration_weight: float) -> IterationTraversalPlan:
    pairs: list[tuple[int, int]] = []
    for sweep_i in range(max(1, cfg.deal_sweeps)):
        for trav_i in range(cfg.traversals_per_iter):
            pairs.append((sweep_i, trav_i))
    return IterationTraversalPlan(iteration_weight=iteration_weight, pairs=pairs)


def shard_traversal_plan(
    plan: IterationTraversalPlan, num_shards: int, shard_id: int
) -> list[tuple[int, int]]:
    """Disjoint partition of (sweep, trav) pairs for worker shard_id."""
    if num_shards < 1 or shard_id < 0 or shard_id >= num_shards:
        raise ValueError("invalid shard partition")
    n = len(plan.pairs)
    if n == 0:
        return []
    # Stable contiguous chunks preserve global order for pending_regret concat.
    base, rem = divmod(n, num_shards)
    start = shard_id * base + min(shard_id, rem)
    size = base + (1 if shard_id < rem else 0)
    return plan.pairs[start : start + size]


def traversal_prng_keys(
    trainer: SDCFRNLHETrainer, iteration_num: int, sweep_i: int, trav_i: int
) -> tuple[Array, Array]:
    """Deterministic deal + traverse keys (order-independent across shards)."""
    base = jax.random.fold_in(trainer._traversal_root_key, int(iteration_num))
    base = jax.random.fold_in(base, int(sweep_i))
    base = jax.random.fold_in(base, int(trav_i))
    return jax.random.split(base)


def _snapshot_regret_tables(trainer: SDCFRNLHETrainer) -> tuple[dict, dict]:
    cum = {k: v.copy() for k, v in trainer._cum_regret.items()}
    strat = {k: v.copy() for k, v in trainer._strat_sum.items()}
    return cum, strat


def _delta_regret_tables(
    before_cum: dict[tuple, np.ndarray],
    before_strat: dict[tuple, np.ndarray],
    after_trainer: SDCFRNLHETrainer,
) -> tuple[dict[tuple, np.ndarray], dict[tuple, np.ndarray]]:
    cum_delta: dict[tuple, np.ndarray] = {}
    for k, v in after_trainer._cum_regret.items():
        prev = before_cum.get(k)
        if prev is None:
            cum_delta[k] = v.copy()
        else:
            d = v - prev
            if np.any(d != 0):
                cum_delta[k] = d.astype(np.float64)
    strat_delta: dict[tuple, np.ndarray] = {}
    for k, v in after_trainer._strat_sum.items():
        prev = before_strat.get(k)
        if prev is None:
            strat_delta[k] = v.copy()
        else:
            d = v - prev
            if np.any(d != 0):
                strat_delta[k] = d.astype(np.float64)
    return cum_delta, strat_delta


def _pending_fragment(
    trainer: SDCFRNLHETrainer, before_lens: dict[tuple, int]
) -> dict[tuple, list[_NLHERegretVisit]]:
    frag: dict[tuple, list[_NLHERegretVisit]] = {}
    for key, visits in trainer._pending_regret.items():
        n = before_lens.get(key, 0)
        if len(visits) > n:
            frag[key] = list(visits[n:])
    return frag


def run_traversal_unit(
    trainer: SDCFRNLHETrainer,
    plan: IterationTraversalPlan,
    sweep_i: int,
    trav_i: int,
    iteration_num: int,
) -> TraversalShardResult:
    """One (sweep, trav) × all live seats; does not flush pending or train net."""
    from pokerbot.training import sd_cfr_nlhe as _sdm

    before_cum, before_strat = _snapshot_regret_tables(trainer)
    before_pending = {k: len(v) for k, v in trainer._pending_regret.items()}
    coll_before = _sdm._collision_count
    trainer._iter_weight = plan.iteration_weight
    occ = jnp.asarray(trainer.occupancy, dtype=bool)
    k1, k2 = traversal_prng_keys(trainer, iteration_num, sweep_i, trav_i)
    state = init_hand(
        k1,
        button=jnp.int32(trainer.config.button_seat),
        occupancy=occ,
    )
    for seat in trainer._live_seats:
        trainer.traverse(state, seat, k2)
    cum_d, strat_d = _delta_regret_tables(before_cum, before_strat, trainer)
    pending = _pending_fragment(trainer, before_pending)
    return TraversalShardResult(
        sweep_trav_pairs=[(sweep_i, trav_i)],
        cum_regret_delta=cum_d,
        strat_sum_delta=strat_d,
        pending_regret=pending,
        collisions=_sdm._collision_count - coll_before,
    )


def run_traversal_shard(
    trainer: SDCFRNLHETrainer,
    plan: IterationTraversalPlan,
    shard_pairs: list[tuple[int, int]],
    iteration_num: int,
) -> list[TraversalShardResult]:
    """Run disjoint (sweep, trav) units; one result per pair for ordered pending merge."""
    results: list[TraversalShardResult] = []
    for sweep_i, trav_i in shard_pairs:
        results.append(run_traversal_unit(trainer, plan, sweep_i, trav_i, iteration_num))
    return results


def merge_traversal_shards(
    trainer: SDCFRNLHETrainer,
    unit_results: list[TraversalShardResult],
    plan: IterationTraversalPlan,
) -> None:
    """
    Merge traversal units into trainer state (SUM regrets/strat_sum; concat pending in global order).

    Must be followed by a single ``_flush_pending_regret()`` on the coordinator — belief-weighted
    aggregation is NOT associative per-shard if flushed independently.

    Idempotent under at-least-once delivery: a (sweep, trav) pair delivered more than once
    (e.g. a slow-but-alive cloud worker plus its reclaim) is merged exactly ONCE. Both the
    regret/strat_sum SUM and the pending concat are keyed off ``pair_to_unit`` / ``plan.pairs``,
    so a duplicate cannot double-count. ``pair_to_unit`` keeps the last-delivered unit per pair.
    """
    pair_to_unit: dict[tuple[int, int], TraversalShardResult] = {}
    for unit in unit_results:
        for p in unit.sweep_trav_pairs:
            pair_to_unit[p] = unit
    # Surface worker-side key collisions on the coordinator (deduped by unit identity below mirrors
    # the regret merge so a re-delivered unit isn't double-counted).

    # Accumulate each unique unit once, in global plan order (deduped by identity so a unit
    # owning >1 pair is not multi-counted, and a re-delivered pair is collapsed via pair_to_unit).
    merged_unit_ids: set[int] = set()
    for pair in plan.pairs:
        unit = pair_to_unit.get(pair)
        if unit is None or id(unit) in merged_unit_ids:
            continue
        merged_unit_ids.add(id(unit))
        trainer._iter_collisions = getattr(trainer, "_iter_collisions", 0) + int(unit.collisions)
        for key, delta in unit.cum_regret_delta.items():
            if key not in trainer._cum_regret:
                trainer._cum_regret[key] = np.zeros(NLHE_NUM_ACTIONS, dtype=np.float64)
            trainer._cum_regret[key] += delta
        for key, delta in unit.strat_sum_delta.items():
            if key not in trainer._strat_sum:
                trainer._strat_sum[key] = np.zeros(NLHE_NUM_ACTIONS, dtype=np.float64)
            trainer._strat_sum[key] += delta

    merged_pending: dict[tuple, list[_NLHERegretVisit]] = defaultdict(list)
    for sweep_i, trav_i in plan.pairs:
        unit = pair_to_unit.get((sweep_i, trav_i))
        if unit is None:
            continue
        for info_key, visits in unit.pending_regret.items():
            if visits:
                merged_pending[info_key].extend(visits)

    trainer._pending_regret = merged_pending


def run_traversal_phase_single_threaded(
    trainer: SDCFRNLHETrainer,
    iteration_num: int,
    iteration_weight: float,
    *,
    logger: IterationEventLogger | None = None,
    log_every_units: int = 45,
) -> None:
    """Reference traversal: one trainer, global pending, fold_in keys (ground truth)."""
    plan = iteration_traversal_plan(trainer.config, iteration_weight)
    trainer._iter_weight = iteration_weight
    trainer._begin_iteration_regret()
    total = len(plan.pairs)
    t0 = time.perf_counter()
    if logger:
        logger.emit(
            "traversal_phase_start",
            phase="single_threaded",
            iteration=iteration_num,
            units_total=total,
        )
    for idx, (sweep_i, trav_i) in enumerate(plan.pairs):
        k1, k2 = traversal_prng_keys(trainer, iteration_num, sweep_i, trav_i)
        occ = jnp.asarray(trainer.occupancy, dtype=bool)
        state = init_hand(
            k1,
            button=jnp.int32(trainer.config.button_seat),
            occupancy=occ,
        )
        for seat in trainer._live_seats:
            trainer.traverse(state, seat, k2)
        done = idx + 1
        if done == total or done % max(1, log_every_units) == 0:
            _emit_progress(
                logger,
                "traversal_progress",
                phase="single_threaded",
                done=done,
                total=total,
                elapsed_sec=time.perf_counter() - t0,
                sweep=sweep_i,
                trav=trav_i,
            )
    if logger:
        logger.emit(
            "traversal_phase_done",
            phase="single_threaded",
            iteration=iteration_num,
            units_total=total,
            elapsed_sec=round(time.perf_counter() - t0, 1),
        )


def run_traversal_phase_sharded(
    trainer: SDCFRNLHETrainer,
    iteration_num: int,
    iteration_weight: float,
    num_shards: int,
    *,
    logger: IterationEventLogger | None = None,
) -> list[TraversalShardResult]:
    """Distributed traversal: worker copies → merge into coordinator (one flush after)."""
    plan = iteration_traversal_plan(trainer.config, iteration_weight)
    trainer._begin_iteration_regret()
    base_cum, base_strat = _snapshot_regret_tables(trainer)
    total_units = len(plan.pairs)
    t0 = time.perf_counter()
    units_done = 0
    if logger:
        logger.emit(
            "traversal_phase_start",
            phase="sharded",
            iteration=iteration_num,
            num_shards=num_shards,
            units_total=total_units,
        )

    unit_results: list[TraversalShardResult] = []
    for shard_id in range(num_shards):
        pairs = shard_traversal_plan(plan, num_shards, shard_id)
        if not pairs:
            continue
        if logger:
            logger.emit(
                "shard_start",
                iteration=iteration_num,
                shard_id=shard_id,
                n_pairs=len(pairs),
            )
        t_shard = time.perf_counter()
        worker = copy_trainer_for_shard(trainer, base_cum, base_strat)
        worker._begin_iteration_regret()
        unit_results.extend(run_traversal_shard(worker, plan, pairs, iteration_num))
        units_done += len(pairs)
        if logger:
            logger.emit(
                "shard_done",
                iteration=iteration_num,
                shard_id=shard_id,
                elapsed_sec=round(time.perf_counter() - t_shard, 1),
                units_done=units_done,
                units_total=total_units,
            )
            _emit_progress(
                logger,
                "traversal_progress",
                phase="sharded",
                done=units_done,
                total=total_units,
                elapsed_sec=time.perf_counter() - t0,
                last_shard=shard_id,
            )

    merge_traversal_shards(trainer, unit_results, plan)
    if logger:
        logger.emit(
            "merge_done",
            iteration=iteration_num,
            n_units=len(unit_results),
            elapsed_sec=round(time.perf_counter() - t0, 1),
        )
    return unit_results


def train_iteration_sharded(
    trainer: SDCFRNLHETrainer,
    iteration_weight: float,
    num_shards: int,
    *,
    logger: IterationEventLogger | None = None,
) -> float:
    """One bulk-sync iteration: sharded traversals → merge → single flush/train."""
    iteration_num = int(iteration_weight)
    trainer._iter_samples = []
    run_traversal_phase_sharded(
        trainer, iteration_num, iteration_weight, num_shards, logger=logger
    )
    if logger:
        logger.emit("net_train_start", iteration=iteration_num)
    t0 = time.perf_counter()
    loss = trainer._finish_iteration_after_traversal()
    if logger:
        logger.emit(
            "net_train_done",
            iteration=iteration_num,
            loss=float(loss),
            elapsed_sec=round(time.perf_counter() - t0, 1),
        )
    return loss


def copy_trainer_for_shard(
    trainer: SDCFRNLHETrainer,
    cum_regret: dict[tuple, np.ndarray],
    strat_sum: dict[tuple, np.ndarray],
) -> SDCFRNLHETrainer:
    """Frozen policy + regret tables for parallel traversal workers."""
    worker = SDCFRNLHETrainer(trainer.config)
    worker.state = trainer.state
    worker._batch_stats = trainer._batch_stats
    worker._traversal_root_key = trainer._traversal_root_key
    worker._cum_regret = {k: v.copy() for k, v in cum_regret.items()}
    worker._strat_sum = defaultdict(
        lambda: np.zeros(NLHE_NUM_ACTIONS, dtype=np.float64),
        {k: v.copy() for k, v in strat_sum.items()},
    )
    worker._reservoir = list(trainer._reservoir)
    worker._reservoir_insert_count = trainer._reservoir_insert_count
    return worker


def compare_trainer_states(
    reference: SDCFRNLHETrainer,
    candidate: SDCFRNLHETrainer,
    *,
    params_atol: float = 1e-5,
    params_rtol: float = 1e-4,
    regret_atol: float = 1e-9,
) -> dict[str, Any]:
    """Ground-truth check: candidate vs single-threaded reference (not inter-worker)."""
    from flax.core import unfreeze

    ref_p = unfreeze(reference.state.params)
    cand_p = unfreeze(candidate.state.params)

    def _tree_allclose(a, b) -> bool:
        if isinstance(a, dict):
            return all(_tree_allclose(a[k], b[k]) for k in a)
        return bool(np.allclose(np.asarray(a), np.asarray(b), atol=params_atol, rtol=params_rtol))

    params_match = _tree_allclose(ref_p, cand_p)

    ref_keys = set(reference._cum_regret.keys())
    cand_keys = set(candidate._cum_regret.keys())
    regret_keys_match = ref_keys == cand_keys
    max_regret_diff = 0.0
    if regret_keys_match:
        for k in ref_keys:
            d = float(np.max(np.abs(reference._cum_regret[k] - candidate._cum_regret[k])))
            max_regret_diff = max(max_regret_diff, d)
    regret_match = regret_keys_match and max_regret_diff <= regret_atol

    ref_strat_keys = set(reference._strat_sum.keys())
    cand_strat_keys = set(candidate._strat_sum.keys())
    strat_keys_match = ref_strat_keys == cand_strat_keys
    max_strat_diff = 0.0
    if strat_keys_match:
        for k in ref_strat_keys:
            d = float(np.max(np.abs(reference._strat_sum[k] - candidate._strat_sum[k])))
            max_strat_diff = max(max_strat_diff, d)
    strat_match = strat_keys_match and max_strat_diff <= regret_atol

    bit_identical_regret = regret_keys_match and max_regret_diff == 0.0
    bit_identical_strat = strat_keys_match and max_strat_diff == 0.0

    return {
        "params_match": params_match,
        "cum_regret_match": regret_match,
        "max_cum_regret_diff": max_regret_diff,
        "cum_regret_bit_identical": bit_identical_regret,
        "strat_sum_match": strat_match,
        "max_strat_sum_diff": max_strat_diff,
        "strat_sum_bit_identical": bit_identical_strat,
        "reservoir_len_match": len(reference._reservoir) == len(candidate._reservoir),
        "reservoir_insert_match": reference._reservoir_insert_count
        == candidate._reservoir_insert_count,
    }


class IterationEventLogger:
    """JSONL observability for bulk-sync traversal (cloud console analogue)."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh: TextIO = self.path.open("a", encoding="utf-8")

    def emit(self, event: str, **fields: Any) -> None:
        row = {"ts": time.time(), "event": event, **fields}
        self._fh.write(json.dumps(row, default=str) + "\n")
        self._fh.flush()
        print(json.dumps(row, default=str), flush=True)

    def close(self) -> None:
        self._fh.close()


def _emit_progress(
    logger: IterationEventLogger | None,
    event: str,
    *,
    phase: str,
    done: int,
    total: int,
    elapsed_sec: float,
    **extra: Any,
) -> None:
    pct = round(100.0 * done / max(total, 1), 1)
    msg = (
        f"  [{phase}] {done}/{total} units ({pct}%) "
        f"elapsed={elapsed_sec:.0f}s"
    )
    print(msg, flush=True)
    if logger is not None:
        logger.emit(
            event,
            phase=phase,
            units_done=done,
            units_total=total,
            pct=pct,
            elapsed_sec=round(elapsed_sec, 1),
            **extra,
        )


def cloud_target_config(**overrides: Any) -> SDCFRNLHEConfig:
    """Default HU convergence-scale config for merge gate / cloud validate."""
    base = dict(
        iterations=1,
        traversals_per_iter=HU_CLOUD_TARGET_TRAVERSALS_PER_ITER,
        deal_sweeps=HU_CLOUD_TARGET_DEAL_SWEEPS,
        train_batch_size=2048,
        sft_checkpoint=None,
        seed=42,
        max_traverse_depth=64,
    )
    base.update(overrides)
    return SDCFRNLHEConfig(**base)


def run_traversal_phase_coordinated(
    trainer: SDCFRNLHETrainer,
    iteration_num: int,
    iteration_weight: float,
    num_shards: int,
    *,
    killed_shard_ids: frozenset[int] = frozenset(),
    shuffle_delivery_order: bool = False,
    logger: IterationEventLogger | None = None,
) -> list[TraversalShardResult]:
    """
    Coordinator traversal with induced worker loss + reclaim.

    First pass skips ``killed_shard_ids`` (simulates mid-iter worker death). Reclaim pass
    runs only missing shards. Merge uses global (sweep, trav) order regardless of delivery order.
    """
    plan = iteration_traversal_plan(trainer.config, iteration_weight)
    trainer._begin_iteration_regret()
    base_cum, base_strat = _snapshot_regret_tables(trainer)
    completed: dict[int, list[TraversalShardResult]] = {}

    def _run_shard(shard_id: int, *, reclaim: bool) -> None:
        pairs = shard_traversal_plan(plan, num_shards, shard_id)
        if not pairs:
            return
        if logger:
            logger.emit(
                "shard_start",
                iteration=iteration_num,
                shard_id=shard_id,
                reclaim=reclaim,
                n_pairs=len(pairs),
            )
        t0 = time.perf_counter()
        worker = copy_trainer_for_shard(trainer, base_cum, base_strat)
        worker._begin_iteration_regret()
        completed[shard_id] = run_traversal_shard(worker, plan, pairs, iteration_num)
        if logger:
            logger.emit(
                "shard_done",
                iteration=iteration_num,
                shard_id=shard_id,
                reclaim=reclaim,
                elapsed_sec=round(time.perf_counter() - t0, 2),
                n_units=len(completed[shard_id]),
            )

    for shard_id in range(num_shards):
        if shard_id in killed_shard_ids:
            if logger:
                logger.emit(
                    "shard_killed",
                    iteration=iteration_num,
                    shard_id=shard_id,
                    reason="induced_worker_loss",
                )
            continue
        _run_shard(shard_id, reclaim=False)

    for shard_id in sorted(killed_shard_ids):
        if shard_id not in completed:
            if logger:
                logger.emit(
                    "shard_reclaim",
                    iteration=iteration_num,
                    shard_id=shard_id,
                )
            _run_shard(shard_id, reclaim=True)

    missing = [s for s in range(num_shards) if s not in completed]
    if missing:
        raise RuntimeError(f"traversal barrier incomplete: missing shards {missing}")

    unit_results: list[TraversalShardResult] = []
    for shard_id in range(num_shards):
        unit_results.extend(completed[shard_id])

    if shuffle_delivery_order:
        random.shuffle(unit_results)
        if logger:
            logger.emit("delivery_shuffled", iteration=iteration_num, n_units=len(unit_results))

    merge_traversal_shards(trainer, unit_results, plan)
    if logger:
        logger.emit(
            "merge_done",
            iteration=iteration_num,
            n_units=len(unit_results),
            n_pending_keys=len(trainer._pending_regret),
        )
    return unit_results


def train_iteration_coordinated(
    trainer: SDCFRNLHETrainer,
    iteration_weight: float,
    num_shards: int,
    *,
    killed_shard_ids: frozenset[int] = frozenset(),
    shuffle_delivery_order: bool = False,
    logger: IterationEventLogger | None = None,
) -> float:
    iteration_num = int(iteration_weight)
    trainer._iter_samples = []
    run_traversal_phase_coordinated(
        trainer,
        iteration_num,
        iteration_weight,
        num_shards,
        killed_shard_ids=killed_shard_ids,
        shuffle_delivery_order=shuffle_delivery_order,
        logger=logger,
    )
    return trainer._finish_iteration_after_traversal()
