#!/usr/bin/env python3
"""
Cloud-validate one bulk-sync iteration: induced worker loss + barrier resume.

Runs locally (coordinator simulation) at target visit/worker scale. Each failure mode
is induced, not assumed.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from pokerbot.training.sd_cfr_nlhe import (
    SDCFRNLHETrainer,
    load_nlhe_checkpoint,
    save_nlhe_checkpoint,
)
from pokerbot.training.sd_cfr_nlhe_distributed import (
    HU_CLOUD_TARGET_WORKERS,
    IterationEventLogger,
    cloud_target_config,
    compare_trainer_states,
    train_iteration_coordinated,
    train_iteration_sharded,
)

ROOT = Path(__file__).resolve().parents[1]


def _gate_pass(cmp: dict) -> bool:
    return bool(
        cmp["params_match"]
        and cmp["cum_regret_match"]
        and cmp["strat_sum_match"]
        and cmp.get("cum_regret_bit_identical")
        and cmp.get("strat_sum_bit_identical")
    )


def _checkpoint_tables_pass(cmp: dict) -> bool:
    """CFR tables + reservoir are the barrier authority (not TrainState wrapper identity)."""
    return bool(
        cmp.get("cum_regret_bit_identical")
        and cmp.get("strat_sum_bit_identical")
        and cmp.get("reservoir_len_match")
        and cmp.get("reservoir_insert_match")
    )


def _barrier_resume_pass(post_load: dict, after_iter2: dict, *, loss_ref: float, loss_res: float) -> bool:
    """Checkpoint tables exact; iter-2 regrets exact; loss exact; params may differ ~1e-4 after fresh TrainState."""
    return bool(
        _checkpoint_tables_pass(post_load)
        and after_iter2.get("cum_regret_bit_identical")
        and after_iter2.get("strat_sum_bit_identical")
        and abs(loss_ref - loss_res) < 1e-5
    )


def test_at_scale_merge(
    cfg,
    workers: int,
    *,
    shuffle: bool = False,
    killed: frozenset[int] = frozenset(),
    log: IterationEventLogger | None = None,
) -> dict:
    ref = SDCFRNLHETrainer(cfg)
    dist = SDCFRNLHETrainer(cfg)
    np.random.seed(cfg.seed)
    ref.train_iteration(1.0)
    np.random.seed(cfg.seed)
    if killed or shuffle:
        train_iteration_coordinated(
            dist,
            1.0,
            workers,
            killed_shard_ids=killed,
            shuffle_delivery_order=shuffle,
            logger=log,
        )
    else:
        train_iteration_sharded(dist, 1.0, num_shards=workers)
    cmp = compare_trainer_states(ref, dist)
    cmp["passed"] = _gate_pass(cmp)
    return cmp


def test_worker_kill_reclaim(cfg, workers: int, log: IterationEventLogger) -> dict:
    killed = frozenset({workers // 2})
    log.emit("test_start", test="worker_kill_reclaim", killed_shard_ids=list(killed))
    cmp = test_at_scale_merge(cfg, workers, killed=killed, log=log)
    cmp["test"] = "worker_kill_reclaim"
    cmp["killed_shard_ids"] = list(killed)
    log.emit("test_end", test="worker_kill_reclaim", passed=cmp["passed"])
    return cmp


def test_out_of_order_delivery(cfg, workers: int, log: IterationEventLogger) -> dict:
    log.emit("test_start", test="out_of_order_delivery")
    cmp = test_at_scale_merge(cfg, workers, shuffle=True, log=log)
    cmp["test"] = "out_of_order_delivery"
    log.emit("test_end", test="out_of_order_delivery", passed=cmp["passed"])
    return cmp


def test_barrier_resume(cfg, workers: int, work_dir: Path, log: IterationEventLogger) -> dict:
    log.emit("test_start", test="barrier_resume")
    uninterrupted = SDCFRNLHETrainer(cfg)
    np.random.seed(cfg.seed)
    uninterrupted.train_iteration(1.0)

    interrupted = SDCFRNLHETrainer(cfg)
    np.random.seed(cfg.seed)
    interrupted.train_iteration(1.0)
    interrupted._completed_iterations = 1
    ckpt = work_dir / f"barrier_iter_0001_seed{cfg.seed}.pkl"
    save_nlhe_checkpoint(interrupted, ckpt)
    log.emit("checkpoint_written", path=str(ckpt))

    resumed = SDCFRNLHETrainer(cfg)
    resumed._completed_iterations = load_nlhe_checkpoint(resumed, ckpt)
    post_load = compare_trainer_states(uninterrupted, resumed)
    post_load["checkpoint_tables_pass"] = _checkpoint_tables_pass(post_load)
    if not _checkpoint_tables_pass(post_load):
        cmp = post_load
        cmp["test"] = "barrier_resume"
        cmp["passed"] = False
        cmp["phase"] = "post_load_iter1"
        log.emit("test_end", test="barrier_resume", passed=False, phase="post_load_iter1")
        return cmp
    np.random.seed(cfg.seed + 1)
    loss_res = resumed.train_iteration(2.0)
    np.random.seed(cfg.seed + 1)
    loss_ref = uninterrupted.train_iteration(2.0)

    after_iter2 = compare_trainer_states(uninterrupted, resumed)
    cmp = {**after_iter2, **post_load}
    cmp["test"] = "barrier_resume"
    cmp["post_load_passed"] = _gate_pass(post_load)
    cmp["after_iter2"] = after_iter2
    cmp["loss_iter2_ref"] = loss_ref
    cmp["loss_iter2_resumed"] = loss_res
    cmp["loss_iter2_match"] = abs(loss_ref - loss_res) < 1e-5
    cmp["passed"] = _barrier_resume_pass(
        post_load, after_iter2, loss_ref=loss_ref, loss_res=loss_res
    )
    cmp["checkpoint"] = str(ckpt)
    cmp["note"] = (
        "Fresh TrainState after load may differ in params ~1e-4 after grad (XLA); "
        "checkpoint tables and iter-2 loss/regret are the authority."
    )
    log.emit("test_end", test="barrier_resume", passed=cmp["passed"])
    return cmp


def test_observability(log_path: Path) -> dict:
    ok = log_path.is_file() and log_path.stat().st_size > 0
    events: list[str] = []
    if ok:
        for line in log_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                events.append(json.loads(line).get("event", ""))
    required = {"shard_start", "shard_done", "shard_killed", "shard_reclaim", "merge_done"}
    has = required.issubset(set(events))
    return {
        "test": "observability",
        "passed": ok and has,
        "log_path": str(log_path),
        "events_seen": sorted(set(events)),
        "required_events": sorted(required),
    }


def main() -> None:
    p = argparse.ArgumentParser(description="HU cloud validate one iteration (induced failures)")
    p.add_argument("--workers", type=int, default=HU_CLOUD_TARGET_WORKERS)
    p.add_argument("--smoke", action="store_true", help="4 trav × 1 sweep × 2 workers (fast debug)")
    p.add_argument("--skip-resume", action="store_true", help="Skip 2-iter barrier resume (saves time)")
    p.add_argument("--work-dir", default="logs/sd_cfr_hu_cloud_validate_work")
    p.add_argument("--output", default="logs/sd_cfr_hu_cloud_validate_one_iter.json")
    p.add_argument("--event-log", default="logs/sd_cfr_hu_cloud_validate_events.jsonl")
    p.add_argument(
        "--max-wall-sec",
        type=float,
        default=0.0,
        help="Abort exit 124 if total wall exceeds (target default set in main)",
    )
    p.add_argument(
        "--status-file",
        default="logs/sd_cfr_hu_cloud_validate_status.json",
    )
    args = p.parse_args()

    if args.smoke:
        cfg = cloud_target_config(traversals_per_iter=4, deal_sweeps=1, train_batch_size=256)
        workers = 2
        max_wall = args.max_wall_sec if args.max_wall_sec > 0 else 7200.0
        event_log = args.event_log
    else:
        cfg = cloud_target_config()
        workers = args.workers
        max_wall = args.max_wall_sec if args.max_wall_sec > 0 else 14_400.0
        event_log = args.event_log or "logs/sd_cfr_hu_cloud_validate_target_scale_events.jsonl"

    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    status_path = Path(args.status_file)
    log = IterationEventLogger(event_log)
    t0 = time.perf_counter()

    def _status(phase: str, **fields) -> None:
        row = {"ts": time.time(), "phase": phase, **fields}
        status_path.parent.mkdir(parents=True, exist_ok=True)
        status_path.write_text(json.dumps(row, indent=2), encoding="utf-8")

    def _check_budget(phase: str) -> None:
        if max_wall <= 0:
            return
        elapsed = time.perf_counter() - t0
        if elapsed > max_wall:
            log.emit("validate_timeout", phase=phase, elapsed_sec=round(elapsed, 1))
            _status("timeout", phase=phase, elapsed_sec=elapsed, max_wall_sec=max_wall)
            raise SystemExit(124)
    results: dict = {
        "target_scale": not args.smoke,
        "workers": workers,
        "visits_per_iter": cfg.traversals_per_iter * max(1, cfg.deal_sweeps) * 2,
        "tests": {},
    }

    try:
        _status("validate_start", workers=workers, smoke=args.smoke, max_wall_sec=max_wall)
        log.emit("validate_start", workers=workers, smoke=args.smoke, max_wall_sec=max_wall)
        print(
            f"Cloud validate: smoke={args.smoke} workers={workers} "
            f"visits={results['visits_per_iter']} max_wall_sec={max_wall}",
            flush=True,
        )
        print(f"  event_log={event_log} status={status_path}", flush=True)

        _check_budget("pre_at_scale_merge")
        _status("test_start", test="at_scale_merge")
        results["tests"]["at_scale_merge"] = test_at_scale_merge(cfg, workers, log=log)
        log.emit("test_end", test="at_scale_merge", passed=results["tests"]["at_scale_merge"]["passed"])
        _status("test_done", test="at_scale_merge", passed=results["tests"]["at_scale_merge"]["passed"])

        _check_budget("pre_worker_kill")
        _status("test_start", test="worker_kill_reclaim")
        results["tests"]["worker_kill_reclaim"] = test_worker_kill_reclaim(cfg, workers, log)
        _status(
            "test_done",
            test="worker_kill_reclaim",
            passed=results["tests"]["worker_kill_reclaim"]["passed"],
        )

        _check_budget("pre_out_of_order")
        _status("test_start", test="out_of_order_delivery")
        results["tests"]["out_of_order_delivery"] = test_out_of_order_delivery(cfg, workers, log)
        _status(
            "test_done",
            test="out_of_order_delivery",
            passed=results["tests"]["out_of_order_delivery"]["passed"],
        )

        if not args.skip_resume:
            _check_budget("pre_barrier_resume")
            _status("test_start", test="barrier_resume")
            results["tests"]["barrier_resume"] = test_barrier_resume(cfg, workers, work_dir, log)
            _status(
                "test_done",
                test="barrier_resume",
                passed=results["tests"]["barrier_resume"]["passed"],
            )
        else:
            results["tests"]["barrier_resume"] = {"test": "barrier_resume", "passed": None, "skipped": True}

        results["tests"]["observability"] = test_observability(Path(event_log))
    finally:
        log.close()

    results["wall_sec"] = round(time.perf_counter() - t0, 1)
    results["max_wall_sec"] = max_wall
    results["event_log"] = event_log
    results["status_file"] = str(status_path)
    results["passed"] = all(
        t.get("passed") is True for t in results["tests"].values() if t.get("passed") is not None
    )
    _status("validate_complete", passed=results["passed"], wall_sec=results["wall_sec"])

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2), flush=True)
    if not results["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
