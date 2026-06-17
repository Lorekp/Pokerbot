#!/usr/bin/env python3
"""Gate: distributed merge == single-threaded trainer (ground truth)."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from pokerbot.training.sd_cfr_nlhe import SDCFRNLHEConfig, SDCFRNLHETrainer
from pokerbot.training.sd_cfr_nlhe_distributed import (
    HU_CLOUD_TARGET_DEAL_SWEEPS,
    HU_CLOUD_TARGET_TRAVERSALS_PER_ITER,
    HU_CLOUD_TARGET_VISITS_PER_ITER,
    HU_CLOUD_TARGET_WORKERS,
    IterationEventLogger,
    cloud_target_config,
    compare_trainer_states,
    train_iteration_sharded,
)

ROOT = Path(__file__).resolve().parents[1]

# Local wall-clock budget: ~2× single-threaded extrap (~137 min/iter) capped for debug loop.
DEFAULT_TARGET_MAX_WALL_SEC = 10_800  # 3h total; kill if exceeded


class _TimeoutError(Exception):
    pass


def _check_wall_budget(t0: float, max_wall_sec: float, phase: str) -> None:
    if max_wall_sec <= 0:
        return
    elapsed = time.perf_counter() - t0
    if elapsed > max_wall_sec:
        raise _TimeoutError(
            f"{phase} exceeded max_wall_sec={max_wall_sec} (elapsed={elapsed:.0f}s). "
            "Kill run — too slow for local gate; use cloud coordinator for target-scale."
        )


def _write_status(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload["ts"] = time.time()
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--iters", type=int, default=1)
    p.add_argument("--traversals", type=int, default=None)
    p.add_argument("--deal-sweeps", type=int, default=None)
    p.add_argument("--workers", type=int, default=None)
    p.add_argument(
        "--target-scale",
        action="store_true",
        help=(
            f"Cloud convergence scale: {HU_CLOUD_TARGET_TRAVERSALS_PER_ITER} trav × "
            f"{HU_CLOUD_TARGET_DEAL_SWEEPS} sweeps × 2 seats = {HU_CLOUD_TARGET_VISITS_PER_ITER} visits, "
            f"{HU_CLOUD_TARGET_WORKERS} workers"
        ),
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", default=None)
    p.add_argument(
        "--max-wall-sec",
        type=float,
        default=None,
        help="Abort with exit 124 if total wall time exceeds this (0=disabled)",
    )
    p.add_argument(
        "--event-log",
        default=None,
        help="JSONL progress log (default: logs/sd_cfr_nlhe_merge_gate_events.jsonl)",
    )
    p.add_argument(
        "--status-file",
        default=None,
        help="Heartbeat JSON (default: logs/sd_cfr_nlhe_merge_gate_status.json)",
    )
    args = p.parse_args()

    if args.target_scale:
        traversals = HU_CLOUD_TARGET_TRAVERSALS_PER_ITER
        sweeps = HU_CLOUD_TARGET_DEAL_SWEEPS
        workers = HU_CLOUD_TARGET_WORKERS
        out_default = "logs/sd_cfr_nlhe_merge_gate_target_scale.json"
        max_wall = (
            args.max_wall_sec
            if args.max_wall_sec is not None
            else DEFAULT_TARGET_MAX_WALL_SEC
        )
        event_log = args.event_log or "logs/sd_cfr_nlhe_merge_gate_target_scale.jsonl"
        status_file = args.status_file or "logs/sd_cfr_nlhe_merge_gate_target_scale_status.json"
    else:
        traversals = args.traversals if args.traversals is not None else 2
        sweeps = args.deal_sweeps if args.deal_sweeps is not None else 1
        workers = args.workers if args.workers is not None else 2
        out_default = "logs/sd_cfr_nlhe_merge_gate.json"
        max_wall = args.max_wall_sec if args.max_wall_sec is not None else 0.0
        event_log = args.event_log or "logs/sd_cfr_nlhe_merge_gate_events.jsonl"
        status_file = args.status_file or "logs/sd_cfr_nlhe_merge_gate_status.json"

    output = args.output or out_default
    visits = traversals * sweeps * 2

    cfg = cloud_target_config(
        traversals_per_iter=traversals,
        deal_sweeps=sweeps,
        seed=args.seed,
    )
    ref = SDCFRNLHETrainer(cfg)
    dist = SDCFRNLHETrainer(cfg)
    logger = IterationEventLogger(event_log)
    reports: list[dict] = []
    passed = True
    t0 = time.perf_counter()
    status_path = Path(status_file)

    print(
        f"Merge gate: {args.iters} iter(s), {visits} visits/iter, {workers} workers, "
        f"target_scale={args.target_scale}, max_wall_sec={max_wall}",
        flush=True,
    )
    print(f"  event_log={event_log}", flush=True)
    print(f"  status_file={status_path}", flush=True)

    try:
        for i in range(args.iters):
            w = float(i + 1)
            iter_t0 = time.perf_counter()
            _write_status(
                status_path,
                {
                    "phase": "iter_start",
                    "iter": i + 1,
                    "visits_per_iter": visits,
                    "workers": workers,
                    "max_wall_sec": max_wall,
                },
            )
            _check_wall_budget(t0, max_wall, "pre_single_threaded")

            np.random.seed(cfg.seed + i)
            logger.emit("iter_start", iter=i + 1, arm="single_threaded")
            print(f"  iter {i + 1}: single-threaded traversal+train ...", flush=True)
            loss_ref = ref.train_iteration(w, traversal_logger=logger)
            ref_wall = round(time.perf_counter() - iter_t0, 1)
            print(
                f"  iter {i + 1}: single-threaded done loss={loss_ref:.6f} wall={ref_wall}s",
                flush=True,
            )
            _write_status(
                status_path,
                {
                    "phase": "single_threaded_done",
                    "iter": i + 1,
                    "loss_ref": loss_ref,
                    "wall_sec": ref_wall,
                },
            )
            _check_wall_budget(t0, max_wall, "pre_sharded")

            np.random.seed(cfg.seed + i)
            shard_t0 = time.perf_counter()
            logger.emit("iter_start", iter=i + 1, arm="sharded", workers=workers)
            print(f"  iter {i + 1}: sharded ({workers} workers) traversal+train ...", flush=True)
            loss_dist = train_iteration_sharded(dist, w, num_shards=workers, logger=logger)
            shard_wall = round(time.perf_counter() - shard_t0, 1)
            print(
                f"  iter {i + 1}: sharded done loss={loss_dist:.6f} wall={shard_wall}s",
                flush=True,
            )

            cmp = compare_trainer_states(ref, dist)
            cmp["iter"] = i + 1
            cmp["loss_ref"] = loss_ref
            cmp["loss_dist"] = loss_dist
            cmp["loss_match"] = abs(loss_ref - loss_dist) < 1e-5
            cmp["wall_sec"] = round(time.perf_counter() - iter_t0, 1)
            cmp["wall_sec_single"] = ref_wall
            cmp["wall_sec_sharded"] = shard_wall
            reports.append(cmp)
            ok = (
                cmp["params_match"]
                and cmp["cum_regret_match"]
                and cmp["strat_sum_match"]
                and cmp["loss_match"]
            )
            passed = passed and ok
            logger.emit(
                "iter_compare",
                iter=i + 1,
                passed=ok,
                bit_identical_regret=cmp.get("cum_regret_bit_identical"),
                max_regret_diff=cmp["max_cum_regret_diff"],
            )
            print(
                f"  iter {i + 1}: passed={ok} bit_identical_regret={cmp.get('cum_regret_bit_identical')} "
                f"max_regret_diff={cmp['max_cum_regret_diff']} loss={loss_ref:.6f} "
                f"wall={cmp['wall_sec']}s",
                flush=True,
            )
            _write_status(
                status_path,
                {
                    "phase": "iter_done",
                    "iter": i + 1,
                    "passed": ok,
                    "compare": cmp,
                },
            )
            _check_wall_budget(t0, max_wall, "post_iter")
    except _TimeoutError as e:
        logger.emit("gate_timeout", error=str(e))
        _write_status(
            status_path,
            {"phase": "timeout", "passed": False, "error": str(e)},
        )
        print(str(e), flush=True)
        logger.close()
        raise SystemExit(124) from e
    finally:
        logger.close()

    out = {
        "gate": "distributed_merge_equals_single_threaded",
        "comparison_target": "single_threaded_trainer_not_inter_worker",
        "target_scale": args.target_scale,
        "passed": passed,
        "wall_sec_total": round(time.perf_counter() - t0, 1),
        "max_wall_sec": max_wall,
        "event_log": event_log,
        "status_file": str(status_path),
        "config": {
            "iters": args.iters,
            "traversals_per_iter": traversals,
            "deal_sweeps": sweeps,
            "visits_per_iter": visits,
            "workers": workers,
            "seed": args.seed,
            "max_traverse_depth": cfg.max_traverse_depth,
            "train_batch_size": cfg.train_batch_size,
        },
        "iter_reports": reports,
    }
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    _write_status(status_path, {"phase": "gate_complete", "passed": passed, "result": out})
    print(json.dumps(out, indent=2), flush=True)
    print(f"Wrote {path}", flush=True)
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
