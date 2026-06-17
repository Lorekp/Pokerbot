#!/usr/bin/env python3
"""Cloud label worker — S3 claim coordination + per-spot solve + shard upload."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pokerbot.solver.cloud_spot_solver import (
    CloudSpotSolver,
    board_timing_from_manifest,
    order_pending_for_worker,
)
from pokerbot.solver.label_completion import spot_key_hash
from pokerbot.solver.s3_coordination import ClaimResult, S3Coordinator

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BUCKET = os.environ.get("LABELGEN_S3_BUCKET", "video-processing-6789")
DEFAULT_PREFIX = os.environ.get("LABELGEN_S3_PREFIX", "pokerbot/prod6000")
DEFAULT_CONFIG = "config/solver_labels_prod6000.yaml"
TIMEOUT_SEC = 3600


def _solve_one_spot_subprocess(
    *,
    spot_key: str,
    config_path: Path,
    output_dir: Path,
    cli_path: str | None,
    solver_threads_per_worker: int | None,
) -> dict[str, Any]:
    """Legacy fallback: spawn build_solver_labels.py per spot."""
    manifest = {
        "run_id": "cloud_worker_single",
        "spot_keys": [spot_key],
        "entries": {spot_key: {"status": "pending"}},
    }
    manifest_path = output_dir / "single_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    ledger_path = output_dir / "completed.jsonl"
    shard_dir = output_dir / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(ROOT / "scripts/build_solver_labels.py"),
        "--config",
        str(config_path),
        "--work-manifest",
        str(manifest_path),
        "--output",
        str(output_dir / "train.parquet"),
        "--manifest",
        str(output_dir / "manifest.json"),
        "--report",
        str(output_dir / "run.md"),
        "--weva-index",
        str(ROOT / "data/weva"),
        "--solver-backend",
        "subprocess",
        "--num-workers",
        "1",
        "--resume",
        "--completion-ledger",
        str(ledger_path),
        "--shard-dir",
        str(shard_dir),
    ]
    if cli_path:
        cmd.extend(["--cli-path", cli_path])
    if solver_threads_per_worker is not None:
        cmd.extend(["--solver-threads-per-worker", str(solver_threads_per_worker)])

    t0 = time.time()
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    elapsed = time.time() - t0

    if proc.returncode != 0:
        return {
            "ok": False,
            "spot_key": spot_key,
            "elapsed_sec": elapsed,
            "stderr": proc.stderr[-2000:],
            "stdout": proc.stdout[-2000:],
        }

    manifest_out = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    per_spot = manifest_out.get("per_spot_results", [])
    meta: dict[str, Any] = {"ok": True, "spot_key": spot_key, "elapsed_sec": elapsed}
    if per_spot:
        row = per_spot[0]
        meta["exploitability_pct"] = row.get("exploitability_pct_of_pot")
        meta["iterations_used"] = row.get("iterations_used")
        meta["converged"] = row.get("converged", True)
        meta["near_timeout"] = elapsed >= 0.95 * TIMEOUT_SEC

    shard_path = shard_dir / f"{spot_key_hash(spot_key)}.parquet"
    if not shard_path.is_file():
        meta["ok"] = False
        meta["error"] = "missing_shard"
    else:
        meta["shard_path"] = str(shard_path)

    return meta


def main() -> None:
    p = argparse.ArgumentParser(description="S3-coordinated cloud label worker")
    p.add_argument("--bucket", default=DEFAULT_BUCKET)
    p.add_argument("--prefix", default=DEFAULT_PREFIX)
    p.add_argument("--worker-id", default=os.environ.get("HOSTNAME", "worker-1"))
    p.add_argument("--config", type=Path, default=Path(DEFAULT_CONFIG))
    p.add_argument("--max-spots", type=int, default=0, help="0 = until queue empty")
    p.add_argument("--idle-exit-sec", type=int, default=120)
    p.add_argument("--claim-ttl-sec", type=int, default=7200)
    p.add_argument("--cli-path", type=str, default=os.environ.get("POSTFLOP_SOLVER_CLI"))
    p.add_argument("--region", default=os.environ.get("AWS_DEFAULT_REGION", "us-east-1"))
    p.add_argument(
        "--solver-backend",
        choices=["worker", "subprocess"],
        default="worker",
        help="worker=persistent JSONL Rust process; subprocess=legacy per-spot spawn",
    )
    p.add_argument(
        "--solver-threads-per-worker",
        type=int,
        default=int(os.environ.get("RAYON_NUM_THREADS", "1")),
        help="RAYON_NUM_THREADS per solver process (default 1 for fleet)",
    )
    p.add_argument(
        "--weva-index",
        type=Path,
        default=Path("data/weva"),
        help="WEVA index directory for in-process encoding",
    )
    args = p.parse_args()

    coord = S3Coordinator(
        bucket=args.bucket,
        prefix=args.prefix,
        worker_id=args.worker_id,
        region=args.region,
        claim_ttl_sec=args.claim_ttl_sec,
    )

    work = coord.load_work_manifest()
    spot_keys: list[str] = list(work.get("spot_keys", []))
    board_timing = board_timing_from_manifest(work)
    slow_threshold = work.get("slow_board_threshold_sec")
    if slow_threshold is not None:
        slow_threshold = float(slow_threshold)
    completed = coord.refresh_completed(manifest_keys=spot_keys, force=True)

    pending = [k for k in spot_keys if k not in completed]
    print(
        json.dumps(
            {
                "worker_id": args.worker_id,
                "total_spots": len(spot_keys),
                "already_completed": len(completed),
                "pending": len(pending),
                "solver_backend": args.solver_backend,
                "solver_threads_per_worker": args.solver_threads_per_worker,
            }
        ),
        flush=True,
    )

    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    weva_index = args.weva_index if args.weva_index.is_absolute() else ROOT / args.weva_index
    work_dir = Path(tempfile.mkdtemp(prefix="labelgen_"))
    shard_dir = work_dir / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)

    processed = 0
    current_claim: str | None = None
    last_progress = time.time()
    poll_sleep_sec = 5
    spot_solver: CloudSpotSolver | None = None
    cli_path = Path(args.cli_path) if args.cli_path else None

    if args.solver_backend == "worker":
        spot_solver = CloudSpotSolver(
            config_path=config_path,
            weva_index=weva_index,
            shard_dir=shard_dir,
            cli_path=cli_path,
            solver_backend="worker",
            solver_threads_per_worker=args.solver_threads_per_worker,
            spot_keys=set(spot_keys),
            timeout_sec=TIMEOUT_SEC,
            board_timing=board_timing or None,
        )

    def _handle_interrupt(signum: int, _frame: Any) -> None:
        print(f"signal {signum}; releasing claim for {current_claim}", flush=True)
        coord.metrics.spot_interruptions += 1
        if spot_solver is not None:
            spot_solver.close()
        if current_claim:
            coord.release_claim(current_claim)
        coord.flush_metrics()
        sys.exit(128 + signum)

    signal.signal(signal.SIGTERM, _handle_interrupt)
    signal.signal(signal.SIGINT, _handle_interrupt)

    def _pending_spots() -> list[str]:
        coord.refresh_completed(manifest_keys=spot_keys, force=True)
        pending_keys = [k for k in spot_keys if not coord.is_completed(k)]
        return order_pending_for_worker(
            pending_keys,
            args.worker_id,
            board_timing=board_timing or None,
            slow_threshold_sec=slow_threshold,
        )

    try:
        while True:
            pending = _pending_spots()
            if not pending:
                break

            if args.max_spots and processed >= args.max_spots:
                if time.time() - last_progress >= args.idle_exit_sec:
                    break
                time.sleep(poll_sleep_sec)
                continue

            made_progress = False
            for spot_key in pending:
                if args.max_spots and processed >= args.max_spots:
                    break

                if coord.is_completed(spot_key):
                    continue

                claim: ClaimResult = coord.try_claim(spot_key)
                if not claim.claimed:
                    if claim.collision:
                        coord.log_collision_event(spot_key, claim.reason)
                    continue

                current_claim = spot_key
                print(f"solving {spot_key}", flush=True)

                if spot_solver is not None:
                    result = spot_solver.solve(spot_key, request_id=spot_key)
                else:
                    result = _solve_one_spot_subprocess(
                        spot_key=spot_key,
                        config_path=config_path,
                        output_dir=work_dir,
                        cli_path=args.cli_path,
                        solver_threads_per_worker=args.solver_threads_per_worker,
                    )

                if not result.get("ok"):
                    print(json.dumps({"error": result}), flush=True)
                    coord.release_claim(spot_key)
                    current_claim = None
                    continue

                shard_path = Path(result["shard_path"])
                coord.upload_shard(spot_key, shard_path)
                first_complete = coord.mark_completed(
                    spot_key,
                    exploitability_pct=result.get("exploitability_pct"),
                    iterations_used=result.get("iterations_used"),
                    elapsed_sec=result.get("elapsed_sec"),
                    near_timeout=result.get("near_timeout", False),
                )
                current_claim = None
                if first_complete:
                    processed += 1
                    made_progress = True
                    last_progress = time.time()
                coord.flush_metrics()
                print(
                    json.dumps(
                        {
                            "done": spot_key,
                            "processed": processed,
                            "first_complete": first_complete,
                            "exploit": result.get("exploitability_pct"),
                            "iters": result.get("iterations_used"),
                            "elapsed": result.get("elapsed_sec"),
                        }
                    ),
                    flush=True,
                )

            if not made_progress:
                if time.time() - last_progress >= args.idle_exit_sec:
                    break
                time.sleep(poll_sleep_sec)
    finally:
        if current_claim:
            coord.release_claim(current_claim)
            current_claim = None
        if spot_solver is not None:
            spot_solver.close()

    coord.flush_metrics()
    report = {
        "worker_id": args.worker_id,
        "processed": processed,
        "metrics": coord.metrics.to_dict(),
        "finished_at": datetime.now(timezone.utc).isoformat(),
    }
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
