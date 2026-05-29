"""Parallel pool of persistent postflop_solver_cli workers."""

from __future__ import annotations

import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pokerbot.solver.postflop_solver_client import SolverSolveResult, build_solve_request
from pokerbot.solver.solver_worker import PersistentSolverWorker, WorkerError
from pokerbot.solver.spot_enumerator import SolverSpot


def default_num_workers() -> int:
    return max(1, (os.cpu_count() or 4) // 2)


def default_solver_threads_per_worker(num_workers: int) -> int:
    cores = os.cpu_count() or 4
    return max(1, cores // max(1, num_workers))


def default_max_in_flight(num_workers: int) -> int:
    return max(1, num_workers * 2)


@dataclass(frozen=True, slots=True)
class SpotSolveJob:
    spot_idx: int
    candidate_num: int
    spot_key: str
    spot: SolverSpot
    request: dict[str, Any]


@dataclass(slots=True)
class SpotSolveOutcome:
    spot_idx: int
    candidate_num: int
    spot_key: str
    spot: SolverSpot
    result: SolverSolveResult | None = None
    error: Exception | None = None
    error_kind: str | None = None
    elapsed_sec: float = 0.0


@dataclass(slots=True)
class PoolStats:
    num_workers: int
    max_in_flight: int
    solver_threads_per_worker: int
    jobs_submitted: int = 0
    jobs_completed: int = 0
    queue_wait_sec: float = 0.0


_thread_local = threading.local()


@dataclass
class SolverWorkerPool:
    num_workers: int
    cli_path: Path | None = None
    max_in_flight: int | None = None
    solver_threads_per_worker: int | None = None
    disable_worker_cache: bool = False
    worker_cache_size: int | None = None
    timeout_sec: float = 3600.0
    _executor: ThreadPoolExecutor = field(init=False, repr=False)
    _closed: bool = field(init=False, default=False, repr=False)
    stats: PoolStats = field(init=False)

    def __post_init__(self) -> None:
        if self.num_workers < 1:
            raise ValueError("num_workers must be >= 1")
        threads = self.solver_threads_per_worker
        if threads is None:
            threads = default_solver_threads_per_worker(self.num_workers)
        inflight = self.max_in_flight
        if inflight is None:
            inflight = default_max_in_flight(self.num_workers)
        self.solver_threads_per_worker = threads
        self.max_in_flight = inflight
        self.stats = PoolStats(
            num_workers=self.num_workers,
            max_in_flight=self.max_in_flight,
            solver_threads_per_worker=threads,
        )
        self._executor = ThreadPoolExecutor(
            max_workers=self.num_workers,
            thread_name_prefix="solver-worker",
        )

    def _worker_for_thread(self) -> PersistentSolverWorker:
        w = getattr(_thread_local, "worker", None)
        if w is None:
            w = PersistentSolverWorker(
                cli_path=self.cli_path,
                solver_threads_per_worker=self.solver_threads_per_worker,
                disable_worker_cache=self.disable_worker_cache,
                worker_cache_size=self.worker_cache_size,
            )
            _thread_local.worker = w
        return w

    def _run_job(self, job: SpotSolveJob) -> SpotSolveOutcome:
        t_queue = time.perf_counter()
        worker = self._worker_for_thread()
        queue_wait = time.perf_counter() - t_queue
        self.stats.queue_wait_sec += queue_wait
        t0 = time.perf_counter()
        try:
            result = worker.solve(
                job.request,
                request_id=str(job.spot_idx),
                timeout_sec=self.timeout_sec,
            )
            return SpotSolveOutcome(
                spot_idx=job.spot_idx,
                candidate_num=job.candidate_num,
                spot_key=job.spot_key,
                spot=job.spot,
                result=result,
                elapsed_sec=time.perf_counter() - t0,
            )
        except WorkerError as exc:
            return SpotSolveOutcome(
                spot_idx=job.spot_idx,
                candidate_num=job.candidate_num,
                spot_key=job.spot_key,
                spot=job.spot,
                error=exc,
                error_kind=type(exc).__name__,
                elapsed_sec=time.perf_counter() - t0,
            )
        except Exception as exc:
            return SpotSolveOutcome(
                spot_idx=job.spot_idx,
                candidate_num=job.candidate_num,
                spot_key=job.spot_key,
                spot=job.spot,
                error=exc,
                error_kind=type(exc).__name__,
                elapsed_sec=time.perf_counter() - t0,
            )

    def solve_jobs(self, jobs: list[SpotSolveJob]) -> dict[int, SpotSolveOutcome]:
        if self._closed:
            raise RuntimeError("pool already closed")
        outcomes: dict[int, SpotSolveOutcome] = {}
        if not jobs:
            return outcomes
        self.stats.jobs_submitted += len(jobs)
        futures = [self._executor.submit(self._run_job, job) for job in jobs]
        for fut in as_completed(futures):
            outcome = fut.result()
            outcomes[outcome.spot_idx] = outcome
            self.stats.jobs_completed += 1
        return outcomes

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._executor.shutdown(wait=True)
        w = getattr(_thread_local, "worker", None)
        if w is not None:
            w.close()
            _thread_local.worker = None

    def __enter__(self) -> SolverWorkerPool:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


def build_job_from_spot(
    spot_idx: int,
    candidate_num: int,
    spot_key: str,
    spot: SolverSpot,
    raw_cfg: dict[str, Any],
) -> SpotSolveJob:
    bet_tree = raw_cfg.get("bet_tree", {})
    solver_cfg = raw_cfg.get("solver", {})
    request = build_solve_request(
        spot,
        bet_fractions=[float(x) for x in bet_tree.get("pot_fractions", [0.33, 0.5, 0.75, 1.0])],
        raise_multiplier=float(bet_tree.get("raise_multiplier", 2.5)),
        max_iterations=int(solver_cfg.get("max_iterations", 10000)),
        target_exploitability_pct=float(solver_cfg.get("target_exploitability_pct", 0.5)),
        chip_unit_bb=float(solver_cfg.get("chip_unit_bb", 100)),
    )
    return SpotSolveJob(
        spot_idx=spot_idx,
        candidate_num=candidate_num,
        spot_key=spot_key,
        spot=spot,
        request=request,
    )
