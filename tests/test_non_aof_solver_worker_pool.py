"""Tests for SolverWorkerPool ordering and scheduling."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from pokerbot.solver.postflop_solver_client import SolverSolveResult
from pokerbot.solver.solver_worker_pool import SpotSolveJob, SolverWorkerPool


def _fake_result() -> SolverSolveResult:
    return SolverSolveResult(
        exploitability=0.01,
        target_exploitability=0.01,
        exploitability_pct_of_pot=0.5,
        converged=True,
        iterations=100,
        iterations_used=50,
        available_actions=["check"],
        action_distribution={"check": 1.0},
        solver_version="test",
    )


def test_pool_out_of_order_completion_preserves_index_map() -> None:
    jobs = [
        SpotSolveJob(0, 1, "a", MagicMock(), {"board": ["Td", "9d", "6h"]}),
        SpotSolveJob(1, 2, "b", MagicMock(), {"board": ["Ah", "7c", "2s"]}),
        SpotSolveJob(2, 3, "c", MagicMock(), {"board": ["Ks", "8h", "3d"]}),
    ]

    def fake_run(job: SpotSolveJob):
        from pokerbot.solver.solver_worker_pool import SpotSolveOutcome

        # Simulate reversed completion order via sleep-free idx ordering
        return SpotSolveOutcome(
            spot_idx=job.spot_idx,
            candidate_num=job.candidate_num,
            spot_key=job.spot_key,
            spot=job.spot,
            result=_fake_result(),
            elapsed_sec=float(3 - job.spot_idx),
        )

    pool = SolverWorkerPool(num_workers=2, cli_path=None)
    with patch.object(pool, "_run_job", side_effect=fake_run):
        outcomes = pool.solve_jobs(jobs)
    pool.close()

    assert list(sorted(outcomes.keys())) == [0, 1, 2]
    ordered = [outcomes[i].spot_key for i in sorted(outcomes.keys())]
    assert ordered == ["a", "b", "c"]
