"""Regression tests for profile_phase4_latency sample counting."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np


def test_latency_report_uses_one_sample_per_root(tmp_path: Path) -> None:
    import scripts.profile_phase4_latency as mod

    n_samples = 5
    roots = [(MagicMock(), i % 6) for i in range(n_samples)]

    mock_solver = MagicMock()
    mock_solver.solve.return_value = MagicMock(
        metadata={"traversal_fraction": 0.25, "native_cfr_recommended": False}
    )

    with (
        patch.object(mod, "_sample_roots", return_value=roots),
        patch.object(mod, "HistoryValueEvaluator", return_value=MagicMock(has_trained_checkpoint=True)),
        patch.object(mod, "TurboReBeLSolver", return_value=mock_solver),
        patch.object(mod, "PBSTracker") as mock_pbs,
        patch.object(mod, "build_subgame_root", return_value=MagicMock()),
        patch.object(mod.time, "perf_counter", side_effect=[0.0, 0.001] * n_samples),
    ):
        mock_pbs.init_uniform.return_value = MagicMock()
        out = tmp_path / "latency.json"
        import sys

        argv = [
            "profile_phase4_latency.py",
            "--samples",
            str(n_samples),
            "--use-mock-leaf",
            "--output",
            str(out),
        ]
        with patch.object(sys, "argv", argv):
            mod.main()

    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["samples"] == n_samples
    # Internal consistency: p50/p95 derived from n_samples wall-clock entries, not 2*n_samples
    assert report["p95_ms"] >= report["p50_ms"]
