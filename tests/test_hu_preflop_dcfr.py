"""HU push/fold DCFR solver smoke tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("pokerkit")
pytest.importorskip("phevaluator")

from aof_equity import warmup
from pokerbot.solver.hu_preflop_dcfr import load_hu_solution, solve_hu_pushfold

ROOT = Path(__file__).resolve().parents[1]
HU_JSON = ROOT / "data" / "aof" / "hu_pushfold_8bb.json"


@pytest.fixture(scope="module", autouse=True)
def _warm() -> None:
    warmup()


@pytest.mark.slow
def test_solver_exploitability_small() -> None:
    r = solve_hu_pushfold(
        iterations=400,
        equity_samples=300,
        bucket_stride=8,
        seed=1,
    )
    assert r.exploitability_bb < 0.15
    assert r.call_range_str
    assert r.shove_range_str


@pytest.mark.slow
def test_aa_pushes_more_than_weakest_bucket() -> None:
    r = solve_hu_pushfold(
        iterations=500,
        equity_samples=300,
        bucket_stride=4,
        seed=2,
    )
    aa = r.bucket_labels.index("AA")
    weakest = int(np.argmin(r.sb_push_freq))
    assert r.sb_push_freq[aa] > r.sb_push_freq[weakest] + 0.15


def test_hu_manifest_round_trip() -> None:
    if not HU_JSON.is_file():
        pytest.skip("run scripts/build_aof_hu_ranges.py")
    data = load_hu_solution(HU_JSON)
    manifest = data["manifest"]
    assert manifest["exploitability_bb"] < 0.1
    assert manifest.get("stack_bb") == 8.0
    labels = data["bucket_labels"]
    push = np.asarray(data["sb_push_freq"], dtype=np.float64)
    aa = labels.index("AA")
    weakest = int(np.argmin(push))
    assert push[aa] > push[weakest]
    assert data["call_range_str"]
    assert data["shove_range_str"]
