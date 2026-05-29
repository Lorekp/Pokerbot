"""Independence vs sequential open-EV report."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "data" / "aof" / "independence_report.json"


def test_independence_report_exists() -> None:
    if not REPORT.is_file():
        pytest.skip("run scripts/measure_aof_independence.py first")
    data = json.loads(REPORT.read_text(encoding="utf-8"))
    assert "max_abs_delta_bb" in data
    assert data["spots"]
    assert data["max_abs_delta_bb"] < 2.0
