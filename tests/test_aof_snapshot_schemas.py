"""Priority 7: snapshot JSON schema checks (fast, no slow MC)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from aof_validation.snapshot_schema import (
    check_call_behind_snapshot,
    check_multiway_equity_snapshot,
    check_open_multiway_snapshot,
    check_range_sensitivity_snapshot,
    check_teacher_mismatch_report,
)

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
ARTIFACTS = ROOT / "artifacts" / "pytest_snapshots"


@pytest.fixture(scope="module")
def snapshot_dir(tmp_path_factory) -> Path:
    d = tmp_path_factory.mktemp("aof_snapshots")
    return d


def _run(script: str, out: Path, extra: list[str] | None = None) -> None:
    cmd = [sys.executable, str(SCRIPTS / script), "-o", str(out)]
    if extra:
        cmd.extend(extra)
    subprocess.run(cmd, cwd=ROOT, check=True, timeout=120)


def test_schema_helpers_minimal() -> None:
    check_open_multiway_snapshot(
        {
            "schema_version": 1,
            "spots": [
                {
                    "spot": "AKo UTG 8bb",
                    "new_action": "ALL IN",
                    "new_ev": 0.1,
                    "legacy_shortcut_ev": 0.0,
                    "delta_bb": 0.1,
                    "action_flip": False,
                    "edge_bb": 0.1,
                }
            ],
        }
    )
    check_teacher_mismatch_report(
        {
            "schema_version": 1,
            "obs_version": 1,
            "teacher_version": 1,
            "num_hands": 1,
            "spot_kind_counts": {"open": 1},
            "teacher_path_counts": {"static_profile_open": 1},
            "terminal_counts": {},
        }
    )


@pytest.mark.slow
def test_multiway_equity_snapshot_schema(snapshot_dir: Path) -> None:
    out = snapshot_dir / "multiway_equity.json"
    _run("aof_multiway_equity_snapshot.py", out)
    data = json.loads(out.read_text(encoding="utf-8"))
    check_multiway_equity_snapshot(data)
    for spot in data["spots"]:
        if spot.get("equity_method") == "exact":
            assert spot["abs_delta"] < 0.08


@pytest.mark.slow
def test_open_multiway_snapshot_schema(snapshot_dir: Path) -> None:
    out = snapshot_dir / "open_multiway.json"
    _run("aof_open_multiway_snapshot.py", out)
    data = json.loads(out.read_text(encoding="utf-8"))
    check_open_multiway_snapshot(data)
    assert any("UTG" in s["spot"] for s in data["spots"])


@pytest.mark.slow
def test_call_behind_snapshot_schema(snapshot_dir: Path) -> None:
    out = snapshot_dir / "call_behind.json"
    _run("aof_call_behind_snapshot.py", out)
    data = json.loads(out.read_text(encoding="utf-8"))
    check_call_behind_snapshot(data)
    assert any(abs(s["delta_bb"]) > 1e-6 for s in data["spots"])


@pytest.mark.slow
def test_range_sensitivity_snapshot_schema(snapshot_dir: Path) -> None:
    out = snapshot_dir / "range_sensitivity.json"
    _run("aof_range_sensitivity_snapshot.py", out)
    data = json.loads(out.read_text(encoding="utf-8"))
    check_range_sensitivity_snapshot(data)


@pytest.mark.slow
def test_teacher_mismatch_report_schema(snapshot_dir: Path) -> None:
    out = snapshot_dir / "teacher_mismatch.json"
    _run(
        "aof_rl_teacher_mismatch_report.py",
        out,
        ["--hands", "10", "--teacher-version", "1", "--teacher-samples", "50"],
    )
    data = json.loads(out.read_text(encoding="utf-8"))
    check_teacher_mismatch_report(data)
    assert "spot_kind_counts" in data


@pytest.mark.slow
def test_teacher_mismatch_v2_schema(snapshot_dir: Path) -> None:
    out = snapshot_dir / "teacher_mismatch_v2.json"
    _run(
        "aof_rl_teacher_mismatch_report.py",
        out,
        ["--hands", "10", "--teacher-version", "2", "--teacher-samples", "50"],
    )
    data = json.loads(out.read_text(encoding="utf-8"))
    check_teacher_mismatch_report(data)


def test_production_no_legacy_open_import() -> None:
    text = (ROOT / "aof.py").read_text(encoding="utf-8")
    assert "legacy_max_1v1" not in text
    assert "max_1v1" not in text.lower() or "multiway" in text
