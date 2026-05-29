"""Static range audit schema tests."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_static_range_audit_json_schema() -> None:
    root = Path(__file__).resolve().parents[1]
    out = root / "artifacts" / "_test_static_range_audit.json"
    subprocess.run(
        [sys.executable, str(root / "scripts/aof_static_range_audit.py"), "--json-out", str(out)],
        cwd=root,
        check=True,
    )
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["schema_version"] == 1
    assert data["static_profile_only"] is True
    assert data["not_solver_backed_34max"] is True
    assert "by_table_size" in data


def test_default_shove_keys_present() -> None:
    from aof_ranges import SHOVE_RANGE_STR

    assert 8 in SHOVE_RANGE_STR
    assert "BTN" in SHOVE_RANGE_STR[8]
