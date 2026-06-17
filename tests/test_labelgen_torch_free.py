"""Verify label-generation runs without torch installed (real absence, not mocked)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import venv
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
REQUIREMENTS = ROOT / "infra/labelgen/requirements.txt"
SOLVER_CANDIDATES = (
    ROOT / "external/postflop_solver_cli/target/release/postflop_solver_cli",
    ROOT / "external/postflop_solver_cli/target/release/postflop_solver_cli.exe",
)


def _solver_cli() -> Path | None:
    env = os.environ.get("POSTFLOP_SOLVER_CLI")
    if env and Path(env).is_file():
        return Path(env)
    for path in SOLVER_CANDIDATES:
        if path.is_file():
            return path
    return None


def _run_torch_free_labelgen(*, max_spots: int, output_dir: Path) -> subprocess.CompletedProcess[str]:
    cli = _solver_cli()
    if cli is None:
        pytest.skip("postflop_solver_cli not built locally")

    venv_dir = output_dir / "venv"
    venv.create(venv_dir, with_pip=True)
    py = venv_dir / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")

    subprocess.run(
        [str(py), "-m", "pip", "install", "-r", str(REQUIREMENTS)],
        check=True,
        capture_output=True,
        text=True,
    )
    torch_check = subprocess.run(
        [str(py), "-c", "import torch"],
        capture_output=True,
        text=True,
    )
    assert torch_check.returncode != 0, "torch must not be installed in labelgen venv"

    manifest_path = output_dir / "work_manifest.json"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/plan_label_work.py"),
            "--config",
            "config/solver_labels_validate50.yaml",
            "--output",
            str(manifest_path),
            "--max-spots",
            str(max_spots),
            "--run-id",
            "torch_free_smoke",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)
    env["POSTFLOP_SOLVER_CLI"] = str(cli)

    return subprocess.run(
        [
            str(py),
            str(ROOT / "scripts/build_solver_labels.py"),
            "--config",
            "config/solver_labels_validate50.yaml",
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
            "--shard-dir",
            str(output_dir / "shards"),
        ],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
    )


@pytest.mark.slow
def test_weva_mapper_imports_without_torch() -> None:
    """WEVAMapper.from_directory must not require torch at import or lookup time."""
    cli = _solver_cli()
    if cli is None:
        pytest.skip("postflop_solver_cli not built locally")

    venv_dir = Path(tempfile.mkdtemp()) / "venv"
    venv.create(venv_dir, with_pip=True)
    py = venv_dir / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    subprocess.run([str(py), "-m", "pip", "install", "-r", str(REQUIREMENTS)], check=True)
    env = {"PYTHONPATH": str(ROOT)}
    r = subprocess.run(
        [
            str(py),
            "-c",
            "from pokerbot.encoding.weva_encoder import WEVAMapper; "
            "from pathlib import Path; "
            "from pokerbot.engine.cards import Card, Rank, Suit; "
            "m = WEVAMapper.from_directory(Path('data/weva')); "
            "bid = m.lookup((Card(Rank(14), Suit(0)), Card(Rank(13), Suit(1)))); "
            "print(bid)",
        ],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, r.stderr
    assert "torch" not in (r.stdout + r.stderr).lower()


@pytest.mark.slow
def test_build_solver_labels_torch_free_multi_spot(tmp_path: Path) -> None:
    """Full build_solver_labels path on diverse spots with torch genuinely absent."""
    output_dir = tmp_path / "labelgen_smoke"
    output_dir.mkdir()
    (output_dir / "shards").mkdir()

    result = _run_torch_free_labelgen(max_spots=5, output_dir=output_dir)
    combined = (result.stdout or "") + (result.stderr or "")
    assert "ImportError" not in combined
    assert "ModuleNotFoundError" not in combined
    assert "torch" not in combined.lower() or "no module named 'torch'" in combined.lower()
    assert result.returncode == 0, combined[-4000:]

    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    spot_types = set()
    boards = set()
    for row in manifest.get("per_spot_results", []):
        spot = str(row.get("spot", ""))
        if "|" in spot:
            spot_types.add(spot.split("|", 1)[0])
            parts = spot.split("|")
            if len(parts) > 1:
                boards.add(parts[1])
    assert len(manifest.get("per_spot_results", [])) >= 5
    assert "bunching_postflop" in spot_types
    assert len(boards) >= 3
