"""Golden and differential tests for non-AoF postflop solver labels."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from pokerbot.encoding.postflop_solver_encoder import record_to_parquet_dict
from pokerbot.encoding.weva_encoder import WEVAMapper
from pokerbot.solver.label_diff import (
    DiffMode,
    canonicalize_action_distribution,
    canonicalize_manifest,
    compare_label_dirs,
    compare_manifests,
    compare_parquet_rows,
    row_sort_key,
)
from pokerbot.solver.postflop_solver_client import SolverSolveResult, parse_solve_response
from pokerbot.solver.spot_enumerator import enumerate_spots, load_solver_config

FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures" / "non_aof_labels"
RESPONSES_DIR = FIXTURE_DIR / "solver_responses"
GOLDEN_CONFIG = FIXTURE_DIR / "golden_config.yaml"
WEVA_INDEX = Path(__file__).resolve().parents[1] / "data" / "weva"


def _load_fixture_response(name: str) -> dict:
    return json.loads((RESPONSES_DIR / name).read_text(encoding="utf-8"))


def _solver_result_from_fixture(fixture: dict) -> SolverSolveResult:
    return parse_solve_response(fixture["solver_response"])


def _enumerate_first_spot():
    config = load_solver_config(GOLDEN_CONFIG)
    spots = list(enumerate_spots(config))
    assert spots, "golden_config produced no spots"
    return spots[0]


@pytest.mark.skipif(not WEVA_INDEX.is_dir(), reason="WEVA index required")
def test_golden_fixture_encode_produces_valid_row() -> None:
    fixture = _load_fixture_response("sb_vs_bb_flop_Td9d6h_hero4.json")
    spot = _enumerate_first_spot()
    result = _solver_result_from_fixture(fixture)
    mapper = WEVAMapper.from_directory(WEVA_INDEX)
    from pokerbot.encoding.postflop_solver_encoder import encode_solver_spot_row

    record = encode_solver_spot_row(
        spot,
        result,
        mapper=mapper,
        bet_sizes=fixture["bet_sizes"],
        raise_sizes=fixture["raise_sizes"],
        target_exploitability_pct=fixture["target_exploitability_pct"],
    )
    row = record_to_parquet_dict(record)
    assert row["label_source"] == "postflop_solver"
    assert row["solver_converged"] is True
    assert float(np.sum(row["solver_policy"])) == pytest.approx(1.0, abs=1e-5)
    assert row["oop_range"] == fixture["oop_range"]
    assert row["ip_range"] == fixture["ip_range"]


def test_action_distribution_order_independent_semantic_diff() -> None:
    dist_a = {"check": 0.6, "bet_0.50": 0.25, "all_in": 0.15}
    dist_b = {"all_in": 0.15, "check": 0.6, "bet_0.50": 0.25}
    assert canonicalize_action_distribution(dist_a) == canonicalize_action_distribution(dist_b)

    row_a = {
        "raw_solver_json": json.dumps(
            {"spot_id": "s1", "solver_request": {"board": ["Td", "9d", "6h"]}, "hero_hole": ["Ah", "Kd"], "action_distribution": dist_a}
        ),
        "oop_range": "QQ+,AK",
        "ip_range": "TT-22,AQs+",
        "solver_policy": [0.0] * 11,
        "solver_label": 0,
        "label_source": "postflop_solver",
    }
    row_b = dict(row_a)
    row_b["raw_solver_json"] = json.dumps(
        {
            "spot_id": "s1",
            "solver_request": {"board": ["Td", "9d", "6h"]},
            "hero_hole": ["Ah", "Kd"],
            "action_distribution": dist_b,
        }
    )
    result = compare_parquet_rows([row_a], [row_b], DiffMode.SEMANTIC)
    assert result.passed, result.issues


def test_manifest_timing_ignored_in_semantic_diff() -> None:
    base = {
        "config_hash": "abc",
        "enumeration": {"included": 1, "total_candidates": 1},
        "convergence": {"median_solve_time_sec": 90.0, "max_solve_time_sec": 95.0},
        "per_spot_results": [{"spot": "a|b|c", "converged": True, "elapsed_sec": 95.0}],
        "generated_at": "2026-01-01T00:00:00+00:00",
        "cli_path": "/old/path",
    }
    variant = {
        **base,
        "generated_at": "2026-06-01T00:00:00+00:00",
        "cli_path": "/new/path",
        "convergence": {"median_solve_time_sec": 120.0, "max_solve_time_sec": 130.0},
        "per_spot_results": [{"spot": "a|b|c", "converged": True, "elapsed_sec": 120.0}],
    }
    result = compare_manifests(base, variant, DiffMode.SEMANTIC)
    assert result.passed, result.issues


def test_strict_manifest_diff_fails_on_timestamp() -> None:
    base = {"generated_at": "a", "seed": 42}
    variant = {"generated_at": "b", "seed": 42}
    result = compare_manifests(base, variant, DiffMode.STRICT)
    assert not result.passed


def test_not_converged_fixture_parsed() -> None:
    fixture = _load_fixture_response("not_converged.json")
    result = _solver_result_from_fixture(fixture)
    assert result.converged is False
    assert result.iterations_used == 10000


def test_golden_config_enumerates_expected_spot_count() -> None:
    config = load_solver_config(GOLDEN_CONFIG)
    spots = list(enumerate_spots(config))
    # 3 spots × 3 boards × 2 hero seats = 18
    assert len(spots) == 18


def test_row_sort_key_stable() -> None:
    row = {
        "raw_solver_json": json.dumps(
            {
                "spot_id": "sb_vs_bb_flop",
                "solver_request": {"board": ["Td", "9d", "6h"]},
                "hero_hole": ["Ah", "Kd"],
                "hero_position": "SB",
            }
        ),
        "oop_range": "QQ+,AK",
        "ip_range": "TT-22,AQs+",
    }
    assert row_sort_key(row) == row_sort_key(dict(row))


def test_compare_label_dirs_missing_parquet(tmp_path: Path) -> None:
    exp = tmp_path / "expected"
    act = tmp_path / "actual"
    exp.mkdir()
    act.mkdir()
    manifest = {"enumeration": {"included": 0, "total_candidates": 0}, "seed": 42}
    exp.joinpath("manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    act.joinpath("manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    result = compare_label_dirs(exp, act, DiffMode.SEMANTIC)
    assert result.passed


@pytest.mark.slow
@pytest.mark.skipif(not WEVA_INDEX.is_dir(), reason="WEVA index required")
def test_full_pipeline_single_spot_golden_semantic(tmp_path: Path) -> None:
    """Slow integration: run subprocess backend on 1 spot and self-diff."""
    import subprocess
    import sys

    root = Path(__file__).resolve().parents[1]
    out = tmp_path / "run_a"
    cmd = [
        sys.executable,
        str(root / "scripts" / "build_solver_labels.py"),
        "--config",
        str(GOLDEN_CONFIG),
        "--max-spots",
        "1",
        "--output",
        str(out / "train.parquet"),
        "--manifest",
        str(out / "manifest.json"),
        "--report",
        str(out / "report.md"),
        "--weva-index",
        str(WEVA_INDEX),
    ]
    subprocess.run(cmd, cwd=root, check=True, timeout=600)

    out_b = tmp_path / "run_b"
    out_b.mkdir()
    out_b.joinpath("manifest.json").write_text(
        out.joinpath("manifest.json").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    out_b.joinpath("train.parquet").write_bytes(out.joinpath("train.parquet").read_bytes())

    result = compare_label_dirs(out, out_b, DiffMode.SEMANTIC)
    assert result.passed, result.issues

    canon_a = canonicalize_manifest(
        json.loads(out.joinpath("manifest.json").read_text(encoding="utf-8")),
        DiffMode.SEMANTIC,
    )
    canon_b = canonicalize_manifest(
        json.loads(out_b.joinpath("manifest.json").read_text(encoding="utf-8")),
        DiffMode.SEMANTIC,
    )
    assert canon_a == canon_b
