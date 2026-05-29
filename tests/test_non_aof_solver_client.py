"""Unit tests for postflop_solver_client request/response parsing."""

from __future__ import annotations

import json
from pathlib import Path

from pokerbot.solver.postflop_solver_client import (
    build_solve_request,
    parse_solve_response,
)
from pokerbot.solver.spot_enumerator import enumerate_spots, load_solver_config

FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures" / "non_aof_labels"
GOLDEN_CONFIG = FIXTURE_DIR / "golden_config.yaml"
CLI_TEST_REQUEST = (
    Path(__file__).resolve().parents[1]
    / "external"
    / "postflop_solver_cli"
    / "test_request.json"
)


def test_build_solve_request_fields() -> None:
    config = load_solver_config(GOLDEN_CONFIG)
    spot = next(iter(enumerate_spots(config)))
    req = build_solve_request(
        spot,
        bet_fractions=[0.5],
        raise_multiplier=2.5,
        max_iterations=10000,
        target_exploitability_pct=0.5,
        chip_unit_bb=100.0,
    )
    assert req["oop_range"] == spot.oop_range
    assert req["ip_range"] == spot.ip_range
    assert req["board"] == [c.to_str() for c in spot.board_cards]
    assert req["bet_sizes"] == "50%,a"
    assert req["raise_sizes"] == "2.5x"
    assert req["max_iterations"] == 10000


def test_parse_solve_response_roundtrip_fixture() -> None:
    data = json.loads(
        (FIXTURE_DIR / "solver_responses" / "sb_vs_bb_flop_Td9d6h_hero4.json").read_text(
            encoding="utf-8"
        )
    )["solver_response"]
    result = parse_solve_response(data)
    assert result.converged is True
    assert result.available_actions == ["check", "bet_0.50", "all_in"]
    assert result.action_distribution["check"] == data["action_distribution"]["check"]
    assert result.error is None


def test_parse_solve_response_error_field() -> None:
    result = parse_solve_response({"error": "range parse: bad", "converged": False})
    assert result.error == "range parse: bad"
    assert result.converged is False


def test_cli_test_request_json_valid() -> None:
    assert CLI_TEST_REQUEST.is_file()
    data = json.loads(CLI_TEST_REQUEST.read_text(encoding="utf-8"))
    assert "oop_range" in data
    assert "board" in data
