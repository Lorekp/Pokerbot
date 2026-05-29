"""Tests for interleaved spot enumeration order."""

from __future__ import annotations

from pathlib import Path

from pokerbot.solver.spot_enumerator import (
    SolverSpotConfig,
    enumerate_spot_keys,
    load_solver_config,
    spot_key,
)
from pokerbot.solver.spot_enumerator import enumerate_spots


def _spot_type(key: str) -> str:
    return key.split("|", 1)[0]


def test_interleaved_smoke_first_12_has_all_spot_types():
    config = load_solver_config(Path("config/solver_labels_smoke.yaml"))
    assert config.enumeration_mode == "interleaved"
    keys = enumerate_spot_keys(config)
    # 3 boards × 1 stack × 3 spot types × 2 heroes = 18
    assert len(keys) == 18
    first12_types = {_spot_type(k) for k in keys[:12]}
    assert first12_types == {"sb_vs_bb_flop", "btn_vs_bb_flop", "bunching_postflop"}


def test_interleaved_per_board_unit_has_2_to_1_ratio():
    config = load_solver_config(Path("config/solver_labels_smoke.yaml"))
    keys = enumerate_spot_keys(config)
    # First board unit: 6 keys (sb×2 + btn×2 + bunch×2)
    unit = keys[:6]
    types = [_spot_type(k) for k in unit]
    assert types.count("sb_vs_bb_flop") == 2
    assert types.count("btn_vs_bb_flop") == 2
    assert types.count("bunching_postflop") == 2


def test_legacy_sequential_preserves_spot_block_order():
    config = load_solver_config(Path("config/solver_labels_smoke.yaml"))
    config.enumeration_mode = "legacy_sequential"
    keys = enumerate_spot_keys(config)
    # All sb first (3 boards × 2 heroes = 6)
    assert all(_spot_type(k) == "sb_vs_bb_flop" for k in keys[:6])
    assert all(_spot_type(k) == "btn_vs_bb_flop" for k in keys[6:12])
    assert all(_spot_type(k) == "bunching_postflop" for k in keys[12:18])


def test_disable_bunching_drops_bunching_spots():
    config = load_solver_config(Path("config/solver_labels_smoke.yaml"))
    config.disable_bunching = True
    keys = enumerate_spot_keys(config)
    assert len(keys) == 12
    assert not any(_spot_type(k) == "bunching_postflop" for k in keys)


def test_spot_key_format():
    config = load_solver_config(Path("config/solver_labels_smoke.yaml"))
    spot = next(enumerate_spots(config))
    key = spot_key(spot)
    assert "|" in key
    assert key.startswith(spot.spot_id)
    assert key.endswith(f"hero={spot.hero_seat}")


def test_prod6000_interleaved_first_600_ratio():
    config = load_solver_config(Path("config/solver_labels_prod6000.yaml"))
    keys = enumerate_spot_keys(config)
    assert len(keys) == 6000
    head = keys[:600]
    vanilla = sum(1 for k in head if _spot_type(k) in ("sb_vs_bb_flop", "btn_vs_bb_flop"))
    bunching = sum(1 for k in head if _spot_type(k) == "bunching_postflop")
    assert vanilla == 400
    assert bunching == 200
