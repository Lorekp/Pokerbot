"""Tests for cloud spot solver helpers and optional CLI parity."""

from __future__ import annotations

from pathlib import Path

import pytest

from pokerbot.solver.cloud_spot_solver import (
    board_timing_from_manifest,
    estimate_spot_seconds,
    order_pending_for_worker,
    slow_board_threshold_sec,
    sort_spot_keys_slow_first,
)
from pokerbot.solver.postflop_solver_client import DEFAULT_CLI_PATH


def test_board_timing_from_manifest() -> None:
    manifest = {
        "board_timing": {
            "Td9d6h": {"count": 2, "median_sec": 130.0, "max_sec": 140.0},
            "9hKh4h": {"count": 1, "median_sec": 12.0, "max_sec": 12.0},
        }
    }
    timing = board_timing_from_manifest(manifest)
    assert timing["Td9d6h"] == 130.0
    assert timing["9hKh4h"] == 12.0


def test_sort_spot_keys_slow_first() -> None:
    keys = [
        "sb_vs_bb_flop|9hKh4h|hero=4",
        "sb_vs_bb_flop|Td9d6h|hero=4",
        "sb_vs_bb_flop|QcJsTs|hero=4",
    ]
    timing = {"Td9d6h": 130.0, "9hKh4h": 12.0, "QcJsTs": 50.0}
    ordered = sort_spot_keys_slow_first(keys, timing)
    assert ordered[0].endswith("Td9d6h|hero=4")
    assert ordered[-1].endswith("9hKh4h|hero=4")


def test_estimate_spot_seconds_unknown_board() -> None:
    assert estimate_spot_seconds("sb_vs_bb_flop|9hKh4h|hero=4", {}) == 0.0


def test_order_pending_slow_bucket_first_per_worker_shuffle() -> None:
    timing = {"Td9d6h": 130.0, "9hKh4h": 12.0, "QcJsTs": 50.0, "Ah7c2s": 200.0}
    keys = [
        "sb_vs_bb_flop|9hKh4h|hero=4",
        "sb_vs_bb_flop|Td9d6h|hero=4",
        "sb_vs_bb_flop|QcJsTs|hero=4",
        "sb_vs_bb_flop|Ah7c2s|hero=4",
    ]
    threshold = slow_board_threshold_sec(timing)
    w1 = order_pending_for_worker(keys, "worker-a", board_timing=timing, slow_threshold_sec=threshold)
    w2 = order_pending_for_worker(keys, "worker-b", board_timing=timing, slow_threshold_sec=threshold)
    # Both workers prioritize slow bucket before fast bucket
    assert estimate_spot_seconds(w1[0], timing) >= threshold
    assert estimate_spot_seconds(w1[1], timing) >= threshold
    assert estimate_spot_seconds(w1[2], timing) < threshold
    # Different workers get different slow-bucket order (collision avoidance)
    assert w1 != w2


@pytest.mark.skipif(not DEFAULT_CLI_PATH.is_file(), reason="postflop_solver_cli not built")
def test_cloud_spot_solver_writes_shard(tmp_path: Path) -> None:
    from pokerbot.solver.cloud_spot_solver import CloudSpotSolver

    root = Path(__file__).resolve().parents[1]
    config = root / "config" / "solver_labels_smoke.yaml"
    weva = root / "data" / "weva"
    if not config.is_file() or not weva.is_dir():
        pytest.skip("smoke config or weva index missing")

    # Pick first valid spot key from enumeration via a tiny manifest slice.
    from pokerbot.solver.spot_enumerator import enumerate_spots, load_solver_config, spot_key
    from pokerbot.solver.spot_validity import is_two_player_valid, spot_node_from_state

    cfg = load_solver_config(config)
    spot_key_str = None
    for spot in enumerate_spots(cfg):
        key = spot_key(spot)
        validity = is_two_player_valid(
            spot_node_from_state(spot.state, spot.hero_seat, bunching=spot.bunching)
        )
        if validity.valid:
            spot_key_str = key
            break
    if spot_key_str is None:
        pytest.skip("no valid spot in smoke config")

    shard_dir = tmp_path / "shards"
    with CloudSpotSolver(
        config_path=config,
        weva_index=weva,
        shard_dir=shard_dir,
        cli_path=DEFAULT_CLI_PATH,
        solver_backend="worker",
        solver_threads_per_worker=1,
        spot_keys={spot_key_str},
        timeout_sec=3600.0,
    ) as solver:
        result = solver.solve(spot_key_str)

    assert result["ok"] is True, result
    assert Path(result["shard_path"]).is_file()
