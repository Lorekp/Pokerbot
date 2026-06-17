"""Gate 0 tests: WEVA encoding injectivity and Parquet replay."""

from __future__ import annotations

from pathlib import Path

import pytest

from pokerbot.engine.jax_weva_encode import NLHE_FLAT_DIM, encode_nlhe_flat
from pokerbot.training.nlhe_collision_check import (
    check_nlhe_encode_collisions,
    check_parquet_spot_replay,
)


def test_nlhe_flat_dim_positive() -> None:
    assert NLHE_FLAT_DIM > 64


@pytest.mark.slow
def test_nlhe_encode_no_collisions() -> None:
    result = check_nlhe_encode_collisions(n_hands=8, seed=1)
    assert result["passed"], result


def test_parquet_replay_vanilla_if_present() -> None:
    path = Path("data/solver/heldout_prod600_vanilla.parquet")
    if not path.exists():
        pytest.skip("heldout vanilla parquet missing")
    result = check_parquet_spot_replay(path, max_rows=5)
    assert result["rows_checked"] > 0
    assert result["passed"], result
