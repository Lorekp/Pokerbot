"""Fast NLHE encoder parity and collision checks."""

from __future__ import annotations

from pokerbot.training.nlhe_collision_check import check_fast_encode_parity, check_nlhe_encode_collisions


def test_fast_path_matches_slow_bridge() -> None:
    rep = check_fast_encode_parity(n_hands=16, seed=1)
    assert rep["passed"], rep


def test_gate0_collisions_on_fast_path() -> None:
    rep = check_nlhe_encode_collisions(n_hands=12, seed=2)
    assert rep["passed"], rep
