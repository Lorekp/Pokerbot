"""Independent side-pot cross-check: replay identical scenarios through pokerkit and JAX."""

from __future__ import annotations

import pytest

pytest.importorskip("pokerkit")

from pokerbot.engine.pokerkit_replay import CASE_A, CASE_G, assert_replay_match


def test_pokerkit_replay_case_a_three_way_distinct_stacks() -> None:
    """Case A: 3+ way all-ins at distinct stacks — terminal payouts match exactly."""
    pk, jax = assert_replay_match(CASE_A)
    assert pk[0] == 9000
    assert pk[1] == -6000
    assert pk[2] == -3000


def test_pokerkit_replay_case_g_folded_contributor_multi_layer() -> None:
    """Case G: folded contributor across multi-layer side pots — payouts match exactly."""
    pk, jax = assert_replay_match(CASE_G)
    assert pk[0] == -4000  # folded contributor loses 4000
    assert pk[1] > 0  # best hand wins
    assert pk[2] == -6000
    assert pk[3] == -4000
