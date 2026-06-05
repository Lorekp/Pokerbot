"""Permanent sentinel: native OMPEval vs numba _eval7 orientation for N-way showdown."""

from __future__ import annotations

import numpy as np
import pytest

from aof_cfr.buckets import LABELS_169
from aof_cfr.nway_eval_orientation import (
    eval7_seat_rank,
    native_eval7_available,
    pick_nway_winners,
    seat_share_from_ranks,
)
from aof_cfr.terminal_ev import _exact_nway_shares_numba
from aof_cfr.trio_eval_mc import mc_nway_shares
from aof_cfr.trio_payoff_tensor import _all_combo_triples
from aof_equity import warmup
from aof_equity_grids import _card_str_to_id, _eval7, _exact_six_card_combo_metrics

pytestmark = pytest.mark.skipif(
    not native_eval7_available(),
    reason="hu_table_native extension not available",
)


@pytest.fixture(scope="module", autouse=True)
def _warm() -> None:
    warmup()


def test_pairwise_win_tie_loss_sentinel() -> None:
    win_board = (0, 1, 2, 3, 4)
    tie_board = (8, 12, 16, 20, 24)

    hr, hn = eval7_seat_rank(48, 49, *win_board)
    vr, vn = eval7_seat_rank(46, 45, *win_board)
    assert hn and vn
    native_win = seat_share_from_ranks([hr, vr], native_omp=True)
    numba_win = seat_share_from_ranks(
        [_eval7(48, 49, *win_board), _eval7(46, 45, *win_board)], native_omp=False
    )
    assert np.allclose(native_win, [1.0, 0.0])
    assert np.allclose(native_win, numba_win)

    hr, _ = eval7_seat_rank(0, 4, *tie_board)
    vr, _ = eval7_seat_rank(1, 5, *tie_board)
    native_tie = seat_share_from_ranks([hr, vr], native_omp=True)
    numba_tie = seat_share_from_ranks(
        [_eval7(0, 4, *tie_board), _eval7(1, 5, *tie_board)], native_omp=False
    )
    assert np.allclose(native_tie, [0.5, 0.5])
    assert np.allclose(native_tie, numba_tie)


def test_native_exact_metrics_match_numba_oracle() -> None:
    from tools.ompeval_cli import native as omp_native

    idx = {lab: i for i, lab in enumerate(LABELS_169)}
    combo = _all_combo_triples(idx["22"], idx["22"], idx["33"])[0]
    cards = np.array([_card_str_to_id(c) for c in sum(combo, ())], dtype=np.int32)
    py = _exact_six_card_combo_metrics(*cards)
    nat = omp_native.exact_six_card_combo_metrics(cards)
    assert np.max(np.abs(np.array(py) - np.array(nat))) < 1e-9


def test_native_mc_matches_numba_oracle_on_sentinel_board() -> None:
    """Native MC path must match numba exact oracle (same comparison semantics via shares)."""
    board_holes = [
        (_card_str_to_id("Ac"), _card_str_to_id("Ad")),
        (_card_str_to_id("Ah"), _card_str_to_id("As")),
        (_card_str_to_id("3c"), _card_str_to_id("3d")),
    ]
    flat = np.array([c for pair in board_holes for c in pair], dtype=np.int32)
    exact = _exact_nway_shares_numba(flat, 3)
    mc = mc_nway_shares(flat, 3, 4096, 42)
    assert np.max(np.abs(mc - exact)) < 0.004


def test_collision_triple_22_22_33_native_mc_matches_exact() -> None:
    idx = {lab: i for i, lab in enumerate(LABELS_169)}
    combo = _all_combo_triples(idx["22"], idx["22"], idx["33"])[0]
    flat = np.array([_card_str_to_id(c) for c in sum(combo, ())], dtype=np.int32)
    exact = _exact_nway_shares_numba(flat, 3)
    acc = np.zeros(3, dtype=np.float64)
    for seed in range(20):
        acc += mc_nway_shares(flat, 3, 4096, seed)
    mean = acc / 20.0
    assert np.max(np.abs(mean - exact)) < 0.004


def test_comparator_sentinel_would_fail_if_flipped() -> None:
    win_board = (0, 1, 2, 3, 4)
    hr, _ = eval7_seat_rank(48, 49, *win_board)
    vr, _ = eval7_seat_rank(46, 45, *win_board)
    assert hr > vr
    assert pick_nway_winners([hr, vr], native_omp=True)[0] == hr
    # Applying numba (lower-is-better) to OMPEval ranks inverts the winner
    wrong_best, _ = pick_nway_winners([hr, vr], native_omp=False)
    assert wrong_best == vr
