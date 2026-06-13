"""Game-model v2 terminal chip model: pot = k*stack + FOLDED blinds, exactly zero-sum.

Pencil-and-paper expected values + the zero-sum identity over random terminals for every
(n, jammer-set, folded-set) class. v1 (legacy, pot = k*stack + 1.5 dead) is pinned too — its
sum violates zero-sum by exactly ``1.5 - folded_blinds``, the audited ~3.75-equity-point
looseness on call thresholds. Fold-branch terminals are identical across models.
"""

from __future__ import annotations

import random

import pytest

from aof_cfr.combo_deal import FULL_DECK, card_id
from aof_cfr.exploit_mc import _terminal_returns
from aof_cfr.pushfold_payoffs import BLINDS_DEAD, contested_pot_bb
from aof_ranges import active_seats


def _holes(n, seed=1):
    rng = random.Random(seed)
    deck = list(FULL_DECK)
    rng.shuffle(deck)
    return [(card_id(deck[2 * i]), card_id(deck[2 * i + 1])) for i in range(n)]


def _blind(seat):
    return {"SB": 0.5, "BB": 1.0}.get(seat, 0.0)


@pytest.fixture(autouse=True)
def _pin_no_fee(monkeypatch):
    """Pencil-and-paper expectations here are PRE-FEE; pin the fee to 0 regardless of the
    shipped manifest's flat_fee_bb (tests that exercise the fee set their own env)."""
    monkeypatch.setenv("AOF_FLAT_FEE_BB", "0")
    yield


def test_contested_pot_values_8bb() -> None:
    # v2: HU contested = 16 (both blinds inside the jams); 3-max pots history-dependent.
    # fee_bb=0.0 pins the PRE-FEE pot regardless of the shipped manifest's flat_fee_bb.
    assert contested_pot_bb(2, 0.0, 8.0, game_model=2, fee_bb=0.0) == 16.0  # sb_bb
    assert contested_pot_bb(2, 0.5, 8.0, game_model=2, fee_bb=0.0) == 16.5  # btn_bb (SB folded)
    assert contested_pot_bb(2, 1.0, 8.0, game_model=2, fee_bb=0.0) == 17.0  # btn_sb (BB folded)
    assert contested_pot_bb(3, 0.0, 8.0, game_model=2, fee_bb=0.0) == 24.0
    assert contested_pot_bb(4, 1.5, 8.0, game_model=2, fee_bb=0.0) == 33.5  # both blinds folded
    # v1 legacy: flat +1.5 dead regardless of who folded
    assert contested_pot_bb(2, 0.0, 8.0, game_model=1, fee_bb=0.0) == 17.5
    assert contested_pot_bb(3, 0.0, 8.0, game_model=1, fee_bb=0.0) == 25.5


def test_flat_fee_deducted_from_contested_pot(monkeypatch) -> None:
    """DC2 (capture 2026-06-12): the showdown fee comes out of the contested pot — folded
    players pay nothing, uncontested steals pay nothing. Official schedule: 0.12BB at
    0.25/0.50 Hold'em."""
    from aof_cfr import pushfold_payoffs as pp

    assert pp.FLAT_FEE_BB_OFFICIAL == pytest.approx(0.12)
    assert pp.OFFICIAL_HOLDEM_FEE_BB[(0.25, 0.50)] == pytest.approx(0.12)
    assert contested_pot_bb(2, 0.0, 8.0, game_model=2, fee_bb=0.12) == pytest.approx(15.88)
    assert contested_pot_bb(2, 1.0, 8.0, game_model=2, fee_bb=0.12) == pytest.approx(16.88)
    # env override beats the manifest default
    monkeypatch.setenv("AOF_FLAT_FEE_BB", "0.12")
    assert pp.flat_fee_default() == pytest.approx(0.12)
    assert contested_pot_bb(2, 0.0, 8.0, game_model=2) == pytest.approx(15.88)
    # zero-sum identity becomes sum(payoffs) == -fee at contested terminals ...
    seats = active_seats(2)
    out = _terminal_returns(2, seats, _holes(2), set(), 8.0, game_model=2)
    assert sum(out) == pytest.approx(-0.12, abs=1e-9)
    # ... while uncontested walks stay fee-free (DC1 hand: folders lost exactly their posts)
    walk = _terminal_returns(2, seats, _holes(2), {0}, 8.0, game_model=2)
    assert sum(walk) == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize("n", [2, 3, 4])
def test_zero_sum_identity_v2_all_terminal_classes(n, monkeypatch) -> None:
    """Sum(payoffs) == 0 (pre-fee) for EVERY terminal: contested with any jammer subset,
    every folded-blind combination, and uncontested walks."""
    monkeypatch.setenv("AOF_FLAT_FEE_BB", "0")    # pin the pre-fee identity
    seats = active_seats(n)
    for mask in range(2 ** n):
        folded = {i for i in range(n) if (mask >> i) & 1}
        if len(folded) == n:                      # everyone folding is not a terminal class
            continue
        for seed in (1, 2):
            out = _terminal_returns(n, seats, _holes(n, seed), folded, 8.0, game_model=2)
            assert sum(out) == pytest.approx(0.0, abs=1e-9), (n, sorted(folded))
            for j in folded:                       # folders lose exactly their blind
                assert out[j] == pytest.approx(-_blind(seats[j]))


@pytest.mark.parametrize("n", [2, 3, 4])
def test_v1_violates_zero_sum_by_exactly_the_double_count(n) -> None:
    seats = active_seats(n)
    folded = set()                                 # everyone jams: no folded blinds
    out = _terminal_returns(n, seats, _holes(n), folded, 8.0, game_model=1)
    assert sum(out) == pytest.approx(BLINDS_DEAD)  # +1.5 appears from nothing


def test_fold_terminals_identical_across_models() -> None:
    seats = active_seats(3)
    holes = _holes(3)
    for folded in ({1, 2}, {0, 2}, {0, 1}):        # uncontested walks
        v1 = _terminal_returns(3, seats, holes, folded, 8.0, game_model=1)
        v2 = _terminal_returns(3, seats, holes, folded, 8.0, game_model=2)
        assert v1 == v2
        assert sum(v1) == pytest.approx(0.0)


def test_pencil_vectors_8bb_v2() -> None:
    """Hand-checked: 3-max BTN+SB all-in, BB folds. pot = 17; jammer_i = share*17 - 8; BB -1."""
    seats = active_seats(3)                        # (BTN, SB, BB)
    holes = _holes(3, seed=7)
    out = _terminal_returns(3, seats, holes, {2}, 8.0, game_model=2)
    assert out[2] == pytest.approx(-1.0)
    # back out the shares: (out + 8) / 17 must be a probability pair summing to 1
    s0, s1 = (out[0] + 8.0) / 17.0, (out[1] + 8.0) / 17.0
    assert 0.0 <= s0 <= 1.0 and 0.0 <= s1 <= 1.0
    assert s0 + s1 == pytest.approx(1.0)


def test_build_static_payoffs_pots_by_model() -> None:
    """The 3-max static payoff layer prices each pairing's pot per the model."""
    import numpy as np

    from aof_cfr.pushfold_payoffs import build_static_payoffs
    shares = np.full((3, 3, 3, 3), 1.0 / 3.0)      # tiny synthetic tensor, equal shares
    pair_eq = np.full((3, 3, 3, 3), 0.5)
    v2 = build_static_payoffs(shares, pair_eq, stack_bb=8.0, game_model=2)
    v1 = build_static_payoffs(shares, pair_eq, stack_bb=8.0, game_model=1)
    assert v2.sb_bb_call1.flat[0] == pytest.approx(0.5 * 16.0 - 8.0)    # = 0: zero-sum HU
    assert v2.btn_bb_call0.flat[0] == pytest.approx(0.5 * 16.5 - 8.0)
    assert v2.btn_sb_call0.flat[0] == pytest.approx(0.5 * 17.0 - 8.0)
    assert v2.three0.flat[0] == pytest.approx(24.0 / 3.0 - 8.0)          # = 0
    assert v1.sb_bb_call1.flat[0] == pytest.approx(0.5 * 17.5 - 8.0)     # legacy +0.75 subsidy


def test_hu_marginals_and_solver_pot_follow_model(monkeypatch) -> None:
    from aof_cfr.hu_br_legal import _pot_called
    assert _pot_called(8.0, game_model=2) == 16.0
    assert _pot_called(8.0, game_model=1) == 17.5
    # env override beats the manifest default
    monkeypatch.setenv("AOF_GAME_MODEL", "2")
    assert _pot_called(8.0) == 16.0
    monkeypatch.setenv("AOF_GAME_MODEL", "1")
    assert _pot_called(8.0) == 17.5
