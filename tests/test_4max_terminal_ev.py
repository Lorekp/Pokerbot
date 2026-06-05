"""4-max FP terminal path uses exact enumeration (not mc_nway_shares)."""

from __future__ import annotations

import numpy as np

from aof_cfr.exploit_mc import _terminal_returns
from aof_cfr.pushfold_payoffs import BLINDS_DEAD, STACK_BB
from aof_cfr.terminal_ev import _exact_nway_shares, _exact_nway_shares_numba, showdown_terminal_returns
from aof_equity import warmup
from aof_equity_grids import _card_str_to_id
from aof_ranges import active_seats


def test_showdown_terminal_matches_exact_on_collision_deal() -> None:
    warmup()
    cards = ["2c", "2d", "2h", "2s", "3c", "3d", "4c", "5d"]
    hole_ids = [
        ( _card_str_to_id(cards[0]), _card_str_to_id(cards[1])),
        (_card_str_to_id(cards[2]), _card_str_to_id(cards[3])),
        (_card_str_to_id(cards[4]), _card_str_to_id(cards[5])),
        (_card_str_to_id(cards[6]), _card_str_to_id(cards[7])),
    ]
    flat = np.array([c for pair in hole_ids for c in pair], dtype=np.int32)
    exact_shares = _exact_nway_shares_numba(flat, 4)
    invested = np.array([8.0, 8.0, 8.0, 8.0], dtype=np.float64)
    rets = showdown_terminal_returns(hole_ids, invested)
    pot = float(invested.sum())
    for i, sh in enumerate(exact_shares):
        expected = sh * pot - float(invested[i])
        assert abs(rets[i] - expected) < 1e-9


def test_exploit_mc_terminal_matches_pushfold_model() -> None:
    """4-max all-in showdown reproduces the validated pushfold_payoffs chip model:
    pot = k*STACK + dead blinds, each all-in net = share*pot - STACK (exact runout shares)."""
    warmup()
    cards = ["2c", "2d", "2h", "2s", "3c", "3d", "4c", "5d"]
    holes = [
        (_card_str_to_id(cards[0]), _card_str_to_id(cards[1])),
        (_card_str_to_id(cards[2]), _card_str_to_id(cards[3])),
        (_card_str_to_id(cards[4]), _card_str_to_id(cards[5])),
        (_card_str_to_id(cards[6]), _card_str_to_id(cards[7])),
    ]
    seats = active_seats(4)
    rets = _terminal_returns(4, seats, holes, set())
    pot = 4 * STACK_BB + BLINDS_DEAD
    # Each all-in net must be share*pot - STACK with shares (recovered) a valid distribution.
    recovered = [(rets[i] + STACK_BB) / pot for i in range(4)]
    assert abs(sum(recovered) - 1.0) < 1e-9
    assert all(-1e-9 <= s <= 1.0 + 1e-9 for s in recovered)
    # uncontested: everyone folds to the last seat (BB), who wins the folded blinds.
    rets_unc = _terminal_returns(4, seats, holes, {0, 1, 2})
    blinds = [0.0 if s not in ("SB", "BB") else (0.5 if s == "SB" else 1.0) for s in seats]
    assert abs(rets_unc[3] - sum(blinds[j] for j in (0, 1, 2))) < 1e-9
    for j in (0, 1, 2):
        assert abs(rets_unc[j] - (-blinds[j])) < 1e-9
