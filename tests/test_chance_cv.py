"""Chance (board-reveal) control variate — unbiasedness + the play_hand sink byte-identity guard.

The control variate's whole safety claim is that it CANNOT bias the mean: ``c_C = pot_pre*(eq(realized) -
mean_c eq(c))`` has exact zero mean over the engine's uniform deal, for any baseline. These tests pin that
(enumerated zero-mean over the full candidate set), the fold/all-in gating, and that wiring the
``chance_sink`` into ``play_hand`` leaves the played hand byte-identical (the sink is observe-only).
"""

from __future__ import annotations

import random
from types import SimpleNamespace

import pytest

import pokerbot.eval.chance_cv as ccv
from pokerbot.engine.actions import Action, ActionType
from pokerbot.engine.cards import Card
from pokerbot.engine.game import GameState, play_hand

pytestmark = pytest.mark.skipif(ccv._phe is None, reason="phevaluator not installed")


def _c(s: str) -> Card:
    return Card.from_str(s)


def _fake_state(holes, board, pool, *, folded=None, pot=10.0, actors=True):
    """A duck-typed GameState exposing exactly what ChanceCV.on_reveal reads. ``board`` is the realized
    board (last card = the reveal); ``pool`` = the candidate cards the deal drew from minus the realized
    one (i.e. the undealt deck)."""
    folded = folded or [False] * len(holes)
    players = [SimpleNamespace(seat=i, folded=folded[i], hole=list(h), all_in=False)
               for i, h in enumerate(holes)]
    deck = SimpleNamespace(_cards=list(pool), _idx=0)
    return SimpleNamespace(players=players, board=list(board), _deck=deck, pot_bb=pot,
                           actors_remaining=lambda: actors)


def _mean_correction_over_deal(holes, base_board, candidates, hero_seat, *, pot=10.0):
    """Average the single-card-reveal correction over EVERY candidate being the realized card. The deal is
    uniform over ``candidates``, so an unbiased control variate must average to exactly 0."""
    corrections = []
    for i, realized in enumerate(candidates):
        remaining = [c for j, c in enumerate(candidates) if j != i]  # undealt deck for this realization
        st = _fake_state(holes, base_board + [realized], remaining, pot=pot)
        cv = ccv.ChanceCV(hero_seat)
        cv.on_reveal(st, len(base_board))
        corrections.append(cv.total())
    return sum(corrections) / len(corrections), corrections


def test_hero_share_basic():
    board = [c.index for c in map(_c, ["2c", "7d", "9s", "Tc", "Jd"])]
    aces = [(_c("As").index, _c("Ad").index), (_c("Ks").index, _c("Kd").index)]
    assert ccv._hero_share(0, aces, board) == 1.0   # pair of aces beats pair of kings
    assert ccv._hero_share(1, aces, board) == 0.0
    # exact tie -> split (both make the same ace-high straight off the board)
    straight = [(_c("As").index, _c("Ks").index), (_c("Ah").index, _c("Kh").index)]
    sb = [c.index for c in map(_c, ["Qc", "Jc", "Tc", "2d", "3d"])]
    assert ccv._hero_share(0, straight, sb) == 0.5


def test_river_correction_zero_mean():
    # Turn done (4-card board); the river is the reveal. Small candidate universe (zero-mean is exact for
    # ANY pool the deal is uniform over).
    holes = [[_c("As"), _c("Kd")], [_c("Qh"), _c("Qc")]]
    base = list(map(_c, ["2c", "7d", "9s", "Th"]))
    cands = list(map(_c, ["Ah", "Kc", "Qs", "Js", "5d", "5h", "8c", "2d"]))
    mean, corr = _mean_correction_over_deal(holes, base, cands, 0, pot=12.0)
    assert any(abs(x) > 1e-9 for x in corr)   # the CV is actually doing something (non-degenerate)
    assert abs(mean) < 1e-9                    # ...and is exactly mean-zero over the deal (unbiased)


def test_turn_correction_zero_mean():
    # Flop done (3-card board); the turn is the reveal — need_after=1, so each candidate integrates the
    # river over the remaining pool. Still exactly mean-zero over the turn deal.
    holes = [[_c("As"), _c("Kd")], [_c("Qh"), _c("Jc")]]
    base = list(map(_c, ["2c", "7d", "9s"]))
    cands = list(map(_c, ["Ah", "Kc", "Qs", "Jd", "5d", "8c", "Td"]))
    mean, corr = _mean_correction_over_deal(holes, base, cands, 0, pot=8.0)
    assert any(abs(x) > 1e-9 for x in corr)
    assert abs(mean) < 1e-9


def test_no_correction_when_hero_folded():
    holes = [[_c("As"), _c("Kd")], [_c("Qh"), _c("Qc")]]
    base = [c.index for c in map(_c, ["2c", "7d", "9s", "Th"])]
    st = _fake_state(holes, base + [_c("Ah").index], [_c("Kc").index, _c("Qs").index],
                     folded=[True, False])
    cv = ccv.ChanceCV(0)
    cv.on_reveal(st, len(base))
    assert cv.total() == 0.0


def test_no_correction_during_allin_runout():
    holes = [[_c("As"), _c("Kd")], [_c("Qh"), _c("Qc")]]
    base = [c.index for c in map(_c, ["2c", "7d", "9s", "Th"])]
    st = _fake_state(holes, base + [_c("Ah").index], [_c("Kc").index, _c("Qs").index], actors=False)
    cv = ccv.ChanceCV(0)
    cv.on_reveal(st, len(base))
    assert cv.total() == 0.0   # an all-in runout is integrated by _allin_ev_settle, not corrected here


def _always_call(state):
    legal = state.legal_actions()
    for at in (ActionType.CHECK, ActionType.CALL, ActionType.FOLD):
        a = next((x for x in legal if x.action_type == at), None)
        if a is not None:
            return a
    return legal[0]


def test_play_hand_chance_sink_byte_identical():
    # The sink is observe-only: a hand played with a recording sink must match one played with sink=None.
    def run(sink):
        st = GameState.new_hand(button_seat=3, rng=random.Random(12345))
        policies = {s: _always_call for s in range(6)}
        return play_hand(policies, state=st, chance_sink=sink)

    seen = []
    fin_a, deltas_a = run(None)
    fin_b, deltas_b = run(lambda state, n: seen.append((n, len(state.board))))
    assert deltas_a == deltas_b
    assert [c.index for c in fin_a.board] == [c.index for c in fin_b.board]
    # the sink fired on board growth only (post >= pre+1), if it fired at all
    assert all(post > pre for pre, post in seen)
