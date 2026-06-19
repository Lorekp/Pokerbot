"""WS7: resolve_showdown must layer the pot by contribution (side pots), not split the whole pot among
the overall winners. A short all-in that wins only the main pot must not be paid the side pot."""

from __future__ import annotations

from pokerbot.engine.cards import Card
from pokerbot.engine.game import GameState, PlayerState, Position, Street

_DRY = ["2c", "7d", "9h", "Js", "3s"]   # no board pair/flush -> pocket pairs rank AA>KK>QQ


def _c(s):
    return Card.from_str(s)


def _mk(seat, pos, ct, hole=None, folded=False):
    p = PlayerState(seat=seat, position=pos, stack_bb=0.0,
                    hole=(_c(hole[0]), _c(hole[1])) if hole else None, folded=folded)
    p.committed_total_bb = ct
    return p


def test_short_allin_wins_only_main_pot():
    st = GameState()
    st.board = [_c(x) for x in _DRY]
    st.street = Street.SHOWDOWN
    st.pot_bb = 220.0
    st.players = [
        _mk(0, Position.BTN, 20.0, ("As", "Ah")),   # short all-in, BEST hand
        _mk(1, Position.SB, 100.0, ("Ks", "Kh")),   # wins the side pot
        _mk(2, Position.BB, 100.0, ("Qs", "Qh")),
        _mk(3, Position.UTG, 0.0, folded=True),
        _mk(4, Position.HJ, 0.0, folded=True),
        _mk(5, Position.CO, 0.0, folded=True),
    ]
    d = st.resolve_showdown()
    assert abs(d[0] - 40.0) < 1e-6     # main pot 60 - 20 committed (NOT the whole 220)
    assert abs(d[1] - 60.0) < 1e-6     # side pot 160 - 100 committed
    assert abs(d[2] + 100.0) < 1e-6    # loses the 100 it put in
    assert abs(sum(d.values())) < 1e-6  # zero-sum


def test_uncalled_excess_is_returned():
    st = GameState()
    st.board = [_c(x) for x in _DRY]
    st.street = Street.SHOWDOWN
    st.pot_bb = 160.0
    st.players = [
        _mk(0, Position.BTN, 100.0, ("Qs", "Qh")),  # bet 100 but only 60 was called -> 40 uncalled
        _mk(1, Position.SB, 60.0, ("Ks", "Kh")),    # all-in 60, BETTER hand, wins
    ] + [_mk(s, Position.BB, 0.0, folded=True) for s in (2, 3, 4, 5)]
    d = st.resolve_showdown()
    assert abs(d[0] + 60.0) < 1e-6     # loses only the 60 that was called (40 refunded)
    assert abs(d[1] - 60.0) < 1e-6     # wins the 120 matched pot - 60 committed
    assert abs(sum(d.values())) < 1e-6
