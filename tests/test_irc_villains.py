"""IRC realistic villains: sized betting (the task #12 fix, isolated) + archetype separation.

These back the Phase-2 value-extraction gauntlet. The point of the IRC villains over the hs_* bots is
that they express aggression as REAL sized bets (so the read can tell a value-only station from a bluffy
lag), instead of collapsing to a forced check when no sized bet clears min_raise.
"""

from __future__ import annotations

import random

import numpy as np

from pokerbot.engine.actions import ActionType
from pokerbot.engine.game import GameState
from pokerbot.eval.irc_villains import _pick_sized_bet, build_irc_villain
from pokerbot.training.self_play import _legal_mask

ARCHES = ("irc_station", "irc_lag", "irc_nit", "irc_tag")


def test_build_all_archetypes():
    for a in ARCHES:
        bot = build_irc_villain(a, random.Random(0))
        assert callable(bot)


def test_unknown_archetype_raises():
    import pytest
    with pytest.raises(ValueError):
        build_irc_villain("irc_nonsense")


def test_pick_sized_bet_targets_fraction():
    """On an unopened postflop node with a real pot, _pick_sized_bet returns a sized BET whose pot
    fraction is close to the target (not a jam, not a check)."""
    state = GameState.new_hand(button_seat=3, rng=random.Random(1))
    # Build a pot: limp/call around to the flop so sized bets are legal.
    guard = 0
    while state.street.name == "PREFLOP" and guard < 40:
        guard += 1
        seat = state.action_seat
        legal = state.legal_actions(seat)
        if not legal:
            break
        call = next((a for a in legal if a.action_type == ActionType.CALL), None)
        check = next((a for a in legal if a.action_type == ActionType.CHECK), None)
        state.apply_action(call or check or legal[0], seat)
    if state.street.name == "PREFLOP":
        return  # couldn't reach a postflop pot this seed; the other tests cover behavior
    seat = state.action_seat
    legal = state.legal_actions(seat)
    if not any(a.action_type in (ActionType.BET, ActionType.RAISE) for a in legal):
        return  # no sized bet legal here; covered by the jam-fallback test
    a = _pick_sized_bet(state, seat, target_frac=0.5)
    assert a is not None and a.action_type in (ActionType.BET, ActionType.RAISE)
    frac = float(a.amount_bb) / max(float(state.pot_bb), 1e-9)
    assert 0.2 <= frac <= 1.2  # a realistic, non-jam sizing near the 0.5 target


def _unopened_bet_rate(name: str, hands: int = 200) -> tuple[float, float]:
    """Self-play the archetype; return (unopened bet rate, facing call rate) — the behavioural
    signatures the read keys on."""
    bet = unopened = call = facing = 0
    for h in range(hands):
        state = GameState.new_hand(button_seat=3, rng=random.Random(100 + h))
        bots = {s: build_irc_villain(name, random.Random(200 + h + 11 * s)) for s in range(6)}
        guard = 0
        while state.street.name != "SHOWDOWN" and guard < 300:
            guard += 1
            seat = state.action_seat
            legal = state.legal_actions(seat)
            if not legal:
                break
            lm = np.asarray(_legal_mask(state, seat), dtype=bool)
            check_legal = bool(lm.size > 1 and lm[1])
            a = bots[seat](state)
            if check_legal:
                unopened += 1
                if a.action_type in (ActionType.BET, ActionType.RAISE, ActionType.ALL_IN):
                    bet += 1
            else:
                facing += 1
                if a.action_type == ActionType.CALL:
                    call += 1
            state.apply_action(a, seat)
    return (bet / max(unopened, 1), call / max(facing, 1))


def test_lag_bets_more_unopened_than_station():
    """The core separation the gate needs: a lag's own-initiative bet rate is materially higher than a
    station's (and a station calls wider when facing). With the hs_* bots this signal was destroyed by
    forced checks."""
    lag_bet, _ = _unopened_bet_rate("irc_lag")
    sta_bet, sta_call = _unopened_bet_rate("irc_station")
    _, nit_call = _unopened_bet_rate("irc_nit")
    assert lag_bet > sta_bet + 0.10, f"lag {lag_bet:.2f} not > station {sta_bet:.2f}+0.10"
    assert sta_call > nit_call, f"station call {sta_call:.2f} should exceed nit {nit_call:.2f}"
