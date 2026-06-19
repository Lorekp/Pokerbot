"""Double-count guard: the unopened VALUE tilt must DEFER to the solver when the base already
best-responds to the profiled opponent (the pluribus search vehicle's solver.opponent_model already
value-bets the fish). solver_best_responds=True => value branch returns the base unchanged; the FACING
branch is unaffected (it has its own defer-to-search-policy gate). The heuristic/Hybrid vehicle keeps
the full tilt (solver_best_responds=False, the battery-validated path).
"""

from __future__ import annotations

import random

import numpy as np

from pokerbot.engine.actions import Action, ActionType
from pokerbot.engine.game import GameState
from pokerbot.opponent.population_prior import context_prior
from pokerbot.opponent.root_mixer import apply_root_exploit_mixer
from pokerbot.opponent.stats import OpponentStats

_ST = ("flop", "turn", "river")
_TEX = ("dry", "wet", "paired")


def _station():  # calls everything facing a bet, passive unopened -> a confirmed value-fish
    prof = OpponentStats()
    prof.hands = 200
    for st in _ST:
        for tex in _TEX:
            for fac in ("unopened", "facing_bet"):
                ck = f"{st}|{tex}|{fac}"
                for h in range(1, 50):
                    prof.decay_and_add(ck, [0, 1, 0, 0], context_prior(ck), h)
                prof.add_ctx(ck, [0, 24, 0, 0])
    return prof


def _facing_state(seed: int):
    st = GameState.new_hand(button_seat=0, rng=random.Random(seed))
    st.deal_hole_cards()
    for _ in range(6):
        if st.action_seat == 0:
            break
        legal = st.legal_actions()
        raise_a = next((a for a in legal if a.action_type == ActionType.RAISE), None)
        st.apply_action(raise_a if (st.action_seat == 3 and raise_a) else Action(ActionType.FOLD))
        st.advance_to_next_actor()
    return st


def _unopened_flop_state(seed: int):
    """Advance to a postflop spot where the current actor can CHECK (unopened / not facing a bet)."""
    st = GameState.new_hand(button_seat=0, rng=random.Random(seed))
    st.deal_hole_cards()
    for _ in range(60):
        legal = st.legal_actions()
        if len(getattr(st, "board", []) or []) >= 3 and any(
                a.action_type == ActionType.CHECK for a in legal):
            return st, int(st.action_seat)
        # progress the hand cheaply: prefer CHECK, else CALL, else FOLD
        nxt = (next((a for a in legal if a.action_type == ActionType.CHECK), None)
               or next((a for a in legal if a.action_type == ActionType.CALL), None)
               or Action(ActionType.FOLD))
        st.apply_action(nxt)
        st.advance_to_next_actor()
    return None, -1


def test_flag_does_not_touch_facing_branch():
    # The flag only gates the unopened VALUE branch; on a facing decision the output is identical.
    for seed in range(8):
        st = _facing_state(seed)
        if st.action_seat != 0:
            continue
        base = np.full(11, 1.0 / 11)
        out_off = apply_root_exploit_mixer(st, 0, base.copy(), {1: _station()},
                                           unopened=True, solver_best_responds=False)
        out_on = apply_root_exploit_mixer(st, 0, base.copy(), {1: _station()},
                                          unopened=True, solver_best_responds=True)
        assert np.allclose(out_off, out_on, atol=1e-12)


def test_unopened_value_branch_defers_under_flag():
    st, seat = _unopened_flop_state(3)
    if st is None:
        import pytest
        pytest.skip("could not construct an unopened postflop spot in this engine build")
    n = len(st.legal_actions()) and 11
    base = np.full(11, 1.0 / 11)
    profs = {s: _station() for s in range(len(st.players)) if s != seat}
    # deferred: value branch returns the (renormalized) base unchanged
    out_defer = apply_root_exploit_mixer(st, seat, base.copy(), profs,
                                         unopened=True, solver_best_responds=True)
    assert np.allclose(out_defer, base / base.sum(), atol=1e-9)
    # not deferred: the value tilt is free to move the policy (>= no change). Either way it must stay a
    # valid distribution; the point is the deferred arm is identity, proving the guard fires.
    out_full = apply_root_exploit_mixer(st, seat, base.copy(), profs,
                                        unopened=True, solver_best_responds=False)
    assert abs(out_full.sum() - 1.0) < 1e-6
