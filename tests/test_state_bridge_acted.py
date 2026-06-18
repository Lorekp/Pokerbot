"""Regression: game_state_to_env must populate EnvState.acted from the GameState's per-street
acted set. Leaving it all-False (the original bridge bug) makes the solver believe NO opponent has
acted yet, so from any non-first-to-act hero seat it searches a subgame with phantom extra betting
rounds (already-acted opponents are expected to act again). That corrupts the EVs out of position and
made the seat-rotated win-rate report the search bot LOSING hundreds of bb/100 (it over-jammed OOP
preflop). Seat 0 (UTG, first to act) was immune because nobody had acted yet -- which is exactly why
the non-rotated eval (hero only ever at seat 0) looked fine.
"""

from __future__ import annotations

import copy
import random

import numpy as np
import jax.numpy as jnp

from pokerbot.engine.actions import ActionType
from pokerbot.engine.game import GameState
from pokerbot.engine.jax_engine import _street_complete, apply_action, legal_action_mask
from pokerbot.engine.np_engine import CALL as NP_CALL, CHECK as NP_CHECK
from pokerbot.search.state_bridge import game_state_to_env


def _call_or_check(state):
    legal = state.legal_actions()
    for at in (ActionType.CALL, ActionType.CHECK):
        a = next((x for x in legal if x.action_type == at), None)
        if a is not None:
            return a
    return legal[0]


def _drive_to_hero(hero_seat, button=3, seed=1234):
    st = GameState.new_hand(button_seat=button, rng=random.Random(seed))
    st.deal_hole_cards()
    guard = 0
    while st.action_seat != hero_seat and guard < 20 and not st.is_terminal():
        st.apply_action(_call_or_check(st))
        st.advance_to_next_actor()
        guard += 1
    return st


def test_acted_reflects_prior_actors():
    # With button=3, preflop order is seats 0,1,2,3,4(SB),5(BB). When the BB (seat 5) is to act, seats
    # 0..4 have already acted this street; the bridge must mark exactly those as acted.
    st = _drive_to_hero(5)
    assert st.action_seat == 5 and not st.is_terminal()
    env = game_state_to_env(st, hero_seat=5)
    acted = np.asarray(env.acted, dtype=bool)
    assert acted.tolist() == [True, True, True, True, True, False]


def test_seat0_first_to_act_has_no_prior_actors():
    # Seat 0 is UTG (first preflop) -> nobody has acted yet -> all-False is CORRECT here (this is why
    # the seat-0-only non-rotated eval never saw the bug).
    st = _drive_to_hero(0)
    assert st.action_seat == 0
    env = game_state_to_env(st, hero_seat=0)
    assert not np.asarray(env.acted, dtype=bool).any()


def test_bb_closing_completes_the_street_in_the_bridged_engine():
    # The decisive check: when the BB closes a limped preflop pot, the REAL engine advances to the flop.
    # The bridged jax engine must agree (street complete) -- it does NOT if `acted` is all-False.
    st = _drive_to_hero(5)
    env = game_state_to_env(st, hero_seat=5)
    mask = np.asarray(legal_action_mask(env, jnp.int32(5)), dtype=bool)
    act_idx = NP_CALL if mask[NP_CALL] else (NP_CHECK if mask[NP_CHECK] else int(np.argmax(mask)))
    env2 = apply_action(env, jnp.int32(act_idx))
    assert bool(np.asarray(_street_complete(env2)))

    # ... and reality agrees the preflop street advances after the BB acts.
    real = copy.deepcopy(st)
    before = real.street
    real.apply_action(_call_or_check(real))
    assert real.street != before
