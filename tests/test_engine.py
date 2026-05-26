"""Tests for game engine."""

import random

import numpy as np
import pytest

from pokerbot.engine.actions import Action, ActionType
from pokerbot.engine.cards import Card, parse_cards, parse_holding
from pokerbot.engine.game import GameState, play_hand
from pokerbot.engine.hand_eval import evaluate_hand
from pokerbot.encoding.tensors import encode_state, SPEC


def test_card_parse():
    c = parse_holding("AhKd")
    assert c[0].rank.name == "ACE"
    assert c[1].suit.name == "DIAMONDS"


def test_hand_eval_pair_beats_high_card():
    pair = list(parse_cards("AhAdKsQhJc")) + list(parse_cards("2s3h"))
    high = list(parse_cards("AhKdQsJh9c")) + list(parse_cards("2s3h"))
    assert evaluate_hand(pair) > evaluate_hand(high)


def test_new_hand_blinds():
    state = GameState.new_hand(button_seat=3, rng=random.Random(0))
    assert state.pot_bb == 1.5
    assert state.current_bet_bb == 1.0


def test_legal_actions_preflop_utg():
    state = GameState.new_hand(rng=random.Random(1))
    state.deal_hole_cards()
    utg = state.action_seat
    legal = state.legal_actions()
    types = {a.action_type for a in legal}
    assert ActionType.FOLD in types
    assert ActionType.CALL in types or ActionType.CHECK in types


def test_tensor_shapes():
    state = GameState.new_hand(rng=random.Random(2))
    state.deal_hole_cards()
    enc = encode_state(state, hero_seat=0)
    assert isinstance(enc["weva_id"], (int, np.integer))
    assert enc["actions"].shape == SPEC.action_shape


def test_play_hand_to_completion():
    rng = random.Random(42)

    def random_policy(s: GameState) -> Action:
        legal = s.legal_actions()
        return rng.choice(legal)

    policies = {i: random_policy for i in range(6)}
    state, deltas = play_hand(policies, rng=rng)
    assert state.is_terminal()
    assert abs(sum(deltas.values())) < 0.01
