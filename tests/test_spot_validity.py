"""Tests for is_two_player_valid scope guard."""

from __future__ import annotations

import random

import pytest

from pokerbot.engine.actions import Action, ActionType
from pokerbot.engine.cards import Card
from pokerbot.engine.game import GameState, Position, Street
from pokerbot.solver.spot_validity import (
    SpotClass,
    SpotNode,
    is_two_player_valid,
    is_two_player_valid_state,
    spot_node_from_state,
)


def _fold_all_except(state: GameState, keep: set[int]) -> None:
    for p in state.players:
        if p.seat not in keep:
            p.folded = True


def _hu_sb_bb_flop() -> tuple[GameState, int]:
    """SB (4) vs BB (5) on flop, all others folded."""
    rng = random.Random(42)
    state = GameState.new_hand(button_seat=3, rng=rng)
    state.deal_hole_cards()
    _fold_all_except(state, {4, 5})
    while state.street != Street.FLOP:
        if state.street == Street.PREFLOP:
            state.apply_action(Action(ActionType.FOLD), seat=0)
            state.advance_to_next_actor()
            state.apply_action(Action(ActionType.FOLD), seat=1)
            state.advance_to_next_actor()
            state.apply_action(Action(ActionType.FOLD), seat=2)
            state.advance_to_next_actor()
            state.apply_action(Action(ActionType.FOLD), seat=3)
            state.advance_to_next_actor()
            state.apply_action(Action(ActionType.CALL), seat=4)
            state.advance_to_next_actor()
            state.apply_action(Action(ActionType.CHECK), seat=5)
        else:
            break
    state.board = [
        Card.from_str("Td"),
        Card.from_str("9d"),
        Card.from_str("6h"),
    ]
    state.street = Street.FLOP
    state.action_seat = 4
    return state, 4


def test_sb_vs_bb_flop_valid():
    state, hero = _hu_sb_bb_flop()
    result = is_two_player_valid_state(state, hero)
    assert result.valid is True
    assert result.reason == "hu_postflop_valid"


def test_three_players_on_flop_invalid():
    state, hero = _hu_sb_bb_flop()
    state.players[3].folded = False
    node = spot_node_from_state(state, hero)
    assert len(node.live_seats) == 3
    result = is_two_player_valid(node)
    assert result.valid is False
    assert "multiway" in result.reason


def test_preflop_collapsed_hu_deferred_v1():
    rng = random.Random(1)
    state = GameState.new_hand(button_seat=3, rng=rng)
    _fold_all_except(state, {4, 5})
    node = spot_node_from_state(state, hero_seat=4)
    assert node.street == Street.PREFLOP
    result = is_two_player_valid(node)
    assert result.valid is False
    assert result.reason == "deferred_preflop_v1"


def test_bunching_postflop_valid():
    state, hero = _hu_sb_bb_flop()
    result = is_two_player_valid_state(state, hero, bunching=True)
    assert result.valid is True
    node = spot_node_from_state(state, hero, bunching=True)
    assert node.spot_class == SpotClass.PSEUDO_HU_BUNCHING


def test_hero_not_live_invalid():
    node = SpotNode(
        street=Street.FLOP,
        live_seats=frozenset({4, 5}),
        folded_seats=frozenset({0, 1, 2, 3}),
        hero_seat=0,
        spot_class=SpotClass.HU_POSTFLOP_2P,
    )
    result = is_two_player_valid(node)
    assert result.valid is False
    assert result.reason == "hero_not_live"


def test_multiway_preflop_rfi_invalid():
    """UTG open with players yet to act — 3+ live."""
    node = SpotNode(
        street=Street.PREFLOP,
        live_seats=frozenset({0, 1, 2, 3, 4, 5}),
        folded_seats=frozenset(),
        hero_seat=0,
        spot_class=SpotClass.MULTIWAY_6MAX,
    )
    result = is_two_player_valid(node)
    assert result.valid is False


def test_inactive_not_folded_invalid():
    node = SpotNode(
        street=Street.FLOP,
        live_seats=frozenset({4, 5}),
        folded_seats=frozenset({0, 1, 2}),
        hero_seat=4,
        spot_class=SpotClass.HU_POSTFLOP_2P,
    )
    result = is_two_player_valid(node)
    assert result.valid is False
    assert result.reason == "inactive_seats_not_folded"
