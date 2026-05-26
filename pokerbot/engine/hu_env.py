"""Heads-up NLH environment (50BB eval track for AlphaNLHoldem)."""

from __future__ import annotations

import random
from dataclasses import dataclass

from pokerbot.engine.actions import Action, ActionType, BET_SIZE_BUCKETS
from pokerbot.engine.cards import Card, Deck
from pokerbot.engine.game import GameState, PlayerState, Position, Street, play_hand
from pokerbot.engine.hand_eval import compare_hands


STARTING_STACK_BB = 50.0
SB_BB = 0.5
BB_BB = 1.0


@dataclass
class HUState:
    """Minimal 2-player state compatible with bot interface."""

    game: GameState
    hero_seat: int = 0

    @classmethod
    def new_hand(cls, rng: random.Random | None = None, hero_seat: int = 0) -> HUState:
        rng = rng or random.Random()
        g = GameState.new_hand(button_seat=0 if hero_seat == 1 else 1, rng=rng)
        for i in range(2, 6):
            g.players[i].folded = True
        g.players[0].stack_bb = STARTING_STACK_BB - g.players[0].committed_total_bb
        g.players[1].stack_bb = STARTING_STACK_BB - g.players[1].committed_total_bb
        g.deal_hole_cards()
        return cls(game=g, hero_seat=hero_seat)

    def legal_actions(self, seat: int | None = None) -> list[Action]:
        seat = self.game.action_seat if seat is None else seat
        return self.game.legal_actions(seat)

    def apply_action(self, action: Action, seat: int | None = None) -> None:
        self.game.apply_action(action, seat)
        if not self.game.is_terminal():
            self.game.advance_to_next_actor()

    def is_terminal(self) -> bool:
        return self.game.is_terminal()

    def resolve(self) -> dict[int, float]:
        if not self.game.is_terminal():
            if not self.game.actors_remaining():
                self.game.run_out_board()
        return self.game.resolve_showdown()


def play_hu_hand(
    hero_policy,
    villain_policy,
    rng: random.Random | None = None,
    hero_seat: int = 0,
) -> tuple[HUState, dict[int, float]]:
    rng = rng or random.Random()
    policies = {0: hero_policy, 1: villain_policy}
    if hero_seat == 1:
        policies = {1: hero_policy, 0: villain_policy}

    state = HUState.new_hand(rng, hero_seat=hero_seat)
    g = state.game

    max_actions = 300
    for _ in range(max_actions):
        if g.is_terminal():
            break
        if not g.actors_remaining():
            g.run_out_board()
            break
        seat = g.action_seat
        if seat > 1 or g.players[seat].folded or g.players[seat].all_in:
            g.advance_to_next_actor()
            continue
        policy = policies.get(seat)
        action = policy(g) if policy else Action(ActionType.FOLD)
        legal = g.legal_actions(seat)
        if action not in legal:
            action = next(
                (a for a in legal if a.action_type in (ActionType.CHECK, ActionType.CALL, ActionType.FOLD)),
                Action(ActionType.FOLD),
            )
        g.apply_action(action)
        if g.is_terminal():
            break
        g.advance_to_next_actor()

    return state, g.resolve_showdown()
