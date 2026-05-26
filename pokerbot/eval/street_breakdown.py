"""Per-street P&L breakdown for head-to-head evaluation."""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass, field

from pokerbot.engine.actions import Action, ActionType
from pokerbot.engine.game import NUM_PLAYERS, GameState, Street, play_hand


STREETS = (Street.PREFLOP, Street.FLOP, Street.TURN, Street.RIVER)


def _street_key(street: Street) -> str:
    return street.name if street.name in {s.name for s in STREETS} else Street.RIVER.name


@dataclass
class StreetBreakdown:
    hands: int = 0
    total_bb: float = 0.0
    pnl_by_street: dict[str, float] = field(default_factory=lambda: {s.name: 0.0 for s in STREETS})
    invested_by_street: dict[str, float] = field(default_factory=lambda: {s.name: 0.0 for s in STREETS})
    folds_by_street: dict[str, int] = field(default_factory=lambda: {s.name: 0 for s in STREETS})
    showdowns_by_street: dict[str, int] = field(default_factory=lambda: {s.name: 0 for s in STREETS})

    @property
    def bb_per_100(self) -> float:
        return (self.total_bb / max(self.hands, 1)) * 100

    def to_dict(self) -> dict:
        return {
            "hands": self.hands,
            "total_bb": round(self.total_bb, 2),
            "bb_per_100": round(self.bb_per_100, 2),
            "pnl_by_street_bb100": {
                k: round(v / max(self.hands, 1) * 100, 2) for k, v in self.pnl_by_street.items()
            },
            "invested_by_street_bb100": {
                k: round(v / max(self.hands, 1) * 100, 2) for k, v in self.invested_by_street.items()
            },
            "folds_by_street": dict(self.folds_by_street),
            "showdowns_by_street": dict(self.showdowns_by_street),
        }


def _hero_terminal_street(state: GameState, hero_seat: int) -> Street:
    hero = state.players[hero_seat]
    if hero.folded:
        for seat, action in reversed(state.action_history()):
            if seat == hero_seat and action.action_type == ActionType.FOLD:
                # Walk back to find street at fold time by replaying (cheap for 6-max).
                return _street_at_hero_fold(state, hero_seat)
    return _max_street_reached(state)


def _street_at_hero_fold(state: GameState, hero_seat: int) -> Street:
    """Replay actions to find the street when hero folded."""
    replay = GameState.new_hand(button_seat=state.button_seat, rng=random.Random(0))
    replay.deal_hole_cards()
    for seat, action in state.action_history():
        if replay.is_terminal():
            break
        if replay.players[seat].folded or replay.players[seat].all_in:
            replay.advance_to_next_actor()
            continue
        replay.apply_action(action)
        if seat == hero_seat and action.action_type == ActionType.FOLD:
            return replay.street
        replay.advance_to_next_actor()
    return state.street


def _max_street_reached(state: GameState) -> Street:
    if len(state.board) >= 5:
        return Street.RIVER
    if len(state.board) >= 4:
        return Street.TURN
    if len(state.board) >= 3:
        return Street.FLOP
    return Street.PREFLOP


def _hero_invested_by_street(state: GameState, hero_seat: int) -> dict[str, float]:
    invested = {s.name: 0.0 for s in STREETS}
    replay = GameState.new_hand(button_seat=state.button_seat, rng=random.Random(0))
    replay.deal_hole_cards()
    for seat, action in state.action_history():
        if replay.is_terminal():
            break
        if replay.players[seat].folded or replay.players[seat].all_in:
            replay.advance_to_next_actor()
            continue
        before = replay.players[hero_seat].committed_total_bb
        replay.apply_action(action)
        if seat == hero_seat:
            after = replay.players[hero_seat].committed_total_bb
            street = replay.street.name
            if street in invested:
                invested[street] += after - before
        replay.advance_to_next_actor()
    return invested


def run_street_breakdown(
    hero_policy,
    opponent_policy,
    hands: int = 500,
    seed: int = 42,
) -> StreetBreakdown:
    rng = random.Random(seed)
    result = StreetBreakdown()

    for h in range(hands):
        hero_seat = h % NUM_PLAYERS
        policies = {i: opponent_policy for i in range(NUM_PLAYERS)}
        policies[hero_seat] = hero_policy

        state, deltas = play_hand(policies, rng=rng)
        hand_delta = deltas.get(hero_seat, 0.0)
        terminal = _hero_terminal_street(state, hero_seat)
        invested = _hero_invested_by_street(state, hero_seat)
        term_key = _street_key(terminal)

        result.hands += 1
        result.total_bb += hand_delta
        result.pnl_by_street[term_key] += hand_delta
        for street, amt in invested.items():
            result.invested_by_street[street] += amt

        if state.players[hero_seat].folded:
            result.folds_by_street[term_key] += 1
        elif not state.players[hero_seat].folded and state.active_player_count() >= 1:
            max_st = _max_street_reached(state)
            result.showdowns_by_street[max_st.name] += 1

    return result


def evaluate_street_breakdown_vs_bots(
    hero_policy,
    bot_names: list[str],
    hands_per_bot: int = 500,
    seed: int = 42,
) -> dict[str, dict]:
    from pokerbot.eval.bots import BOT_REGISTRY

    out: dict[str, dict] = {}
    for name in bot_names:
        cls = BOT_REGISTRY[name]
        opp = cls(random.Random(seed))
        bd = run_street_breakdown(hero_policy, opp, hands=hands_per_bot, seed=seed)
        out[name] = bd.to_dict()
    return out
