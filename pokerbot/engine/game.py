"""6-max No-Limit Hold'em game engine."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from enum import Enum, IntEnum, auto
from typing import Iterator

from pokerbot.engine.actions import Action, ActionType, BET_SIZE_BUCKETS
from pokerbot.engine.cards import Card, Deck
from pokerbot.engine.hand_eval import compare_hands


class Street(Enum):
    PREFLOP = auto()
    FLOP = auto()
    TURN = auto()
    RIVER = auto()
    SHOWDOWN = auto()


class Position(IntEnum):
    UTG = 0
    HJ = 1
    CO = 2
    BTN = 3
    SB = 4
    BB = 5

    @classmethod
    def from_str(cls, name: str) -> Position:
        return cls[name.upper()]


NUM_PLAYERS = 6
STARTING_STACK_BB = 100.0
SB_BB = 0.5
BB_BB = 1.0


@dataclass
class PlayerState:
    seat: int
    position: Position
    stack_bb: float = STARTING_STACK_BB
    hole: tuple[Card, Card] | None = None
    folded: bool = False
    all_in: bool = False
    committed_street_bb: float = 0.0
    committed_total_bb: float = 0.0

    @property
    def active(self) -> bool:
        return not self.folded and self.stack_bb > 0 or self.committed_street_bb > 0


@dataclass
class GameState:
    """Mutable 6-max NLH table state."""

    players: list[PlayerState] = field(default_factory=list)
    board: list[Card] = field(default_factory=list)
    street: Street = Street.PREFLOP
    pot_bb: float = 0.0
    current_bet_bb: float = 0.0
    button_seat: int = 3
    action_seat: int = 0
    last_aggressor: int | None = None
    hand_number: int = 0
    _deck: Deck | None = field(default=None, repr=False)
    _rng: random.Random = field(default_factory=random.Random, repr=False)
    _acted_this_street: set[int] = field(default_factory=set, repr=False)
    _history: list[tuple[int, Action]] = field(default_factory=list, repr=False)

    @classmethod
    def new_hand(cls, button_seat: int = 3, rng: random.Random | None = None) -> GameState:
        rng = rng or random.Random()
        deck = Deck(rng)
        state = cls(button_seat=button_seat, _deck=deck, _rng=rng)
        pos_order = [Position.BTN, Position.SB, Position.BB, Position.UTG, Position.HJ, Position.CO]
        state.players = []
        for i in range(NUM_PLAYERS):
            offset = (i - button_seat) % NUM_PLAYERS
            state.players.append(PlayerState(seat=i, position=pos_order[offset]))
        state._post_blinds()
        state.action_seat = state._first_preflop_actor()
        return state

    def _post_blinds(self) -> None:
        sb_seat = (self.button_seat + 1) % NUM_PLAYERS
        bb_seat = (self.button_seat + 2) % NUM_PLAYERS
        self._commit(sb_seat, SB_BB)
        self._commit(bb_seat, BB_BB)
        self.current_bet_bb = BB_BB
        self.pot_bb = SB_BB + BB_BB

    def _first_preflop_actor(self) -> int:
        return (self.button_seat + 3) % NUM_PLAYERS

    def _first_postflop_actor(self) -> int:
        for offset in range(1, NUM_PLAYERS + 1):
            seat = (self.button_seat + offset) % NUM_PLAYERS
            if not self.players[seat].folded and not self.players[seat].all_in:
                return seat
        return self.button_seat

    def _commit(self, seat: int, amount_bb: float) -> float:
        player = self.players[seat]
        amount_bb = min(amount_bb, player.stack_bb)
        player.stack_bb -= amount_bb
        player.committed_street_bb += amount_bb
        player.committed_total_bb += amount_bb
        self.pot_bb += amount_bb
        if player.stack_bb <= 1e-9:
            player.all_in = True
        return amount_bb

    def deal_hole_cards(self) -> None:
        assert self._deck is not None
        for p in self.players:
            hole = self._deck.deal(2)
            p.hole = (hole[0], hole[1])

    def active_player_count(self) -> int:
        return sum(1 for p in self.players if not p.folded)

    def players_in_hand(self) -> list[PlayerState]:
        return [p for p in self.players if not p.folded]

    def legal_actions(self, seat: int | None = None) -> list[Action]:
        seat = self.action_seat if seat is None else seat
        player = self.players[seat]
        if player.folded or player.all_in:
            return []

        to_call = max(0.0, self.current_bet_bb - player.committed_street_bb)
        actions: list[Action] = [Action(ActionType.FOLD)]

        if to_call <= 1e-9:
            actions.append(Action(ActionType.CHECK))
        else:
            actions.append(Action(ActionType.CALL))

        min_raise = max(BB_BB, self.current_bet_bb * 2 - player.committed_street_bb)
        if player.stack_bb > to_call:
            for frac in BET_SIZE_BUCKETS:
                target = max(self.current_bet_bb, self.pot_bb * frac)
                bet_total = target - player.committed_street_bb
                if bet_total >= min_raise and bet_total <= player.stack_bb:
                    atype = ActionType.BET if self.current_bet_bb <= 1e-9 else ActionType.RAISE
                    actions.append(Action(atype, round(bet_total, 2)))
            if player.stack_bb > to_call:
                actions.append(Action(ActionType.ALL_IN, player.stack_bb))
        return actions

    def apply_action(self, action: Action, seat: int | None = None) -> None:
        seat = self.action_seat if seat is None else seat
        player = self.players[seat]
        self._history.append((seat, action))

        if action.action_type == ActionType.FOLD:
            player.folded = True
        elif action.action_type == ActionType.CHECK:
            pass
        elif action.action_type == ActionType.CALL:
            to_call = self.current_bet_bb - player.committed_street_bb
            self._commit(seat, to_call)
        elif action.action_type in (ActionType.BET, ActionType.RAISE, ActionType.ALL_IN):
            if action.action_type == ActionType.ALL_IN:
                amount = player.stack_bb
            else:
                amount = action.amount_bb
            to_call = self.current_bet_bb - player.committed_street_bb
            prev_bet = self.current_bet_bb
            total = to_call + amount
            self._commit(seat, total)
            if player.committed_street_bb > prev_bet + 1e-9:
                self.current_bet_bb = player.committed_street_bb
                self.last_aggressor = seat
                self._acted_this_street = {seat}
        else:
            raise ValueError(f"Unknown action {action}")

        if seat not in self._acted_this_street:
            self._acted_this_street.add(seat)
        if self.active_player_count() <= 1:
            self.street = Street.SHOWDOWN
            return
        if self._street_complete():
            self._advance_street()

    def _street_complete(self) -> bool:
        # NOTE: no ``len(active) <= 1`` shortcut — it skipped the LAST live player's response to an
        # all-in raise (the jammer goes all_in ⇒ drops out of ``active`` ⇒ the lone remaining player
        # was never asked to call/fold and free-rolled the showdown for their blind). Betting on a
        # street is complete iff EVERY live (non-folded, non-all-in) player has acted and matched the
        # current bet; with zero live players (everyone all-in) the loop is empty ⇒ complete (run out).
        active = [p for p in self.players if not p.folded and not p.all_in]
        for p in active:
            if p.seat not in self._acted_this_street:
                return False
            if abs(p.committed_street_bb - self.current_bet_bb) > 1e-9:
                return False
        return True

    def _advance_street(self) -> None:
        if self.active_player_count() <= 1:
            self.street = Street.SHOWDOWN
            return
        for p in self.players:
            p.committed_street_bb = 0.0
        self.current_bet_bb = 0.0
        self._acted_this_street.clear()

        if self.street == Street.PREFLOP:
            assert self._deck is not None
            self.board.extend(self._deck.deal(3))
            self.street = Street.FLOP
        elif self.street == Street.FLOP:
            assert self._deck is not None
            self.board.extend(self._deck.deal(1))
            self.street = Street.TURN
        elif self.street == Street.TURN:
            assert self._deck is not None
            self.board.extend(self._deck.deal(1))
            self.street = Street.RIVER
        else:
            self.street = Street.SHOWDOWN
            return

        self.action_seat = self._first_postflop_actor()

    def advance_to_next_actor(self) -> None:
        if self.street == Street.SHOWDOWN:
            return
        for _ in range(NUM_PLAYERS):
            self.action_seat = (self.action_seat + 1) % NUM_PLAYERS
            p = self.players[self.action_seat]
            if not p.folded and not p.all_in:
                return

    def actors_remaining(self) -> bool:
        return any(not p.folded and not p.all_in for p in self.players)

    def run_out_board(self) -> None:
        """Deal remaining community cards when no further betting is possible."""
        assert self._deck is not None
        while self.street != Street.SHOWDOWN:
            if self.street == Street.PREFLOP:
                self.board.extend(self._deck.deal(3))
                self.street = Street.FLOP
            elif self.street == Street.FLOP:
                self.board.extend(self._deck.deal(1))
                self.street = Street.TURN
            elif self.street == Street.TURN:
                self.board.extend(self._deck.deal(1))
                self.street = Street.RIVER
            else:
                self.street = Street.SHOWDOWN
                break

    def is_terminal(self) -> bool:
        return self.street == Street.SHOWDOWN or self.active_player_count() <= 1

    def resolve_showdown(self) -> dict[int, float]:
        """Return chip deltas in BB per seat for this hand."""
        if self.active_player_count() == 1:
            winner = next(p.seat for p in self.players if not p.folded)
            return self._award_pot([winner])

        hands = []
        seats = []
        for p in self.players:
            if not p.folded and p.hole:
                seats.append(p.seat)
                hands.append(list(p.hole) + list(self.board))
        winners = compare_hands(hands)
        winner_seats = [seats[i] for i in winners]
        return self._award_pot(winner_seats)

    def _award_pot(self, winner_seats: list[int]) -> dict[int, float]:
        share = self.pot_bb / len(winner_seats)
        deltas = {i: -self.players[i].committed_total_bb for i in range(NUM_PLAYERS)}
        for w in winner_seats:
            deltas[w] += share
        return deltas

    def action_history(self) -> list[tuple[int, Action]]:
        return list(self._history)

    def clone(self) -> GameState:
        import copy

        return copy.deepcopy(self)


def play_hand(
    policies: dict[int, callable],
    state: GameState | None = None,
    rng: random.Random | None = None,
    allin_settle: "callable | None" = None,
) -> tuple[GameState, dict[int, float]]:
    """Play one hand; policies map seat -> callable(state) -> Action.

    ``allin_settle``: optional hook ``(state, n_board_seen) -> deltas | None`` called when betting is
    over (everyone left is all-in). ``n_board_seen`` = board cards the LAST action conditioned on —
    cards dealt after that (``_advance_street`` auto-deals the next street when the closing call
    completes it) are information-free runout luck the hook may integrate out. If it returns a deltas
    dict, the hand settles on that (all-in EV / equity settlement — a mean-preserving variance reducer
    for evals); on None the board runs out as usual. Default None ⇒ behavior unchanged."""
    state = state or GameState.new_hand(rng=rng)
    state.deal_hole_cards()
    n_board_seen = 0

    def _runout_or_settle():
        if allin_settle is not None:
            deltas = allin_settle(state, n_board_seen)
            if deltas is not None:
                return deltas
        state.run_out_board()
        return None

    max_actions = 500
    for _ in range(max_actions):
        if state.is_terminal():
            break
        if not state.actors_remaining():
            settled = _runout_or_settle()
            if settled is not None:
                return state, settled
            break
        seat = state.action_seat
        if state.players[seat].folded or state.players[seat].all_in:
            state.advance_to_next_actor()
            if not state.actors_remaining():
                settled = _runout_or_settle()
                if settled is not None:
                    return state, settled
                break
            continue
        policy = policies.get(seat)
        if policy is None:
            legal = state.legal_actions()
            action = legal[0] if legal else Action(ActionType.FOLD)
        else:
            action = policy(state)
            legal = state.legal_actions()
            if action not in legal:
                action = next(
                    (a for a in legal if a.action_type in (ActionType.CHECK, ActionType.CALL, ActionType.FOLD)),
                    Action(ActionType.FOLD),
                )
        n_board_seen = len(state.board)
        state.apply_action(action)
        if state.is_terminal():
            break
        state.advance_to_next_actor()
    else:
        raise RuntimeError("Hand exceeded maximum action count")

    return state, state.resolve_showdown()
