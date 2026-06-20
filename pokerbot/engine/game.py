"""6-max No-Limit Hold'em game engine."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from enum import Enum, IntEnum, auto
from typing import Iterator

from pokerbot.engine.actions import Action, ActionType, BET_SIZE_BUCKETS
from pokerbot.engine.betting_rules import reopen_step
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
_CHIP = 0.01  # smallest indivisible unit (matches the engine's round(_, 2) bet granularity)


def _split_layer_chips(layer_pot: float, winners: list[int], button_seat: int) -> dict[int, float]:
    """Award a tied pot layer in whole chips (0.01bb), routing the indivisible remainder to the first
    eligible winner left of the button.

    The remainder ROUTING matches the int-chip JAX settlement (``side_pots_jax.award_layer``) exactly —
    proven over odd-chip multiway scenarios by ``tests/test_game_jax_equiv.py``. The one residual
    divergence risk is UPSTREAM, not here: ``layer_pot`` derives from float-BB ``committed_total_bb`` that
    game.py accumulates during play, so a fractional bet (e.g. 1/3 pot) can leave sub-0.01bb float drift
    that ``round(layer_pot / _CHIP)`` resolves differently than the engine's pure-int chips. game.py is a
    float-BB eval engine by convention; the settlement-parity gate feeds clean integer chips and passes."""
    n = len(winners)
    chips = round(layer_pot / _CHIP)
    base, rem = divmod(chips, n)
    out = {w: base * _CHIP for w in winners}
    if rem:
        wset = set(winners)
        odd = next(
            (s for off in range(NUM_PLAYERS)
             for s in ((button_seat + 1 + off) % NUM_PLAYERS,) if s in wset),
            winners[0],
        )
        out[odd] += rem * _CHIP
    return out


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
        # parenthesized: a FOLDED player with chips already in this street must NOT read active
        # (operator precedence made `committed_street_bb > 0` an independent OR term before).
        return not self.folded and (self.stack_bb > 0 or self.committed_street_bb > 0)


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
        sb_paid = self._commit(sb_seat, SB_BB)  # _commit clamps to stack + flags all_in
        bb_paid = self._commit(bb_seat, BB_BB)
        # B3: standing bet = largest actual street commitment, NOT an unconditional full BB (a BB
        # all-in for < 1BB sets to_call to its short commit). _commit already accumulated pot_bb from
        # the clamped amounts -- do NOT overwrite it with SB_BB+BB_BB (which would ignore clamping).
        # For full stacks max(0.5, 1.0) == BB_BB and the pot is 1.5, so this is byte-identical.
        self.current_bet_bb = max(sb_paid, bb_paid)

    def _nth_live_after(self, start: int, n: int) -> int:
        """The n-th non-folded seat clockwise after ``start`` (n >= 1), skipping folded/empty seats.
        Mirrors jax ``_nth_live_after`` so short-handed/occupancy play is correct, not accidental."""
        seat = start
        for _ in range(max(1, n)):
            for _ in range(NUM_PLAYERS):
                seat = (seat + 1) % NUM_PLAYERS
                if not self.players[seat].folded:
                    break
        return seat

    def _first_preflop_actor(self) -> int:
        # BUG-B7: occupancy-aware (was ``button+3`` unconditionally, wrong for non-6-handed). Heads-up =>
        # the button (== SB) acts first; otherwise the 3rd live seat after the button (UTG). At a full
        # ring with no folds this is exactly ``button+3``, so 6-max behaviour is byte-identical.
        live = sum(1 for p in self.players if not p.folded)
        return self._nth_live_after(self.button_seat, 1 if live <= 2 else 3)

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

        # Minimum legal raise TOTAL (above committed) = the call plus the last FULL-raise increment (>= BB),
        # NOT current_bet*2 (which wrongly assumes the next increment equals the whole standing bet).
        min_inc = max(BB_BB, getattr(self, "last_raise_increment", BB_BB))
        min_raise = to_call + min_inc
        if player.stack_bb > to_call:
            for i, frac in enumerate(BET_SIZE_BUCKETS):
                target = max(self.current_bet_bb, self.pot_bb * frac)
                bet_total = target - player.committed_street_bb
                if bet_total >= min_raise and bet_total <= player.stack_bb:
                    atype = ActionType.BET if self.current_bet_bb <= 1e-9 else ActionType.RAISE
                    # Tag the EXACT abstraction menu index (3 + frac position) so the serve path
                    # (action_to_index) recovers the trained bucket instead of mis-bucketing the
                    # absolute-BB amount. The frac IS the bucket here — no pot-fraction re-derivation.
                    actions.append(Action(atype, round(bet_total, 2), bucket=3 + i))
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
            # BUG-B4: the reopen / min-raise update is the ONE canonical rule in betting_rules.reopen_step
            # (mirrors the deployed engine), no longer re-derived here. A FULL raise reopens (resets the
            # acted set); a short all-in under-raise advances current_bet but does NOT reopen.
            prior_min_raise = max(BB_BB, getattr(self, "last_raise_increment", BB_BB))
            res = reopen_step(prev_bet, player.committed_street_bb, prior_min_raise, BB_BB, eps=1e-9)
            if res.raised:
                self.current_bet_bb = res.new_current_bet
                self.last_aggressor = seat
                self.last_raise_increment = res.new_min_raise
                if res.resets_acted:
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
        self.last_raise_increment = BB_BB    # new street: the minimum raise increment resets to one BB
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
        """Return chip deltas in BB per seat for this hand. Side-pot aware: the pot is layered by each
        seat's committed_total, and each layer is awarded to the best hand among the contenders still
        ELIGIBLE for it (committed at least that much). A single largest contributor's uncalled excess is
        returned. (The old code split the WHOLE pot among the overall winners, overpaying a short all-in
        that won the main pot and shorting the side-pot contenders — eval-only: the live JAX serve path
        settles via side_pots_jax, which was already correct.)"""
        if self.active_player_count() == 1:
            winner = next(p.seat for p in self.players if not p.folded)
            return self._award_pot([winner])

        deltas = {i: -self.players[i].committed_total_bb for i in range(NUM_PLAYERS)}
        totals = [self.players[i].committed_total_bb for i in range(NUM_PLAYERS)]
        # Uncalled-bet return: the single largest contribution is only matched up to the 2nd largest.
        srt = sorted(totals)
        if len(srt) >= 2 and srt[-1] > srt[-2] + 1e-9:
            top = max(range(NUM_PLAYERS), key=lambda i: totals[i])
            deltas[top] += totals[top] - srt[-2]   # refund the uncalled excess
            totals[top] = srt[-2]                  # cap it for the layering below

        hole = {p.seat: list(p.hole) for p in self.players if p.hole}
        contenders = [p.seat for p in self.players if not p.folded and p.hole]
        levels = sorted({t for t in totals if t > 1e-9})
        prev = 0.0
        for lvl in levels:
            layer_pot = (lvl - prev) * sum(1 for t in totals if t >= lvl - 1e-9)
            prev = lvl
            if layer_pot <= 1e-12:
                continue
            elig = [s for s in contenders if totals[s] >= lvl - 1e-9]
            if not elig:
                continue   # layer made up purely of folded dead money with no eligible claimant
            win_idx = compare_hands([hole[s] + list(self.board) for s in elig])
            winners = [elig[i] for i in win_idx]
            for w, amt in _split_layer_chips(layer_pot, winners, self.button_seat).items():
                deltas[w] += amt
        return deltas

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
    chance_sink: "callable | None" = None,
) -> tuple[GameState, dict[int, float]]:
    """Play one hand; policies map seat -> callable(state) -> Action.

    ``allin_settle``: optional hook ``(state, n_board_seen) -> deltas | None`` called when betting is
    over (everyone left is all-in). ``n_board_seen`` = board cards the LAST action conditioned on —
    cards dealt after that (``_advance_street`` auto-deals the next street when the closing call
    completes it) are information-free runout luck the hook may integrate out. If it returns a deltas
    dict, the hand settles on that (all-in EV / equity settlement — a mean-preserving variance reducer
    for evals); on None the board runs out as usual. Default None ⇒ behavior unchanged.

    ``chance_sink``: optional hook ``(state, prev_board_len) -> None`` called immediately AFTER an action
    that revealed new board cards (``_advance_street`` dealt them inside ``apply_action``), with
    ``prev_board_len`` = the board length before the reveal. Used by the chance control variate
    (``pokerbot.eval.chance_cv``) to correct board-runout luck; observe-only, never mutates state, so
    default None ⇒ behavior byte-identical."""
    state = state or GameState.new_hand(rng=rng)
    state.deal_hole_cards()
    # BUG-P0: seed from the CURRENT board, not 0. ``play_hand`` may be called on a RESUMED mid-hand state
    # (e.g. an eval subgame root already past preflop) that immediately has no actors remaining (all-in);
    # the top-of-loop ``not actors_remaining()`` branch would then pass ``n_board_seen=0`` to
    # ``allin_settle``, telling it 0 board cards were seen when the board is already dealt -> wrong
    # variance-reduction baseline. A fresh hand has an empty board, so this is identical there.
    n_board_seen = len(state.board)

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
        if chance_sink is not None and len(state.board) > n_board_seen:
            chance_sink(state, n_board_seen)
        if state.is_terminal():
            break
        state.advance_to_next_actor()
    else:
        raise RuntimeError("Hand exceeded maximum action count")

    return state, state.resolve_showdown()
