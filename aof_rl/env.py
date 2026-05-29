"""AoF preflop push/fold hand environment."""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from aof_ranges import active_seats

from aof_rl.payoff import compute_payoffs

ACTION_FOLD = 0
ACTION_ALL_IN = 1


def payoff_step_seed(hand_id: int, guard: int) -> int:
    """MC payoff seed for one env step (league rollout + batched collector)."""
    return hand_id * 1009 + guard

RANKS = "23456789TJQKA"
SUITS = "cdhs"

_HAND_COUNTER = 0


def _new_hand_id() -> int:
    global _HAND_COUNTER
    _HAND_COUNTER += 1
    return _HAND_COUNTER


def _full_deck() -> list[str]:
    return [f"{r}{s}" for r in RANKS for s in SUITS]


def _post_blinds(seats: tuple[str, ...], stack_bb: float) -> dict[str, float]:
    """Return initial invested bb per seat (blinds only)."""
    inv = {s: 0.0 for s in seats}
    if "SB" in seats:
        inv["SB"] = min(0.5, stack_bb)
    if "BB" in seats:
        inv["BB"] = min(1.0, stack_bb)
    return inv


@dataclass
class AoFHandState:
    table_size: int
    stack_bb: float
    seats: tuple[str, ...]
    holes: dict[str, tuple[str, str]]
    folded: set[str] = field(default_factory=set)
    all_in: set[str] = field(default_factory=set)
    acted: set[str] = field(default_factory=set)
    shover: str | None = None
    action_seat: str | None = None
    invested: dict[str, float] = field(default_factory=dict)
    hand_id: int = 0
    terminal: bool = False
    payoffs: dict[str, float] = field(default_factory=dict)
    _rng: random.Random = field(default_factory=random.Random, repr=False)

    def copy_for_decision(self) -> AoFHandState:
        return AoFHandState(
            table_size=self.table_size,
            stack_bb=self.stack_bb,
            seats=self.seats,
            holes=dict(self.holes),
            folded=set(self.folded),
            all_in=set(self.all_in),
            acted=set(self.acted),
            shover=self.shover,
            action_seat=self.action_seat,
            invested=dict(self.invested),
            hand_id=self.hand_id,
            terminal=self.terminal,
            payoffs=dict(self.payoffs),
            _rng=self._rng,
        )

    def facing_shove(self) -> bool:
        return self.shover is not None

    def players_behind(self, hero: str) -> tuple[str, ...]:
        if not self.shover or hero not in self.seats:
            return ()
        order = self.seats
        hi = order.index(hero)
        return tuple(s for s in order[hi + 1 :] if s not in self.folded and s not in self.acted)

    def legal_actions(self) -> list[int]:
        if self.terminal or self.action_seat is None:
            return []
        return [ACTION_FOLD, ACTION_ALL_IN]

    def is_terminal(self) -> bool:
        return self.terminal


def shuffle_deck(seed: int) -> list[str]:
    """Reproducible full-deck order from integer seed (no global RNG)."""
    deck = _full_deck()
    random.Random(seed).shuffle(deck)
    return deck


def reset_hand_from_holes(
    seats: tuple[str, ...],
    holes: dict[str, tuple[str, str]],
    *,
    table_size: int,
    stack_bb: float = 8.0,
    hand_id: int | None = None,
) -> AoFHandState:
    """Start a hand with fixed hole cards and blind posting."""
    invested = _post_blinds(seats, stack_bb)
    state = AoFHandState(
        table_size=table_size,
        stack_bb=stack_bb,
        seats=seats,
        holes=dict(holes),
        invested=invested,
        hand_id=hand_id if hand_id is not None else _new_hand_id(),
    )
    state.action_seat = _next_actor(state)
    if state.action_seat is None:
        _finalize(state)
    return state


def reset_hand_from_deal_index(
    deal_index: int,
    *,
    table_size: int,
    stack_bb: float = 8.0,
    base_seed: int = 0,
) -> AoFHandState:
    """Deal hands deterministically from ``base_seed ^ deal_index``."""
    seats = active_seats(table_size)
    deck = shuffle_deck(base_seed ^ (deal_index * 0x9E3779B9))
    holes: dict[str, tuple[str, str]] = {}
    idx = 0
    for s in seats:
        holes[s] = (deck[idx], deck[idx + 1])
        idx += 2
    return reset_hand_from_holes(
        seats,
        holes,
        table_size=table_size,
        stack_bb=stack_bb,
        hand_id=deal_index,
    )


def reset_hand(
    rng: random.Random,
    *,
    table_size: int = 4,
    stack_bb: float = 8.0,
) -> AoFHandState:
    seats = active_seats(table_size)
    deck = _full_deck()
    rng.shuffle(deck)
    holes: dict[str, tuple[str, str]] = {}
    idx = 0
    for s in seats:
        holes[s] = (deck[idx], deck[idx + 1])
        idx += 2

    invested = _post_blinds(seats, stack_bb)
    state = AoFHandState(
        table_size=table_size,
        stack_bb=stack_bb,
        seats=seats,
        holes=holes,
        invested=invested,
        hand_id=_new_hand_id(),
        _rng=rng,
    )
    state.action_seat = _next_actor(state)
    if state.action_seat is None:
        _finalize(state)
    return state


def _next_actor(state: AoFHandState) -> str | None:
    alive_not_acted = [
        s
        for s in state.seats
        if s not in state.folded and s not in state.acted
    ]
    if not alive_not_acted:
        return None
    return alive_not_acted[0]


def step_hand(
    state: AoFHandState,
    action: int,
    *,
    payoff_samples: int = 4000,
    payoff_seed: int | None = None,
) -> AoFHandState:
    if state.terminal:
        raise RuntimeError("Hand already terminal")
    seat = state.action_seat
    if seat is None:
        raise RuntimeError("No actor")

    if action not in (ACTION_FOLD, ACTION_ALL_IN):
        raise ValueError(f"Invalid action {action}")

    state.acted.add(seat)

    if action == ACTION_FOLD:
        state.folded.add(seat)
    else:
        state.all_in.add(seat)
        to_add = state.stack_bb - state.invested[seat]
        state.invested[seat] = state.stack_bb
        if state.shover is None:
            state.shover = seat

    # Everyone folded to one player
    alive = [s for s in state.seats if s not in state.folded]
    if len(alive) == 1:
        _finalize(state, payoff_samples=payoff_samples, payoff_seed=payoff_seed)
        return state

    state.action_seat = _next_actor(state)
    if state.action_seat is None:
        _finalize(state, payoff_samples=payoff_samples, payoff_seed=payoff_seed)
    return state


def _finalize(
    state: AoFHandState,
    *,
    payoff_samples: int = 4000,
    payoff_seed: int | None = None,
) -> None:
    from contextlib import nullcontext

    from aof_rl.profiling.league_profiler import get_profiler

    seed = payoff_seed if payoff_seed is not None else payoff_step_seed(state.hand_id, 0)
    alive = [s for s in state.seats if s not in state.folded]
    if len(alive) == 2:
        bucket = "rollout.step_payoff_hu"
    elif len(alive) > 2:
        bucket = "rollout.step_payoff_multiway"
    else:
        bucket = "rollout.step_payoff"
    prof = get_profiler()
    ctx = prof.measure(bucket) if prof is not None else nullcontext()
    with ctx:
        state.payoffs = compute_payoffs(state, samples=payoff_samples, seed=seed)
    state.terminal = True
    state.action_seat = None


def play_hand_to_completion(
    policies: dict[str, callable],
    rng: random.Random,
    *,
    table_size: int = 4,
    stack_bb: float = 8.0,
    payoff_samples: int = 4000,
) -> AoFHandState:
    """Drive a hand until terminal using per-seat policy callables."""
    state = reset_hand(rng, table_size=table_size, stack_bb=stack_bb)
    guard = 0
    while not state.terminal:
        guard += 1
        if guard > 32:
            raise RuntimeError("Hand action guard exceeded")
        seat = state.action_seat
        if seat is None:
            break
        fn = policies.get(seat)
        if fn is None:
            action = ACTION_FOLD
        else:
            action = int(fn(state, seat))
        seed = payoff_step_seed(state.hand_id, guard)
        state = step_hand(state, action, payoff_samples=payoff_samples, payoff_seed=seed)
    return state
