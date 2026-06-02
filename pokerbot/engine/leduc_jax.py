"""Leduc Hold'em JAX environment for SD-CFR validation."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import Array

NUM_PLAYERS = 2
NUM_ACTIONS = 3  # fold, call/check, raise
NUM_CARD_RANKS = 3
MAX_HISTORY = 16
ANTE = 1
RAISE_SIZE = 2
MAX_RAISES = 2

FOLD = 0
CALL = 1
RAISE = 2

ROUND_PRIVATE = 0
ROUND_PUBLIC = 1


class LeducState(NamedTuple):
    cards: Array  # (2,) private ranks 0..2
    public: Array  # () -1 until dealt
    pot: Array
    stacks: Array  # (2,)
    committed: Array  # (2,) street commitments
    invested: Array  # (2,) total chips put in this hand
    current_bet: Array
    raises: Array
    round: Array
    acting: Array
    acted: Array  # (2,)
    done: Array
    rewards: Array
    rng_key: Array
    hist_action: Array
    hist_len: Array
    fixed_public: Array  # () predetermined public card when >= 0 (tabular enumeration)


def _hand_score(card: Array, public: Array) -> Array:
    return jnp.where(card == public, jnp.int32(10 + card), card)


def _resolve_showdown(state: LeducState) -> LeducState:
    s0 = _hand_score(state.cards[0], state.public)
    s1 = _hand_score(state.cards[1], state.public)
    half = state.pot // 2
    rem = state.pot - half * 2
    inv = state.invested
    r = -inv
    r = r.at[0].set(
        jnp.where(
            s0 > s1,
            state.pot - inv[0],
            jnp.where(s0 == s1, half + rem - inv[0], -inv[0]),
        )
    )
    r = r.at[1].set(state.pot - inv[0] - inv[1] - r[0])
    return state._replace(done=jnp.bool_(True), rewards=r)


def init_hand(rng_key: Array) -> LeducState:
    key, k_deck = jax.random.split(rng_key)
    deck = jax.random.permutation(k_deck, jnp.array([0, 0, 1, 1, 2, 2], dtype=jnp.int32))
    return LeducState(
        cards=deck[:2],
        public=jnp.int32(-1),
        pot=jnp.int32(ANTE * 2),
        stacks=jnp.array([100 - ANTE, 100 - ANTE], dtype=jnp.int32),
        committed=jnp.array([ANTE, ANTE], dtype=jnp.int32),
        invested=jnp.array([ANTE, ANTE], dtype=jnp.int32),
        current_bet=jnp.int32(ANTE),
        raises=jnp.int32(0),
        round=jnp.int32(ROUND_PRIVATE),
        acting=jnp.int32(0),
        acted=jnp.zeros(NUM_PLAYERS, dtype=bool),
        done=jnp.bool_(False),
        rewards=jnp.zeros(NUM_PLAYERS, dtype=jnp.int32),
        rng_key=key,
        hist_action=jnp.full(MAX_HISTORY, -1, dtype=jnp.int32),
        hist_len=jnp.int32(0),
        fixed_public=jnp.int32(-1),
    )


def init_from_deal(cards: tuple[int, int], fixed_public: int) -> LeducState:
    """Deterministic deal for exhaustive tabular CFR (private cards + public card)."""
    return LeducState(
        cards=jnp.array(cards, dtype=jnp.int32),
        public=jnp.int32(-1),
        pot=jnp.int32(ANTE * 2),
        stacks=jnp.array([100 - ANTE, 100 - ANTE], dtype=jnp.int32),
        committed=jnp.array([ANTE, ANTE], dtype=jnp.int32),
        invested=jnp.array([ANTE, ANTE], dtype=jnp.int32),
        current_bet=jnp.int32(ANTE),
        raises=jnp.int32(0),
        round=jnp.int32(ROUND_PRIVATE),
        acting=jnp.int32(0),
        acted=jnp.zeros(NUM_PLAYERS, dtype=bool),
        done=jnp.bool_(False),
        rewards=jnp.zeros(NUM_PLAYERS, dtype=jnp.int32),
        rng_key=jnp.zeros(2, dtype=jnp.uint32),
        hist_action=jnp.full(MAX_HISTORY, -1, dtype=jnp.int32),
        hist_len=jnp.int32(0),
        fixed_public=jnp.int32(fixed_public),
    )


def legal_actions(state: LeducState) -> Array:
    seat = state.acting
    to_call = jnp.maximum(0, state.current_bet - state.committed[seat])
    can_raise = (state.raises < MAX_RAISES) & (state.stacks[seat] > to_call)
    return jnp.array([True, True, can_raise], dtype=bool)


def _round_complete(state: LeducState) -> Array:
    return (state.committed[0] == state.committed[1]) & state.acted[0] & state.acted[1]


def _start_public_round(state: LeducState) -> LeducState:
    key, sub = jax.random.split(state.rng_key)
    deck = jax.random.permutation(sub, jnp.array([0, 0, 1, 1, 2, 2], dtype=jnp.int32))
    public = jnp.where(state.fixed_public >= 0, state.fixed_public, deck[0])
    return state._replace(
        public=public,
        round=jnp.int32(ROUND_PUBLIC),
        current_bet=jnp.int32(0),
        committed=jnp.zeros(NUM_PLAYERS, dtype=jnp.int32),
        invested=state.invested,
        raises=jnp.int32(0),
        acted=jnp.zeros(NUM_PLAYERS, dtype=bool),
        acting=jnp.int32(1),
        rng_key=key,
    )


def apply_action(state: LeducState, action: int) -> LeducState:
    seat = state.acting
    to_call = jnp.maximum(0, state.current_bet - state.committed[seat])

    def do_fold(s: LeducState) -> LeducState:
        winner = 1 - seat
        r = (-s.invested).at[winner].add(s.pot)
        return s._replace(done=jnp.bool_(True), rewards=r)

    def do_call(s: LeducState) -> LeducState:
        pay = jnp.minimum(to_call, s.stacks[seat])
        ns = s.stacks.at[seat].add(-pay)
        nc = s.committed.at[seat].add(pay)
        ni = s.invested.at[seat].add(pay)
        return s._replace(
            stacks=ns,
            committed=nc,
            invested=ni,
            pot=s.pot + pay,
            acting=1 - seat,
            acted=s.acted.at[seat].set(True),
        )

    def do_raise(s: LeducState) -> LeducState:
        target = s.current_bet + RAISE_SIZE
        pay = jnp.minimum(target - s.committed[seat], s.stacks[seat])
        nc = s.committed.at[seat].add(pay)
        ni = s.invested.at[seat].add(pay)
        return s._replace(
            stacks=s.stacks.at[seat].add(-pay),
            committed=nc,
            invested=ni,
            pot=s.pot + pay,
            current_bet=nc[seat],
            raises=s.raises + 1,
            acting=1 - seat,
            acted=jnp.zeros(NUM_PLAYERS, dtype=bool).at[seat].set(True),
        )

    state = jax.lax.switch(action, [do_fold, do_call, do_raise], state)
    private_complete = (
        _round_complete(state) & (~state.done) & (state.round == ROUND_PRIVATE)
    )
    state = jax.lax.cond(
        private_complete,
        _start_public_round,
        lambda s: s,
        state,
    )
    public_complete = (
        _round_complete(state) & (~state.done) & (state.round == ROUND_PUBLIC)
    )
    state = jax.lax.cond(
        public_complete,
        _resolve_showdown,
        lambda s: s,
        state,
    )
    idx = state.hist_len
    return state._replace(
        hist_action=state.hist_action.at[idx].set(action),
        hist_len=idx + 1,
    )


def is_terminal(state: LeducState) -> Array:
    return state.done


def returns(state: LeducState) -> Array:
    return state.rewards
