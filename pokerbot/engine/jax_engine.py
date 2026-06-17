"""JAX-vectorized 6-max No-Limit Hold'em environment (Phase 2a CFR-capable).

Pure-functional transitions, fixed-shape int32 chip math (1 BB = 100 chips),
11-action vocab aligned with ``pokerbot.engine.actions``.
"""

from __future__ import annotations

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
from jax import Array

from pokerbot.engine.hand_eval_jax import card_rank
from pokerbot.engine.side_pots_jax import SEAT_ALLIN, SEAT_FOLDED, SEAT_LIVE, compute_returns

SEAT_EMPTY = jnp.int8(0)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NUM_PLAYERS: int = 6
NUM_ACTIONS: int = 11
NUM_STREETS: int = 4
MAX_BOARD: int = 5
MAX_HISTORY: int = 64
MAX_POTS: int = NUM_PLAYERS
CHIP_UNIT: int = 100
STARTING_STACK_BB: float = 100.0
STARTING_STACK_CHIPS: int = int(STARTING_STACK_BB * CHIP_UNIT)
SB_CHIPS: int = CHIP_UNIT // 2
BB_CHIPS: int = CHIP_UNIT
BET_SIZE_BUCKETS: tuple[float, ...] = (1.0 / 3, 0.50, 0.66, 0.75, 1.0, 1.5, 2.0)
INFO_FEATURE_DIM: int = 256
HISTORY_FEATURE_DIM: int = 128

FOLD = 0
CHECK = 1
CALL = 2
ALL_IN = NUM_ACTIONS - 1
BET_BUCKET_START = 3

STREET_PREFLOP = 0
STREET_FLOP = 1
STREET_TURN = 2
STREET_RIVER = 3
STREET_TERMINAL = 4
STREET_CHANCE = -1  # internal: waiting to deal next street

BET_BUCKETS = jnp.array(BET_SIZE_BUCKETS, dtype=jnp.float32)


class EnvState(NamedTuple):
    holes: Array  # (6, 2) int32
    board: Array  # (5,) int32
    seat_status: Array  # (6,) int8
    stacks: Array  # (6,) int32
    committed_street: Array  # (6,) int32
    committed_total: Array  # (6,) int32
    street: Array  # () int32
    pot_dead: Array  # () int32
    current_bet: Array  # () int32
    min_raise: Array  # () int32
    button: Array  # () int32
    action_seat: Array  # () int32, -1 terminal/chance
    last_aggressor: Array  # () int32
    acted: Array  # (6,) bool
    last_raise_full: Array  # () bool
    deck: Array  # (52,) int32
    deck_ptr: Array  # () int32
    rng_key: Array  # (2,) uint32
    hist_seat: Array  # (MAX_HISTORY,) int32
    hist_action: Array  # (MAX_HISTORY,) int32
    hist_amount: Array  # (MAX_HISTORY,) int32
    hist_len: Array  # () int32
    done: Array  # () bool
    rewards: Array  # (6,) int32
    pending_deal: Array  # () bool — street closed, cards not yet dealt


class TimeStep(NamedTuple):
    observation: Array
    legal_mask: Array
    state: EnvState
    reward: Array
    discount: Array
    step_type: Array


# ---------------------------------------------------------------------------
# Seat helpers
# ---------------------------------------------------------------------------


def _is_live(seat_status: Array) -> Array:
    return (seat_status == SEAT_LIVE) | (seat_status == SEAT_ALLIN)


def _can_act(state: EnvState, seat: Array) -> Array:
    return state.seat_status[seat] == SEAT_LIVE


def _live_count(seat_status: Array) -> Array:
    return jnp.sum((seat_status == SEAT_LIVE) | (seat_status == SEAT_ALLIN))


def _pot_for_sizing(state: EnvState) -> Array:
    return state.pot_dead + jnp.sum(state.committed_street)


# ---------------------------------------------------------------------------
# Actor order (respects occupancy / empty seats)
# ---------------------------------------------------------------------------


def _next_seat_cw(from_seat: Array, offset: Array) -> Array:
    return (from_seat + offset) % NUM_PLAYERS


def _first_live_after(seat_status: Array, from_seat: Array) -> Array:
    offsets = jnp.arange(1, NUM_PLAYERS + 1)
    seats = (from_seat + offsets) % NUM_PLAYERS
    live = seat_status[seats] == SEAT_LIVE
    idx = jnp.argmax(live)
    return jnp.where(jnp.any(live), seats[idx], jnp.int32(-1))


def _nth_live_after(seat_status: Array, from_seat: Array, n: Array) -> Array:
    """Return the n-th live seat clockwise after from_seat (n>=1)."""

    def step(_: Array, carry: tuple[Array, Array]) -> tuple[Array, Array]:
        seat, remaining = carry
        seat = _first_live_after(seat_status, seat)
        remaining = remaining - 1
        return seat, remaining

    seat, _ = jax.lax.fori_loop(0, n, step, (from_seat, n))
    return seat


def _first_postflop_actor(state: EnvState) -> Array:
    return _nth_live_after(state.seat_status, state.button, 1)


def _next_actor(state: EnvState) -> Array:
    return _first_live_after(state.seat_status, state.action_seat)


# ---------------------------------------------------------------------------
# Street / chance
# ---------------------------------------------------------------------------


@jax.jit
def is_chance_node(state: EnvState) -> Array:
    return state.pending_deal & (~state.done)


def is_terminal(state: EnvState) -> Array:
    return state.done


def current_player(state: EnvState) -> Array:
    return jnp.where(state.done | state.pending_deal, jnp.int32(-1), state.action_seat)


def _street_complete(state: EnvState) -> Array:
    can_act = jax.vmap(lambda s: _can_act(state, s))(jnp.arange(NUM_PLAYERS))
    n_can_act = jnp.sum(can_act)
    matched = jnp.all(jnp.where(can_act, state.committed_street >= state.current_bet, True))
    all_acted = jnp.all(jnp.where(can_act, state.acted, True))
    back_to_aggressor = state.action_seat == state.last_aggressor
    no_aggressor = state.last_aggressor < 0
    # Close when nobody can act (everyone all-in/folded) or all actors matched + acted.
    closed = (n_can_act == 0) | (matched & all_acted & (back_to_aggressor | no_aggressor))
    return closed


def _deal_n_cards(state: EnvState, n_new: Array) -> EnvState:
    ptr = state.deck_ptr
    n_board = jnp.sum(state.board >= 0)

    def write_one(i: Array, carry: tuple[Array, Array, Array]) -> tuple[Array, Array, Array]:
        board, p, key = carry
        key, sub = jax.random.split(key)
        card = state.deck[p]
        should = i < n_new
        board = board.at[n_board + i].set(jnp.where(should, card, board[n_board + i]))
        p = p + should.astype(jnp.int32)
        return board, p, key

    board, ptr, key = jax.lax.fori_loop(0, 3, write_one, (state.board, ptr, state.rng_key))
    return state._replace(board=board, deck_ptr=ptr, rng_key=key, pending_deal=jnp.bool_(False))


@jax.jit
def sample_chance(state: EnvState) -> EnvState:
    """Deal next street after betting closed; pure given rng_key."""

    new_street = state.street + 1
    n_new = jnp.where(new_street == STREET_FLOP, 3, jnp.where(new_street <= STREET_RIVER, 1, 0))
    state = state._replace(street=new_street)
    state = jax.lax.cond(
        n_new > 0,
        lambda s: _deal_n_cards(s, n_new),
        lambda s: s._replace(pending_deal=jnp.bool_(False)),
        state,
    )
    new_street = jnp.where(new_street > STREET_RIVER, STREET_TERMINAL, state.street)
    actor = jax.lax.cond(
        new_street >= STREET_TERMINAL,
        lambda _: jnp.int32(-1),
        lambda _: _first_postflop_actor(state),
        None,
    )
    return state._replace(
        street=new_street,
        action_seat=actor,
        committed_street=jnp.zeros(NUM_PLAYERS, dtype=jnp.int32),
        current_bet=jnp.int32(0),
        min_raise=jnp.int32(BB_CHIPS),
        acted=jnp.zeros(NUM_PLAYERS, dtype=bool),
        last_aggressor=jnp.int32(-1),
        last_raise_full=jnp.bool_(True),
    )


def _run_out(state: EnvState) -> EnvState:
    def write_one(_: Array, carry: EnvState) -> EnvState:
        s = carry
        nb = jnp.sum(s.board >= 0)
        should = nb < MAX_BOARD
        card = s.deck[s.deck_ptr]
        board = s.board.at[nb].set(jnp.where(should, card, s.board[nb]))
        ptr = s.deck_ptr + should.astype(jnp.int32)
        return s._replace(board=board, deck_ptr=ptr)

    state = jax.lax.fori_loop(0, MAX_BOARD, write_one, state)
    return state._replace(street=STREET_TERMINAL, pending_deal=jnp.bool_(False))


def _close_street(state: EnvState) -> EnvState:
    pot_dead = state.pot_dead + jnp.sum(state.committed_street)
    state = state._replace(
        pot_dead=pot_dead,
        committed_street=jnp.zeros(NUM_PLAYERS, dtype=jnp.int32),
        current_bet=jnp.int32(0),
        pending_deal=jnp.bool_(True),
        action_seat=jnp.int32(-1),
    )
    actors = jnp.any(jax.vmap(lambda s: _can_act(state, s))(jnp.arange(NUM_PLAYERS)))
    state = jax.lax.cond(
        (~actors) & (state.street < STREET_RIVER),
        _run_out,
        lambda s: s,
        state,
    )
    return state


# ---------------------------------------------------------------------------
# Betting
# ---------------------------------------------------------------------------


def _commit(state: EnvState, seat: Array, amount: Array) -> EnvState:
    amount = jnp.minimum(amount, state.stacks[seat])
    stacks = state.stacks.at[seat].add(-amount)
    committed_street = state.committed_street.at[seat].add(amount)
    committed_total = state.committed_total.at[seat].add(amount)
    all_in = stacks[seat] <= 0
    seat_status = jnp.where(all_in, state.seat_status.at[seat].set(SEAT_ALLIN), state.seat_status)
    return state._replace(
        stacks=stacks,
        committed_street=committed_street,
        committed_total=committed_total,
        seat_status=seat_status,
    )


@jax.jit
def legal_action_mask(state: EnvState, seat: Array | None = None) -> Array:
    seat = state.action_seat if seat is None else seat
    status = state.seat_status[seat]
    stack = state.stacks[seat]
    committed = state.committed_street[seat]
    to_call = jnp.maximum(0, state.current_bet - committed)
    pot = _pot_for_sizing(state)

    mask = jnp.zeros(NUM_ACTIONS, dtype=bool)
    mask = mask.at[FOLD].set(to_call > 0)
    mask = mask.at[CHECK].set(to_call == 0)
    mask = mask.at[CALL].set(to_call > 0)

    min_raise_amt = state.min_raise
    full_raise_target = state.current_bet + min_raise_amt

    def bucket_legal(i: Array) -> Array:
        frac = BET_BUCKETS[i]
        target_total = jnp.maximum(state.current_bet, (pot * frac).astype(jnp.int32))
        raise_by = target_total - committed
        is_full = target_total >= full_raise_target
        reopen_ok = (~state.acted[seat]) | state.last_raise_full | (to_call == 0)
        return (
            (status == SEAT_LIVE)
            & (stack > to_call)
            & (raise_by >= min_raise_amt)
            & (raise_by <= stack)
            & reopen_ok
        )

    bucket_mask = jax.vmap(bucket_legal)(jnp.arange(len(BET_SIZE_BUCKETS)))
    mask = mask.at[BET_BUCKET_START : BET_BUCKET_START + len(BET_SIZE_BUCKETS)].set(bucket_mask)
    reopen_ok = (~state.acted[seat]) | state.last_raise_full | (to_call == 0)
    mask = mask.at[ALL_IN].set((status == SEAT_LIVE) & (stack > 0) & reopen_ok)
    mask = jnp.where(status == SEAT_LIVE, mask, False)
    mask = jnp.where(jnp.any(mask), mask, mask.at[CHECK].set(True))
    return mask


legal_actions = legal_action_mask


def _action_amount(state: EnvState, seat: Array, action: Array) -> Array:
    committed = state.committed_street[seat]
    to_call = jnp.maximum(0, state.current_bet - committed)
    stack = state.stacks[seat]
    pot = _pot_for_sizing(state)

    def bucket_amt(i: Array) -> Array:
        frac = BET_BUCKETS[i]
        target = jnp.maximum(state.current_bet, (pot * frac).astype(jnp.int32))
        return jnp.maximum(0, target - committed)

    bucket_amounts = jax.vmap(bucket_amt)(jnp.arange(len(BET_SIZE_BUCKETS)))
    bucket_idx = action - BET_BUCKET_START
    bucket_val = jnp.where(
        (action >= BET_BUCKET_START) & (action < ALL_IN),
        bucket_amounts[bucket_idx],
        0,
    )
    amount = jnp.where(action == CALL, to_call, 0)
    amount = jnp.where(action == ALL_IN, stack, amount)
    amount = jnp.where((action >= BET_BUCKET_START) & (action < ALL_IN), to_call + bucket_val, amount)
    return amount


@jax.jit
def step_state(state: EnvState, action: Array) -> EnvState:
    return apply_action(state, action)


@jax.jit
def apply_action(state: EnvState, action: Array) -> EnvState:
    seat = state.action_seat
    amount = _action_amount(state, seat, action)

    seat_status = jnp.where(action == FOLD, state.seat_status.at[seat].set(SEAT_FOLDED), state.seat_status)
    state = state._replace(seat_status=seat_status)

    is_call = action == CALL
    is_aggressive = (action >= BET_BUCKET_START) | (action == ALL_IN)
    state = jax.lax.cond(is_call | is_aggressive, lambda s: _commit(s, seat, amount), lambda s: s, state)

    prev_bet = state.current_bet
    new_street_bet = state.committed_street[seat]
    raised = is_aggressive & (new_street_bet > prev_bet)
    full_raise = raised & (new_street_bet >= prev_bet + state.min_raise)
    acted = state.acted.at[seat].set(True)
    acted = jnp.where(full_raise, jnp.zeros(NUM_PLAYERS, dtype=bool).at[seat].set(True), acted)
    current_bet = jnp.where(raised, new_street_bet, state.current_bet)
    raise_size = jnp.where(raised, new_street_bet - prev_bet, state.min_raise)
    min_raise = jnp.where(raised, jnp.maximum(BB_CHIPS, raise_size), state.min_raise)
    last_aggressor = jnp.where(raised, seat, state.last_aggressor)
    last_raise_full = jnp.where(raised, full_raise, state.last_raise_full)

    state = state._replace(
        acted=acted,
        current_bet=current_bet,
        min_raise=min_raise,
        last_aggressor=last_aggressor,
        last_raise_full=last_raise_full,
    )

    idx = state.hist_len
    state = state._replace(
        hist_seat=state.hist_seat.at[idx].set(seat),
        hist_action=state.hist_action.at[idx].set(action),
        hist_amount=state.hist_amount.at[idx].set(amount),
        hist_len=jnp.minimum(idx + 1, MAX_HISTORY),
    )

    n_live = _live_count(state.seat_status)
    terminal = n_live <= 1

    state = jax.lax.cond(
        terminal,
        lambda s: s._replace(street=STREET_TERMINAL, pending_deal=jnp.bool_(False)),
        lambda s: jax.lax.cond(_street_complete(s), _close_street, lambda x: x, s),
        state,
    )

    terminal = terminal | (state.street >= STREET_TERMINAL) | (n_live <= 1)
    state = state._replace(
        action_seat=jax.lax.cond(terminal | state.pending_deal, lambda: jnp.int32(-1), lambda: _next_actor(state)),
        done=terminal,
    )

    rets = jax.lax.cond(
        terminal,
        lambda s: compute_returns(
            holes=s.holes,
            board=s.board,
            seat_status=s.seat_status,
            committed_total=s.committed_total,
            button=s.button,
            n_showdown=n_live,
        ),
        lambda _: jnp.zeros(NUM_PLAYERS, dtype=jnp.int32),
        state,
    )
    return state._replace(rewards=rets)


def returns(state: EnvState) -> Array:
    """Terminal per-seat net chip deltas (int32). Only valid when ``state.done``."""
    return state.rewards


# ---------------------------------------------------------------------------
# Init
# ---------------------------------------------------------------------------


def _post_blinds(stacks: Array, seat_status: Array, button: Array) -> tuple[Array, Array, Array, Array, Array]:
    sb = _nth_live_after(seat_status, button, 1)
    bb = _nth_live_after(seat_status, button, 2)
    committed = jnp.zeros(NUM_PLAYERS, dtype=jnp.int32)
    sb_live = seat_status[sb] != SEAT_EMPTY
    bb_live = seat_status[bb] != SEAT_EMPTY
    stacks = stacks.at[sb].add(-SB_CHIPS * sb_live.astype(jnp.int32))
    stacks = stacks.at[bb].add(-BB_CHIPS * bb_live.astype(jnp.int32))
    committed = committed.at[sb].set(SB_CHIPS * sb_live.astype(jnp.int32))
    committed = committed.at[bb].set(BB_CHIPS * bb_live.astype(jnp.int32))
    pot_dead = jnp.int32(0)
    current_bet = jnp.int32(BB_CHIPS)
    min_raise = jnp.int32(BB_CHIPS)
    return stacks, committed, pot_dead, current_bet, min_raise


@jax.jit
def reset_state(
    key: Array,
    button: Array | None = None,
    occupancy: Array | None = None,
) -> EnvState:
    return init_hand(key, button, None, occupancy)


@jax.jit
def init_hand(
    rng_key: Array,
    button: Array | None = None,
    stacks: Array | None = None,
    occupancy: Array | None = None,
) -> EnvState:
    key, k_deck, k_btn = jax.random.split(rng_key, 3)
    button = jax.random.randint(k_btn, (), 0, NUM_PLAYERS) if button is None else button

    if occupancy is None:
        seat_status = jnp.full(NUM_PLAYERS, SEAT_LIVE, dtype=jnp.int8)
    else:
        seat_status = jnp.where(occupancy, SEAT_LIVE, SEAT_EMPTY)

    deck = jax.random.permutation(k_deck, jnp.arange(52, dtype=jnp.int32))
    if stacks is None:
        stacks = jnp.full(NUM_PLAYERS, STARTING_STACK_CHIPS, dtype=jnp.int32)
    else:
        stacks = stacks.astype(jnp.int32)

    stacks, committed, pot_dead, current_bet, min_raise = _post_blinds(stacks, seat_status, button)
    ptr = jnp.int32(0)
    holes = jnp.full((NUM_PLAYERS, 2), -1, dtype=jnp.int32)

    def deal_hole(i: Array, carry: tuple[Array, Array]) -> tuple[Array, Array]:
        h, p = carry
        live = seat_status[i] != SEAT_EMPTY
        c0 = deck[p]
        c1 = deck[p + 1]
        h = h.at[i].set(jnp.where(live, jnp.array([c0, c1], dtype=jnp.int32), h[i]))
        p = p + (2 * live.astype(jnp.int32))
        return h, p

    holes, ptr = jax.lax.fori_loop(0, NUM_PLAYERS, deal_hole, (holes, ptr))

    n_live = jnp.sum(seat_status == SEAT_LIVE)
    utg = _nth_live_after(seat_status, button, 3)
    btn_first = _nth_live_after(seat_status, button, 1)
    action_seat = jax.lax.cond(n_live <= 2, lambda: btn_first, lambda: utg)

    return EnvState(
        holes=holes,
        board=jnp.full(MAX_BOARD, -1, dtype=jnp.int32),
        seat_status=seat_status,
        stacks=stacks,
        committed_street=committed,
        committed_total=committed,
        street=jnp.int32(STREET_PREFLOP),
        pot_dead=pot_dead,
        current_bet=current_bet,
        min_raise=min_raise,
        button=button,
        action_seat=action_seat,
        last_aggressor=jnp.int32(-1),
        acted=jnp.zeros(NUM_PLAYERS, dtype=bool),
        last_raise_full=jnp.bool_(True),
        deck=deck,
        deck_ptr=ptr,
        rng_key=key,
        hist_seat=jnp.full(MAX_HISTORY, -1, dtype=jnp.int32),
        hist_action=jnp.full(MAX_HISTORY, -1, dtype=jnp.int32),
        hist_amount=jnp.zeros(MAX_HISTORY, dtype=jnp.int32),
        hist_len=jnp.int32(0),
        done=jnp.bool_(False),
        rewards=jnp.zeros(NUM_PLAYERS, dtype=jnp.int32),
        pending_deal=jnp.bool_(False),
    )


# ---------------------------------------------------------------------------
# Feature encoders (normalize by STARTING_STACK_CHIPS)
# ---------------------------------------------------------------------------


def _one_hot(index: Array, size: int) -> Array:
    return jax.nn.one_hot(index, size, dtype=jnp.float32)


def _folded_mask(seat_status: Array) -> Array:
    return (seat_status == SEAT_FOLDED).astype(jnp.float32)


def _all_in_mask(seat_status: Array) -> Array:
    return (seat_status == SEAT_ALLIN).astype(jnp.float32)


@jax.jit
def encode_info_state(state: EnvState, seat: Array) -> Array:
    scale = float(STARTING_STACK_CHIPS)
    pot_live = _pot_for_sizing(state)
    parts = [
        _one_hot(jnp.maximum(state.street, 0), NUM_STREETS + 1),
        _one_hot(seat, NUM_PLAYERS),
        _one_hot(state.button, NUM_PLAYERS),
        state.stacks.astype(jnp.float32) / scale,
        state.committed_street.astype(jnp.float32) / scale,
        state.committed_total.astype(jnp.float32) / scale,
        _folded_mask(state.seat_status),
        _all_in_mask(state.seat_status),
        jnp.array([pot_live.astype(jnp.float32) / scale, state.current_bet.astype(jnp.float32) / scale]),
        (state.holes[seat] >= 0).astype(jnp.float32) * (card_rank(state.holes[seat]) / 14.0),
        (state.board >= 0).astype(jnp.float32) * (card_rank(state.board) / 14.0),
    ]
    feat = jnp.concatenate([p.flatten() for p in parts])
    pad = jnp.zeros(INFO_FEATURE_DIM, dtype=jnp.float32)
    return jax.lax.dynamic_update_slice(pad, feat, (0,))


HISTORY_SLOTS: int = 32


@jax.jit
def encode_history_state(state: EnvState) -> Array:
    scale = float(STARTING_STACK_CHIPS)
    pot_live = _pot_for_sizing(state)
    seats = state.hist_seat[:HISTORY_SLOTS].astype(jnp.float32) / NUM_PLAYERS
    actions = state.hist_action[:HISTORY_SLOTS].astype(jnp.float32) / NUM_ACTIONS
    amounts = state.hist_amount[:HISTORY_SLOTS].astype(jnp.float32) / scale
    hist = jnp.concatenate([seats, actions, amounts])
    pad = jnp.zeros(HISTORY_FEATURE_DIM, dtype=jnp.float32)
    pad = jax.lax.dynamic_update_slice(pad, hist, (0,))
    ctx = jnp.array(
        [
            pot_live.astype(jnp.float32) / (scale * NUM_PLAYERS),
            state.current_bet.astype(jnp.float32) / scale,
            jnp.maximum(state.street, 0).astype(jnp.float32) / NUM_STREETS,
            state.hist_len.astype(jnp.float32) / MAX_HISTORY,
        ],
        dtype=jnp.float32,
    )
    return jax.lax.dynamic_update_slice(pad, ctx, (HISTORY_FEATURE_DIM - ctx.shape[0],))


class JAXNLHEEnv:
    def __init__(self, num_envs: int = 1) -> None:
        self.num_envs = num_envs

    @property
    def observation_spec(self) -> dict[str, tuple[int, ...]]:
        return {"info_state": (INFO_FEATURE_DIM,), "legal_mask": (NUM_ACTIONS,)}

    @property
    def action_spec(self) -> dict[str, Any]:
        return {"num_actions": NUM_ACTIONS}

    def reset(self, key: Array) -> TimeStep:
        if self.num_envs == 1:
            state = reset_state(key)
        else:
            keys = jax.random.split(key, self.num_envs)
            state = jax.vmap(reset_state)(keys)
        return self._pack_timestep(state)

    def step(self, state: EnvState, action: Array) -> TimeStep:
        if self.num_envs == 1:
            next_state = step_state(state, action)
        else:
            next_state = jax.vmap(step_state)(state, action)
        return self._pack_timestep(next_state)

    def _pack_timestep(self, state: EnvState) -> TimeStep:
        if self.num_envs == 1:
            obs = encode_info_state(state, state.action_seat)
            mask = legal_action_mask(state)
            reward = jnp.float32(0.0)
            discount = jnp.where(state.done, 0.0, 1.0)
            step_type = jnp.where(state.done, 1, 0).astype(jnp.int32)
        else:
            obs = jax.vmap(encode_info_state)(state, state.action_seat)
            mask = jax.vmap(legal_action_mask)(state)
            reward = jnp.zeros(self.num_envs, dtype=jnp.float32)
            discount = jnp.where(state.done, 0.0, 1.0)
            step_type = jnp.where(state.done, 1, 0).astype(jnp.int32)
        return TimeStep(obs, mask, state, reward, discount, step_type)


batched_reset = jax.jit(jax.vmap(reset_state))
batched_step = jax.jit(jax.vmap(step_state))
batched_legal_mask = jax.jit(jax.vmap(legal_action_mask))
batched_info_state = jax.jit(jax.vmap(encode_info_state, in_axes=(0, 0)))

__all__ = [
    "SEAT_EMPTY",
    "SEAT_ALLIN",
    "SEAT_FOLDED",
    "SEAT_LIVE",
    "EnvState",
    "TimeStep",
    "JAXNLHEEnv",
    "batched_reset",
    "batched_step",
    "batched_legal_mask",
    "batched_info_state",
    "init_hand",
    "apply_action",
    "legal_actions",
    "legal_action_mask",
    "returns",
    "current_player",
    "is_chance_node",
    "is_terminal",
    "sample_chance",
    "reset_state",
    "step_state",
    "encode_info_state",
    "encode_history_state",
]
