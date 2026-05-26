"""JAX-vectorized 6-max No-Limit Hold'em environment.

All game state is represented as fixed-shape ``jnp.ndarray`` fields suitable
for ``jax.jit`` / ``jax.vmap`` without Python fallbacks.  Side pots, terminal
payoffs, and legal action masking are handled entirely in JAX.
"""

from __future__ import annotations

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
from jax import Array

# ---------------------------------------------------------------------------
# Constants (mirror pokerbot.engine.actions / game)
# ---------------------------------------------------------------------------

NUM_PLAYERS: int = 6
NUM_ACTIONS: int = 11
NUM_STREETS: int = 4
MAX_BOARD: int = 5
MAX_HISTORY: int = 64
MAX_POTS: int = NUM_PLAYERS
STARTING_STACK_BB: float = 100.0
SB_BB: float = 0.5
BB_BB: float = 1.0
BET_SIZE_BUCKETS: tuple[float, ...] = (0.33, 0.50, 0.66, 0.75, 1.0, 1.5, 2.0)
INFO_FEATURE_DIM: int = 256
HISTORY_FEATURE_DIM: int = 128

# Action indices
FOLD = 0
CHECK = 1
CALL = 2
ALL_IN = NUM_ACTIONS - 1
BET_BUCKET_START = 3

# Street indices
STREET_PREFLOP = 0
STREET_FLOP = 1
STREET_TURN = 2
STREET_RIVER = 3
STREET_TERMINAL = 4

BET_BUCKETS = jnp.array(BET_SIZE_BUCKETS, dtype=jnp.float32)


class EnvState(NamedTuple):
    """Rigidly sized poker table state."""

    holes: Array  # (6, 2) int32 card indices 0-51, -1 unknown
    board: Array  # (5,) int32, -1 empty
    stacks: Array  # (6,) float32 remaining stack
    committed_street: Array  # (6,) float32
    committed_total: Array  # (6,) float32
    folded: Array  # (6,) bool
    all_in: Array  # (6,) bool
    street: Array  # () int32 0..4
    pot: Array  # () float32
    current_bet: Array  # () float32
    button: Array  # () int32
    action_seat: Array  # () int32
    last_aggressor: Array  # () int32, -1 none
    acted: Array  # (6,) bool acted this street
    deck: Array  # (52,) int32 permutation of 0..51
    deck_ptr: Array  # () int32 next deal index
    hist_seat: Array  # (MAX_HISTORY,) int32
    hist_action: Array  # (MAX_HISTORY,) int32
    hist_amount: Array  # (MAX_HISTORY,) float32
    hist_len: Array  # () int32
    done: Array  # () bool
    rewards: Array  # (6,) float32 terminal chip deltas


class TimeStep(NamedTuple):
    """DeepMind-style environment transition."""

    observation: Array  # (INFO_FEATURE_DIM,) for acting player
    legal_mask: Array  # (NUM_ACTIONS,) bool
    state: EnvState
    reward: Array  # () float32 hero reward (unused during hand)
    discount: Array  # () float32
    step_type: Array  # () int32 0=mid, 1=last


# ---------------------------------------------------------------------------
# Card / hand evaluation (pure JAX)
# ---------------------------------------------------------------------------


def _card_rank(card: Array) -> Array:
    return card // 4 + 2


def _card_suit(card: Array) -> Array:
    return card % 4


# All straight rank windows (high to low); last row is ace-low wheel.
_STRAIGHT_WINDOWS = jnp.array(
    [
        [14, 13, 12, 11, 10],
        [13, 12, 11, 10, 9],
        [12, 11, 10, 9, 8],
        [11, 10, 9, 8, 7],
        [10, 9, 8, 7, 6],
        [9, 8, 7, 6, 5],
        [8, 7, 6, 5, 4],
        [7, 6, 5, 4, 3],
        [6, 5, 4, 3, 2],
        [14, 5, 4, 3, 2],
    ],
    dtype=jnp.int32,
)


def _straight_high(ranks: Array) -> Array:
    """Return high card of best straight, or 0 if none."""

    def window_high(window: Array) -> Array:
        hits = jnp.array([jnp.any(ranks == window[i]) for i in range(5)])
        matched = jnp.all(hits)
        high = jnp.where(window[0] == 14, jnp.where(window[1] == 5, 5, 14), window[0])
        return jnp.where(matched, high, 0)

    highs = jax.vmap(window_high)(_STRAIGHT_WINDOWS)
    return jnp.max(highs)


def _score_five(ranks: Array, suits: Array) -> Array:
    """Return comparable int64 score for five cards (ranks/suits shape (5,))."""

    ranks = jnp.sort(ranks)[::-1]
    counts = jnp.bincount(ranks, length=15)
    is_flush = jnp.all(suits == suits[0])
    straight_high = _straight_high(ranks)
    is_straight = straight_high > 0

    count_vals = jnp.sort(counts)[::-1]
    quad_rank = jnp.argmax(counts == 4)
    trips_rank = jnp.argmax(counts == 3)
    pair_ranks = jnp.nonzero(counts == 2, size=2, fill_value=0)[0]
    high_card_score = jnp.sum(ranks * jnp.array([15**4, 15**3, 15**2, 15, 1], dtype=jnp.float32)).astype(jnp.int64)

    cat = jnp.int32(0)
    primary = high_card_score

    cat = jnp.where(is_flush & is_straight, 8, cat)
    primary = jnp.where(is_flush & is_straight, straight_high, primary)
    cat = jnp.where((count_vals[0] == 4) & (cat < 8), 7, cat)
    primary = jnp.where((count_vals[0] == 4) & (cat < 8), quad_rank * 100 + jnp.max(ranks), primary)
    cat = jnp.where((count_vals[0] == 3) & (count_vals[1] >= 2) & (cat < 7), 6, cat)
    primary = jnp.where(
        (count_vals[0] == 3) & (count_vals[1] >= 2) & (cat < 7),
        trips_rank * 100 + pair_ranks[0],
        primary,
    )
    cat = jnp.where(is_flush & ~is_straight & (cat < 6), 5, cat)
    primary = jnp.where(is_flush & ~is_straight & (cat < 6), high_card_score, primary)
    cat = jnp.where(is_straight & ~is_flush & (cat < 5), 4, cat)
    primary = jnp.where(is_straight & ~is_flush & (cat < 5), straight_high, primary)
    cat = jnp.where((count_vals[0] == 3) & (count_vals[1] < 2) & (cat < 4), 3, cat)
    primary = jnp.where((count_vals[0] == 3) & (count_vals[1] < 2) & (cat < 4), trips_rank * 10000 + high_card_score, primary)
    cat = jnp.where((count_vals[0] == 2) & (count_vals[1] == 2) & (cat < 3), 2, cat)
    high_pair = jnp.maximum(pair_ranks[0], pair_ranks[1])
    low_pair = jnp.minimum(pair_ranks[0], pair_ranks[1])
    primary = jnp.where(
        (count_vals[0] == 2) & (count_vals[1] == 2) & (cat < 3),
        high_pair * 10000 + low_pair * 100 + high_card_score,
        primary,
    )
    cat = jnp.where((count_vals[0] == 2) & (count_vals[1] < 2) & (cat < 2), 1, cat)
    primary = jnp.where((count_vals[0] == 2) & (count_vals[1] < 2) & (cat < 2), pair_ranks[0] * 10000 + high_card_score, primary)

    return (cat.astype(jnp.int64) << 32) + primary.astype(jnp.int64)


def _evaluate_seven(cards: Array) -> Array:
    """Best 5-card score from seven card indices."""

    def score_combo(idx: Array) -> Array:
        chosen = cards[idx]
        ranks = _card_rank(chosen)
        suits = _card_suit(chosen)
        return _score_five(ranks, suits)

    # 21 five-card combinations from 7 cards (fixed unroll)
    combos = jnp.array(
        [
            [0, 1, 2, 3, 4],
            [0, 1, 2, 3, 5],
            [0, 1, 2, 3, 6],
            [0, 1, 2, 4, 5],
            [0, 1, 2, 4, 6],
            [0, 1, 2, 5, 6],
            [0, 1, 3, 4, 5],
            [0, 1, 3, 4, 6],
            [0, 1, 3, 5, 6],
            [0, 1, 4, 5, 6],
            [0, 2, 3, 4, 5],
            [0, 2, 3, 4, 6],
            [0, 2, 3, 5, 6],
            [0, 2, 4, 5, 6],
            [0, 3, 4, 5, 6],
            [1, 2, 3, 4, 5],
            [1, 2, 3, 4, 6],
            [1, 2, 3, 5, 6],
            [1, 2, 4, 5, 6],
            [1, 3, 4, 5, 6],
            [2, 3, 4, 5, 6],
        ],
        dtype=jnp.int32,
    )
    scores = jax.vmap(score_combo)(combos)
    return jnp.max(scores)


# ---------------------------------------------------------------------------
# Side pots & showdown
# ---------------------------------------------------------------------------


def _side_pot_payoffs(state: EnvState) -> Array:
    """Compute terminal chip deltas including side pots."""

    active = ~state.folded
    n_active = jnp.sum(active)
    contributions = state.committed_total

    def single_winner(_: Any) -> Array:
        winner = jnp.argmax(active.astype(jnp.int32))
        deltas = -contributions
        deltas = deltas.at[winner].add(jnp.sum(contributions))
        return deltas

    def multi_showdown(_: Any) -> Array:
        board_cards = state.board
        board_valid = board_cards >= 0
        board_filled = jnp.where(board_valid, board_cards, 0)

        def player_score(seat: Array) -> Array:
            hole = state.holes[seat]
            seven = jnp.concatenate([hole, board_filled])
            valid_mask = jnp.concatenate([jnp.ones(2, bool), board_valid])
            seven = jnp.where(valid_mask, seven, 0)
            return _evaluate_seven(seven)

        scores = jax.vmap(player_score)(jnp.arange(NUM_PLAYERS))
        scores = jnp.where(active, scores, -1)

        # Layered side pots by contribution level
        levels = jnp.sort(jnp.unique(contributions, size=MAX_POTS, fill_value=-1.0))
        levels = jnp.where(levels >= 0, levels, jnp.inf)
        deltas = jnp.zeros(NUM_PLAYERS, dtype=jnp.float32)
        prev = jnp.float32(0.0)

        def pot_step(carry: Array, level: Array) -> tuple[Array, None]:
            deltas_acc = carry
            level = jnp.minimum(level, jnp.max(contributions))
            layer = level - prev
            eligible = active & (contributions >= level - 1e-5)
            pot_size = layer * jnp.sum(eligible.astype(jnp.float32))
            best = jnp.max(jnp.where(eligible, scores, -1))
            winners = eligible & (scores == best) & (best >= 0)
            n_winners = jnp.maximum(jnp.sum(winners), 1)
            share = pot_size / n_winners.astype(jnp.float32)
            deltas_acc = deltas_acc + winners.astype(jnp.float32) * share
            return deltas_acc, None

        winnings, _ = jax.lax.scan(pot_step, deltas, levels)
        return winnings - contributions

    return jax.lax.cond(n_active <= 1, single_winner, multi_showdown, None)


# ---------------------------------------------------------------------------
# Betting logic
# ---------------------------------------------------------------------------


def _active_count(folded: Array) -> Array:
    return jnp.sum(~folded)


def _seat_active(state: EnvState, seat: Array) -> Array:
    return (~state.folded[seat]) & (~state.all_in[seat])


def _next_actor(state: EnvState) -> Array:
    seat = state.action_seat
    def cond_fn(i: Array) -> Array:
        return i < NUM_PLAYERS
    def body_fn(i: Array) -> Array:
        s = (seat + 1 + i) % NUM_PLAYERS
        return i + 1
    # find first active seat after current
    offsets = jnp.arange(NUM_PLAYERS)
    seats = (state.action_seat + 1 + offsets) % NUM_PLAYERS
    active = jax.vmap(lambda s: _seat_active(state, s))(seats)
    idx = jnp.argmax(active)
    return jnp.where(jnp.any(active), seats[idx], state.action_seat)


def _first_postflop_actor(state: EnvState) -> Array:
    offsets = jnp.arange(1, NUM_PLAYERS + 1)
    seats = (state.button + offsets) % NUM_PLAYERS
    active = jax.vmap(lambda s: _seat_active(state, s))(seats)
    idx = jnp.argmax(active)
    return jnp.where(jnp.any(active), seats[idx], state.button)


def _first_preflop_actor(button: Array) -> Array:
    return (button + 3) % NUM_PLAYERS


def _street_complete(state: EnvState) -> Array:
    active = jax.vmap(lambda s: _seat_active(state, s))(jnp.arange(NUM_PLAYERS))
    n_active = jnp.sum(active)
    all_acted = jnp.all(jnp.where(active, state.acted, True))
    matched = jnp.all(jnp.where(active, state.committed_street >= state.current_bet - 1e-5, True))
    return (n_active <= 1) | (all_acted & matched)


def _deal_n_cards(state: EnvState, n_new: Array) -> EnvState:
    """Deal up to three community cards (static loop for JIT)."""

    ptr = state.deck_ptr
    n_board = jnp.sum(state.board >= 0)

    def write_one(i: Array, carry: tuple[Array, Array]) -> tuple[Array, Array]:
        board, p = carry
        card = state.deck[p]
        should = i < n_new
        board = board.at[n_board + i].set(jnp.where(should, card, board[n_board + i]))
        p = p + should.astype(jnp.int32)
        return board, p

    new_board, new_ptr = jax.lax.fori_loop(0, 3, write_one, (state.board, ptr))
    return state._replace(board=new_board, deck_ptr=new_ptr)


def _advance_street(state: EnvState) -> EnvState:
    new_street = state.street + 1
    state = state._replace(
        committed_street=jnp.zeros(NUM_PLAYERS, dtype=jnp.float32),
        current_bet=jnp.float32(0.0),
        acted=jnp.zeros(NUM_PLAYERS, dtype=bool),
        street=new_street,
    )
    n_new = jnp.where(new_street == STREET_FLOP, 3, jnp.where(new_street <= STREET_RIVER, 1, 0))
    state = jax.lax.cond(n_new > 0, lambda s: _deal_n_cards(s, n_new), lambda s: s, state)
    new_street = jnp.where(new_street > STREET_RIVER, STREET_TERMINAL, new_street)
    actor = jax.lax.cond(
        new_street >= STREET_TERMINAL,
        lambda _: state.action_seat,
        lambda _: _first_postflop_actor(state),
        None,
    )
    return state._replace(street=new_street, action_seat=actor)


def _run_out(state: EnvState) -> EnvState:
    """Deal remaining board when betting closed (static loop up to 5 cards)."""

    def write_one(i: Array, carry: tuple[EnvState, Array]) -> tuple[EnvState, Array]:
        s, p = carry
        nb = jnp.sum(s.board >= 0)
        should = nb < MAX_BOARD
        card = s.deck[p]
        board = s.board.at[nb].set(jnp.where(should, card, s.board[nb]))
        p = p + should.astype(jnp.int32)
        return s._replace(board=board, deck_ptr=p), p

    state, _ = jax.lax.fori_loop(0, MAX_BOARD, write_one, (state, state.deck_ptr))
    return state._replace(street=STREET_TERMINAL)


def _commit(state: EnvState, seat: Array, amount: Array) -> EnvState:
    amount = jnp.minimum(amount, state.stacks[seat])
    stacks = state.stacks.at[seat].add(-amount)
    committed_street = state.committed_street.at[seat].add(amount)
    committed_total = state.committed_total.at[seat].add(amount)
    pot = state.pot + amount
    all_in = state.all_in.at[seat].set(stacks[seat] <= 1e-9)
    return state._replace(
        stacks=stacks,
        committed_street=committed_street,
        committed_total=committed_total,
        pot=pot,
        all_in=all_in,
    )


@jax.jit
def legal_action_mask(state: EnvState, seat: Array | None = None) -> Array:
    """Return (NUM_ACTIONS,) bool mask for ``seat`` (default action_seat)."""

    seat = state.action_seat if seat is None else seat
    folded = state.folded[seat]
    all_in = state.all_in[seat]
    stack = state.stacks[seat]
    committed = state.committed_street[seat]
    to_call = jnp.maximum(0.0, state.current_bet - committed)

    mask = jnp.zeros(NUM_ACTIONS, dtype=bool)
    mask = mask.at[FOLD].set(True)
    mask = mask.at[CHECK].set(to_call <= 1e-9)
    mask = mask.at[CALL].set(to_call > 1e-9)

    min_raise = jnp.maximum(BB_BB, state.current_bet * 2.0 - committed)
    pot = state.pot

    def bucket_legal(i: Array) -> Array:
        frac = BET_BUCKETS[i]
        target = jnp.maximum(state.current_bet, pot * frac)
        bet_total = target - committed
        return (stack > to_call) & (bet_total >= min_raise) & (bet_total <= stack)

    bucket_mask = jax.vmap(bucket_legal)(jnp.arange(len(BET_SIZE_BUCKETS)))
    mask = mask.at[BET_BUCKET_START : BET_BUCKET_START + len(BET_SIZE_BUCKETS)].set(bucket_mask)
    mask = mask.at[ALL_IN].set(stack > to_call)
    mask = jnp.where(folded | all_in, False, mask)
    # ensure at least fold
    mask = jnp.where(jnp.any(mask), mask, mask.at[FOLD].set(True))
    return mask


def _action_amount(state: EnvState, seat: Array, action: Array) -> Array:
    committed = state.committed_street[seat]
    to_call = jnp.maximum(0.0, state.current_bet - committed)
    stack = state.stacks[seat]

    def bucket_amt(i: Array) -> Array:
        frac = BET_BUCKETS[i]
        target = jnp.maximum(state.current_bet, state.pot * frac)
        return jnp.maximum(0.0, target - committed)

    bucket_amounts = jax.vmap(bucket_amt)(jnp.arange(len(BET_SIZE_BUCKETS)))

    amount = jnp.where(action == FOLD, 0.0, 0.0)
    amount = jnp.where(action == CHECK, 0.0, amount)
    amount = jnp.where(action == CALL, to_call, amount)
    amount = jnp.where(action == ALL_IN, stack, amount)

    def pick_bucket(i: Array) -> Array:
        return bucket_amounts[i]

    bucket_idx = action - BET_BUCKET_START
    bucket_val = jnp.where(
        (action >= BET_BUCKET_START) & (action < ALL_IN),
        pick_bucket(bucket_idx),
        0.0,
    )
    amount = jnp.where((action >= BET_BUCKET_START) & (action < ALL_IN), to_call + bucket_val, amount)
    return amount


@jax.jit
def step_state(state: EnvState, action: Array) -> EnvState:
    """Apply discrete action index and return updated state."""

    seat = state.action_seat
    amount = _action_amount(state, seat, action)

    folded = state.folded.at[seat].set(action == FOLD)
    state = state._replace(folded=folded)

    is_call = action == CALL
    is_aggressive = (action >= BET_BUCKET_START) | (action == ALL_IN)
    state = jax.lax.cond(
        is_call | is_aggressive,
        lambda s: _commit(s, seat, amount),
        lambda s: s,
        state,
    )

    prev_bet = state.current_bet
    new_street_bet = state.committed_street[seat]
    raised = is_aggressive & (new_street_bet > prev_bet + 1e-9)
    acted = state.acted.at[seat].set(True)
    acted = jnp.where(raised, jnp.zeros(NUM_PLAYERS, dtype=bool).at[seat].set(True), acted)
    current_bet = jnp.where(raised, new_street_bet, state.current_bet)
    last_aggressor = jnp.where(raised, seat, state.last_aggressor)

    state = state._replace(acted=acted, current_bet=current_bet, last_aggressor=last_aggressor)

    # append history
    idx = state.hist_len
    hist_seat = state.hist_seat.at[idx].set(seat)
    hist_action = state.hist_action.at[idx].set(action)
    hist_amount = state.hist_amount.at[idx].set(amount)
    hist_len = jnp.minimum(idx + 1, MAX_HISTORY)
    state = state._replace(hist_seat=hist_seat, hist_action=hist_action, hist_amount=hist_amount, hist_len=hist_len)

    n_active = _active_count(state.folded)
    terminal = n_active <= 1

    state = jax.lax.cond(
        terminal,
        lambda s: s._replace(street=STREET_TERMINAL),
        lambda s: jax.lax.cond(_street_complete(s), _advance_street, lambda x: x, s),
        state,
    )

    actors_left = jnp.any(jax.vmap(lambda s: _seat_active(state, s))(jnp.arange(NUM_PLAYERS)))
    state = jax.lax.cond(
        (~terminal) & (~actors_left) & (state.street < STREET_TERMINAL),
        _run_out,
        lambda s: s,
        state,
    )

    terminal = terminal | (state.street >= STREET_TERMINAL) | (n_active <= 1)
    state = state._replace(
        action_seat=_next_actor(state),
        done=terminal,
    )

    rewards = jax.lax.cond(terminal, _side_pot_payoffs, lambda _: jnp.zeros(NUM_PLAYERS, dtype=jnp.float32), state)
    return state._replace(rewards=rewards)


def _post_blinds(stacks: Array, button: Array) -> tuple[Array, Array, Array, Array]:
    sb = (button + 1) % NUM_PLAYERS
    bb = (button + 2) % NUM_PLAYERS
    committed = jnp.zeros(NUM_PLAYERS, dtype=jnp.float32)
    stacks = stacks.at[sb].add(-SB_BB)
    stacks = stacks.at[bb].add(-BB_BB)
    committed = committed.at[sb].set(SB_BB)
    committed = committed.at[bb].set(BB_BB)
    pot = SB_BB + BB_BB
    current_bet = BB_BB
    return stacks, committed, pot, current_bet


@jax.jit
def reset_state(key: Array, button: Array | None = None) -> EnvState:
    """Reset a single table for a new hand."""

    key, k_deck, k_btn = jax.random.split(key, 3)
    button = jax.random.randint(k_btn, (), 0, NUM_PLAYERS) if button is None else button

    deck = jax.random.permutation(k_deck, jnp.arange(52, dtype=jnp.int32))
    stacks = jnp.full(NUM_PLAYERS, STARTING_STACK_BB, dtype=jnp.float32)
    stacks, committed, pot, current_bet = _post_blinds(stacks, button)

    ptr = jnp.int32(0)
    holes = jnp.full((NUM_PLAYERS, 2), -1, dtype=jnp.int32)

    def deal_hole(i: Array, carry: tuple[Array, Array]) -> tuple[Array, Array]:
        h, p = carry
        c0 = deck[p]
        c1 = deck[p + 1]
        h = h.at[i].set(jnp.array([c0, c1], dtype=jnp.int32))
        return h, p + 2

    holes, ptr = jax.lax.fori_loop(0, NUM_PLAYERS, deal_hole, (holes, ptr))

    return EnvState(
        holes=holes,
        board=jnp.full(MAX_BOARD, -1, dtype=jnp.int32),
        stacks=stacks,
        committed_street=committed,
        committed_total=committed,
        folded=jnp.zeros(NUM_PLAYERS, dtype=bool),
        all_in=jnp.zeros(NUM_PLAYERS, dtype=bool),
        street=jnp.int32(STREET_PREFLOP),
        pot=jnp.float32(pot),
        current_bet=jnp.float32(current_bet),
        button=button,
        action_seat=_first_preflop_actor(button),
        last_aggressor=jnp.int32(-1),
        acted=jnp.zeros(NUM_PLAYERS, dtype=bool),
        deck=deck,
        deck_ptr=ptr,
        hist_seat=jnp.zeros(MAX_HISTORY, dtype=jnp.int32),
        hist_action=jnp.zeros(MAX_HISTORY, dtype=jnp.int32),
        hist_amount=jnp.zeros(MAX_HISTORY, dtype=jnp.float32),
        hist_len=jnp.int32(0),
        done=jnp.bool_(False),
        rewards=jnp.zeros(NUM_PLAYERS, dtype=jnp.float32),
    )


# ---------------------------------------------------------------------------
# Feature encoders for neural CFR
# ---------------------------------------------------------------------------


def _one_hot(index: Array, size: int) -> Array:
    return jax.nn.one_hot(index, size, dtype=jnp.float32)


@jax.jit
def encode_info_state(state: EnvState, seat: Array) -> Array:
    """Fixed-size information-set features for player ``seat``."""

    parts = [
        _one_hot(state.street, NUM_STREETS + 1),
        _one_hot(seat, NUM_PLAYERS),
        _one_hot(state.button, NUM_PLAYERS),
        state.stacks / STARTING_STACK_BB,
        state.committed_street / STARTING_STACK_BB,
        state.committed_total / STARTING_STACK_BB,
        state.folded.astype(jnp.float32),
        state.all_in.astype(jnp.float32),
        jnp.array([state.pot / (STARTING_STACK_BB * NUM_PLAYERS), state.current_bet / STARTING_STACK_BB]),
        (state.holes[seat] >= 0).astype(jnp.float32) * (_card_rank(state.holes[seat]) / 14.0),
        (state.board >= 0).astype(jnp.float32) * (_card_rank(state.board) / 14.0),
    ]
    feat = jnp.concatenate([p.flatten() for p in parts])
    pad = jnp.zeros(INFO_FEATURE_DIM, dtype=jnp.float32)
    return jax.lax.dynamic_update_slice(pad, feat, (0,))


HISTORY_SLOTS: int = 32


@jax.jit
def encode_history_state(state: EnvState) -> Array:
    """History node features for the value network Q(h, a)."""

    seats = state.hist_seat[:HISTORY_SLOTS].astype(jnp.float32) / NUM_PLAYERS
    actions = state.hist_action[:HISTORY_SLOTS].astype(jnp.float32) / NUM_ACTIONS
    amounts = state.hist_amount[:HISTORY_SLOTS] / STARTING_STACK_BB
    hist = jnp.concatenate([seats, actions, amounts])
    pad = jnp.zeros(HISTORY_FEATURE_DIM, dtype=jnp.float32)
    pad = jax.lax.dynamic_update_slice(pad, hist, (0,))
    ctx = jnp.array(
        [
            state.pot / (STARTING_STACK_BB * NUM_PLAYERS),
            state.current_bet / STARTING_STACK_BB,
            state.street.astype(jnp.float32) / NUM_STREETS,
            state.hist_len.astype(jnp.float32) / MAX_HISTORY,
        ],
        dtype=jnp.float32,
    )
    return jax.lax.dynamic_update_slice(pad, ctx, (HISTORY_FEATURE_DIM - ctx.shape[0],))


# ---------------------------------------------------------------------------
# Batched DeepMind-style environment wrapper
# ---------------------------------------------------------------------------


class JAXNLHEnv:
    """JIT-compiled 6-max NLHE environment for massive batch processing."""

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


# Vectorized helpers exported for training loops
batched_reset = jax.jit(jax.vmap(reset_state))
batched_step = jax.jit(jax.vmap(step_state))
batched_legal_mask = jax.jit(jax.vmap(legal_action_mask))
batched_info_state = jax.jit(jax.vmap(encode_info_state, in_axes=(0, 0)))
