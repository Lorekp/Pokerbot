"""Pure-NumPy port of the hot-path JAX NLHE engine (``pokerbot.engine.jax_engine``).

CORRECTNESS-CRITICAL: every transition here must produce BIT-IDENTICAL results to the
JAX engine.  A CPU game-tree traversal uses this to avoid per-node XLA eager-dispatch
overhead, but a single divergence would silently corrupt a CFR blueprint.  The acceptance
gate is ``tests/test_np_engine_equiv.py`` (lockstep equivalence over thousands of hands).

NO ``import jax``.  Operates on :class:`pokerbot.engine.jax_engine.EnvState` instances whose
fields are NumPy arrays.  ``init_hand`` is NOT ported — callers keep the JAX init and then
call :func:`to_numpy_state`.

Constants/semantics are mirrored verbatim from ``jax_engine.py``, ``side_pots_jax.py`` and
``hand_eval_jax.py``.
"""

from __future__ import annotations

import numpy as np

from pokerbot.engine.jax_engine import (
    ALL_IN,
    BB_CHIPS,
    BET_BUCKET_START,
    BET_SIZE_BUCKETS,
    CALL,
    CHECK,
    EnvState,
    FOLD,
    MAX_BOARD,
    MAX_HISTORY,
    NUM_ACTIONS,
    NUM_PLAYERS,
    STREET_FLOP,
    STREET_RIVER,
    STREET_TERMINAL,
)

# Seat-status int8 sentinels (mirror side_pots_jax / jax_engine).
SEAT_EMPTY = np.int8(0)
SEAT_FOLDED = np.int8(1)
SEAT_LIVE = np.int8(2)
SEAT_ALLIN = np.int8(3)

MAX_POTS = NUM_PLAYERS
BET_BUCKETS = np.asarray(BET_SIZE_BUCKETS, dtype=np.float32)
_N_BUCKETS = len(BET_SIZE_BUCKETS)


# ---------------------------------------------------------------------------
# State conversion
# ---------------------------------------------------------------------------

# Per-field numpy dtypes matching the JAX EnvState exactly.
_FIELD_DTYPES = {
    "holes": np.int32,
    "board": np.int32,
    "seat_status": np.int8,
    "stacks": np.int32,
    "committed_street": np.int32,
    "committed_total": np.int32,
    "street": np.int32,
    "pot_dead": np.int32,
    "current_bet": np.int32,
    "min_raise": np.int32,
    "button": np.int32,
    "action_seat": np.int32,
    "last_aggressor": np.int32,
    "acted": np.bool_,
    "last_raise_full": np.bool_,
    "deck": np.int32,
    "deck_ptr": np.int32,
    "rng_key": np.uint32,
    "hist_seat": np.int32,
    "hist_action": np.int32,
    "hist_amount": np.int32,
    "hist_len": np.int32,
    "done": np.bool_,
    "rewards": np.int32,
    "pending_deal": np.bool_,
}

# Fields that are 0-d scalars in the JAX state (kept as 0-d numpy arrays so that
# np.array_equal and downstream `int(...)`/`np.asarray(...)` behave identically).
_SCALAR_FIELDS = {
    "street",
    "pot_dead",
    "current_bet",
    "min_raise",
    "button",
    "action_seat",
    "last_aggressor",
    "last_raise_full",
    "deck_ptr",
    "hist_len",
    "done",
    "pending_deal",
}


def to_numpy_state(state: EnvState) -> EnvState:
    """Convert a (JAX or numpy) EnvState to a numpy-backed EnvState with exact dtypes."""
    out = {}
    for field in state._fields:
        arr = np.asarray(getattr(state, field))
        dt = _FIELD_DTYPES[field]
        out[field] = arr.astype(dt, copy=True)
    return EnvState(**out)


def _scalar(value, dtype) -> np.ndarray:
    """Return a 0-d numpy array, matching JAX 0-d scalar semantics."""
    return np.asarray(value, dtype=dtype)


# ---------------------------------------------------------------------------
# Hand evaluation (port of hand_eval_jax.py)
# ---------------------------------------------------------------------------

_STRAIGHT_WINDOWS = np.array(
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
    dtype=np.int32,
)

_COMBOS = np.array(
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
    dtype=np.int32,
)

_HIGH_WEIGHTS = np.array([15**4, 15**3, 15**2, 15, 1], dtype=np.float32)
_HIST_WEIGHTS = np.array([15**4, 15**3, 15**2, 15, 1], dtype=np.int64)  # base-15 positional, int (exact)
_CAT_SCALE = 15**5  # 759375: category multiplier; within-code in [0, 15**5-1] => no overlap, fits int32


def card_rank(card):
    return card // 4 + 2


def card_suit(card):
    return card % 4


def _straight_high(ranks: np.ndarray) -> np.int32:
    """ranks: (5,) int array.  Mirror straight_high()."""
    best = np.int32(0)
    for w in range(_STRAIGHT_WINDOWS.shape[0]):
        window = _STRAIGHT_WINDOWS[w]
        matched = True
        for i in range(5):
            if not np.any(ranks == window[i]):
                matched = False
                break
        if window[0] == 14:
            high = 5 if window[1] == 5 else 14
        else:
            high = int(window[0])
        high = np.int32(high if matched else 0)
        if high > best:
            best = high
    return best


def score_five(ranks: np.ndarray, suits: np.ndarray) -> np.int32:
    """Correct 5-card hand score (kept bit-identical to hand_eval_jax.score_five).

    Returns ``cat * 15**5 + within`` in int32: the hand CATEGORY (0 high card .. 8 straight
    flush) dominates, and ``within`` is a base-15 positional tiebreaker.

    The within-category code orders the 5 cards by (count of that rank desc, then rank desc) and
    reads off base-15 digits, so paired/tripped/quad ranks sort ahead of kickers automatically and
    one formula correctly tiebreaks every category. Straights/straight-flushes use ``s_high`` so the
    wheel (A-2-3-4-5) ranks as 5-high. Verified 0 disagreements vs an independent evaluator
    (phevaluator) over 20k random 7-card matchups.

    FIX 2026-06-08: the previous packing ``(cat << 32) + primary`` silently dropped the category
    (repo runs JAX x64-OFF, so int32 ``<< 32`` == 0) AND used mis-scaled per-category tiebreakers,
    mis-ranking showdowns (a flush could lose to two pair). Both are corrected here.
    """
    ranks = np.sort(ranks)[::-1]
    counts = np.bincount(ranks.astype(np.int64), minlength=15)[:15]
    is_flush = bool(np.all(suits == suits[0]))
    s_high = int(_straight_high(ranks))
    is_straight = s_high > 0
    count_vals = np.sort(counts)[::-1]

    cat = 0
    if is_flush and is_straight:
        cat = 8
    if (count_vals[0] == 4) and (cat < 8):
        cat = 7
    if (count_vals[0] == 3) and (count_vals[1] >= 2) and (cat < 7):
        cat = 6
    if is_flush and (not is_straight) and (cat < 6):
        cat = 5
    if is_straight and (not is_flush) and (cat < 5):
        cat = 4
    if (count_vals[0] == 3) and (count_vals[1] < 2) and (cat < 4):
        cat = 3
    if (count_vals[0] == 2) and (count_vals[1] == 2) and (cat < 3):
        cat = 2
    if (count_vals[0] == 2) and (count_vals[1] < 2) and (cat < 2):
        cat = 1

    # within-category positional code (cards ordered by count desc, then rank desc)
    key = counts[ranks] * 15 + ranks
    ordered = ranks[np.argsort(key, kind="stable")[::-1]]
    hist = int(np.sum(ordered.astype(np.int64) * _HIST_WEIGHTS))
    within = s_high if cat in (4, 8) else hist
    return np.int32(cat * _CAT_SCALE + within)


# Memoization for evaluate_seven.  ``evaluate_seven`` is a PURE deterministic function of its 7-card
# argument (no RNG, no global state), so caching its result by the exact card-id tuple returns the
# BIT-IDENTICAL ``np.int32`` for any repeated input.  This is the dominant search hot path: the gadget
# leaf replays the SAME CRN board across all 16 hero×field matrix cells with fixed hole cards, so the
# same 7-card showdown sets recur ~16x per leaf.  Caching collapses that redundant evaluation with
# zero numerical change.  The key space is bounded (C(52,7) ≈ 133M worst case, but only a tiny working
# set per solve); a soft cap keeps memory bounded over a long eval without affecting results.
_EVAL7_CACHE: dict[bytes, np.int32] = {}
_EVAL7_CACHE_CAP = 1_000_000


def evaluate_seven(cards: np.ndarray) -> np.int32:
    cards = np.ascontiguousarray(cards, dtype=np.int32)
    ckey = cards.tobytes()
    cached = _EVAL7_CACHE.get(ckey)
    if cached is not None:
        return cached
    best = np.int32(np.iinfo(np.int32).min)
    for c in range(_COMBOS.shape[0]):
        chosen = cards[_COMBOS[c]]
        ranks = card_rank(chosen)
        suits = card_suit(chosen)
        sc = score_five(ranks, suits)
        if sc > best:
            best = sc
    if len(_EVAL7_CACHE) < _EVAL7_CACHE_CAP:
        _EVAL7_CACHE[ckey] = best
    return best


def player_showdown_score(holes: np.ndarray, board: np.ndarray, board_valid: np.ndarray) -> np.int32:
    # Build the 7-card array in one int32 buffer instead of two np.concatenate + np.where passes.
    # Identical result: holes are always valid; invalid board slots map to 0 (the original masked the
    # board to 0 via board_valid then re-masked via valid_mask — net effect is exactly this).
    seven = np.empty(2 + MAX_BOARD, dtype=np.int32)
    seven[0] = holes[0]
    seven[1] = holes[1]
    seven[2:] = np.where(board_valid, board, 0)
    return np.int32(evaluate_seven(seven))


# ---------------------------------------------------------------------------
# Side pots / terminal returns (port of side_pots_jax.py)
# ---------------------------------------------------------------------------


def _eligible_to_win(seat_status: np.ndarray) -> np.ndarray:
    return (seat_status == SEAT_LIVE) | (seat_status == SEAT_ALLIN)


def _cap_contributions(committed_total: np.ndarray) -> np.ndarray:
    positive = np.where(committed_total > 0, committed_total, 0)
    sorted_c = np.sort(positive)[::-1]
    second = sorted_c[1]
    n_pos = int(np.sum(positive > 0))
    cap = second if n_pos >= 2 else sorted_c[0]
    cap = cap if n_pos >= 1 else np.int32(0)
    return np.where(positive > 0, np.minimum(positive, cap), 0).astype(np.int32)


def _odd_chip_seat(eligible: np.ndarray, button) -> np.int32:
    offsets = np.arange(NUM_PLAYERS)
    seats = (int(button) + 1 + offsets) % NUM_PLAYERS
    is_elig = eligible[seats]
    idx = int(np.argmax(is_elig))
    return np.int32(seats[idx]) if bool(np.any(is_elig)) else np.int32(button)


def award_layer(amount, eligible: np.ndarray, scores: np.ndarray, button) -> np.ndarray:
    # scores are int32 (JAX runs without x64); sentinel for non-eligible is int32(-1).
    best = np.max(np.where(eligible, scores, np.int32(-1)))
    winners = eligible & (scores == best) & (best >= 0)
    n_winners = max(int(np.sum(winners.astype(np.int32))), 1)
    amount = np.int32(amount)
    base = amount // n_winners
    rem = amount - base * n_winners
    odd_seat = _odd_chip_seat(winners, button)
    winnings = winners.astype(np.int32) * base
    if (rem > 0) and bool(winners[odd_seat]):
        winnings[odd_seat] += rem
    return winnings.astype(np.int32)


def compute_returns_np(
    *,
    holes: np.ndarray,
    board: np.ndarray,
    seat_status: np.ndarray,
    committed_total: np.ndarray,
    button,
    n_showdown,
) -> np.ndarray:
    """Per-seat net chip delta (int32), sums to zero.  Port of compute_returns."""
    committed_total = committed_total.astype(np.int32)
    if int(n_showdown) <= 1:
        # fold_win
        live = _eligible_to_win(seat_status)
        winner = int(np.argmax(live.astype(np.int32)))
        total = np.int32(np.sum(committed_total))
        out = (-committed_total).astype(np.int32).copy()
        out[winner] += total
        return out.astype(np.int32)

    # showdown
    board_valid = board >= 0
    # Ineligible (folded/empty) seats are overwritten with the -1 sentinel below, so their showdown
    # score never affects the result.  Evaluate only eligible seats and fill the rest with -1 directly:
    # this yields the BIT-IDENTICAL ``scores`` array while skipping the expensive hand evaluation for
    # seats whose value would be discarded.  (Pure refactor of the np.where mask — no numerical change.)
    eligible_mask = _eligible_to_win(seat_status)
    scores = np.full(NUM_PLAYERS, np.int32(-1), dtype=np.int32)
    for s in range(NUM_PLAYERS):
        if eligible_mask[s]:
            scores[s] = player_showdown_score(holes[s], board, board_valid)

    capped = _cap_contributions(committed_total)
    excess = committed_total - capped
    winnings = np.zeros(NUM_PLAYERS, dtype=np.int32)

    positive = np.where(capped > 0, capped, -1)
    # jnp.unique(positive, size=MAX_POTS, fill_value=-1) -> sorted unique, padded with -1.
    uniq = np.unique(positive)  # sorted ascending
    levels = np.full(MAX_POTS, -1, dtype=positive.dtype)
    take = min(len(uniq), MAX_POTS)
    levels[:take] = uniq[:take]
    levels = np.sort(levels)  # JAX sorts again
    levels = np.where(levels >= 0, levels, -1)

    prev_level = np.int32(0)
    for li in range(MAX_POTS):
        level = levels[li]
        valid = level >= 0
        layer_size = (level - prev_level) if valid else np.int32(0)
        contributing = (capped >= level) & valid
        n_contrib = np.int32(np.sum(contributing.astype(np.int32)))
        pot_amt = np.int32(layer_size) * n_contrib
        eligible = _eligible_to_win(seat_status) & contributing
        win = award_layer(pot_amt, eligible, scores, button)
        winnings = winnings + win
        if valid:
            prev_level = np.int32(level)

    return (winnings + excess - committed_total).astype(np.int32)


# ---------------------------------------------------------------------------
# Seat helpers (port of jax_engine.py)
# ---------------------------------------------------------------------------


def _is_live(seat_status: np.ndarray) -> np.ndarray:
    return (seat_status == SEAT_LIVE) | (seat_status == SEAT_ALLIN)


def _can_act(state: EnvState, seat) -> bool:
    return bool(state.seat_status[seat] == SEAT_LIVE)


def _can_act_vec(seat_status: np.ndarray) -> np.ndarray:
    return seat_status == SEAT_LIVE


def _live_count(seat_status: np.ndarray) -> np.int32:
    # tight Python loop over 6 elems is faster than numpy ufunc dispatch on tiny arrays.
    c = 0
    for v in seat_status.tolist():
        if v == 2 or v == 3:  # SEAT_LIVE / SEAT_ALLIN
            c += 1
    return np.int32(c)


def _pot_for_sizing(state: EnvState) -> np.int32:
    cs = state.committed_street
    return np.int32(int(state.pot_dead) + int(cs[0]) + int(cs[1]) + int(cs[2]) + int(cs[3]) + int(cs[4]) + int(cs[5]))


def _next_seat_cw(from_seat, offset) -> np.int32:
    return np.int32((int(from_seat) + int(offset)) % NUM_PLAYERS)


def _first_live_after(seat_status: np.ndarray, from_seat) -> np.int32:
    # jnp argmax-of-(seat_status==LIVE) over seats (from+1..from+6); first live clockwise.
    ss = seat_status
    base = int(from_seat)
    for off in range(1, NUM_PLAYERS + 1):
        seat = (base + off) % NUM_PLAYERS
        if ss[seat] == SEAT_LIVE:
            return np.int32(seat)
    return np.int32(-1)


def _nth_live_after(seat_status: np.ndarray, from_seat, n) -> np.int32:
    seat = np.int32(from_seat)
    for _ in range(int(n)):
        seat = _first_live_after(seat_status, seat)
    return np.int32(seat)


def _first_postflop_actor(state: EnvState) -> np.int32:
    return _nth_live_after(state.seat_status, state.button, 1)


def _next_actor(state: EnvState) -> np.int32:
    return _first_live_after(state.seat_status, state.action_seat)


# ---------------------------------------------------------------------------
# Street / chance
# ---------------------------------------------------------------------------


def is_chance_node_np(state: EnvState) -> bool:
    return bool(bool(state.pending_deal) and (not bool(state.done)))


def is_terminal_np(state: EnvState) -> bool:
    return bool(state.done)


def current_player_np(state: EnvState) -> int:
    if bool(state.done) or bool(state.pending_deal):
        return -1
    return int(state.action_seat)


def _street_complete(state: EnvState) -> bool:
    ss = state.seat_status.tolist()
    cs = state.committed_street.tolist()
    acted = state.acted.tolist()
    cur = int(state.current_bet)
    n_can_act = 0
    matched = True
    all_acted = True
    for i in range(NUM_PLAYERS):
        if ss[i] == 2:  # SEAT_LIVE
            n_can_act += 1
            if cs[i] < cur:
                matched = False
            if not acted[i]:
                all_acted = False
    back_to_aggressor = int(state.action_seat) == int(state.last_aggressor)
    no_aggressor = int(state.last_aggressor) < 0
    closed = (n_can_act == 0) or (matched and all_acted and (back_to_aggressor or no_aggressor))
    return bool(closed)


def _deal_n_cards(state: EnvState, n_new: int) -> EnvState:
    """Deterministic deal of the next street's cards from the already-shuffled deck.

    Mirrors _deal_n_cards: writes n_new cards at board[n_board:], advances deck_ptr,
    and crucially advances rng_key by 3 jax.random.split steps (one per fori loop iter,
    regardless of `should`).  rng_key parity is required by the equivalence gate.
    """
    ptr = int(state.deck_ptr)
    n_board = int(np.sum(state.board >= 0))
    board = state.board.copy()
    key = state.rng_key.copy()
    for i in range(3):
        key = _split_consume(key)
        should = i < n_new
        if should:
            board[n_board + i] = state.deck[ptr]
            ptr += 1
    return state._replace(
        board=board,
        deck_ptr=_scalar(ptr, np.int32),
        rng_key=key,
        pending_deal=_scalar(False, np.bool_),
    )


def sample_chance_np(state: EnvState) -> EnvState:
    """Deal next street after betting closed; deterministic given rng_key/deck."""
    new_street = int(state.street) + 1
    if new_street == STREET_FLOP:
        n_new = 3
    elif new_street <= STREET_RIVER:
        n_new = 1
    else:
        n_new = 0
    state = state._replace(street=_scalar(new_street, np.int32))
    if n_new > 0:
        state = _deal_n_cards(state, n_new)
    else:
        state = state._replace(pending_deal=_scalar(False, np.bool_))

    new_street = STREET_TERMINAL if new_street > STREET_RIVER else int(state.street)
    if new_street >= STREET_TERMINAL:
        actor = np.int32(-1)
    else:
        actor = _first_postflop_actor(state)

    return state._replace(
        street=_scalar(new_street, np.int32),
        action_seat=_scalar(actor, np.int32),
        committed_street=np.zeros(NUM_PLAYERS, dtype=np.int32),
        current_bet=_scalar(0, np.int32),
        min_raise=_scalar(BB_CHIPS, np.int32),
        acted=np.zeros(NUM_PLAYERS, dtype=np.bool_),
        last_aggressor=_scalar(-1, np.int32),
        last_raise_full=_scalar(True, np.bool_),
    )


def _run_out(state: EnvState) -> EnvState:
    board = state.board.copy()
    ptr = int(state.deck_ptr)
    for _ in range(MAX_BOARD):
        nb = int(np.sum(board >= 0))
        if nb < MAX_BOARD:
            board[nb] = state.deck[ptr]
            ptr += 1
    return state._replace(
        board=board,
        deck_ptr=_scalar(ptr, np.int32),
        street=_scalar(STREET_TERMINAL, np.int32),
        pending_deal=_scalar(False, np.bool_),
    )


def _close_street(state: EnvState) -> EnvState:
    pot_dead = int(state.pot_dead) + int(np.sum(state.committed_street))
    state = state._replace(
        pot_dead=_scalar(pot_dead, np.int32),
        committed_street=np.zeros(NUM_PLAYERS, dtype=np.int32),
        current_bet=_scalar(0, np.int32),
        pending_deal=_scalar(True, np.bool_),
        action_seat=_scalar(-1, np.int32),
    )
    actors = bool(np.any(_can_act_vec(state.seat_status)))
    if (not actors) and (int(state.street) < STREET_RIVER):
        state = _run_out(state)
    return state


# ---------------------------------------------------------------------------
# Betting
# ---------------------------------------------------------------------------


def _commit(state: EnvState, seat, amount) -> EnvState:
    seat = int(seat)
    amount = int(min(int(amount), int(state.stacks[seat])))
    stacks = state.stacks.copy()
    stacks[seat] -= amount
    committed_street = state.committed_street.copy()
    committed_street[seat] += amount
    committed_total = state.committed_total.copy()
    committed_total[seat] += amount
    seat_status = state.seat_status
    if stacks[seat] <= 0:
        seat_status = state.seat_status.copy()
        seat_status[seat] = SEAT_ALLIN
    return state._replace(
        stacks=stacks,
        committed_street=committed_street,
        committed_total=committed_total,
        seat_status=seat_status,
    )


def legal_action_mask_np(state: EnvState, seat=None) -> np.ndarray:
    seat = int(state.action_seat) if seat is None else int(seat)
    status = state.seat_status[seat]
    stack = int(state.stacks[seat])
    committed = int(state.committed_street[seat])
    to_call = max(0, int(state.current_bet) - committed)
    pot = int(_pot_for_sizing(state))

    mask = np.zeros(NUM_ACTIONS, dtype=np.bool_)
    mask[FOLD] = to_call > 0
    mask[CHECK] = to_call == 0
    mask[CALL] = to_call > 0

    min_raise_amt = int(state.min_raise)
    # (JAX computes full_raise_target/is_full inside bucket_legal but never uses is_full;
    #  we omit that dead branch — the legality conditions below are identical.)
    cur_bet = int(state.current_bet)
    is_live = bool(status == SEAT_LIVE)
    reopen_ok = (not bool(state.acted[seat])) or bool(state.last_raise_full) or (to_call == 0)

    for i in range(_N_BUCKETS):
        frac = BET_BUCKETS[i]
        # (pot * frac).astype(int32): float32 product truncated toward zero.
        prod = np.int32(np.float32(np.float32(pot) * frac))
        target_total = max(cur_bet, int(prod))
        raise_by = target_total - committed
        bucket_ok = (
            is_live
            and (stack > to_call)
            and (raise_by >= min_raise_amt)
            and (raise_by <= stack)
            and reopen_ok
        )
        mask[BET_BUCKET_START + i] = bucket_ok

    mask[ALL_IN] = is_live and (stack > 0) and reopen_ok
    if not is_live:
        mask[:] = False
    if not bool(np.any(mask)):
        mask[CHECK] = True
    return mask


def _action_amount(state: EnvState, seat, action: int) -> int:
    seat = int(seat)
    committed = int(state.committed_street[seat])
    to_call = max(0, int(state.current_bet) - committed)
    stack = int(state.stacks[seat])
    pot = int(_pot_for_sizing(state))

    if action == CALL:
        return to_call
    if action == ALL_IN:
        return stack
    if BET_BUCKET_START <= action < ALL_IN:
        frac = BET_BUCKETS[action - BET_BUCKET_START]
        prod = np.int32(np.float32(np.float32(pot) * frac))
        target = max(int(state.current_bet), int(prod))
        bucket_val = max(0, target - committed)
        return to_call + bucket_val
    return 0


def apply_action_np(state: EnvState, action: int) -> EnvState:
    action = int(action)
    seat = int(state.action_seat)
    amount = _action_amount(state, seat, action)

    if action == FOLD:
        seat_status = state.seat_status.copy()
        seat_status[seat] = SEAT_FOLDED
        state = state._replace(seat_status=seat_status)

    is_call = action == CALL
    is_aggressive = (action >= BET_BUCKET_START) or (action == ALL_IN)
    if is_call or is_aggressive:
        state = _commit(state, seat, amount)

    prev_bet = int(state.current_bet)
    new_street_bet = int(state.committed_street[seat])
    raised = is_aggressive and (new_street_bet > prev_bet)
    full_raise = raised and (new_street_bet >= prev_bet + int(state.min_raise))

    acted = state.acted.copy()
    acted[seat] = True
    if full_raise:
        acted = np.zeros(NUM_PLAYERS, dtype=np.bool_)
        acted[seat] = True

    if raised:
        current_bet = new_street_bet
        raise_size = new_street_bet - prev_bet
        min_raise = max(BB_CHIPS, raise_size)
        last_aggressor = seat
        last_raise_full = full_raise
    else:
        current_bet = int(state.current_bet)
        min_raise = int(state.min_raise)
        last_aggressor = int(state.last_aggressor)
        last_raise_full = bool(state.last_raise_full)

    state = state._replace(
        acted=acted,
        current_bet=_scalar(current_bet, np.int32),
        min_raise=_scalar(min_raise, np.int32),
        last_aggressor=_scalar(last_aggressor, np.int32),
        last_raise_full=_scalar(last_raise_full, np.bool_),
    )

    idx = int(state.hist_len)
    hist_seat = state.hist_seat.copy()
    hist_action = state.hist_action.copy()
    hist_amount = state.hist_amount.copy()
    if idx < MAX_HISTORY:
        hist_seat[idx] = seat
        hist_action[idx] = action
        hist_amount[idx] = amount
    state = state._replace(
        hist_seat=hist_seat,
        hist_action=hist_action,
        hist_amount=hist_amount,
        hist_len=_scalar(min(idx + 1, MAX_HISTORY), np.int32),
    )

    n_live = int(_live_count(state.seat_status))
    terminal = n_live <= 1

    if terminal:
        state = state._replace(
            street=_scalar(STREET_TERMINAL, np.int32),
            pending_deal=_scalar(False, np.bool_),
        )
    else:
        if _street_complete(state):
            state = _close_street(state)

    terminal = terminal or (int(state.street) >= STREET_TERMINAL) or (n_live <= 1)

    if terminal or bool(state.pending_deal):
        action_seat = np.int32(-1)
    else:
        action_seat = _next_actor(state)
    state = state._replace(
        action_seat=_scalar(action_seat, np.int32),
        done=_scalar(terminal, np.bool_),
    )

    if terminal:
        rets = compute_returns_np(
            holes=state.holes,
            board=state.board,
            seat_status=state.seat_status,
            committed_total=state.committed_total,
            button=state.button,
            n_showdown=n_live,
        )
    else:
        rets = np.zeros(NUM_PLAYERS, dtype=np.int32)
    return state._replace(rewards=rets.astype(np.int32))


# step_state alias
step_state_np = apply_action_np


def returns_np(state: EnvState) -> np.ndarray:
    return state.rewards


# ---------------------------------------------------------------------------
# JAX threefry RNG replication (so rng_key stays bit-identical across deals)
# ---------------------------------------------------------------------------
#
# sample_chance/_deal_n_cards advance state.rng_key via jax.random.split exactly 3 times
# per call (once per fori_loop iter, regardless of whether a card is written).  The numpy
# state must carry a byte-identical rng_key after each deal or the equivalence test fails.
# We replicate JAX's default Threefry2x32 split here.

_ROT_BLOCKS = (
    (13, 15, 26, 6),
    (17, 29, 16, 24),
    (13, 15, 26, 6),
    (17, 29, 16, 24),
    (13, 15, 26, 6),
)


def _rotl32(x: np.uint32, r: int) -> np.uint32:
    r = int(r)
    return np.uint32((np.uint32(x) << np.uint32(r)) | (np.uint32(x) >> np.uint32(32 - r)))


def _threefry2x32(keypair, count0, count1):
    """Threefry-2x32 (20 rounds) matching jax's default PRNG.  All uint32 math.

    Returns (out0, out1) for the single counter (count0, count1).  Overflow in uint32
    addition is the desired wraparound; we silence the numpy scalar-overflow warning by
    masking to 32 bits via the uint32 dtype (np.errstate covers the RuntimeWarning).
    """
    k0 = np.uint32(keypair[0])
    k1 = np.uint32(keypair[1])
    k2 = np.uint32(np.uint32(0x1BD11BDA) ^ k0 ^ k1)
    ks = (k0, k1, k2)
    with np.errstate(over="ignore"):
        x0 = np.uint32(np.uint32(count0) + ks[0])
        x1 = np.uint32(np.uint32(count1) + ks[1])
        for block in range(5):
            rots = _ROT_BLOCKS[block]
            for i in range(4):
                x0 = np.uint32(x0 + x1)
                x1 = _rotl32(x1, rots[i])
                x1 = np.uint32(x1 ^ x0)
            x0 = np.uint32(x0 + ks[(block + 1) % 3])
            x1 = np.uint32(x1 + ks[(block + 2) % 3] + np.uint32(block + 1))
    return np.uint32(x0), np.uint32(x1)


def _split_consume(key: np.ndarray) -> np.ndarray:
    """Replicate `key, sub = jax.random.split(key)` returning the NEW carry key.

    jax's default (partitionable / foldlike) split runs threefry2x32 elementwise over
    iota_2x32_shape((2,)) = counts (hi=[0,0], lo=[0,1]); the result is stacked so row 0
    (the carry key kept by _deal_n_cards) = threefry2x32(key, count=(hi=0, lo=0)).
    """
    k = np.asarray(key, dtype=np.uint32).reshape(2)
    o0, o1 = _threefry2x32(k, np.uint32(0), np.uint32(0))
    return np.array([o0, o1], dtype=np.uint32)


__all__ = [
    "SEAT_EMPTY",
    "SEAT_FOLDED",
    "SEAT_LIVE",
    "SEAT_ALLIN",
    "to_numpy_state",
    "legal_action_mask_np",
    "apply_action_np",
    "step_state_np",
    "sample_chance_np",
    "is_chance_node_np",
    "is_terminal_np",
    "current_player_np",
    "returns_np",
    "compute_returns_np",
    "score_five",
    "evaluate_seven",
    "player_showdown_score",
    "card_rank",
    "card_suit",
]
