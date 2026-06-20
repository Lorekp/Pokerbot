"""Fixed-max-layer side-pot construction and terminal returns (int32 chips)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array

from pokerbot.engine.hand_eval_jax import player_showdown_score

# NUM_PLAYERS / MAX_POTS are the 6-max DEFAULTS (cash). They are NOT used for SIZING any more: every
# function below derives the seat count ``n`` from the shape of its ``seat_status`` / ``committed_total``
# input (``.shape[0]``, which is STATIC under jit, so jit-safety is preserved) and uses that for array
# allocation and seat wraps. This lets a 7-wide (or N-wide) state flow through unchanged while keeping
# byte-identity for the deployed 6-wide path. Do NOT change these module constants.
NUM_PLAYERS: int = 6
MAX_POTS: int = NUM_PLAYERS
SEAT_FOLDED = jnp.int8(1)
SEAT_LIVE = jnp.int8(2)
SEAT_ALLIN = jnp.int8(3)


def _eligible_to_win(seat_status: Array) -> Array:
    return (seat_status == SEAT_LIVE) | (seat_status == SEAT_ALLIN)


def _cap_contributions(committed_total: Array) -> Array:
    positive = jnp.where(committed_total > 0, committed_total, 0)
    sorted_c = jnp.sort(positive)[::-1]
    second = sorted_c[1]
    cap = jnp.where(jnp.sum(positive > 0) >= 2, second, sorted_c[0])
    cap = jnp.where(jnp.sum(positive > 0) >= 1, cap, 0)
    return jnp.where(positive > 0, jnp.minimum(positive, cap), 0)


def _odd_chip_seat(eligible: Array, button: Array) -> Array:
    n = eligible.shape[0]  # seat count (static under jit)
    offsets = jnp.arange(n)
    seats = (button + 1 + offsets) % n
    is_elig = eligible[seats]
    idx = jnp.argmax(is_elig)
    return jnp.where(jnp.any(is_elig), seats[idx], button)


def award_layer(amount: Array, eligible: Array, scores: Array, button: Array) -> Array:
    best = jnp.max(jnp.where(eligible, scores, jnp.int32(-1)))
    winners = eligible & (scores == best) & (best >= 0)
    n_winners = jnp.maximum(jnp.sum(winners.astype(jnp.int32)), 1)
    base = amount // n_winners
    rem = amount - base * n_winners
    odd_seat = _odd_chip_seat(winners, button)
    winnings = winners.astype(jnp.int32) * base
    winnings = winnings.at[odd_seat].add(jnp.where((rem > 0) & winners[odd_seat], rem, 0))
    return winnings


def compute_returns(
    *,
    holes: Array,
    board: Array,
    seat_status: Array,
    committed_total: Array,
    button: Array,
    n_showdown: Array,
) -> Array:
    """Per-seat net chip delta; sums to zero."""

    def fold_win(_: None) -> Array:
        live = _eligible_to_win(seat_status)
        winner = jnp.argmax(live.astype(jnp.int32))
        total = jnp.sum(committed_total)
        return (-committed_total).at[winner].add(total)

    # Seat count (and hence the max distinct side-pot levels) is the width of the dealt state, derived
    # from the input arrays. Static under jit, so the kernel re-traces cleanly per seat count and stays
    # byte-identical for n == 6.
    n = seat_status.shape[0]
    max_pots = n

    def showdown(_: None) -> Array:
        board_valid = board >= 0
        scores = jax.vmap(lambda s: player_showdown_score(holes[s], board, board_valid))(jnp.arange(n))
        scores = jnp.where(_eligible_to_win(seat_status), scores, jnp.int32(-1))

        capped = _cap_contributions(committed_total)
        excess = committed_total - capped
        winnings = jnp.zeros(n, dtype=jnp.int32)

        positive = jnp.where(capped > 0, capped, -1)
        levels = jnp.sort(jnp.unique(positive, size=max_pots, fill_value=-1))
        levels = jnp.where(levels >= 0, levels, -1)
        prev = jnp.int32(0)

        def layer_step(carry: tuple[Array, Array], level: Array) -> tuple[tuple[Array, Array], None]:
            win_acc, prev_level = carry
            valid = level >= 0
            layer_size = jnp.where(valid, level - prev_level, 0)
            contributing = (capped >= level) & valid
            n_contrib = jnp.sum(contributing.astype(jnp.int32))
            pot_amt = layer_size * n_contrib
            eligible = _eligible_to_win(seat_status) & contributing
            win = award_layer(pot_amt, eligible, scores, button)
            win_acc = win_acc + win
            prev_level = jnp.where(valid, level, prev_level)
            return (win_acc, prev_level), None

        (winnings, _), _ = jax.lax.scan(layer_step, (winnings, prev), levels)
        return winnings + excess - committed_total

    return jax.lax.cond(n_showdown <= 1, fold_win, showdown, None)
