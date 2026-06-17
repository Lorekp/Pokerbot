"""Fixed-max-layer side-pot construction and terminal returns (int32 chips)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array

from pokerbot.engine.hand_eval_jax import player_showdown_score

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
    offsets = jnp.arange(NUM_PLAYERS)
    seats = (button + 1 + offsets) % NUM_PLAYERS
    is_elig = eligible[seats]
    idx = jnp.argmax(is_elig)
    return jnp.where(jnp.any(is_elig), seats[idx], button)


def award_layer(amount: Array, eligible: Array, scores: Array, button: Array) -> Array:
    best = jnp.max(jnp.where(eligible, scores, jnp.int64(-1)))
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

    def showdown(_: None) -> Array:
        board_valid = board >= 0
        scores = jax.vmap(lambda s: player_showdown_score(holes[s], board, board_valid))(jnp.arange(NUM_PLAYERS))
        scores = jnp.where(_eligible_to_win(seat_status), scores, jnp.int64(-1))

        capped = _cap_contributions(committed_total)
        excess = committed_total - capped
        winnings = jnp.zeros(NUM_PLAYERS, dtype=jnp.int32)

        positive = jnp.where(capped > 0, capped, -1)
        levels = jnp.sort(jnp.unique(positive, size=MAX_POTS, fill_value=-1))
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
