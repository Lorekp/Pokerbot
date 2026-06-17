"""Known-answer tests for JAX hand evaluator."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from pokerbot.engine.hand_eval_jax import evaluate_seven, score_five, card_rank, card_suit


def _cards(*indices: int) -> jnp.ndarray:
    return jnp.array(indices, dtype=jnp.int32)


def test_royal_flush_beats_quads() -> None:
    royal = _cards(48, 44, 40, 36, 32, 28, 24)  # Ah Kh Qh Jh Th + 2 hole
    quads = _cards(0, 4, 8, 12, 16, 20, 24)
    assert int(evaluate_seven(royal)) > int(evaluate_seven(quads))


def test_wheel_straight() -> None:
    # A-5 wheel on board
    ranks = jnp.array([14, 5, 4, 3, 2], dtype=jnp.int32)
    suits = jnp.array([0, 1, 2, 3, 0], dtype=jnp.int32)
    s = score_five(ranks, suits)
    assert int(s) > 0


def test_pair_vs_high_card() -> None:
    pair = jnp.array([14, 14, 13, 12, 11], dtype=jnp.int32)
    high = jnp.array([14, 13, 12, 11, 9], dtype=jnp.int32)
    suits = jnp.zeros(5, dtype=jnp.int32)
    assert int(score_five(pair, suits)) > int(score_five(high, suits))
