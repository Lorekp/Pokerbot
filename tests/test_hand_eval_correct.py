"""Regression tests for the NLHE hand evaluator correctness.

These guard the 2026-06-08 fix: under JAX x64-off the old ``(cat << 32) + primary`` packing
truncated to int32, discarding the hand CATEGORY, so showdowns were mis-ranked (a flush could
lose to two pair). The evaluator now packs ``cat * 15**5 + within`` with a correct base-15
within-category code. We assert:
  1. the category ladder is strictly monotonic (royal > quads > ... > high),
  2. the np engine and the JAX engine agree (they must stay bit-identical),
  3. the specific bug case (flush beats two pair) is fixed,
  4. (if phevaluator is installed) zero winner/tie disagreements vs an independent evaluator.
"""

from __future__ import annotations

import numpy as np
import pytest

from pokerbot.engine import hand_eval_jax as HE
from pokerbot.engine.np_engine import evaluate_seven as ev_np, player_showdown_score

import jax.numpy as jnp

RANKS = "23456789TJQKA"
SUITS = "cdhs"


def _cid(r: str, s: str) -> int:
    return RANKS.index(r) * 4 + SUITS.index(s)


def _ev_np(cards) -> int:
    return int(ev_np(np.asarray(cards, dtype=np.int32)))


def _ev_jax(cards) -> int:
    return int(HE.evaluate_seven(jnp.asarray(list(cards), dtype=jnp.int32)))


_LADDER = {
    "royal": [("A", "s"), ("K", "s"), ("Q", "s"), ("J", "s"), ("T", "s"), ("2", "c"), ("3", "d")],
    "quads": [("9", "s"), ("9", "c"), ("9", "d"), ("9", "h"), ("K", "s"), ("2", "c"), ("3", "d")],
    "fullhouse": [("A", "s"), ("A", "c"), ("A", "d"), ("K", "h"), ("K", "s"), ("2", "c"), ("3", "d")],
    "flush": [("A", "h"), ("J", "h"), ("8", "h"), ("5", "h"), ("2", "h"), ("K", "s"), ("Q", "d")],
    "straight": [("5", "s"), ("6", "c"), ("7", "d"), ("8", "h"), ("9", "s"), ("2", "c"), ("3", "d")],
    "trips": [("9", "s"), ("9", "c"), ("9", "d"), ("K", "h"), ("5", "s"), ("2", "c"), ("3", "h")],
    "twopair": [("A", "s"), ("A", "c"), ("K", "d"), ("K", "h"), ("5", "s"), ("2", "c"), ("3", "d")],
    "pair": [("A", "s"), ("A", "c"), ("9", "d"), ("7", "h"), ("5", "s"), ("2", "c"), ("3", "d")],
    "high": [("A", "s"), ("J", "c"), ("9", "d"), ("7", "h"), ("5", "s"), ("2", "c"), ("3", "d")],
}
_ORDER = ["royal", "quads", "fullhouse", "flush", "straight", "trips", "twopair", "pair", "high"]


def _hand(name: str) -> list[int]:
    return [_cid(r, s) for r, s in _LADDER[name]]


def test_category_ladder_monotonic_np():
    vals = [_ev_np(_hand(n)) for n in _ORDER]
    assert all(vals[i] > vals[i + 1] for i in range(len(vals) - 1)), dict(zip(_ORDER, vals))


def test_category_ladder_monotonic_jax():
    vals = [_ev_jax(_hand(n)) for n in _ORDER]
    assert all(vals[i] > vals[i + 1] for i in range(len(vals) - 1)), dict(zip(_ORDER, vals))


def test_flush_beats_two_pair_showdown():
    # The exact bug: K-high flush must beat AA/JJ two pair at showdown.
    board = np.array([_cid("A", "h"), _cid("J", "h"), _cid("8", "h"), _cid("5", "s"), _cid("2", "d")], np.int32)
    valid = np.ones(5, bool)
    flush = np.array([_cid("K", "h"), _cid("3", "h")], np.int32)
    twopair = np.array([_cid("A", "s"), _cid("J", "s")], np.int32)
    assert int(player_showdown_score(flush, board, valid)) > int(player_showdown_score(twopair, board, valid))


def test_within_category_tiebreak():
    # JJ44 (8 kicker) beats JJ33 (A kicker): second pair decides before kicker.
    board = [_cid("3", "d"), _cid("4", "s"), _cid("J", "h"), _cid("2", "s"), _cid("J", "c")]
    jj44 = _ev_np([_cid("8", "h"), _cid("4", "d")] + board)
    jj33 = _ev_np([_cid("3", "h"), _cid("A", "d")] + board)
    assert jj44 > jj33


def test_np_jax_identical_random():
    rng = np.random.default_rng(2024)
    for _ in range(500):
        d = rng.permutation(52)
        ha, hb = np.concatenate([d[:2], d[4:9]]), np.concatenate([d[2:4], d[4:9]])
        na, nb = _ev_np(ha), _ev_np(hb)
        ja, jb = _ev_jax(ha), _ev_jax(hb)
        assert (na > nb) == (ja > jb) and (na == nb) == (ja == jb)


def test_agrees_with_phevaluator():
    phe = pytest.importorskip("phevaluator")
    evaluate_cards = phe.evaluate_cards

    def to_str(x: int) -> str:
        return RANKS[x // 4] + SUITS[x % 4]

    rng = np.random.default_rng(99)
    disagreements = 0
    N = 3000
    for _ in range(N):
        d = rng.permutation(52)
        ha, hb = np.concatenate([d[:2], d[4:9]]), np.concatenate([d[2:4], d[4:9]])
        na, nb = _ev_np(ha), _ev_np(hb)
        pa = evaluate_cards(*[to_str(int(c)) for c in ha])
        pb = evaluate_cards(*[to_str(int(c)) for c in hb])
        ours = 1 if na > nb else (-1 if na < nb else 0)
        ref = 1 if pa < pb else (-1 if pa > pb else 0)  # phevaluator: lower is better
        disagreements += ours != ref
    assert disagreements == 0, f"{disagreements}/{N} disagreements vs phevaluator"
