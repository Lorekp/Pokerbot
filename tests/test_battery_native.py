"""Bit-identity gate for the native (Rust) engine drop-in (tools/battery_native via
pokerbot/search/native_solver.py).

Tier A: native apply_action / legal_mask / sample_chance / showdown returns must be byte-identical to the
pure-Python np_engine over random hands (incl. showdowns + side pots) and evaluate_seven fuzz.

If the extension isn't built, the wrapper falls back to pure Python and these tests are skipped (the
fallback path is exercised by the rest of the suite). When it IS built, this pins the equivalence so a
miscompile / drift is caught in CI.
"""
from __future__ import annotations

import random

import numpy as np
import pytest

jax = pytest.importorskip("jax")

from pokerbot.engine.jax_engine import init_hand
from pokerbot.engine.np_engine import (
    apply_action_np as apply_py,
    compute_returns_np as returns_py,
    evaluate_seven as eval7_py,
    is_chance_node_np as is_chance_py,
    legal_action_mask_np as mask_py,
    sample_chance_np as chance_py,
    to_numpy_state,
)
from pokerbot.search import native_solver as ns

pytestmark = pytest.mark.skipif(not ns.available(), reason="battery_native extension not built")


def _eq_state(a, b):
    for f in a._fields:
        if not np.array_equal(np.asarray(getattr(a, f)), np.asarray(getattr(b, f))):
            return f
    return None


def test_extension_selfcheck_passed():
    assert ns._SELFCHECK_OK, "native startup self-check must pass when the extension is available"


def test_engine_transitions_bit_identical():
    ns.enable()
    rng = random.Random(7)
    n = 0
    for _ in range(150):
        key = jax.random.PRNGKey(rng.randint(0, 10**9))
        env = to_numpy_state(init_hand(key, button=rng.randint(0, 5)))
        depth = 0
        while not bool(env.done) and depth < 250:
            if is_chance_py(env):
                assert ns.is_chance_node_np(env) is True
                e_n, e_p = ns.sample_chance_np(env), chance_py(env)
                assert _eq_state(e_n, e_p) is None
                env = e_p
                n += 1
                continue
            seat = int(env.action_seat)
            if seat < 0:
                break
            m_n, m_p = ns.legal_action_mask_np(env, seat), mask_py(env, seat)
            assert np.array_equal(m_n, m_p)
            legal = np.flatnonzero(m_p)
            if legal.size == 0:
                break
            passive = [a for a in legal.tolist() if a in (1, 2)]
            a = int(rng.choice(passive if (passive and rng.random() < 0.7) else list(legal)))
            e_n, e_p = ns.apply_action_np(env, a), apply_py(env, a)
            assert _eq_state(e_n, e_p) is None
            env = e_p
            depth += 1
            n += 1
    assert n > 2000  # enough coverage to be meaningful


def test_evaluate_seven_bit_identical():
    ns.enable()
    rng = random.Random(11)
    for _ in range(4000):
        cards = np.asarray(rng.sample(range(52), 7), dtype=np.int32)
        assert int(ns.evaluate_seven(cards)) == int(eval7_py(cards))


def test_compute_returns_sidepots_bit_identical():
    rng = random.Random(13)
    for _ in range(2000):
        cards = rng.sample(range(52), 17)
        holes = np.asarray(cards[:12], dtype=np.int32).reshape(6, 2)
        board = np.asarray(cards[12:17], dtype=np.int32)
        status = np.asarray([rng.choice([1, 2, 3]) for _ in range(6)], dtype=np.int8)
        committed = np.asarray([rng.choice([0, 50, 100, 200, 300, 500]) for _ in range(6)], dtype=np.int32)
        button = rng.randint(0, 5)
        n_showdown = int(((status == 2) | (status == 3)).sum())
        r_p = returns_py(holes=holes, board=board, seat_status=status,
                         committed_total=committed, button=button, n_showdown=n_showdown)
        r_n = ns._EXT.compute_returns([int(x) for x in holes.reshape(-1)],
                                      [int(x) for x in board], [int(x) for x in status],
                                      [int(x) for x in committed], button, n_showdown)
        assert np.array_equal(np.asarray(r_n, dtype=np.int32), r_p)


def test_disable_falls_back_to_python():
    ns.disable()
    try:
        rng = random.Random(3)
        env = to_numpy_state(init_hand(jax.random.PRNGKey(123), button=2))
        # With native disabled, the wrapper must return exactly the pure-Python result.
        m = ns.legal_action_mask_np(env, int(env.action_seat))
        assert np.array_equal(m, mask_py(env, int(env.action_seat)))
    finally:
        ns.enable()
