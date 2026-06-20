"""Native ICM (Rust) <-> numpy oracle (`pokerbot.tournament.icm`) equivalence gate.

The Rust `icm.rs` port is the in-loop hot path for tournament-aware solving (chip-EV solve in
`subgame.rs`, ICM bend near the money, push/fold Nash tables). It is NOT trusted until it
reproduces the certified numpy oracle. ICM is a float recursion, so the bar is tolerance
(~1e-9), not bit-identity -- but the arithmetic order mirrors the oracle, so agreement is tight.

Covers the edge classes that matter: busted seats (stack<=0 -> bottom-rank split), payouts
shorter than the field (zero-pad), ties, hero-locked bubble_factor (+inf), and pool conservation.
Skips cleanly if the native extension is unavailable (e.g. not built on this box).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from pokerbot.tournament import icm as oracle

try:
    from pokerbot.search import native_solver

    _EXT = native_solver._try_load()
except Exception:  # noqa: BLE001
    _EXT = None

pytestmark = pytest.mark.skipif(
    _EXT is None or not hasattr(_EXT, "icm_equity"),
    reason="native battery_native (with icm_equity) not available",
)

TOL = 1e-9


def _rng_case(rng: np.random.Generator):
    """A random ICM scenario: n in 2..9, varied payout length, some busted seats, occasional ties."""
    n = int(rng.integers(2, 10))
    # stacks: positive draws, with a chance of busted (0) seats and a chance of duplicate stacks
    stacks = rng.uniform(1.0, 500.0, size=n)
    if rng.random() < 0.35:  # introduce 1..n-1 busted seats
        z = int(rng.integers(1, n))
        for i in rng.choice(n, size=z, replace=False):
            stacks[i] = 0.0
    if rng.random() < 0.25 and n >= 3:  # introduce a tie
        stacks[1] = stacks[0]
    # payouts: descending, length 1..n (sometimes shorter than field -> zero-pad path)
    k = int(rng.integers(1, n + 1))
    raw = np.sort(rng.uniform(0.1, 10.0, size=k))[::-1]
    payouts = raw.tolist()
    return stacks.tolist(), payouts


def test_icm_equity_matches_oracle_random():
    rng = np.random.default_rng(20260620)
    for _ in range(4000):
        stacks, payouts = _rng_case(rng)
        want = np.asarray(oracle.icm_equity(stacks, payouts), dtype=float)
        got = np.asarray(_EXT.icm_equity(stacks, payouts), dtype=float)
        assert got.shape == want.shape
        assert np.allclose(got, want, atol=TOL, rtol=0.0), (stacks, payouts, want, got)
        # pool conservation: sum(equity) == sum(payouts[:n])
        n = len(stacks)
        pool = float(sum(payouts[:n]))
        assert abs(float(got.sum()) - pool) <= 1e-7


def test_bubble_factor_matches_oracle_random():
    rng = np.random.default_rng(7771)
    finite_seen = 0
    inf_seen = 0
    for _ in range(2000):
        stacks, payouts = _rng_case(rng)
        n = len(stacks)
        hero, villain = rng.choice(n, size=2, replace=False)
        want = oracle.bubble_factor(stacks, payouts, int(hero), int(villain))
        got = _EXT.bubble_factor(stacks, payouts, int(hero), int(villain))
        if math.isinf(want):
            assert math.isinf(got)
            inf_seen += 1
        else:
            assert abs(got - want) <= 1e-7 * max(1.0, abs(want)), (stacks, payouts, hero, villain, want, got)
            finite_seen += 1
    assert finite_seen > 0  # sanity: the random sweep actually exercised finite bubble factors


def test_icm_delta_matches_oracle_random():
    rng = np.random.default_rng(424242)
    for _ in range(2000):
        stacks, payouts = _rng_case(rng)
        n = len(stacks)
        # a chip-conserving delta: move chips between two seats
        a, b = rng.choice(n, size=2, replace=False)
        amt = float(rng.uniform(1.0, min(stacks[a] if stacks[a] > 0 else 50.0, 200.0)))
        delta = [0.0] * n
        delta[int(a)] = amt
        delta[int(b)] = -amt
        want = np.asarray(oracle.icm_delta(stacks, delta, payouts), dtype=float)
        got = np.asarray(_EXT.icm_delta(stacks, delta, payouts), dtype=float)
        assert np.allclose(got, want, atol=TOL, rtol=0.0), (stacks, delta, payouts, want, got)


def test_known_anchors():
    # winner-take-all, equal HU stacks -> 0.5/0.5
    assert np.allclose(_EXT.icm_equity([100.0, 100.0], [1.0, 0.0]), [0.5, 0.5], atol=TOL)
    # classic 3-way ICM textbook-shape: leader has < proportional equity (ICM compression)
    eq = np.asarray(_EXT.icm_equity([5000.0, 3000.0, 2000.0], [50.0, 30.0, 20.0]))
    assert eq[0] < 50.0  # chip leader's $-share is below its chip-share*pool (ICM)
    assert abs(eq.sum() - 100.0) < 1e-7
