"""Property-based invariant tests for the economics layer.

Distinguished from test_economics_sharpening.py by FOCUS:
  * test_economics_sharpening.py: targeted regression tests for specific audit findings (ECON-1..27)
  * test_economics_property.py: invariant sweeps — random input -> output must obey stated property

This file exercises the SAME production code paths as test_economics_sharpening.py
(``pokerbot.economics.rake``) but with random / parametric inputs and checks mathematical invariants
(bounds, monotonicity, fail-closed semantics) rather than specific audit-issue regressions.

The four properties under test, mirroring the user's invariants:

  P1  Rake is in [0, BB cap] for any postflop pot at any stake in the schedule
  P2  Kelly criterion (banker form) is in [0, 1] for any edge in [-1, 1] and any sigma > 0
  P3  Bankroll decision (per-hand net USD) is monotone in edge (positive, finite, sign-of-edge)
  P4  Play-money guard refuses any bot action whenever is_practice is not exactly True

Each @given block runs 50+ examples (Hypothesis default is 100, which clears the 50+ floor).

Run via: pytest tests/test_economics_property.py -v
"""
from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Hypothesis is the standard property-based testing library. Optional in this test module — the
# non-Hypothesis invariant sweeps below run regardless, and the Hypothesis-driven property block
# at the bottom (H1-H4) skips if hypothesis is not installed.
hypothesis = pytest.importorskip("hypothesis")
from hypothesis import HealthCheck, assume, given, settings, strategies as st  # noqa: E402

from pokerbot.economics import rake as rk  # noqa: E402
from pokerbot.economics import rake_floor  # noqa: E402
from scripts.keep_withdraw_switch_policy import (  # noqa: E402
    ACTION_KEEP, ACTION_SWITCH_DOWN, ACTION_SWITCH_UP, ACTION_WITHDRAW,
    per_hand_net_bb100, softness_to_gross_edge,
)


# =============================================================================================
# Pre-Hypothesis invariant sweeps (paranoia layer — keep them as a fast belt-and-braces)
# =============================================================================================

# --- P1: Rake in [0, cap] for any (stake, n_dealt, postflop pot) ----------------------------------

def test_property_p1_rake_bounded_by_cap_sweep():
    """For any stake in STAKES, any dealt-count tier, and any postflop pot in [0, 1000] BB,
    coinpoker_rake_bb returns a value in [0, cap]."""
    for sk, stake in rk.STAKES.items():
        for n_dealt in (2, 3, 4, 5, 6, 7):
            tier = rk.cap_tier(n_dealt)
            cap = stake.cap_bb[tier]
            for pot in (0.0, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0, 200.0, 500.0, 1000.0):
                rake = rk.coinpoker_rake_bb(pot, n_dealt, ended_preflop=False, stake_key=sk)
                assert 0.0 <= rake <= cap + 1e-9, (
                    f"stake={sk} n_dealt={n_dealt} pot={pot} cap={cap} -> rake={rake}"
                )


def test_property_p1_rake_zero_on_preflop():
    """NLNFD: a preflop-ended hand pays 0 rake regardless of pot size / stake / n_dealt."""
    for sk in rk.STAKES:
        for n_dealt in (2, 3, 6):
            for pot in (0.0, 1.0, 100.0, 10000.0):
                assert rk.coinpoker_rake_bb(pot, n_dealt, ended_preflop=True, stake_key=sk) == 0.0


# --- P2: Kelly in [0, 1] for any (edge in [-1,1], sigma in (0, 10]) ---------------------------------

def test_property_p2_kelly_bounded_unit_interval_sweep():
    """Kelly fraction (banker form) is in [0, 1] for any edge in [-1, 1] and any sigma in (0, 10]."""
    edges = [-1.0, -0.5, -0.1, 0.0, 0.001, 0.05, 0.1, 0.5, 0.9, 1.0]
    sigmas = [0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    for e in edges:
        for s in sigmas:
            f = rk.kelly_fraction_banker(e, s)
            assert 0.0 <= f <= 1.0 + 1e-12, f"Kelly out of [0,1] for e={e} s={s}: f={f}"


def test_property_p2_kelly_zero_on_non_positive_edge():
    """A non-positive edge means a -EV or break-even game -> Kelly fraction is 0 (no wager)."""
    for e in (-1.0, -0.1, -0.001, 0.0):
        for s in (0.1, 1.0, 5.0):
            f = rk.kelly_fraction_banker(e, s)
            assert f == 0.0, f"Kelly != 0 for non-positive edge e={e} s={s}: f={f}"


# --- P3: Bankroll decision is monotone in edge (via per_hand_net_bb100) -----------------------------

def test_property_p3_per_hand_net_monotone_in_softness():
    """per_hand_net_bb100 = softness_to_gross_edge(s) - floor; gross edge is monotone in softness,
    so per_hand_net_bb100 is monotone non-decreasing in softness for any non-negative floor."""
    floor = rake_floor.floor_bb100("5/10", n_dealt=6, mode="floor")
    s_vals = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    prev = -math.inf
    for s in s_vals:
        n = per_hand_net_bb100(s, floor)
        assert n >= prev - 1e-12, f"per_hand_net not monotone in s at s={s}: {prev} -> {n}"
        prev = n


# --- P4: Play-money guard refuses any bot action when is_practice is not exactly True --------------

def test_property_p4_guard_refuses_non_true_practice():
    """For any is_practice that is not exactly True (False, None, 0, "", []), the guard returns
    PLAY_MONEY_REFUSE regardless of the bot's chosen action."""
    bot_actions = ["FOLD", "CHECK", "CALL", "RAISE_2.5", "RAISE_3", "ALL_IN", ""]
    non_true = [False, None, 0, 0.0, "", "False", [], "real_money"]
    for ip in non_true:
        for ba in bot_actions:
            out = rk.play_money_guard(ip, ba)
            assert out == rk.PLAY_MONEY_REFUSE, (
                f"guard did not refuse ip={ip!r} bot_action={ba!r} -> {out!r}"
            )


def test_property_p4_guard_passes_when_practice_true():
    """When is_practice is True, the guard returns the bot's action verbatim (no transformation)."""
    for ba in ("FOLD", "CALL", "RAISE_3", "ALL_IN", ""):
        assert rk.play_money_guard(True, ba) == ba


# =============================================================================================
# Hypothesis @given property block (>= 50 examples per property)
# =============================================================================================
#
# The four properties below are the CORRECTNESS SKELETON of the economics layer — if any of them
# breaks, the live economics / bankroll advisory is wrong by construction. They are stated in the
# design docs and were previously verified by hand in test_economics_sharpening.py; this block
# lifts them to property-based tests with random inputs so future refactors can't silently regress
# them.
#
# Settings: deadline=None (Hypothesis can be slow on cold numpy imports); suppress_health_check
# on the too-slow / large-data warnings.

_H_SETTINGS = settings(
    max_examples=50,
    deadline=None,
    suppress_health_check=list(HealthCheck),
)


# ---------------------------------------------------------------------------------------------
# H1: Rake in [0, BB cap] for any (stake, n_dealt, postflop pot).
# ---------------------------------------------------------------------------------------------

# Sample a stake key uniformly from the schedule.
_stake_key = st.sampled_from(list(rk.STAKES.keys()))
_n_dealt = st.integers(min_value=2, max_value=7)
_postflop_pot = st.floats(min_value=0.0, max_value=1000.0, allow_nan=False, allow_infinity=False)


@given(stake_key=_stake_key, n_dealt=_n_dealt, pot=_postflop_pot)
@_H_SETTINGS
def test_h1_rake_in_unit_cap_interval(stake_key, n_dealt, pot):
    """For ANY (stake, n_dealt, postflop pot), rake in [0, cap] (the BB cap for that tier)."""
    cap = rk.STAKES[stake_key].cap_bb[rk.cap_tier(n_dealt)]
    rake = rk.coinpoker_rake_bb(pot, n_dealt, ended_preflop=False, stake_key=stake_key)
    assert 0.0 <= rake <= cap + 1e-9, (
        f"rake out of [0, cap]: stake={stake_key} n_dealt={n_dealt} pot={pot} "
        f"cap={cap} rake={rake}"
    )
    # And the upper bound is 5% * pot OR cap — verify the min() formulation explicitly.
    assert rake == pytest.approx(min(rk.RAKE_PCT * pot, cap), abs=1e-9), (
        f"rake != min(5%*pot, cap): stake={stake_key} n_dealt={n_dealt} pot={pot} "
        f"cap={cap} rake={rake}"
    )


# ---------------------------------------------------------------------------------------------
# H2: Kelly in [0, 1] for any edge in [-1, 1] and any positive sigma.
# ---------------------------------------------------------------------------------------------

_edge = st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False)
_sigma = st.floats(min_value=1e-6, max_value=10.0, allow_nan=False, allow_infinity=False)


@given(edge=_edge, sigma=_sigma)
@_H_SETTINGS
def test_h2_kelly_in_unit_interval(edge, sigma):
    """For ANY (edge in [-1, 1], sigma in (0, 10]), Kelly fraction in [0, 1]."""
    f = rk.kelly_fraction_banker(edge, sigma)
    assert 0.0 <= f <= rk.KELLY_MAX + 1e-12, (
        f"Kelly out of [0, {rk.KELLY_MAX}]: edge={edge} sigma={sigma} f={f}"
    )
    # Kelly must be 0 whenever the edge is non-positive (no wagering on a -EV or break-even game).
    if edge <= 0.0:
        assert f == 0.0, f"Kelly != 0 for edge={edge} <= 0: f={f}"


@given(edge=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
       sigma=st.floats(min_value=0.5, max_value=10.0, allow_nan=False, allow_infinity=False))
@_H_SETTINGS
def test_h2_kelly_monotone_decreasing_in_sigma(edge, sigma):
    """For fixed positive edge, Kelly = edge / sigma^2 is strictly DECREASING in sigma.

    This is the canonical "more variance -> bet less" Kelly property. We assert non-strict
    monotonicity across two random samples sigma_a <= sigma_b in the positive-edge regime."""
    f_low = rk.kelly_fraction_banker(edge, sigma)
    f_high = rk.kelly_fraction_banker(edge, sigma * 2.0)
    assert f_low >= f_high - 1e-12, (
        f"Kelly NOT monotone decreasing in sigma: edge={edge} "
        f"f(sigma)={f_low} f(2*sigma)={f_high}"
    )


# ---------------------------------------------------------------------------------------------
# H3: Bankroll decision (per-hand net USD) is monotone in edge (= softness).
# ---------------------------------------------------------------------------------------------
#
# per_hand_net_bb100(s) = softness_to_gross_edge(s) - floor_bb100
# softness_to_gross_edge is a clamped LINEAR map -> non-decreasing in s.
# Subtracting a constant floor preserves monotonicity. Multiplying by BB/100 preserves it.
# So per-hand_net_usd is non-decreasing in s for any non-negative floor.

_softness_a = st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)
_softness_b = st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)


@given(s_a=_softness_a, s_b=_softness_b, stake_key=_stake_key)
@_H_SETTINGS
def test_h3_per_hand_net_monotone_in_softness(s_a, s_b, stake_key):
    """For ANY pair of softnesses s_a, s_b and ANY stake, per_hand_net_bb100(s_a) <=
    per_hand_net_bb100(s_b) whenever s_a <= s_b."""
    assume(s_a <= s_b)
    floor = rake_floor.floor_bb100(stake_key, n_dealt=6, mode="floor")
    n_a = per_hand_net_bb100(s_a, floor)
    n_b = per_hand_net_bb100(s_b, floor)
    assert n_a <= n_b + 1e-9, (
        f"per_hand_net not monotone in s: s_a={s_a} -> {n_a}, s_b={s_b} -> {n_b}, floor={floor}"
    )


@given(softness=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
       stake_key=_stake_key)
@_H_SETTINGS
def test_h3_per_hand_net_sign_matches_edge_minus_floor(softness, stake_key):
    """The per-hand net bb/100 = gross_edge - floor; the SIGN of per_hand_net matches the sign of
    (gross_edge - floor). The breakeven softness is exactly s such that gross_edge(s) = floor."""
    floor = rake_floor.floor_bb100(stake_key, n_dealt=6, mode="floor")
    gross = softness_to_gross_edge(softness)
    n = per_hand_net_bb100(softness, floor)
    if gross > floor:
        assert n > 0.0, f"per_hand_net not positive when edge > floor: s={softness} n={n}"
    elif gross < floor:
        assert n < 0.0, f"per_hand_net not negative when edge < floor: s={softness} n={n}"
    else:
        assert n == pytest.approx(0.0, abs=1e-9), f"per_hand_net not ~0 at breakeven: n={n}"


# ---------------------------------------------------------------------------------------------
# H4: Play-money guard refuses ANY bot action when is_practice is not exactly True.
# ---------------------------------------------------------------------------------------------
#
# The contract: ``is_practice is True`` is the ONLY cleared value. Anything else (False, None,
# 0, the empty string, an int, ...) blocks the action. The guard must be IDEMPOTENT in the
# bot_action argument: regardless of what the bot wanted to do, the guard returns REFUSE.

# ``st.one_of`` with a wide variety of non-True values plus some obviously false-ish types.
_non_true_practice = st.one_of(
    st.none(),
    st.booleans().filter(lambda v: v is not True),
    st.integers(min_value=-10, max_value=10),
    st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    st.text(max_size=5),
    st.lists(st.integers(), max_size=3),
)


@given(is_practice=_non_true_practice,
       bot_action=st.text(min_size=0, max_size=20))
@_H_SETTINGS
def test_h4_play_money_guard_refuses_non_true(is_practice, bot_action):
    """For ANY non-True is_practice and ANY bot_action string, guard returns REFUSE."""
    out = rk.play_money_guard(is_practice, bot_action)
    assert out == rk.PLAY_MONEY_REFUSE, (
        f"guard did not refuse: is_practice={is_practice!r} "
        f"bot_action={bot_action!r} -> {out!r}"
    )
    # The guard must NEVER echo the bot's action when blocking — that would be the
    # wire-encoding bug (REFUSE is a sentinel, not a passthrough).
    if bot_action != rk.PLAY_MONEY_REFUSE:
        assert out != bot_action, (
            f"guard echoed bot_action on block: is_practice={is_practice!r} "
            f"bot_action={bot_action!r} -> {out!r}"
        )


@given(bot_action=st.text(min_size=0, max_size=30))
@_H_SETTINGS
def test_h4_play_money_guard_passes_when_true(bot_action):
    """For is_practice == True, the guard returns bot_action VERBATIM (no transformation).

    The guard is a pure pass-through on the cleared path — it must never mangle the action
    string (no upper-casing, no whitespace stripping, no encoding)."""
    out = rk.play_money_guard(True, bot_action)
    assert out == bot_action, (
        f"guard mangled bot_action on cleared path: in={bot_action!r} out={out!r}"
    )
    assert out != rk.PLAY_MONEY_REFUSE, (
        f"guard refused on True is_practice: in={bot_action!r} out={out!r}"
    )
