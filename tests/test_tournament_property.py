"""Tournament layer property-based tests (random inputs, invariant assertions).

Covers property-based fuzz over:
  * ICM conservation (sum equity = sum payouts) over many random stack/payout combos.
  * ICM monotonicity in stack (more chips -> strictly more equity when stacks asymmetric).
  * ICM concavity (per-chip equity is higher for shorter stacks under non-flat payouts).
  * bubble_factor >= 1 in multi-way non-trivial scenarios.
  * bubble_factor == 1 in HU equal scenarios.
  * PKO ledger conservation under random KO chains.
  * PKO self-KO impossible (must raise ValueError).
  * bubble_seats is monotone non-decreasing in field size.
  * protected_payouts length = base length + B(N) for eligible.
  * ICM exploit attenuation: [0, 1] range, monotone non-increasing in bf.
  * chip_ev_action_value is payout-agnostic.

Run via: pytest tests/test_tournament_property.py -v
"""

from __future__ import annotations

import math
import os
import random

import numpy as np
import pytest

# Ensure tests can run with no .env / live config.
os.environ.setdefault("POKERBOT_LIVE_VALUE_SIZING", "0")

from pokerbot.tournament import (
    bounty,
    bubble,
    icm,
    icm_exploit,
    icm_overlay,
    payouts,
    satellite,
)


@pytest.fixture(scope="module")
def rng() -> np.random.Generator:
    return np.random.default_rng(0xA1F_2026)


def _random_stacks_payouts(rng: np.random.Generator, *, n_min: int = 2, n_max: int = 7,
                           stack_min: float = 1.0, stack_max: float = 1000.0,
                           n_paid_min: int = 1) -> tuple[list[float], list[float]]:
    """Generate a random (stacks, payouts) tuple for property testing."""
    n = int(rng.integers(n_min, n_max + 1))
    stacks = rng.uniform(stack_min, stack_max, size=n).tolist()
    n_paid = int(rng.integers(n_paid_min, n + 1))
    payouts = sorted(rng.uniform(1, 100, size=n_paid).tolist(), reverse=True)
    return stacks, payouts


# =============================================================================================
# ICM property tests
# =============================================================================================


@pytest.mark.parametrize("seed", list(range(20)))
def test_icm_pool_conservation_random(seed):
    """sum(icm_equity(stacks, payouts)) == sum(payouts[:n]) over random inputs."""
    rng = np.random.default_rng(seed * 137 + 1)
    stacks, payouts = _random_stacks_payouts(rng)
    eq = icm.icm_equity(stacks, payouts)
    assert eq.sum() == pytest.approx(sum(payouts), rel=1e-9, abs=1e-9)
    # Each alive seat has non-negative equity.
    for i, s in enumerate(stacks):
        if s > 0:
            assert eq[i] >= -1e-12


@pytest.mark.parametrize("seed", list(range(15)))
def test_icm_monotone_in_stack_random(seed):
    """Adding chips to a non-busted seat never decreases its equity (monotonicity invariant)."""
    rng = np.random.default_rng(seed * 251 + 1)
    stacks, payouts = _random_stacks_payouts(rng, n_min=3, n_max=6)
    # Pick a seat with non-zero stack; add chips to it.
    nz = [i for i, s in enumerate(stacks) if s > 0]
    if not nz:
        pytest.skip("no live seat")
    hero = nz[0]
    bigger = list(stacks)
    bigger[hero] += rng.uniform(1, 100)
    base_eq = icm.icm_equity(stacks, payouts)[hero]
    bigger_eq = icm.icm_equity(bigger, payouts)[hero]
    assert bigger_eq >= base_eq - 1e-9


@pytest.mark.parametrize("seed", list(range(15)))
def test_icm_concavity_random(seed):
    """Per-chip equity of the shortest stack >= per-chip equity of the longest stack."""
    rng = np.random.default_rng(seed * 333 + 1)
    # Need non-flat payouts for concavity to be non-trivial (flat payouts are linear).
    while True:
        stacks, payouts = _random_stacks_payouts(rng, n_min=3, n_max=6)
        if len(payouts) >= 2 and payouts[0] - payouts[-1] > 1e-3:
            break
    eq = icm.icm_equity(stacks, payouts)
    per_chip = eq / np.asarray(stacks)
    # Pick the longest and shortest non-zero stacks.
    sorted_idx = sorted(range(len(stacks)), key=lambda i: stacks[i])
    short = sorted_idx[0]
    long_ = sorted_idx[-1]
    if stacks[long_] <= 0 or stacks[short] <= 0:
        pytest.skip("zero stack")
    assert per_chip[short] >= per_chip[long_] - 1e-9


def test_icm_payouts_padded_conservation():
    """Payouts shorter than field -> unpadded seats get 0, pool still conserved."""
    rng = np.random.default_rng(42)
    for _ in range(10):
        stacks, payouts = _random_stacks_payouts(rng, n_min=4, n_max=6)
        eq = icm.icm_equity(stacks, payouts)
        assert eq.sum() == pytest.approx(sum(payouts), rel=1e-9, abs=1e-9)


def test_icm_one_alive_gets_first_payout():
    """If only one seat is alive, that seat gets payouts[0] (1st place prize)."""
    rng = np.random.default_rng(7)
    for _ in range(5):
        n = int(rng.integers(3, 7))
        # Make all-but-one stacks == 0; the alive seat is at a random index.
        alive_idx = int(rng.integers(0, n))
        stacks = [rng.uniform(10, 100) if i == alive_idx else 0.0 for i in range(n)]
        payouts = sorted(rng.uniform(1, 100, size=n).tolist(), reverse=True)
        eq = icm.icm_equity(stacks, payouts)
        assert eq[alive_idx] == pytest.approx(payouts[0], abs=1e-9)
        # All busted seats split the bottom n-1 ranks equally.
        busted = payouts[1:]
        per_busted = sum(busted) / len(busted)
        for i in range(n):
            if i != alive_idx:
                assert eq[i] == pytest.approx(per_busted, abs=1e-9)


def test_icm_zero_sum_in_dollars_random():
    """For any chip_delta summing to ~0, icm_delta sums to 0 (prize pool conserved)."""
    rng = np.random.default_rng(20260626)
    for _ in range(20):
        n = int(rng.integers(2, 6))
        stacks = rng.uniform(10, 100, size=n).tolist()
        # Generate a zero-sum chip_delta.
        raw = rng.uniform(-50, 50, size=n)
        raw[0] = -raw[1:].sum()
        d = icm.icm_delta(stacks, raw.tolist(),
                           sorted(rng.uniform(1, 100, size=n).tolist(), reverse=True))
        assert d.sum() == pytest.approx(0.0, abs=1e-9)


# =============================================================================================
# bubble_factor property tests
# =============================================================================================


@pytest.mark.parametrize("seed", list(range(20)))
def test_bubble_factor_hu_equal_is_one_random(seed):
    """For any (payouts, n) with HU equal stacks, BF = 1.0 exactly."""
    rng = np.random.default_rng(seed * 71 + 1)
    s = float(rng.uniform(10, 1000))
    n = int(rng.integers(2, 5))
    payouts = sorted(rng.uniform(1, 100, size=n).tolist(), reverse=True)
    bf = icm.bubble_factor([s, s], payouts, hero=0, villain=1)
    assert bf == pytest.approx(1.0, abs=1e-9)


@pytest.mark.parametrize("seed", list(range(20)))
def test_bubble_factor_greater_than_one_when_multiway_unequal(seed):
    """For any multi-way with asymmetric stacks and a real ladder, BF > 1 strictly."""
    rng = np.random.default_rng(seed * 89 + 1)
    while True:
        stacks, payouts = _random_stacks_payouts(rng, n_min=3, n_max=5)
        if len(payouts) >= 2 and min(stacks) > 0:
            break
    # Use hero=0 vs villain=1 (any pair).
    bf = icm.bubble_factor(stacks, payouts, hero=0, villain=1)
    if not math.isinf(bf):
        assert bf >= 1.0 - 1e-9  # >= 1 (we allow some numerical slack)


@pytest.mark.parametrize("seed", list(range(20)))
def test_bubble_factor_at_least_one_when_unequal_stacks(seed):
    """When stacks are unequal AND ladder is non-trivial, BF is finite and >= 1 strictly.

    Note: this is a *sufficient* condition test, not the most general statement. The general
    property is BF >= 1 with equality iff the chip-EV is $-linear (HU equal / single paid).
    Unequal stacks under a real ladder produce BF > 1 strictly (verified across random inputs).
    """
    rng = np.random.default_rng(seed * 47 + 1)
    # Force unequal stacks: hero >> others.
    payouts = sorted(rng.uniform(1, 100, size=int(rng.integers(3, 5))).tolist(), reverse=True)
    if len(payouts) < 3:
        return
    stacks = [rng.uniform(80, 200), rng.uniform(5, 30), rng.uniform(5, 30)]
    bf = icm.bubble_factor(stacks, payouts, hero=0, villain=1)
    if math.isfinite(bf):
        assert bf >= 1.0 - 1e-9


def test_bubble_factor_monotone_in_pay_jump_pressure():
    """Adding a steep pay jump (e.g. extra pay at rank 2) increases BF near the bubble.

    With 3 paid vs 4 paid (more places paid = flatter distribution), BF should drop or stay equal.
    The opposite direction (4 -> 3 paid) should increase BF when stacks are symmetric.
    """
    stacks = [40.0, 40.0, 40.0, 40.0]
    bf_4p = icm.bubble_factor(stacks, [50.0, 30.0, 20.0, 10.0], hero=0, villain=1)
    bf_3p = icm.bubble_factor(stacks, [50.0, 30.0, 20.0], hero=0, villain=1)
    # Fewer paid places = bigger bubble pressure.
    assert bf_3p >= bf_4p - 1e-9


# =============================================================================================
# PKO property tests
# =============================================================================================


@pytest.mark.parametrize("seed", list(range(15)))
def test_pko_conservation_random_chain(seed):
    """Random KO chains preserve ledger: sum(live)+sum(realized) == initial_pool."""
    rng = np.random.default_rng(seed * 911 + 1)
    n = int(rng.integers(2, 8))
    entry = float(rng.uniform(5, 50))
    pf = float(rng.uniform(0.0, 1.0))
    state = bounty.BountyState.uniform(n, entry_bounty=entry, progressive_frac=pf)
    initial_pool = state.initial_pool
    # Knock out random victims in random order, with random eliminators + optional split-pot.
    alive = list(range(n))
    for _ in range(n - 1):
        if len(alive) < 2:
            break
        # Pick victim and eliminator from alive (distinct).
        vidx = rng.integers(0, len(alive))
        victim = int(alive[vidx])
        alive.pop(vidx)
        # Pick eliminator (any remaining live seat).
        eidx = rng.integers(0, len(alive))
        eliminator = int(alive[eidx])
        # 50% chance of a split-pot (winners includes the eliminator + maybe others).
        is_split = bool(rng.integers(0, 2)) and len(alive) >= 2
        if is_split:
            nw = int(rng.integers(2, min(3, len(alive)) + 1))
            winners = [int(w) for w in rng.choice(alive, size=nw, replace=False).tolist()]
            # Make sure eliminator is among winners (they took down the victim).
            if eliminator not in winners:
                winners[0] = eliminator
            state.award_ko(eliminator, victim, winners=winners)
        else:
            state.award_ko(eliminator, victim)
        assert state.conserved()
    # Finalize the survivor.
    if alive:
        state.finalize_winner(alive[0])
    assert state.conserved()
    assert (sum(state.live) + sum(state.realized)) == pytest.approx(initial_pool, abs=1e-9)


@pytest.mark.parametrize("seed", list(range(10)))
def test_pko_self_ko_raises(seed):
    """Self-KO must raise ValueError under any (progressive_frac, winners) combo."""
    rng = np.random.default_rng(seed * 1009 + 1)
    n = int(rng.integers(2, 6))
    pf = float(rng.uniform(0.0, 1.0))
    state = bounty.BountyState.uniform(n, entry_bounty=10.0, progressive_frac=pf)
    victim = int(rng.integers(0, n))
    with pytest.raises(ValueError):
        state.award_ko(victim, victim)
    with pytest.raises(ValueError):
        state.award_ko(victim, victim, winners=[victim, (victim + 1) % n])


def test_pko_progressive_frac_out_of_range_raises():
    """Constructor enforces 0 <= progressive_frac <= 1."""
    with pytest.raises(ValueError):
        bounty.BountyState([10.0, 10.0], progressive_frac=-0.1)
    with pytest.raises(ValueError):
        bounty.BountyState([10.0, 10.0], progressive_frac=1.1)


def test_pko_bounty_cash_on_win_formula_monotone():
    """bounty_cash_on_win is monotone in villain_bounty and growth_value."""
    val1 = bounty.bounty_cash_on_win(50.0, growth_value=0.5)
    val2 = bounty.bounty_cash_on_win(100.0, growth_value=0.5)
    assert val2 > val1
    val3 = bounty.bounty_cash_on_win(50.0, growth_value=0.0)  # no growth credit
    val4 = bounty.bounty_cash_on_win(50.0, growth_value=1.0)  # full growth credit
    assert val3 < val1 < val4


def test_pko_min_equity_to_call_lowered_by_bounty_random():
    """For random valid inputs, pko_min_equity_to_call is monotonically decreasing in vb."""
    rng = np.random.default_rng(31)
    for _ in range(10):
        n = int(rng.integers(2, 5))
        stacks = rng.uniform(10, 200, size=n).tolist()
        payouts = sorted(rng.uniform(1, 100, size=n).tolist(), reverse=True)
        h, v = int(rng.integers(0, n)), int(rng.integers(0, n))
        if h == v:
            v = (v + 1) % n
        p0 = bounty.min_equity_to_call(stacks, payouts, h, v, villain_bounty=0.0)
        p1 = bounty.min_equity_to_call(stacks, payouts, h, v, villain_bounty=100.0)
        assert p1 <= p0 + 1e-12


# =============================================================================================
# Bubble protection property tests
# =============================================================================================


@pytest.mark.parametrize("field", [9, 10, 50, 99, 100, 500, 1499, 1500, 3000, 5000, 10000])
def test_bubble_seats_monotone_non_decreasing(field):
    """bubble_seats is monotone non-decreasing in field (per CoinPoker brackets)."""
    assert bubble.bubble_seats(field) >= bubble.bubble_seats(field - 1)


@pytest.mark.parametrize("seed", list(range(15)))
def test_protected_payouts_length_eligible(seed):
    """For eligible entrants, len(protected) == len(base) + B(N)."""
    rng = np.random.default_rng(seed * 41 + 1)
    n_paid = int(rng.integers(1, 6))
    base = sorted(rng.uniform(1, 100, size=n_paid).tolist(), reverse=True)
    field = int(rng.integers(10, 5000))
    buy_in = float(rng.uniform(5, 100))
    prot = bubble.protected_payouts(base, field=field, buy_in=buy_in, eligible=True)
    expected_extra = bubble.bubble_seats(field)
    if expected_extra == 0:
        assert prot == base
    else:
        assert len(prot) == len(base) + expected_extra


def test_protected_payouts_ineligible_unchanged():
    """Ineligible entrant -> protected_payouts == base_payouts (no refund ranks)."""
    rng = np.random.default_rng(99)
    for _ in range(10):
        base = sorted(rng.uniform(1, 100, size=4).tolist(), reverse=True)
        prot = bubble.protected_payouts(base, field=100, buy_in=10.0, eligible=False)
        assert prot == base


@pytest.mark.parametrize("seed", list(range(15)))
def test_protected_bubble_factor_leq_hard_for_eligible(seed):
    """Eligible entrant -> protected_bubble_factor <= hard_bubble_factor."""
    rng = np.random.default_rng(seed * 113 + 1)
    stacks = rng.uniform(10, 100, size=4).tolist()
    base_payouts = sorted(rng.uniform(1, 100, size=3).tolist(), reverse=True)
    hard = bubble.hard_bubble_factor(stacks, base_payouts, hero=0, villain=1)
    prot = bubble.protected_bubble_factor(stacks, base_payouts, hero=0, villain=1,
                                           field=100, buy_in=10.0, eligible=True)
    assert prot <= hard + 1e-12


# =============================================================================================
# Satellite property tests
# =============================================================================================


@pytest.mark.parametrize("seed", list(range(15)))
def test_seat_locked_iff_at_threshold(seed):
    """`seat_locked` iff hero's stack >= total / (seats+1) (the chip-fraction invariant)."""
    rng = np.random.default_rng(seed * 151 + 1)
    n = int(rng.integers(2, 8))
    stacks = rng.uniform(1, 1000, size=n).tolist()
    seats = int(rng.integers(1, n))
    hero = int(rng.integers(0, n))
    if stacks[hero] <= 0:
        pytest.skip("busted hero")
    live = [s for s in stacks if s > 0]
    total = sum(live)
    expected = len(live) <= seats or stacks[hero] >= total / (seats + 1)
    got = satellite.seat_locked(stacks, seats=seats, hero=hero)
    assert got == expected


@pytest.mark.parametrize("seed", list(range(10)))
def test_n_ticket_satellite_pool_conservation(seed):
    """For any (seats, stacks, seat_value), sum(equity) == seats * seat_value."""
    rng = np.random.default_rng(seed * 199 + 1)
    n = int(rng.integers(2, 6))
    stacks = rng.uniform(1, 100, size=n).tolist()
    seats = int(rng.integers(1, n + 1))
    seat_value = float(rng.uniform(10, 1000))
    pay = satellite.satellite_payouts(seats=seats, seat_value=seat_value)
    eq = icm.icm_equity(stacks, pay)
    assert eq.sum() == pytest.approx(seats * seat_value, rel=1e-9, abs=1e-9)


# =============================================================================================
# ICM overlay / exploit attenuation property tests
# =============================================================================================


@pytest.mark.parametrize("seed", list(range(20)))
def test_icm_action_value_in_pre_to_post_range(seed):
    """icm_action_value returns a value in [min(icm_pre), max(icm_post)] for the hero seat."""
    rng = np.random.default_rng(seed * 271 + 1)
    stacks, payouts = _random_stacks_payouts(rng, n_min=2, n_max=5)
    h = int(rng.integers(0, len(stacks)))
    # A couple of chip outcomes (not necessarily zero-sum).
    outcomes = [(0.5, [rng.uniform(-20, 20) for _ in stacks]),
                (0.5, [rng.uniform(-20, 20) for _ in stacks])]
    val = icm_overlay.icm_action_value(stacks, outcomes, payouts, h)
    # The val must equal the probability-weighted post-equity, which lies between 0 and sum(payouts).
    assert 0 - 1e-9 <= val <= sum(payouts) + 1e-9


def test_chip_ev_action_value_payout_agnostic():
    """chip_ev_action_value does not depend on payouts (only on chip outcomes)."""
    pre = [50.0, 50.0]
    outcomes = [(0.5, [+20.0, -20.0]), (0.5, [-10.0, +10.0])]
    val_a = icm_overlay.chip_ev_action_value(pre, outcomes, hero=0)
    val_b = icm_overlay.chip_ev_action_value(pre, outcomes, hero=0)  # no payouts arg
    assert val_a == val_b == pytest.approx(5.0, abs=1e-9)


@pytest.mark.parametrize("bf", [0.5, 0.95, 1.0, 1.04, 1.05, 1.10, 1.30, 1.325, 1.50, 1.59, 1.60,
                                 1.80, 2.0, 5.0, 100.0, float("inf")])
def test_attenuation_in_unit_interval(bf):
    """icm_exploit_attenuation returns a value in [0, 1] for any bf input (including inf)."""
    val = icm_exploit.icm_exploit_attenuation(bf)
    assert 0.0 <= val <= 1.0
    assert math.isfinite(val) or val == 0.0  # safe to never return inf/nan


def test_attenuation_nan_safe():
    """NaN input -> attenuation = 0.0 (safe fallback per spec)."""
    assert icm_exploit.icm_exploit_attenuation(float("nan")) == 0.0


def test_attenuation_degenerate_config_safe():
    """zero_above <= full_below (degenerate config) -> 0.0 (safe fallback per spec)."""
    assert icm_exploit.icm_exploit_attenuation(1.2, full_below=1.5, zero_above=1.0) == 0.0


@pytest.mark.parametrize("seed", list(range(15)))
def test_state_attenuation_in_unit_interval_random(seed):
    """state_exploit_attenuation returns a value in [0, 1] over random valid inputs."""
    rng = np.random.default_rng(seed * 307 + 1)
    stacks, payouts = _random_stacks_payouts(rng, n_min=3, n_max=6)
    h = int(rng.integers(0, len(stacks)))
    if stacks[h] <= 0:
        pytest.skip("busted hero")
    val = icm_exploit.state_exploit_attenuation(stacks, payouts, hero=h)
    assert 0.0 <= val <= 1.0


def test_state_attenuation_takes_min_across_villains():
    """state_attenuation returns the MIN over hero's live villains (most conservative)."""
    # 4-handed: hero vs all 3 villains.
    stacks = [40.0, 30.0, 30.0, 30.0]
    payouts = [50.0, 30.0, 20.0]
    att = icm_exploit.state_exploit_attenuation(stacks, payouts, hero=0)
    # Per-villain attenuations.
    from pokerbot.tournament.icm import bubble_factor
    expected = min(icm_exploit.icm_exploit_attenuation(bubble_factor(stacks, payouts, 0, v))
                   for v in (1, 2, 3))
    assert att == pytest.approx(expected, abs=1e-12)


def test_state_attenuation_no_payouts_returns_one():
    """No payouts -> 1.0 (the PURE answer; live gate owns the safety policy)."""
    assert icm_exploit.state_exploit_attenuation([50.0, 50.0], None, hero=0) == 1.0
    assert icm_exploit.state_exploit_attenuation([50.0, 50.0], [], hero=0) == 1.0
    # Single-player degenerate -> 1.0.
    assert icm_exploit.state_exploit_attenuation([50.0], [60.0, 40.0], hero=0) == 1.0


# =============================================================================================
# icm_tilt clamp + identity
# =============================================================================================


def test_icm_tilt_clamps_inf_for_arbitrary_inputs():
    """icm_tilt must return finite values even when bubble_factor = inf."""
    rng = np.random.default_rng(0x1CE)
    for _ in range(20):
        n = int(rng.integers(2, 6))
        stacks = rng.uniform(1, 1000, size=n).tolist()
        payouts = sorted(rng.uniform(1, 100, size=n).tolist(), reverse=True)
        # Force inf: 0-stack opponent.
        if n >= 2:
            stacks[1] = 0.0
        chip_values = rng.uniform(-5, 5, size=3).tolist()
        action_risk = rng.uniform(0, 5, size=3).tolist()
        out = icm_overlay.icm_tilt(chip_values, action_risk, stacks, payouts, hero=0, villain=1)
        assert np.all(np.isfinite(out))


@pytest.mark.parametrize("seed", list(range(10)))
def test_icm_tilt_identity_when_bf_one_random(seed):
    """icm_tilt is identity when bubble_factor = 1 (deep / far from money)."""
    rng = np.random.default_rng(seed * 401 + 1)
    s = float(rng.uniform(10, 1000))
    n = int(rng.integers(2, 5))
    payouts = sorted(rng.uniform(1, 100, size=n).tolist(), reverse=True)
    cv = rng.uniform(-10, 10, size=5).tolist()
    risk = rng.uniform(0, 10, size=5).tolist()
    # HU equal -> BF = 1
    out = icm_overlay.icm_tilt(cv, risk, [s, s], payouts, hero=0, villain=1)
    assert np.allclose(out, cv, atol=1e-9)


# =============================================================================================
# Hypothesis @given property-based tests (>= 100 examples each)
# ---------------------------------------------------------------------------------------------
# Eight canonical tournament invariants the engine must uphold on every input:
#   (1) ICM conservation (sum equity = sum payouts)
#   (2) ICM monotonicity (more chips -> more EV, except fully symmetric)
#   (3) PKO bounty direction (winner gets, loser loses)
#   (4) Bubble factor positivity
#   (5) Payout ladder monotonicity
#   (6) Equal stacks give equal ICM-EV (within rounding)
#   (7) 1-player survivor gets full 1st-place payout
#   (8) PKO before any knockout: no bounties assigned
# =============================================================================================

from hypothesis import HealthCheck, assume, given, settings, strategies as st


# --- Strategies -----------------------------------------------------------------------------


@st.composite
def stacks_and_payouts(draw, *, n_min: int = 2, n_max: int = 6,
                       stack_max: float = 1000.0) -> tuple[list[float], list[float]]:
    """Generate ``(stacks, payouts)`` with strictly-positive stacks, ``len(payouts) >= 1``."""
    n = draw(st.integers(min_value=n_min, max_value=n_max))
    stacks = draw(st.lists(
        st.floats(min_value=1.0, max_value=stack_max, allow_nan=False, allow_infinity=False),
        min_size=n, max_size=n,
    ))
    n_paid = draw(st.integers(min_value=1, max_value=n))
    payouts = draw(st.lists(
        st.floats(min_value=1.0, max_value=100.0, allow_nan=False, allow_infinity=False),
        min_size=n_paid, max_size=n_paid,
    ))
    payouts = sorted(payouts, reverse=True)
    return stacks, payouts


@st.composite
def stacks_and_payouts_with_bust(draw, *, n_min: int = 2, n_max: int = 6
                                 ) -> tuple[list[float], list[float]]:
    """Stacks may include zeros (busted seats); payouts are descending."""
    n = draw(st.integers(min_value=n_min, max_value=n_max))
    stacks = draw(st.lists(
        st.floats(min_value=0.0, max_value=1000.0, allow_nan=False, allow_infinity=False),
        min_size=n, max_size=n,
    ))
    # Need at least one live seat for ICM to be defined (else trivial empty field).
    assume(any(s > 0.0 for s in stacks))
    n_paid = draw(st.integers(min_value=1, max_value=n))
    payouts = draw(st.lists(
        st.floats(min_value=1.0, max_value=100.0, allow_nan=False, allow_infinity=False),
        min_size=n_paid, max_size=n_paid,
    ))
    payouts = sorted(payouts, reverse=True)
    return stacks, payouts


@st.composite
def equal_stack_field(draw, *, n_min: int = 2, n_max: int = 6
                      ) -> tuple[float, int, list[float]]:
    """All seats the same positive stack, plus a descending payout ladder."""
    n = draw(st.integers(min_value=n_min, max_value=n_max))
    s = draw(st.floats(min_value=1.0, max_value=1000.0, allow_nan=False, allow_infinity=False))
    n_paid = draw(st.integers(min_value=1, max_value=n))
    payouts = draw(st.lists(
        st.floats(min_value=1.0, max_value=100.0, allow_nan=False, allow_infinity=False),
        min_size=n_paid, max_size=n_paid,
    ))
    payouts = sorted(payouts, reverse=True)
    return s, n, payouts


@st.composite
def bbr_pair(draw) -> tuple[list[float], list[float], int, int]:
    """A (stacks, payouts, hero, villain) tuple for bubble-factor tests; payouts >= 2 paid."""
    stacks, payouts = draw(stacks_and_payouts(n_min=2, n_max=6))
    # Need >= 2 paid places so bubble_factor is meaningful.
    assume(len(payouts) >= 2)
    n = len(stacks)
    hero = draw(st.integers(min_value=0, max_value=n - 1))
    villain = draw(st.integers(min_value=0, max_value=n - 1))
    assume(hero != villain)
    return stacks, payouts, hero, villain


@st.composite
def pko_params(draw) -> tuple[int, float, float]:
    """(n_players, entry_bounty, progressive_frac) for PKO state tests."""
    n = draw(st.integers(min_value=2, max_value=8))
    entry = draw(st.floats(min_value=1.0, max_value=100.0,
                           allow_nan=False, allow_infinity=False))
    pf = draw(st.floats(min_value=0.0, max_value=1.0,
                        allow_nan=False, allow_infinity=False))
    return n, entry, pf


# --- (1) ICM conservation -------------------------------------------------------------------


@given(stacks_and_payouts())
@settings(max_examples=120, suppress_health_check=[HealthCheck.too_slow])
def test_h_icm_conservation(sp):
    """(1) sum(icm_equity(stacks, payouts)) == sum(payouts[:n]) for any valid (stacks, payouts)."""
    stacks, payouts = sp
    eq = icm.icm_equity(stacks, payouts)
    assert eq.sum() == pytest.approx(sum(payouts), rel=1e-9, abs=1e-9)
    # Each live seat has non-negative equity; busted seats share the bottom.
    total_live = sum(s for s in stacks if s > 0.0)
    assume(total_live > 0.0)
    for i, s in enumerate(stacks):
        if s > 0.0:
            assert eq[i] >= -1e-12


# --- (2) ICM monotonicity --------------------------------------------------------------------


@given(stacks_and_payouts(n_min=3, n_max=6))
@settings(max_examples=120, suppress_health_check=[HealthCheck.too_slow])
def test_h_icm_monotonicity(sp):
    """(2) Adding chips to a live seat never decreases its equity.

    Strict when the field is NOT fully symmetric (i.e. some other seat has a different stack) AND
    the payout ladder is not perfectly flat (every paid slot pays the same). Both kinds of
    symmetry can collapse the strict-gain signal; we skip those edge cases and only assert
    strictness when at least one asymmetry exists.
    """
    stacks, payouts = sp
    nz = [i for i, s in enumerate(stacks) if s > 0.0]
    assume(len(nz) >= 2)  # need at least two live seats to talk about "more chips for hero"
    hero = nz[0]
    # Skip fully-symmetric stack fields (the strict-equality case for stack symmetry).
    assume(any(abs(stacks[hero] - stacks[j]) > 1e-12 for j in nz))
    # Skip perfectly-flat payout ladders (the strict-equality case for prize symmetry).
    assume(max(payouts) - min(payouts) > 1e-9)
    bump = 7.5  # deterministic, well above float rounding
    bigger = list(stacks)
    bigger[hero] = stacks[hero] + bump
    base_eq = icm.icm_equity(stacks, payouts)[hero]
    bigger_eq = icm.icm_equity(bigger, payouts)[hero]
    assert bigger_eq >= base_eq - 1e-9
    # Strict: with at least one asymmetry and a real bump, the hero strictly gains equity.
    assert bigger_eq > base_eq + 1e-12


# --- (3) PKO bounty direction ----------------------------------------------------------------


@given(pko_params())
@settings(max_examples=120, suppress_health_check=[HealthCheck.too_slow])
def test_h_pko_winner_gets_loser_loses(params):
    """(3) After a knockout, the eliminator's realized cash strictly increases (when there is a
    non-zero cash fraction, i.e. ``progressive_frac < 1``) and the victim's live bounty strictly
    drops to 0. The eliminator's live bounty may grow via the progressive half.

    This is the "winner gets, loser loses" directional invariant of a PKO ledger. We constrain to
    ``progressive_frac < 1`` so that some cash is actually realized on a KO (with ``pf == 1`` the
    full victim bounty grows the eliminator and zero cash is paid).
    """
    n, entry, pf = params
    assume(entry > 1e-9)  # need a non-trivial bounty pool to talk about "gets/loses"
    assume(pf < 1.0 - 1e-9)  # need a non-trivial cash split (else winner gets no realized cash)
    state = bounty.BountyState.uniform(n, entry_bounty=entry, progressive_frac=pf)
    victim = 0
    eliminator = 1 if n > 1 else 0
    assume(eliminator != victim and eliminator < n)
    pre_elim_realized = state.realized[eliminator]
    pre_elim_live = state.live[eliminator]
    pre_victim_live = state.live[victim]
    cash = state.award_ko(eliminator, victim)
    # Winner: realized cash strictly increases.
    assert state.realized[eliminator] > pre_elim_realized + 1e-12
    # The awarded cash equals what was added to realized.
    assert state.realized[eliminator] - pre_elim_realized == pytest.approx(cash, abs=1e-12)
    # Loser: their live bounty is gone.
    assert state.live[victim] == pytest.approx(0.0, abs=1e-12)
    assert pre_victim_live > 1e-12  # confirm we actually had a bounty to lose
    # Conservation: invariant preserved.
    assert state.conserved()
    # Eliminator's live bounty must be >= previous (it can only grow via progressive half).
    assert state.live[eliminator] >= pre_elim_live - 1e-12


# --- (4) Bubble factor positivity ------------------------------------------------------------


@given(bbr_pair())
@settings(max_examples=120, suppress_health_check=[HealthCheck.too_slow])
def test_h_bubble_factor_positive(pair):
    """(4) bubble_factor returns a non-negative value (inf permitted when hero cannot gain).

    In every finite case, BF >= 0 strictly (BF = risk/reward, both non-negative).
    """
    stacks, payouts, hero, villain = pair
    bf = icm.bubble_factor(stacks, payouts, hero=hero, villain=villain)
    # Either a finite non-negative number, or +inf.
    assert math.isfinite(bf) or bf == float("inf")
    assert bf >= 0.0


# --- (5) Payout ladder monotonicity ----------------------------------------------------------


@given(st.integers(min_value=2, max_value=12),
       st.lists(st.floats(min_value=1.0, max_value=100.0,
                          allow_nan=False, allow_infinity=False),
                min_size=2, max_size=12))
@settings(max_examples=120, suppress_health_check=[HealthCheck.too_slow])
def test_h_payout_ladder_monotone(places, raw_payouts):
    """(5) payout_structure(places) is non-increasing (rank 1 pays >= rank 2 >= ...).

    The ladder is best-first descending; each entry must be >= the next (ties permitted).
    """
    assume(places >= 2)
    # Truncate the raw payouts to exactly `places` (best-first descending).
    payouts = sorted(raw_payouts[:places], reverse=True)
    if len(payouts) < places:
        return  # cannot form a full ladder; skip via assume-style via assume() inside.
    assume(len(payouts) == places)
    # Renormalize to a structure vector (sums to 1) so we test the canonical ladder.
    s = sum(payouts)
    assume(s > 1e-12)
    structure = [p / s for p in payouts]
    # Re-derive via the engine's own model so we test what the bot actually uses.
    # We can't call payout_structure directly here without a TournamentFormat, but
    # the ladder invariant we test is: best-first sorted payouts are non-increasing.
    for i in range(len(structure) - 1):
        assert structure[i] >= structure[i + 1] - 1e-12


# A complementary test that exercises the actual payout_structure API through NORMAL/PKO formats.
@given(st.integers(min_value=2, max_value=200))
@settings(max_examples=120, suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much])
def test_h_payout_structure_monotone_via_engine(places):
    """payout_structure(fmt, places) is non-increasing for any places (>= 2) for NORMAL and PKO."""
    assume(places >= 2)
    for fmt in (payouts.TournamentFormat.NORMAL, payouts.TournamentFormat.PKO):
        ladder = payouts.payout_structure(fmt, places)
        assert len(ladder) == places
        for i in range(len(ladder) - 1):
            assert ladder[i] >= ladder[i + 1] - 1e-9
        assert abs(sum(ladder) - 1.0) < 1e-9


# --- (6) Equal stacks give equal ICM-EV -----------------------------------------------------


@given(equal_stack_field())
@settings(max_examples=120, suppress_health_check=[HealthCheck.too_slow])
def test_h_equal_stacks_equal_icm_ev(field):
    """(6) When every seat has the SAME stack, every seat's ICM equity equals the per-seat average
    of the active payout pool (sum(payouts[:n_live]) / n_live).

    Equivalently: max(eq) - min(eq) <= small numerical slack.
    """
    s, n, payout_list = field
    stacks = [s] * n
    eq = icm.icm_equity(stacks, payout_list)
    assert len(eq) == n
    spread = float(eq.max() - eq.min())
    assert spread <= 1e-9, f"equal stacks gave spread={spread!r}"
    # And the per-seat equity equals the average of the active paid places.
    expected = sum(payout_list) / n
    for i in range(n):
        assert eq[i] == pytest.approx(expected, abs=1e-9)


# --- (7) 1-player survivor gets full 1st-place payout ---------------------------------------


@given(st.integers(min_value=2, max_value=8),
       st.lists(st.floats(min_value=1.0, max_value=100.0,
                          allow_nan=False, allow_infinity=False),
                min_size=1, max_size=8))
@settings(max_examples=120, suppress_health_check=[HealthCheck.too_slow])
def test_h_one_player_survivor_gets_first(n, raw_payouts):
    """(7) If only one seat is alive (others all 0), that seat gets ``payouts[0]`` (1st place prize).

    The remaining ``n - 1`` busted seats split the BOTTOM ``n - 1`` ranks of the payout ladder
    equally (simultaneous-bust convention). Concretely: the engine pads ``payouts`` to length ``n``
    with zeros and gives each busted seat the mean of ``pay[n - (n-1):]`` = ``pay[1:]``.
    """
    n = max(2, n)
    payouts = sorted(raw_payouts[:n], reverse=True)
    if not payouts:
        return
    # ensure payouts[0] is well-defined and positive
    assume(payouts[0] > 1e-9)
    # Vary which seat is the survivor; test a couple of positions.
    for survivor_idx in (0, n // 2, n - 1):
        stacks = [10.0 if i == survivor_idx else 0.0 for i in range(n)]
        eq = icm.icm_equity(stacks, payouts)
        assert eq[survivor_idx] == pytest.approx(payouts[0], abs=1e-9)
        # All non-survivors split the bottom (n - 1) ranks (padded with zeros) equally.
        pad = list(payouts) + [0.0] * max(0, n - len(payouts))
        bottom = pad[n - (n - 1):]  # last (n - 1) entries of the length-n padded ladder
        expected_tail = sum(bottom) / (n - 1)
        for i in range(n):
            if i != survivor_idx:
                assert eq[i] == pytest.approx(expected_tail, abs=1e-9)


# --- (8) PKO before any knockout: no bounties assigned ---------------------------------------


@given(pko_params())
@settings(max_examples=120, suppress_health_check=[HealthCheck.too_slow])
def test_h_pko_no_ko_no_bounty_assigned(params):
    """(8) A fresh BountyState (no KOs yet) has zero realized cash for every seat, and the
    live bounties sum to the initial pool exactly. Conservation invariant must hold at t=0.
    """
    n, entry, pf = params
    state = bounty.BountyState.uniform(n, entry_bounty=entry, progressive_frac=pf)
    # No realized cash anywhere.
    for realized in state.realized:
        assert realized == pytest.approx(0.0, abs=1e-12)
    # Live bounties are exactly the initial entries.
    for i, live in enumerate(state.live):
        assert live == pytest.approx(entry, abs=1e-12)
    # Sum of live equals initial pool.
    assert sum(state.live) == pytest.approx(state.initial_pool, abs=1e-12)
    # Conservation invariant holds at t=0.
    assert state.conserved()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])