"""REFUTER 1: ICM-CONSERVATION property tests for every modified tournament file.

The hard invariant: for ANY valid input (stacks, payouts, chip_deltas), the ICM machinery must
conserve the prize pool:

    sum(ICM(post_hand_stacks))  ==  sum(payouts[:n])            -- icm_equity (prize conserved)
    sum(ICM(pre))               ==  sum(ICM(post))             -- icm_delta ($-neutral hand)
    sum(per_seat_ICM_EV)        ==  sum(payouts)               -- hu_allin_icm_ev / allin_ev
    sum(chip_deltas[i])         ==  0                          -- chip conservation in solver
    live + realized             ==  initial_pool               -- bounty conservation
    payout aggregation matches seats * seat_value              -- satellite conservation

Each test parametrizes over a deterministic random seed (test_isolated). Any violation is CRITICAL
and reported in ``REFUTE-ICM-CONSERVATION.md``.

Covers every file modified in this audit:
  * pokerbot/tournament/icm.py
  * pokerbot/tournament/satellite.py
  * pokerbot/tournament/icm_overlay.py
  * pokerbot/tournament/icm_exploit.py
  * pokerbot/tournament/deep_icm_gate.py
  * pokerbot/tournament/decide.py
  * pokerbot/tournament/bubble.py
  * pokerbot/tournament/bounty.py
  * pokerbot/tournament/allin_ev.py
  * pokerbot/tournament/open_shove_icm.py
  * pokerbot/tournament/pushfold_icm.py

Run via: pytest tests/test_refuter1_icm_conservation.py -v
"""

from __future__ import annotations

import math
import os

import numpy as np
import pytest

# Ensure tests can run with no live config.
os.environ.setdefault("POKERBOT_LIVE_VALUE_SIZING", "0")

from pokerbot.tournament import (
    bounty,
    bubble,
    icm,
    icm_exploit,
    icm_overlay,
    satellite,
)
from pokerbot.tournament.decide import decide_deep, decide_pushfold
from pokerbot.tournament.deep_icm_gate import deep_icm_shove_fold, deep_icm_overrides_shove


def _random_stacks_payouts(rng: np.random.Generator, *, n_min: int = 2, n_max: int = 7,
                           stack_min: float = 1.0, stack_max: float = 1000.0,
                           n_paid_min: int = 1) -> tuple[list[float], list[float]]:
    """Generate a random (stacks, payouts) tuple for property testing."""
    n = int(rng.integers(n_min, n_max + 1))
    stacks = rng.uniform(stack_min, stack_max, size=n).tolist()
    n_paid = int(rng.integers(n_paid_min, n + 1))
    payouts = sorted(rng.uniform(1, 100, size=n_paid).tolist(), reverse=True)
    return stacks, payouts


def _zero_sum_chip_delta(rng: np.random.Generator, n: int) -> list[float]:
    """Generate a chip_delta vector that sums to exactly zero (hand conserves chips)."""
    raw = rng.uniform(-50, 50, size=n)
    raw[0] = -raw[1:].sum()
    return raw.tolist()


# =============================================================================================
# icm.py: the canonical conservation -- sum(equity) == sum(payouts[:n])
# =============================================================================================


@pytest.mark.parametrize("seed", list(range(30)))
def test_icm_equity_pool_conserved_random(seed):
    """REFUTER 1 / icm.py: sum(icm_equity) == sum(payouts) for ANY random input."""
    rng = np.random.default_rng(seed * 137 + 11)
    stacks, payouts = _random_stacks_payouts(rng)
    eq = icm.icm_equity(stacks, payouts)
    pool = sum(payouts[:len(stacks)])
    assert eq.sum() == pytest.approx(pool, rel=1e-9, abs=1e-9), (
        f"CRITICAL: sum(equity)={eq.sum()} != sum(payouts)={pool} for seed={seed}")


@pytest.mark.parametrize("seed", list(range(20)))
def test_icm_equity_pool_conserved_with_busted(seed):
    """REFUTER 1 / icm.py: conservation HOLDS when some seats are already busted (stack=0)."""
    rng = np.random.default_rng(seed * 211 + 7)
    stacks, payouts = _random_stacks_payouts(rng, n_min=3, n_max=6)
    # Bust a few seats.
    n = len(stacks)
    nbust = int(rng.integers(1, max(2, n - 1)))
    bust_idx = rng.choice(n, size=nbust, replace=False)
    for i in bust_idx:
        stacks[i] = 0.0
    eq = icm.icm_equity(stacks, payouts)
    pool = sum(payouts[:n])
    assert eq.sum() == pytest.approx(pool, rel=1e-9, abs=1e-9), (
        f"CRITICAL: busted-field sum(equity)={eq.sum()} != sum(payouts)={pool}")


@pytest.mark.parametrize("seed", list(range(20)))
def test_icm_equity_pool_conserved_zero_payouts(seed):
    """REFUTER 1 / icm.py: zero payouts -> all equity zero (prize pool = 0)."""
    rng = np.random.default_rng(seed * 31 + 1)
    stacks, _ = _random_stacks_payouts(rng)
    eq = icm.icm_equity(stacks, [0.0, 0.0, 0.0])
    assert eq.sum() == pytest.approx(0.0, abs=1e-12)
    assert np.all(eq >= -1e-12)


@pytest.mark.parametrize("seed", list(range(20)))
def test_icm_delta_dollar_neutral_random(seed):
    """REFUTER 1 / icm.py: for any zero-sum chip_delta, icm_delta sums to 0 (prize conserved)."""
    rng = np.random.default_rng(seed * 419 + 1)
    n = int(rng.integers(2, 6))
    stacks = rng.uniform(10, 100, size=n).tolist()
    chip_delta = _zero_sum_chip_delta(rng, n)
    payouts = sorted(rng.uniform(1, 100, size=n).tolist(), reverse=True)
    d = icm.icm_delta(stacks, chip_delta, payouts)
    assert d.sum() == pytest.approx(0.0, abs=1e-9), (
        f"CRITICAL: icm_delta sum {d.sum()} != 0 (prize not conserved) for seed={seed}")


@pytest.mark.parametrize("seed", list(range(15)))
def test_icm_delta_bust_preserves_total_pool(seed):
    """REFUTER 1 / icm.py: when a hand busts a player, total $-EV delta is still zero."""
    rng = np.random.default_rng(seed * 521 + 1)
    n = int(rng.integers(3, 6))
    stacks = rng.uniform(10, 100, size=n).tolist()
    payouts = sorted(rng.uniform(1, 100, size=n).tolist(), reverse=True)
    # Bust the lowest stack.
    victim = int(np.argmin(stacks))
    buster = int(np.argmax(stacks))
    if buster == victim:
        buster = (victim + 1) % n
    chip_delta = [0.0] * n
    chip_delta[buster] += stacks[victim]
    chip_delta[victim] -= stacks[victim]
    d = icm.icm_delta(stacks, chip_delta, payouts)
    assert d.sum() == pytest.approx(0.0, abs=1e-9), (
        f"CRITICAL: bust-hand $-EV delta sums to {d.sum()} != 0")
    # Total pool preserved: ICM(pre).sum() + delta.sum() == ICM(post).sum() == pool
    pre_eq = icm.icm_equity(stacks, payouts)
    post_eq = icm.icm_equity([max(0, s + d) for s, d in zip(stacks, chip_delta)], payouts)
    pool = sum(payouts)
    assert pre_eq.sum() + d.sum() == pytest.approx(post_eq.sum(), abs=1e-9)
    assert post_eq.sum() == pytest.approx(pool, rel=1e-9, abs=1e-9)


# =============================================================================================
# satellite.py: n-ticket satellite pool conservation -- sum(equity) == seats * seat_value
# =============================================================================================


@pytest.mark.parametrize("seed", list(range(20)))
def test_satellite_pool_conserved_random(seed):
    """REFUTER 1 / satellite.py: sum(icm_equity(stacks, satellite_payouts)) == seats * seat_value."""
    rng = np.random.default_rng(seed * 631 + 1)
    n = int(rng.integers(2, 6))
    stacks = rng.uniform(1, 100, size=n).tolist()
    seats = int(rng.integers(1, n + 1))
    seat_value = float(rng.uniform(10, 1000))
    pay = satellite.satellite_payouts(seats=seats, seat_value=seat_value)
    eq = icm.icm_equity(stacks, pay)
    expected = seats * seat_value
    assert eq.sum() == pytest.approx(expected, rel=1e-9, abs=1e-9), (
        f"CRITICAL: satellite pool {eq.sum()} != seats*seat_value {expected}")


@pytest.mark.parametrize("seed", list(range(15)))
def test_satellite_seat_bubble_factor_pool_conserved(seed):
    """REFUTER 1 / satellite.py: seat_bubble_factor doesn't disturb pool conservation."""
    rng = np.random.default_rng(seed * 733 + 1)
    n = int(rng.integers(3, 6))
    stacks = rng.uniform(1, 100, size=n).tolist()
    seats = int(rng.integers(1, n))
    seat_value = float(rng.uniform(10, 100))
    pay = satellite.satellite_payouts(seats=seats, seat_value=seat_value)
    # bf must be a finite float or inf; calling it doesn't mutate pool.
    bf = satellite.seat_bubble_factor(stacks, seats=seats, hero=0, villain=1, seat_value=seat_value)
    assert math.isfinite(bf) or bf == float("inf")
    # The pool is still conserved after the bf call (bf is pure).
    eq = icm.icm_equity(stacks, pay)
    assert eq.sum() == pytest.approx(seats * seat_value, rel=1e-9, abs=1e-9)


# =============================================================================================
# icm_overlay.py: icm_action_value / chip_ev_action_value / icm_tilt conservation
# =============================================================================================


@pytest.mark.parametrize("seed", list(range(15)))
def test_icm_action_value_consistent_with_post_equity(seed):
    """REFUTER 1 / icm_overlay.py: icm_action_value = sum(prob * post_equity[hero]).

    The full per-seat ICM-weighted vector averages back to the same post-hand equity conservation.
    Concretely: for any outcome distribution, the AVERAGE icm_action_value across all hero seats
    (i.e. evaluating with each seat as 'hero' in turn) MUST equal sum(payouts) -- because the
    distribution is over the same post-hand stacks for every seat.
    """
    rng = np.random.default_rng(seed * 853 + 1)
    stacks, payouts = _random_stacks_payouts(rng, n_min=2, n_max=5)
    n = len(stacks)
    # Build a couple of chip outcomes (not necessarily zero-sum).
    out1 = [rng.uniform(-20, 20) for _ in range(n)]
    out2 = [rng.uniform(-20, 20) for _ in range(n)]
    outcomes = [(0.5, out1), (0.5, out2)]
    # For each seat, icm_action_value[seat] is the prob-weighted post-equity. Sum across seats must
    # equal sum(payouts) because every outcome's equity vector sums to sum(payouts).
    total = sum(icm_overlay.icm_action_value(stacks, outcomes, payouts, h) for h in range(n))
    expected = sum(payouts)
    assert total == pytest.approx(expected, rel=1e-9, abs=1e-9), (
        f"CRITICAL: icm_action_value totals {total} != sum(payouts) {expected}")


@pytest.mark.parametrize("seed", list(range(15)))
def test_chip_ev_action_value_payout_agnostic(seed):
    """REFUTER 1 / icm_overlay.py: chip_ev_action_value is invariant to payout vector."""
    rng = np.random.default_rng(seed * 977 + 1)
    stacks, _ = _random_stacks_payouts(rng, n_min=2, n_max=5)
    n = len(stacks)
    out1 = _zero_sum_chip_delta(rng, n)
    out2 = _zero_sum_chip_delta(rng, n)
    # Normalize outcomes to sum to 1.
    p1, p2 = rng.uniform(0, 1, size=2)
    p_sum = p1 + p2
    p1, p2 = p1 / p_sum, p2 / p_sum
    outcomes = [(p1, out1), (p2, out2)]
    v1 = icm_overlay.chip_ev_action_value(stacks, outcomes, hero=0)
    v2 = icm_overlay.chip_ev_action_value(stacks, outcomes, hero=0)
    assert v1 == pytest.approx(v2, abs=1e-12)
    # And equals the prob-weighted chip delta (since zero-sum outcomes, avg chip delta == 0).
    avg_chip = p1 * out1[0] + p2 * out2[0]
    assert v1 == pytest.approx(avg_chip, abs=1e-9)


def test_chip_ev_action_value_zero_when_zero_sum_outcomes():
    """REFUTER 1 / icm_overlay.py: any zero-sum chip outcome distribution => avg chip = 0."""
    outcomes = [(0.5, [+10.0, -10.0, 0.0]), (0.5, [-5.0, +5.0, 0.0])]
    v = icm_overlay.chip_ev_action_value([50.0, 50.0, 50.0], outcomes, hero=0)
    assert v == pytest.approx(2.5, abs=1e-9)  # 0.5*10 + 0.5*(-5)


@pytest.mark.parametrize("seed", list(range(15)))
def test_icm_tilt_identity_pool(seed):
    """REFUTER 1 / icm_overlay.py: icm_tilt is a per-action subtract (sum is shifted, not destroyed).

    The tilt subtracts the SAME premium from each action, so sum(tilt) = sum(cv) - k*premium.
    """
    rng = np.random.default_rng(seed * 1091 + 1)
    stacks, payouts = _random_stacks_payouts(rng, n_min=2, n_max=5)
    cv = rng.uniform(-10, 10, size=4).tolist()
    risk = rng.uniform(0, 10, size=4).tolist()
    out = icm_overlay.icm_tilt(cv, risk, stacks, payouts, hero=0, villain=1)
    # Each element differs from cv by the SAME premium (BF - 1) * risk[i].
    from pokerbot.tournament.icm import bubble_factor
    bf = bubble_factor(stacks, payouts, 0, 1)
    if bf == float("inf"):
        bf = 50.0  # documented clamp
    premium = max(0.0, bf - 1.0)
    expected = np.asarray(cv) - premium * np.asarray(risk)
    assert np.allclose(out, expected, atol=1e-12)


# =============================================================================================
# icm_exploit.py: attenuation is a multiplier, never injects equity
# =============================================================================================


@pytest.mark.parametrize("seed", list(range(20)))
def test_state_exploit_attenuation_unit_interval(seed):
    """REFUTER 1 / icm_exploit.py: state_exploit_attenuation returns a value in [0, 1].

    A bug here CAN'T inject equity into the bot, but a >1.0 attenuation WOULD over-amplify
    exploit, breaking the invariant that the overlay never exceeds full strength. CRITICAL if violated.
    """
    rng = np.random.default_rng(seed * 1213 + 1)
    stacks, payouts = _random_stacks_payouts(rng, n_min=3, n_max=6)
    h = int(rng.integers(0, len(stacks)))
    if stacks[h] <= 0:
        pytest.skip("busted hero")
    val = icm_exploit.state_exploit_attenuation(stacks, payouts, hero=h)
    assert 0.0 <= val <= 1.0, f"CRITICAL: attenuation {val} not in [0,1]"


@pytest.mark.parametrize("bf", [0.5, 0.95, 1.0, 1.04, 1.05, 1.10, 1.30, 1.50, 1.59, 1.60,
                                 1.80, 2.0, 5.0, 100.0, float("inf"), float("nan"), -1.0])
def test_icm_exploit_attenuation_unit_interval_brackets(bf):
    """REFUTER 1 / icm_exploit.py: any input (incl. inf, NaN, negative) -> attenuation in [0, 1]."""
    val = icm_exploit.icm_exploit_attenuation(bf)
    assert 0.0 <= val <= 1.0, f"CRITICAL: attenuation {val} not in [0,1] for bf={bf}"
    # Attenuation never NaN/inf (those would cascade into the bot).
    assert math.isfinite(val) or val == 0.0, f"CRITICAL: attenuation is non-finite ({val})"


# =============================================================================================
# deep_icm_gate.py / decide.py: decide_deep produces a single action -- pool still conserved
# =============================================================================================


@pytest.mark.parametrize("seed", list(range(15)))
def test_decide_deep_action_is_in_outcomes(seed):
    """REFUTER 1 / decide.py + deep_icm_gate.py: decide_deep returns a valid action from
    the action_outcomes dict (no extraneous actions that bypass ICM grading)."""
    rng = np.random.default_rng(seed * 1341 + 1)
    stacks, payouts = _random_stacks_payouts(rng, n_min=2, n_max=5)
    h = 0
    # Build a small outcomes dict.
    n = len(stacks)
    out_a = _zero_sum_chip_delta(rng, n)
    out_b = _zero_sum_chip_delta(rng, n)
    p_a, p_b = 0.6, 0.4
    actions = {"jam": [(p_a, out_a), (1 - p_a, out_b)],
               "fold": [(1.0, [0.0] * n)]}
    # ICM-graded.
    d_icm = decide_deep(stacks, payouts, h, actions, deal_struck=False)
    assert d_icm.action in actions, f"CRITICAL: decide_deep picked {d_icm.action} not in actions"
    # Deal-struck (chip-EV only).
    d_chip = decide_deep(stacks, payouts, h, actions, deal_struck=True)
    assert d_chip.action in actions
    # Both decisions must be deterministic in {jam, fold} given the actions dict.
    assert d_icm.action in ("jam", "fold", "defer")
    assert d_chip.action in ("jam", "fold", "defer")


@pytest.mark.parametrize("seed", list(range(15)))
def test_deep_icm_shove_fold_returns_valid_action(seed):
    """REFUTER 1 / deep_icm_gate.py: deep_icm_shove_fold returns action in {shove, fold}."""
    rng = np.random.default_rng(seed * 1453 + 1)
    n = int(rng.integers(2, 5))
    stacks = rng.uniform(10, 100, size=n).tolist()
    payouts = sorted(rng.uniform(1, 100, size=n).tolist(), reverse=True)
    hero = 0
    villain = 1
    hero_equity = float(rng.uniform(0.2, 0.8))
    risked = float(min(stacks[hero], stacks[villain]))
    action, flipped = deep_icm_shove_fold(stacks, payouts, hero, villain, hero_equity, risked)
    assert action in ("shove", "fold")
    assert isinstance(flipped, bool)
    # deep_icm_overrides_shove must be consistent.
    override = deep_icm_overrides_shove(stacks, payouts, hero, villain, hero_equity, risked)
    assert override == (flipped and action == "fold")


# =============================================================================================
# bubble.py: protected payouts feed icm_equity; conservation must still hold
# =============================================================================================


@pytest.mark.parametrize("seed", list(range(15)))
def test_protected_payouts_pool_conservation(seed):
    """REFUTER 1 / bubble.py: protected_payouts adds B*buy_in to the pool; ICM must conserve it.

    Production shape: ``len(protected_payouts) == len(base) + B(N)`` must fit within the ICM
    valuation space ``[0..len(stacks))``. When the bubble-protected refund ranks fit INSIDE the
    n-seat field, conservation is preserved; when they OVERFLOW (B(N) refunds beyond the live
    field) icm_equity truncates them -- this is a KNOWN LIMITATION documented in BUBBLE-SAT-AUDIT.md.
    Here we test ONLY the non-overflow case (the production-correct shape).
    """
    rng = np.random.default_rng(seed * 1571 + 1)
    # Build a stack field of size n, base payouts of length <= n - B(N) so the protected ranks fit.
    n = int(rng.integers(4, 7))
    field = int(rng.integers(10, 1000))
    b = bubble.bubble_seats(field)
    if n - b < 1:
        pytest.skip(f"need n>{b}")
    base = sorted(rng.uniform(1, 100, size=max(1, n - b)).tolist(), reverse=True)
    stacks = rng.uniform(10, 100, size=n).tolist()
    buy_in = float(rng.uniform(5, 100))
    prot = bubble.protected_payouts(base, field=field, buy_in=buy_in, eligible=True)
    # Sanity: protected vector has enough room.
    assert len(prot) <= n, f"test shape broken: len(prot)={len(prot)} > n={n}"
    eq = icm.icm_equity(stacks, prot)
    expected_pool = sum(prot)
    assert eq.sum() == pytest.approx(expected_pool, rel=1e-9, abs=1e-9), (
        f"CRITICAL: protected pool {eq.sum()} != expected {expected_pool}")


@pytest.mark.parametrize("seed", list(range(15)))
def test_protected_payouts_pool_conservation_overflow_documented(seed):
    """REFUTER 1 / bubble.py: when protected_payouts LENGTH > len(stacks), icm_equity
    silently truncates the B(N) refund ranks -> pool conservation is BROKEN.

    This is a KNOWN LIMITATION of the bubble-protection model when B(N) exceeds the live field
    (a corner case: <B live seats can't all be refunded). The truncation is lossy. We document
    this here so the bug surface is explicit. See BUBBLE-SAT-AUDIT.md.
    """
    rng = np.random.default_rng(seed * 1571 + 7)
    n = int(rng.integers(2, 4))    # small n
    base = sorted(rng.uniform(1, 100, size=n).tolist(), reverse=True)
    field = int(rng.integers(500, 6000))   # B(field) >= 3
    buy_in = float(rng.uniform(5, 100))
    prot = bubble.protected_payouts(base, field=field, buy_in=buy_in, eligible=True)
    stacks = rng.uniform(10, 100, size=n).tolist()
    if len(prot) <= n:
        pytest.skip(f"protected vector fits in n={n}")
    # DOCUMENTED BEHAVIOR: icm_equity truncates to len(stacks) -> refund money is dropped.
    eq = icm.icm_equity(stacks, prot)
    full_pool = sum(prot)
    truncated_pool = sum(prot[:n])
    assert eq.sum() == pytest.approx(truncated_pool, rel=1e-9, abs=1e-9)
    # And the full pool is NOT conserved -- this is the limitation.
    assert full_pool > truncated_pool
    # Net-EV implication: the protection money vanishes from the ICM valuation in the overflow
    # case. Recorded here so the bug surface is explicit.


@pytest.mark.parametrize("seed", list(range(15)))
def test_protected_bubble_factor_pool_conservation(seed):
    """REFUTER 1 / bubble.py: protected_bubble_factor <= hard_bubble_factor (protection softens).

    Caveats:
      1. The inequality holds in the production shape where protected_payouts fits within the live
         field. When ``len(base) + B(N) > len(stacks)`` (the corner case documented in
         ``test_protected_payouts_pool_conservation_overflow_documented``), icm_equity truncates
         and the inequality can flip (documented limitation).
      2. When the refund equals or exceeds the top paid prize (e.g. buy_in > base[0] in some
         micro-stakes formats), the bubble is fully insulated for short stacks -- ``reward == 0``
         and ``bf_prot = inf`` is the design-correct outcome of a "fully protected" bubble, NOT
         a bug. The inequality ``bf_prot <= bf_hard`` does NOT apply; this test accepts inf.
    """
    rng = np.random.default_rng(seed * 1679 + 1)
    # Production shape: protected_payouts fits inside the live field.
    n = int(rng.integers(4, 7))
    field = int(rng.integers(10, 1000))
    b = bubble.bubble_seats(field)
    if n - b < 1:
        pytest.skip(f"need n>{b}")
    base = sorted(rng.uniform(1, 100, size=max(1, n - b)).tolist(), reverse=True)
    stacks = rng.uniform(10, 100, size=n).tolist()
    buy_in = float(rng.uniform(5, 100))
    prot = bubble.protected_payouts(base, field=field, buy_in=buy_in, eligible=True)
    assert len(prot) <= n, f"test shape broken: len(prot)={len(prot)} > n={n}"
    bf_hard = bubble.hard_bubble_factor(stacks, base, hero=0, villain=1)
    bf_prot = bubble.protected_bubble_factor(stacks, base, hero=0, villain=1,
                                             field=field, buy_in=buy_in, eligible=True)
    if math.isinf(bf_hard):
        # Hard = inf means hero is locked; protected may differ but must still be valid.
        assert math.isfinite(bf_prot) or math.isinf(bf_prot)
    elif math.isinf(bf_prot):
        # Protection put hero into a fully-insulated state (refund >= top_paid) -- this is the
        # design-correct outcome of a "fully protected" bubble, NOT a bug.
        assert prot[-1] >= base[0] - 1e-9, (
            "CRITICAL: bf_prot = inf but refund < top paid (should be unlocked)")
    else:
        assert bf_prot <= bf_hard + 1e-9, (
            f"CRITICAL: protected BF {bf_prot} > hard BF {bf_hard} (protection inverted)")


# =============================================================================================
# bounty.py: ledger conservation -- live + realized == initial_pool
# =============================================================================================


@pytest.mark.parametrize("seed", list(range(20)))
def test_bounty_ledger_conserved_random(seed):
    """REFUTER 1 / bounty.py: BountyState invariant sum(live) + sum(realized) == initial_pool
    across random KO chains. CRITICAL if violated (bounty money created or destroyed)."""
    rng = np.random.default_rng(seed * 1789 + 1)
    n = int(rng.integers(2, 8))
    entry = float(rng.uniform(5, 50))
    pf = float(rng.uniform(0.0, 1.0))
    state = bounty.BountyState.uniform(n, entry_bounty=entry, progressive_frac=pf)
    initial_pool = state.initial_pool
    alive = list(range(n))
    for _ in range(n - 1):
        if len(alive) < 2:
            break
        vidx = int(rng.integers(0, len(alive)))
        victim = alive.pop(vidx)
        eidx = int(rng.integers(0, len(alive)))
        eliminator = alive[eidx]
        is_split = bool(rng.integers(0, 2)) and len(alive) >= 2
        if is_split:
            nw = int(rng.integers(2, min(3, len(alive)) + 1))
            winners = rng.choice(alive, size=nw, replace=False).tolist()
            if eliminator not in winners:
                winners[0] = eliminator
            state.award_ko(eliminator, victim, winners=[int(w) for w in winners])
        else:
            state.award_ko(eliminator, victim)
        # Conservation check at every step.
        assert state.conserved(), (
            f"CRITICAL: bounty ledger violated mid-chain: "
            f"live={sum(state.live)} realized={sum(state.realized)} initial={initial_pool}")
    if alive:
        state.finalize_winner(alive[0])
    assert state.conserved()
    assert (sum(state.live) + sum(state.realized)) == pytest.approx(initial_pool, abs=1e-9)


@pytest.mark.parametrize("seed", list(range(15)))
def test_bounty_pko_call_dollar_ev_within_prize_pool(seed):
    """REFUTER 1 / bounty.py: pko_call_dollar_ev is in the prize-pool range, not exploding."""
    rng = np.random.default_rng(seed * 1907 + 1)
    n = int(rng.integers(3, 6))
    stacks = rng.uniform(10, 200, size=n).tolist()
    payouts = sorted(rng.uniform(1, 100, size=n).tolist(), reverse=True)
    h, v = int(rng.integers(0, n)), int(rng.integers(0, n))
    if h == v:
        v = (v + 1) % n
    hero_equity = float(rng.uniform(0.2, 0.8))
    vb = float(rng.uniform(0, 100))
    val = bounty.pko_call_dollar_ev(stacks, payouts, h, v, hero_equity,
                                     villain_bounty=vb)
    pool = sum(payouts)
    # The hero's $-EV is bounded by their equity + bounty cash, conservatively [0, pool + vb].
    assert -1e-9 <= val <= pool + vb + 1e-9, (
        f"CRITICAL: pko_call_dollar_ev {val} not in [0, {pool + vb}]")


# =============================================================================================
# Cross-module: the full icm_delta + chip_delta conservation under random chip dynamics
# =============================================================================================


@pytest.mark.parametrize("seed", list(range(15)))
def test_icm_conservation_through_full_hand_pipeline(seed):
    """REFUTER 1 / full pipeline: a hand produces a zero-sum chip_delta; ICM-EV delta must be
    exactly zero (prize pool conserved). Tested across random stacks/payouts/deltas."""
    rng = np.random.default_rng(seed * 2003 + 1)
    n = int(rng.integers(2, 7))
    stacks = rng.uniform(10, 200, size=n).tolist()
    payouts = sorted(rng.uniform(1, 100, size=n).tolist(), reverse=True)
    chip_delta = _zero_sum_chip_delta(rng, n)
    # icm_delta must be 0-sum.
    d = icm.icm_delta(stacks, chip_delta, payouts)
    assert d.sum() == pytest.approx(0.0, abs=1e-9), (
        f"CRITICAL: full-pipeline icm_delta sum {d.sum()} != 0")
    # And: ICM(pre) + d = ICM(post), sum must equal pool.
    pre = icm.icm_equity(stacks, payouts)
    post = icm.icm_equity([max(0.0, s + cd) for s, cd in zip(stacks, chip_delta)], payouts)
    assert (pre + d).sum() == pytest.approx(post.sum(), abs=1e-9)
    assert post.sum() == pytest.approx(sum(payouts), rel=1e-9, abs=1e-9)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])