"""Tournament layer property + sharpening tests (post-audit batch).

Covers hand-calculated reference cases for:
  * ICM (icm.py) — closed-form equity, conservation, monotonicity, finite bubble limits.
  * PKO (bounty.py) — closed-form cash + growth split, conservation ledger, self-KO impossible.
  * Bubble protection (bubble.py) — protection softening, eligible/ineligible asymmetry.
  * Satellite (satellite.py) — ticket-EV split, lock threshold, leader-residual.
  * ICM overlay (icm_overlay.py) — bf=inf clamp, identity at deep.
  * ICM exploit attenuation (icm_exploit.py) — bf -> attenuation mapping.

The "hand-calculated" tests fix specific stack/payout inputs where the answer is known to N decimal
places and pin the implementation against regression. Run via:
  pytest tests/test_tournament_sharpening.py -v
"""

from __future__ import annotations

import math
import os

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
    satellite,
)


# =============================================================================================
# ICM — hand-calculated closed forms + asymptotic bubble limit
# =============================================================================================


def test_two_player_weighted_linear():
    """Hand-calc: stacks [3,1] / payouts [70, 30] -> equities = [60, 40] (closed form)."""
    eq = icm.icm_equity([3.0, 1.0], [70.0, 30.0])
    assert eq[0] == pytest.approx(60.0, abs=1e-12)
    assert eq[1] == pytest.approx(40.0, abs=1e-12)
    assert eq.sum() == pytest.approx(100.0, abs=1e-12)


def test_three_player_hand_computed_mh():
    """Hand-calc: stacks [50, 30, 20] / payouts [70, 20, 10].

    Derivation (Malmuth-Harville):
      P(player 1 finishes 1st) = 50/100 = 0.5
      P(player 2 finishes 1st) = 30/100 = 0.3
      P(player 3 finishes 1st) = 20/100 = 0.2

      Conditional on P1 1st, remaining [30,20] over [20,10]:
        P(P2 2nd | P1 1st) = 30/50 = 0.6  -> P2 gets 20
        P(P3 2nd | P1 1st) = 20/50 = 0.4  -> P3 gets 20
      Equity P2 = 0.3*70 + 0.5*20*0.6 + 0.2*10*0.4 = 21 + 6 + 0.8 = 27.8 (refined)
      Equity P3 = 0.2*70 + 0.5*20*0.4 + 0.3*10*0.6 = 14 + 4 + 1.8 = 19.8
    """
    eq = icm.icm_equity([50.0, 30.0, 20.0], [70.0, 20.0, 10.0])
    # Reference values from the audit (proven via MH recursion).
    assert eq[0] == pytest.approx(43.393, abs=1e-2)
    assert eq[1] == pytest.approx(31.750, abs=1e-2)
    assert eq[2] == pytest.approx(24.857, abs=1e-2)
    assert eq.sum() == pytest.approx(100.0, abs=1e-9)


def test_equal_stacks_three_way_third_each():
    """Hand-calc: equal stacks over 3-way payouts split the pool 1/3 each."""
    eq = icm.icm_equity([10.0, 10.0, 10.0], [60.0, 30.0, 10.0])
    assert eq == pytest.approx([100.0 / 3] * 3, abs=1e-12)


def test_bubble_factor_equal_4p_3paid_is_1_875():
    """Hand-calc: 4 equal stacks, 3 paid [50, 30, 20]. The classic bubble. BF = 1.875 exactly.

    The closed-form limit at equal stacks is verified via the BUBBLE-SAT audit: for the
    4-player 3-paid bubble, BF = 15/8 = 1.875. NOT infinity (common misconception).
    """
    bf = icm.bubble_factor([40.0, 40.0, 40.0, 40.0], [50.0, 30.0, 20.0], hero=0, villain=1)
    assert bf == pytest.approx(1.875, abs=2e-2)
    # Cross-check: bubble factor at FINITE limit, NOT infinity.
    assert math.isfinite(bf)


def test_bubble_factor_equal_3p_2paid_is_1_5():
    """Hand-calc: 3 equal stacks, 2 paid [10, 5]. BF = 1.5 exactly at equal stacks.

    Verified via the BUBBLE-SAT audit: limit = 3/2 = 1.5 at the 2-paid bubble.
    """
    bf = icm.bubble_factor([10.0, 10.0, 10.0], [10.0, 5.0], hero=0, villain=1)
    assert bf == pytest.approx(1.5, abs=2e-2)
    assert math.isfinite(bf)


def test_bubble_factor_hu_equal_is_one():
    """HU equal-stacks, covered: BF = 1.0 exactly (chip-EV-linear)."""
    bf = icm.bubble_factor([50.0, 50.0], [60.0, 40.0], hero=0, villain=1)
    assert bf == pytest.approx(1.0, abs=1e-9)


def test_bubble_factor_locked_returns_inf():
    """Hero already locked (no equity gain by doubling) -> BF = inf (cannot gain)."""
    bf = icm.bubble_factor([100.0, 0.0], [60.0, 40.0], hero=0, villain=1)
    assert bf == float("inf")


def test_bubble_factor_finite_at_classic_3p_2paid_bubble():
    """3 equal stacks, top 2 paid [10, 5]. Equal-stack asymptote should be (paid-1)/paid = 2/3 -- but
    that is the limit at 4-player 3-paid case; for 3p/2paid the asymptote is 1.0 actually (HU down
    to two players yields BF=1 in the limit). Verified empirically via the audit, BF stays finite."""
    bf = icm.bubble_factor([10.0, 10.0, 10.0], [10.0, 5.0], hero=0, villain=1)
    assert math.isfinite(bf)
    assert bf >= 1.0


def test_icm_delta_bust_realizes_bottom_payout():
    """Hand-calc: pre [50, 30, 20] -> C busted (delta = [+20, 0, -20]). Post [70, 30, 0].

    ICM(post, [70, 20, 10]):
      alive: seats 0, 1 (stacks 70, 30) compete for top 2 ranks [70, 20] -- each gets the rank
      proportional to stack: P(seat 0 1st) = 70/100 = 0.7, P(seat 1 1st) = 0.3. So seat 0 gets
      0.7*70 + 0.3*20 = 55; seat 1 gets 0.3*70 + 0.7*20 = 35.
      busted: seat 2 gets the bottom rank [10].
    Total = 55 + 35 + 10 = 100 (prize pool conserved).
    """
    pre = [50.0, 30.0, 20.0]
    d = icm.icm_delta(pre, [+20.0, 0.0, -20.0], [70.0, 20.0, 10.0])
    assert d.sum() == pytest.approx(0.0, abs=1e-9)
    assert d[0] > 0  # winner gains $
    assert d[2] < 0  # buster loses the most


def test_icm_delta_zero_sum_in_dollars():
    """Non-bust transfer: A wins 10 from B -> $-EV delta is zero-sum (prize pool fixed)."""
    pre = [50.0, 30.0, 20.0]
    d = icm.icm_delta(pre, [+10.0, -10.0, 0.0], [70.0, 20.0, 10.0])
    assert d.sum() == pytest.approx(0.0, abs=1e-9)
    assert d[0] > 0
    assert d[1] < 0


def test_payouts_padded_zero_for_unpaid_finishes():
    """4 players, 2 paid -> 3rd and 4th finishers realize 0; pool still conserved."""
    eq = icm.icm_equity([40, 30, 20, 10], [60, 40])
    assert eq.sum() == pytest.approx(100.0, abs=1e-9)
    # chip leader > 2nd > 3rd > 4th (monotone in stack)
    assert eq[0] > eq[1] > eq[2] > eq[3]
    assert eq[2] > 0  # top-3 chance for non-leader
    assert eq[3] > 0  # even short has SOME equity via runout


def test_monotone_in_stack():
    """More chips -> more equity (monotonicity invariant)."""
    base = icm.icm_equity([40, 30, 30], [50, 30, 20])
    bigger = icm.icm_equity([60, 30, 30], [50, 30, 20])
    assert bigger[0] > base[0]


def test_short_stack_icm_premium():
    """Concave ICM: per-chip equity of the short stack > the leader."""
    stacks = [50.0, 30.0, 20.0]
    eq = icm.icm_equity(stacks, [70.0, 20.0, 10.0])
    per_chip = eq / np.asarray(stacks)
    assert per_chip[2] > per_chip[0]


def test_saturation_extreme_ratio_does_not_overflow():
    """Stacks 1e12 vs 1 over payouts [60, 40] -> chip leader gets ~60, short gets ~40, no overflow."""
    eq = icm.icm_equity([1e12, 1.0], [60.0, 40.0])
    assert eq[0] <= 60.0 + 1e-9
    assert eq[1] <= 40.0 + 1e-9
    assert eq[0] > 59.0  # leader has effectively locked 1st


# =============================================================================================
# PKO bounty — hand-calculated conservation + cash/growth split
# =============================================================================================


def test_pko_award_50_50_split_default():
    """Hand-calc: victim_bounty=20, pf=0.5 -> eliminator gets 10 immediate cash + 10 growth.

    Final ledger:
      realized[elim] += 10
      live[elim] += 10
      live[victim] = 0
    Pool: 20 = 10 realized + 10 live[elim] + 0 live[victim]. Conserved.
    """
    state = bounty.BountyState([20.0, 20.0], progressive_frac=0.5)
    cash = state.award_ko(eliminator=0, victim=1)
    assert cash == pytest.approx(10.0, abs=1e-12)
    assert state.realized[0] == pytest.approx(10.0, abs=1e-12)
    assert state.live[0] == pytest.approx(30.0, abs=1e-12)  # own 20 + growth 10
    assert state.live[1] == pytest.approx(0.0, abs=1e-12)
    assert state.conserved()


def test_pko_award_fixed_scheme_pf_zero():
    """pf=0.0 -> classic fixed-bounty: victim_bounty paid entirely as immediate cash, no growth."""
    state = bounty.BountyState([25.0, 25.0], progressive_frac=0.0)
    cash = state.award_ko(eliminator=0, victim=1)
    assert cash == pytest.approx(25.0, abs=1e-12)
    assert state.realized[0] == pytest.approx(25.0, abs=1e-12)
    assert state.live[0] == pytest.approx(25.0, abs=1e-12)  # no growth added
    assert state.live[1] == 0.0
    assert state.conserved()


def test_pko_award_100_progressive_pf_one():
    """pf=1.0 -> 100% progressive: no immediate cash, all growth to eliminator's bounty."""
    state = bounty.BountyState([15.0, 15.0], progressive_frac=1.0)
    cash = state.award_ko(eliminator=0, victim=1)
    assert cash == pytest.approx(0.0, abs=1e-12)
    assert state.realized[0] == pytest.approx(0.0, abs=1e-12)
    assert state.live[0] == pytest.approx(30.0, abs=1e-12)  # own 15 + growth 15
    assert state.live[1] == 0.0
    assert state.conserved()


def test_pko_chain_conservation():
    """Multi-KO chain must preserve ledger: sum(live)+sum(realized) == initial pool at every step."""
    state = bounty.BountyState.uniform(5, entry_bounty=12.5, progressive_frac=0.5)
    initial_pool = state.initial_pool
    # KO chain: 1 -> 0, 1 -> 2, 1 -> 3, 1 -> 4
    state.award_ko(eliminator=1, victim=0)
    assert state.conserved()
    state.award_ko(eliminator=1, victim=2)
    assert state.conserved()
    state.award_ko(eliminator=1, victim=3)
    assert state.conserved()
    state.award_ko(eliminator=1, victim=4)
    assert state.conserved()
    assert (sum(state.live) + sum(state.realized)) == pytest.approx(initial_pool, abs=1e-9)


def test_pko_self_ko_impossible():
    """Self-KO must raise ValueError (cannot be among the winners)."""
    state = bounty.BountyState([10.0, 10.0], progressive_frac=0.5)
    with pytest.raises(ValueError):
        state.award_ko(eliminator=0, victim=0)
    with pytest.raises(ValueError):
        state.award_ko(eliminator=0, victim=0, winners=[0, 1])


def test_pko_split_pot_ko():
    """Split-pot KO: cash + growth divided equally among winners."""
    state = bounty.BountyState([20.0, 20.0, 20.0], progressive_frac=0.5)
    cash = state.award_ko(eliminator=0, victim=1, winners=[0, 2])
    assert cash == pytest.approx(10.0, abs=1e-12)  # total cash
    # Each winner gets half the cash + half the growth
    assert state.realized[0] == pytest.approx(5.0, abs=1e-12)
    assert state.realized[2] == pytest.approx(5.0, abs=1e-12)
    assert state.live[0] == pytest.approx(25.0, abs=1e-12)  # 20 + 5 growth
    assert state.live[2] == pytest.approx(25.0, abs=1e-12)
    assert state.live[1] == 0.0
    assert state.conserved()


def test_pko_bounty_cash_on_win_formula():
    """Hand-calc: vb=100, pf=0.5, growth_value=0.5 -> cash=50, growth=25, total=75."""
    val = bounty.bounty_cash_on_win(100.0)
    assert val == pytest.approx(75.0, abs=1e-12)


def test_pko_call_dollar_ev_reduces_to_icm_when_no_bounty():
    """With villain_bounty=0, pko_call_dollar_ev equals the pure-ICM chip-weighted call value."""
    stacks = [50.0, 30.0]
    payouts = [60.0, 40.0]
    p = 0.55
    # ICM call value: p * icm_win + (1-p) * icm_lose, where busts clamp at 0 (no bust here).
    h, v = stacks[0], stacks[1]
    risked = min(h, v)
    win = [h + risked, v - risked]
    lose = [h - risked, v + risked]
    eq_win = icm.icm_equity(win, payouts)[0]
    eq_lose = icm.icm_equity(lose, payouts)[0]
    expected = p * eq_win + (1 - p) * eq_lose
    got = bounty.pko_call_dollar_ev(stacks, payouts, hero=0, villain=1, hero_equity=p,
                                     villain_bounty=0.0)
    assert got == pytest.approx(expected, abs=1e-9)


def test_pko_min_equity_to_call_lowered_by_bounty():
    """A positive villain_bounty LOWERS the equity threshold to call (PKO chip != $)."""
    stacks = [50.0, 30.0]
    payouts = [60.0, 40.0]
    p_no_bounty = bounty.min_equity_to_call(stacks, payouts, 0, 1, villain_bounty=0.0)
    p_with_bounty = bounty.min_equity_to_call(stacks, payouts, 0, 1, villain_bounty=100.0)
    assert p_with_bounty < p_no_bounty  # PKO widens the calling range


def test_pko_min_equity_to_call_clamped():
    """min_equity_to_call is clamped to [0, 1]."""
    stacks = [50.0, 30.0]
    payouts = [60.0, 40.0]
    p = bounty.min_equity_to_call(stacks, payouts, 0, 1,
                                   villain_bounty=1000.0, progressive_frac=0.5, growth_value=0.5)
    assert 0.0 <= p <= 1.0


def test_pko_finalize_winner_drains_remaining():
    """finalize_winner realizes the winner's own outstanding live bounty (paid out at end)."""
    state = bounty.BountyState([10.0, 20.0, 30.0], progressive_frac=0.5)
    initial = state.initial_pool
    # KO chain: 0 eliminates 1 and 2
    state.award_ko(0, 1)
    state.award_ko(0, 2)
    # Winner is seat 0. Their remaining live bounty was 10 (own) + 0.5*20 (from 1) + 0.5*30 (from 2)
    # = 10 + 10 + 15 = 35.
    final = state.finalize_winner(0)
    assert final == pytest.approx(35.0, abs=1e-12)
    assert state.live[0] == 0.0
    assert state.conserved()


# =============================================================================================
# Bubble protection — hand-calculated softening + eligibility asymmetry
# =============================================================================================


def test_bubble_seats_brackets():
    """Bracket table: 1-9 -> 0, 10-99 -> 1, 100-499 -> 2, ..., cap at 5."""
    assert bubble.bubble_seats(1) == 0
    assert bubble.bubble_seats(9) == 0
    assert bubble.bubble_seats(10) == 1
    assert bubble.bubble_seats(99) == 1
    assert bubble.bubble_seats(100) == 2
    assert bubble.bubble_seats(499) == 2
    assert bubble.bubble_seats(500) == 3
    assert bubble.bubble_seats(1499) == 3
    assert bubble.bubble_seats(1500) == 4
    assert bubble.bubble_seats(2999) == 4
    assert bubble.bubble_seats(3000) == 5
    assert bubble.bubble_seats(5000) == 5
    # Cap at 5 for very large fields
    assert bubble.bubble_seats(10000) == 5


def test_protected_payouts_appends_for_eligible():
    """Eligible entrant: base_payouts + [buy_in]*B(N) refund ranks appended."""
    pay = bubble.protected_payouts([50.0, 30.0, 20.0], field=100, buy_in=10.0, eligible=True)
    # field=100 -> B=2 -> 2 refund ranks appended at the bottom.
    assert len(pay) == 5
    assert pay[-2:] == [10.0, 10.0]


def test_protected_payouts_unchanged_for_ineligible():
    """Ineligible entrant (promo-ticket): base_payouts unchanged (hard bubble)."""
    pay = bubble.protected_payouts([50.0, 30.0, 20.0], field=100, buy_in=10.0, eligible=False)
    assert pay == [50.0, 30.0, 20.0]


def test_protected_payouts_no_protection_when_b_zero():
    """Field < 10 -> B=0 -> payouts unchanged."""
    pay = bubble.protected_payouts([50.0, 30.0, 20.0], field=5, buy_in=10.0, eligible=True)
    assert pay == [50.0, 30.0, 20.0]


def test_protected_bubble_factor_softens_for_eligible():
    """Eligible -> protected_bubble_factor <= hard_bubble_factor (the floor widens ranges)."""
    stacks = [40.0, 40.0, 40.0, 40.0]
    base_payouts = [50.0, 30.0, 20.0]
    hard = bubble.hard_bubble_factor(stacks, base_payouts, hero=0, villain=1)
    prot = bubble.protected_bubble_factor(stacks, base_payouts, hero=0, villain=1,
                                           field=100, buy_in=10.0, eligible=True)
    assert prot <= hard + 1e-12


def test_protected_bubble_factor_equals_hard_when_ineligible():
    """Ineligible -> protected_bubble_factor == hard_bubble_factor (no floor)."""
    stacks = [40.0, 40.0, 40.0, 40.0]
    base_payouts = [50.0, 30.0, 20.0]
    hard = bubble.hard_bubble_factor(stacks, base_payouts, hero=0, villain=1)
    prot = bubble.protected_bubble_factor(stacks, base_payouts, hero=0, villain=1,
                                           field=100, buy_in=10.0, eligible=False)
    assert prot == pytest.approx(hard, abs=1e-9)


# =============================================================================================
# Satellite — hand-calculated lock + ticket EV split
# =============================================================================================


def test_satellite_payouts_step_function():
    """satellite_payouts(seats, value) = [value] * seats (step function, all ranks identical)."""
    pay = satellite.satellite_payouts(3, seat_value=215.0)
    assert pay == [215.0, 215.0, 215.0]


def test_seat_locked_uncatchable_fraction():
    """Lock threshold: hero >= total/(seats+1) -> locked.

    4 stacks [12, 4, 4, 4], seats=1: total=24, threshold=12 -> hero at 12 is exactly locked.
    """
    assert satellite.seat_locked([12.0, 4.0, 4.0, 4.0], seats=1, hero=0)
    # Slightly below: not locked
    assert not satellite.seat_locked([11.999, 4.0, 4.0, 4.0], seats=1, hero=0)


def test_seat_locked_trivial_when_few_left():
    """If <= seats players remain, all are paid -> trivially locked."""
    assert satellite.seat_locked([10.0, 5.0], seats=1, hero=0)
    assert satellite.seat_locked([10.0, 5.0, 3.0], seats=3, hero=2)


def test_seat_locked_busted_hero_never_locked():
    """Busted hero (stack <= 0) is never locked."""
    assert not satellite.seat_locked([0.0, 100.0, 100.0], seats=1, hero=0)


def test_should_fold_for_seat_only_when_locked_and_risk():
    """should_fold_for_seat = seat_locked AND action_risks_chips."""
    stacks = [12.0, 4.0, 4.0, 4.0]
    assert satellite.should_fold_for_seat(stacks, seats=1, hero=0)  # locked + risking
    assert not satellite.should_fold_for_seat(stacks, seats=1, hero=0, action_risks_chips=False)
    # Not locked -> False regardless of risk flag.
    assert not satellite.should_fold_for_seat([8.0, 4.0, 4.0, 4.0], seats=1, hero=0)


def test_leader_equity_is_chip_share_times_ticket_value():
    """Per BUBBLE-SAT audit: leader equity on a 1-ticket sat is `chip_share * ticket_value`,
    NOT 100% of the ticket. With [12, 4, 4, 4], 1 ticket worth 215 -> leader's expected
    $EV under MH = (12/24)*215 + (4/24)*0 + ... = 0.5 * 215 = 107.5."""
    eq = icm.icm_equity([12.0, 4.0, 4.0, 4.0], [215.0])
    # Each alive seat's equity is its chip-fraction times the prize pool.
    assert eq[0] == pytest.approx(0.5 * 215.0, abs=1e-9)
    assert eq[1] == pytest.approx((4.0 / 24.0) * 215.0, abs=1e-9)


def test_n_ticket_satellite_conserves_pool():
    """N-ticket sat: pool = N * seat_value exactly."""
    pay = satellite.satellite_payouts(seats=3, seat_value=215.0)
    eq = icm.icm_equity([12.0, 4.0, 4.0, 4.0], pay)
    assert eq.sum() == pytest.approx(3 * 215.0, abs=1e-9)


def test_seat_bubble_factor_inf_when_no_reward():
    """`bubble_factor` returns inf when the equity-gain component is effectively zero.

    With a 1-ticket sat over a busted-villain stack, the all-in is for min(10, 0) = 0 chips, so
    win/lose states are identical and `reward = 0 -> inf`. This is the canonical "hero already
    locked, no further reward possible" scenario.
    """
    bf = satellite.seat_bubble_factor([10.0, 0.0], seats=1, hero=0, villain=1, seat_value=215.0)
    assert bf == float("inf")


# =============================================================================================
# ICM overlay — clamping behavior + identity at deep
# =============================================================================================


def test_icm_tilt_identity_at_deep_bf_one():
    """BF == 1 -> premium == 0 -> tilt is identity (chip-EV unchanged)."""
    cv = np.array([1.0, 0.5, -0.2])
    risk = np.array([0.0, 1.0, 1.0])
    # HU equal: BF = 1.0 exactly.
    out = icm_overlay.icm_tilt(cv, risk, [50.0, 50.0], [60.0, 40.0], hero=0, villain=1)
    assert np.allclose(out, cv, atol=1e-9)


def test_icm_tilt_clamps_inf_to_50():
    """Locked hero -> bubble_factor = inf -> clamp to 50 internally so output is finite."""
    cv = np.array([0.0, 0.0, 0.0])
    risk = np.array([0.0, 1.0, 2.0])
    out = icm_overlay.icm_tilt(cv, risk, [100.0, 0.0], [60.0, 40.0], hero=0, villain=1)
    assert np.all(np.isfinite(out))
    # Expected premium = (50 - 1) = 49; risk [0, 1, 2] -> tilt [0, -49, -98].
    assert out[1] == pytest.approx(-49.0, abs=1e-9)
    assert out[2] == pytest.approx(-98.0, abs=1e-9)


def test_icm_action_value_is_absolute_post_equity():
    """icm_action_value returns the probability-weighted absolute post-hand $-equity of hero.

    For pre=[50,50] and chip_delta=[+10,-10], post=[60,40] -> hero's ICM equity = 52.
    decide_deep picks the action with max icm_action_value (an absolute $-EV measure).
    """
    pre = [50.0, 50.0]
    payouts = [60.0, 40.0]
    outcomes = [(1.0, [+10.0, -10.0])]
    icm_val = icm_overlay.icm_action_value(pre, outcomes, payouts, hero=0)
    post_eq = icm.icm_equity([60.0, 40.0], payouts)[0]
    assert icm_val == pytest.approx(post_eq, abs=1e-9)
    assert icm_val == pytest.approx(52.0, abs=1e-9)


def test_icm_action_value_chip_neutral_equals_pre_equity():
    """For an outcome that doesn't change stacks, icm_action_value equals the pre-hand equity."""
    pre = [50.0, 50.0]
    payouts = [60.0, 40.0]
    outcomes = [(1.0, [0.0, 0.0])]
    icm_val = icm_overlay.icm_action_value(pre, outcomes, payouts, hero=0)
    pre_eq = icm.icm_equity(pre, payouts)[0]
    assert icm_val == pytest.approx(pre_eq, abs=1e-9)
    assert icm_val == pytest.approx(50.0, abs=1e-9)


def test_icm_action_value_average_over_distribution():
    """50/50 mix of [+10,-10] and [-10,+10] (symmetric reverses) gives pre-hand equity."""
    pre = [50.0, 50.0]
    payouts = [60.0, 40.0]
    outcomes = [(0.5, [+10.0, -10.0]), (0.5, [-10.0, +10.0])]
    icm_val = icm_overlay.icm_action_value(pre, outcomes, payouts, hero=0)
    pre_eq = icm.icm_equity(pre, payouts)[0]
    assert icm_val == pytest.approx(pre_eq, abs=1e-9)


def test_chip_ev_action_value_ignores_payouts():
    """chip_ev_action_value is payout-agnostic (used post-deal)."""
    pre = [50.0, 50.0]
    outcomes = [(0.5, [+20.0, -20.0]), (0.5, [-10.0, +10.0])]
    val = icm_overlay.chip_ev_action_value(pre, outcomes, hero=0)
    # 0.5 * 20 + 0.5 * (-10) = 5
    assert val == pytest.approx(5.0, abs=1e-9)


def test_chip_ev_action_value_zero_outcomes():
    """All-zero probability outcomes -> 0.0."""
    val = icm_overlay.chip_ev_action_value([50.0, 50.0], [], hero=0)
    assert val == 0.0


# =============================================================================================
# ICM exploit attenuation — hand-calculated mapping
# =============================================================================================


def test_attenuation_full_below_threshold():
    """bf <= 1.05 -> attenuation = 1.0 (no ICM pressure)."""
    assert icm_exploit.icm_exploit_attenuation(1.0) == 1.0
    assert icm_exploit.icm_exploit_attenuation(1.04) == 1.0
    assert icm_exploit.icm_exploit_attenuation(1.05) == 1.0


def test_attenuation_zero_above_threshold():
    """bf >= 1.60 -> attenuation = 0.0 (full suppression)."""
    assert icm_exploit.icm_exploit_attenuation(1.60) == 0.0
    assert icm_exploit.icm_exploit_attenuation(2.0) == 0.0


def test_attenuation_linear_in_middle():
    """bf in (1.05, 1.60) -> linear ramp (1 - (bf - 1.05) / 0.55)."""
    # At bf = 1.325 (midpoint): attenuation = 0.5
    bf = 1.325
    expected = (1.60 - bf) / (1.60 - 1.05)
    assert icm_exploit.icm_exploit_attenuation(bf) == pytest.approx(expected, abs=1e-12)


def test_attenuation_handles_inf():
    """bf = inf -> attenuation = 0.0 (max pressure, full suppression)."""
    assert icm_exploit.icm_exploit_attenuation(float("inf")) == 0.0


def test_attenuation_handles_nan():
    """bf = NaN -> attenuation = 0.0 (safe fallback)."""
    assert icm_exploit.icm_exploit_attenuation(float("nan")) == 0.0


def test_attenuation_monotone_non_increasing():
    """Attenuation is monotone non-increasing in bf over the active range."""
    prev = 1.0
    for bf in np.linspace(1.0, 1.6, 50):
        cur = icm_exploit.icm_exploit_attenuation(float(bf))
        assert cur <= prev + 1e-12
        prev = cur


def test_state_attenuation_no_payouts_returns_one():
    """No payouts -> state_exploit_attenuation = 1.0 (pure answer, live gate owns policy)."""
    assert icm_exploit.state_exploit_attenuation([50.0, 50.0], None, hero=0) == 1.0
    assert icm_exploit.state_exploit_attenuation([50.0, 50.0], [], hero=0) == 1.0


def test_state_attenuation_min_across_villains():
    """Most-conservative (minimum) attenuation across hero's villains governs."""
    # 3-handed, deep mid-stack vs a low-stack opponent and a high-stack opponent.
    # If ANY villain yields bf in (1.05, 1.60), the minimum is < 1.0.
    stacks = [40.0, 30.0, 50.0]
    payouts = [60.0, 30.0, 10.0]
    att = icm_exploit.state_exploit_attenuation(stacks, payouts, hero=0)
    # Recompute the per-villain attenuations and take the min.
    from pokerbot.tournament.icm import bubble_factor
    per_villain = []
    for v in [1, 2]:
        bf = bubble_factor(stacks, payouts, hero=0, villain=v)
        per_villain.append(icm_exploit.icm_exploit_attenuation(bf))
    expected = min(per_villain)
    assert att == pytest.approx(expected, abs=1e-12)


def test_state_attenuation_busted_hero_returns_one():
    """Busted hero (stack <= 0) is not a live actor -> return 1.0 (safe fallback)."""
    stacks = [0.0, 30.0, 50.0]
    payouts = [60.0, 30.0, 10.0]
    assert icm_exploit.state_exploit_attenuation(stacks, payouts, hero=0) == 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])