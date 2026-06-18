"""Unit tests for the CoinPoker rake + CoinRewards rakeback model."""

from __future__ import annotations

import pytest

from pokerbot.economics import rake as rk


def test_nlnfd_no_rake_preflop():
    # A hand that ends preflop pays zero regardless of pot size.
    assert rk.coinpoker_rake_bb(200.0, 6, ended_preflop=True, stake_key="0.05/0.10") == 0.0


def test_five_percent_uncapped_when_cap_far():
    # Micro 5+ cap is 10bb -> a 40bb pot pays full 5% = 2.0bb (cap does not bind).
    assert rk.coinpoker_rake_bb(40.0, 6, False, "0.05/0.10") == pytest.approx(2.0)


def test_cap_binds():
    # High stakes ($25/50) 5+ cap = 0.40bb -> binds for any pot >= 8bb.
    assert rk.coinpoker_rake_bb(8.0, 6, False, "25/50") == pytest.approx(0.40)
    assert rk.coinpoker_rake_bb(500.0, 6, False, "25/50") == pytest.approx(0.40)
    # Just under the binding point: 6bb pot -> 5% = 0.30 < 0.40 cap.
    assert rk.coinpoker_rake_bb(6.0, 6, False, "25/50") == pytest.approx(0.30)


def test_dealt_count_tiers():
    assert rk.cap_tier(2) == 2
    assert rk.cap_tier(3) == 3
    assert rk.cap_tier(4) == 3
    assert rk.cap_tier(5) == 5
    assert rk.cap_tier(6) == 5
    # Tier raises the cap at a fixed stake (more dealt -> higher cap).
    s = rk.STAKES["0.10/0.25"]
    assert s.cap_bb[2] <= s.cap_bb[3] <= s.cap_bb[5]


def test_cap_collapses_with_stake():
    # The 5+ cap in BB strictly shrinks from micro to high (the load-bearing structural fact).
    caps = [rk.STAKES[s].cap_bb[5] for s in rk.STAKE_ORDER]
    assert caps[0] == 10.0 and caps[-1] == 0.40
    # Monotone non-increasing across the ordered stakes.
    assert all(caps[i] >= caps[i + 1] for i in range(len(caps) - 1))


def test_contributed_method_sums_to_total_rake():
    # Two players each committing half of a 40bb pot at micro: each pays half the 2.0bb rake.
    pot, half = 40.0, 20.0
    each = rk.contributed_rake_bb(pot, half, 6, False, "0.05/0.10")
    assert each == pytest.approx(1.0)
    assert 2 * each == pytest.approx(rk.coinpoker_rake_bb(pot, 6, False, "0.05/0.10"))


def test_contributed_zero_pot():
    assert rk.contributed_rake_bb(0.0, 0.0, 6, False, "1/2") == 0.0


def test_rakeback_modes():
    # Floor = 15% everywhere.
    for s in rk.STAKE_ORDER:
        assert rk.rakeback_fraction(s, "floor") == pytest.approx(0.15)
    # Modeled: micro ~100%, decreasing band to high.
    assert rk.rakeback_fraction("0.01/0.02", "modeled") == pytest.approx(1.00)
    assert rk.rakeback_fraction("0.10/0.25", "modeled") == pytest.approx(0.40)
    assert rk.rakeback_fraction("1/2", "modeled") == pytest.approx(0.25)
    assert rk.rakeback_fraction("25/50", "modeled") == pytest.approx(0.20)
    with pytest.raises(ValueError):
        rk.rakeback_fraction("1/2", "bogus")


def test_net_rake_le_gross():
    gross = rk.contributed_rake_bb(40.0, 20.0, 6, False, "0.05/0.10")
    net_floor = rk.net_rake_bb(40.0, 20.0, 6, False, "0.05/0.10", rakeback_mode="floor")
    net_modeled = rk.net_rake_bb(40.0, 20.0, 6, False, "0.05/0.10", rakeback_mode="modeled")
    assert net_modeled <= net_floor <= gross
    # '0.05/0.10' is the LOW band (40% modeled) -> net = 60% of gross.
    assert net_modeled == pytest.approx(0.6 * gross)
    # A true MICRO stake has ~100% modeled rakeback -> net micro rake ~ 0.
    micro_net = rk.net_rake_bb(40.0, 20.0, 6, False, "0.01/0.02", rakeback_mode="modeled")
    assert micro_net == pytest.approx(0.0, abs=1e-9)


# --- Splash fee (0.1bb/dealt hand at the non-micro low stakes) -------------------------------------

def test_splash_eligible_only_low_band():
    # Eligible at the three non-micro low stakes; NOT at micro / mid / high.
    assert rk.splash_eligible("0.05/0.10")
    assert rk.splash_eligible("0.10/0.25")
    assert rk.splash_eligible("0.25/0.50")
    assert not rk.splash_eligible("0.01/0.02")  # micro: infused, no fee
    assert not rk.splash_eligible("0.02/0.05")  # micro
    assert not rk.splash_eligible("0.50/1")     # mid: no splash program
    assert not rk.splash_eligible("25/50")      # high


def test_splash_zero_off_band():
    # No fee at micro / mid / high regardless of pot.
    for sk in ("0.01/0.02", "0.50/1", "25/50"):
        assert rk.splash_fee_bb(40.0, 20.0, sk) == 0.0


def test_splash_charged_on_preflop_hands():
    # Unlike rake (no-flop-no-drop), the splash fee is taken even on a hand that ends preflop.
    # A preflop pot of just the blinds (1.5bb) where the hero posted the BB (1.0): hero share = 1/1.5.
    fee = rk.splash_fee_bb(1.5, 1.0, "0.10/0.25")
    assert fee == pytest.approx(rk.SPLASH_FEE_BB * (1.0 / 1.5))
    # The rake on that same preflop hand is zero (NLNFD), proving splash is a SEPARATE cost.
    assert rk.coinpoker_rake_bb(1.5, 6, ended_preflop=True, stake_key="0.10/0.25") == 0.0


def test_splash_contributed_shares_sum_to_fee():
    # Contributed attribution: each player's splash share sums to the full 0.1bb fee.
    pot = 40.0
    shares = [10.0, 10.0, 20.0]  # three contributors summing to the pot
    total = sum(rk.splash_fee_bb(pot, c, "0.05/0.10") for c in shares)
    assert total == pytest.approx(rk.SPLASH_FEE_BB)


def test_splash_zero_pot():
    assert rk.splash_fee_bb(0.0, 0.0, "0.05/0.10") == 0.0


def test_net_cost_is_rake_plus_splash():
    pot, contrib = 40.0, 20.0
    for sk, mode in (("0.05/0.10", "modeled"), ("0.10/0.25", "floor"), ("0.50/1", "modeled"),
                     ("0.01/0.02", "modeled"), ("25/50", "floor")):
        rake = rk.net_rake_bb(pot, contrib, 6, False, sk, rakeback_mode=mode)
        splash = rk.splash_fee_bb(pot, contrib, sk)
        cost = rk.net_cost_bb(pot, contrib, 6, False, sk, rakeback_mode=mode)
        assert cost == pytest.approx(rake + splash)
    # At an eligible low stake the full cost strictly exceeds the rake-only figure.
    assert rk.net_cost_bb(40.0, 20.0, 6, False, "0.10/0.25", rakeback_mode="modeled") > \
        rk.net_rake_bb(40.0, 20.0, 6, False, "0.10/0.25", rakeback_mode="modeled")


def test_splash_does_not_perturb_rake_numbers():
    # Adding the splash model must NOT change any existing rake/net-rake number (byte-stable additivity).
    assert rk.coinpoker_rake_bb(40.0, 6, False, "0.05/0.10") == pytest.approx(2.0)
    assert rk.net_rake_bb(40.0, 20.0, 6, False, "0.05/0.10", rakeback_mode="modeled") == \
        pytest.approx(0.6 * rk.contributed_rake_bb(40.0, 20.0, 6, False, "0.05/0.10"))
    # net_cost at a NON-eligible stake equals net_rake exactly (splash term is 0 there).
    assert rk.net_cost_bb(40.0, 20.0, 6, False, "0.50/1", rakeback_mode="modeled") == \
        pytest.approx(rk.net_rake_bb(40.0, 20.0, 6, False, "0.50/1", rakeback_mode="modeled"))
