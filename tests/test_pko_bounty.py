"""PKO bounty tests: conservation invariant + chip-EV != $-EV (wider calling for bounties)."""

from __future__ import annotations

from pokerbot.tournament.bounty import (
    BountyState,
    min_equity_to_call,
    pko_call_dollar_ev,
)
from pokerbot.tournament.icm import icm_equity


def test_award_ko_splits_half_cash_half_growth():
    st = BountyState.uniform(3, 10.0)
    cash = st.award_ko(eliminator=0, victim=1)
    assert cash == 5.0                 # 50% of victim's 10.0 as immediate cash
    assert st.realized[0] == 5.0
    assert st.live[0] == 15.0          # own bounty grew by the other 50%
    assert st.live[1] == 0.0           # victim eliminated
    assert st.conserved()


def test_split_pot_ko_divides_equally():
    st = BountyState.uniform(4, 8.0)
    st.award_ko(eliminator=0, victim=3, winners=[0, 1])
    assert st.realized[0] == 2.0 and st.realized[1] == 2.0   # 0.5*8 cash split two ways
    assert st.live[0] == 10.0 and st.live[1] == 10.0         # 0.5*8 growth split two ways
    assert st.live[3] == 0.0
    assert st.conserved()


def test_conservation_through_full_tournament():
    n = 6
    st = BountyState.uniform(n, 12.5)
    # a plausible KO chain ending with one winner
    st.award_ko(0, 5)
    st.award_ko(2, 4)
    st.award_ko(0, 2)
    st.award_ko(3, 1)
    st.award_ko(0, 3)   # seat 0 wins, everyone else out
    assert st.conserved()
    winner_residual = st.finalize_winner(0)
    assert winner_residual > 0.0
    assert st.conserved()
    assert abs(sum(st.live)) < 1e-9                      # ledger fully drained
    assert abs(sum(st.realized) - st.initial_pool) < 1e-9


def test_bounty_lowers_call_threshold():
    stacks = [100.0, 100.0, 100.0]
    payouts = [50.0, 30.0, 20.0]
    p_no = min_equity_to_call(stacks, payouts, 0, 1, villain_bounty=0.0)
    p_b = min_equity_to_call(stacks, payouts, 0, 1, villain_bounty=20.0)
    assert 0.0 <= p_b < p_no <= 1.0                     # bounty -> strictly wider calling range


def test_chip_ev_not_dollar_ev_demonstration():
    stacks = [100.0, 100.0, 100.0]
    payouts = [50.0, 30.0, 20.0]
    p_no = min_equity_to_call(stacks, payouts, 0, 1, villain_bounty=0.0)
    p_b = min_equity_to_call(stacks, payouts, 0, 1, villain_bounty=20.0)
    # pick an equity strictly between the two thresholds: calling is -$EV WITHOUT the bounty but
    # +$EV WITH it -> covering for the bounty is correct at a chip-EV (ICM-only) loss.
    p = 0.5 * (p_b + p_no)
    fold = float(icm_equity(stacks, payouts)[0])
    ev_no = pko_call_dollar_ev(stacks, payouts, 0, 1, p, villain_bounty=0.0)
    ev_b = pko_call_dollar_ev(stacks, payouts, 0, 1, p, villain_bounty=20.0)
    assert ev_no < fold < ev_b


def test_call_ev_reduces_to_icm_without_bounty():
    stacks = [120.0, 80.0, 100.0]
    payouts = [50.0, 30.0, 20.0]
    p = 0.55
    s = stacks
    risked = min(s[0], s[1])
    win = [s[0] + risked, s[1] - risked, s[2]]
    lose = [s[0] - risked, s[1] + risked, s[2]]
    icm_win = float(icm_equity(win, payouts)[0])
    icm_lose = float(icm_equity(lose, payouts)[0])
    want = p * icm_win + (1.0 - p) * icm_lose
    got = pko_call_dollar_ev(stacks, payouts, 0, 1, p, villain_bounty=0.0)
    assert abs(got - want) < 1e-9
