"""Multi-table MTT simulator (pokerbot/tournament/mtt_sim.py).

Correctness anchors: a tournament always completes to a valid finish + ladder payout; a HOMOGENEOUS field
(hero plays the same strategy as everyone) is unbiased (mean finish ~ (N+1)/2, ROI ~ 0) — proving the sim has
no seat/table/hero-tracking bias through table breaks; and the value lever is directional (a sound hero beats
a spew-y field's $ROI). Sizes kept small so the suite stays fast.
"""

from __future__ import annotations

import random

from pokerbot.tournament.mtt_sim import (
    chip_ev_strategy,
    icm_aware_strategy,
    measure_roi,
    simulate_mtt,
)


def test_completes_to_valid_finish_and_payout():
    payouts = [50.0, 30.0, 20.0]
    r = simulate_mtt(12, payouts, chip_ev_strategy(), chip_ev_strategy(),
                     rng=random.Random(1), bf_sims=50)
    assert 1 <= r.hero_finish <= 12
    assert r.players == 12
    pay = payouts + [0.0] * (12 - len(payouts))
    assert r.hero_payout == pay[r.hero_finish - 1]      # payout matches the ladder for the finish


def test_homogeneous_field_is_unbiased():
    """Hero plays the SAME strategy as the field -> by symmetry mean finish ~ (N+1)/2 and ROI ~ 0. This is the
    decisive no-bias check (it exercises multi-table balancing + hero identity tracking through breaks)."""
    N = 18                                          # 2 tables of 9 -> balancing + a table break will occur
    payouts = [40.0, 25.0, 18.0, 10.0, 7.0]         # 5 of 18 paid
    strat = chip_ev_strategy()
    res = measure_roi(N, payouts, strat, strat, n_sims=120, rng=random.Random(4), bf_sims=1)
    expected_finish = (N + 1) / 2                    # 9.5
    assert abs(res["mean_finish"] - expected_finish) < 2.0, res
    assert abs(res["roi"]) < 0.45, res               # fair share within MC noise at 120 sims


def test_payout_and_itm_accounting():
    N = 16
    payouts = [60.0, 40.0]                           # 2 of 16 paid
    res = measure_roi(N, payouts, chip_ev_strategy(), chip_ev_strategy(),
                      n_sims=100, rng=random.Random(2), bf_sims=1)
    # buy-in share = total prize / entrants
    assert abs(res["buy_in_share"] - sum(payouts) / N) < 1e-9
    # ITM rate ~ paid/entrants for a homogeneous field (2/16 = 0.125), within MC noise
    assert 0.02 <= res["itm_rate"] <= 0.30, res


def test_sound_hero_beats_spewy_field():
    """A hero with disciplined thresholds vs a spew-y (over-loose) field ladders better -> higher $ROI. Flat-ish
    payout so survival is rewarded (the value-extraction direction)."""
    N = 18
    payouts = [30.0, 22.0, 16.0, 12.0, 9.0, 6.0, 5.0]   # 7 of 18 paid (flat-ish)
    sound = chip_ev_strategy(jam=0.50, call=0.66)
    spewy = chip_ev_strategy(jam=0.18, call=0.28)        # jams/calls far too wide -> busts itself
    hero_vs_spewy = measure_roi(N, payouts, sound, spewy, n_sims=160, rng=random.Random(11), bf_sims=1)
    # a sound hero in a field of spew should beat its fair share
    assert hero_vs_spewy["roi"] > 0.0, hero_vs_spewy
    # control: spewy hero vs spewy field is ~fair (no skill edge)
    spewy_vs_spewy = measure_roi(N, payouts, spewy, spewy, n_sims=160, rng=random.Random(11), bf_sims=1)
    assert hero_vs_spewy["roi"] > spewy_vs_spewy["roi"], (hero_vs_spewy, spewy_vs_spewy)


def test_realistic_resolver_equities_sane():
    """The real-equity resolver loads the HU grid and gives sane preflop equities (best bucket ~AA crushes the
    worst; cutoffs tighten with quantile)."""
    from pokerbot.tournament.mtt_sim import _RealEquity

    r = _RealEquity.get()
    assert r.eq.shape == (169, 169)
    assert 0.30 <= float(r.bucket_strength.min()) <= 0.40   # ~72o
    assert 0.83 <= float(r.bucket_strength.max()) <= 0.90   # ~AA
    assert r.strength_cutoff(0.85) > r.strength_cutoff(0.5)  # tighter range = higher cutoff
    best, worst = int(r.bucket_strength.argmax()), int(r.bucket_strength.argmin())
    assert r.win_prob(best, worst) > 0.80                    # AA vs trash ~87%


def test_realistic_mode_completes_and_unbiased():
    """A REAL-card/REAL-equity tournament completes to a valid finish; a homogeneous field stays ~unbiased."""
    res = measure_roi(12, [50.0, 30.0, 20.0], chip_ev_strategy(), chip_ev_strategy(),
                      n_sims=120, rng=random.Random(5), realistic=True, bf_sims=1)
    assert 1.0 <= res["mean_finish"] <= 12.0
    assert abs(res["mean_finish"] - 6.5) < 2.0               # ~fair under symmetry
    assert abs(res["roi"]) < 0.45


def test_icm_aware_strategy_runs_and_ladders():
    """An ICM-aware hero (tightens near the global bubble) completes and produces a sane ITM rate -> the field
    bubble-factor path is exercised end-to-end in the sim (not just unit-tested)."""
    N = 12
    payouts = [50.0, 30.0, 20.0]
    res = measure_roi(N, payouts, icm_aware_strategy(), chip_ev_strategy(),
                      n_sims=40, rng=random.Random(7), bf_sims=120)
    assert 1.0 <= res["mean_finish"] <= N
    assert 0.0 <= res["itm_rate"] <= 1.0
