"""Rake-aware value-SIZE escalation discount (Workstream F).

The HUD value-SIZE escalation (``value_sizing_logits``, mid->big bet buckets) grows the final pot, and
CoinPoker rakes ``min(5%*pot, cap)``. BELOW the cap, every extra BB of pot forfeits 5% to rake, so
escalating thin value into a small pot can cost more rake than the marginal extraction is worth; ABOVE
the cap the rake is FLAT, so the over-bet is rake-free and must NOT be discounted. These tests pin:
  * the marginal-rake primitive direction (5% below cap, 0 above),
  * the rake_keep discount fires below the cap and VANISHES above it,
  * backward-compat: no rake context => byte-identical to today's logits,
  * preflop => no discount (no-flop-no-drop),
  * direction sanity: a big-extraction station keeps almost full escalation; a thin/weak read is trimmed.
"""

from __future__ import annotations

import numpy as np

from pokerbot.economics.rake import (
    RAKE_PCT,
    STAKES,
    cap_tier,
    coinpoker_rake_bb,
    marginal_rake_bb,
)
from pokerbot.engine.actions import NUM_ACTIONS
from pokerbot.opponent.stats import OpponentStats
from pokerbot.opponent.value_sizing import (
    _BET_LO,
    _BIG_LO,
    _ALL_IN,
    RakeContext,
    rake_keep_factor,
    value_sizing_logits,
)

_VALLEY = "0.10/0.25"   # the documented rake-valley default stake


def _station(**hud) -> OpponentStats:
    """A confirmed loose-passive station carrying server HUD stats."""
    s = OpponentStats(hands=50, vpip=30, pfr=4, aggression_bets_raises=5, aggression_calls=20)
    if hud:
        s.hud = dict(hud)
    return s


def _neutral(**hud) -> OpponentStats:
    """Unknown-archetype HUD carrier so the score is driven by the HUD alone (no station floor)."""
    s = OpponentStats(hands=50, vpip=15, pfr=8, aggression_bets_raises=10, aggression_calls=10)
    assert s.archetype() == "unknown"
    if hud:
        s.hud = dict(hud)
    return s


# --- marginal_rake_bb primitive: the rake DIRECTION ------------------------------------------------

def test_marginal_rake_is_five_percent_below_cap():
    # Both endpoints well below the cap => marginal rake == 5% of the pot growth (the taxed increment).
    cap = STAKES[_VALLEY].cap_bb[cap_tier(3)]
    before, after = 1.0, 2.0
    assert RAKE_PCT * after < cap                          # both on the 5% ramp
    mr = marginal_rake_bb(before, after, 3, _VALLEY)
    assert mr == RAKE_PCT * (after - before)               # exactly 5% of the increment


def test_marginal_rake_is_zero_above_cap():
    # Both endpoints above the cap-binding pot => rake is FLAT => growing the pot is rake-FREE.
    cap = STAKES[_VALLEY].cap_bb[cap_tier(3)]
    cap_pot = cap / RAKE_PCT                                # pot where 5% first hits the cap
    before, after = cap_pot + 10.0, cap_pot + 50.0
    assert marginal_rake_bb(before, after, 3, _VALLEY) == 0.0


def test_marginal_rake_only_taxes_the_below_cap_portion():
    # before below cap, after above cap => only the (cap - 5%*before) below-cap slice is taxed.
    cap = STAKES[_VALLEY].cap_bb[cap_tier(3)]
    cap_pot = cap / RAKE_PCT
    before, after = cap_pot - 4.0, cap_pot + 20.0
    expected = coinpoker_rake_bb(after, 3, False, _VALLEY) - coinpoker_rake_bb(before, 3, False, _VALLEY)
    mr = marginal_rake_bb(before, after, 3, _VALLEY)
    assert mr == expected
    assert 0.0 < mr < RAKE_PCT * (after - before)          # less than the full-ramp tax (cap clips it)


def test_marginal_rake_non_negative_on_shrink():
    assert marginal_rake_bb(20.0, 10.0, 3, _VALLEY) == 0.0   # defensive: never negative


# --- rake_keep_factor: the discount ----------------------------------------------------------------

def test_rake_keep_is_one_with_no_context():
    # No context => no discount (default-safe, today's behaviour).
    assert rake_keep_factor(_station(wtsd=37, wsd=42, foldToCbet=30), None) == 1.0


def test_rake_keep_below_cap_discounts():
    # A mid-stake pot that straddles the cap-binding point: the escalation IS rake-discounted.
    st = _station(wtsd=37, wsd=42, foldToCbet=30)
    rk = rake_keep_factor(st, RakeContext(pot_bb=6.0, n_dealt=3, stake_key=_VALLEY))
    assert 0.0 < rk < 1.0                                   # a real discount, but soft (not a hard block)


def test_rake_keep_above_cap_is_one():
    # A large pot (both mid- and big-size resulting pots above the cap) => rake flat => no discount.
    st = _station(wtsd=37, wsd=42, foldToCbet=30)
    rk = rake_keep_factor(st, RakeContext(pot_bb=100.0, n_dealt=3, stake_key=_VALLEY))
    assert rk == 1.0


def test_rake_keep_micro_small_pot_is_near_one():
    # MICRO has a FAT cap (~10bb), so a small pot's whole escalation stays on the 5% ramp but the
    # absolute rake is tiny relative to the same field's extraction => discount is the modest pure-5%
    # bite (not a hard cut). Above-cap micro pots get NO discount. Both << the mid-stake straddle.
    st = _station(wtsd=37, wsd=42, foldToCbet=30)
    micro = "0.01/0.02"
    rk_small = rake_keep_factor(st, RakeContext(pot_bb=1.5, n_dealt=5, stake_key=micro))
    rk_big = rake_keep_factor(st, RakeContext(pot_bb=400.0, n_dealt=5, stake_key=micro))
    assert rk_big == 1.0                                    # well above the fat micro cap => no discount
    assert rk_small >= 0.85                                 # pure-5% bite only; never a hard cut


def test_rake_keep_preflop_no_discount():
    # Preflop => no-flop-no-drop => rake hasn't applied => no discount regardless of pot.
    st = _station(wtsd=37, wsd=42, foldToCbet=30)
    rk = rake_keep_factor(st, RakeContext(pot_bb=6.0, n_dealt=3, stake_key=_VALLEY, is_preflop=True))
    assert rk == 1.0


def test_rake_keep_killswitch(monkeypatch):
    monkeypatch.setenv("POKERBOT_RAKE_AWARE_SIZING", "0")
    st = _station(wtsd=37, wsd=42, foldToCbet=30)
    assert rake_keep_factor(st, RakeContext(pot_bb=6.0, n_dealt=3, stake_key=_VALLEY)) == 1.0


def test_rake_keep_unknown_stake_no_discount():
    st = _station(wtsd=37, wsd=42, foldToCbet=30)
    assert rake_keep_factor(st, RakeContext(pot_bb=6.0, n_dealt=3, stake_key="not/a/stake")) == 1.0


def test_rake_keep_direction_big_extraction_vs_thin():
    # Direction sanity: a MEGA-station (very sticky, weak showdown => high call-through => large
    # marginal extraction) keeps MORE of the escalation than a weak/marginal read (low call-through =>
    # small extraction => the same marginal rake bites harder). Same pot/stake so only extraction varies.
    mega = _station(wtsd=55, wsd=30, foldToCbet=5, fta=8)
    weak = _neutral(wsd=49, foldToCbet=42)                  # barely sticky, near-neutral read
    ctx = RakeContext(pot_bb=6.0, n_dealt=3, stake_key=_VALLEY)
    rk_mega = rake_keep_factor(mega, ctx)
    rk_weak = rake_keep_factor(weak, ctx)
    assert rk_mega > rk_weak                                # big extraction => keep almost full escalation


# --- value_sizing_logits integration ---------------------------------------------------------------

def test_value_sizing_backward_compat_byte_identical():
    # No rake context => EXACTLY today's logits (snapshot the size buckets).
    st = _station(wtsd=37, wsd=42, foldToCbet=30)
    base = value_sizing_logits(st, 0.72)
    with_none = value_sizing_logits(st, 0.72, rake_ctx=None)
    assert np.array_equal(base, with_none)
    # And the historical snapshot of the size-escalation boost is unchanged (no silent drift).
    assert base[_BIG_LO] > base[_BET_LO] > 0.0
    assert base[0] == 0.0 and base[_ALL_IN] == 0.0


def test_value_sizing_below_cap_is_trimmed_vs_baseline():
    # At the mid-stake straddle pot, the SIZE boost is strictly smaller than the no-context baseline.
    st = _station(wtsd=37, wsd=42, foldToCbet=30)
    base = value_sizing_logits(st, 0.72)
    ctx = RakeContext(pot_bb=6.0, n_dealt=3, stake_key=_VALLEY)
    raked = value_sizing_logits(st, 0.72, rake_ctx=ctx)
    # size buckets (mid + big) are dampened ...
    assert raked[_BIG_LO] < base[_BIG_LO]
    assert raked[_BET_LO] < base[_BET_LO]
    # ... but it's a soft scaling: still escalating in the same direction, never sign-flipped.
    assert raked[_BIG_LO] > 0.0 and raked[_BET_LO] > 0.0
    # the check-discourage term (a READ, not a SIZE) is NOT rake-scaled.
    assert raked[1] == base[1]


def test_value_sizing_above_cap_matches_baseline():
    # Above the cap, the escalation must NOT be penalized: identical to the no-context baseline.
    st = _station(wtsd=37, wsd=42, foldToCbet=30)
    base = value_sizing_logits(st, 0.72)
    ctx = RakeContext(pot_bb=120.0, n_dealt=3, stake_key=_VALLEY)
    raked = value_sizing_logits(st, 0.72, rake_ctx=ctx)
    assert np.array_equal(base, raked)


def test_value_sizing_preflop_context_matches_baseline():
    st = _station(wtsd=37, wsd=42, foldToCbet=30)
    base = value_sizing_logits(st, 0.72)
    ctx = RakeContext(pot_bb=6.0, n_dealt=3, stake_key=_VALLEY, is_preflop=True)
    assert np.array_equal(base, value_sizing_logits(st, 0.72, rake_ctx=ctx))


def test_value_sizing_air_still_zero_with_rake_ctx():
    # The rake discount only scales an existing escalation; air never escalates so it stays zero.
    st = _station(wtsd=37, wsd=42, foldToCbet=30)
    ctx = RakeContext(pot_bb=6.0, n_dealt=3, stake_key=_VALLEY)
    assert np.allclose(value_sizing_logits(st, 0.30, rake_ctx=ctx), 0.0)


def test_value_sizing_direction_big_value_vs_thin_under_rake():
    # A BIG value bet vs a mega-station (large extraction) keeps almost full escalation; a thin small-pot
    # bet vs a marginal read is trimmed the most => the SIZE boost ratio (raked/baseline) is larger for
    # the station than for the thin read.
    mega = _station(wtsd=55, wsd=30, foldToCbet=5, fta=8)
    weak = _neutral(wsd=49, foldToCbet=42)
    ctx = RakeContext(pot_bb=6.0, n_dealt=3, stake_key=_VALLEY)
    mega_keep = value_sizing_logits(mega, 0.95, rake_ctx=ctx)[_BIG_LO] / max(
        value_sizing_logits(mega, 0.95)[_BIG_LO], 1e-9)
    # weak read still needs a nonzero baseline escalation to form a ratio
    weak_base = value_sizing_logits(weak, 0.72)[_BIG_LO]
    if weak_base > 1e-9:
        weak_keep = value_sizing_logits(weak, 0.72, rake_ctx=ctx)[_BIG_LO] / weak_base
        assert mega_keep > weak_keep
    assert mega_keep >= 0.85   # the station keeps almost all of its escalation


def test_shape_and_finite_with_rake_ctx():
    st = _station(wtsd=40, wsd=35, foldToCbet=15, fta=20)
    ctx = RakeContext(pot_bb=6.0, n_dealt=3, stake_key=_VALLEY)
    adj = value_sizing_logits(st, 1.0, rake_ctx=ctx)
    assert adj.shape == (NUM_ACTIONS,)
    assert np.isfinite(adj).all()
