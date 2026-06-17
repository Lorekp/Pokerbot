"""R4: live gift-budget safety clamp (gift-modulated bounded-ε, Ganzfried-Sandholm).

Validates the deterministic mechanism: zero gift ⇒ behavior identical to the static Pmax cap (the
bounded-ε floor is never breached at cold start); banked gifts monotonically relax the cap up to a
hard ceiling; the gift signal grows vs a demonstrably-leaky opponent.
"""

from __future__ import annotations

import random

import numpy as np

from pokerbot.opponent.safety_budget import (
    ALPHA_CEILING,
    MAX_BONUS,
    GiftRegister,
    gift_from_qrow,
    gift_to_alpha_bonus,
)


def test_zero_gift_is_a_noop_bounded_epsilon_preserved():
    reg = GiftRegister()
    assert reg.available("v") == 0.0
    assert reg.alpha_bonus("v") == 0.0
    # capped_max_alpha returns the base cap EXACTLY ⇒ bit-identical to the static Pmax cap.
    assert reg.capped_max_alpha("v", 0.7) == 0.7
    assert gift_to_alpha_bonus(0.0) == 0.0


def test_session_budget_bounds_and_halts_gift_funded_deviation():
    """H7.3 (BEFEWP session bound): gift-funded EXTRA deviation across a session may never exceed the
    gifts actually received. Cold start ⇒ 0 (bit-identical floor); banked gifts open headroom; spending
    beyond the banked total HALTS the bonus back to 0 (base bounded-ε only)."""
    reg = GiftRegister()
    # Cold start: no session headroom ⇒ no bonus (bit-identical to the static cap).
    assert reg.session_available() == 0.0
    assert reg.alpha_bonus("v") == 0.0
    assert reg.capped_max_alpha("v", 0.7) == 0.7

    # Bank gifts ⇒ session headroom + a positive bonus.
    reg.credit("v", 3.0)
    assert reg.session_available() == 3.0
    assert reg.alpha_bonus("v") > 0.0

    # Spending beyond the banked total exhausts the session ⇒ bonus halts to 0 even though the per-key
    # budget might still look non-zero — the cumulative guarantee dominates.
    reg.spend("v", 100.0)
    assert reg.session_available() == 0.0
    assert reg.alpha_bonus("v") == 0.0
    assert reg.capped_max_alpha("v", 0.7) == 0.7

    # A fresh session resets the ledger.
    reg.reset_session()
    reg.credit("v", 2.0)
    assert reg.session_available() == 2.0


def test_alpha_bonus_monotone_saturating_and_ceiling():
    vals = [gift_to_alpha_bonus(x) for x in (0.0, 0.5, 1.0, 2.0, 5.0, 50.0)]
    assert vals == sorted(vals)                      # monotone non-decreasing
    assert all(0.0 <= v <= MAX_BONUS + 1e-9 for v in vals)
    assert vals[-1] > vals[0]                         # a banked budget DOES add headroom
    reg = GiftRegister()
    reg.credit("v", 1e6)                              # absurd budget
    assert reg.capped_max_alpha("v", 0.7) <= ALPHA_CEILING + 1e-9   # hard ceiling holds
    assert reg.capped_max_alpha("v", 0.7) > 0.7       # but it did relax the cap


def test_credit_decays_toward_recent_headroom():
    reg = GiftRegister()
    reg.credit("v", 5.0)
    hi = reg.available("v")
    for _ in range(100):      # ~100 gift-free observations fade the stale budget (decay ~34-hand t½)
        reg.credit("v", 0.0)
    assert reg.available("v") < 0.2 * hi


def test_gift_from_qrow_is_surrendered_ev():
    q = np.array([0.0, 0.2, 1.0, -0.5, 0.4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9])
    legal = np.zeros(11, dtype=bool)
    legal[[1, 2, 10]] = True                          # check/call/all-in legal; max legal Q = call(1.0)
    assert abs(gift_from_qrow(q, taken_idx=1, legal_mask=legal) - (1.0 - 0.2)) < 1e-9  # checked, gifted 0.8
    assert gift_from_qrow(q, taken_idx=2, legal_mask=legal) == 0.0   # took the max ⇒ no gift
    assert gift_from_qrow(q, taken_idx=99, legal_mask=legal) == 0.0  # out of range ⇒ 0


def test_high_budget_raises_effective_alpha_when_confident():
    # With a saturated read (conf high) the gift bonus lifts the realized alpha above the base 0.7 cap.
    from pokerbot.opponent.bayes_exploit import DEFAULT_MAX_ALPHA

    reg = GiftRegister()
    reg.credit("v", 10.0)
    base_cap = DEFAULT_MAX_ALPHA
    relaxed = reg.capped_max_alpha("v", base_cap)
    conf = 0.95
    assert min(conf, relaxed) > min(conf, base_cap)   # more deviation funded by the banked gift


def test_battery_credits_gifts_vs_leaky_opponent():
    from pokerbot.eval.exploitation_battery import run_archetype

    reg = GiftRegister()
    r = run_archetype("calling_station", hands=200, seed=2, gift_register=reg)
    assert "gift_budget" in r and "gift_alpha_bonus" in r
    assert r["gift_budget"] > 0.0                      # a leaky station banks a positive budget
    assert r["gift_alpha_bonus"] >= 0.0


def test_battery_zero_register_is_unchanged():
    # Without a register the summary has no gift keys and the run still works (R4 fully off ⇒ unchanged).
    from pokerbot.eval.exploitation_battery import run_archetype

    r = run_archetype("calling_station", hands=60, seed=2)
    assert "gift_budget" not in r
    assert r["hands"] == 60
