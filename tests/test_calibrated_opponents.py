"""Tests for the IRC-calibrated skill-tier opponents (pokerbot/eval/calibrated_opponents.py) and
their wiring into the exploitation battery.

The headline test is ``test_realized_hud_matches_target``: it plays each tier bot in the real 6-max
engine and asserts the realized HUD stats the bot DIRECTLY controls (VPIP / PFR / AF / fold_to_cbet)
land within tolerance of the IRC-grounded target. WTSD / WSD are showdown-reach stats with a known
structural floor in homogeneous self-play (every existing caricature bot shows WTSD > 0.74 in the same
harness), so they are reported, not asserted — mirroring the validation script.
"""

from __future__ import annotations

import random

import pytest

from pokerbot.engine.actions import ActionType
from pokerbot.engine.cards import Card, Rank, Suit
from pokerbot.engine.game import GameState
from pokerbot.eval.calibrated_opponents import (
    CalibratedOpponentBot,
    build_calibrated_bot,
    load_skill_tiers,
    postflop_threshold_for_bottom_fraction,
    strength_threshold_for_top_fraction,
    tier_target,
)


# --- skill-tier data ------------------------------------------------------------------------------
def test_skill_tiers_load_and_grounding():
    tiers = load_skill_tiers()
    # the four-tier ladder exists
    for name in ("recreational", "weak_reg", "solid_reg", "pro"):
        assert name in tiers, name
    # HUD-target tiers carry the six stat keys; pro is the equilibrium bot
    for name in ("recreational", "weak_reg", "solid_reg"):
        t = tier_target(name)
        for k in ("vpip", "pfr", "af", "wtsd", "wsd", "fold_to_cbet"):
            assert k in t
    assert "target" not in tiers["pro"]
    assert tiers["pro"].get("bot") == "blueprint"
    with pytest.raises(ValueError):
        tier_target("pro")  # equilibrium tier has no HUD target


def test_tier_monotonicity_recreational_to_solid():
    # discipline increases recreational -> weak_reg -> solid_reg: VPIP falls, PFR/AF/fold_to_cbet rise
    rec, weak, solid = (tier_target(t) for t in ("recreational", "weak_reg", "solid_reg"))
    assert rec["vpip"] > weak["vpip"] > solid["vpip"]
    assert rec["pfr"] < weak["pfr"] < solid["pfr"]
    assert rec["af"] < weak["af"] < solid["af"]
    assert rec["fold_to_cbet"] < weak["fold_to_cbet"] < solid["fold_to_cbet"]


# --- CDF inversion --------------------------------------------------------------------------------
def test_preflop_cdf_inversion_monotone_and_bracketing():
    # a larger top-fraction -> a LOWER threshold (we play more, so the cutoff drops)
    t10 = strength_threshold_for_top_fraction(0.10)
    t50 = strength_threshold_for_top_fraction(0.50)
    t90 = strength_threshold_for_top_fraction(0.90)
    assert t10 > t50 > t90
    # extremes play nothing / everything
    assert strength_threshold_for_top_fraction(0.0) > strength_threshold_for_top_fraction(0.5)
    assert strength_threshold_for_top_fraction(1.0) < strength_threshold_for_top_fraction(0.5)


def test_preflop_cdf_inversion_realizes_fraction():
    # the threshold for top-X% should leave ~X% of random starting hands at or above it
    from pokerbot.eval.calibrated_opponents import _preflop_strength_proxy

    cards = [Card(r, s) for r in Rank for s in Suit]
    combos = [(cards[i], cards[j]) for i in range(len(cards)) for j in range(i + 1, len(cards))]
    for frac in (0.20, 0.48, 0.61):
        thr = strength_threshold_for_top_fraction(frac)
        realized = sum(1 for c in combos if _preflop_strength_proxy(c) >= thr) / len(combos)
        assert abs(realized - frac) < 0.06, (frac, realized)


def test_postflop_cdf_inversion_monotone():
    f10 = postflop_threshold_for_bottom_fraction(0.10)
    f50 = postflop_threshold_for_bottom_fraction(0.50)
    f90 = postflop_threshold_for_bottom_fraction(0.90)
    assert f10 < f50 < f90  # fold MORE -> a HIGHER continue threshold


# --- bot construction + legality ------------------------------------------------------------------
def test_bot_always_returns_legal_action():
    bot = build_calibrated_bot("weak_reg", random.Random(0))
    for h in range(60):
        state = GameState.new_hand(button_seat=h % 6, rng=random.Random(h))
        state.deal_hole_cards()
        # walk a few decision nodes
        for _ in range(8):
            legal = state.legal_actions()
            if not legal:
                break
            a = bot(state)
            assert a in legal, (a, [str(x) for x in legal])
            state.apply_action(a)
            if state.is_terminal():
                break
            state.advance_to_next_actor()


def test_range_structure_continues_stronger_than_folds():
    # The human prior: facing a bet, the hands the calibrated bot CONTINUES with (call/raise) are
    # stronger than the hands it FOLDS — i.e. it has real range structure, not random aggression.
    # (The bot's BET range is intentionally polarized — value + a bluff tail — so the cleaner
    # structure claim is on the continue/fold split.)
    from pokerbot.engine.game import play_hand
    from pokerbot.eval.calibrated_opponents import _hand_strength

    hs_continue, hs_fold = [], []
    for h in range(500):
        st = GameState.new_hand(button_seat=h % 6, rng=random.Random(h))
        bots = {s: build_calibrated_bot("weak_reg", random.Random(31 * (s + 1) + h)) for s in range(6)}

        def mk(s, _b):
            def pol(state):
                legal = state.legal_actions()
                facing_bet = not any(a.action_type == ActionType.CHECK for a in legal)
                hs = _hand_strength(state, state.action_seat)
                a = _b(state)
                if state.street.name != "PREFLOP" and facing_bet:
                    if a.action_type in (ActionType.CALL, ActionType.BET, ActionType.RAISE,
                                         ActionType.ALL_IN):
                        hs_continue.append(hs)
                    elif a.action_type == ActionType.FOLD:
                        hs_fold.append(hs)
                return a
            return pol

        play_hand({s: mk(s, bots[s]) for s in range(6)}, state=st)
    assert hs_continue and hs_fold
    assert sum(hs_continue) / len(hs_continue) > sum(hs_fold) / len(hs_fold)


# --- the key validation: realized HUD matches target ----------------------------------------------
ASSERTED_TOL = {"vpip": 0.07, "pfr": 0.05, "af": 0.32, "fold_to_cbet": 0.12}


@pytest.mark.parametrize("tier", ["recreational", "weak_reg", "solid_reg"])
def test_realized_hud_matches_target(tier):
    # uses the validation script's measurement harness (the production source of truth)
    from scripts.validate_calibrated_opponents import measure_tier, pooled

    target = tier_target(tier)
    agg = pooled(measure_tier(tier, hands=1500, seed=3))
    realized = agg.realized()
    for stat, tol in ASSERTED_TOL.items():
        assert abs(realized[stat] - target[stat]) <= tol, (
            f"{tier}.{stat}: realized {realized[stat]:.3f} vs target {target[stat]:.3f} "
            f"(tol {tol})")
    # sanity: enough support
    assert agg.cbet_faced > 200
    assert agg.showdowns > 200


def test_knob_cache_is_deterministic():
    # building the same tier twice yields identical solved knobs (cache hit) -> stable bot
    a = build_calibrated_bot("recreational", random.Random(0))
    b = build_calibrated_bot("recreational", random.Random(1))
    assert a._call_vs_bet_hs == b._call_vs_bet_hs
    assert a._bet_freq == b._bet_freq
    assert a._bluff_freq == b._bluff_freq


# --- battery wiring -------------------------------------------------------------------------------
def test_battery_fields_and_builder():
    from pokerbot.eval.exploitation_battery import (
        FIELDS,
        build_calibrated_field_bot,
        field_is_calibrated,
        field_spec,
    )

    for f in ("real_recreational_table", "real_weak_table", "real_solid_table",
              "real_pro_table", "real_mixed_soft", "real_mixed_tough"):
        assert f in FIELDS
        spec = field_spec(f)
        assert len(spec) == 5
        assert field_is_calibrated(f)
    # the calibrated builder resolves tiers AND legacy bots
    assert build_calibrated_field_bot("recreational", random.Random(0)) is not None
    assert build_calibrated_field_bot("hs_station", random.Random(0)) is not None
    # legacy caricature fields are NOT flagged calibrated (additive, non-breaking)
    assert not field_is_calibrated("realistic_table")


def test_calibrate_off_env_skips_sim(monkeypatch):
    # POKERBOT_CALIBRATE_OPP=0 returns default knobs without running the calibration self-play
    monkeypatch.setenv("POKERBOT_CALIBRATE_OPP", "0")
    from pokerbot.eval.calibrated_opponents import _DEFAULT_KNOBS, _calibrated_knobs

    k = _calibrated_knobs(0.5, 0.1, 0.67, 0.37, 0.46, 0.36)
    assert k == _DEFAULT_KNOBS
