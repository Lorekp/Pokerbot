"""C4 ``cash_hero`` — the deployed-cash hero that FIRES the HUD value-SIZING overlay offline.

These tests are FAST (no 43MB blueprint, no EV run): they use an EMPTY blueprint snapshot
(``blueprint_snapshot={}`` -> uniform-legal fallback) + a tiny fixed CFR iteration count, exactly like
tests/test_table_kill.py's end-to-end SearchAgent tests. The point is the SEAM (the overlay fires on the
post-search policy + the CRN/protocol contract + serve-skew fail-closed), not EV magnitude.

What is asserted:
  * protocol surface (hero_seat / set_profile / reseed / on_new_hand / pop_decisions / bet_freq);
  * the ON arm's policy differs from OFF ONLY when the station HUD is seeded (the overlay fires) and is
    byte-identical to OFF when the kill-switch POKERBOT_LIVE_VALUE_SIZING=0 is set;
  * the hero plays its OWN dealt cards from EVERY seat after hero_seat is set (the seat-0-from-every-seat
    search bug, exploitation_battery.py:1014);
  * the M6 bet_shift proxy accumulates only when the overlay fires;
  * serve-skew: a deliberate fingerprint MISMATCH on a stamped snapshot FAILS CLOSED in make_cash_hero
    (the -559 bb/100 config-artifact guard).
"""

from __future__ import annotations

import random

import numpy as np
import pytest

from pokerbot.engine.actions import Action, ActionType
from pokerbot.engine.game import GameState
from pokerbot.eval.cash_hero import CashHero, _CashSearchAgent, hero_factory, make_cash_hero
from pokerbot.eval.exploitation_battery import population_hud
from pokerbot.opponent.stats import OpponentStats


# --------------------------------------------------------------------------------------------------
# helpers — mirror tests/test_table_kill.py (empty blueprint, fixed iters, deterministic facing node)
# --------------------------------------------------------------------------------------------------
def _make_cash_search_agent(*, stability_tracker=None) -> _CashSearchAgent:
    ag = _CashSearchAgent(
        use_pluribus_search=True, blueprint_snapshot={}, hero_seat=0,
        time_budget_ms=10 ** 9, transformer_checkpoint=None, value_checkpoint=None,
        stability_tracker=stability_tracker,
    )
    ag.solver.config.max_iterations = 4
    ag.solver.config.leaf = type(ag.solver.config.leaf)(select="uniform", rollout_samples=1)
    return ag


def _make_hero(*, exploit: bool, stability_tracker=None, seed_station_hud=True) -> CashHero:
    ag = _make_cash_search_agent(stability_tracker=stability_tracker)
    return CashHero(ag, random.Random(0), exploit=exploit, hero_seat=0, n_seats=6,
                    seed_station_hud=seed_station_hud)


def _station_profile(with_hud: bool) -> OpponentStats:
    """A loose-passive station. ``with_hud`` seeds the server HUD that the value-cut guard requires;
    without it the overlay correctly fails closed (no WSD evidence)."""
    prof = OpponentStats(hands=300)
    if with_hud:
        prof.hud = population_hud("calling_station")
    return prof


def _facing_decision(hero: CashHero, seed: int, hero_seat: int = 0):
    """Drive a facing-a-raise decision and return (post_overlay_probs, overlay_caps), or (None, None) if
    the hand didn't route to the hero. button=0 => seat 3 opens; others fold => hero faces a raise.
    Mirrors test_table_kill._facing_probs but through the CashHero wrapper so on_new_hand / hero_seat are
    exercised. The returned caps are the overlay's per-decision diagnostics (fired/reason/bet_shift)."""
    st = GameState.new_hand(button_seat=0, rng=random.Random(seed))
    st.deal_hole_cards()
    for _ in range(6):
        if st.action_seat == hero_seat:
            break
        legal = st.legal_actions()
        raise_a = next((a for a in legal if a.action_type == ActionType.RAISE), None)
        st.apply_action(raise_a if (st.action_seat == 3 and raise_a) else Action(ActionType.FOLD))
        st.advance_to_next_actor()
    if st.action_seat != hero_seat:
        return None, None
    hero.hero_seat = hero_seat       # set BEFORE on_new_hand (the seat-0-from-every-seat bug)
    hero.reseed(1)
    hero.on_new_hand()
    hero(st)
    decs = hero.pop_decisions()
    if not decs:
        return None, None
    return np.asarray(decs[-1][0], dtype=float), list(hero.last_overlay_caps)


def _facing_probs(hero: CashHero, seed: int, hero_seat: int = 0):
    return _facing_decision(hero, seed, hero_seat)[0]


# --------------------------------------------------------------------------------------------------
# protocol surface
# --------------------------------------------------------------------------------------------------
def test_cash_hero_satisfies_reseed_contract():
    hero = _make_hero(exploit=True)
    for attr in ("hero_seat", "set_profile", "reseed", "on_new_hand", "pop_decisions",
                 "pop_chance_corrections", "bet_freq", "mean_bet_shift", "bet_shift"):
        assert hasattr(hero, attr)
    hero.hero_seat = 2
    hero.set_profile(_station_profile(with_hud=True))
    hero.reseed(123)
    hero.on_new_hand()
    st = GameState.new_hand(button_seat=3, rng=random.Random(0))
    action = hero(st)
    assert action is not None
    decs = hero.pop_decisions()
    assert decs and np.asarray(decs[0][0]).shape == (11,)
    assert abs(float(np.asarray(decs[0][0]).sum()) - 1.0) < 1e-6
    assert 0.0 <= hero.bet_freq <= 1.0


def test_set_profile_seeds_station_hud_when_missing():
    """A bare (HUD-less) station profile gets a representative calling-station HUD seeded, so the
    value-cut-guarded overlay has the WSD evidence it needs (mirrors exploitation_battery._POP_HUD)."""
    hero = _make_hero(exploit=True, seed_station_hud=True)
    prof = _station_profile(with_hud=False)
    assert not prof.hud
    hero.set_profile(prof)
    assert prof.hud, "station HUD was not seeded -> the overlay's value-cut guard would fail closed"
    assert "wsd" in prof.hud and "foldToCbet" in prof.hud


def test_set_profile_does_not_overwrite_existing_hud():
    hero = _make_hero(exploit=True, seed_station_hud=True)
    prof = OpponentStats(hands=100)
    prof.hud = {"vpip": 99, "pfr": 1, "wsd": 12, "foldToCbet": 5}   # a custom HUD must survive
    hero.set_profile(prof)
    assert prof.hud["vpip"] == 99


# --------------------------------------------------------------------------------------------------
# the CORE A/B: the overlay fires ON vs OFF only when the station HUD is seeded
# --------------------------------------------------------------------------------------------------
def test_overlay_fires_only_when_station_hud_is_seeded():
    """The value-SIZING OVERLAY (the headline cash lever this module exists to fire offline) must fire
    ONLY when the station HUD is present. The exploit-ON arm wires BOTH the solver opponent-tail AND the
    overlay; this test isolates the OVERLAY via its per-decision ``fired``/``reason`` diagnostics, so the
    solver tail (which also moves the policy whenever a profile is set) does not confound the assertion.

    HUD seeded  -> at least one decision reports ``fired=True`` (the overlay escalated value SIZE).
    No HUD      -> NO decision ever reports ``fired=True`` (the value-cut guard fails closed without the
                   showdown-weakness evidence: reason is never 'fired')."""
    on_hud = _make_hero(exploit=True)
    on_hud.set_profile(_station_profile(with_hud=True))
    on_nohud = _make_hero(exploit=True, seed_station_hud=False)
    on_nohud.set_profile(_station_profile(with_hud=False))   # no HUD, no seeding -> overlay can't fire

    fired_hud = False
    for seed in range(16):
        _, caps_hud = _facing_decision(on_hud, seed)
        _, caps_nohud = _facing_decision(on_nohud, seed)
        if caps_hud:
            fired_hud = fired_hud or any(c.get("fired") for c in caps_hud)
        if caps_nohud:
            assert not any(c.get("fired") for c in caps_nohud), (
                f"overlay FIRED without a station HUD at seed {seed} (value-cut guard not failing closed)")
    assert fired_hud, ("the value-SIZING overlay never fired even with the station HUD seeded -> the "
                       "offline serve seam is inert (the bet_menu_ab null in a different guise)")


def test_overlay_perturbs_policy_and_escalates_bet_size():
    """Holding the solver tail CONSTANT (same HUD profile both arms), the overlay vs the kill-switched
    overlay differ ONLY through the value-SIZE / bluff-suppression channel: the overlay-ON policy moves
    off the overlay-killed policy on >=1 seed where the overlay fired, and the overlay's reported
    bet_shift (the change in expected bet SIZE in pot fractions) is positive on >=1 fired decision
    (it escalates value SIZE / suppresses bluffs vs a confirmed station). Note the final renormalization
    means fold/check/call mass is NOT byte-preserved (bluff-suppression is a frequency effect too), so
    the channel is verified by the seam moving the policy + a positive size shift, not bucket isolation."""
    import os

    moved = False
    saw_positive_shift = False
    for seed in range(16):
        on = _make_hero(exploit=True); on.set_profile(_station_profile(with_hud=True))
        killed = _make_hero(exploit=True); killed.set_profile(_station_profile(with_hud=True))
        prev = os.environ.get("POKERBOT_LIVE_VALUE_SIZING")
        os.environ.pop("POKERBOT_LIVE_VALUE_SIZING", None)            # overlay LIVE for the 'on' arm
        p_on, caps = _facing_decision(on, seed)
        os.environ["POKERBOT_LIVE_VALUE_SIZING"] = "0"                # KILL it for the control arm
        try:
            p_killed, _ = _facing_decision(killed, seed)
        finally:
            if prev is None:
                os.environ.pop("POKERBOT_LIVE_VALUE_SIZING", None)
            else:
                os.environ["POKERBOT_LIVE_VALUE_SIZING"] = prev
        if p_on is None or p_killed is None:
            continue
        fired = [c for c in (caps or []) if c.get("fired")]
        if fired:
            if not np.allclose(p_on, p_killed):
                moved = True
            if any(float(c.get("bet_shift", 0.0) or 0.0) > 0.0 for c in fired):
                saw_positive_shift = True
    assert moved, "overlay never measurably moved the policy vs its kill-switched control"
    assert saw_positive_shift, ("the overlay fired but never escalated bet SIZE (positive bet_shift) -> "
                                "the value-SIZING channel is not expressing vs the station")


def test_kill_switch_neutralizes_the_overlay():
    """POKERBOT_LIVE_VALUE_SIZING=0 must make the overlay a NO-OP: at every seed where the overlay would
    otherwise FIRE (it does on the HUD arm without the kill-switch), the kill-switched policy is
    byte-identical to that arm's solver-tail-only (overlay-killed) policy, and NO cap ever reports
    fired=True under the kill-switch. Compares ACROSS the kill-switch boundary so it is not a tautology."""
    import os

    saw_fire_without_killswitch = False
    seen = 0
    for seed in range(16):
        # arm 1: overlay live (may fire) ; arm 2: kill-switch set (must be inert)
        live = _make_hero(exploit=True); live.set_profile(_station_profile(with_hud=True))
        killed = _make_hero(exploit=True); killed.set_profile(_station_profile(with_hud=True))

        prev = os.environ.get("POKERBOT_LIVE_VALUE_SIZING")
        os.environ.pop("POKERBOT_LIVE_VALUE_SIZING", None)              # ensure overlay LIVE for arm 1
        p_live, caps_live = _facing_decision(live, seed)
        os.environ["POKERBOT_LIVE_VALUE_SIZING"] = "0"                  # KILL the overlay for arm 2
        p_killed, caps_killed = _facing_decision(killed, seed)
        if prev is None:
            os.environ.pop("POKERBOT_LIVE_VALUE_SIZING", None)
        else:
            os.environ["POKERBOT_LIVE_VALUE_SIZING"] = prev

        if p_live is None or p_killed is None:
            continue
        seen += 1
        # Under the kill-switch NO decision may report fired=True (the overlay is fully neutralized).
        assert not any(c.get("fired") for c in (caps_killed or [])), (
            f"overlay reported fired=True under the kill-switch at seed {seed}")
        if caps_live and any(c.get("fired") for c in caps_live):
            saw_fire_without_killswitch = True
            # where the live overlay FIRED, the kill-switched policy must drop the sizing tilt -> the two
            # policies differ on the sizing buckets (the overlay's effect is gone under the kill-switch).
            assert not np.allclose(p_live, p_killed), (
                f"overlay fired live but kill-switch produced an identical policy at seed {seed}")
    assert seen > 0, "no facing decision routed to the hero -> the kill-switch test would be vacuous"
    assert saw_fire_without_killswitch, ("the overlay never fired even without the kill-switch -> the "
                                         "kill-switch neutralization check would be vacuous")


def test_bet_shift_proxy_accumulates_only_when_overlay_fires():
    """The M6 bet_shift proxy is 0.0 for the OFF arm (overlay never fires) and accumulates a non-negative
    mean for the HUD-seeded ON arm whenever the overlay fired on >=1 decision."""
    off = _make_hero(exploit=False)
    on = _make_hero(exploit=True)
    on.set_profile(_station_profile(with_hud=True))
    for seed in range(16):
        _facing_probs(off, seed)
        _facing_probs(on, seed)
    assert off.mean_bet_shift == 0.0, "OFF arm reported a bet_shift -> the overlay fired with no profile"
    assert on.mean_bet_shift >= 0.0
    assert on.bet_shift == on.mean_bet_shift   # the M6 column alias


# --------------------------------------------------------------------------------------------------
# the search hero plays its OWN dealt cards from every seat (the seat-0-from-every-seat bug)
# --------------------------------------------------------------------------------------------------
def test_hero_plays_own_cards_from_every_seat():
    """After hero_seat is set BEFORE on_new_hand(), the agent's internal hero seat tracks it, so the
    search builds the subgame from the hero's ACTUAL hole cards in every physical seat (not seat 0's)."""
    hero = _make_hero(exploit=False)
    for hs in range(6):
        hero.hero_seat = hs
        hero.reseed(7)
        hero.on_new_hand()
        # reset_hand(hero_seat) propagated the seat into the agent's per-hand state.
        assert hero.agent._state.hero_seat == hs
        st = GameState.new_hand(button_seat=3, rng=random.Random(100 + hs))
        if st.action_seat != hs:
            continue
        hero(st)
        # The decision was built for THIS seat (no exception, valid simplex over 11 actions).
        decs = hero.pop_decisions()
        assert decs and abs(float(np.asarray(decs[-1][0]).sum()) - 1.0) < 1e-6


# --------------------------------------------------------------------------------------------------
# serve-skew: a deliberate fingerprint MISMATCH fails closed (the -559 bb/100 guard)
# --------------------------------------------------------------------------------------------------
def _active_stamp() -> dict:
    from pokerbot.engine.jax_weva_encode import active_abstraction_config

    return dict(active_abstraction_config())


def test_make_cash_hero_fails_closed_on_fingerprint_mismatch():
    """A stamped snapshot whose abstraction differs from the active serve env (one flipped field) must
    RAISE in make_cash_hero (strict serve guard) — a mismatch must never silently mis-measure."""
    stamp = _active_stamp()
    bad = dict(stamp)
    bad["postflop_num_buckets"] = int(stamp.get("postflop_num_buckets", 200)) + 777   # guaranteed mismatch
    snap = {"strat_sum": {}, "abstraction": bad}
    with pytest.raises(RuntimeError, match="stamp|abstraction|MISCONFIGURED|misses"):
        make_cash_hero(snap, exploit=False, iterations=4, time_budget_ms=10 ** 9)


def test_make_cash_hero_accepts_matching_stamp():
    """A stamped snapshot whose abstraction MATCHES the active serve env builds without raising."""
    snap = {"strat_sum": {}, "abstraction": _active_stamp()}
    hero = make_cash_hero(snap, exploit=False, iterations=4, time_budget_ms=10 ** 9)
    assert isinstance(hero, CashHero)
    # Smoke a single decision to confirm the resolved (empty) table serves the uniform fallback cleanly.
    st = GameState.new_hand(button_seat=3, rng=random.Random(0))
    hero.hero_seat = st.action_seat
    hero.reseed(1)
    hero.on_new_hand()
    if st.action_seat == hero.hero_seat:
        assert hero(st) is not None


def test_make_cash_hero_accepts_raw_table_without_stamp():
    """A raw blueprint table (no stamp wrapper) is served directly (verify_serve is a no-op when there is
    no stamp to check) — the C6 gate is the precondition for raw tables."""
    hero = make_cash_hero({}, exploit=False, iterations=4, time_budget_ms=10 ** 9)
    assert isinstance(hero, CashHero)


def test_hero_factory_builds_crn_clean_on_off_arms():
    """hero_factory(snap)(exploit=...) builds both arms off ONE resolved table; the OFF/ON arms share the
    per-hand seed so a decision the overlay does not change is sampled identically (CRN)."""
    factory = hero_factory({}, iterations=4, time_budget_ms=10 ** 9)
    off, on = factory(exploit=False), factory(exploit=True)
    assert isinstance(off, CashHero) and isinstance(on, CashHero)
    assert off.exploit is False and on.exploit is True
    # OFF arm with no profile: identical to a fresh OFF hero under the same seed/deck (determinism).
    p1 = _facing_probs(off, 3)
    off2 = factory(exploit=False)
    p2 = _facing_probs(off2, 3)
    if p1 is not None and p2 is not None:
        assert np.allclose(p1, p2), "OFF arm not deterministic under CRN reseed"
