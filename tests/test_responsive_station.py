"""Tests for the SIZE-ELASTIC calibrated opponents (pokerbot/eval/responsive_station.py) and the
BLINDNESS REGRESSION that motivates them.

Two things are proven here:

(a) RESPONSIVENESS — the station's continue probability reads the FACING BET SIZE: P(call) is strictly
    DECREASING in ``size_frac`` and stays ABOVE MDF for small/mid sizes (the over-defending payoff fish
    pays off bets past the GTO minimum-defense frequency), while the reg control defends ~MDF (slack=0).

(b) BLINDNESS REGRESSION — a value-SIZING overlay (a hero that escalates its bet size for value hands)
    produces a NON-ZERO, faithful uplift against ResponsiveStation but ~0 against the legacy SIZE-BLIND
    ``CalibratedOpponentBot``. The legacy bot's continue decision ignores size, so the SAME escalation
    that the new field rewards cannot be measured by the old one — proving the new field captures the
    mechanism (the value-sizing/frequency overlay) the old field is blind to. This is the #1 substantive
    fix: today's field is size-blind, so the deployed value-SIZING edge measures as ~0.

No rake / variance-reduction here — these are VILLAINS (rake + VR live on the hero/measurement side).
The measurements use small N on CPU against engine showdowns (net-free, no blueprint/GPU).
"""

from __future__ import annotations

import copy
import os
import random

import pytest

# These villains build on the calibrated postflop knobs. The knob self-calibration (POKERBOT_CALIBRATE_OPP
# default ON) runs a few hundred self-play hands per target vector; for these UNIT tests we only need the
# size-elasticity of the FACING branch (which is overlay logic, not a calibrated knob), so skip the heavy
# calibration self-play with default knobs — keeps the test fast and deterministic.
os.environ.setdefault("POKERBOT_CALIBRATE_OPP", "0")

from pokerbot.engine.actions import ActionType  # noqa: E402
from pokerbot.engine.game import GameState, Street, advance_to_terminal  # noqa: E402
from pokerbot.eval.calibrated_opponents import CalibratedOpponentBot, _hand_strength  # noqa: E402
from pokerbot.eval.responsive_station import (  # noqa: E402
    ResponsiveReg,
    ResponsiveStation,
    build_responsive_bot,
    build_responsive_reg,
    build_responsive_station,
    facing_size_frac,
    mdf,
    p_call,
)

# IRC recreational (loose-passive) HUD target — the modal soft-field player. Passed verbatim to the
# inherited CalibratedOpponentBot calibration so the realized frequencies match the population.
_STATION_HUD = {"vpip": 0.61, "pfr": 0.06, "af": 0.45, "wtsd": 0.42, "wsd": 0.41, "fold_to_cbet": 0.27}

_FRACS = [0.33, 0.5, 0.75, 1.0, 1.5, 2.0]


# --- pure-formula sanity ---------------------------------------------------------------------------
def test_mdf_matches_pot_odds_identity():
    # mdf(size_frac) == pot_odds when size_frac == to_call/pot, i.e. mdf(f) == f/(1+f).
    for f in _FRACS:
        to_call = f * 10.0          # pot = 10, bet = f*pot
        pot_odds = to_call / (10.0 + to_call)
        assert abs(mdf(f) - pot_odds) < 1e-12, (f, mdf(f), pot_odds)
    # monotone increasing in size
    assert all(mdf(_FRACS[i]) < mdf(_FRACS[i + 1]) for i in range(len(_FRACS) - 1))


def test_p_call_decreasing_in_size_and_slack_shifts_up():
    hs = 0.55
    # strictly decreasing in size (bigger bet -> mdf rises -> fewer continues)
    pcs = [p_call(hs, f, slack=0.18) for f in _FRACS]
    assert all(pcs[i] > pcs[i + 1] for i in range(len(pcs) - 1)), pcs
    # slack shifts the whole curve UP (over-defense): station continues more than the reg at every size
    for f in _FRACS:
        assert p_call(hs, f, slack=0.18) >= p_call(hs, f, slack=0.0)


# --- (a) RESPONSIVENESS: realized continue rate reads bet size -------------------------------------
def _realized_continue_rate(make_villain, frac: float, n: int = 3000) -> float:
    """Realized continue (call/raise) frequency of ``make_villain`` facing a bet of ``frac``·pot on the
    river, over ``n`` random deals. Constructs a clean facing spot so size_frac == frac exactly."""
    cont = tot = 0
    for h in range(n):
        st = GameState.new_hand(button_seat=0, rng=random.Random(h))
        st.deal_hole_cards()
        st.board = st._deck.deal(5)
        st.street = Street.RIVER
        for p in st.players:
            p.committed_street_bb = 0.0
        st.pot_bb = 10.0
        st.current_bet_bb = frac * st.pot_bb   # size_frac = to_call/pot = frac
        st.action_seat = 1
        legal = st.legal_actions(1)
        if not any(a.action_type == ActionType.CALL for a in legal):
            continue
        a = make_villain(random.Random(7777 * (h + 1)))(st)
        tot += 1
        if a.action_type in (ActionType.CALL, ActionType.RAISE, ActionType.BET, ActionType.ALL_IN):
            cont += 1
    return cont / tot if tot else float("nan")


def _mk_station(rng):
    return ResponsiveStation(rng, slack=0.18, **_STATION_HUD)


def _mk_reg(rng):
    return ResponsiveReg(rng, slack=0.0, **_STATION_HUD)


def _mk_legacy(rng):
    return CalibratedOpponentBot(rng, **_STATION_HUD)


def test_station_continue_rate_strictly_decreasing_in_size():
    rates = [_realized_continue_rate(_mk_station, f) for f in _FRACS]
    # strictly decreasing across the size grid (allow tiny MC noise via a small slack)
    for i in range(len(rates) - 1):
        assert rates[i] > rates[i + 1] - 0.02, (rates, i)
    # and meaningfully elastic end-to-end (small bet continued far more than a big over-bet)
    assert rates[0] - rates[-1] > 0.20, rates


def test_station_overcalls_past_mdf_small_and_mid():
    # The over-defending payoff fish: at small/mid sizes it continues MORE than the GTO MDF (over-bets
    # are still paid by its wide showdown range). Use small sizes where the over-defense is unambiguous.
    for f in (0.33, 0.5):
        c = _realized_continue_rate(_mk_station, f)
        assert c > mdf(f), (f, c, mdf(f))


def test_reg_defends_near_mdf_not_overcalling():
    # The do-no-harm control: the reg (slack=0) defends ~MDF — it does NOT over-call, so it sits at or
    # below the station's continue rate at every size (and near the MDF baseline, not far above it).
    for f in (0.33, 0.5, 0.75, 1.0):
        c_reg = _realized_continue_rate(_mk_reg, f)
        c_st = _realized_continue_rate(_mk_station, f)
        assert c_reg <= c_st + 1e-6, (f, c_reg, c_st)
        # not wildly over-defending past MDF (a reg is disciplined). A generous ceiling that the
        # over-defending station would BREACH but the reg does not.
        assert c_reg <= mdf(f) + 0.30, (f, c_reg, mdf(f))


def test_legacy_calibrated_bot_is_size_blind():
    # The crux of the blindness claim: the legacy CalibratedOpponentBot's continue rate is INVARIANT to
    # bet size (it never reads to_call/pot). So the elasticity slope is ~0, unlike the station's.
    rates = [_realized_continue_rate(_mk_legacy, f) for f in _FRACS]
    assert max(rates) - min(rates) < 0.05, rates   # flat across all sizes


# --- (b) BLINDNESS REGRESSION: value-SIZING uplift faithful only vs the size-elastic field ----------
# A minimal "cash_hero" value-sizing overlay: ON escalates a value bet from a base size to a bigger size;
# OFF bets the base size. The ONLY difference between the arms is the bet SIZE on value hands. We measure
# the hero's realized chip uplift (ON - OFF) under CRN (same deck + same villain seed per hand).
_OFF_FRAC = 1.0 / 3.0      # base value-bet size
_ON_FRAC = 0.75            # the value-SIZING overlay's escalated size (modest over-bet)
_THIN_LO, _THIN_HI = 0.62, 0.80   # realistic THIN-value band (top pair / weak two pair, not the nuts)


def _hero_policy(frac: float):
    """Hero that value-bets ``frac``·pot on the river when it holds a value hand, else checks/calls."""
    def pol(state: GameState):
        seat = state.action_seat
        legal = state.legal_actions(seat)
        hs = _hand_strength(state, seat)
        types = {a.action_type for a in legal}
        if hs is not None and hs >= _THIN_LO and ActionType.CHECK in types:
            pot = max(state.pot_bb, 1e-9)
            sized = [a for a in legal if a.action_type in (ActionType.BET, ActionType.RAISE)]
            if sized:
                target = frac * pot
                return min(sized, key=lambda a: abs(float(a.amount_bb) - target))
        for t in (ActionType.CHECK, ActionType.CALL, ActionType.FOLD):
            a = next((a for a in legal if a.action_type == t), None)
            if a is not None:
                return a
        return legal[0]
    return pol


def _sizing_overlay_uplift(make_villain, n: int = 6000) -> tuple[float, int]:
    """Hero EV uplift (ON over-bet minus OFF base) per thin-value river spot, heads-up vs the villain,
    under CRN. Real engine showdowns determine win-prob-when-called, so the uplift is faithful (no
    assumed equity). Returns (uplift_bb_per_spot, n_spots)."""
    on_total = off_total = 0.0
    cnt = 0
    for h in range(n):
        st0 = GameState.new_hand(button_seat=0, rng=random.Random(h))
        st0.deal_hole_cards()
        for s in range(2, 6):          # fold everyone except hero (0) and villain (1) -> heads-up
            st0.players[s].folded = True
        st0.board = st0._deck.deal(5)
        st0.street = Street.RIVER
        for p in st0.players:
            p.committed_street_bb = 0.0
        st0.pot_bb = 10.0
        st0.current_bet_bb = 0.0
        st0.action_seat = 0
        hs = _hand_strength(st0, 0)
        if hs is None or not (_THIN_LO <= hs < _THIN_HI):
            continue                   # only THIN-value spots (the discriminating band)
        cnt += 1
        for on, frac in ((True, _ON_FRAC), (False, _OFF_FRAC)):
            st = copy.deepcopy(st0)
            villain = make_villain(random.Random(7777 * (h + 1)))
            pols = {0: _hero_policy(frac), 1: villain}
            for s in range(2, 6):
                pols[s] = lambda state: state.legal_actions()[0]
            _, deltas = advance_to_terminal(st, pols, rng=random.Random(h))
            d = deltas.get(0, 0.0)
            if on:
                on_total += d
            else:
                off_total += d
    return ((on_total - off_total) / cnt if cnt else float("nan")), cnt


def test_blindness_regression_uplift_faithful_only_vs_responsive():
    ul_legacy, n1 = _sizing_overlay_uplift(_mk_legacy)
    ul_station, n2 = _sizing_overlay_uplift(_mk_station)
    assert n1 > 500 and n2 > 500, (n1, n2)   # enough thin-value spots for a stable read
    # vs the SIZE-BLIND legacy bot the value-SIZING escalation prints ~0 (its continue decision ignores
    # size, so a thin over-bet is called by a range matching our thin equity = ~break-even).
    assert abs(ul_legacy) < 0.30, ul_legacy
    # vs the SIZE-ELASTIC station the SAME escalation prints materially (the over-defense pays it off).
    assert ul_station > 0.40, ul_station
    # and the new field captures what the old cannot: the station uplift dwarfs the legacy uplift.
    assert ul_station > ul_legacy + 0.40, (ul_station, ul_legacy)


# --- construction / wiring -------------------------------------------------------------------------
def test_factories_build_and_inherit_calibration():
    st = build_responsive_station(random.Random(0))
    rg = build_responsive_reg(random.Random(0))
    assert isinstance(st, ResponsiveStation) and st.slack > 0.0
    assert isinstance(rg, ResponsiveReg) and rg.slack == 0.0
    # inherits the IRC HUD calibration fields from CalibratedOpponentBot
    for attr in ("vpip", "pfr", "af", "wtsd", "wsd", "fold_to_cbet"):
        assert hasattr(st, attr) and hasattr(rg, attr)
    # build_responsive_bot dispatch
    assert isinstance(build_responsive_bot("responsive_station", random.Random(0)), ResponsiveStation)
    assert isinstance(build_responsive_bot("responsive_reg", random.Random(0)), ResponsiveReg)
    with pytest.raises(ValueError):
        build_responsive_bot("nope", random.Random(0))


def test_params_override_responsiveness_knobs():
    st = build_responsive_station(random.Random(0), {"slack": 0.05, "value_size": 0.8})
    assert st.slack == 0.05
    assert st.value_size == 0.8


def test_always_returns_legal_action():
    bot = build_responsive_station(random.Random(0))
    for h in range(60):
        state = GameState.new_hand(button_seat=h % 6, rng=random.Random(h))
        state.deal_hole_cards()
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


def test_facing_size_frac_none_when_not_facing():
    st = GameState.new_hand(button_seat=0, rng=random.Random(0))
    st.deal_hole_cards()
    st.board = st._deck.deal(5)
    st.street = Street.RIVER
    for p in st.players:
        p.committed_street_bb = 0.0
    st.pot_bb = 10.0
    st.current_bet_bb = 0.0          # no bet to face
    assert facing_size_frac(st, 1) is None
    st.current_bet_bb = 5.0          # half-pot bet faced
    assert abs(facing_size_frac(st, 1) - 0.5) < 1e-9


# --- battery wiring --------------------------------------------------------------------------------
def test_battery_wiring_registers_responsive():
    from pokerbot.eval.bots import build_bot
    from pokerbot.eval.exploitation_battery import (
        FIELDS,
        build_calibrated_field_bot,
        field_is_calibrated,
        field_spec,
    )

    # build_bot + the calibrated field builder both resolve the names
    assert isinstance(build_bot("responsive_station", random.Random(0)), ResponsiveStation)
    assert isinstance(build_calibrated_field_bot("responsive_reg", random.Random(0)), ResponsiveReg)
    # new fields exist, are well-formed, and are flagged calibrated (so a driver uses the right builder)
    for f in ("responsive_station_table", "responsive_reg_table", "responsive_mixed_soft"):
        assert f in FIELDS
        spec = field_spec(f)
        assert len(spec) == 5
        assert field_is_calibrated(f)
    # additive: existing calibrated/legacy fields are untouched
    assert field_is_calibrated("real_recreational_table")
    assert not field_is_calibrated("realistic_table")
