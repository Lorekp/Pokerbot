"""Workstream F: the 6-max NLHE Restricted-Nash-Response (RNR) exploitation cap.

Three layers of guarantees:

  * PARITY — the non-AoF reimplementation (``pokerbot.opponent.rnr_cap``) reproduces the AoF math
    (``aof_cfr.exploit_router`` cap formula + ``aof_cfr.adaptive`` dual-EWMA stability) byte-for-byte
    over the AoF domain (n in {2, 3, 4}). The AoF module is imported READ-ONLY here, for comparison
    only — the live module never imports it (firewall).
  * BEHAVIOR — a stable read commits to the ceiling; a caught adapter clamps to CAP_FLOOR; the
    ceiling is monotone non-increasing in N; effective_alpha never exceeds cap_eff.
  * WIRING — root_mixer.apply_live_value_sizing caps alpha at the calibrated 0.5 ceiling at n=6,
    allows up to 1.0 at n=2, and an adapting villain clamps it to the 0.25 floor.

The 5/6-max ceilings (EXPLOIT_CAP_BY_N[5]=0.55, [6]=0.50) are CALIBRATED by the 6-max NLHE
counter-exploiter battery (scripts/hud_sizing_acceptance.py, artifacts/hud_sizing_acceptance.json):
bounded below the 4-max 0.6 and strictly decreasing in N. Raising 2/3-max 0.7 -> 1.0 is an intended
RNR change (minimax-safe).
"""

from __future__ import annotations

import numpy as np
import pytest

from pokerbot.opponent import rnr_cap


# ----------------------------------------------------------------------------------------------
# PARITY vs the AoF module (read-only import, for number comparison only).
# ----------------------------------------------------------------------------------------------
def _aof_available() -> bool:
    try:
        import aof_cfr.adaptive  # noqa: F401
        import aof_cfr.exploit_router  # noqa: F401
        return True
    except Exception:
        return False


_HAVE_AOF = _aof_available()
aof_only = pytest.mark.skipif(not _HAVE_AOF, reason="aof_cfr not importable for parity comparison")


@aof_only
@pytest.mark.parametrize("n", [2, 3, 4])
@pytest.mark.parametrize("stability", [0.0, 0.25, 0.5, 0.75, 1.0])
def test_cap_eff_parity_with_aof(n, stability):
    """cap_eff(stability, n) reproduces the AoF cap formula floor + (ceil - floor)*stability."""
    from aof_cfr.exploit_router import CAP_FLOOR as A_FLOOR
    from aof_cfr.exploit_router import default_exploit_cap as a_ceil

    aof = A_FLOOR + (a_ceil(n) - A_FLOOR) * stability
    assert rnr_cap.cap_eff(stability, n) == pytest.approx(aof)


@aof_only
@pytest.mark.parametrize("n", [2, 3, 4])
def test_default_exploit_cap_parity(n):
    from aof_cfr.exploit_router import default_exploit_cap as a_ceil

    assert rnr_cap.default_exploit_cap(n) == pytest.approx(a_ceil(n))


@aof_only
@pytest.mark.parametrize("n", [2, 3, 4])
@pytest.mark.parametrize("stability,confidence", [(1.0, 0.9), (0.0, 0.9), (0.5, 0.3), (1.0, 0.05)])
def test_effective_alpha_parity(n, stability, confidence):
    """effective_alpha = min(confidence, cap_eff) reproduces the AoF c_play = min(conf, cap_eff)."""
    from aof_cfr.exploit_router import CAP_FLOOR as A_FLOOR
    from aof_cfr.exploit_router import EXPLOIT_CAP as A_CAP
    from aof_cfr.exploit_router import default_exploit_cap as a_ceil

    aof_cap_eff = A_FLOOR + (a_ceil(n) - A_FLOOR) * stability
    aof_c_play = min(confidence, aof_cap_eff)
    assert rnr_cap.effective_alpha(confidence, n, stability) == pytest.approx(aof_c_play)
    # The AoF EXPLOIT_CAP constant (1.0) == our heads-up ceiling.
    assert A_CAP == pytest.approx(rnr_cap.default_exploit_cap(2))


@aof_only
def test_dual_ewma_stability_parity_with_aof():
    """The dual-EWMA opponent_stability reproduces aof_cfr.adaptive on an identical signal sequence.

    Drive both with the SAME 0/1 sequence at ONE infoset. The AoF model keys per (identity, infoset);
    our tracker keys per identity with one aggregate cell — identical when only one infoset is fed."""
    from aof_cfr.adaptive import AdaptiveOpponentModel, PopulationPrior

    # A gear flip: 40 pushes then 40 folds (the same pattern the AoF cap test uses).
    seq = [1.0] * 40 + [0.0] * 40

    aof = AdaptiveOpponentModel(prior=PopulationPrior())
    ours = rnr_cap.StabilityTracker()
    for x in seq:
        aof.observe("villain", 3, "BB", "FA", 8.0, pushed=bool(x))
        ours.observe("villain", x)

    assert ours.opponent_stability("villain") == pytest.approx(aof.opponent_stability("villain"))
    # Sanity: the flip was actually detected (stability strictly below 1).
    assert ours.opponent_stability("villain") < 1.0


@aof_only
def test_dual_ewma_stable_sequence_parity():
    """A constant signal => no divergence => stability 1.0 in both implementations."""
    from aof_cfr.adaptive import AdaptiveOpponentModel, PopulationPrior

    aof = AdaptiveOpponentModel(prior=PopulationPrior())
    ours = rnr_cap.StabilityTracker()
    for _ in range(80):
        aof.observe("rock", 3, "BB", "FA", 8.0, pushed=False)
        ours.observe("rock", 0.0)
    assert ours.opponent_stability("rock") == pytest.approx(aof.opponent_stability("rock"))
    assert ours.opponent_stability("rock") == pytest.approx(1.0)


# ----------------------------------------------------------------------------------------------
# BEHAVIOR (no AoF dependency).
# ----------------------------------------------------------------------------------------------
def test_stable_read_commits_to_ceiling():
    """Stability 1.0 -> cap_eff == the full N-ceiling (a static leak has no downside)."""
    for n in (2, 3, 4, 5, 6):
        assert rnr_cap.cap_eff(1.0, n) == pytest.approx(rnr_cap.default_exploit_cap(n))


def test_adapting_read_clamps_to_floor():
    """Stability 0.0 -> cap_eff == CAP_FLOOR for every N (a caught adapter is clamped hard)."""
    for n in (2, 3, 4, 5, 6):
        assert rnr_cap.cap_eff(0.0, n) == pytest.approx(rnr_cap.CAP_FLOOR)


def test_ceiling_monotone_non_increasing_in_n():
    caps = [rnr_cap.default_exploit_cap(n) for n in (2, 3, 4, 5, 6)]
    for a, b in zip(caps, caps[1:]):
        assert b <= a + 1e-12


def test_effective_alpha_never_exceeds_cap_eff():
    rng = np.random.default_rng(0)
    for _ in range(500):
        conf = float(rng.uniform(0, 1.5))
        stab = float(rng.uniform(0, 1))
        n = int(rng.integers(2, 8))
        ea = rnr_cap.effective_alpha(conf, n, stab)
        assert ea <= rnr_cap.cap_eff(stab, n) + 1e-12
        assert ea <= conf + 1e-12
        assert ea >= 0.0


def test_n_clamping():
    assert rnr_cap.default_exploit_cap(1) == pytest.approx(rnr_cap.default_exploit_cap(2))  # n<2 -> 1.0
    assert rnr_cap.default_exploit_cap(9) == pytest.approx(rnr_cap.default_exploit_cap(6))  # n>6 -> 6val
    assert rnr_cap.default_exploit_cap(2) == pytest.approx(1.0)
    assert rnr_cap.default_exploit_cap(6) == pytest.approx(0.5)  # CALIBRATED (6-max counter-exploiter)


def test_calibrated_56_ceilings_are_bounded_and_decreasing():
    """The 5/6-max ceilings are CALIBRATED (scripts/hud_sizing_acceptance.py, 2026-06-21): 5-max=0.55,
    6-max=0.50. They must stay <= the legacy 0.7 cap (never raise exploitability vs the old flat
    behaviour), <= the 4-max ceiling 0.6, and STRICTLY DECREASING in N (less minimax safety multiway)."""
    assert rnr_cap.EXPLOIT_CAP_BY_N[5] == pytest.approx(0.55)
    assert rnr_cap.EXPLOIT_CAP_BY_N[6] == pytest.approx(0.50)
    assert rnr_cap.EXPLOIT_CAP_BY_N[5] <= 0.7 and rnr_cap.EXPLOIT_CAP_BY_N[6] <= 0.7
    assert rnr_cap.EXPLOIT_CAP_BY_N[5] <= rnr_cap.EXPLOIT_CAP_BY_N[4] + 1e-12   # <= 4-max
    assert rnr_cap.EXPLOIT_CAP_BY_N[6] < rnr_cap.EXPLOIT_CAP_BY_N[5]            # strictly decreasing


def test_unknown_identity_is_stable():
    t = rnr_cap.StabilityTracker()
    assert t.opponent_stability("never-seen") == pytest.approx(1.0)
    assert t.opponent_stability("") == pytest.approx(1.0)
    assert t.opponent_stability(None) == pytest.approx(1.0)


def test_tracker_catches_gear_flip_and_decays():
    t = rnr_cap.StabilityTracker()
    for _ in range(40):
        t.observe("v", 1.0)
    for _ in range(40):
        t.observe("v", 0.0)
    flipped = t.opponent_stability("v")
    assert flipped < 1.0
    # Hysteresis: keeping the new (now-stable) behaviour decays instability slowly back up.
    for _ in range(300):
        t.observe("v", 0.0)
    assert t.opponent_stability("v") > flipped


# ----------------------------------------------------------------------------------------------
# WIRING into root_mixer.apply_live_value_sizing.
# ----------------------------------------------------------------------------------------------
def _station_hud():
    from pokerbot.opponent.stats import OpponentStats

    s = OpponentStats()
    s.hands = 0
    s.hud = {"wsd": 40, "foldToCbet": 25, "wtsd": 46, "vpip": 62, "pfr": 9}
    return s


def _state(hole, board, *, n_seats=6, hero=1, folded=()):
    from pokerbot.engine.cards import Card
    from pokerbot.engine.game import GameState, PlayerState, Position, Street

    positions = [Position.BTN, Position.SB, Position.BB, Position.UTG, Position.HJ, Position.CO]
    st = GameState()
    st.players = [PlayerState(seat=i, position=positions[i], stack_bb=100.0) for i in range(n_seats)]
    for s in folded:
        st.players[s].folded = True
    st.button_seat = 0
    st.street = Street.FLOP if len(board) == 3 else (Street.TURN if len(board) == 4 else Street.RIVER)
    st.board = [Card.from_str(c) for c in board]
    st.pot_bb = 6.0
    st.current_bet_bb = 0.0
    st.action_seat = hero
    st.players[hero].hole = (Card.from_str(hole[0]), Card.from_str(hole[1]))
    return st


def _alpha_used(monkeypatch, st, hero, profiles, **kwargs):
    """Fire apply_live_value_sizing and capture the (alpha, max_alpha) passed to safe_mix_policies."""
    from pokerbot.opponent import root_mixer

    monkeypatch.setenv("POKERBOT_LIVE_VALUE_SIZING", "1")
    monkeypatch.setenv("POKERBOT_VALUE_SIZING", "1")
    monkeypatch.setenv("POKERBOT_BLUFF_SUPPRESS", "1")
    monkeypatch.setattr(root_mixer, "_all_live_villains_fish", lambda *a, **k: True)
    monkeypatch.setattr(root_mixer, "_table_too_aggressive", lambda *a, **k: False)

    captured = {}
    import pokerbot.opponent.mixer as mixer_mod

    real_mix = mixer_mod.safe_mix_policies

    def _spy(gto, logits, alpha, max_alpha=0.7):
        captured["alpha"] = float(alpha)
        captured["max_alpha"] = float(max_alpha)
        return real_mix(gto, logits, alpha, max_alpha=max_alpha)

    monkeypatch.setattr(mixer_mod, "safe_mix_policies", _spy)

    from pokerbot.training.self_play import _legal_mask

    legal = np.asarray(_legal_mask(st, hero), dtype=bool)
    probs = legal.astype(float) / legal.sum()
    out = root_mixer.apply_live_value_sizing(st, hero, probs.copy(), profiles, **kwargs)
    return captured, probs, out


def test_wiring_n6_caps_at_calibrated_half(monkeypatch):
    """At a full 6-handed table the value-sizing overlay caps alpha at the calibrated 0.5 RNR
    ceiling — MORE conservative than the legacy flat 0.7."""
    st = _state(("As", "Ah"), ("2c", "7d", "Qh"), n_seats=6)
    profiles = {s: _station_hud() for s in range(6) if s != 1}
    cap, probs, out = _alpha_used(monkeypatch, st, 1, profiles)
    assert cap["max_alpha"] == pytest.approx(0.5)
    assert cap["alpha"] <= 0.5 + 1e-9
    assert not np.allclose(out, probs)            # it still fired


def test_wiring_n2_allows_up_to_one(monkeypatch):
    """Heads-up (2 live seats) the RNR ceiling is 1.0 (>0.7) — the intended minimax-safe RNR change."""
    # 6 seats but 4 folded -> 2 live (hero + 1 villain).
    st = _state(("As", "Ah"), ("2c", "7d", "Qh"), n_seats=6, hero=1, folded=(0, 3, 4, 5))
    profiles = {s: _station_hud() for s in range(6) if s != 1}
    cap, probs, out = _alpha_used(monkeypatch, st, 1, profiles)
    assert cap["max_alpha"] == pytest.approx(1.0)


def test_wiring_adapting_villain_clamps_to_floor(monkeypatch):
    """An adapting villain (stability 0) clamps the ceiling to CAP_FLOOR (0.25) regardless of N."""
    st = _state(("As", "Ah"), ("2c", "7d", "Qh"), n_seats=6, hero=1)
    # last_aggressor identifies the read villain; the tracker reports it as a caught adapter.
    st.last_aggressor = 0
    profiles = {s: _station_hud() for s in range(6) if s != 1}

    class _AdaptingTracker:
        def opponent_stability(self, key):
            return 0.0

    cap, probs, out = _alpha_used(monkeypatch, st, 1, profiles,
                                  stability_tracker=_AdaptingTracker(), villain_key=0)
    assert cap["max_alpha"] == pytest.approx(rnr_cap.CAP_FLOOR)
    assert cap["alpha"] <= rnr_cap.CAP_FLOOR + 1e-9


def test_wiring_explicit_max_alpha_override(monkeypatch):
    """Passing max_alpha explicitly overrides the RNR ceiling (back-compat / tests)."""
    st = _state(("As", "Ah"), ("2c", "7d", "Qh"), n_seats=6, hero=1)
    profiles = {s: _station_hud() for s in range(6) if s != 1}
    cap, probs, out = _alpha_used(monkeypatch, st, 1, profiles, max_alpha=0.7)
    assert cap["max_alpha"] == pytest.approx(0.7)


def test_wiring_gift_relaxes_base_but_not_past_rnr(monkeypatch):
    """A banked gift relaxes the legacy base cap toward the RNR ceiling but NEVER past it (n=6 -> 0.5)."""
    from pokerbot.opponent.safety_budget import GiftRegister

    st = _state(("As", "Ah"), ("2c", "7d", "Qh"), n_seats=6, hero=1)
    st.last_aggressor = 0
    profiles = {s: _station_hud() for s in range(6) if s != 1}
    reg = GiftRegister()
    for _ in range(50):            # bank a large gift -> would push base 0.7 well up toward 0.85
        reg.credit(0, 5.0)
    cap, probs, out = _alpha_used(monkeypatch, st, 1, profiles, gift_register=reg, villain_key=0)
    # Even a fully-banked gift cannot push the n=6 ceiling past the RNR bound 0.5.
    assert cap["max_alpha"] <= 0.5 + 1e-9
