"""Tests for the cap-changedetect remediation group (audit wiup89smb refresh): EG-10, EG-13, EG-16.

  - EG-10: outcome-based exploit-abandon hook feeding the RNR cap (bayes_exploit).
  - EG-13: n_live_players' sit-out exclusion made correct against the REAL live state object
           (per-seat status string + sitting_out bool), and shown inert against the engine PlayerState.
  - EG-16: fish_relaxed_n env-parse aligned to the codebase OFF-set convention + except-fallback.
"""

from __future__ import annotations

import pytest

from pokerbot.engine.game import GameState, PlayerState, Position
from pokerbot.opponent import bayes_exploit
from pokerbot.opponent.bayes_exploit import (
    OUTCOME_ABANDON_BB,
    outcome_stability,
    profile_stability,
    update_exploit_outcome,
)
from pokerbot.opponent.rnr_cap import _seat_is_live, fish_relaxed_n, n_live_players
from pokerbot.opponent.stats import OpponentStats


# ============================ EG-10: outcome-based exploit-abandon =================================

def test_update_exploit_outcome_accumulates_ewma():
    """update_exploit_outcome folds observations into a dual-mass/sum EWMA, creating the dict when None."""
    acc = update_exploit_outcome(None, 2.0)
    assert acc == {"w": 1.0, "s": 2.0}
    acc = update_exploit_outcome(acc, 2.0)
    # w = 1*0.9 + 1 = 1.9 ; s = 2*0.9 + 2 = 3.8 -> running mean stays 2.0
    assert acc["w"] == pytest.approx(1.9)
    assert acc["s"] == pytest.approx(3.8)
    assert acc["s"] / acc["w"] == pytest.approx(2.0)


def test_outcome_stability_none_and_low_evidence_is_one():
    """No accumulator / too little effective mass -> 1.0 (trust the read; outcome imposes no clamp)."""
    assert outcome_stability(None) == 1.0
    assert outcome_stability({}) == 1.0
    # one big loss but mass < OUTCOME_MIN_EVIDENCE -> not yet trusted
    assert outcome_stability({"w": 1.0, "s": -5.0}) == 1.0


def test_outcome_stability_winning_deviation_no_clamp():
    """A winning / break-even realized deviation never clamps (and never raises above 1.0)."""
    acc = None
    for _ in range(40):
        acc = update_exploit_outcome(acc, +0.5)   # consistently winning the exploit line
    assert outcome_stability(acc) == 1.0


def test_outcome_stability_losing_deviation_abandons():
    """A losing realized deviation (the 'fish' beats our over-extraction) drops stability toward 0."""
    acc = None
    for _ in range(40):
        acc = update_exploit_outcome(acc, -OUTCOME_ABANDON_BB)   # mean loss at the abandon threshold
    assert outcome_stability(acc) == pytest.approx(0.0, abs=1e-9)
    # a partial loss lands proportionally between 1 and 0
    acc2 = None
    for _ in range(40):
        acc2 = update_exploit_outcome(acc2, -0.5 * OUTCOME_ABANDON_BB)
    assert outcome_stability(acc2) == pytest.approx(0.5, abs=1e-6)


def test_profile_stability_without_accumulator_is_byte_identical():
    """No exploit_outcome field on the stats object -> outcome multiplier 1.0 -> frequency-only result."""
    # a profile with NO contextual data returns 1.0 today and must stay 1.0.
    assert profile_stability(OpponentStats(hands=10)) == 1.0


def test_profile_stability_outcome_abandons_a_stationary_trap():
    """EG-10 core: a STATIONARY line (frequency stability ~1.0) still collapses when the deviation loses.

    We feed an exploit_outcome accumulator straight onto the stats object (the DEFERRED live feed will
    populate exactly this field). The frequency change-detector cannot see the trap, but the realized-EV
    contradiction does.
    """
    losing = {"w": 0.0, "s": 0.0}
    for _ in range(40):
        losing = update_exploit_outcome(losing, -OUTCOME_ABANDON_BB)
    stats = OpponentStats(hands=200)
    # attach the (deferred) accumulator the way the live join will
    stats.exploit_outcome = losing  # type: ignore[attr-defined]
    # no ctx data -> frequency path returns the early-exit; the outcome multiplier now drives it to ~0.
    assert profile_stability(stats) == pytest.approx(0.0, abs=1e-9)
    # and a winning deviation leaves the stationary read at full stability
    winning = {"w": 0.0, "s": 0.0}
    for _ in range(40):
        winning = update_exploit_outcome(winning, +OUTCOME_ABANDON_BB)
    stats.exploit_outcome = winning  # type: ignore[attr-defined]
    assert profile_stability(stats) == 1.0


def test_profile_stability_product_with_frequency_signal():
    """When BOTH a frequency divergence AND a losing outcome are present they compound (product)."""
    # Build a profile with a real fast/slow divergence so the frequency path returns < 1.0.
    from pokerbot.opponent.context import NUM_BINS
    stats = OpponentStats(hands=200)
    # slow read concentrated in bin 0; fast read PARTIALLY shifted (10% mass) -> small TV (0.1) ->
    # freq stability strictly between 0 and 1 (1 - 0.1/CHANGE_TV(0.2) = 0.5).
    key = "flop|ip|single|facing_bet"
    slow = [0.0] * NUM_BINS
    fast = [0.0] * NUM_BINS
    slow[0] = 50.0
    fast[0] = 45.0
    fast[1] = 5.0
    stats.ctx_slow = {key: slow}
    stats.ctx_fast = {key: fast}
    freq_only = profile_stability(stats)
    assert 0.0 < freq_only < 1.0      # the gear-change alone already partially drops it
    # now add a half-abandon outcome and confirm the result is the product (strictly lower).
    half = {"w": 0.0, "s": 0.0}
    for _ in range(40):
        half = update_exploit_outcome(half, -0.5 * OUTCOME_ABANDON_BB)
    stats.exploit_outcome = half  # type: ignore[attr-defined]
    combined = profile_stability(stats)
    assert combined == pytest.approx(freq_only * 0.5, abs=1e-6)
    assert combined < freq_only


# ============================ EG-13: n_live_players sit-out exclusion ==============================

class _StatusSeat:
    """A minimal live-seat stand-in exposing the REAL signal: a per-seat status string."""

    def __init__(self, status: str = "active", folded: bool = False):
        self.status = status
        self.folded = folded


class _StateLike:
    def __init__(self, players):
        self.players = players


def test_seat_is_live_status_string():
    """_seat_is_live honors the per-seat status string the live builders produce."""
    assert _seat_is_live(_StatusSeat("active"))
    assert _seat_is_live(_StatusSeat("all_in"))      # all-in still contests the pot
    assert not _seat_is_live(_StatusSeat("sitting_out"))
    assert not _seat_is_live(_StatusSeat("folded"))
    assert not _seat_is_live(_StatusSeat("empty"))
    assert not _seat_is_live(_StatusSeat("waiting"))
    assert not _seat_is_live(_StatusSeat("disconnected"))
    # case / whitespace robust
    assert not _seat_is_live(_StatusSeat("  Sitting_Out "))


def test_n_live_players_excludes_sitting_out_via_status():
    """EG-13: 6 dealt seats, 4 sat-out (status string) -> N collapses to 2 (the HU ceiling input)."""
    players = [_StatusSeat("active"), _StatusSeat("active")] + [_StatusSeat("sitting_out")] * 4
    assert n_live_players(_StateLike(players), hero=0) == 2


def test_n_live_players_excludes_sitting_out_via_bool():
    """The legacy synthetic ``sitting_out`` boolean path still excludes (back-compat)."""
    class _BoolSeat:
        def __init__(self, sitting_out=False, folded=False):
            self.sitting_out = sitting_out
            self.folded = folded
    players = [_BoolSeat(), _BoolSeat()] + [_BoolSeat(sitting_out=True)] * 4
    assert n_live_players(_StateLike(players), hero=0) == 2


def test_n_live_players_never_below_two_and_counts_active():
    players = [_StatusSeat("active")] * 5 + [_StatusSeat("folded")]
    assert n_live_players(_StateLike(players), hero=0) == 5
    # all sat out -> floored at 2 (never returns < 2)
    assert n_live_players(_StateLike([_StatusSeat("sitting_out")] * 6), hero=0) == 2


def test_n_live_players_inert_against_real_engine_playerstate():
    """EG-13 honesty: the REAL engine PlayerState carries NO sit-out field, so against a freshly built
    GameState every dealt seat reads live (conservative — clamps N HIGHER, never the over-extraction
    direction). Populating ``status`` on the live PlayerState is DEFERRED to the builder owners."""
    state = GameState()
    state.players = [PlayerState(seat=i, position=Position.BTN) for i in range(6)]
    # no status / sitting_out attribute on PlayerState -> all six count as live.
    assert not hasattr(state.players[0], "status")
    assert not hasattr(state.players[0], "sitting_out")
    assert n_live_players(state, hero=0) == 6


# ============================ EG-16: fish_relaxed_n env parse ======================================

def _value_fish():
    # loose-passive HUD that _is_value_fish admits and _is_dangerous_aggressor does not flag.
    return OpponentStats(hands=30, hud={"vpip": 58, "pfr": 9, "wtsd": 33})


@pytest.mark.parametrize(
    "env_val,expect_relaxed",
    [
        ("1", True),
        ("true", True),
        ("on", True),
        ("yes", True),
        (" 1", True),       # leading whitespace stripped -> still enables (audit repro)
        ("TRUE", True),     # upper-case enables (lowercased convention)
        ("On", True),
        ("0", False),
        ("", False),
        ("false", False),
        ("False", False),
        ("FALSE", False),   # was MIS-handled by the old non-lowercased 4-string match
        ("no", False),      # codebase OFF-set (root_mixer.py:538) — was treated as ON before
        ("off", False),
        ("Off", False),
    ],
)
def test_fish_relax_env_parse_convention(monkeypatch, env_val, expect_relaxed):
    """EG-16: the env parse follows the codebase OFF-set convention (lowercase + {0,'',false,no,off})."""
    monkeypatch.setenv("POKERBOT_RNR_FISH_RELAX", env_val)
    model = {s: _value_fish() for s in range(1, 6)}   # 5 confirmed fish -> no punisher
    result = fish_relaxed_n(6, model)
    assert result == (2 if expect_relaxed else 6)


def test_fish_relax_except_fallback_is_conservative(monkeypatch):
    """EG-16: a profile that RAISES inside the fish-check must fall back to the conservative n, never 2.

    Force the classification to raise (the try/except around the relaxation guards exactly this) and
    assert the conservative N is returned even with the flag ON.
    """
    monkeypatch.setenv("POKERBOT_RNR_FISH_RELAX", "1")

    class _Boom:
        """A profile-like object whose attribute access blows up inside _is_value_fish."""
        def __getattr__(self, name):
            raise RuntimeError("boom")

    model = {1: _Boom(), 2: _Boom(), 3: _Boom(), 4: _Boom(), 5: _Boom()}
    # 5 villains at a 6-max -> passes the len(profs) >= n-1 gate -> reaches the try block -> raises ->
    # except returns the conservative n (6), NOT the relaxed HU ceiling (2).
    assert fish_relaxed_n(6, model) == 6


def test_fish_relax_default_off(monkeypatch):
    monkeypatch.delenv("POKERBOT_RNR_FISH_RELAX", raising=False)
    model = {s: _value_fish() for s in range(1, 6)}
    assert fish_relaxed_n(6, model) == 6
