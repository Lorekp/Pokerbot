"""RNR restriction of the solver-internal best-response (#28).

The solver models the opponent deviating from blueprint and hero CFR best-responds to THAT. Left
unbounded (the flat _CTX_MAX_ALPHA = 0.7) the response is a pure best-response — it crushes a STATIC
reg but is maximally COUNTER-exploitable by an ADAPTIVE pro. These tests pin the restriction:
``cap_eff(stability, n)`` bounds the modeled deviation, monotonically shrinking it for multiway tables
(no minimax safety) and caught adapters (stability -> 0), and is byte-identical when the flag is off.
"""

from __future__ import annotations

import numpy as np

from pokerbot.engine.game import GameState
from pokerbot.opponent.rnr_cap import CAP_FLOOR, restricted_max_alpha
from pokerbot.opponent.stats import OpponentStats
from pokerbot.search.leaf_evaluator import MockHistoryValueEvaluator
from pokerbot.search.state_bridge import game_state_to_env
from pokerbot.search.turbo_rebel_search import (
    TurboReBeLConfig,
    TurboReBeLSolver,
    _context_key_from_env,
)


# ---- the pure helper -----------------------------------------------------------------------------

def test_restricted_max_alpha_never_exceeds_base():
    # The whole safety argument: turning the restriction on can only LOWER the modeled deviation.
    for stab in (0.0, 0.5, 1.0):
        for n in (2, 3, 4, 5, 6):
            assert restricted_max_alpha(0.7, stab, n) <= 0.7 + 1e-12


def test_restricted_max_alpha_anchors():
    assert restricted_max_alpha(0.7, 1.0, 2) == 0.7                       # HU stable: cap_eff=1.0 -> keep 0.7
    assert abs(restricted_max_alpha(0.7, 1.0, 6) - 0.5) < 1e-12           # 6-max stable: N-ceiling 0.5
    assert abs(restricted_max_alpha(0.7, 0.0, 6) - CAP_FLOOR) < 1e-12     # caught adapter -> floor 0.25


def test_restricted_max_alpha_monotone():
    # decreasing in N (more players -> weaker minimax safety -> tighter cap)
    assert (restricted_max_alpha(0.7, 1.0, 2) >= restricted_max_alpha(0.7, 1.0, 4)
            >= restricted_max_alpha(0.7, 1.0, 6))
    # increasing in stability (a stable read may exploit harder; an adapter is clamped)
    assert (restricted_max_alpha(0.7, 0.0, 6) <= restricted_max_alpha(0.7, 0.5, 6)
            <= restricted_max_alpha(0.7, 1.0, 6))


# ---- end-to-end at _opponent_strategy (same fixture as test_opponent_exploitation) ---------------

def _fixture():
    state = GameState.new_hand(button_seat=3)
    hero = state.button_seat
    villain = (hero + 1) % 6
    env = game_state_to_env(state, hero)
    legal = np.zeros(11, dtype=np.float64)
    legal[[0, 2, 3, 4]] = 1.0
    regrets = np.zeros(11, dtype=np.float64)
    ck = _context_key_from_env(env, villain)
    station = OpponentStats(hands=80, ctx_counts={ck: [2, 60, 1, 1]})   # heavy over-call read -> bias
    return env, legal, regrets, villain, station


def _dev(s, base) -> float:
    return float(np.abs(np.asarray(s) - np.asarray(base)).sum())


def _solve(cfg, model, *, stability=None, n_live=None):
    solv = TurboReBeLSolver(MockHistoryValueEvaluator(), cfg, opponent_model=model)
    if stability is not None:
        solv.opponent_stability = stability
    if n_live is not None:
        solv.opponent_n_live = n_live
    env, legal, regrets, villain, _ = _FIX
    return solv._opponent_strategy(env, villain, legal, regrets)


_FIX = _fixture()


def test_rnr_restriction_shrinks_deviation_monotonically():
    env, legal, regrets, villain, station = _FIX
    cfg_off = TurboReBeLConfig(max_depth=2, rnr_restrict=False)
    cfg_on = TurboReBeLConfig(max_depth=2, rnr_restrict=True)

    base = _solve(cfg_off, None)
    s_off = _solve(cfg_off, {villain: station})                                  # un-restricted (0.7)
    s_hu = _solve(cfg_on, {villain: station}, stability={villain: 1.0}, n_live=2)   # cap_eff(1,2)=1.0
    s_6s = _solve(cfg_on, {villain: station}, stability={villain: 1.0}, n_live=6)   # cap_eff(1,6)=0.5
    s_6a = _solve(cfg_on, {villain: station}, stability={villain: 0.0}, n_live=6)   # cap_eff(0,6)=0.25

    d_off, d_hu, d_6s, d_6a = (_dev(s_off, base), _dev(s_hu, base),
                               _dev(s_6s, base), _dev(s_6a, base))
    assert d_off > 1e-6, "the station read must produce a real deviation to exercise the restriction"
    assert abs(d_hu - d_off) < 1e-9, "HU stable read (cap_eff 1.0) must be byte-identical to un-restricted"
    assert d_6s < d_off - 1e-9, "6-max stable: the N-ceiling 0.5 must shrink the modeled deviation"
    assert d_6a < d_6s - 1e-9, "a caught adapter (stability 0) must shrink it further toward GTO"


def test_rnr_off_ignores_stability_inputs():
    env, legal, regrets, villain, station = _FIX
    cfg_off = TurboReBeLConfig(max_depth=2, rnr_restrict=False)
    s_plain = _solve(cfg_off, {villain: station})
    # stability/n_live set to maximally-clamping values: with the flag OFF they must be ignored entirely.
    s_with = _solve(cfg_off, {villain: station}, stability={villain: 0.0}, n_live=6)
    assert np.array_equal(s_plain, s_with), "rnr_restrict=False must ignore opponent_stability/opponent_n_live"


def test_profile_stability_from_fast_slow_divergence():
    from pokerbot.opponent.bayes_exploit import profile_stability

    # no maintained dual-EWMA data => trust the read (stability 1.0)
    assert profile_stability(OpponentStats()) == 1.0
    # fast == slow (no gear change) => stable
    ck = "flop|dry|facing_bet"
    steady = OpponentStats(ctx_fast={ck: [10.0, 10.0, 10.0, 10.0]},
                           ctx_slow={ck: [10.0, 10.0, 10.0, 10.0]})
    assert profile_stability(steady) == 1.0
    # fast strongly diverged from slow with ample fast evidence => caught adapting => stability << 1
    adapt = OpponentStats(ctx_fast={ck: [40.0, 0.0, 0.0, 0.0]},
                          ctx_slow={ck: [10.0, 10.0, 10.0, 10.0]})
    s = profile_stability(adapt)
    assert 0.0 <= s < 0.5, f"a hard gear-change must collapse stability, got {s}"


def test_profile_stability_drives_the_solver_clamp():
    # end-to-end: the SAME station read, but one profile also shows a fast/slow gear change -> the derived
    # stability shrinks the modeled deviation vs the steady profile, all via profile_stability (no tracker).
    from pokerbot.opponent.bayes_exploit import profile_stability

    env, legal, regrets, villain, station = _FIX
    cfg_on = TurboReBeLConfig(max_depth=2, rnr_restrict=True)
    base = _solve(TurboReBeLConfig(max_depth=2, rnr_restrict=False), None)
    steady_stab = {villain: 1.0}
    adapt_stab = {villain: 0.0}
    s_steady = _solve(cfg_on, {villain: station}, stability=steady_stab, n_live=6)
    s_adapt = _solve(cfg_on, {villain: station}, stability=adapt_stab, n_live=6)
    assert _dev(s_adapt, base) < _dev(s_steady, base) - 1e-9
    # and profile_stability yields exactly such a scalar from a real profile
    assert 0.0 <= profile_stability(station) <= 1.0


def test_rnr_n_live_fallback_from_model_size():
    cfg = TurboReBeLConfig(rnr_restrict=True)
    solv = TurboReBeLSolver(MockHistoryValueEvaluator(), cfg, opponent_model={1: None, 2: None, 3: None})
    assert solv._rnr_n_live() == 4          # 3 villains + hero
    solv.opponent_n_live = 6
    assert solv._rnr_n_live() == 6          # explicit caller value wins
    empty = TurboReBeLSolver(MockHistoryValueEvaluator(), cfg, opponent_model=None)
    assert empty._rnr_n_live() == 2         # no model -> HU floor


# ---- no-punisher (all-fish) wipe-the-floor relaxation (default-OFF) -------------------------------

def _value_fish():
    return OpponentStats(hands=30, hud={"vpip": 58, "pfr": 9, "wtsd": 33})   # loose-passive => _is_value_fish


def test_fish_relax_off_by_default(monkeypatch):
    from pokerbot.opponent.rnr_cap import fish_relaxed_n
    monkeypatch.delenv("POKERBOT_RNR_FISH_RELAX", raising=False)
    model = {s: _value_fish() for s in range(1, 6)}   # 5 confirmed fish
    assert fish_relaxed_n(6, model) == 6              # default OFF -> unchanged, even all-fish


def test_fish_relax_all_fish_collapses_to_hu(monkeypatch):
    from pokerbot.opponent.rnr_cap import fish_relaxed_n
    monkeypatch.setenv("POKERBOT_RNR_FISH_RELAX", "1")
    model = {s: _value_fish() for s in range(1, 6)}   # all 5 villains confirmed fish -> no punisher
    assert fish_relaxed_n(6, model) == 2              # HU ceiling -> full exploitation


def test_fish_relax_unknown_seat_blocks(monkeypatch):
    from pokerbot.opponent.rnr_cap import fish_relaxed_n
    monkeypatch.setenv("POKERBOT_RNR_FISH_RELAX", "1")
    model = {s: _value_fish() for s in range(1, 5)}   # only 4 villain profiles at a 6-max (1 unknown)
    assert fish_relaxed_n(6, model) == 6              # an unprofiled seat could be a punisher -> no relax


def test_fish_relax_one_reg_blocks(monkeypatch):
    from pokerbot.opponent.rnr_cap import fish_relaxed_n
    monkeypatch.setenv("POKERBOT_RNR_FISH_RELAX", "1")
    reg = OpponentStats(hands=200, hud={"vpip": 20, "pfr": 17, "wsd": 54, "foldToCbet": 50})  # not a fish
    model = {1: _value_fish(), 2: _value_fish(), 3: _value_fish(), 4: _value_fish(), 5: reg}
    assert fish_relaxed_n(6, model) == 6              # one punisher present -> stay at the conservative cap


def test_fish_relax_drives_solver_n(monkeypatch):
    # integration: the solver's _rnr_n_live picks up the relaxation, collapsing a 6-max all-fish table to 2.
    monkeypatch.setenv("POKERBOT_RNR_FISH_RELAX", "1")
    cfg = TurboReBeLConfig(rnr_restrict=True)
    model = {s: _value_fish() for s in range(1, 6)}
    solv = TurboReBeLSolver(MockHistoryValueEvaluator(), cfg, opponent_model=model)
    solv.opponent_n_live = 6
    assert solv._rnr_n_live() == 2                    # all-fish table -> HU ceiling
    monkeypatch.delenv("POKERBOT_RNR_FISH_RELAX")
    assert solv._rnr_n_live() == 6                    # OFF -> the conservative 6-max ceiling
