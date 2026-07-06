"""P1 verification: Pluribus-faithful continuation-strategy search (blueprint adapter + gadget + solver).

Covers the plan's gate: determinism, the exploit-engine FIREWALL invariants, zero-deviation bit-identity,
continuation-strategy direction, the gadget matrix-game solver, and terminal-leaf passthrough. The strong
"search >= blueprint in 6-max self-play" signal needs a trained blueprint + many hands and lives in P3.
"""

from __future__ import annotations

import os

os.environ.setdefault("POKERBOT_BETTING_ABSTRACTION", "1")

import numpy as np
import jax

from pokerbot.engine.jax_engine import NUM_ACTIONS, NUM_PLAYERS, init_hand
from pokerbot.engine.np_engine import (
    CALL,
    CHECK,
    FOLD,
    BET_BUCKET_START,
    ALL_IN,
    apply_action_np,
    is_chance_node_np,
    legal_action_mask_np,
    sample_chance_np,
    to_numpy_state,
)
from pokerbot.search.state_bridge import SubgameRoot
from pokerbot.search.tabular_blueprint_adapter import (
    TabularBlueprintAdapter,
    bias_continuation,
    CONTINUATION_NAMES,
    NUM_CONTINUATIONS,
)
from pokerbot.search.pluribus_rollout import (
    ContinuationLeafConfig,
    continuation_leaf_value,
    _solve_zero_sum,
)
from pokerbot.search.pluribus_search import PluribusReBeLSolver, PluribusReBeLConfig

_OCC = np.array([True] * 6, dtype=bool)


class _StubPBS:
    """Returns the env's own dealt opponent holes (real PBSTracker integration is P2)."""

    def __init__(self, env, hero):
        self._h = np.asarray(env.holes)
        self.hero = hero

    def sample_opponent_holes(self, rng):
        return {s: (int(self._h[s, 0]), int(self._h[s, 1])) for s in range(NUM_PLAYERS) if s != self.hero}


def _postflop_env(seed: int = 7):
    env = to_numpy_state(init_hand(jax.random.PRNGKey(seed), occupancy=_OCC))
    for _ in range(60):
        if bool(env.done) or is_chance_node_np(env):
            break
        seat = int(env.action_seat)
        if seat < 0:
            break
        m = legal_action_mask_np(env, seat).astype(bool)
        env = apply_action_np(env, CALL if m[CALL] else CHECK)
    if is_chance_node_np(env):
        env = sample_chance_np(env)
    return env


# --------------------------------------------------------------------------- adapter / continuations
def test_continuation_directions_and_normalization():
    env = to_numpy_state(init_hand(jax.random.PRNGKey(0), occupancy=_OCC))
    while is_chance_node_np(env):
        env = sample_chance_np(env)
    seat = int(env.action_seat)
    ad = TabularBlueprintAdapter(snapshot={}, client_bet_sizing=True)
    cs = ad.continuation_set(env, seat)
    assert cs.shape == (NUM_CONTINUATIONS, NUM_ACTIONS)
    assert np.allclose(cs.sum(axis=1), 1.0)
    bp, fold, call, raise_ = cs
    aggro = list(range(BET_BUCKET_START, NUM_ACTIONS))
    # Each continuation does not REDUCE its target group's mass vs the blueprint.
    assert fold[FOLD] >= bp[FOLD] - 1e-9
    assert call[[CHECK, CALL]].sum() >= bp[[CHECK, CALL]].sum() - 1e-9
    assert raise_[aggro].sum() >= bp[aggro].sum() - 1e-9
    assert CONTINUATION_NAMES[0] == "blueprint"


def test_bias_floor_lets_zeroed_action_receive_mass():
    # Blueprint never folds at this node; with bias_floor>0 the fold continuation still tilts to fold.
    mask = np.zeros(NUM_ACTIONS, dtype=bool)
    mask[[CHECK, BET_BUCKET_START]] = True  # no FOLD legal here -> stays zero even with floor
    bp = np.zeros(NUM_ACTIONS)
    bp[CHECK] = 1.0
    out = bias_continuation(bp, np.array([FOLD]), mask, bias_mult=10.0, bias_floor=1e-2)
    assert out[FOLD] == 0.0  # illegal -> never gets mass
    # But a LEGAL zeroed action does receive floor mass.
    mask2 = mask.copy()
    mask2[FOLD] = True
    out2 = bias_continuation(bp, np.array([FOLD]), mask2, bias_mult=10.0, bias_floor=1e-2)
    assert out2[FOLD] > 0.0


# --------------------------------------------------------------------------- gadget solver
def test_zero_sum_solver_matching_pennies():
    # Matching-pennies payoff to the row player has game value 0.
    M = np.array([[1.0, -1.0], [-1.0, 1.0]])
    v = _solve_zero_sum(M, iters=2000)
    assert abs(v) < 0.05


def test_zero_sum_solver_dominant_row():
    # Row 0 dominates row 1 everywhere; the field minimizes -> value = max_i min_j = min of row 0.
    M = np.array([[3.0, 5.0], [0.0, 1.0]])
    v = _solve_zero_sum(M, iters=2000)
    assert abs(v - 3.0) < 0.1


def test_terminal_leaf_passthrough():
    # A terminal env returns rewards[hero]/CHIP_UNIT exactly (no rollout).
    env = to_numpy_state(init_hand(jax.random.PRNGKey(1), occupancy=_OCC))
    # Drive everyone but one to fold preflop -> terminal.
    for _ in range(60):
        if bool(env.done) or is_chance_node_np(env):
            break
        seat = int(env.action_seat)
        m = legal_action_mask_np(env, seat).astype(bool)
        env = apply_action_np(env, FOLD if m[FOLD] else CHECK)
    if bool(env.done):
        from pokerbot.engine.jax_engine import CHIP_UNIT

        ad = TabularBlueprintAdapter(snapshot={}, client_bet_sizing=True)
        hero = 0
        v = continuation_leaf_value(env, hero, ad, ContinuationLeafConfig(select="gadget"))
        assert abs(v - float(env.rewards[hero]) / CHIP_UNIT) < 1e-9


# --------------------------------------------------------------------------- solver behavior
def test_solver_determinism_and_normalized():
    env = _postflop_env()
    hero = int(env.action_seat)
    root = SubgameRoot(env=env, hero_seat=hero, pbs_vector=None,
                       legal_mask=legal_action_mask_np(env, hero).astype(bool))
    cfg = PluribusReBeLConfig(max_iterations=12, time_budget_ms=20000,
                              leaf=ContinuationLeafConfig(select="uniform", rollout_samples=2))
    r1 = PluribusReBeLSolver({}, cfg, rng=np.random.default_rng(0)).solve(root, _StubPBS(env, hero))
    r2 = PluribusReBeLSolver({}, cfg, rng=np.random.default_rng(0)).solve(root, _StubPBS(env, hero))
    assert np.allclose(r1.action_probs, r2.action_probs)
    assert abs(float(r1.action_probs.sum()) - 1.0) < 1e-5
    # probs are supported only on abstracted-legal actions
    legal = PluribusReBeLSolver({}, cfg)._legal(env, hero)
    assert np.all(r1.action_probs[~legal] == 0.0)


def test_per_decision_seed_history_independence():
    """The CRN-cancellation property: a solve at a given PUBLIC state is bitwise identical no matter
    how many (different) solves the same solver instance ran before — i.e. the search's Monte-Carlo
    stream is a pure function of (public state, config), not of process history. This is what lets
    the exploitation battery's ON/OFF arms draw identical search randomness where situations match."""
    env = _postflop_env()
    hero = int(env.action_seat)
    root = SubgameRoot(env=env, hero_seat=hero, pbs_vector=None,
                       legal_mask=legal_action_mask_np(env, hero).astype(bool))
    cfg = PluribusReBeLConfig(max_iterations=8, time_budget_ms=20000,
                              leaf=ContinuationLeafConfig(select="uniform", rollout_samples=2))

    # Solver A burns RNG/leaf-counter history on a DIFFERENT decision first.
    other = _postflop_env(seed=11)
    ohero = int(other.action_seat)
    oroot = SubgameRoot(env=other, hero_seat=ohero, pbs_vector=None,
                        legal_mask=legal_action_mask_np(other, ohero).astype(bool))
    sa = PluribusReBeLSolver({}, cfg)
    sa.solve(oroot, _StubPBS(other, ohero))
    ra = sa.solve(root, _StubPBS(env, hero))

    # Solver B goes straight to the target decision.
    rb = PluribusReBeLSolver({}, cfg).solve(root, _StubPBS(env, hero))
    assert np.array_equal(ra.action_probs, rb.action_probs)


def test_legacy_running_stream_still_deterministic_when_flag_off():
    # per_decision_seed=False restores the ctor-rng running stream; same ctor seed => same result.
    env = _postflop_env()
    hero = int(env.action_seat)
    root = SubgameRoot(env=env, hero_seat=hero, pbs_vector=None,
                       legal_mask=legal_action_mask_np(env, hero).astype(bool))
    cfg = PluribusReBeLConfig(max_iterations=8, time_budget_ms=20000, per_decision_seed=False,
                              leaf=ContinuationLeafConfig(select="uniform", rollout_samples=2))
    r1 = PluribusReBeLSolver({}, cfg, rng=np.random.default_rng(4)).solve(root, _StubPBS(env, hero))
    r2 = PluribusReBeLSolver({}, cfg, rng=np.random.default_rng(4)).solve(root, _StubPBS(env, hero))
    assert np.array_equal(r1.action_probs, r2.action_probs)


def test_zero_deviation_bit_identity():
    """opponent_model=None must produce the SAME hero policy as a sparse/low-confidence profile."""
    from pokerbot.opponent.stats import OpponentStats

    env = _postflop_env()
    hero = int(env.action_seat)
    root = SubgameRoot(env=env, hero_seat=hero, pbs_vector=None,
                       legal_mask=legal_action_mask_np(env, hero).astype(bool))
    cfg = PluribusReBeLConfig(max_iterations=10, time_budget_ms=20000,
                              leaf=ContinuationLeafConfig(select="uniform", rollout_samples=2))
    base = PluribusReBeLSolver({}, cfg, rng=np.random.default_rng(3),
                               opponent_model=None).solve(root, _StubPBS(env, hero))
    sparse = {s: OpponentStats(hands=2) for s in range(NUM_PLAYERS) if s != hero}
    dev = PluribusReBeLSolver({}, cfg, rng=np.random.default_rng(3),
                              opponent_model=sparse).solve(root, _StubPBS(env, hero))
    assert np.allclose(base.action_probs, dev.action_probs, atol=1e-9)


def test_serve_final_iterate_golden_and_zero_gate():
    """R5a: serve_final_iterate=True serves the regret-match of the ROOT's cumulative (CFR+-clamped)
    regrets after the last iteration (the final-iterate strategy) instead of the averaged
    strategy_sum; False (default) is byte-identical to the legacy averaged serve."""
    from pokerbot.search.pluribus_search import _regret_match

    env = _postflop_env()
    hero = int(env.action_seat)
    root = SubgameRoot(env=env, hero_seat=hero, pbs_vector=None,
                       legal_mask=legal_action_mask_np(env, hero).astype(bool))

    def _cfg(**kw):
        return PluribusReBeLConfig(max_iterations=12, time_budget_ms=20000,
                                   leaf=ContinuationLeafConfig(select="uniform", rollout_samples=2),
                                   **kw)

    # Zero-gate: explicit False == default (byte-identical).
    r_def = PluribusReBeLSolver({}, _cfg()).solve(root, _StubPBS(env, hero))
    r_off = PluribusReBeLSolver({}, _cfg(serve_final_iterate=False)).solve(root, _StubPBS(env, hero))
    assert np.array_equal(r_def.action_probs, r_off.action_probs)

    # Golden: ON == regret-match over the solver's FINAL root regrets, masked+renormalized.
    sv = PluribusReBeLSolver({}, _cfg(serve_final_iterate=True))
    r_on = sv.solve(root, _StubPBS(env, hero))
    base_state = to_numpy_state(env)
    root_key = sv._infoset_key(base_state, hero)
    legal = sv._legal(base_state, hero)
    expect = _regret_match(sv._regrets[root_key], legal.astype(np.float64))
    expect = expect * legal
    expect = expect / max(expect.sum(), 1e-12)
    assert np.allclose(r_on.action_probs, expect.astype(np.float32), atol=1e-7)
    # The knob must actually move the served policy here (final != average) — not a placebo.
    assert float(np.abs(r_on.action_probs - r_off.action_probs).sum()) > 1e-6


def test_serve_final_iterate_exploit_path_gated():
    """R5a firewall: with opponent_model set (exploit path) the knob is INERT — the exploit serve is
    always the certified averaged strategy (mirrors the two_sided gating)."""
    from pokerbot.opponent.stats import OpponentStats

    env = _postflop_env()
    hero = int(env.action_seat)
    root = SubgameRoot(env=env, hero_seat=hero, pbs_vector=None,
                       legal_mask=legal_action_mask_np(env, hero).astype(bool))
    model = {s: OpponentStats(hands=2) for s in range(NUM_PLAYERS) if s != hero}

    def _cfg(**kw):
        return PluribusReBeLConfig(max_iterations=10, time_budget_ms=20000,
                                   leaf=ContinuationLeafConfig(select="uniform", rollout_samples=2),
                                   **kw)

    r_off = PluribusReBeLSolver({}, _cfg(serve_final_iterate=False),
                                opponent_model=model).solve(root, _StubPBS(env, hero))
    r_on = PluribusReBeLSolver({}, _cfg(serve_final_iterate=True),
                               opponent_model=model).solve(root, _StubPBS(env, hero))
    assert np.array_equal(r_on.action_probs, r_off.action_probs)


def test_preflop_served_from_blueprint():
    env = to_numpy_state(init_hand(jax.random.PRNGKey(5), occupancy=_OCC))
    while is_chance_node_np(env):
        env = sample_chance_np(env)
    hero = int(env.action_seat)
    root = SubgameRoot(env=env, hero_seat=hero, pbs_vector=None,
                       legal_mask=legal_action_mask_np(env, hero).astype(bool))
    cfg = PluribusReBeLConfig(search_preflop=False)
    res = PluribusReBeLSolver({}, cfg).solve(root, _StubPBS(env, hero))
    assert res.metadata.get("served") == "blueprint_preflop"
    assert abs(float(res.action_probs.sum()) - 1.0) < 1e-5


def test_serve_survives_untraversed_root_infoset():
    """Regression (2026-07-05 dev.fun playground KeyError): a solve whose CFR traversal never
    populates the ROOT infoset — budget-bound to zero iterations, cancelled before the first
    traversal, or (under two-sided cycling) the hero seat never traversing it — must still serve a
    valid uniform-over-legal distribution and MUST NOT raise KeyError from the serve-side strategy
    read. The playground surfaced exactly this: an action-infoset node reached during serving
    (key ``seat|street|board|seats|actions|sprN``, no ``cont!`` prefix) that the traversal never
    populated; the arena's decider-error fallback caught it, but on the engine path it silently
    degrades a decision to a crude fallback. Mirrors the arena config (two_sided)."""
    env = _postflop_env()
    hero = int(env.action_seat)
    root = SubgameRoot(env=env, hero_seat=hero, pbs_vector=None,
                       legal_mask=legal_action_mask_np(env, hero).astype(bool))
    # max_iterations=0 => the traversal loop never runs => _cfv never accumulates the root infoset.
    cfg = PluribusReBeLConfig(max_iterations=0, time_budget_ms=20000, native=False, two_sided=True,
                              leaf=ContinuationLeafConfig(select="uniform", rollout_samples=1))
    sv = PluribusReBeLSolver({}, cfg)
    res = sv.solve(root, _StubPBS(env, hero))            # must NOT raise KeyError
    probs = np.asarray(res.action_probs, dtype=float)
    assert abs(probs.sum() - 1.0) < 1e-6
    # A zero-traversal serve is uniform over the abstracted-legal actions (regret-match of zeros).
    alegal = sv._legal(to_numpy_state(env), hero)
    assert np.all(probs[~alegal] == 0.0)
    expect = alegal.astype(np.float64) / max(alegal.sum(), 1)
    assert np.allclose(probs, expect, atol=1e-6)
    # SERVE-KEY GUARANTEE: the root infoset is registered in BOTH tables even with zero traversals,
    # so the serve-side read can never face a missing key.
    root_key = sv._infoset_key(to_numpy_state(env), hero)
    assert root_key in sv._regrets and root_key in sv._strategy_sum


# --------------------------------------------------------------------------- FIREWALL
def test_firewall_invariants_unchanged():
    from pokerbot.opponent.context import context_spec_hash, action_bin, NUM_BINS
    from pokerbot.opponent import player_profiles

    assert context_spec_hash() == "c960cfceab695388"
    assert NUM_PLAYERS == 6 and NUM_ACTIONS == 11 and NUM_BINS == 4
    # 11 -> 4 action-bin ladder unchanged.
    assert [action_bin(a) for a in range(NUM_ACTIONS)] == [0, 1, 1, 2, 2, 2, 3, 3, 3, 3, 3]
    # v5 = ADDITIVE bluff-ratio Beta counts (ctx_bluff/ctx_bluff_last); the locked context spec,
    # bin ladder, and back-compat loading (v1–v4 files open with new fields empty — pinned by
    # tests/test_bluff_model.py) are all unchanged.
    assert player_profiles.SCHEMA_VERSION == 7   # unified cash+MTT store (v7); bump in lockstep with the schema


def test_solver_does_not_load_neural_stack():
    # The Pluribus path must never construct the poisoned HistoryValueEvaluator / AverageStrategyNet.
    sol = PluribusReBeLSolver({}, PluribusReBeLConfig())
    assert not hasattr(sol, "leaf_eval")
    assert not hasattr(sol, "_blueprint_module")


# --------------------------------------------------------------------------- P2 integration
def test_searchagent_pluribus_wiring():
    from pokerbot.eval.search_agent import SearchAgent

    ag = SearchAgent(use_pluribus_search=True, hero_seat=0, time_budget_ms=1000)
    assert isinstance(ag.solver, PluribusReBeLSolver)
    assert ag.leaf_eval is None and ag._value_ready  # no poisoned value net loaded
    # opponent_model seam used by production_hero._apply_profile.
    ag.solver.opponent_model = {1: None}
    assert ag.solver.opponent_model is not None


def test_leaf_q_accepts_jax_env_for_aivat():
    from pokerbot.engine.game import GameState
    from pokerbot.search.state_bridge import game_state_to_env

    st = GameState.new_hand(button_seat=3)
    env = game_state_to_env(st, st.action_seat)  # JAX env (AIVAT passes root.env)
    cfg = PluribusReBeLConfig(leaf=ContinuationLeafConfig(select="uniform", rollout_samples=1))
    q = PluribusReBeLSolver({}, cfg)._leaf_q(env)
    assert q.shape == (NUM_ACTIONS,)
    assert np.all(np.isfinite(q))


def test_root_exploit_mixer_facing_only_contract():
    """The pruned root-mixer contract: (a) profiles=None is bit-identical to flag-off (the
    zero-deviation OFF arm); (b) UNOPENED decisions are NEVER touched (the unopened branch lost in
    every battery configuration and is disabled); (c) a FACING decision (genuine raise in history)
    with an over-aggressive read shifts the policy when hand strength clears the pot-odds gate."""
    import random

    from pokerbot.engine.actions import Action, ActionType
    from pokerbot.engine.game import GameState
    from pokerbot.eval.search_agent import SearchAgent
    from pokerbot.opponent.population_prior import context_prior
    from pokerbot.opponent.stats import OpponentStats

    def make_agent(root_mixer: bool) -> SearchAgent:
        ag = SearchAgent(use_pluribus_search=True, blueprint_snapshot={}, hero_seat=0,
                         time_budget_ms=10 ** 9, transformer_checkpoint=None,
                         root_exploit_mixer=root_mixer)
        ag.solver.config.max_iterations = 4
        ag.solver.config.leaf = type(ag.solver.config.leaf)(select="uniform", rollout_samples=1)
        return ag

    # Over-aggressive (lag-like) profile: bets/raises dominate every context.
    prof = OpponentStats()
    prof.hands = 200
    for st in ("flop", "turn", "river"):
        for tex in ("dry", "wet", "paired"):
            for facing in ("unopened", "facing_bet"):
                ck = f"{st}|{tex}|{facing}"
                for h in range(1, 50):
                    prof.decay_and_add(ck, [0, 0, 1, 1], context_prior(ck), h)
                prof.add_ctx(ck, [0, 0, 24, 24])

    def unopened_probs(agent, seed=5):
        st = GameState.new_hand(button_seat=3, rng=random.Random(seed))
        st.deal_hole_cards()
        agent.enable_decision_capture(random.Random(1))
        agent.reset_hand(0)
        agent(st)                       # hero UTG, unraised pot => unopened
        return np.asarray(agent.pop_decisions_raw()[-1][0], dtype=float)

    def facing_probs(agent, seed):
        # button=0 => seat 3 acts first preflop; it raises, 4/5 fold => hero (seat 0) faces a raise.
        st = GameState.new_hand(button_seat=0, rng=random.Random(seed))
        st.deal_hole_cards()
        for _ in range(6):
            if st.action_seat == 0:
                break
            legal = st.legal_actions()
            raise_a = next((a for a in legal if a.action_type == ActionType.RAISE), None)
            st.apply_action(raise_a if (st.action_seat == 3 and raise_a) else Action(ActionType.FOLD))
            st.advance_to_next_actor()
        assert st.action_seat == 0
        agent.enable_decision_capture(random.Random(1))
        agent.reset_hand(0)
        agent(st)
        return np.asarray(agent.pop_decisions_raw()[-1][0], dtype=float)

    # Profile ONLY the aggressor seat (seat 3 raises in facing_probs). Profiling all 5 seats with this
    # loose-aggressor would trip the table-aggression kill-switch (>=3 lags => whole exploit OFF), which
    # is correct behavior but not what this facing-branch contract is testing. One aggressor stays below
    # the kill threshold so the facing branch fires; the kill-switch itself is covered in test_table_kill.
    base = make_agent(root_mixer=False)
    on = make_agent(root_mixer=True)
    on.set_opponent_profiles({3: prof})
    off = make_agent(root_mixer=True)
    off.set_opponent_profiles(None)

    # (a) OFF arm bit-identity and (b) unopened never touched, even with a strong profile.
    assert np.array_equal(unopened_probs(base), unopened_probs(off))
    assert np.array_equal(unopened_probs(base), unopened_probs(on))

    # (c) some facing spot shifts (hand-strength gated, so scan a few seeds for one that clears).
    shifted = False
    for seed in range(12):
        pb = facing_probs(make_agent(root_mixer=False), seed)
        on2 = make_agent(root_mixer=True)
        on2.set_opponent_profiles({3: prof})  # only the aggressor (seat 3) -> below the kill threshold
        po = facing_probs(on2, seed)
        assert abs(float(po.sum()) - 1.0) < 1e-6
        if not np.allclose(po, pb):
            shifted = True
            break
    assert shifted, "no facing decision shifted across 12 seeds — the facing branch never fired"


def test_root_mixer_unopened_flag_and_shared_module_parity():
    """Phase-0 arm B contract: (a) ``root_mixer_unopened=True`` lets UNOPENED decisions shift
    (the branch is alive again for the fair-bot re-test) while the FOLD-LEAK CLAMP still holds
    (our fold mass never increases on a continue-read); (b) ``SearchAgent._apply_root_mixer`` is
    a pure delegate of the shared ``pokerbot.opponent.root_mixer.apply_root_exploit_mixer`` —
    identical output for identical inputs (the live bridge consumes the shared module)."""
    import random

    from pokerbot.engine.game import GameState
    from pokerbot.eval.search_agent import SearchAgent
    from pokerbot.opponent.population_prior import context_prior
    from pokerbot.opponent.root_mixer import apply_root_exploit_mixer
    from pokerbot.opponent.stats import OpponentStats

    def make_agent(unopened: bool) -> SearchAgent:
        ag = SearchAgent(use_pluribus_search=True, blueprint_snapshot={}, hero_seat=0,
                         time_budget_ms=10 ** 9, transformer_checkpoint=None,
                         root_exploit_mixer=True, root_mixer_unopened=unopened)
        ag.solver.config.max_iterations = 4
        ag.solver.config.leaf = type(ag.solver.config.leaf)(select="uniform", rollout_samples=1)
        return ag

    # Sticky no-fold profile (station-like RESPONSE read: never folds, always continues) — the
    # unopened branch's mixer_logits should cut bluffs / boost passive continues, never add folds.
    prof = OpponentStats()
    prof.hands = 200
    for st in ("flop", "turn", "river"):
        for tex in ("dry", "wet", "paired"):
            for facing in ("unopened", "facing_bet"):
                ck = f"{st}|{tex}|{facing}"
                for h in range(1, 50):
                    prof.decay_and_add(ck, [0, 2, 0, 0], context_prior(ck), h)
                prof.add_ctx(ck, [0, 48, 0, 0])

    profiles = {s: prof for s in range(1, 6)}

    def unopened_decision(agent, seed=5):
        st = GameState.new_hand(button_seat=3, rng=random.Random(seed))
        st.deal_hole_cards()
        agent.enable_decision_capture(random.Random(1))
        agent.reset_hand(0)
        agent(st)                       # hero UTG, unraised pot => unopened (steal spot)
        return st, np.asarray(agent.pop_decisions_raw()[-1][0], dtype=float)

    shifted = False
    for seed in range(12):
        base_st, pb = unopened_decision(make_agent(unopened=False), seed)
        on = make_agent(unopened=True)
        on.set_opponent_profiles(profiles)
        st_on, po = unopened_decision(on, seed)
        assert abs(float(po.sum()) - 1.0) < 1e-6
        # Fold-leak clamp: a continue-read never increases OUR fold mass in unopened spots.
        assert po[0] <= pb[0] + 1e-9
        if not np.allclose(po, pb):
            shifted = True
    assert shifted, "no unopened decision shifted across 12 seeds — the unopened branch never fired"

    # (b) Delegate parity: the agent method and the shared module are the same function.
    ag = make_agent(unopened=True)
    ag.set_opponent_profiles(profiles)
    st = GameState.new_hand(button_seat=3, rng=random.Random(7))
    st.deal_hole_cards()
    probs = np.full(NUM_ACTIONS, 1.0 / NUM_ACTIONS)
    via_agent = ag._apply_root_mixer(st, st.action_seat, probs.copy())
    via_module = apply_root_exploit_mixer(st, st.action_seat, probs.copy(), profiles,
                                          unopened=True)
    assert np.array_equal(via_agent, via_module)


def test_model_aware_leaf_direction_and_zero_deviation():
    """MODEL-AWARE LEAF: (a) with cont_weights=None the leaf is byte-identical to the legacy path
    (zero-deviation gate); (b) a fold-heavy leaf world values the hero's spot at least as much as a
    call/raise-heavy one for the SAME state (continuations that fold surrender the pot)."""
    env = _postflop_env()
    hero = int(env.action_seat)
    ad = TabularBlueprintAdapter(snapshot={}, client_bet_sizing=True)
    cfg = ContinuationLeafConfig(select="uniform", rollout_samples=8, base_seed=12345)

    v_legacy = continuation_leaf_value(env, hero, ad, cfg)
    v_none = continuation_leaf_value(env, hero, ad, cfg, cont_weights=None)
    assert v_legacy == v_none  # bit-identical

    others = [s for s in range(NUM_PLAYERS) if s != hero]
    foldy = {s: np.array([0.1, 0.8, 0.05, 0.05]) for s in others}
    cally = {s: np.array([0.1, 0.05, 0.8, 0.05]) for s in others}
    v_fold = np.mean([continuation_leaf_value(
        env, hero, ad, ContinuationLeafConfig(select="uniform", rollout_samples=16,
                                              base_seed=12345 + 7 * k),
        cont_weights=foldy) for k in range(6)])
    v_call = np.mean([continuation_leaf_value(
        env, hero, ad, ContinuationLeafConfig(select="uniform", rollout_samples=16,
                                              base_seed=12345 + 7 * k),
        cont_weights=cally) for k in range(6)])
    # Folding fields surrender the pot to the hero more often.
    assert v_fold >= v_call - 1e-9


def test_solver_builds_leaf_weights_only_with_reads():
    from pokerbot.opponent.population_prior import context_prior
    from pokerbot.opponent.stats import OpponentStats

    sol = PluribusReBeLSolver({}, PluribusReBeLConfig())
    assert sol._build_leaf_cont_weights(0) is None          # no model
    sol.opponent_model = {s: OpponentStats(hands=2) for s in range(1, 6)}
    assert sol._build_leaf_cont_weights(0) is None          # sparse => no read => None

    prof = OpponentStats(); prof.hands = 300
    for st in ("flop", "turn", "river"):
        for tex in ("dry", "wet", "paired"):
            for facing in ("unopened", "facing_bet"):
                ck = f"{st}|{tex}|{facing}"
                for h in range(1, 50):
                    prof.decay_and_add(ck, [0, 1, 0, 0], context_prior(ck), h)
                prof.add_ctx(ck, [0, 48, 0, 0])
    sol.opponent_model = {s: prof for s in range(1, 6)}
    w = sol._build_leaf_cont_weights(0)
    assert w is not None and set(w) == {1, 2, 3, 4, 5}
    for s, ws in w.items():
        assert abs(ws.sum() - 1.0) < 1e-9
        assert ws[2] == max(ws)        # station read => call-continuation dominates


def test_leaf_memo_bit_identical_and_deterministic():
    """The per-leaf continuation-set memo (fast path, env POKERBOT_LEAF_MEMO) is a PURE cache: the leaf
    value with the memo ON is byte-identical to the legacy (memo OFF) path for BOTH leaf modes, and is
    deterministic given the seed (the CRN contract the exploitation battery relies on)."""
    from pokerbot.search import pluribus_rollout as R

    ad = TabularBlueprintAdapter(snapshot={}, client_bet_sizing=True)
    try:
        for mode in ("uniform", "gadget"):
            cfg = ContinuationLeafConfig(select=mode, rollout_samples=3, base_seed=4242)
            for seed in (7, 11, 13):
                env = _postflop_env(seed=seed)
                hero = int(env.action_seat) if int(env.action_seat) >= 0 else 0
                R.set_leaf_memo(False)
                v_off = continuation_leaf_value(env, hero, ad, cfg)
                R.set_leaf_memo(True)
                v_on1 = continuation_leaf_value(env, hero, ad, cfg)
                v_on2 = continuation_leaf_value(env, hero, ad, cfg)  # same seed => identical
                assert v_on1 == v_off, f"{mode}/{seed}: memo changed leaf value {v_on1} != {v_off}"
                assert v_on1 == v_on2, f"{mode}/{seed}: memo non-deterministic given seed"
    finally:
        R.set_leaf_memo(True)


def test_leaf_gadget_batch_bit_identical():
    """The gadget-leaf prefix-sharing DAG (fast path, env POKERBOT_LEAF_GADGET_BATCH) is BYTE-IDENTICAL
    to the legacy per-cell double loop: it only shares engine work on common path prefixes while feeding
    each (i,j) cell the SAME depth-indexed action draws a freshly-seeded baseline cell would consume. The
    gadget leaf value (and thus the served policy) must be bit-identical with the DAG ON vs OFF, for the
    memo BOTH on and off (the two fast paths must compose), across seeds. OFF is the default so this is a
    pure speed optimization guarded against any divergence."""
    from pokerbot.search import pluribus_rollout as R

    ad = TabularBlueprintAdapter(snapshot={}, client_bet_sizing=True)
    try:
        for memo_on in (False, True):
            R.set_leaf_memo(memo_on)
            for rollout in (1, 4):
                cfg = ContinuationLeafConfig(select="gadget", rollout_samples=rollout, base_seed=4242)
                for seed in (7, 11, 13, 17):
                    env = _postflop_env(seed=seed)
                    hero = int(env.action_seat) if int(env.action_seat) >= 0 else 0
                    R.set_leaf_gadget_batch(False)
                    v_off = continuation_leaf_value(env, hero, ad, cfg)
                    R.set_leaf_gadget_batch(True)
                    v_on = continuation_leaf_value(env, hero, ad, cfg)
                    assert v_on == v_off, (
                        f"memo={memo_on} rollout={rollout} seed={seed}: DAG changed gadget leaf "
                        f"value {v_on} != {v_off}")
    finally:
        R.set_leaf_gadget_batch(False)
        R.set_leaf_memo(True)


def test_model_aware_leaf_direction_and_zero_deviation():
    """MODEL-AWARE LEAF: (a) with cont_weights=None the leaf is byte-identical to the legacy path
    (zero-deviation gate); (b) a fold-heavy leaf world values the hero's spot at least as much as a
    call/raise-heavy one for the SAME state (continuations that fold surrender the pot)."""
    env = _postflop_env()
    hero = int(env.action_seat)
    ad = TabularBlueprintAdapter(snapshot={}, client_bet_sizing=True)
    cfg = ContinuationLeafConfig(select="uniform", rollout_samples=8, base_seed=12345)

    v_legacy = continuation_leaf_value(env, hero, ad, cfg)
    v_none = continuation_leaf_value(env, hero, ad, cfg, cont_weights=None)
    assert v_legacy == v_none  # bit-identical

    others = [s for s in range(NUM_PLAYERS) if s != hero]
    foldy = {s: np.array([0.1, 0.8, 0.05, 0.05]) for s in others}
    cally = {s: np.array([0.1, 0.05, 0.8, 0.05]) for s in others}
    v_fold = np.mean([continuation_leaf_value(
        env, hero, ad, ContinuationLeafConfig(select="uniform", rollout_samples=16,
                                              base_seed=12345 + 7 * k),
        cont_weights=foldy) for k in range(6)])
    v_call = np.mean([continuation_leaf_value(
        env, hero, ad, ContinuationLeafConfig(select="uniform", rollout_samples=16,
                                              base_seed=12345 + 7 * k),
        cont_weights=cally) for k in range(6)])
    # Folding fields surrender the pot to the hero more often.
    assert v_fold >= v_call - 1e-9


def test_solver_builds_leaf_weights_only_with_reads():
    from pokerbot.opponent.population_prior import context_prior
    from pokerbot.opponent.stats import OpponentStats

    sol = PluribusReBeLSolver({}, PluribusReBeLConfig())
    assert sol._build_leaf_cont_weights(0) is None          # no model
    sol.opponent_model = {s: OpponentStats(hands=2) for s in range(1, 6)}
    assert sol._build_leaf_cont_weights(0) is None          # sparse => no read => None

    prof = OpponentStats(); prof.hands = 300
    for st in ("flop", "turn", "river"):
        for tex in ("dry", "wet", "paired"):
            for facing in ("unopened", "facing_bet"):
                ck = f"{st}|{tex}|{facing}"
                for h in range(1, 50):
                    prof.decay_and_add(ck, [0, 1, 0, 0], context_prior(ck), h)
                prof.add_ctx(ck, [0, 48, 0, 0])
    sol.opponent_model = {s: prof for s in range(1, 6)}
    w = sol._build_leaf_cont_weights(0)
    assert w is not None and set(w) == {1, 2, 3, 4, 5}
    for s, ws in w.items():
        assert abs(ws.sum() - 1.0) < 1e-9
        assert ws[2] == max(ws)        # station read => call-continuation dominates
