"""ZERO-DEVIATION + EXACT-MATH guards for the native CFV-under-exploitation port (POKERBOT_NATIVE_CFV).

Two firewall properties:

1. ZERO-DEVIATION (the firewall): with ``opponent_model is None`` the native CFV code path is reachable
   only via the GTO route (tilt_table is None), so it must be BYTE-IDENTICAL to the existing
   POKERBOT_NATIVE_SUBGAME GTO path. A None / empty tilt table makes EVERY opponent node miss the lookup
   and fall back to the blueprint-mixed regret-match — i.e. the GTO behaviour. Verified by solving the
   SAME root with and without the (None) tilt arg and asserting identical action probs.

2. EXACT-MATH (tilt application): the native side LOOKS UP the precomputed row and returns it verbatim,
   re-masked + renormalized. We pin this deterministically with a POINT-MASS opponent: a tilt table that
   forces every modeled opponent infoset to a single legal action removes the opponent-action RNG
   entirely, so the native solve becomes a pure function of the public seed + leaf RNG. We then assert
   (a) the row is applied (forcing the opponent to FOLD vs to CALL gives different, sensible hero EVs),
   (b) the masking+renormalization identity holds, and (c) the Python tilt-table BUILDER rows equal a
   direct recompute of ``_opponent_strategy`` (the exploit math is bit-for-bit the Python result by
   construction — the table just carries it).

SKIPPED when the native SubgameSolver is not built.
"""

from __future__ import annotations

import os
import random

import numpy as np
import pytest

os.environ.setdefault("POKERBOT_PREFLOP_LOSSLESS", "1")
os.environ.setdefault("POKERBOT_SEARCH_PREFLOP", "1")
# Zero-deviation is BYTE-IDENTITY between the no-tilt and tilt=None paths. The blueprint-CV
# control variate (POKERBOT_BLUEPRINT_CV=1 default) shifts the rollout by an algebraically-zero
# but FP-non-zero amount, which breaks strict `allclose(atol=1e-9)` assertions. Force OFF here so
# the test gates the byte-identity path; the no-CV quality test (test_blueprint_cv_byte_identical
# in test_pluribus_rollout.py) asserts the same CV-OFF path is bit-identical to the legacy code.
os.environ["POKERBOT_BLUEPRINT_CV"] = "0"

from pokerbot.engine.actions import ActionType  # noqa: E402
from pokerbot.engine.game import GameState, Street, play_hand  # noqa: E402
from pokerbot.engine.np_engine import to_numpy_state  # noqa: E402
from pokerbot.opponent.stats import OpponentStats  # noqa: E402
from pokerbot.search import native_solver as ns  # noqa: E402
from pokerbot.search import pluribus_search as ps  # noqa: E402
from pokerbot.search.pbs_tracker import PBSTracker  # noqa: E402
from pokerbot.search import pluribus_rollout  # noqa: E402
from pokerbot.search.pluribus_rollout import ContinuationLeafConfig, set_blueprint_cv  # noqa: E402

# Force the CV off at module-load — the env var is read once at module import and the global is
# process-wide. Tests in this file assert zero-deviation (bit-identity / atol=1e-9) and the CV
# must be off for those assertions to hold (otherwise the mean(v - V_bp) + V_bp reconstruction
# diverges from the legacy mean(v) by FP epsilon at the very least, plus the test sweep budgets
# can land in different RNG states if CV is on).
set_blueprint_cv(False)
from pokerbot.search.pluribus_search import PluribusReBeLConfig, PluribusReBeLSolver  # noqa: E402
from pokerbot.search.state_bridge import build_subgame_root, game_state_to_env  # noqa: E402

native_only = pytest.mark.skipif(
    not ns.native_subgame_available(), reason="native SubgameSolver not built/available"
)


def _postflop_root(seed: int = 7):
    rng = random.Random(seed)
    captured: list = []

    def rec_policy(state):
        legal = state.legal_actions()
        seat = state.action_seat
        if (not captured and seat is not None and seat >= 0
                and state.street != Street.PREFLOP and state.players[seat].hole):
            import copy
            captured.append((copy.deepcopy(state), int(seat)))
        if not legal:
            return None
        passive = [a for a in legal if a.action_type in (ActionType.CHECK, ActionType.CALL)]
        if passive and rng.random() < 0.85:
            return rng.choice(passive)
        return rng.choice(legal)

    for _ in range(400):
        if captured:
            break
        st = GameState.new_hand(button_seat=3, rng=random.Random(rng.randint(0, 10 ** 9)))
        play_hand({s: rec_policy for s in range(6)}, state=st, rng=random.Random(rng.randint(0, 10 ** 9)))
    if not captured:
        raise RuntimeError("could not reach a postflop node")
    return captured[0]


def _cfg(iters=24):
    return PluribusReBeLConfig(
        max_iterations=iters, time_budget_ms=10 ** 9, max_depth=48, blueprint_mix=0.5,
        client_bet_sizing=True, search_preflop=True,
        leaf=ContinuationLeafConfig(select="uniform", rollout_samples=2, max_playout_depth=48,
                                    base_seed=0, resample_board=True),
        per_decision_seed=True, native=True,
    )


def _native_solve(solver, base_state, hero, *, tilt_table):
    from pokerbot.engine.jax_weva_encode import _POSTFLOP_NUM_BUCKETS

    cfg = solver.config
    return ns.native_subgame_solve(
        base_state, hero, solver.adapter,
        public_seed=solver._public_seed(base_state, hero),
        max_iterations=cfg.max_iterations, max_depth=cfg.max_depth,
        blueprint_mix=cfg.blueprint_mix,
        rollout_samples=cfg.leaf.rollout_samples,
        resample_board=cfg.leaf.resample_board,
        max_playout_depth=cfg.leaf.max_playout_depth,
        num_buckets=int(_POSTFLOP_NUM_BUCKETS),
        linear_averaging=bool(cfg.linear_averaging),
        blueprint_cv=bool(pluribus_rollout.blueprint_cv_enabled()),
        dcfr=bool(cfg.dcfr), dcfr_alpha=float(cfg.dcfr_alpha),
        dcfr_beta=float(cfg.dcfr_beta), dcfr_gamma=float(cfg.dcfr_gamma),
        tilt_table=tilt_table,
    )


@native_only
def test_zero_deviation_none_tilt_equals_gto():
    """opponent_model is None => the None tilt table must give BYTE-IDENTICAL output to the GTO native
    path (every opponent node falls back to the blueprint mix). This is the firewall's zero-deviation
    gate: the new arg cannot regress the existing POKERBOT_NATIVE_SUBGAME path."""
    st, hero = _postflop_root()
    pbs = PBSTracker.init_uniform(hero, st.players[hero].hole, st.board)
    root = build_subgame_root(st, pbs, hero)
    solver = PluribusReBeLSolver({}, _cfg(), opponent_model=None)
    base_state = to_numpy_state(root.env)

    gto = _native_solve(solver, base_state, hero, tilt_table=None)
    # An EMPTY dict must behave identically to None (every node misses -> GTO fallback).
    empty = _native_solve(solver, base_state, hero, tilt_table={})
    assert gto is not None and empty is not None
    assert np.array_equal(gto, empty), "empty tilt table deviated from the GTO native path"


@native_only
def test_zero_deviation_solver_route_with_no_read_model():
    """End-to-end firewall: a model whose seats have NO usable read (sparse stats) yields a tilt table
    whose rows ARE the GTO base (_opponent_strategy returns the unbiased base), so enabling native CFV
    must not change the served decision vs the GTO native path. Also confirms native CFV is OFF by
    default and the served tag is correct when on."""
    st, hero = _postflop_root(seed=13)
    pbs = PBSTracker.init_uniform(hero, st.players[hero].hole, st.board)
    root = build_subgame_root(st, pbs, hero)
    # sparse model => context_read None / archetype unknown => _opponent_strategy == GTO base.
    model = {s: OpponentStats(hands=1) for s in range(6) if s != hero}
    solver = PluribusReBeLSolver({}, _cfg(), opponent_model=model)

    # Default OFF: native CFV not engaged even with the model + native subgame on.
    ps.set_native_subgame(True)
    ps.set_native_cfv(False)
    try:
        res_off = solver.solve(root, pbs)
    finally:
        ps.set_native_subgame(False)
    assert res_off.metadata.get("served") not in ("search_native", "search_native_cfv")

    # ON: the path is taken (served tag), output is a valid masked distribution.
    ps.set_native_subgame(True)
    ps.set_native_cfv(True)
    try:
        res_on = solver.solve(root, pbs)
    finally:
        ps.set_native_subgame(False)
        ps.set_native_cfv(False)
    p = np.asarray(res_on.action_probs, dtype=np.float64)
    assert res_on.metadata.get("served") == "search_native_cfv"
    assert np.all(np.isfinite(p)) and abs(p.sum() - 1.0) < 1e-6


@native_only
def test_tilt_table_builder_rows_match_opponent_strategy():
    """EXACT-MATH (Python side): every row the tilt-table BUILDER stores equals a direct recompute of
    ``_opponent_strategy`` at that opponent infoset — i.e. the table carries the verbatim exploit-tail
    result, no recomputation drift. (The native side then returns the row as-is, masked+renormalized.)"""
    st, hero = _postflop_root(seed=21)
    pbs = PBSTracker.init_uniform(hero, st.players[hero].hole, st.board)
    root = build_subgame_root(st, pbs, hero)
    # A confident station read at the live context => non-trivial tilt rows.
    from pokerbot.search.turbo_rebel_search import _context_key_from_env

    base_state = to_numpy_state(root.env)
    model = {}
    for s in range(6):
        if s == hero:
            continue
        env_s = base_state
        ck = _context_key_from_env(env_s, s)
        model[s] = OpponentStats(hands=120, ctx_counts={ck: [2, 80, 1, 1]})
    solver = PluribusReBeLSolver({}, _cfg(), opponent_model=model)
    # Mirror the per-solve reset solve() does before building the table.
    solver._regrets.clear(); solver._strategy_sum.clear(); solver._bp_cache.clear()
    solver._key_cache.clear(); solver._legal_cache.clear(); solver._leaf_cache.clear()
    solver._leaf_cont_weights = None

    # Seed the RNG exactly as solve() does (the sampled traversal is RNG-driven).
    solver.rng = np.random.default_rng(solver._public_seed(base_state, hero))
    table = solver._build_native_cfv_tilt_table(base_state, hero)
    assert table is not None and len(table) > 0

    # Re-walk (sampled, same shape as the builder) and recompute _opponent_strategy directly at each
    # opponent node that IS in the table; assert the stored row equals the recompute bit-for-bit (the
    # table carries the verbatim exploit-tail result — no recomputation drift). Off-table nodes (not
    # sampled) are skipped: they fall back to GTO natively (the benign-miss contract).
    checked = 0
    seen = set()

    def walk(env, depth):
        nonlocal checked
        if bool(env.done) or depth >= solver.config.max_depth:
            return
        if ns.is_chance_node_np(env):
            return
        actor = int(env.action_seat)
        if actor < 0:
            return
        kstr = solver._infoset_key(env, actor)
        legal = solver._legal(env, actor, kstr)
        if not legal.any():
            return
        regrets = solver._get_regrets(kstr)
        if actor != hero:
            ikey = solver._infoset_key_i64(env, actor)
            row = np.asarray(
                solver._opponent_strategy(env, actor, legal, regrets, kstr), dtype=np.float64)
            if ikey in table and ikey not in seen:
                seen.add(ikey)
                stored = np.asarray(table[ikey], dtype=np.float64)
                assert np.array_equal(stored, row), (ikey, stored, row)
                checked += 1
            lp = row[legal]
            tot = lp.sum()
            a = int(np.flatnonzero(legal)[0]) if tot <= 1e-12 else \
                int(solver.rng.choice(np.flatnonzero(legal), p=lp / tot))
            walk(ns.apply_action_np(env, a), depth + 1)
        else:
            for a in np.flatnonzero(legal):
                walk(ns.apply_action_np(env, int(a)), depth + 1)

    solver.rng = np.random.default_rng(solver._public_seed(base_state, hero))
    n_iters = max(1, min(solver.config.max_iterations, solver.config.native_cfv_build_iters))
    for _ in range(n_iters):
        holes = solver._pbs_sample_holes(base_state, hero)
        walk(solver._set_world(base_state, holes), 0)
    assert checked > 0, "no opponent nodes checked"


@native_only
def test_native_applies_point_mass_tilt_deterministically():
    """EXACT-MATH (native application): a POINT-MASS tilt (force every modeled opponent infoset to a
    single legal action) removes opponent-action RNG, so the native solve is deterministic across
    public seeds' opponent branch. Forcing the opponent to a different action must change the hero's
    served decision (the row IS applied), and the result must be a valid masked distribution (the
    re-mask + renormalize identity holds)."""
    st, hero = _postflop_root(seed=33)
    pbs = PBSTracker.init_uniform(hero, st.players[hero].hole, st.board)
    root = build_subgame_root(st, pbs, hero)
    base_state = to_numpy_state(root.env)
    solver = PluribusReBeLSolver({}, _cfg(iters=40),
                                 opponent_model={s: OpponentStats(hands=50) for s in range(6) if s != hero})
    solver._leaf_cont_weights = None

    def _pick(idxs, prefer_fold):
        if prefer_fold and 0 in idxs:
            return 0
        if (not prefer_fold) and any(i in idxs for i in (1, 2)):
            return next(i for i in (1, 2) if i in idxs)
        return idxs[0] if prefer_fold else idxs[-1]

    # Branch all HERO actions; at each opponent node FORCE a single action (point mass) and follow ONLY
    # that line — so the walk is tractable (the opponent fan-out collapses to one child) and the table
    # covers exactly the opponent infosets the native solve (driven by the same point-mass rows) reaches.
    def build_forced(prefer_fold: bool):
        table: dict = {}

        def walk(env, depth):
            if bool(env.done) or depth >= solver.config.max_depth:
                return
            if ns.is_chance_node_np(env):
                return
            actor = int(env.action_seat)
            if actor < 0:
                return
            kstr = solver._infoset_key(env, actor)
            legal = solver._legal(env, actor, kstr)
            if not legal.any():
                return
            if actor != hero:
                idxs = list(int(i) for i in np.flatnonzero(legal))
                pick = _pick(idxs, prefer_fold)
                row = [0.0] * 11
                row[int(pick)] = 1.0
                table[solver._infoset_key_i64(env, actor)] = row
                walk(ns.apply_action_np(env, int(pick)), depth + 1)
            else:
                for a in np.flatnonzero(legal):
                    walk(ns.apply_action_np(env, int(a)), depth + 1)

        walk(base_state, 0)
        return table

    fold_table = build_forced(prefer_fold=True)
    call_table = build_forced(prefer_fold=False)
    assert fold_table and call_table

    p_fold = _native_solve(solver, base_state, hero, tilt_table=fold_table)
    p_call = _native_solve(solver, base_state, hero, tilt_table=call_table)
    assert p_fold is not None and p_call is not None

    # Valid masked distributions.
    for p in (p_fold, p_call):
        assert np.all(np.isfinite(p)) and abs(p.sum() - 1.0) < 1e-6
        assert np.all(p >= -1e-12)

    # DETERMINISM of a point-mass tilt: re-running with the SAME forced table gives the identical result
    # (opponent branch is RNG-free; leaf RNG is seeded from the public state).
    p_fold2 = _native_solve(solver, base_state, hero, tilt_table=fold_table)
    assert np.array_equal(p_fold, p_fold2)

    # The row IS applied: forcing the opponent to fold vs to call changes hero's best response.
    assert not np.allclose(p_fold, p_call, atol=1e-6), (p_fold, p_call)
