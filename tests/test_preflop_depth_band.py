"""A1 stack-agnostic preflop: the 100bb-trained preflop blueprint is served only when the table is within
its depth band; off-band (deep/short) the solver falls through to the depth-general search instead of an
off-depth range. Standard 100bb tables stay on the blueprint path bit-for-bit."""

from __future__ import annotations

import jax
import numpy as np
import pytest

from pokerbot.engine.jax_engine import CHIP_UNIT, NUM_PLAYERS, init_hand
from pokerbot.engine.cards import Card
from pokerbot.engine.np_engine import legal_action_mask_np, to_numpy_state
from pokerbot.search.pluribus_rollout import ContinuationLeafConfig
from pokerbot.search.pluribus_search import PluribusReBeLConfig, PluribusReBeLSolver
from pokerbot.search.state_bridge import SubgameRoot

_OCC = np.array([True] * 6, dtype=bool)


class _StubPBS:
    def __init__(self, env, hero):
        self._h = np.asarray(env.holes)
        self.hero = hero

    def sample_opponent_holes(self, rng):
        return {s: (int(self._h[s, 0]), int(self._h[s, 1])) for s in range(NUM_PLAYERS) if s != self.hero}


def _preflop_env(stack_bb: float):
    import jax.numpy as jnp
    stacks = jnp.full(NUM_PLAYERS, int(stack_bb * CHIP_UNIT), dtype=jnp.int32)
    return to_numpy_state(init_hand(jax.random.PRNGKey(0), occupancy=jnp.asarray(_OCC), stacks=stacks))


def _snapshot_row(env, hero):
    from pokerbot.encoding.fingerprint_native import fingerprint_fast
    from pokerbot.engine.jax_weva_fast import default_weva_bucket_table

    key = fingerprint_fast(env, hero, weva_table=default_weva_bucket_table())
    row = legal_action_mask_np(env, hero).astype(np.float32)
    row /= row.sum()
    return {key: row}


def _solve(stack_bb: float, band, *, source_hit: bool = True):
    env = _preflop_env(stack_bb)
    hero = int(env.action_seat)
    root = SubgameRoot(env=env, hero_seat=hero, pbs_vector=None,
                       legal_mask=legal_action_mask_np(env, hero).astype(bool))
    cfg = PluribusReBeLConfig(max_iterations=4, time_budget_ms=20000, preflop_blueprint_band_bb=band,
                              leaf=ContinuationLeafConfig(select="uniform", rollout_samples=1))
    snapshot = _snapshot_row(env, hero) if source_hit else {}
    res = PluribusReBeLSolver(snapshot, cfg, rng=np.random.default_rng(0)).solve(
        root, _StubPBS(env, hero))
    return res.metadata.get("served")


def test_effective_stack_bb_reads_table_depth():
    assert abs(PluribusReBeLSolver._effective_stack_bb(_preflop_env(100.0)) - 100.0) < 1e-6
    assert abs(PluribusReBeLSolver._effective_stack_bb(_preflop_env(200.0)) - 200.0) < 1e-6
    assert abs(PluribusReBeLSolver._effective_stack_bb(_preflop_env(40.0)) - 40.0) < 1e-6


def test_blueprint_depth_uses_action_relevant_pairwise_stack_not_table_minimum():
    """One short opponent must not disable a 127bb opening blueprint globally."""
    env = _preflop_env(127.0)
    hero = int(env.action_seat)
    short = next(s for s in range(NUM_PLAYERS) if s != hero)
    deep = next(s for s in range(NUM_PLAYERS) if s not in (hero, short))
    stacks = np.asarray(env.stacks).copy()
    total = np.zeros_like(np.asarray(env.committed_total))
    stacks[hero] = 127 * CHIP_UNIT
    stacks[short] = 40 * CHIP_UNIT
    stacks[deep] = 200 * CHIP_UNIT
    env = env._replace(stacks=stacks, committed_total=total)

    assert PluribusReBeLSolver._effective_stack_bb(env) == pytest.approx(40.0)
    assert PluribusReBeLSolver._preflop_relevant_stack_bb(
        env, hero) == pytest.approx(127.0)

    legal = legal_action_mask_np(env, hero).astype(bool)
    solver = PluribusReBeLSolver(
        _snapshot_row(env, hero),
        PluribusReBeLConfig(preflop_blueprint_band_bb=(80.0, 130.0)),
        rng=np.random.default_rng(0))
    assert solver._trusted_root_blueprint(env, hero, legal) is not None


def test_facing_short_aggressor_uses_that_pairwise_depth():
    env = _preflop_env(127.0)
    hero = int(env.action_seat)
    short = next(s for s in range(NUM_PLAYERS) if s != hero)
    stacks = np.asarray(env.stacks).copy()
    total = np.zeros_like(np.asarray(env.committed_total))
    stacks[hero] = 127 * CHIP_UNIT
    stacks[short] = 40 * CHIP_UNIT
    hist_seat = np.asarray(env.hist_seat).copy()
    hist_action = np.asarray(env.hist_action).copy()
    hist_seat[0], hist_action[0] = short, 5
    env = env._replace(
        stacks=stacks, committed_total=total, hist_len=np.int32(1),
        hist_seat=hist_seat, hist_action=hist_action)

    assert PluribusReBeLSolver._preflop_relevant_stack_bb(
        env, hero) == pytest.approx(40.0)


def test_in_band_serves_blueprint_offband_searches():
    band = (80.0, 130.0)
    assert _solve(100.0, band) == "blueprint_preflop"   # standard depth -> blueprint (unchanged)
    assert _solve(200.0, band) != "blueprint_preflop"   # deep -> falls through to search (A1)
    assert _solve(40.0, band) == "preflop_offdepth_prior"  # short -> trusted passive prior


def test_band_none_is_legacy_always_blueprint():
    assert _solve(200.0, None) == "blueprint_preflop"   # disabled band -> legacy fixed-100bb behaviour


def test_in_band_lookup_miss_uses_passive_prior_not_uniform_legal():
    assert _solve(100.0, (80.0, 130.0), source_hit=False) == "preflop_lookup_miss_prior"


def test_short_unraised_state_addresses_source_depth_row_then_projects_actions():
    """Short stacks may borrow a learned opening range, never 100bb terminal semantics."""
    from pokerbot.encoding.fingerprint_native import fingerprint_fast
    from pokerbot.engine.jax_weva_fast import default_weva_bucket_table

    short = _preflop_env(40.0)
    source = _preflop_env(100.0)
    hero = int(short.action_seat)
    assert hero == int(source.action_seat)
    cfg = PluribusReBeLConfig(
        preflop_blueprint_band_bb=(80.0, 130.0),
        short_unraised_preflop_projection=True,
    )
    solver = PluribusReBeLSolver(
        _snapshot_row(source, hero), cfg, rng=np.random.default_rng(0))
    shadow = solver._reference_depth_preflop_state(
        short, normalize_short=True)
    assert fingerprint_fast(
        shadow, hero, weva_table=default_weva_bucket_table()) == fingerprint_fast(
            source, hero, weva_table=default_weva_bucket_table())

    legal = legal_action_mask_np(short, hero).astype(bool)
    root = SubgameRoot(
        env=short, hero_seat=hero, pbs_vector=None, legal_mask=legal)
    result = solver.solve(root, _StubPBS(short, hero))

    assert result.metadata["served"] == "blueprint_preflop_short_projected"
    assert result.metadata["blueprint_source_hit"] is False
    assert result.metadata["reference_projection_source_hit"] is True
    assert result.metadata["projection_source_hit"] is True
    assert result.action_probs[-1] == pytest.approx(0.0)
    assert float(result.action_probs.sum()) == pytest.approx(1.0)


def test_short_projection_abstains_when_facing_aggression():
    """A 100bb response range is not silently reused against a short aggressor."""
    from pokerbot.search.native_solver import apply_action_np

    env = _preflop_env(40.0)
    opener = int(env.action_seat)
    legal = legal_action_mask_np(env, opener).astype(bool)
    raise_action = next(a for a in (7, 6, 5, 4, 3) if legal[a])
    env = apply_action_np(env, raise_action)
    hero = int(env.action_seat)
    legal = legal_action_mask_np(env, hero).astype(bool)
    cfg = PluribusReBeLConfig(
        preflop_blueprint_band_bb=(80.0, 130.0),
        short_unraised_preflop_projection=True,
    )
    solver = PluribusReBeLSolver({}, cfg, rng=np.random.default_rng(0))
    root = SubgameRoot(env=env, hero_seat=hero, pbs_vector=None, legal_mask=legal)

    result = solver.solve(root, _StubPBS(env, hero))

    assert result.metadata["served"] == "preflop_offdepth_prior"
    assert result.metadata["preflop_miss_opener_position"] is not None


def test_deep_direct_miss_retries_the_true_source_depth_key():
    """State projection must address the 100bb fingerprint, not merely use its
    legal menu after a lookup against the deep-SPR key."""
    from pokerbot.search.native_solver import apply_action_np
    from pokerbot.encoding.fingerprint_native import fingerprint_fast
    from pokerbot.engine.jax_weva_fast import default_weva_bucket_table

    def reach_after_raise_and_two_calls(env):
        first = int(env.action_seat)
        mask = legal_action_mask_np(env, first).astype(bool)
        raise_action = next(a for a in (7, 6, 5, 4, 3) if mask[a])
        env = apply_action_np(env, raise_action)
        for _ in range(2):
            mask = legal_action_mask_np(env, int(env.action_seat)).astype(bool)
            env = apply_action_np(env, 2 if mask[2] else 1)
        return env

    deep = reach_after_raise_and_two_calls(_preflop_env(200.0))
    source = reach_after_raise_and_two_calls(_preflop_env(100.0))
    hero = int(deep.action_seat)
    assert hero == int(source.action_seat)
    assert fingerprint_fast(
        deep, hero, weva_table=default_weva_bucket_table()) != fingerprint_fast(
            source, hero, weva_table=default_weva_bucket_table())
    legal = legal_action_mask_np(deep, hero).astype(bool)
    cfg = PluribusReBeLConfig(
        max_iterations=4,
        time_budget_ms=20_000,
        preflop_blueprint_band_bb=(80.0, 130.0),
        deep_preflop_projection=True,
        leaf=ContinuationLeafConfig(select="uniform", rollout_samples=1),
    )
    solver = PluribusReBeLSolver(
        _snapshot_row(source, hero), cfg, rng=np.random.default_rng(0))
    root = SubgameRoot(env=deep, hero_seat=hero, pbs_vector=None, legal_mask=legal)

    result = solver.solve(root, _StubPBS(deep, hero))

    assert result.metadata["served"] == "blueprint_preflop_deep_projected"
    assert result.metadata["blueprint_source_hit"] is False
    assert result.metadata["reference_projection_source_hit"] is True
    assert result.metadata["projection_source_hit"] is True
    assert float(result.action_probs.sum()) == pytest.approx(1.0)


def test_reference_depth_state_preserves_public_state_and_does_not_mutate_live():
    deep = _preflop_env(200.0)
    hero = int(deep.action_seat)
    solver = PluribusReBeLSolver(
        {}, PluribusReBeLConfig(preflop_blueprint_reference_bb=100.0),
        rng=np.random.default_rng(0))
    old_stacks = np.asarray(deep.stacks).copy()

    shadow = solver._reference_depth_preflop_state(deep)

    assert np.min(np.asarray(shadow.stacks) + np.asarray(shadow.committed_total)) == (
        100 * CHIP_UNIT)
    np.testing.assert_array_equal(shadow.committed_total, deep.committed_total)
    np.testing.assert_array_equal(shadow.committed_street, deep.committed_street)
    np.testing.assert_array_equal(shadow.hist_action, deep.hist_action)
    np.testing.assert_array_equal(shadow.hist_seat, deep.hist_seat)
    np.testing.assert_array_equal(deep.stacks, old_stacks)
    assert legal_action_mask_np(shadow, hero).any()


def test_material_high_exposure_blueprint_mass_routes_to_realtime_search():
    """A real row hit is not enough evidence for a 100bb terminal action.

    The trigger keys on material high-exposure mass, not merely argmax: even a
    fold-dominant row with 20% jam mass is resolved before an action selector can
    deterministically or stochastically expose that noisy tail.
    """
    env = _preflop_env(100.0)
    hero = int(env.action_seat)
    legal = legal_action_mask_np(env, hero).astype(bool)
    from pokerbot.encoding.fingerprint_native import fingerprint_fast
    from pokerbot.engine.jax_weva_fast import default_weva_bucket_table

    row = np.zeros(11, dtype=np.float32)
    row[0], row[10] = 0.8, 0.2
    key = fingerprint_fast(env, hero, weva_table=default_weva_bucket_table())
    cfg = PluribusReBeLConfig(
        max_iterations=4, time_budget_ms=20_000,
        preflop_blueprint_band_bb=(80.0, 130.0),
        offdepth_preflop_pluribus=True,
        preflop_risky_action_resolve=True,
        leaf=ContinuationLeafConfig(select="uniform", rollout_samples=1),
    )
    root = SubgameRoot(env=env, hero_seat=hero, pbs_vector=None, legal_mask=legal)
    res = PluribusReBeLSolver(
        {key: row}, cfg, rng=np.random.default_rng(0)).solve(
            root, _StubPBS(env, hero))

    assert res.metadata["served"] in ("search", "search_native")
    assert res.metadata["preflop_risky_resolve"] is True
    assert res.metadata["preflop_risky_action"] == 10
    assert res.metadata["preflop_risky_mass"] == pytest.approx(0.2)
    assert res.metadata["preflop_risky_commit_pot_ratio"] > 3.0


def test_low_exposure_blueprint_row_stays_on_fast_path():
    env = _preflop_env(100.0)
    hero = int(env.action_seat)
    legal = legal_action_mask_np(env, hero).astype(bool)
    from pokerbot.encoding.fingerprint_native import fingerprint_fast
    from pokerbot.engine.jax_weva_fast import default_weva_bucket_table

    row = np.zeros(11, dtype=np.float32)
    row[0], row[7] = 0.4, 0.6  # ordinary pot-sized raise, not >3 pots
    key = fingerprint_fast(env, hero, weva_table=default_weva_bucket_table())
    cfg = PluribusReBeLConfig(
        preflop_blueprint_band_bb=(80.0, 130.0),
        offdepth_preflop_pluribus=True,
        preflop_risky_action_resolve=True,
    )
    root = SubgameRoot(env=env, hero_seat=hero, pbs_vector=None, legal_mask=legal)
    res = PluribusReBeLSolver(
        {key: row}, cfg, rng=np.random.default_rng(0)).solve(
            root, _StubPBS(env, hero))
    assert res.metadata["served"] == "blueprint_preflop"


def test_trusted_blueprint_refuses_off_depth_preflop():
    cfg = PluribusReBeLConfig(preflop_blueprint_band_bb=(80.0, 130.0))
    deep = _preflop_env(500.0)
    hero = int(deep.action_seat)
    legal = legal_action_mask_np(deep, hero).astype(bool)
    solver = PluribusReBeLSolver(_snapshot_row(deep, hero), cfg,
                                 rng=np.random.default_rng(0))
    assert solver._trusted_root_blueprint(deep, hero, legal) is None

    normal = _preflop_env(100.0)
    hero = int(normal.action_seat)
    legal = legal_action_mask_np(normal, hero).astype(bool)
    solver = PluribusReBeLSolver(_snapshot_row(normal, hero), cfg,
                                 rng=np.random.default_rng(0))
    trusted = solver._trusted_root_blueprint(normal, hero, legal)
    assert trusted is not None
    assert float(np.asarray(trusted).sum()) == pytest.approx(1.0)


def test_trusted_blueprint_rejects_uniform_compatibility_fallback_on_any_street():
    cfg = PluribusReBeLConfig(preflop_blueprint_band_bb=(80.0, 130.0))
    solver = PluribusReBeLSolver({}, cfg, rng=np.random.default_rng(0))
    env = _preflop_env(100.0)
    hero = int(env.action_seat)
    legal = legal_action_mask_np(env, hero).astype(bool)
    assert solver._trusted_root_blueprint(env, hero, legal) is None

    # The lookup-status invariant is street-independent.  The adapter is
    # already a miss for this empty snapshot; changing only the street proves
    # postflop raw/under-search guards cannot trust its uniform fallback either.
    postflop = env._replace(street=np.int8(1))
    assert solver._trusted_root_blueprint(postflop, hero, legal) is None
    fallback, meta = solver._trusted_root_fallback(postflop, hero, legal)
    assert fallback is None
    assert meta["blueprint_source_hit"] is False


def test_trusted_offdepth_fallback_is_card_aware_and_never_aggressive():
    cfg = PluribusReBeLConfig(preflop_blueprint_band_bb=(80.0, 130.0))
    solver = PluribusReBeLSolver({}, cfg, rng=np.random.default_rng(0))
    env = _preflop_env(40.0)
    hero = int(env.action_seat)
    legal = legal_action_mask_np(env, hero).astype(bool)

    def with_hole(c0, c1):
        holes = np.asarray(env.holes).copy()
        holes[hero] = [c0, c1]
        return env._replace(holes=holes)

    # AQs (AcQc) continues for the cheap blind price; 93o (9c3d) folds.
    aq, aq_meta = solver._trusted_root_fallback(with_hole(48, 40), hero, legal)
    trash, trash_meta = solver._trusted_root_fallback(with_hole(28, 5), hero, legal)
    assert aq_meta["fallback_source"] == "preflop_miss_prior"
    assert aq_meta["preflop_miss_score"] >= aq_meta["preflop_miss_threshold"]
    assert int(np.argmax(aq)) == 2
    assert trash_meta["preflop_miss_score"] < trash_meta["preflop_miss_threshold"]
    assert int(np.argmax(trash)) == 0
    assert float(np.asarray(aq)[3:].sum()) == pytest.approx(0.0)
    assert float(np.asarray(trash)[3:].sum()) == pytest.approx(0.0)


def test_coverage_miss_wiring_recovers_btn_kjo_vs_hj_cheap_open():
    """Production geometry from cmrkyh1l: BTN KJo faced a 2.5bb HJ open.

    The blueprint row was missing. Position/price context must reach the one
    centralized passive prior, which calls but can never raise or jam.
    """
    env = _preflop_env(193.0)
    hero, opener = 5, 3  # button=5 => seat 3 is HJ on a six-seat ring
    holes = np.asarray(env.holes).copy()
    holes[hero] = [Card.from_str("Kh").index, Card.from_str("Jc").index]
    committed = np.zeros_like(np.asarray(env.committed_street))
    committed[opener] = int(2.5 * CHIP_UNIT)
    hist_seat = np.asarray(env.hist_seat).copy()
    hist_action = np.asarray(env.hist_action).copy()
    hist_seat[0], hist_action[0] = opener, 5  # aggressive bet bucket
    env = env._replace(
        button=np.int32(hero), action_seat=np.int32(hero), holes=holes,
        committed_street=committed, current_bet=np.int32(2.5 * CHIP_UNIT),
        hist_len=np.int32(1), hist_seat=hist_seat, hist_action=hist_action,
    )
    legal = np.zeros(11, dtype=bool)
    legal[[0, 2, 7, 10]] = True
    solver = PluribusReBeLSolver(
        {}, PluribusReBeLConfig(preflop_blueprint_band_bb=(80.0, 130.0)),
        rng=np.random.default_rng(0))

    probs, meta = solver._preflop_coverage_policy(env, hero, legal, 193.0)

    assert meta["preflop_miss_hero_position"] == "BTN"
    assert meta["preflop_miss_opener_position"] == "HJ"
    assert meta["preflop_miss_to_call_bb"] == 2.5
    assert meta["preflop_miss_threshold"] == 7.0
    assert int(np.argmax(probs)) == 2
    assert float(probs[3:].sum()) == pytest.approx(0.0)


def test_short_offdepth_wiring_uses_equity_override_for_aks_vs_co_open():
    """End-to-end regression for the live 291bb hero / short aggressor spot."""
    env = _preflop_env(291.0)
    hero, opener = 5, 4  # button=5; seat 4 is CO
    holes = np.asarray(env.holes).copy()
    holes[hero] = [Card.from_str("Ac").index, Card.from_str("Kc").index]
    committed = np.zeros_like(np.asarray(env.committed_street))
    committed[0] = int(0.5 * CHIP_UNIT)
    committed[1] = int(1.0 * CHIP_UNIT)
    committed[opener] = int(3.25 * CHIP_UNIT)
    stacks = np.asarray(env.stacks).copy()
    stacks[opener] = int(13.0 * CHIP_UNIT) - committed[opener]
    hist_seat = np.asarray(env.hist_seat).copy()
    hist_action = np.asarray(env.hist_action).copy()
    hist_seat[0], hist_action[0] = opener, 5
    env = env._replace(
        button=np.int32(hero), action_seat=np.int32(hero), holes=holes,
        stacks=stacks, committed_street=committed,
        current_bet=np.int32(3.25 * CHIP_UNIT), hist_len=np.int32(1),
        hist_seat=hist_seat, hist_action=hist_action,
    )
    legal = np.zeros(11, dtype=bool)
    legal[[0, 2, 7, 10]] = True
    solver = PluribusReBeLSolver(
        {}, PluribusReBeLConfig(preflop_blueprint_band_bb=(80.0, 130.0)),
        rng=np.random.default_rng(0))

    probs, meta = solver._preflop_coverage_policy(env, hero, legal, 13.0)

    assert meta["preflop_miss_opener_position"] == "CO"
    assert meta["preflop_miss_score"] == 12.0
    assert meta["preflop_miss_threshold"] == 14.0
    assert meta["preflop_miss_equity_override"] is True
    assert meta["preflop_miss_equity"] > meta["preflop_miss_required_equity"]
    assert int(np.argmax(probs)) == 2
    assert float(probs[3:].sum()) == pytest.approx(0.0)


def test_deep_projection_moves_allin_mass_to_largest_legal_raise():
    p = np.zeros(11, dtype=np.float64)
    p[2] = 0.37
    p[7] = 0.11
    p[10] = 0.52
    legal = np.zeros(11, dtype=bool)
    legal[[0, 2, 5, 7, 10]] = True

    projected, target, moved = PluribusReBeLSolver._project_deep_preflop_blueprint(p, legal)

    assert target == 7
    assert moved == pytest.approx(0.52)
    assert projected[10] == 0.0
    assert projected[7] == pytest.approx(0.63)
    assert projected.sum() == pytest.approx(1.0)


def test_deep_projection_maps_duplicate_allin_call_to_call_when_no_raise():
    p = np.zeros(11, dtype=np.float64)
    p[0] = 0.2
    p[2] = 0.3
    p[10] = 0.5
    legal = np.zeros(11, dtype=bool)
    legal[[0, 2, 10]] = True

    projected, target, moved = PluribusReBeLSolver._project_deep_preflop_blueprint(p, legal)

    assert target == 2
    assert moved == pytest.approx(0.5)
    assert projected[2] == pytest.approx(0.8)
    assert projected[10] == 0.0


def test_reference_depth_shadow_removes_live_93o_330bb_reraise_geometry():
    """Production-state regression for the live 1/2 line 15 -> 64 -> 195.

    At 500bb, the pot-sized menu exposes further enormous raises. In the same
    public state capped to the blueprint's 100bb training stack, hero has only
    fold/call/all-in: every nominal raise bucket is terminal semantics and must
    be projected out.
    """
    env = _preflop_env(500.0)
    hero = int(env.action_seat)
    opp = next(i for i in range(NUM_PLAYERS) if i != hero)
    stacks = np.asarray(env.stacks).copy()
    street = np.zeros_like(np.asarray(env.committed_street))
    total = np.zeros_like(np.asarray(env.committed_total))
    # Normalize wire chips by the 2-chip BB: hero raised to 64 (32bb), villain
    # reraised to 195 (97.5bb), leaving a 65.5bb call and min-reraise increment.
    street[hero], street[opp] = 3_200, 9_750
    total[:] = street
    stacks[hero] -= 3_200
    stacks[opp] -= 9_750
    acted = np.asarray(env.acted).copy()
    acted[hero] = True
    env = env._replace(
        stacks=stacks,
        committed_street=street,
        committed_total=total,
        current_bet=np.int32(9_750),
        min_raise=np.int32(6_550),
        acted=acted,
        last_raise_full=np.bool_(True),
        action_seat=np.int32(hero),
    )
    solver = PluribusReBeLSolver(
        {}, PluribusReBeLConfig(preflop_blueprint_band_bb=(80.0, 130.0)),
        rng=np.random.default_rng(0))
    live = np.asarray(solver._legal(env, hero)).astype(bool)
    source = solver._reference_depth_preflop_legal(env, hero)

    assert live[3:10].any()                    # 500bb menu offers the bad reraises
    assert not source[3:10].any()              # none existed nonterminally at 100bb
    assert source[0] and source[2] and source[10]


def test_reference_depth_shadow_keeps_normal_first_in_raise_buckets():
    """Regression for live QQ first-in becoming a limp with target=None.

    The shadow-action probe must call apply_action_np(state, action), not pass a
    redundant hero argument that throws and silently marks every raise terminal.
    """
    env = _preflop_env(262.0)
    hero = int(env.action_seat)
    solver = PluribusReBeLSolver(
        {}, PluribusReBeLConfig(
            preflop_blueprint_reference_bb=100.0,
            preflop_blueprint_nonterminal_spr_floor=0.5),
        rng=np.random.default_rng(0))
    source = solver._reference_depth_preflop_legal(env, hero)

    assert source[10]                         # all-in stays legal/terminal
    assert source[5:10].any()                 # normal 100bb raises stay nonterminal

    p = np.zeros(11)
    p[2], p[10] = 0.20, 0.80                 # live QQ-like limp + terminal mass
    live = legal_action_mask_np(env, hero).astype(bool)
    projected, target, moved = solver._project_deep_preflop_blueprint(
        p, live, source_hit=True, source_legal=source)
    assert target is not None and 5 <= target < 10
    assert moved == pytest.approx(0.80)
    assert int(np.argmax(projected)) == target
    assert projected[10] == 0.0


def test_deep_projection_route_fails_closed_on_blueprint_miss():
    env = _preflop_env(500.0)
    hero = int(env.action_seat)
    root = SubgameRoot(env=env, hero_seat=hero, pbs_vector=None,
                       legal_mask=legal_action_mask_np(env, hero).astype(bool))
    cfg = PluribusReBeLConfig(
        preflop_blueprint_band_bb=(80.0, 130.0),
        deep_preflop_projection=True,
    )

    res = PluribusReBeLSolver({}, cfg, rng=np.random.default_rng(0)).solve(
        root, _StubPBS(env, hero))

    assert res.metadata["served"] == "blueprint_preflop_deep_projected"
    assert res.metadata["projection_source_hit"] is False
    assert res.metadata["projected_target_action"] is None
    assert int(np.argmax(res.action_probs)) == 0  # facing the BB: fold, never raise


def test_deep_projection_route_preserves_real_row_provenance():
    from pokerbot.encoding.fingerprint_native import fingerprint_fast
    from pokerbot.engine.jax_weva_fast import default_weva_bucket_table

    env = _preflop_env(500.0)
    hero = int(env.action_seat)
    key = fingerprint_fast(env, hero, weva_table=default_weva_bucket_table())
    row = np.zeros(11, dtype=np.float32)
    row[0], row[2], row[9], row[10] = 0.1, 0.2, 0.3, 0.4
    root = SubgameRoot(env=env, hero_seat=hero, pbs_vector=None,
                       legal_mask=legal_action_mask_np(env, hero).astype(bool))
    cfg = PluribusReBeLConfig(
        preflop_blueprint_band_bb=(80.0, 130.0),
        deep_preflop_projection=True,
    )

    res = PluribusReBeLSolver({key: row}, cfg, rng=np.random.default_rng(0)).solve(
        root, _StubPBS(env, hero))

    assert res.metadata["projection_source_hit"] is True
    assert res.action_probs[10] == 0.0
    assert res.metadata["projected_allin_mass"] > 0.0
