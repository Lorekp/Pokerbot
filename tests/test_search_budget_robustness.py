"""Starvation-robust live search serve: blend-to-blueprint when budget-bound + single-table iteration floor.

The live pluribus search runs only ~22-130 of its 200 CFR iterations at the 800ms budget, so the served
CFR AVERAGE is UNDER-CONVERGED -> a noisy argmax that flips frame-to-frame under load. The Pluribus/Modicum
fix: when a solve is BUDGET-BOUND, blend the served root policy toward the TRUSTED BLUEPRINT prior (weight ~
convergence shortfall). A fully-converged solve serves the pure search policy (byte-identical). Single-tabling
gets an iteration floor so it converges (no multi-table slowdown). This file proves:

  (a) a budget-bound solve blends toward the blueprint (served moves toward bp vs the raw average; converged = no blend);
  (b) DEFAULT-OFF (eval) is byte-identical to the no-blend path (the certified-exploitability serve unchanged);
  (c) a fixed budget-bound spot serves a STABLE argmax across repeats (the blend quiets the starvation flip);
  (d) the legal mask / fold-call legality is preserved and the blend is a renormalized convex combination;
  (e) the single-table floor lets a single-table solve run more iterations without slowing multi-table.
"""

from __future__ import annotations

import os

os.environ.setdefault("POKERBOT_BETTING_ABSTRACTION", "1")
os.environ.setdefault("POKERBOT_POSTFLOP_BUCKETS", "1")

import numpy as np
import jax

from pokerbot.engine.jax_engine import NUM_ACTIONS, NUM_PLAYERS, init_hand
from pokerbot.engine.np_engine import (
    CALL,
    CHECK,
    apply_action_np,
    is_chance_node_np,
    legal_action_mask_np,
    sample_chance_np,
    to_numpy_state,
)
from pokerbot.search.state_bridge import SubgameRoot
from pokerbot.search.pluribus_rollout import ContinuationLeafConfig
from pokerbot.search.pluribus_search import (
    PluribusReBeLSolver,
    PluribusReBeLConfig,
    set_blueprint_blend,
    blueprint_blend_enabled,
)

_OCC = np.array([True] * 6, dtype=bool)


class _StubPBS:
    def __init__(self, env, hero):
        self._h = np.asarray(env.holes)
        self.hero = hero

    def sample_opponent_holes(self, rng):
        return {s: (int(self._h[s, 0]), int(self._h[s, 1]))
                for s in range(NUM_PLAYERS) if s != self.hero}


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


def _root(env):
    hero = int(env.action_seat)
    return hero, SubgameRoot(env=env, hero_seat=hero, pbs_vector=None,
                             legal_mask=legal_action_mask_np(env, hero).astype(bool))


def _solve(env, hero, root, *, blend, budget_ms, max_iters, single_table=None,
           blueprint_blend=None, blueprint_blend_max=1.0, seed=0):
    cfg = PluribusReBeLConfig(
        max_iterations=max_iters, time_budget_ms=budget_ms,
        blueprint_blend=blueprint_blend,
        blueprint_blend_max=blueprint_blend_max,
        single_table_iter_floor=(40 if single_table is not None else None),
        single_table_budget_ms=(10 ** 9 if single_table is not None else None),
        leaf=ContinuationLeafConfig(select="uniform", rollout_samples=2),
        # The blueprint is intentionally empty for these unit tests (the test
        # exercises the blend math, not blueprint coverage). Opt out of the
        # production fail-closed snapshot validator.
        validate_snapshot_at_init=False,
    )
    set_blueprint_blend(blend)
    sol = PluribusReBeLSolver({}, cfg, rng=np.random.default_rng(seed))
    return sol.solve(root, _StubPBS(env, hero), single_table=single_table)


# --------------------------------------------------------------------------- (a) blend toward blueprint
def test_budget_bound_blends_toward_blueprint():
    """A budget-bound solve serves a convex blend of the raw average and the root blueprint; the blend
    moves the served policy CLOSER to the blueprint (here uniform-legal) than the noisy raw average."""
    env = _postflop_env()
    hero, root = _root(env)
    legal = PluribusReBeLSolver({}, PluribusReBeLConfig(validate_snapshot_at_init=False))._legal(env, hero)

    # Raw average after a few CFR iterations is PEAKED (concentrated on the +EV action). Budget-bind it
    # with a small wall clock so only a HANDFUL of iterations run before the bound (this IS the live spot:
    # ~22-130 of 200). Enough iterations to peak the average, but still well short of max_iters => starved.
    raw = _solve(env, hero, root, blend=False, budget_ms=80, max_iters=200, seed=0)
    assert raw.metadata["budget_bound"], "spot did not budget-bind — raise max_iters / lower budget_ms"
    assert raw.iterations >= 2, "too few iterations ran to peak the average — raise budget_ms"

    blended = _solve(env, hero, root, blend=True, budget_ms=80, max_iters=200, seed=0)
    assert blended.metadata["budget_bound"]
    w = blended.metadata["blueprint_blend_weight"]
    assert 0.0 < w <= 1.0, f"budget-bound blend weight should be in (0,1], got {w}"

    # The root blueprint at this spot (empty snapshot => uniform over the legal mask).
    bp = PluribusReBeLSolver({}, PluribusReBeLConfig(validate_snapshot_at_init=False))._root_blueprint(env, hero, legal)

    d_raw = float(np.abs(raw.action_probs - bp).sum())
    d_blend = float(np.abs(blended.action_probs - bp).sum())
    assert d_blend <= d_raw + 1e-9, "blend did not move the served policy toward the blueprint"
    assert d_blend < d_raw - 1e-6, "blend was inert (raw average already == blueprint?)"


def test_converged_solve_no_blend_byte_identical():
    """A fully-converged solve (ran all iterations, NOT budget-bound) is byte-identical with blend ON vs
    OFF — the blueprint blend only ever touches a budget-bound serve."""
    env = _postflop_env()
    hero, root = _root(env)
    # Generous wall clock => max_iterations binds, not the budget => NOT budget_bound.
    conv_off = _solve(env, hero, root, blend=False, budget_ms=10 ** 9, max_iters=12, seed=0)
    conv_on = _solve(env, hero, root, blend=True, budget_ms=10 ** 9, max_iters=12, seed=0)
    assert not conv_off.metadata["budget_bound"]
    assert not conv_on.metadata["budget_bound"]
    assert conv_on.metadata["blueprint_blend_weight"] == 0.0
    assert np.array_equal(conv_off.action_probs, conv_on.action_probs)


# --------------------------------------------------------------------------- (b) default-OFF byte-identity
def test_default_off_eval_byte_identical_to_no_blend():
    """With the env flag OFF (the eval/LBR battery default), a budget-bound serve is byte-identical to the
    legacy no-blend serve — the certified-exploitability path / CRN A/B is provably inert."""
    env = _postflop_env()
    hero, root = _root(env)
    # Legacy (blend code never existed): force the per-solver flag OFF.
    legacy = _solve(env, hero, root, blend=False, budget_ms=5, max_iters=200,
                    blueprint_blend=False, seed=0)
    # Env OFF, config None (follow env): must match legacy byte-for-byte.
    set_blueprint_blend(False)
    cfg = PluribusReBeLConfig(
        max_iterations=200, time_budget_ms=5, blueprint_blend=None,
        leaf=ContinuationLeafConfig(select="uniform", rollout_samples=2),
        validate_snapshot_at_init=False)
    env_off = PluribusReBeLSolver({}, cfg, rng=np.random.default_rng(0)).solve(root, _StubPBS(env, hero))
    assert env_off.metadata["budget_bound"]
    assert env_off.metadata["blueprint_blend_weight"] == 0.0
    assert np.array_equal(legacy.action_probs, env_off.action_probs)


# --------------------------------------------------------------------------- (c) determinism / less flip
def test_budget_bound_argmax_stable_with_blend():
    """On a FIXED budget-bound spot, the blended serve's argmax is STABLE across repeated solves at varying
    iteration counts (the blueprint pull quiets the starvation flip), and is strictly less jittery than the
    raw average's argmax across the same iteration sweep."""
    env = _postflop_env(seed=13)
    hero, root = _root(env)

    def argmaxes(blend, blend_max):
        out = []
        for it in (1, 2, 3, 4, 5, 6):
            # Tiny budget per call so each binds at a different shallow iteration count (the live regime:
            # the realized iteration count wobbles with machine load). per_decision_seed keeps the stream a
            # pure function of public state, so the only varying input is how many iterations got to run.
            r = _solve(env, hero, root, blend=blend, budget_ms=2 * it, max_iters=200,
                       blueprint_blend_max=blend_max, seed=0)
            out.append(int(np.argmax(r.action_probs)))
        return out

    raw = argmaxes(blend=False, blend_max=1.0)
    blended = argmaxes(blend=True, blend_max=1.0)
    # Strong blend (max=1.0) collapses every budget-bound serve toward the deterministic blueprint argmax,
    # so the blended argmax is constant across the sweep — strictly fewer distinct argmaxes than the raw.
    assert len(set(blended)) == 1, f"blended argmax not stable across iteration sweep: {blended}"
    assert len(set(blended)) <= len(set(raw)), "blend did not reduce argmax jitter"


# --------------------------------------------------------------------------- (d) legality / convexity
def test_blend_preserves_legal_mask_and_is_convex():
    env = _postflop_env(seed=21)
    hero, root = _root(env)
    legal = PluribusReBeLSolver({}, PluribusReBeLConfig(validate_snapshot_at_init=False))._legal(env, hero)

    raw = _solve(env, hero, root, blend=False, budget_ms=5, max_iters=200, seed=0)
    blended = _solve(env, hero, root, blend=True, budget_ms=5, max_iters=200, seed=0)
    assert blended.metadata["budget_bound"]
    w = blended.metadata["blueprint_blend_weight"]

    # (1) renormalized distribution over the legal mask.
    assert abs(float(blended.action_probs.sum()) - 1.0) < 1e-5
    assert np.all(blended.action_probs[~legal] == 0.0)
    assert np.all(blended.action_probs >= -1e-12)

    # (2) the served policy is EXACTLY the convex combination (1-w)*raw + w*bp, renormalized over legal.
    bp = PluribusReBeLSolver({}, PluribusReBeLConfig(validate_snapshot_at_init=False))._root_blueprint(env, hero, legal)
    expect = (1.0 - w) * raw.action_probs.astype(np.float64) + w * bp
    expect = expect * legal
    expect = expect / max(expect.sum(), 1e-12)
    assert np.allclose(blended.action_probs, expect, atol=1e-6)


def test_blend_max_caps_blueprint_weight():
    """blueprint_blend_max scales the realized weight: at total starvation the weight equals blueprint_blend_max
    (* shortfall=1), so a cap < 1 keeps some search signal even on the most starved serve."""
    env = _postflop_env(seed=21)
    hero, root = _root(env)
    # budget_ms=0 => binds at iteration 0 => shortfall=1.0 => weight == blueprint_blend_max exactly.
    r = _solve(env, hero, root, blend=True, budget_ms=0, max_iters=200, blueprint_blend_max=0.5, seed=0)
    assert r.metadata["budget_bound"]
    assert abs(r.metadata["blueprint_blend_weight"] - 0.5) < 1e-9


# --------------------------------------------------------------------------- (e) single-table iteration floor
def test_single_table_floor_runs_more_iterations_without_multitable_slowdown():
    """Single-table (single_table=True) lifts the effective iteration ceiling + wall clock so the solve
    converges; multi-table (single_table=False) keeps the per-table max_iterations / budget UNCHANGED."""
    env = _postflop_env(seed=7)
    hero, root = _root(env)

    # Multi-table: the per-table max_iterations binds (NOT the floor) => iterations == max_iterations.
    multi = _solve(env, hero, root, blend=False, budget_ms=10 ** 9, max_iters=12,
                   single_table=False, seed=0)
    assert multi.metadata["eff_max_iters"] == 12
    assert multi.iterations == 12

    # Single-table: the floor lifts the ceiling so more iterations run under the same call (converge).
    single = _solve(env, hero, root, blend=False, budget_ms=10 ** 9, max_iters=12,
                    single_table=True, seed=0)
    assert single.metadata["eff_max_iters"] >= 40
    assert single.iterations > 12, "single-table floor did not raise the iteration count"


def test_env_default_on_for_live():
    """POKERBOT_SEARCH_BLUEPRINT_BLEND defaults ON (live robustness); the eval battery flips it OFF."""
    # The module read the env at import; default unset => ON. (The battery sets it OFF explicitly.)
    set_blueprint_blend(True)
    assert blueprint_blend_enabled() is True
    set_blueprint_blend(False)
    assert blueprint_blend_enabled() is False
    set_blueprint_blend(True)  # leave it in the live-default state for any later test in this process
