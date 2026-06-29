"""Regression test for F-NATIVE-8: ``SearchHandle.solve`` must return a real root EV.

Bug: ``tools/battery_native/src/live.rs:618`` (the ``Ok((result_probs.to_vec(), 0.0, iters, d))``
return at the tail of ``solve``) hard-coded the second tuple element to ``0.0``. The live
decision log + coach NLHE row read this slot as "expected value under the converged CFR+
policy" and stored ``0.0`` for every decision — making EV-adjusted statistics
(``scripts/live/live_stats.py`` with ``POKERBOT_EV_ADJUSTED=1``) and the coach row's EV
column permanently wrong.

Fix: capture the root EV from the final completed hero iteration at depth 0 via an
``AtomicU64`` bit-cast (Ungil-safe for ``py.detach``), return it as the second tuple
element, and mirror it into a new ``last_root_ev()`` accessor on the SearchHandle for
the wrapper to read AFTER ``solve`` returns (same shape as ``last_budget_used_ms`` /
``last_iters`` / ``last_served_kind``).

This test pins the SEARCH HANDLE PATH (the live-driver Rust surface that the bug
named) — NOT the Python ``PluribusReBeLSolver.solve`` path (which has its own
hard-coded ``ev=0.0`` at multiple return sites; that is a separate fix to wire the
EV into the public Pluribus API).

Pins:
  (1) ``SearchHandle.solve`` returns an EV that is NOT hard-coded 0.0 on a
      representative postflop root where the hero has a non-trivial hand.
  (2) The returned EV is finite (not NaN/Inf from a bad rollout).
  (3) The returned EV matches ``metadata['root_ev']`` and the post-call accessor
      ``handle.last_root_ev()``.
  (4) The EV is bounded by ``hero_stack + pot`` (the maximum the hero can realize
      at this spot — the upper safety bound).
  (5) Repeated solves on the SAME (state, public_seed) return the SAME EV
      (CRN parity — the Rust xoshiro seeded from ``public_seed`` reproduces the
      same Monte-Carlo stream across calls).

Skip behavior: the test gates on ``native_search_handle_available()`` so it cleanly
skips on a build without the python feature (the rlib / no-python path).
"""

from __future__ import annotations

import math
import os
import sys

os.environ.setdefault("POKERBOT_BETTING_ABSTRACTION", "1")
os.environ.setdefault("POKERBOT_POSTFLOP_BUCKETS", "1")

# Make the freshly built battery_native extension importable from the release build dir
# (other Python tests go through `pokerbot.search.native_solver` which does the same).
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
_BN_RELEASE = os.path.join(_REPO, "tools", "battery_native", "target", "release")
if _BN_RELEASE not in sys.path:
    sys.path.insert(0, _BN_RELEASE)

import numpy as np
import pytest

from pokerbot.engine.actions import ActionType
from pokerbot.engine.game import GameState, Street, play_hand
from pokerbot.engine.np_engine import to_numpy_state
from pokerbot.search import native_solver as ns
from pokerbot.search import pluribus_search as ps
from pokerbot.search.pbs_tracker import PBSTracker
from pokerbot.search.pluribus_rollout import ContinuationLeafConfig
from pokerbot.search.pluribus_search import PluribusReBeLConfig, PluribusReBeLSolver
from pokerbot.search.state_bridge import build_subgame_root


_handle_skipif = pytest.mark.skipif(
    not ns.native_search_handle_available(),
    reason="native SearchHandle not built (cargo build --release --features python)",
)


# ---------------------------------------------------------------------------
# Fixture: drive a hand to a postflop root via a passive recording policy.
# Mirrors test_native_search_wiring._postflop_root so this test stands alone.
# ---------------------------------------------------------------------------
def _postflop_root(seed: int = 11):
    """Play random hands with a passive policy; return the first captured postflop root."""
    import copy
    import random as _random

    rng = _random.Random(seed)
    captured: list = []

    def rec_policy(state):
        legal = state.legal_actions()
        seat = state.action_seat
        if (not captured and seat is not None and seat >= 0
                and state.street != Street.PREFLOP and state.players[seat].hole):
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
        st = GameState.new_hand(button_seat=3, rng=_random.Random(rng.randint(0, 10 ** 9)))
        play_hand({s: rec_policy for s in range(6)}, state=st,
                  rng=_random.Random(rng.randint(0, 10 ** 9)))
    if not captured:
        raise RuntimeError("could not reach a postflop node")
    state, hero = captured[0]
    pbs = PBSTracker.init_uniform(hero, state.players[hero].hole, state.board)
    root = build_subgame_root(state, pbs, hero)
    env = to_numpy_state(root.env)
    return env, hero, root, pbs


def _make_handle(blueprint: dict, *, num_actions: int = 11, num_buckets: int = 200):
    """Build a Rust ``SearchHandle`` directly (no PluribusReBeLSolver wrapper).

    The handle's solve() return tuple's second element is the EV slot we are
    regression-testing. Bypassing PluribusReBeLSolver.solve sidesteps the
    separate (out-of-scope) Python-side ``ev=0.0`` hard-codes."""
    import battery_native as bn

    # All 11 actions allowed at preflop + postflop (the test root is postflop; post-mask
    # is what the native side actually consults via abstracted_legal). Toggling doesn't
    # affect the EV slot — the recursion just picks the same actions either way.
    allowed_pre = [True] * num_actions
    allowed_post = [True] * num_actions
    return bn.SearchHandle(
        blueprint=blueprint,
        allowed_pre=allowed_pre,
        allowed_post=allowed_post,
        bias_mult=1.0,
        bias_floor=0.0,
        leaf_max_depth=4,
        num_buckets=num_buckets,
        dcfr=False,
        dcfr_alpha=1.5,
        dcfr_beta=0.0,
        dcfr_gamma=2.0,
        linear_averaging=True,
        blueprint_cv=False,
    )


def _env_to_flat_bytes(env) -> bytes:
    """Pack a numpy EnvState into the raw little-endian int64 bytes the Rust side reads.

    Uses ``native_solver._to_flat`` (the canonical packing) so the test doesn't drift
    from the live wire format when the buffer layout evolves. Falls back to the test-
    local mirror if the extension isn't available — but at that point the test will
    already have been skipped by ``_handle_skipif``."""
    from pokerbot.search.native_solver import _to_flat
    return np.ascontiguousarray(_to_flat(env)).tobytes()


def _public_seed(env, hero: int) -> int:
    """Mirror ``PluribusReBeLSolver._public_seed`` so the Rust side computes the same
    CRN root seed we feed in via ``public_seed=``."""
    from pokerbot.search.pluribus_search import PluribusReBeLSolver
    solver = PluribusReBeLSolver(
        {bytes([0] * 16): np.full(11, 1.0 / 11.0, dtype=np.float64)},
        PluribusReBeLConfig(),
    )
    return int(solver._public_seed(env, hero))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
@_handle_skipif
def test_search_handle_solve_returns_nonzero_ev_on_postflop_root():
    """The ``solve`` return tuple's second element MUST NOT be hard-coded 0.0.

    On a representative postflop root (passive, 100bb deep) the converged CFR+
    root policy on a non-trivial hand has non-zero EV. Pre-fix the EV was 0.0 by
    construction; the fix captures the last-iteration hero node EV from the Rust
    ``cfv_iter`` recursion.
    """
    ns.enable()
    env, hero, root, pbs = _postflop_root(seed=11)
    blueprint = {bytes([0] * 16): np.full(11, 1.0 / 11.0, dtype=np.float64)}
    handle = _make_handle(blueprint)
    state_bytes = _env_to_flat_bytes(env)
    public_seed = _public_seed(env, hero)

    probs, ev, iters, meta = handle.solve(
        state=state_bytes,
        hero=hero,
        public_seed=public_seed,
        max_iterations=20,
        max_depth=12,
        blueprint_mix=0.5,
        rollout_samples=2,
        resample_board=True,
        time_budget_ms=10 ** 9,
    )

    # CRITICAL: the pre-fix return was hard-coded 0.0. On a postflop root with a
    # non-trivial hand (passive stations + random cards), the realized hero node
    # EV is strictly non-zero in chips. We assert |ev| > 1e-3 chips (CHIP_UNIT is
    # 1; the leaf playout returns chips in the same scale).
    assert math.isfinite(ev), f"solve returned a non-finite EV ({ev!r})"
    assert abs(float(ev)) > 1e-3, (
        f"solve() returned ev=0.0 (the pre-fix hard-coded value) on a non-trivial "
        f"postflop root. F-NATIVE-8 regression — root EV is not being captured from "
        f"the recursion.")

    # Sanity bound: a root EV greater than pot+stack is implausible.
    pot = int(env.pot_dead) + int(np.sum(env.committed_street))
    hero_stack = int(env.stacks[hero])
    upper_safety_bound = float(pot + hero_stack)
    assert -upper_safety_bound < float(ev) < upper_safety_bound, (
        f"solve returned ev={ev} which exceeds the {upper_safety_bound}-chip "
        f"plausibility bound (pot={pot}, hero_stack={hero_stack})")

    # Shape contract: probs is an 11-vector summing to 1 over the legal mask.
    p = np.asarray(probs, dtype=np.float64)
    assert p.shape == (11,)
    assert abs(float(p.sum()) - 1.0) < 1e-6, f"probs do not sum to 1 (sum={p.sum()})"
    assert iters == 20, f"expected 20 iters, got {iters}"


@_handle_skipif
def test_search_handle_solve_ev_matches_metadata_and_accessor():
    """The returned EV MUST agree with ``metadata['root_ev']`` AND with the
    post-call accessor ``handle.last_root_ev()`` (the third read path the
    live decision log uses for non-blocking post-call reads)."""
    ns.enable()
    env, hero, root, pbs = _postflop_root(seed=23)
    blueprint = {bytes([0] * 16): np.full(11, 1.0 / 11.0, dtype=np.float64)}
    handle = _make_handle(blueprint)
    state_bytes = _env_to_flat_bytes(env)
    public_seed = _public_seed(env, hero)

    probs, ev_return, iters, meta = handle.solve(
        state=state_bytes, hero=hero, public_seed=public_seed,
        max_iterations=20, max_depth=12, blueprint_mix=0.5,
        rollout_samples=2, resample_board=True, time_budget_ms=10 ** 9,
    )
    ev_meta = float(meta.get("root_ev", float("nan")))
    ev_accessor = float(handle.last_root_ev())

    assert math.isfinite(ev_return) and math.isfinite(ev_meta) and math.isfinite(ev_accessor), (
        f"non-finite EV: return={ev_return} metadata={ev_meta} accessor={ev_accessor}")
    assert abs(ev_return - ev_meta) < 1e-12, (
        f"return EV ({ev_return}) and metadata['root_ev'] ({ev_meta}) disagree")
    assert abs(ev_return - ev_accessor) < 1e-12, (
        f"return EV ({ev_return}) and handle.last_root_ev() ({ev_accessor}) disagree")


@_handle_skipif
def test_search_handle_solve_ev_reproducible_under_crn_seed():
    """Repeated solves on the SAME (state, public_seed) MUST return the SAME EV
    (CRN parity — the Rust xoshiro seeded from ``public_seed`` reproduces the
    same Monte-Carlo stream across calls)."""
    ns.enable()
    env, hero, root, pbs = _postflop_root(seed=77)
    blueprint = {bytes([0] * 16): np.full(11, 1.0 / 11.0, dtype=np.float64)}
    handle = _make_handle(blueprint)
    state_bytes = _env_to_flat_bytes(env)
    public_seed = _public_seed(env, hero)

    probs1, ev1, iters1, meta1 = handle.solve(
        state=state_bytes, hero=hero, public_seed=public_seed,
        max_iterations=20, max_depth=12, blueprint_mix=0.5,
        rollout_samples=2, resample_board=True, time_budget_ms=10 ** 9,
    )
    probs2, ev2, iters2, meta2 = handle.solve(
        state=state_bytes, hero=hero, public_seed=public_seed,
        max_iterations=20, max_depth=12, blueprint_mix=0.5,
        rollout_samples=2, resample_board=True, time_budget_ms=10 ** 9,
    )
    assert math.isfinite(ev1) and math.isfinite(ev2)
    assert abs(ev1 - ev2) < 1e-12, (
        f"EV is not CRN-reproducible: call 1={ev1} vs call 2={ev2} on the same root")