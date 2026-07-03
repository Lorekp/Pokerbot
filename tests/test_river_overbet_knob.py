"""STAGE-1 KNOB-EFFECTIVENESS PROBE (MANDATORY, tree-divergence rule) — river_overbet.

The 2026-07-02 tree-divergence lesson: a config knob is presumed a SILENT NO-OP until a probe shows
it moves the served policy on EVERY path the battery/live will exercise. This gate proves the
Stage-1 river overbet root menu (artifacts/offtree_resolve_design.md §3/§8) is effective on BOTH the
Python `_cfv` loop (config.native=False) and the native subgame solve (root_unmask through the FFI):

  * knob ON  -> the served root policy at an ENGINE-GENERATED river check-to-hero node puts
    NONZERO mass on bet bucket 6 (2.0x pot, action index BET_BUCKET_START+6);
  * knob OFF -> that index carries EXACTLY zero mass (the static keep-list drops the bucket);
  * the installed battery_native.SubgameSolver.solve signature contains ``root_unmask`` (BUG-4c:
    an armed battery arm against a stale pyd would silently fall back to the Python loop);
  * the armed NATIVE serve reports served=="search_native" (no silent fallback).

STANDING RULE (ensemble/preflop-search verdicts 2026-07-01): game-flow states ONLY — the roots are
collected by walking REAL engine deals (GameState.play_hand) to a river node where the hero is
first-in or facing a check; hand-built postflop nodes hang the native path and are banned.
"""
from __future__ import annotations

import os
import random
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

# ---------------------------------------------------------------------------
# CRITICAL: POKERBOT_* env vars must be set BEFORE any pokerbot import (the abstraction flags are
# read at import time). Deployed EMD-K160 regime — copied from
# scripts/probe_live_reach_replay_perturb.py; setdefault so a pinned external env wins.
# ---------------------------------------------------------------------------
os.environ.setdefault("JAX_PLATFORMS", "cpu")
_FROZEN = os.path.join(REPO, "data", "abstraction", "emd_frozen_flop_turn_160_r400.emdfrzn")
for _k, _v in {
    "POKERBOT_BETTING_ABSTRACTION": "1",
    "POKERBOT_POSTFLOP_BUCKETS": "1",
    "POKERBOT_POSTFLOP_NUM_BUCKETS": "160",
    "POKERBOT_POSTFLOP_MODE": "emd",
    "POKERBOT_POSTFLOP_EMD_FROZEN": _FROZEN,
    "POKERBOT_EMD_FROZEN_MIRROR_FALLBACK": "1",
    "POKERBOT_PREFLOP_LOSSLESS": "1",
}.items():
    os.environ.setdefault(_k, _v)

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from pokerbot.engine.actions import ActionType  # noqa: E402
from pokerbot.engine.jax_engine import BET_BUCKET_START  # noqa: E402
from pokerbot.engine.game import GameState, Street, play_hand  # noqa: E402
from pokerbot.engine.np_engine import (  # noqa: E402
    evaluate_seven, legal_action_mask_np, to_numpy_state,
)
from pokerbot.search import native_solver as ns  # noqa: E402
from pokerbot.search import pluribus_search as ps  # noqa: E402
from pokerbot.search.pbs_tracker import PBSTracker  # noqa: E402
from pokerbot.search.pluribus_rollout import ContinuationLeafConfig  # noqa: E402
from pokerbot.search.pluribus_search import PluribusReBeLConfig, PluribusReBeLSolver  # noqa: E402
from pokerbot.search.state_bridge import build_subgame_root  # noqa: E402

OVERBET_IDX = BET_BUCKET_START + 6  # bet bucket 6 == 2.0x pot == action index 9
K_ITERS = 500

native_only = pytest.mark.skipif(
    not ns.native_subgame_available(), reason="native SubgameSolver not built/available"
)


def _collect_river_checkto_roots(n: int = 6, seed: int = 11):
    """Walk REAL engine deals (game-flow states only) to river nodes where the hero is first-in or
    facing a check (to_call == 0) and the 2x-pot bucket is engine-LEGAL. Returns
    [(GameState, hero, strength)] sorted hero-strongest first (prefer a value-heavy root so the
    overbet child is a meaningful action, not only a structural one)."""
    rng = random.Random(seed)
    captured: list = []

    def rec_policy(state):
        legal = state.legal_actions()
        seat = state.action_seat
        if (seat is not None and seat >= 0 and state.street == Street.RIVER
                and state.players[seat].hole and len(captured) < 64):
            import copy
            captured.append((copy.deepcopy(state), int(seat)))
        if not legal:
            return None
        passive = [a for a in legal if a.action_type in (ActionType.CHECK, ActionType.CALL)]
        if passive and rng.random() < 0.85:
            return rng.choice(passive)
        return rng.choice(legal)

    for _ in range(600):
        if len(captured) >= 32:
            break
        st = GameState.new_hand(button_seat=rng.randint(0, 5),
                                rng=random.Random(rng.randint(0, 10 ** 9)))
        play_hand({s: rec_policy for s in range(6)},
                  state=st, rng=random.Random(rng.randint(0, 10 ** 9)))

    roots = []
    for st, hero in captured:
        pbs = PBSTracker.init_uniform(hero, st.players[hero].hole, st.board)
        root = build_subgame_root(st, pbs, hero)
        env = to_numpy_state(root.env)
        to_call = int(env.current_bet) - int(np.asarray(env.committed_street)[hero])
        if to_call > 0:
            continue  # facing a bet -> the offtree/raise lane, out of Stage-1 scope
        full = legal_action_mask_np(env, hero).astype(bool)
        if not bool(full[OVERBET_IDX]):
            continue  # 2x pot exceeds the stack here -> bucket 6 engine-illegal, not a probe root
        # Hero 7-card strength (higher = stronger) for the strongest-first preference.
        hole = np.asarray(env.holes)[hero]
        board = np.asarray(env.board)
        seven = np.concatenate([hole, board]).astype(np.int64)
        strength = int(evaluate_seven(seven)) if (seven >= 0).all() else 0
        roots.append((st, hero, strength))
    roots.sort(key=lambda t: -t[2])
    return roots[:n]


def _make_solver(*, native: bool, river_overbet: bool) -> PluribusReBeLSolver:
    cfg = PluribusReBeLConfig(
        max_iterations=K_ITERS, time_budget_ms=10 ** 9, max_depth=48, blueprint_mix=0.5,
        client_bet_sizing=True, search_preflop=True,
        leaf=ContinuationLeafConfig(select="uniform", rollout_samples=2, max_playout_depth=48,
                                    base_seed=0, resample_board=True),
        per_decision_seed=True, native=native, river_overbet=river_overbet,
    )
    return PluribusReBeLSolver({}, cfg, opponent_model=None)


def _solve_at(st, hero, *, native: bool, river_overbet: bool):
    pbs = PBSTracker.init_uniform(hero, st.players[hero].hole, st.board)
    root = build_subgame_root(st, pbs, hero)
    ps.set_native_subgame(native)
    sv = _make_solver(native=native, river_overbet=river_overbet)
    res = sv.solve(root, pbs)
    return np.asarray(res.action_probs, dtype=np.float64), res.metadata


@pytest.fixture(scope="module")
def river_root():
    roots = _collect_river_checkto_roots()
    if not roots:
        pytest.skip("could not reach a river check-to-hero node with a legal 2x-pot bet")
    return roots[0][:2]


def test_ffi_signature_contains_root_unmask():
    """BUG-4c: the installed pyd must accept root_unmask BEFORE any battery arm sets the knob."""
    import inspect
    assert ns._EXT is not None and hasattr(ns._EXT, "SubgameSolver"), "native extension not loaded"
    sig = inspect.signature(ns._EXT.SubgameSolver.solve)
    assert "root_unmask" in sig.parameters, (
        f"stale battery_native pyd: solve signature lacks root_unmask ({sig})"
    )


def test_python_path_knob_on_serves_overbet_off_zero(river_root):
    st, hero = river_root
    on, meta_on = _solve_at(st, hero, native=False, river_overbet=True)
    assert meta_on.get("river_overbet_armed") is True, f"knob ON did not arm: {meta_on}"
    assert on[OVERBET_IDX] > 0.0, (
        f"Python path: knob ON must put mass on the 2x-pot bucket, got {on}"
    )
    off, _ = _solve_at(st, hero, native=False, river_overbet=False)
    assert off[OVERBET_IDX] == 0.0, (
        f"Python path: knob OFF must keep the dropped bucket at exactly zero, got {off}"
    )


@native_only
def test_native_path_knob_on_serves_overbet_off_zero(river_root):
    st, hero = river_root
    on, meta_on = _solve_at(st, hero, native=True, river_overbet=True)
    # The armed serve must actually be NATIVE (no silent fallback to the Python loop — the
    # BUG-4c dead-knob trap this whole test exists to catch).
    assert meta_on.get("served") == "search_native", (
        f"armed river_overbet solve fell back off the native path: {meta_on}"
    )
    assert meta_on.get("river_overbet_armed") is True, f"knob ON did not arm: {meta_on}"
    assert on[OVERBET_IDX] > 0.0, (
        f"native path: knob ON must put mass on the 2x-pot bucket, got {on}"
    )
    off, meta_off = _solve_at(st, hero, native=True, river_overbet=False)
    assert meta_off.get("served") == "search_native", f"knob-OFF solve not native: {meta_off}"
    assert off[OVERBET_IDX] == 0.0, (
        f"native path: knob OFF must keep the dropped bucket at exactly zero, got {off}"
    )
