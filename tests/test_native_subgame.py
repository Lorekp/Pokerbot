"""Guards for the native CFR subgame fast path (pokerbot.search.pluribus_search + native_solver).

These are CHEAP invariants (not the statistical-equivalence gate, which lives in
tools/battery_native/crosscheck_subgame.py and the e2e timing): the native path must produce a valid
masked probability distribution, must be OFF by default, and must NEVER be taken when an opponent
model is set (the exploitation firewall — the native solver only implements the GTO base).
"""

from __future__ import annotations

import os
import random

import numpy as np
import pytest

os.environ.setdefault("POKERBOT_PREFLOP_LOSSLESS", "1")
os.environ.setdefault("POKERBOT_SEARCH_PREFLOP", "1")

from pokerbot.engine.actions import ActionType  # noqa: E402
from pokerbot.engine.game import GameState, Street, play_hand  # noqa: E402
from pokerbot.search import native_solver as ns  # noqa: E402
from pokerbot.search import pluribus_search as ps  # noqa: E402
from pokerbot.search.pbs_tracker import PBSTracker  # noqa: E402
from pokerbot.search.pluribus_rollout import ContinuationLeafConfig  # noqa: E402
from pokerbot.search.pluribus_search import PluribusReBeLConfig, PluribusReBeLSolver  # noqa: E402
from pokerbot.search.state_bridge import build_subgame_root  # noqa: E402

native_only = pytest.mark.skipif(
    not ns.native_subgame_available(), reason="native SubgameSolver not built/available"
)


def _postflop_root(seed: int = 7):
    """Play random hands via ``play_hand`` (advances streets + deals board) with a passive recording
    policy; return the first captured postflop (GameState, hero)."""
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


def _make_solver(iters=12, opponent_model=None):
    # Tiny snapshot is fine — the solver falls back to uniform blueprint on misses; we only check the
    # output is a valid distribution and the routing, not the strategy content.
    cfg = PluribusReBeLConfig(
        max_iterations=iters, time_budget_ms=10 ** 9, max_depth=48, blueprint_mix=0.5,
        client_bet_sizing=True, search_preflop=True,
        leaf=ContinuationLeafConfig(select="uniform", rollout_samples=2, max_playout_depth=48,
                                    base_seed=0, resample_board=True),
        per_decision_seed=True, native=True,
    )
    return PluribusReBeLSolver({}, cfg, opponent_model=opponent_model)


@native_only
def test_native_subgame_valid_distribution():
    st, hero = _postflop_root()
    pbs = PBSTracker.init_uniform(hero, st.players[hero].hole, st.board)
    root = build_subgame_root(st, pbs, hero)
    solver = _make_solver()
    ps.set_native_subgame(True)
    try:
        res = solver.solve(root, pbs)
    finally:
        ps.set_native_subgame(False)
    p = np.asarray(res.action_probs, dtype=np.float64)
    assert res.metadata.get("served") == "search_native"
    assert np.all(np.isfinite(p))
    assert abs(p.sum() - 1.0) < 1e-6
    # mass only on legal actions
    legal = solver._legal(__import__("pokerbot.engine.np_engine", fromlist=["to_numpy_state"])
                          .to_numpy_state(root.env), hero)
    assert np.all(p[~legal] == 0.0)


@native_only
def test_native_subgame_skipped_with_opponent_model():
    """Firewall: with an opponent model set, the native (GTO-only) path must NOT be taken."""
    st, hero = _postflop_root(seed=11)
    pbs = PBSTracker.init_uniform(hero, st.players[hero].hole, st.board)
    root = build_subgame_root(st, pbs, hero)
    # A minimal non-None opponent model (content irrelevant; presence flips the firewall).
    solver = _make_solver(opponent_model={s: None for s in range(6) if s != hero})
    ps.set_native_subgame(True)
    try:
        res = solver.solve(root, pbs)
    finally:
        ps.set_native_subgame(False)
    assert res.metadata.get("served") != "search_native"


def test_native_subgame_off_by_default():
    """Default routing must be the Python loop unless explicitly enabled."""
    st, hero = _postflop_root(seed=5)
    pbs = PBSTracker.init_uniform(hero, st.players[hero].hole, st.board)
    root = build_subgame_root(st, pbs, hero)
    solver = _make_solver()
    ps.set_native_subgame(False)
    res = solver.solve(root, pbs)
    assert res.metadata.get("served") != "search_native"
