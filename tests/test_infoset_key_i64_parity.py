"""KEY-EQUALITY GATE for the native CFV-under-exploitation port (POKERBOT_NATIVE_CFV).

The native tilt table (the firewalled exploit rows, carried AS DATA) is keyed by an i64 infoset-key
vector. Python builds that key with ``PluribusReBeLSolver._infoset_key_i64``; Rust looks it up with
``subgame::infoset_key``. These are two INDEPENDENT producers of ONE canonical encoding — any drift
silently mis-applies the tilt to the WRONG opponent infoset (the -559 serve-skew class of disaster).

This test asserts ``tuple(_infoset_key_i64(env, seat)) == SubgameSolver.debug_infoset_key(flat, seat)``
exactly over a large fuzz of REACHABLE (env, seat) across ALL four streets, pinning the two known drift
points: the spr_bucket truncation (Python ``int()`` vs Rust ``as i64``) and the revealed-board filter
(c >= 0). It also fuzzes ``_public_seed`` byte-identity hooks (determinism contract the relaxable RNG
surface depends on). It is SKIPPED when the native SubgameSolver is not built.
"""

from __future__ import annotations

import os
import random

import numpy as np
import pytest

os.environ.setdefault("POKERBOT_PREFLOP_LOSSLESS", "1")
os.environ.setdefault("POKERBOT_SEARCH_PREFLOP", "1")

from pokerbot.engine.actions import ActionType  # noqa: E402
from pokerbot.engine.game import GameState, play_hand  # noqa: E402
from pokerbot.engine.np_engine import to_numpy_state  # noqa: E402
from pokerbot.search import native_solver as ns  # noqa: E402
from pokerbot.search.pbs_tracker import PBSTracker  # noqa: E402
from pokerbot.search.pluribus_rollout import ContinuationLeafConfig  # noqa: E402
from pokerbot.search.pluribus_search import PluribusReBeLConfig, PluribusReBeLSolver  # noqa: E402
from pokerbot.search.state_bridge import build_subgame_root, game_state_to_env  # noqa: E402

native_only = pytest.mark.skipif(
    not ns.native_subgame_available(), reason="native SubgameSolver not built/available"
)


def _solver() -> PluribusReBeLSolver:
    cfg = PluribusReBeLConfig(
        max_iterations=4, time_budget_ms=10 ** 9, max_depth=48, blueprint_mix=0.5,
        client_bet_sizing=True, search_preflop=True,
        leaf=ContinuationLeafConfig(select="uniform", rollout_samples=1, max_playout_depth=48,
                                    base_seed=0, resample_board=True),
        per_decision_seed=True, native=True,
    )
    return PluribusReBeLSolver({}, cfg)


def _native_engine(solver: PluribusReBeLSolver):
    """A native SubgameSolver instance (we only call its read-only debug_infoset_key)."""
    from pokerbot.engine.jax_weva_encode import _POSTFLOP_NUM_BUCKETS

    eng = ns._build_subgame_engine(solver.adapter, int(_POSTFLOP_NUM_BUCKETS), 48)
    assert eng is not None and hasattr(eng, "debug_infoset_key")
    return eng


def _collect_nodes(n_target: int, seed: int):
    """Play random hands; capture (env, seat) at EVERY decision node across all four streets.

    Returns a list of (numpy_env, seat). Uses game_state_to_env(hero=seat) so the captured env is the
    public state at that actor (holes are irrelevant to the key — it reads only public fields)."""
    rng = random.Random(seed)
    nodes: list = []
    streets_seen: set = set()

    while len(nodes) < n_target:
        st = GameState.new_hand(button_seat=rng.randint(0, 5),
                                rng=random.Random(rng.randint(0, 10 ** 9)))

        def rec_policy(state):
            legal = state.legal_actions()
            seat = state.action_seat
            if seat is not None and seat >= 0 and legal:
                env = game_state_to_env(state, int(seat))
                nodes.append((to_numpy_state(env), int(seat)))
                streets_seen.add(int(state.street.value if hasattr(state.street, "value")
                                     else state.street))
            if not legal:
                return None
            # Mixed policy so we reach varied bet sizes / multiway lines.
            passive = [a for a in legal if a.action_type in (ActionType.CHECK, ActionType.CALL)]
            if passive and rng.random() < 0.6:
                return rng.choice(passive)
            return rng.choice(legal)

        play_hand({s: rec_policy for s in range(6)}, state=st,
                  rng=random.Random(rng.randint(0, 10 ** 9)))
    return nodes, streets_seen


@native_only
def test_infoset_key_i64_parity_fuzz():
    """>=10k reachable (env, seat) across all streets: Python i64 key == native infoset_key, exactly."""
    solver = _solver()
    eng = _native_engine(solver)
    nodes, streets_seen = _collect_nodes(10_000, seed=20240624)
    assert len(nodes) >= 10_000
    # All four streets must be represented — the revealed-board filter is the drift point that only bites
    # postflop; an all-preflop fuzz would not exercise it. (Street enum is 1-indexed: PREFLOP..RIVER.)
    assert len(streets_seen) >= 4, f"streets covered: {sorted(streets_seen)}"

    mismatches = 0
    for env, seat in nodes:
        py_key = tuple(solver._infoset_key_i64(env, seat))
        flat = ns._to_flat(env).tobytes()
        rs_key = tuple(int(x) for x in eng.debug_infoset_key(flat, int(seat)))
        if py_key != rs_key:
            mismatches += 1
            if mismatches <= 5:
                print("KEY MISMATCH seat", seat, "\n  py:", py_key, "\n  rs:", rs_key)
    assert mismatches == 0, f"{mismatches}/{len(nodes)} infoset-key mismatches"


@native_only
def test_infoset_key_i64_spr_extremes():
    """Pin the spr_bucket drift point: a near-zero active_mean (deep all-folded) and a very large pot
    must still agree byte-for-byte (truncate-toward-zero on a non-negative ratio, both sides)."""
    solver = _solver()
    eng = _native_engine(solver)
    st = GameState.new_hand(button_seat=2)
    seat = int(st.action_seat)
    env = to_numpy_state(game_state_to_env(st, seat))

    # Mutate pot / stacks to force extreme SPR ratios while keeping the engine layout valid.
    for pot_dead, stack in [(0, 1), (1, 1), (999999, 1), (12345, 7), (0, 100000)]:
        holes = np.asarray(env.holes).copy()
        e2 = env._replace(
            pot_dead=np.asarray(pot_dead, dtype=np.int32),
            stacks=np.asarray([stack] * 6, dtype=np.int32),
            holes=holes,
        )
        py_key = tuple(solver._infoset_key_i64(e2, seat))
        rs_key = tuple(int(x) for x in eng.debug_infoset_key(ns._to_flat(e2).tobytes(), seat))
        assert py_key == rs_key, (pot_dead, stack, py_key, rs_key)


@native_only
def test_infoset_key_hist_len_overflow_no_panic():
    """A hand-corrupted state with hist_len > MAX_HISTORY (64) must NOT abort the native process.

    The native `infoset_key` iterates `0..hist_len` over a fixed `[i32; 64]` array; before the clamp
    fix an over-max hist_len indexed out of bounds and panicked (under `panic = "abort"` that takes
    down the whole serve process), whereas Python merely raises a catchable IndexError. Engine-produced
    states always satisfy hist_len <= 64 (apply_action caps it), so this only guards a corrupt flat
    buffer — but a process abort on the serve path is unacceptable. The native side now clamps to 64 and
    returns a key instead of aborting. (We don't assert key-equality here: the state is out-of-domain
    and Python refuses it; the contract is only that the native call SURVIVES.)
    """
    solver = _solver()
    eng = _native_engine(solver)
    st = GameState.new_hand(button_seat=2)
    seat = int(st.action_seat)
    env = to_numpy_state(game_state_to_env(st, seat))
    for hl in (64, 65, 70, 200):
        e2 = env._replace(hist_len=np.asarray(hl, dtype=np.int32))
        flat = ns._to_flat(e2).tobytes()
        key = eng.debug_infoset_key(flat, seat)  # must not panic/abort
        assert key is not None and len(key) > 0
        if hl <= 64:
            # On the reachable domain the clamp is a no-op: still byte-identical to Python.
            py_key = tuple(solver._infoset_key_i64(ns._from_flat(flat), seat))
            assert tuple(int(x) for x in key) == py_key, hl


@native_only
def test_public_seed_byte_identity():
    """The relaxable RNG/float surface relies on a stable per-public-state seed: the SAME public node
    must always hash to the SAME seed (determinism contract). Fuzz >=1000 roots: re-deriving the seed
    on a fresh solver gives the identical value (independent of process history)."""
    s1 = _solver()
    s2 = _solver()
    rng = random.Random(99)
    for _ in range(1000):
        st = GameState.new_hand(button_seat=rng.randint(0, 5),
                                rng=random.Random(rng.randint(0, 10 ** 9)))
        hero = int(st.action_seat) if st.action_seat is not None else st.button_seat
        env = to_numpy_state(game_state_to_env(st, hero))
        assert s1._public_seed(env, hero) == s2._public_seed(env, hero)
