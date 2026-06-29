"""Regression test for F-NATIVE-1 (tools/battery_native/src/search_session.rs:786).

THE BUG
=======
`SearchSession.solve(cancel_handle=...)` IGNORED the kwarg and created a fresh
`Arc<AtomicBool>::new(false)` instead of resolving the cancel_handle id against the
process-global `live::CancelRegistry`. Net effect: `live.set_cancel(handle_id)` never
flipped the AtomicBool the inner Rust loop was polling, so the watchdog fired
silently and an in-flight solve ran to `max_iterations` regardless of the Python-
side cancel. Live cancel contract broken on the rust path.

THE FIX
=======
`SearchSession.solve` now resolves the cancel_handle id against the same
`cancel_registry()` the SearchHandle (`live.rs`) writes to via `set_cancel(id)`. The
Arc<AtomicBool> it returns is the SAME one the recursion's per-iter cancel check
polls, so a watchdog-triggered `set_cancel(id)` is observed at the LOOP TOP and the
solve exits within ONE more iteration (bounded by `max_depth` recursion cost).

REGRESSION TEST CONTRACT
========================
1. Open a cancel handle via `register_cancel()`.
2. Build a SearchSession that will run for `max_iterations=4000` (slow enough that
   the test thread can call `set_cancel` mid-flight on any reasonable machine).
3. Kick the solve off in a worker thread.
4. Sleep 50 ms (so the worker is INSIDE the iteration loop).
5. Call `set_cancel(handle_id)` from the test thread.
6. Wait for the worker to return — it MUST do so within the cancel-deadline window.
7. Assert `result['budget_bound'] == 2` (the rust run_solve sets `budget_bound = 2`
   on the cancel path; 0 == ran to max, 1 == wall deadline, 2 == cancel).
8. Assert `result['iterations']` is well below `max_iterations` — proves the loop
   actually broke (a buggy implementation would return `iterations == max_iterations`).
9. Always `release_cancel(handle_id)` (registry cleanup).
10. ALSO assert the control case: with `cancel_handle=None` the solve runs to
    `max_iterations` (no spurious cancel).
"""

from __future__ import annotations

import os
import sys
import threading
import time
import importlib
import importlib.util
from pathlib import Path

import pytest


def _load_fresh_ext():
    """Same loader as test_search_session_equiv.py — target/release wins over root .pyd.
    A stale pyd that lacks the PyInit_battery_native symbol (e.g. from an interrupted
    cargo build) must NOT crash collection — we fall through to the standard loader
    which honors POKERBOT_NO_NATIVE_BATTERY / finds the root-level fallback."""
    if os.environ.get("POKERBOT_NO_NATIVE_BATTERY"):
        return None
    sys.modules.pop("battery_native", None)
    here = Path(__file__).resolve().parent.parent
    rel = here / "tools" / "battery_native" / "target" / "release"
    pyd = rel / "battery_native.pyd"
    so = rel / "battery_native.so"
    target = pyd if pyd.exists() else (so if so.exists() else None)
    if target is None:
        from pokerbot.search import native_solver as ns
        return ns._try_load()
    spec = importlib.util.spec_from_file_location("battery_native", str(target))
    if spec is None or spec.loader is None:
        from pokerbot.search import native_solver as ns
        return ns._try_load()
    try:
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    except (ImportError, OSError):
        # Stale / partially-built artifact (e.g. cargo build --features python still broken):
        # fall back to the standard loader which honors POKERBOT_NO_NATIVE_BATTERY.
        from pokerbot.search import native_solver as ns
        return ns._try_load()


_EXT = _load_fresh_ext()
_has_session = _EXT is not None and hasattr(_EXT, "SearchSession")
_has_cancel = _EXT is not None and all(
    hasattr(_EXT, n) for n in ("register_cancel", "set_cancel", "release_cancel")
)

needs_native = pytest.mark.skipif(
    not (_has_session and _has_cancel),
    reason="native SearchSession + register/set/release_cancel not built "
    "(cargo build --release --features python)",
)


# ---------------------------------------------------------------------------
# Build a session + a postflop state. Mirrors test_search_session_equiv's recipe.
# ---------------------------------------------------------------------------
def _build_session_and_state(seed: int = 7):
    """Return (session, state_bytes, hero, public_seed) for a captured postflop node."""
    import random
    from pokerbot.engine.actions import ActionType
    from pokerbot.engine.game import GameState, Street, play_hand
    from pokerbot.engine.np_engine import to_numpy_state
    from pokerbot.search.native_solver import _to_flat
    from pokerbot.search.pbs_tracker import PBSTracker
    from pokerbot.search.state_bridge import build_subgame_root

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
        return passive[0] if passive else legal[0]

    pbs = PBSTracker()
    gs = GameState(rng_seed=seed, num_players=6, small_blind=0.5, big_blind=1.0)
    play_hand(gs, rec_policy, pbs=pbs, max_actions=200)
    assert captured, "rec_policy failed to capture a postflop node"
    state, hero = captured[0]
    root = build_subgame_root(state, hero)
    env = to_numpy_state(state)
    state_bytes = _to_flat(env)
    public_seed = int(root["public_seed"])

    # Empty blueprint (uniform fallback over the legal mask); tilt + leaf = stage-1 GTO.
    session = _EXT.SearchSession(
        {},
        [True] * 11,
        [True] * 11,
        1.0,    # bias_mult
        0.0,    # bias_floor
        48,     # max_depth (PlayoutConfig)
        100,    # num_buckets
        0,      # session_seed
    )
    return session, state_bytes, hero, public_seed


@needs_native
def test_register_set_release_round_trip_round_trip_smoke():
    """Sanity: register_cancel returns a u64, set+release on a fresh id is a no-op after release."""
    handle = _EXT.register_cancel()
    assert isinstance(handle, int) and handle > 0
    _EXT.set_cancel(handle)
    _EXT.release_cancel(handle)
    # After release, set_cancel on the same id must NOT raise (it's a silent no-op).
    _EXT.set_cancel(handle)


@needs_native
def test_cancel_handle_breaks_long_solve_within_one_iteration():
    """THE REGRESSION GATE for F-NATIVE-1.

    The solve below is configured for `max_iterations=4000` — large enough that
    `set_cancel(handle)` fired 50ms after kickoff MUST land well before completion.
    Before the fix the rust path ignored `cancel_handle` entirely and ran to
    `max_iterations`; the assertion `iterations < max_iterations` and
    `budget_bound == 2` would both fail.
    """
    session, state_bytes, hero, public_seed = _build_session_and_state(seed=7)

    max_iterations = 4000
    max_depth = 16

    handle = _EXT.register_cancel()
    try:
        result_box: list = []
        error_box: list = []

        def worker():
            try:
                result_box.append(session.solve(
                    state_bytes, int(hero), int(public_seed),
                    int(max_iterations), int(max_depth),
                    1.0,    # blueprint_mix
                    4,      # rollout_samples
                    True,   # resample_board
                    True,   # linear_averaging
                    False, 1.5, 0.0, 2.0,  # dcfr off
                    None,           # tilt_table
                    None,           # cont_weights
                    0,              # time_budget_ms (no wall gate; cancel-only test)
                    handle,         # cancel_handle
                    0,              # worker_id
                ))
            except BaseException as e:
                error_box.append(e)

        t = threading.Thread(target=worker, daemon=True)
        t0 = time.monotonic()
        t.start()
        # Give the worker ~50ms to enter the iteration loop and burn a few iters.
        time.sleep(0.05)
        _EXT.set_cancel(handle)
        t.join(timeout=5.0)
        elapsed = time.monotonic() - t0

        assert not error_box, f"worker raised: {error_box[0]!r}"
        assert result_box, "worker did not return a result within 5s — solve did not cancel"
        res = result_box[0]

        # The core contract: cancel was honored.
        # run_solve sets budget_bound = 2 on the cancel path (see search_session.rs:380).
        assert res["budget_bound"] == 2, (
            f"expected budget_bound=2 (cancel), got {res['budget_bound']} "
            f"(iters={res['iterations']}, elapsed={elapsed:.3f}s) — "
            f"cancel_handle was likely ignored"
        )
        # And the loop actually broke — well short of max_iterations.
        assert res["iterations"] < max_iterations, (
            f"solve ran to max_iterations ({max_iterations}) despite set_cancel — "
            f"cancel_handle was IGNORED (F-NATIVE-1 regression)"
        )
        # Sanity: it ran at least a couple of iterations before the cancel landed.
        assert res["iterations"] >= 1, (
            f"solve returned 0 iters — too fast to have entered the loop"
        )
    finally:
        _EXT.release_cancel(handle)


@needs_native
def test_no_cancel_handle_runs_to_max_iterations():
    """Control case: with cancel_handle=None the solve runs to max_iterations (no spurious cancel)."""
    session, state_bytes, hero, public_seed = _build_session_and_state(seed=11)

    max_iterations = 64
    max_depth = 16

    res = session.solve(
        state_bytes, int(hero), int(public_seed),
        int(max_iterations), int(max_depth),
        1.0, 4, True,
        True,
        False, 1.5, 0.0, 2.0,
        None,           # tilt_table
        None,           # cont_weights
        0,              # time_budget_ms
        None,           # cancel_handle (no registry id)
        0,              # worker_id
    )
    # No cancel contract wired -> budget_bound=0 (ran to max), iterations==max.
    assert res["budget_bound"] == 0, (
        f"expected budget_bound=0 (ran to max) with cancel_handle=None, "
        f"got {res['budget_bound']}"
    )
    assert res["iterations"] == max_iterations, (
        f"expected iterations=={max_iterations} with cancel_handle=None, "
        f"got {res['iterations']}"
    )


@needs_native
def test_unknown_cancel_handle_id_falls_back_safely():
    """A bogus id (never registered) MUST NOT crash — it falls back to a fresh local-only
    AtomicBool(false), so the solve runs to max_iterations just like cancel_handle=None.
    """
    session, state_bytes, hero, public_seed = _build_session_and_state(seed=13)

    max_iterations = 32
    max_depth = 16

    # A u64 id that was never registered (register_cancel is monotonic, so an id
    # >> any current value is overwhelmingly likely to be unregistered).
    bogus_id = (1 << 62)

    res = session.solve(
        state_bytes, int(hero), int(public_seed),
        int(max_iterations), int(max_depth),
        1.0, 4, True,
        True,
        False, 1.5, 0.0, 2.0,
        None, None, 0,
        bogus_id,        # cancel_handle: unregistered id (safe fallback path)
        0,
    )
    assert res["budget_bound"] == 0, (
        f"unregistered cancel_handle id should behave like None (budget_bound=0); "
        f"got budget_bound={res['budget_bound']}"
    )
    assert res["iterations"] == max_iterations