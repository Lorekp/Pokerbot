"""TASK-75 (2026-07-13) guards: the native subgame solve is INTERRUPTIBLE and GIL-FREE.

Live wedge class this closes: `SubgameSolver.solve` was one non-interruptible FFI call that HELD
THE GIL for its full wall — at pathological deep-SPR geometry (flop pot 6 / stacks 900) a solve sat
9+ minutes, freezing health threads / watchdogs (one child became unkillable). The rebuilt binary
(a) releases the GIL for every serve-path solve and (b) accepts an in-native cooperative wall-clock
cap (`wall_cap_ms`, checked per CFR iteration + per 1024 tree nodes) returning a bounded, honest,
flagged result.

These tests exercise the REBUILT worktree binary through the same loader the serve uses
(pokerbot.search.native_solver); they use a tiny synthetic dict blueprint + the native
`init_hand` root — never the 32GB tier-1 index.
"""

from __future__ import annotations

import os
import threading
import time

import numpy as np
import pytest

os.environ.setdefault("POKERBOT_POSTFLOP_MODE", "fast")

from pokerbot.search import native_solver as ns  # noqa: E402

native_only = pytest.mark.skipif(
    not (ns.available() and ns._EXT is not None and hasattr(ns._EXT, "SubgameSolver")),
    reason="native SubgameSolver not built/available",
)
wall_cap_only = pytest.mark.skipif(
    not (ns.available() and ns.native_wall_cap_capable()),
    reason="battery_native binary lacks the solve_wall_cap capability (stale build)",
)


def _root_state_bytes():
    """A 6-max 50bb root straight from the native engine (no jax import needed)."""
    ext = ns._EXT
    deck = list(range(52))
    return bytes(ext.init_hand(0, [5000] * 6, [1] * 6, 0, deck, [0, 0]))


def _hero_of(state_bytes: bytes) -> int:
    env = ns._from_flat(state_bytes)
    return int(env.action_seat)


def _mk_engine():
    ext = ns._EXT
    return ext.SubgameSolver(
        {},  # empty blueprint -> uniform fallback (chip-EV base); tiny + hermetic
        [True] * 11, [True] * 11, 5.0, 0.0, 24, 200,
    )


# Positional args shared by solve / solve_capped (after state, hero, seed):
#   max_iterations, max_depth, blueprint_mix, rollout_samples, resample_board, linear_averaging,
#   repair_deck, two_sided, decouple_sampler, range_aware, range_uniform_floor, range_reach_full,
#   ensemble_streams, leaf_cache_worlds, avg_skip_frac, leaf_biased_continuations, leaf_bias_mult,
#   serve_final_iterate, root_unmask
def _solve_args(k: int):
    return (int(k), 8, 0.0, 2, True, True,
            False, False, False, False, 0.25, False,
            1, 1, 0.0, False, 5.0,
            False, 0)


@native_only
@wall_cap_only
def test_caps_report_wall_cap():
    assert "solve_wall_cap" in set(ns._EXT.native_caps())
    assert ns.native_wall_cap_capable()


@native_only
@wall_cap_only
def test_solve_capped_cap_off_bit_identical_to_solve():
    """wall_cap=0 (and a huge, unexpired cap) must be BYTE-IDENTICAL to the legacy solve."""
    eng = _mk_engine()
    sb = _root_state_bytes()
    hero = _hero_of(sb)
    legacy = eng.solve(sb, hero, 0xC0FFEE, *_solve_args(32))
    p0, capped0, it0 = eng.solve_capped(sb, hero, 0xC0FFEE, *_solve_args(32), 0, 0)
    assert [x for x in legacy] == [x for x in p0], "solve_capped(cap=0) must equal solve exactly"
    assert capped0 is False
    assert it0 == 32
    ph, cappedh, ith = eng.solve_capped(sb, hero, 0xC0FFEE, *_solve_args(32), 600_000, 0)
    assert [x for x in legacy] == [x for x in ph], "an unexpired cap must not change the solve"
    assert cappedh is False and ith == 32


@native_only
@wall_cap_only
def test_wall_cap_hit_bounded_and_flagged():
    """An absurd iteration target under a small cap returns promptly, flagged, still valid."""
    eng = _mk_engine()
    sb = _root_state_bytes()
    hero = _hero_of(sb)
    t0 = time.perf_counter()
    p, capped, iters = eng.solve_capped(sb, hero, 42, *_solve_args(50_000_000), 200, 0)
    wall_s = time.perf_counter() - t0
    assert capped is True, "a 200ms cap under a 50M-iteration target must flag capped"
    assert 0 <= iters < 50_000_000
    assert wall_s < 10.0, f"cap-hit solve must return promptly (took {wall_s:.1f}s)"
    arr = np.asarray(p, dtype=np.float64)
    assert np.all(np.isfinite(arr)) and abs(arr.sum() - 1.0) < 1e-6


@native_only
@wall_cap_only
def test_raise_cap_valid_and_default_off_identical():
    eng = _mk_engine()
    sb = _root_state_bytes()
    hero = _hero_of(sb)
    legacy = eng.solve(sb, hero, 7, *_solve_args(32))
    p0, _, _ = eng.solve_capped(sb, hero, 7, *_solve_args(32), 0, 0)
    assert [x for x in legacy] == [x for x in p0]
    p2, capped2, it2 = eng.solve_capped(sb, hero, 7, *_solve_args(32), 0, 2)
    assert capped2 is False and it2 == 32
    arr = np.asarray(p2, dtype=np.float64)
    assert np.all(np.isfinite(arr)) and abs(arr.sum() - 1.0) < 1e-6


@native_only
@wall_cap_only
def test_gil_released_watchdog_thread_runs_during_solve():
    """The wedge-class regression test: a watchdog-style Python thread MUST keep running while a
    long native solve is in flight. Pre-change, the solve held the GIL for its full wall, so the
    thread could log NO ticks inside the solve window; post-change (py.detach) it ticks freely."""
    eng = _mk_engine()
    sb = _root_state_bytes()
    hero = _hero_of(sb)

    ticks: list = []
    stop = threading.Event()

    def _watchdog():
        while not stop.is_set():
            ticks.append(time.perf_counter())
            time.sleep(0.01)

    th = threading.Thread(target=_watchdog, daemon=True)
    th.start()
    time.sleep(0.05)  # let the thread reach steady state
    t0 = time.perf_counter()
    # ~1.5s solve, structurally bounded by the wall cap (never unbounded even on a fast box).
    p, capped, _iters = eng.solve_capped(sb, hero, 42, *_solve_args(50_000_000), 1500, 0)
    t1 = time.perf_counter()
    stop.set()
    th.join(timeout=5.0)

    assert capped is True
    assert (t1 - t0) >= 0.5, "the solve must actually have run long enough to test liveness"
    # Ticks strictly INSIDE the solve window (margins exclude entry/exit races). GIL-held would be 0.
    inside = [x for x in ticks if t0 + 0.1 < x < t1 - 0.1]
    assert len(inside) >= 10, (
        f"watchdog thread logged only {len(inside)} ticks inside a {t1 - t0:.2f}s native solve — "
        "the GIL is not being released"
    )
    arr = np.asarray(p, dtype=np.float64)
    assert np.all(np.isfinite(arr)) and abs(arr.sum() - 1.0) < 1e-6


@native_only
@wall_cap_only
def test_native_subgame_solve_cap_info_plumbing():
    """The serve wrapper (native_solver.native_subgame_solve) must route wall_cap_ms through
    solve_capped and surface wall_capped/iters_completed via cap_info."""

    class _FakeAdapter:
        snapshot: dict = {}
        _allowed_pre = [True] * 11
        _allowed_post = [True] * 11
        bias_mult = 5.0
        bias_floor = 0.0

    if not ns.native_subgame_available():
        pytest.skip("native subgame path disabled in this process")
    sb = _root_state_bytes()
    env = ns._from_flat(sb)
    hero = int(env.action_seat)
    adapter = _FakeAdapter()
    ci: dict = {}
    probs = ns.native_subgame_solve(
        env, hero, adapter, public_seed=42, max_iterations=50_000_000, max_depth=8,
        blueprint_mix=0.0, rollout_samples=2, resample_board=True, max_playout_depth=24,
        num_buckets=200, wall_cap_ms=200, cap_info=ci,
    )
    assert probs is not None
    assert ci.get("wall_cap_supported") is True
    assert ci.get("wall_capped") is True
    assert 0 <= int(ci.get("iters_completed", -1)) < 50_000_000
    arr = np.asarray(probs, dtype=np.float64)
    assert np.all(np.isfinite(arr)) and abs(arr.sum() - 1.0) < 1e-6

    # cap off: legacy call, cap_info untouched (no keys written).
    ci2: dict = {}
    probs2 = ns.native_subgame_solve(
        env, hero, adapter, public_seed=42, max_iterations=16, max_depth=8,
        blueprint_mix=0.0, rollout_samples=2, resample_board=True, max_playout_depth=24,
        num_buckets=200, wall_cap_ms=0, cap_info=ci2,
    )
    assert probs2 is not None
    assert ci2 == {}
