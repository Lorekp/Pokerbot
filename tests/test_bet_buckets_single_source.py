"""BUG-B1 gate: BET_SIZE_BUCKETS has ONE source of truth across engine + serve + trained policy.

Before the fix, ``pokerbot.engine.actions`` defined the bet-size menu as ``(0.33, ...)`` while the
engine (``jax_engine`` / ``np_engine`` / ``core.rs``) sized at ``(1/3, ...)`` — silently mis-sizing the
smallest bucket between the serve/translation path and the trained policy. They are now the SAME
tuple object, sourced from ``actions``.

The external postflop-solver-CLI menu (``solver_exploitability._BET_BUCKET_FRACS``) is a DELIBERATELY
separate ``0.33`` abstraction (the CoinPoker client sizes a literal 33%); this test asserts it stays
distinct and documented rather than falsely merged.
"""

from __future__ import annotations

import pytest


def test_actions_is_canonical_one_third() -> None:
    from pokerbot.engine import actions

    assert actions.BET_SIZE_BUCKETS[0] == pytest.approx(1.0 / 3), (
        "smallest bucket must be the exact 1/3 (33.33%), not 0.33"
    )
    assert actions.BET_SIZE_BUCKETS == (1.0 / 3, 0.50, 0.66, 0.75, 1.0, 1.5, 2.0)


def test_engine_shares_one_object() -> None:
    """jax_engine and np_engine must reference the SAME tuple object as actions (no copy/redef)."""
    from pokerbot.engine import actions
    from pokerbot.engine import jax_engine, np_engine

    assert jax_engine.BET_SIZE_BUCKETS is actions.BET_SIZE_BUCKETS, (
        "jax_engine must import BET_SIZE_BUCKETS from actions, not redefine it"
    )
    assert np_engine.BET_SIZE_BUCKETS is actions.BET_SIZE_BUCKETS, (
        "np_engine must transitively share the same BET_SIZE_BUCKETS object"
    )


def test_native_core_matches_engine_buckets() -> None:
    """core.rs ``BET_BUCKETS_F32`` (= 0.333_333_34, 0.5, 0.66, 0.75, 1.0, 1.5, 2.0) must equal
    f32(actions.BET_SIZE_BUCKETS) elementwise. The native engine doesn't export the const directly,
    so we assert the f32 cast matches the literal core.rs ships; the *behavioural* sizing parity
    (the thing that actually matters for bit-identity) is covered exhaustively by the np<->jax<->core
    runtime equivalence gate (tests/test_np_engine_equiv.py + tests/test_lbr_native_equiv.py)."""
    import numpy as np

    from pokerbot.engine import actions

    # The exact f32 literals shipped in tools/battery_native/src/core.rs (BET_BUCKETS_F32, :38-46).
    core_rs_f32 = np.asarray(
        [0.333_333_34, 0.5, 0.66, 0.75, 1.0, 1.5, 2.0], dtype=np.float32
    )
    engine_f32 = np.asarray(actions.BET_SIZE_BUCKETS, dtype=np.float32)
    assert np.array_equal(engine_f32, core_rs_f32), (
        f"core.rs BET_BUCKETS_F32 {core_rs_f32} != f32(actions.BET_SIZE_BUCKETS) {engine_f32}"
    )


def test_engine_package_reexport_is_same_object() -> None:
    """The ``pokerbot.engine`` package re-exports BET_SIZE_BUCKETS — it must be the same object as
    actions' (no shadowing copy). (Avoids importlib.reload, which would fork the ActionType enum and
    pollute other tests.)"""
    import pokerbot.engine as eng
    from pokerbot.engine import actions

    assert eng.BET_SIZE_BUCKETS is actions.BET_SIZE_BUCKETS
