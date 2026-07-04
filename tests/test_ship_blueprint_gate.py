"""The serve-skew firewall gate in scripts/ship_blueprint.py REFUSES a mismatched abstraction.

This is the -559 bb/100 guard: if the snapshot's stamped abstraction does not field-match the active
serve encoder config, every blueprint lookup misses serve and the bot plays a beatable strategy. The
tool must raise ValidationError (and never activate) in that case.

The test imports ship_blueprint via importlib (it lives under scripts/, not an installed package) and
drives validate_serve() with (1) a stamp that mismatches the active serve config and (2) an empty
snapshot — both must be refused — without loading the 20MB native checkpoint.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent


def _load_ship():
    spec = importlib.util.spec_from_file_location("ship_blueprint", ROOT / "scripts" / "ship_blueprint.py")
    m = importlib.util.module_from_spec(spec)
    # ship_blueprint sets the serve abstraction env at import via module-level constants only; the
    # actual env-set happens in main(). Set it here so active_abstraction_config matches a real serve.
    spec.loader.exec_module(m)
    return m


def _pin_serve_config(monkeypatch):
    """Pin the BAKED serve config so the gate tests are import-order independent.

    ``jax_weva_encode`` bakes the POKERBOT_* abstraction env into module constants AT IMPORT. If any
    other test module in the same pytest process imports pokerbot first (with no abstraction env set),
    betting_abstraction=False is baked and ship_blueprint._set_serve_env() — a setdefault that runs
    only later, inside the test — can no longer re-bake it. validate_serve's check (a) then fires
    ("set POKERBOT_BETTING_ABSTRACTION=1 ...") before the check each test actually targets.

    Monkeypatching the baked constants (the same pattern as test_cash_serve_gate._set_serve_env)
    makes ``active_abstraction_config()`` — and therefore both ``_stamp_for_snapshot()`` and the
    check (a) cross-check, which read the SAME live config — deterministic under any import order.
    The gate's fail-closed semantics are untouched; only the test process's baked env is pinned.
    """
    from pokerbot.encoding import weva_encoder
    from pokerbot.engine import jax_weva_encode as enc

    monkeypatch.setattr(enc, "_BETTING_ABSTRACTION", True)
    monkeypatch.setattr(enc, "_POSTFLOP_BUCKETS_ENABLED", True)
    monkeypatch.setattr(enc, "_POSTFLOP_MODE", "fast")
    monkeypatch.setattr(enc, "_POSTFLOP_NUM_BUCKETS", 200)
    monkeypatch.setattr(enc, "_DEPTH_BUCKETS", False)
    monkeypatch.setattr(weva_encoder, "_PREFLOP_LOSSLESS", True)
    # validate_serve's check (0) (EMD frozen-blob SHA256) reads this env var directly. It is
    # orthogonal to the checks under test, and the fast-mode config above carries no EMD stamp
    # fields — drop it so the tests stay hermetic even after another test ran _set_serve_env().
    monkeypatch.delenv("POKERBOT_POSTFLOP_EMD_FROZEN", raising=False)


def _tiny_real_snapshot(ship) -> dict:
    """A minimal but real {fingerprint -> row} table: one key fingerprinted from a fresh 6-max env.

    Using the REAL fingerprint keeps the coverage probe meaningful (the key actually hits serve), so
    the gate's pass/fail hinges on the STAMP cross-check — which is what this test targets.
    """
    import jax

    from pokerbot.engine.jax_engine import init_hand
    from pokerbot.engine.np_engine import to_numpy_state
    from pokerbot.engine.jax_weva_fast import default_weva_bucket_table
    from pokerbot.encoding.fingerprint_native import fingerprint_fast

    env = to_numpy_state(init_hand(jax.random.PRNGKey(0), n_players=6))
    seat = int(env.action_seat)
    key = fingerprint_fast(env, seat, weva_table=default_weva_bucket_table())
    row = np.zeros(11, dtype=np.float64)
    row[2] = 1.0  # all-call: a valid normalized row
    return {bytes(key): row}


def test_gate_refuses_mismatched_stamp(monkeypatch):
    """A snapshot whose stamp disagrees with the active serve config must be REFUSED."""
    ship = _load_ship()
    _pin_serve_config(monkeypatch)

    snap = _tiny_real_snapshot(ship)
    # Active serve config is preflop_lossless=True; hand it a stamp that says False -> strict mismatch.
    active = ship._stamp_for_snapshot()
    bad_stamp = dict(active)
    bad_stamp["preflop_lossless"] = not active["preflop_lossless"]

    with pytest.raises(ship.ValidationError) as ei:
        ship.validate_serve(snap, bad_stamp)
    # the (a) strict serve-abstraction cross-check is what fired
    assert "MISMATCH" in str(ei.value) or "abstraction" in str(ei.value).lower()


def test_gate_refuses_empty_snapshot(monkeypatch):
    """An empty/too-small table must be REFUSED (the infoset floor)."""
    ship = _load_ship()
    _pin_serve_config(monkeypatch)
    active = ship._stamp_for_snapshot()
    with pytest.raises(ship.ValidationError) as ei:
        ship.validate_serve({}, active)
    assert "infoset" in str(ei.value).lower()
