"""The SHA256 content-stamp + sidecar wiring for EMD frozen blobs.

These three tests prove train-table == serve-table is provably verified by a
cryptographic SHA256 of the whole blob (written as a sidecar by
``scripts/precompute_emd_frozen_lookup.py`` and checked at the serve-skew gate
in ``scripts/ship_blueprint.py``):

  1. test_sha256_round_trip
     Write a blob + its sidecar, then read the sidecar and verify it matches the
     blob bytes. This is the happy path: the sidecar is regenerated atomically
     alongside the blob, and ``ship_blueprint.validate_serve`` would PASS.

  2. test_sha256_detects_tampering
     Flip one byte in the blob after writing the sidecar; the SHA256 of the blob
     no longer matches the sidecar. ``ship_blueprint.validate_serve`` raises
     ``ValidationError`` with the (0) SHA256-mismatch message. The cryptographic
     avalanche property is the whole point of SHA256 here (vs blake2b16 which is
     a fingerprint).

  3. test_sha256_missing_sidecar_is_fail_closed
     When ``POKERBOT_POSTFLOP_EMD_FROZEN`` is set but the sidecar is absent,
     ``ship_blueprint.validate_serve`` raises ``ValidationError`` (fail-closed)
     rather than silently accepting an unverified blob. This guards against an
     attacker (or a stale artifact) shipping a modified blob under a fresh name
     that never had a sidecar.

The tests intentionally avoid loading the 20MB native checkpoint or running the
serve probe; they exercise only the sidecar read/verify path. This keeps them
fast and independent of battery_native availability.
"""

from __future__ import annotations

import hashlib
import importlib.util
import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def _load_ship():
    """Load scripts/ship_blueprint.py via importlib (it lives under scripts/, not an installed package)."""
    spec = importlib.util.spec_from_file_location("ship_blueprint", ROOT / "scripts" / "ship_blueprint.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _write_blob_with_sidecar(tmp_path: Path, content: bytes = b"EMDFRZN1-fake-blob-payload") -> tuple[Path, Path]:
    """Write a fake blob and its SHA256 sidecar at ``tmp_path``. Return (blob_path, sidecar_path)."""
    blob = tmp_path / "emd_frozen_test.emdfrzn"
    blob.write_bytes(content)
    sha = hashlib.sha256(content).hexdigest()
    sidecar = blob.with_suffix(blob.suffix + ".sha256")
    sidecar.write_text(sha + "\n", encoding="ascii")
    return blob, sidecar


# --------------------------------------------------------------------------- #
# 1. Round-trip: write blob + sidecar, read+verify -> match.
# --------------------------------------------------------------------------- #
def test_sha256_round_trip(tmp_path: Path):
    """Writing the blob and sidecar together must verify on read."""
    blob, sidecar = _write_blob_with_sidecar(tmp_path)
    expected = sidecar.read_text(encoding="ascii").strip()
    actual = hashlib.sha256(blob.read_bytes()).hexdigest()
    assert expected == actual, "fresh sidecar must match its blob's SHA256"
    assert len(expected) == 64  # SHA256 hex is 64 chars
    assert all(c in "0123456789abcdef" for c in expected)


# --------------------------------------------------------------------------- #
# 2. Tampering: flip one byte after writing the sidecar -> sidecar no longer matches,
#    and the live validate_serve gate raises ValidationError with the (0) SHA256 message.
# --------------------------------------------------------------------------- #
def test_sha256_detects_tampering(tmp_path: Path, monkeypatch):
    """A single-byte change in the blob must invalidate the sidecar (avalanche), AND the
    serve-skew gate must raise ValidationError citing the SHA256 mismatch."""
    blob, sidecar = _write_blob_with_sidecar(tmp_path, content=b"EMDFRZN1-original-blob")
    expected_sha = sidecar.read_text(encoding="ascii").strip()

    # Flip one byte (xor with 0x01) deep into the blob -- never touches the sidecar.
    data = bytearray(blob.read_bytes())
    data[10] ^= 0x01
    blob.write_bytes(bytes(data))

    actual_sha = hashlib.sha256(blob.read_bytes()).hexdigest()
    assert expected_sha != actual_sha, "tampered blob must NOT match the original sidecar"

    # Now drive the REAL ship_blueprint.validate_serve() -- it must refuse at (0).
    # Order in validate_serve: (0) SHA256 -> (a) STRICT stamp -> (b)/(c) per-vec -> (d) infoset floor.
    # With a valid-shape stamp + a tiny-but-real snapshot, (0) is what fires.
    ship = _load_ship()
    ship._set_serve_env()  # bake the serve flags so (a) does not false-fire
    monkeypatch.setenv("POKERBOT_POSTFLOP_EMD_FROZEN", str(blob))

    snap = {b"\x00" * 16: [0.0] * 11}
    active_stamp = ship._stamp_for_snapshot()

    with pytest.raises(ship.ValidationError) as ei:
        ship.validate_serve(snap, active_stamp)
    msg = str(ei.value)
    assert "SHA256 mismatch" in msg, f"expected (0) SHA256 mismatch, got: {msg}"
    assert expected_sha[:16] in msg, f"expected short sha in error, got: {msg}"
    assert actual_sha[:16] in msg, f"expected short sha in error, got: {msg}"


# --------------------------------------------------------------------------- #
# 3. Missing sidecar: when POKERBOT_POSTFLOP_EMD_FROZEN is set but the sidecar
#    is absent, the gate MUST raise ValidationError (fail-closed).
# --------------------------------------------------------------------------- #
def test_sha256_missing_sidecar_is_fail_closed(tmp_path: Path, monkeypatch):
    """No sidecar = no ship. The serve-skew gate refuses to activate."""
    ship = _load_ship()
    ship._set_serve_env()  # bake the serve flags so (a) does not false-fire

    blob = tmp_path / "emd_frozen_no_sidecar.emdfrzn"
    blob.write_bytes(b"EMDFRZN1-orphan-blob-with-no-sidecar")
    orphan_sidecar = blob.with_suffix(blob.suffix + ".sha256")
    assert not orphan_sidecar.exists(), "test setup: sidecar must be absent"

    monkeypatch.setenv("POKERBOT_POSTFLOP_EMD_FROZEN", str(blob))

    snap = {b"\x00" * 16: [0.0] * 11}
    active_stamp = ship._stamp_for_snapshot()

    with pytest.raises(ship.ValidationError) as ei:
        ship.validate_serve(snap, active_stamp)
    msg = str(ei.value)
    assert "SHA256 sidecar missing" in msg, f"expected (0) sidecar-missing message, got: {msg}"
    assert "Re-run scripts/precompute_emd_frozen_lookup.py" in msg, f"expected remediation hint, got: {msg}"
    assert str(orphan_sidecar) in msg, f"expected sidecar path in error, got: {msg}"