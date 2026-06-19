"""Regression gate: ``fingerprint_fast`` is byte-identical to the captured golden vectors.

This is the GO/NO-GO bit-identity test for a future native (Rust) port of the blueprint
infoset fingerprint hot path — analogous to ``test_battery_native`` / the AIVAT golden.

Skips automatically when ``tools/fingerprint_native/golden/golden.json`` has not been
generated yet (it is a large stratified artifact produced by ``golden_harness.py generate``).
Each section verifies in its own subprocess because the abstraction flags are baked at import.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_HARNESS = Path(__file__).resolve().parents[1] / "tools" / "fingerprint_native" / "golden_harness.py"
_GOLDEN_DIR = Path(__file__).resolve().parents[1] / "tools" / "fingerprint_native" / "golden"
_GOLDEN = _GOLDEN_DIR / "golden.json.gz"
if not _GOLDEN.exists():  # accept an uncompressed golden too
    _GOLDEN = _GOLDEN_DIR / "golden.json"


def _load_harness():
    spec = importlib.util.spec_from_file_location("fingerprint_golden_harness", str(_HARNESS))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.skipif(not _GOLDEN.exists(), reason="fingerprint golden.json not generated yet")
def test_fingerprint_fast_matches_golden():
    harness = _load_harness()
    results = harness.verify_against_golden(_GOLDEN, verbose=True)
    assert results["_ok"] is True
    sections = {k: v for k, v in results.items() if not k.startswith("_")}
    assert sections, "golden contained no sections"
    for name, r in sections.items():
        assert r["ok"], f"section {name} failed"
        assert r["cases"] > 0, f"section {name} had no cases"
