"""Priority 4: AOF_STRICT_RANGES raises on missing call charts."""

from __future__ import annotations

import pytest

from aof_ranges import lookup_call


def test_strict_mode_missing_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AOF_STRICT_RANGES", "1")
    with pytest.raises(KeyError, match="Missing call chart"):
        lookup_call("UTG", "XX", 8.0, table_size=4)


def test_default_mode_allows_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AOF_STRICT_RANGES", raising=False)
    pct, rng, prov = lookup_call("UTG", "XX", 8.0, table_size=4)
    assert pct is not None
    assert rng
    assert prov == "unknown"
