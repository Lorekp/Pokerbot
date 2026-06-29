"""EMD frozen lookup miss behavior: fail-closed by default, observable mirror fallback."""

from __future__ import annotations

import importlib
import sys

import numpy as np
import pytest


class _MissFrozen:
    def bucket(self, hole: list[int], board: list[int]) -> int:
        return -1


@pytest.fixture
def emd_encode_module(monkeypatch):
    """Reload jax_weva_encode with controlled env (module caches globals at import)."""
    monkeypatch.setenv("POKERBOT_POSTFLOP_BUCKETS", "1")
    monkeypatch.setenv("POKERBOT_POSTFLOP_MODE", "emd")
    monkeypatch.setenv("POKERBOT_POSTFLOP_NUM_BUCKETS", "200")
    monkeypatch.setenv("POKERBOT_POSTFLOP_EMD_FROZEN", "/fake/frozen.emdfrzn")
    import pokerbot.engine.jax_weva_encode as jwe

    importlib.reload(jwe)
    jwe.reset_emd_frozen_miss_count()
    monkeypatch.setattr(jwe, "_get_emd_frozen_lookup", lambda: _MissFrozen())
    return jwe


def test_frozen_miss_fail_closed_returns_minus_one(emd_encode_module, monkeypatch):
    jwe = emd_encode_module
    monkeypatch.setenv("POKERBOT_EMD_FROZEN_FAIL_ON_MISS", "1")
    monkeypatch.setenv("POKERBOT_EMD_FROZEN_MIRROR_FALLBACK", "0")
    monkeypatch.setenv("POKERBOT_EMD_FROZEN_ALLOW_SILENT_FALLBACK", "0")
    jwe.reset_emd_frozen_miss_count()
    hole = np.array([0, 4], dtype=np.int32)
    board = np.array([8, 12, 16, 20, -1], dtype=np.int32)
    b = jwe.compute_postflop_bucket(hole, board)
    assert b == -1
    assert jwe.emd_frozen_miss_count() == 1
    cfg = jwe.active_abstraction_config()
    assert cfg["postflop_emd_frozen_miss_count"] == 1
    assert cfg["postflop_emd_frozen_fail_on_miss"] is True


def test_frozen_miss_mirror_fallback_observable(emd_encode_module, monkeypatch):
    jwe = emd_encode_module
    monkeypatch.setenv("POKERBOT_EMD_FROZEN_FAIL_ON_MISS", "1")
    monkeypatch.setenv("POKERBOT_EMD_FROZEN_MIRROR_FALLBACK", "1")
    monkeypatch.setenv("POKERBOT_EMD_FROZEN_ALLOW_SILENT_FALLBACK", "0")
    jwe.reset_emd_frozen_miss_count()
    hole = np.array([0, 4], dtype=np.int32)
    board = np.array([8, 12, 16, 20, -1], dtype=np.int32)
    b = jwe.compute_postflop_bucket(hole, board)
    assert b >= 0
    assert jwe.emd_frozen_miss_count() == 1
    cfg = jwe.active_abstraction_config()
    assert cfg["postflop_emd_frozen_mirror_fallback"] is True
