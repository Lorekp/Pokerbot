"""Fail-closed gate for the native fingerprint-key-segment capabilities (native_solver).

# Why this test

``pokerbot/search/native_solver.py::_native_fingerprint_matches_serve_abstraction`` is the native
twin of ``pokerbot/eval/lbr.py::check_blueprint_serve_env``: it must DISABLE the native
subgame/rollout fast paths whenever the loaded ``battery_native`` binary does NOT carry the exact
fingerprint-key port the serve env demands. Each key segment is behind its own env flag + a
``native_caps()`` capability string:

  * POKERBOT_PREFLOP_HIST_KEY=1 -> "preflop_hist_key"   (blueprint-v3)
  * POKERBOT_FULLHIST_KEY=1     -> "full_hist_key"      (blueprint-v4 Stage 0)
  * POKERBOT_PRESUM_KEY=1       -> "presum_key"         (blueprint-v4 arm-4)

REGRESSION (verified 2026-07-08): the gate historically checked ONLY preflop_hist_key. A stale root
``battery_native.pyd`` can advertise caps WITHOUT "presum_key" yet the gate still returned True under
``POKERBOT_FULLHIST_KEY=1 POKERBOT_PRESUM_KEY=1`` — the native fast paths then ran with the stale
binary's fingerprint.rs (FULLHIST/plain keys instead of PRESUM keys) -> 100% postflop blueprint MISS
-> silent uniform-fallback postflop play on any presum checkpoint. This test pins the gate closed for
ALL THREE segments.

The gate reads only ``os.environ`` + the module-global ``_EXT`` (its ``native_caps()``), so it is
unit-testable with a stubbed extension — no compiled binary required.
"""
from __future__ import annotations

import pytest

from pokerbot.search import native_solver as ns

# The full capability set a current binary reports (tools/battery_native/src/lib.rs::native_caps).
_FULL_CAPS = [
    "emd_fingerprint",
    "emd_subgame",
    "emd_rollout",
    "emd_lbr",
    "preflop_hist_key",
    "full_hist_key",
    "presum_key",
]

# (env_var, required native_cap) pairs — the same table the gate iterates.
_KEY_ENV_CAP = [
    ("POKERBOT_PREFLOP_HIST_KEY", "preflop_hist_key"),
    ("POKERBOT_FULLHIST_KEY", "full_hist_key"),
    ("POKERBOT_PRESUM_KEY", "presum_key"),
]


class _FakeExt:
    """Stub battery_native extension exposing native_caps() -> a fixed list."""

    def __init__(self, caps):
        self._caps = list(caps)

    def native_caps(self):
        return list(self._caps)


class _RaisingExt:
    """Stub whose native_caps() raises — models a binary too old to have the symbol behave."""

    def native_caps(self):
        raise RuntimeError("stale binary: native_caps unavailable")


@pytest.fixture(autouse=True)
def _clean_gate_env(monkeypatch):
    """Clear every env var the gate reads so each test starts from a known baseline (the gate's
    fingerprint-key checks are the axis under test; keep POSTFLOP_MODE/RIVER_OCHS/MISKEY out so a
    'cap present' case resolves to the plain True return, not the EMD branch)."""
    for var in (
        "POKERBOT_PREFLOP_HIST_KEY",
        "POKERBOT_FULLHIST_KEY",
        "POKERBOT_PRESUM_KEY",
        "POKERBOT_POSTFLOP_MODE",
        "POKERBOT_RIVER_OCHS",
        "POKERBOT_NATIVE_ALLOW_EMD_MISKEY",
    ):
        monkeypatch.delenv(var, raising=False)
    yield


# ---------------------------------------------------------------------------
# Default-off: no key env set -> gate unchanged (True) regardless of caps.
# ---------------------------------------------------------------------------
def test_gate_open_default_off_even_with_empty_caps(monkeypatch):
    """No fingerprint-key env set: the gate must NOT be gated by any of the three caps, even when
    the binary reports NONE of them (byte-identical default-off behavior)."""
    monkeypatch.setattr(ns, "_EXT", _FakeExt([]))
    assert ns._native_fingerprint_matches_serve_abstraction() is True


def test_gate_open_default_off_with_no_extension(monkeypatch):
    """No fingerprint-key env set and no extension loaded: the gate's key checks do not fire (the
    non-EMD default path returns True)."""
    monkeypatch.setattr(ns, "_EXT", None)
    assert ns._native_fingerprint_matches_serve_abstraction() is True


# ---------------------------------------------------------------------------
# Fail-closed: env set but the matching cap is ABSENT.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("env_var,cap", _KEY_ENV_CAP)
def test_gate_fails_closed_when_cap_absent(monkeypatch, env_var, cap):
    """env=1 but the binary does not advertise the matching cap -> gate returns False (disable the
    native fast paths). Covers preflop_hist_key (regression floor) AND full_hist_key / presum_key
    (the twin checks this test was added for)."""
    caps = [c for c in _FULL_CAPS if c != cap]  # everything EXCEPT the one under test
    monkeypatch.setattr(ns, "_EXT", _FakeExt(caps))
    monkeypatch.setenv(env_var, "1")
    assert ns._native_fingerprint_matches_serve_abstraction() is False


@pytest.mark.parametrize("env_var,cap", _KEY_ENV_CAP)
def test_gate_fails_closed_when_extension_missing(monkeypatch, env_var, cap):
    """env=1 but no extension loaded (``_EXT is None``) -> fail closed."""
    monkeypatch.setattr(ns, "_EXT", None)
    monkeypatch.setenv(env_var, "1")
    assert ns._native_fingerprint_matches_serve_abstraction() is False


@pytest.mark.parametrize("env_var,cap", _KEY_ENV_CAP)
def test_gate_fails_closed_when_caps_raises(monkeypatch, env_var, cap):
    """env=1 but ``native_caps()`` raises (a too-old binary) -> fail closed."""
    monkeypatch.setattr(ns, "_EXT", _RaisingExt())
    monkeypatch.setenv(env_var, "1")
    assert ns._native_fingerprint_matches_serve_abstraction() is False


# ---------------------------------------------------------------------------
# Gate opens when env is set AND the matching cap is present.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("env_var,cap", _KEY_ENV_CAP)
def test_gate_opens_when_cap_present(monkeypatch, env_var, cap):
    """env=1 and the binary advertises the matching cap -> the key check passes; with no EMD regime
    the gate returns True."""
    monkeypatch.setattr(ns, "_EXT", _FakeExt(_FULL_CAPS))
    monkeypatch.setenv(env_var, "1")
    assert ns._native_fingerprint_matches_serve_abstraction() is True


# ---------------------------------------------------------------------------
# The exact 2026-07-08 failure: FULLHIST + PRESUM set, binary lacks presum_key.
# ---------------------------------------------------------------------------
def test_gate_fails_closed_presum_missing_under_fullhist_and_presum(monkeypatch):
    """The reported cascade-miss: POKERBOT_FULLHIST_KEY=1 POKERBOT_PRESUM_KEY=1 against a binary that
    advertises full_hist_key but NOT presum_key. Before the fix the gate returned True (native ran
    with FULLHIST keys -> 100% postflop miss). It must now return False."""
    caps = [c for c in _FULL_CAPS if c != "presum_key"]  # has full_hist_key, lacks presum_key
    assert "full_hist_key" in caps and "presum_key" not in caps
    monkeypatch.setattr(ns, "_EXT", _FakeExt(caps))
    monkeypatch.setenv("POKERBOT_FULLHIST_KEY", "1")
    monkeypatch.setenv("POKERBOT_PRESUM_KEY", "1")
    assert ns._native_fingerprint_matches_serve_abstraction() is False


def test_gate_opens_when_fullhist_and_presum_both_supported(monkeypatch):
    """Both flags set AND both caps present -> the key checks pass (non-EMD -> True)."""
    monkeypatch.setattr(ns, "_EXT", _FakeExt(_FULL_CAPS))
    monkeypatch.setenv("POKERBOT_FULLHIST_KEY", "1")
    monkeypatch.setenv("POKERBOT_PRESUM_KEY", "1")
    assert ns._native_fingerprint_matches_serve_abstraction() is True


# ---------------------------------------------------------------------------
# Helper unit semantics (the fail-closed primitive the gate iterates).
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("env_var,cap", _KEY_ENV_CAP)
def test_cap_missing_helper_semantics(monkeypatch, env_var, cap):
    # env unset -> nothing required, never "missing".
    monkeypatch.setattr(ns, "_EXT", _FakeExt([]))
    assert ns._fingerprint_key_cap_missing(env_var, cap) is False
    # env set + cap absent -> missing.
    monkeypatch.setenv(env_var, "1")
    assert ns._fingerprint_key_cap_missing(env_var, cap) is True
    # env set + no extension -> missing (fail closed).
    monkeypatch.setattr(ns, "_EXT", None)
    assert ns._fingerprint_key_cap_missing(env_var, cap) is True
    # env set + native_caps raises -> missing (fail closed).
    monkeypatch.setattr(ns, "_EXT", _RaisingExt())
    assert ns._fingerprint_key_cap_missing(env_var, cap) is True
    # env set + cap present -> not missing.
    monkeypatch.setattr(ns, "_EXT", _FakeExt([cap]))
    assert ns._fingerprint_key_cap_missing(env_var, cap) is False


def test_gate_table_matches_native_caps_string_contract():
    """Guard against drift: the env->cap table the gate iterates must name the exact cap strings the
    Rust native_caps() emits (tools/battery_native/src/lib.rs)."""
    gate_caps = {cap for _, cap in ns._FINGERPRINT_KEY_CAPS}
    assert gate_caps == {"preflop_hist_key", "full_hist_key", "presum_key"}
