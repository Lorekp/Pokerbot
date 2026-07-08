"""M7 cash serve-fidelity gate (``pokerbot.eval.cash_serve_gate``).

The validating anchor: a deliberate single-field fingerprint MISMATCH must be DETECTED and FAIL CLOSED
*before* any hand is served. This is what makes the -559 bb/100 "config artifact" (a lossless/postflop
flag skew that silently dropped blueprint coverage to ~0%) unrepeatable.

Mirrors ``tests/test_preflop_lossless.py::test_serve_guard_rejects_flag_skew``: it monkeypatches the
encoder's baked flags so ``active_abstraction_config()`` reports a serve env, then asserts the gate
raises when that env differs from the snapshot stamp in EXACTLY ONE field — preflop_lossless,
postflop_num_buckets, and an injected depth_bucketed=True — and does NOT raise when they match.

These tests touch only ``check_blueprint_serve_env`` and the env/stamp portion of ``assert_cash_serve``
(``probe_hands=0``), so they are net-free, deal-free, and sub-second on CPU.
"""

from __future__ import annotations

import pytest

from pokerbot.encoding import weva_encoder
from pokerbot.engine import jax_weva_encode as enc
from pokerbot.eval.lbr import check_blueprint_serve_env


# A correctly-stamped lossless-era snapshot stamp (the deployed cash blueprint regime).
_GOOD_STAMP = {
    "betting_abstraction": True,
    "postflop_buckets": True,
    "postflop_mode": "fast",
    "postflop_num_buckets": 200,
    "preflop_lossless": True,
}


def _set_serve_env(monkeypatch, *, preflop_lossless=True, num_buckets=200, depth_buckets=False,
                   fullhist_key=False, presum_key=False):
    """Force ``active_abstraction_config()`` to report a specific baked serve env (no re-import needed)."""
    monkeypatch.setattr(enc, "_BETTING_ABSTRACTION", True)
    monkeypatch.setattr(enc, "_POSTFLOP_BUCKETS_ENABLED", True)
    monkeypatch.setattr(enc, "_POSTFLOP_MODE", "fast")
    monkeypatch.setattr(enc, "_POSTFLOP_NUM_BUCKETS", num_buckets)
    monkeypatch.setattr(enc, "_DEPTH_BUCKETS", depth_buckets)
    monkeypatch.setattr(enc, "_FULLHIST_KEY", fullhist_key)
    monkeypatch.setattr(enc, "_PRESUM_KEY", presum_key)
    monkeypatch.setattr(weva_encoder, "_PREFLOP_LOSSLESS", preflop_lossless)


# --------------------------------------------------------------------------- #
# check_blueprint_serve_env: one-field mismatches must each RAISE (the anchor)
# --------------------------------------------------------------------------- #
def test_serve_guard_rejects_preflop_lossless_skew(monkeypatch):
    # Serve baked preflop_lossless OFF; snapshot trained with it ON -> every preflop node misses.
    _set_serve_env(monkeypatch, preflop_lossless=False)
    with pytest.raises(RuntimeError, match="preflop_lossless|stamp"):
        check_blueprint_serve_env(strict=True, expected=dict(_GOOD_STAMP))


def test_serve_guard_rejects_postflop_num_buckets_skew(monkeypatch):
    # Serve baked num_buckets=180; snapshot stamped 200 -> postflop keys hash a different bucketer.
    _set_serve_env(monkeypatch, num_buckets=180)
    with pytest.raises(RuntimeError, match="num_buckets|stamp"):
        check_blueprint_serve_env(strict=True, expected=dict(_GOOD_STAMP))


def test_serve_guard_rejects_injected_depth_bucketed(monkeypatch):
    # Snapshot stamp injects depth_bucketed=True; serve env has it OFF -> the depth-byte cross-check fires
    # (a PRE-depth stamp has no such key, so the field-by-field loop alone can't catch it; lbr.py:168-176).
    _set_serve_env(monkeypatch, depth_buckets=False)
    stamp = {**_GOOD_STAMP, "depth_bucketed": True}
    with pytest.raises(RuntimeError, match="depth_bucketed"):
        check_blueprint_serve_env(strict=True, expected=stamp)


def test_serve_guard_accepts_matching_stamp(monkeypatch):
    _set_serve_env(monkeypatch, preflop_lossless=True, num_buckets=200, depth_buckets=False)
    out = check_blueprint_serve_env(strict=True, expected=dict(_GOOD_STAMP))
    assert out["preflop_lossless"] is True
    assert out["postflop_num_buckets"] == 200


# --------------------------------------------------------------------------- #
# fullhist_key / presum_key two-sided guards (blueprint-v4). Each direction of a skew must RAISE:
# a stamp-has-flag / serve-lacks-flag AND a serve-has-flag / stamp-lacks-flag. The field-by-field
# `mism` loop only iterates the stamp's keys, so the serve-has-but-stamp-lacks direction is caught
# ONLY by the explicit two-sided checks (the back-filled fullhist guard + the new presum guard).
# --------------------------------------------------------------------------- #
def test_serve_guard_rejects_fullhist_stamp_serve_off(monkeypatch):
    # Snapshot trained WITH the full-history key; serve baked it OFF -> 100% miss on every street.
    _set_serve_env(monkeypatch, fullhist_key=False)
    stamp = {**_GOOD_STAMP, "fullhist_key": True}
    with pytest.raises(RuntimeError, match="fullhist_key"):
        check_blueprint_serve_env(strict=True, expected=stamp)


def test_serve_guard_rejects_fullhist_serve_on_stamp_off(monkeypatch):
    # Serve baked the full-history key ON; a NON-fullhist snapshot (no stamp) -> 100% miss. This is
    # the direction the field-by-field loop misses (stamp has no "fullhist_key" key) — the back-fill.
    _set_serve_env(monkeypatch, fullhist_key=True)
    with pytest.raises(RuntimeError, match="fullhist_key"):
        check_blueprint_serve_env(strict=True, expected=dict(_GOOD_STAMP))


def test_serve_guard_rejects_presum_stamp_serve_off(monkeypatch):
    # Snapshot keyed POSTFLOP on the preflop pot-type summary; serve baked presum OFF -> the exact
    # preflop ids get hashed instead => 100% postflop miss. (A presum snapshot is also fullhist.)
    _set_serve_env(monkeypatch, fullhist_key=True, presum_key=False)
    stamp = {**_GOOD_STAMP, "fullhist_key": True, "presum_key": True}
    with pytest.raises(RuntimeError, match="presum_key"):
        check_blueprint_serve_env(strict=True, expected=stamp)


def test_serve_guard_rejects_presum_serve_on_stamp_off(monkeypatch):
    # Serve baked presum ON against a plain-fullhist snapshot (no presum stamp) -> the summary bytes
    # the table never saw get hashed => 100% postflop miss. The serve-has / stamp-lacks direction.
    _set_serve_env(monkeypatch, fullhist_key=True, presum_key=True)
    stamp = {**_GOOD_STAMP, "fullhist_key": True}
    with pytest.raises(RuntimeError, match="presum_key"):
        check_blueprint_serve_env(strict=True, expected=stamp)


def test_serve_guard_accepts_matching_presum_stamp(monkeypatch):
    # Correctly matched presum serve + presum snapshot: no problem raised.
    _set_serve_env(monkeypatch, fullhist_key=True, presum_key=True)
    stamp = {**_GOOD_STAMP, "fullhist_key": True, "presum_key": True}
    out = check_blueprint_serve_env(strict=True, expected=stamp)
    assert out["presum_key"] is True and out["fullhist_key"] is True


# --------------------------------------------------------------------------- #
# assert_cash_serve: refuses when the baked env != snapshot stamp (fail-closed)
# --------------------------------------------------------------------------- #
def test_assert_cash_serve_refuses_preflop_lossless_skew(monkeypatch):
    from pokerbot.eval.cash_serve_gate import assert_cash_serve

    # Snapshot trained with lossless ON; encoder baked it OFF. The LBR modules setdefault the other four
    # flags but NOT this one, so this is the canonical cash serve skew — must fail closed before any deal.
    _set_serve_env(monkeypatch, preflop_lossless=False)
    snap = {"strat_sum": {}, "abstraction": dict(_GOOD_STAMP)}
    with pytest.raises(RuntimeError, match="PREFLOP_LOSSLESS|preflop_lossless"):
        assert_cash_serve(snap, probe_hands=0)


def test_assert_cash_serve_refuses_num_buckets_skew(monkeypatch):
    from pokerbot.eval.cash_serve_gate import assert_cash_serve

    _set_serve_env(monkeypatch, num_buckets=180)
    snap = {"strat_sum": {}, "abstraction": dict(_GOOD_STAMP)}
    with pytest.raises(RuntimeError, match="num_buckets|stamp"):
        assert_cash_serve(snap, probe_hands=0)


def test_assert_cash_serve_refuses_injected_depth_bucketed(monkeypatch):
    from pokerbot.eval.cash_serve_gate import assert_cash_serve

    _set_serve_env(monkeypatch, depth_buckets=False)
    snap = {"strat_sum": {}, "abstraction": {**_GOOD_STAMP, "depth_bucketed": True}}
    with pytest.raises(RuntimeError, match="depth_bucketed"):
        assert_cash_serve(snap, probe_hands=0)


def test_assert_cash_serve_refuses_abstraction_off(monkeypatch):
    from pokerbot.eval.cash_serve_gate import assert_cash_serve

    # Betting abstraction baked OFF entirely -> the bounded blueprint table simply doesn't apply.
    _set_serve_env(monkeypatch)
    monkeypatch.setattr(enc, "_BETTING_ABSTRACTION", False)
    snap = {"strat_sum": {}, "abstraction": dict(_GOOD_STAMP)}
    with pytest.raises(RuntimeError, match="BETTING_ABSTRACTION|POSTFLOP_BUCKETS"):
        assert_cash_serve(snap, probe_hands=0)


def test_assert_cash_serve_accepts_matching_env_no_probe(monkeypatch):
    from pokerbot.eval.cash_serve_gate import assert_cash_serve

    # Env matches the stamp; skip the deal probe (probe_hands=0). Must return the active cfg for logging.
    _set_serve_env(monkeypatch, preflop_lossless=True, num_buckets=200, depth_buckets=False)
    snap = {"strat_sum": {}, "abstraction": dict(_GOOD_STAMP)}
    out = assert_cash_serve(snap, probe_hands=0)
    assert out["abstraction"]["preflop_lossless"] is True
    assert out["abstraction"]["postflop_num_buckets"] == 200
    assert out["stamp"] == _GOOD_STAMP
    assert out["coverage"] == {}  # probe skipped
