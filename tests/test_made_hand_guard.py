"""FAIL-CLOSED ship guard for the made-hand-only ('fast') postflop abstraction.

``require_potential_aware_buckets`` must REFUSE to train/save a production NLHE blueprint on the
made-hand 'fast' bucketer (which ignores draw equity = documented +188 bb/100 structural leak) unless
the run is explicitly acknowledged via POKERBOT_ALLOW_MADE_HAND_BUCKETS. Distinct from the advisory
``warn_if_made_hand_only`` (pinned by tests/test_emd_bucketer.py::test_made_hand_only_guard_fires),
which only WARNS and uses a looser mode-only predicate — that test must stay green.

The guard uses the PRECISE predicate ``_made_hand_only_active`` (postflop_buckets ON *and* mode 'fast'),
so a lossless/non-bucketed run (buckets OFF) is never blocked even if the mode reads 'fast'.
"""

import warnings

import pytest

import pokerbot.engine.jax_weva_encode as enc


def test_require_raises_when_made_hand_only_and_unacked(monkeypatch):
    # postflop buckets ON + mode 'fast' + no ack -> hard fail.
    monkeypatch.setattr(enc, "_POSTFLOP_BUCKETS_ENABLED", True)
    monkeypatch.setattr(enc, "_POSTFLOP_MODE", "fast")
    monkeypatch.delenv("POKERBOT_ALLOW_MADE_HAND_BUCKETS", raising=False)
    with pytest.raises(RuntimeError) as ei:
        enc.require_potential_aware_buckets(context="unit test train")
    # actionable message: names the context, the leak, and the two escape hatches.
    msg = str(ei.value)
    assert "unit test train" in msg
    assert "MADE-HAND-ONLY" in msg
    assert "POKERBOT_POSTFLOP_MODE=emd" in msg
    assert "POKERBOT_ALLOW_MADE_HAND_BUCKETS=1" in msg


def test_require_warns_and_returns_true_when_acked(monkeypatch):
    # acknowledged deliberate smoke -> no raise, still loud (warns), returns True.
    monkeypatch.setattr(enc, "_POSTFLOP_BUCKETS_ENABLED", True)
    monkeypatch.setattr(enc, "_POSTFLOP_MODE", "fast")
    monkeypatch.setenv("POKERBOT_ALLOW_MADE_HAND_BUCKETS", "1")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        active = enc.require_potential_aware_buckets(context="unit test smoke")
    assert active is True
    assert any("MADE-HAND-ONLY" in str(x.message) for x in w)


def test_require_returns_false_for_emd_mode(monkeypatch):
    # potential-aware mode -> nothing to guard, no raise, returns False.
    monkeypatch.setattr(enc, "_POSTFLOP_BUCKETS_ENABLED", True)
    monkeypatch.setattr(enc, "_POSTFLOP_MODE", "emd")
    monkeypatch.delenv("POKERBOT_ALLOW_MADE_HAND_BUCKETS", raising=False)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        active = enc.require_potential_aware_buckets(context="unit test emd")
    assert active is False
    assert not w  # silent for emd


def test_require_returns_false_when_buckets_disabled(monkeypatch):
    # PRECISE predicate: buckets OFF means a lossless/non-bucketed run -> never blocked, even mode 'fast'.
    monkeypatch.setattr(enc, "_POSTFLOP_BUCKETS_ENABLED", False)
    monkeypatch.setattr(enc, "_POSTFLOP_MODE", "fast")
    monkeypatch.delenv("POKERBOT_ALLOW_MADE_HAND_BUCKETS", raising=False)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        active = enc.require_potential_aware_buckets(context="unit test lossless")
    assert active is False
    assert not w  # no warning when there is no bucketing to be 'made-hand-only'


def test_made_hand_only_active_precise_predicate(monkeypatch):
    # the precise predicate itself: requires BOTH buckets ON and mode 'fast'.
    monkeypatch.setattr(enc, "_POSTFLOP_BUCKETS_ENABLED", True)
    monkeypatch.setattr(enc, "_POSTFLOP_MODE", "fast")
    assert enc._made_hand_only_active() is True
    monkeypatch.setattr(enc, "_POSTFLOP_BUCKETS_ENABLED", False)
    assert enc._made_hand_only_active() is False
    monkeypatch.setattr(enc, "_POSTFLOP_BUCKETS_ENABLED", True)
    monkeypatch.setattr(enc, "_POSTFLOP_MODE", "emd")
    assert enc._made_hand_only_active() is False
