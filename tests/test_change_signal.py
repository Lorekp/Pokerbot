"""R5: lightweight BOCPD-style run-length change signal.

The per-context MAP run-length (hands since the last detected regime change; reset on a TV spike)
drives a SMOOTH post-change confidence ramp: a gear-change drops confidence toward 0 (cautious / GTO)
then climbs back as stable evidence rebuilds — recovering faster and more smoothly than the prior
binary fast/slow switch. Persists across save/load; legacy profiles fall back gracefully.
"""

from __future__ import annotations

import numpy as np

from pokerbot.opponent.bayes_exploit import context_read
from pokerbot.opponent.population_prior import context_prior
from pokerbot.opponent.stats import OpponentStats

CK = "flop|wet|facing_bet"


def _feed(stats, bins, n, start_hand=1):
    prior = context_prior(CK)
    for h in range(start_hand, start_hand + n):
        stats.hands = h
        stats.decay_and_add(CK, bins, prior, h)
    return start_hand + n


def test_stable_stream_grows_runlength_and_confidence():
    s = OpponentStats()
    _feed(s, [0, 1, 0, 0], 60)          # a consistent station style
    assert s.ctx_runlen.get(CK, 0) > 20  # a long stable run accrued
    read = context_read(s, CK)
    assert read is not None and read.conf > 0.7   # high confidence when stable


def test_gear_change_resets_runlength_and_dips_then_recovers_confidence():
    s = OpponentStats()
    nxt = _feed(s, [0, 1, 0, 0], 60)              # phase 1: over-calls (passive)
    conf_before = context_read(s, CK).conf
    runlen_before = s.ctx_runlen[CK]

    # phase 2: abrupt switch to aggression (a different bin) — the fast model diverges from slow.
    _feed(s, [0, 0, 0, 1], 6, start_hand=nxt)
    assert s.ctx_runlen[CK] < runlen_before        # run-length reset by the detected change
    conf_after = context_read(s, CK).conf
    assert conf_after < conf_before                # confidence dips (cautious ⇒ near-GTO)

    # phase 3: the new style stabilizes ⇒ run-length climbs ⇒ confidence recovers.
    _feed(s, [0, 0, 0, 1], 40, start_hand=nxt + 6)
    assert s.ctx_runlen[CK] > 15
    assert context_read(s, CK).conf > conf_after


def test_runlength_persists_and_legacy_loads_without_it(tmp_path):
    from pokerbot.opponent.player_profiles import PlayerProfileStore

    path = tmp_path / "p.json"
    store = PlayerProfileStore(path)
    s = store.get("999")
    _feed(s, [0, 1, 0, 0], 30)
    s.hands = 30
    assert s.ctx_runlen.get(CK, 0) > 0
    store.save()

    store2 = PlayerProfileStore(path)
    assert store2.get("999").ctx_runlen.get(CK, 0) > 0   # survived the round-trip

    # a legacy (no ctx_runlen) profile loads with an empty map and behaves (falls back gracefully).
    legacy = OpponentStats()
    legacy.ctx_slow[CK] = [1.0, 8.0, 1.0, 1.0]
    legacy.ctx_fast[CK] = [1.0, 8.0, 1.0, 1.0]
    assert legacy.ctx_runlen == {}
    assert context_read(legacy, CK) is not None          # no crash, binary-switch fallback path
