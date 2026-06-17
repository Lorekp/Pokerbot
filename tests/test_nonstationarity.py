"""Non-stationarity: dual-timescale discounted counts + the change signal. The fast timescale tracks
a gear-change while the slow one lags; a detected shift makes the read lean to the recent style and
drops confidence to cautious until the opponent settles."""

from __future__ import annotations

import numpy as np

from pokerbot.opponent.bayes_exploit import CHANGE_TV, context_read
from pokerbot.opponent.context import AGG_LARGE_BIN, PASSIVE_BIN
from pokerbot.opponent.population_prior import context_prior
from pokerbot.opponent.stats import OpponentStats

_CK = "flop|wet|facing_bet"
_PASSIVE = [0, 1, 0, 0]   # a calling-station decision (check/call)
_AGGR = [0, 0, 0, 1]      # a maniac decision (big bet/raise)


def _feed(s: OpponentStats, bins, start_hand: int, n: int) -> None:
    prior = context_prior(_CK)
    for h in range(start_hand, start_hand + n):
        s.hands = h
        s.decay_and_add(_CK, bins, prior, h)


def test_stabilized_update_never_zeroes_unseen_action() -> None:
    s = OpponentStats()
    _feed(s, _PASSIVE, 1, 200)  # only ever passive
    # the fold/agg bins were never observed but must stay > 0 (population floor), not collapse.
    assert all(x > 0.0 for x in s.ctx_fast[_CK])
    assert all(x > 0.0 for x in s.ctx_slow[_CK])


def test_fast_tracks_shift_while_slow_lags() -> None:
    s = OpponentStats()
    _feed(s, _PASSIVE, 1, 60)        # 60 hands of station
    before = context_read(s, _CK)
    assert before.post[PASSIVE_BIN] > 0.6   # reads as passive

    _feed(s, _AGGR, 61, 8)           # opponent switches to maniac for 8 hands
    f = np.asarray(s.ctx_fast[_CK]); sl = np.asarray(s.ctx_slow[_CK])
    p_fast, p_slow = f / f.sum(), sl / sl.sum()
    assert p_fast[AGG_LARGE_BIN] > p_slow[AGG_LARGE_BIN]   # fast moved further toward aggression
    assert p_fast[AGG_LARGE_BIN] > 0.30                    # fast (≈10-hand window): ~8x up from baseline
    assert p_slow[PASSIVE_BIN] > 0.4                       # slow still remembers the station


def test_change_signal_makes_read_cautious_then_recovers() -> None:
    s = OpponentStats()
    _feed(s, _PASSIVE, 1, 80)
    stable = context_read(s, _CK)
    assert 0.5 * (np.abs(np.asarray(s.ctx_fast[_CK]) / sum(s.ctx_fast[_CK])
                  - np.asarray(s.ctx_slow[_CK]) / sum(s.ctx_slow[_CK]))).sum() <= CHANGE_TV  # stable: low TV

    _feed(s, _AGGR, 81, 6)           # gear-change
    just_after = context_read(s, _CK)
    assert just_after.conf < stable.conf                   # confidence drops (cautious) on the shift
    assert just_after.post[AGG_LARGE_BIN] > stable.post[AGG_LARGE_BIN]  # but leans to the new style

    _feed(s, _AGGR, 87, 120)         # new style persists; fast & slow reconverge
    settled = context_read(s, _CK)
    assert settled.conf > just_after.conf                  # confidence recovers once settled
    assert settled.post[AGG_LARGE_BIN] > 0.5               # now reads as aggressive


def test_stable_stream_full_confidence_no_shift() -> None:
    s = OpponentStats()
    _feed(s, _PASSIVE, 1, 120)
    r = context_read(s, _CK)
    f = np.asarray(s.ctx_fast[_CK]); sl = np.asarray(s.ctx_slow[_CK])
    tv = 0.5 * float(np.abs(f / f.sum() - sl / sl.sum()).sum())
    assert tv <= CHANGE_TV                 # no false change signal on a stationary opponent
    assert r.conf > 0.8                    # lifetime confidence is high
    # back-compat: a profile with only static ctx_counts (no fast/slow) still reads via the static path
    legacy = OpponentStats(hands=80, ctx_counts={_CK: [2, 60, 1, 1]})
    assert context_read(legacy, _CK) is not None
