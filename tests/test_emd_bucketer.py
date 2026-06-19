"""Potential-aware EMD postflop bucketer — the draw-aware refinement of the made-hand-only abstraction.

Verifies the property the made-hand bucketer throws away: a DRAW and a MADE hand of different potential
get different hand-strength distributions (=> different EMD buckets), the river collapses to a made-hand
spike (no potential left), the bucketer is deterministic + round-trips, and the made-hand-only guard
fires loudly.
"""

from __future__ import annotations

import numpy as np

from pokerbot.encoding.postflop_bucketer import (
    EMDBucketer, _emd_1d, strength_histogram,
)

# card_id = rank*4 + suit (rank 0=2 .. 12=A; suit 0=c,1=d,2=h,3=s)
def _cid(rank0, suit):
    return rank0 * 4 + suit

# Trip aces on A-7-2 rainbow: very strong, concentrated-high distribution.
_TRIPS_HOLE = (_cid(12, 1), _cid(12, 3))                 # Ad As
_TRIPS_BOARD = (_cid(12, 2), _cid(5, 0), _cid(0, 1))     # Ah 7c 2d
# K-high flush draw on Jh-5h-2c: weak current rank, high potential -> spread distribution.
_DRAW_HOLE = (_cid(10, 2), _cid(11, 2))                  # Qh Kh
_DRAW_BOARD = (_cid(9, 2), _cid(3, 2), _cid(0, 0))       # Jh 5h 2c


def test_river_histogram_is_a_spike():
    # 5-card board: no runouts -> a single deterministic strength bin.
    h = strength_histogram(_TRIPS_HOLE, _TRIPS_BOARD + (_cid(8, 0), _cid(4, 1)), n_runouts=50)
    assert np.count_nonzero(h) == 1
    assert abs(h.sum() - 1.0) < 1e-9


def test_made_hand_more_top_heavy_than_draw():
    trips = strength_histogram(_TRIPS_HOLE, _TRIPS_BOARD, n_bins=20, n_runouts=400, seed=1)
    draw = strength_histogram(_DRAW_HOLE, _DRAW_BOARD, n_bins=20, n_runouts=400, seed=1)
    # the made set is ahead at showdown far more often -> more mass in the top bins than the draw
    assert trips[-5:].sum() > draw[-5:].sum()
    # and their distributions are genuinely different (potential-aware signal the made-hand bucket loses)
    assert _emd_1d(trips, draw) > 0.1


def test_emd_self_distance_zero():
    h = strength_histogram(_DRAW_HOLE, _DRAW_BOARD, n_runouts=200)
    assert _emd_1d(h, h) == 0.0


def test_fit_bucket_deterministic_and_roundtrips(tmp_path):
    rng = np.random.default_rng(0)
    spots = []
    for _ in range(60):
        cards = rng.choice(52, size=5, replace=False)
        spots.append(strength_histogram(tuple(int(c) for c in cards[:2]),
                                        tuple(int(c) for c in cards[2:]), n_runouts=120))
    bk = EMDBucketer(num_buckets=8, n_runouts=120).fit(np.array(spots), iters=8, seed=0)
    b1 = bk.bucket(_TRIPS_HOLE, _TRIPS_BOARD)
    b2 = bk.bucket(_TRIPS_HOLE, _TRIPS_BOARD)
    assert b1 == b2 and 0 <= b1 < 8           # deterministic, in range
    p = tmp_path / "emd.json"
    bk.save(p)
    bk2 = EMDBucketer.load(p)
    assert bk2.bucket(_TRIPS_HOLE, _TRIPS_BOARD) == b1   # round-trip identical


def test_unfitted_bucketer_raises():
    bk = EMDBucketer(num_buckets=8)
    try:
        bk.bucket(_TRIPS_HOLE, _TRIPS_BOARD)
        assert False, "expected RuntimeError on unfitted bucketer"
    except RuntimeError:
        pass


def test_made_hand_only_guard_fires(monkeypatch):
    import warnings
    import pokerbot.engine.jax_weva_encode as enc
    monkeypatch.setattr(enc, "_POSTFLOP_MODE", "fast")
    monkeypatch.delenv("POKERBOT_ALLOW_MADE_HAND_BUCKETS", raising=False)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        active = enc.warn_if_made_hand_only(context="test")
    assert active is True
    assert any("MADE-HAND-ONLY" in str(x.message) for x in w)
    # acknowledged -> no warning
    monkeypatch.setenv("POKERBOT_ALLOW_MADE_HAND_BUCKETS", "1")
    with warnings.catch_warnings(record=True) as w2:
        warnings.simplefilter("always")
        enc.warn_if_made_hand_only(context="test")
    assert not w2
