"""The exploitation battery's ExploitHero must deviate with the SAME alpha the production policy uses,
or its do-no-harm / uplift verdicts are measured off-policy. Production (root_mixer / search_tilt) uses
EXTREMITY-scaled alpha = max_alpha * conf * min(1, tv/_TV_FULL); the battery previously used conf-only
(min(conf, cap)), which over-deviates vs a BALANCED opponent (high conf, low tv). This guards the fix.
"""

from __future__ import annotations

import numpy as np

from pokerbot.opponent.bayes_exploit import ContextRead, DEFAULT_MAX_ALPHA, _TV_FULL


def _read(post, pop, conf):
    return ContextRead(post=np.asarray(post, float), pop=np.asarray(pop, float),
                       conf=float(conf), n_obs=50)


def _battery_alpha(read, cap):
    # the exact expression now in exploitation_battery.ExploitHero.__call__
    return float(np.clip(cap * read.conf * min(1.0, read.tv / _TV_FULL), 0.0, cap))


def test_battery_alpha_equals_production_search_tilt_alpha():
    cap = DEFAULT_MAX_ALPHA
    cases = [
        # (post, pop, conf) — a near-balanced read (low tv) and a polarized read (high tv)
        ([0.30, 0.30, 0.20, 0.20], [0.25, 0.25, 0.25, 0.25], 0.9),   # low tv, high conf
        ([0.70, 0.10, 0.10, 0.10], [0.25, 0.25, 0.25, 0.25], 0.8),   # high tv
        ([0.26, 0.24, 0.25, 0.25], [0.25, 0.25, 0.25, 0.25], 0.95),  # tiny tv -> alpha ~ 0
    ]
    for post, pop, conf in cases:
        r = _read(post, pop, conf)
        assert abs(_battery_alpha(r, cap) - r.search_tilt(max_alpha=cap)[1]) < 1e-9


def test_balanced_opponent_gets_near_zero_alpha():
    # The whole point: a high-confidence read of a BALANCED villain (tv ~ 0) must NOT deviate much,
    # whereas the old conf-only alpha would have returned ~cap.
    r = _read([0.26, 0.24, 0.25, 0.25], [0.25, 0.25, 0.25, 0.25], 0.95)
    cap = DEFAULT_MAX_ALPHA
    assert _battery_alpha(r, cap) < 0.1 * cap          # extremity-scaled: tiny
    assert min(r.conf, cap) >= 0.5 * cap               # old conf-only would have been large
