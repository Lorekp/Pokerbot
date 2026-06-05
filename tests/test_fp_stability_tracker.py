"""StabilityTracker must not certify convergence at high Nash gap."""

from __future__ import annotations

import numpy as np

from aof_cfr.fp_br import StabilityTracker


def test_high_gap_plateau_not_converged() -> None:
    tr = StabilityTracker(gap_floor_mbb_g=762.0, window=5)
    btn = np.full(169, 0.75)
    for g in [1700.0, 1701.0, 1700.5, 1700.8, 1700.2, 1700.9]:
        assert not tr.observe(g, btn)


def test_low_gap_plateau_can_converge() -> None:
    tr = StabilityTracker(gap_floor_mbb_g=762.0, window=5, policy_tol=1.0)
    btn = np.full(169, 0.5)
    out = False
    for g in [400.0, 400.1, 400.0, 400.2, 400.1, 400.0]:
        out = tr.observe(g, btn)
    assert out
