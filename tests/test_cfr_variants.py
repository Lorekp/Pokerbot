"""P0 #10: DCFR + VR-MCCFR flags — math correctness, in-place discount, and OFF = bit-identical."""

from __future__ import annotations

import os

os.environ.setdefault("POKERBOT_BETTING_ABSTRACTION", "1")

import numpy as np

from pokerbot.training.shared_hogwild_table import SharedHogwildTable, dcfr_weights
from pokerbot.training.sd_cfr_nlhe_hogwild import dcfr_discount_snapshots, run_nlhe_hogwild


# ----------------------------------------------------------------- DCFR math
def test_dcfr_weights_t1():
    # t=1, (alpha=1.5,beta=0,gamma=2): pos=1/(1+1)=0.5, neg=1/(1+1)=0.5, strat=(1/2)^2=0.25.
    pos, neg, strat = dcfr_weights(1, alpha=1.5, beta=0.0, gamma=2.0)
    assert abs(pos - 0.5) < 1e-9
    assert abs(neg - 0.5) < 1e-9
    assert abs(strat - 0.25) < 1e-9


def test_dcfr_weights_monotone():
    # As t grows, positive-regret factor -> 1 and strat factor -> 1 (old mass discounted less).
    p2, _, s2 = dcfr_weights(2)
    p100, _, s100 = dcfr_weights(100)
    assert p2 < p100 < 1.0
    assert s2 < s100 < 1.0


def test_dcfr_discount_snapshots_inplace():
    reg = {b"k0": np.array([2.0, -4.0, 0.0] + [0.0] * 8, dtype=np.float64)}
    strat = {b"k0": np.array([1.0] * 11, dtype=np.float64)}
    pos, neg, sfac = dcfr_weights(1)
    dcfr_discount_snapshots(reg, strat, 1)
    assert abs(reg[b"k0"][0] - 2.0 * pos) < 1e-9   # positive scaled by pos
    assert abs(reg[b"k0"][1] - (-4.0) * neg) < 1e-9  # negative scaled by neg
    assert abs(strat[b"k0"][0] - 1.0 * sfac) < 1e-9


def test_apply_dcfr_discount_table():
    t = SharedHogwildTable.create(64, 11)
    try:
        slot = t.slot_for(b"x" * 16)
        t.regret[slot] = np.array([3.0, -6.0] + [0.0] * 9, dtype=np.float64)
        t.strat[slot] = np.ones(11, dtype=np.float64)
        pos, neg, sfac = dcfr_weights(5)
        t.apply_dcfr_discount(5)
        assert abs(t.regret[slot][0] - 3.0 * pos) < 1e-9
        assert abs(t.regret[slot][1] - (-6.0) * neg) < 1e-9
        assert abs(t.strat[slot][0] - 1.0 * sfac) < 1e-9
    finally:
        t.close()
        t.unlink()


# ----------------------------------------------------------------- VR-MCCFR baseline table
def test_baseline_array_conditional():
    t0 = SharedHogwildTable.create(16, 11, with_baseline=False)
    try:
        assert t0.base is None and t0.base_row(0) is None
        assert "base" not in t0._shms  # default layout byte-identical (no extra segment)
    finally:
        t0.close(); t0.unlink()
    t1 = SharedHogwildTable.create(16, 11, with_baseline=True)
    try:
        assert t1.base is not None and t1.base_row(0) is not None
        assert t1.base.shape == (16, 11)
    finally:
        t1.close(); t1.unlink()


# ----------------------------------------------------------------- end-to-end flags
def test_use_baseline_false_is_deterministic():
    # OFF path must be unperturbed: same seed/workers => identical snapshot (bit-identity guard).
    a = run_nlhe_hogwild(traversals=6, num_workers=1, capacity=1 << 15,
                         occupancy=(True,) * 6, client_bet_sizing=True, seed=0, use_baseline=False)
    b = run_nlhe_hogwild(traversals=6, num_workers=1, capacity=1 << 15,
                         occupancy=(True,) * 6, client_bet_sizing=True, seed=0, use_baseline=False)
    assert a["strat"].keys() == b["strat"].keys()
    for k in a["strat"]:
        assert np.array_equal(a["strat"][k], b["strat"][k])


def test_use_baseline_true_runs_and_is_valid():
    r = run_nlhe_hogwild(traversals=6, num_workers=1, capacity=1 << 15,
                         occupancy=(True,) * 6, client_bet_sizing=True, seed=0,
                         cfr_plus=False, use_baseline=True)
    assert r["n_infosets"] > 0 and len(r["strat"]) > 0
    row = next(iter(r["strat"].values()))
    assert np.all(np.isfinite(np.asarray(row)))
