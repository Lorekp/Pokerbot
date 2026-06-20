"""Validate-the-validator for the native LBR harness (`tools/battery_native/src/lbr.rs`).

The native LBR is an independent Monte-Carlo estimator (fast xoshiro RNG, not numpy PCG64), so it can't
be bit-compared to the Python `lbr6`. Instead we gate it on:
  1. CALIBRATION anchors -- always-fold exploitability must hit the documented references
     (~ +750 mbb/g HU, ~ +128 bb/100 6-max), the same self-test the Python docstrings cite.
  2. CI-OVERLAP equivalence -- on a shared policy/config, |ev_native - ev_python| must be within the
     combined 95% CI of the two estimators (a two-sided z-test that they estimate the same quantity).

Calibration uses the always-fold policy (near-deterministic -> tiny CI -> a sharp test). Run with:
  POKERBOT_PREFLOP_LOSSLESS=1 python -m pytest tests/test_lbr_native_equiv.py -q
"""

from __future__ import annotations

import numpy as np
import pytest

from pokerbot.eval.lbr import LBRConfig
from pokerbot.eval.lbr6 import LBR6Config


def _native_available():
    try:
        from pokerbot.eval import lbr_native as L

        return L.available()
    except Exception:  # noqa: BLE001
        return False


pytestmark = pytest.mark.skipif(not _native_available(), reason="native battery_native not built")


def test_calibration_fold_6max():
    """Native always-fold 6-max exploitability hits the documented ~ +128 bb/100 anchor."""
    from pokerbot.eval import lbr_native as L

    r = L.lbr6_exploitability_native(None, LBR6Config(hands=8000, seed=0, n_seats=6), policy="fold")
    ci = 1.96 * r.stderr_bb_per_hand * 100.0
    assert 110.0 <= r.bb_per_100 <= 145.0, f"fold 6-max {r.bb_per_100:+.1f} bb/100 off anchor +128"
    assert ci < 10.0, f"fold calibration CI unexpectedly wide ({ci:.1f}) -- estimator broken?"


def test_calibration_fold_hu():
    """Native always-fold HU exploitability hits the documented ~ +750 mbb/g anchor."""
    from pokerbot.eval import lbr_native as L

    r = L.lbr_hu_tabular_native(None, LBRConfig(hands=8000, seed=0), policy="fold")
    assert 680.0 <= r.mbb_per_g <= 820.0, f"fold HU {r.mbb_per_g:+.0f} mbb/g off anchor +750"


def test_native_matches_python_fold_6max_ci_overlap():
    """Native and Python lbr6 must agree within their combined 95% CI on the shared fold policy."""
    from pokerbot.eval import lbr_native as L
    from pokerbot.eval.lbr import always_fold_policy
    from pokerbot.eval.lbr6 import lbr6_exploitability

    cfg = LBR6Config(hands=600, seed=0, n_seats=6, track_ranges=False)
    rp = lbr6_exploitability(always_fold_policy, cfg)
    rn = L.lbr6_exploitability_native(None, cfg, policy="fold")
    sp, sn = rp.stderr_bb_per_hand * 100.0, rn.stderr_bb_per_hand * 100.0
    combined = 1.96 * float(np.hypot(sp, sn)) + 1.0  # +1 guard for the near-zero-variance fold case
    diff = abs(rp.bb_per_100 - rn.bb_per_100)
    assert diff <= combined, (
        f"native {rn.bb_per_100:+.1f} vs python {rp.bb_per_100:+.1f} bb/100; "
        f"|diff|={diff:.1f} > combined95={combined:.1f}")


def test_multiway_equity_matches_phevaluator():
    """The native LBR showdown equity (`core::evaluate_seven`, higher=better) must agree with phevaluator
    (lower=better) at a full board -- the win/tie/loss flip is the #1 gotcha. Exact, 5000 random spots."""
    from phevaluator import evaluate_cards as phe

    from pokerbot.search import native_solver

    ext = native_solver._try_load()
    cs = ["23456789TJQKA"[c // 4] + "cdhs"[c % 4] for c in range(52)]

    def py_eq(lbr, opps, board):
        bs = [cs[c] for c in board]
        lh = phe(cs[lbr[0]], cs[lbr[1]], *bs)
        ohs = [phe(cs[o[0]], cs[o[1]], *bs) for o in opps]
        best = min(ohs)
        return 1.0 if lh < best else (1.0 / (1 + sum(o == lh for o in ohs)) if lh == best else 0.0)

    rng = np.random.default_rng(0)
    for _ in range(5000):
        n_opp = int(rng.integers(1, 4))
        cards = list(rng.choice(52, size=2 + 2 * n_opp + 5, replace=False))
        lbr = (int(cards[0]), int(cards[1]))
        opps = [(int(cards[2 + 2 * i]), int(cards[3 + 2 * i])) for i in range(n_opp)]
        board = [int(c) for c in cards[2 + 2 * n_opp:]]
        pe = py_eq(lbr, opps, board)
        ne = ext.multiway_equity_river(lbr[0], lbr[1], [c for o in opps for c in o], board)
        assert abs(pe - ne) < 1e-9, f"equity flip bug: py={pe} native={ne} board={board}"


def test_per_seat_breakdown_present():
    """The 6-max result carries a per-seat breakdown (used by the blueprint validator report)."""
    from pokerbot.eval import lbr_native as L

    r = L.lbr6_exploitability_native(None, LBR6Config(hands=1200, seed=0, n_seats=6), policy="fold")
    ps = r.detail["per_seat"]
    assert set(ps.keys()) == set(range(6)), f"expected 6 seats, got {sorted(ps.keys())}"
    assert all(v["n"] > 0 for v in ps.values())
