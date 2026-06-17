"""Bit-identity tests for the native AIVAT core (``pokerbot.eval.aivat_native``).

Every native output MUST equal the Python oracle (``pokerbot.eval.aivat`` / ``chance_term``) bit-for-bit
(exact f64 equality, NOT approximate). Covers:
  * the captured golden vectors (incl. NumPy pairwise-summation trap sizes + real blueprint reveals),
  * a fresh live cross-check vs the oracle on random inputs (decision / chance_reduce / accumulator),
  * the CPython-`sum()` Neumaier-compensation trap in ``AIVATAccumulator.add_hand``.

These pass on BOTH backends (compiled Rust extension and the pure-python fallback) — the fallback is
itself bit-identical, so the suite is green whether or not the .pyd is built.
"""

from __future__ import annotations

import os
import struct

os.environ.setdefault("POKERBOT_BETTING_ABSTRACTION", "1")

import numpy as np

from pokerbot.eval import aivat as oracle
from pokerbot.eval import aivat_native as nat


def _bits(x: float) -> bytes:
    return struct.pack("<d", float(x))


def _assert_bits(a: float, b: float, msg: str = "") -> None:
    assert _bits(a) == _bits(b), f"{msg}: {a!r} != {b!r}"


def test_golden_vectors_bit_identical():
    res = nat.verify_against_golden(verbose=False)
    assert all(v["ok"] for k, v in res.items() if k != "_backend")
    assert sum(v["cases"] for k, v in res.items() if k != "_backend") >= 44


def test_decision_correction_matches_oracle_random():
    rng = np.random.default_rng(101)
    for _ in range(5000):
        k = int(rng.integers(2, 12))
        p = rng.random(k)
        if rng.random() < 0.1:
            p[:] = 0.0
        bv = rng.standard_normal(k) * 9.0
        ti = int(rng.integers(-1, k + 1))
        _assert_bits(oracle.decision_correction(p, bv, ti),
                     nat.decision_correction(p, bv, ti), "decision_correction")


def test_chance_correction_matches_oracle():
    rng = np.random.default_rng(202)
    for _ in range(2000):
        e = float(rng.standard_normal() * 7)
        r = float(rng.standard_normal() * 7)
        _assert_bits(oracle.chance_correction(e, r), nat.chance_correction(e, r),
                     "chance_correction")


def test_chance_reduce_matches_oracle_across_pairwise_trap_sizes():
    """``realized - float(np.mean(vals))`` over sizes straddling NumPy's 128-elt pairwise threshold."""
    rng = np.random.default_rng(303)
    for sz in (0, 1, 7, 37, 47, 64, 127, 128, 129, 200, 257, 1000, 4096):
        for _ in range(50):
            realized = float(rng.standard_normal() * 13)
            vals = (rng.standard_normal(sz) * 13).tolist()
            expected = float(np.mean(vals)) if vals else float(realized)
            want = float(realized) - expected
            _assert_bits(want, nat.chance_reduce(realized, vals), f"chance_reduce(sz={sz})")


def test_accumulator_summary_matches_oracle_random():
    rng = np.random.default_rng(404)
    for _ in range(60):
        H = int(rng.integers(0, 600))
        oa = oracle.AIVATAccumulator()
        na = nat.AIVATAccumulator()
        for _ in range(H):
            raw = float(rng.standard_normal() * 20)
            nc = int(rng.integers(0, 4))
            corr = rng.standard_normal(nc).tolist() if nc else None
            nch = int(rng.integers(0, 3))
            ch = rng.standard_normal(nch).tolist() if nch else None
            oa.add_hand(raw, corr, ch)
            na.add_hand(raw, corr, ch)
        os_, ns_ = oa.summary(), na.summary()
        assert os_["hands"] == ns_["hands"]
        for kk in ("raw_mean_bb", "adj_mean_bb", "raw_std", "adj_std", "variance_reduction"):
            _assert_bits(os_[kk], ns_[kk], f"accumulator.{kk} (H={H})")


def test_add_hand_neumaier_sum_trap():
    """CPython `sum()` uses Neumaier compensation, NOT a naive fold — the native core must too.

    This specific 3-element correction list differs by 1 ULP between a naive fold and `sum()`."""
    raw = -0.12697153429002567
    corr = [-0.021187260672077567, 1.1874175363532256, -0.37966138518168174]
    oa = oracle.AIVATAccumulator()
    oa.add_hand(raw, corr, None)
    _assert_bits(oa._adj[0], nat.add_hand_adjusted(raw, corr, None), "neumaier add_hand")
