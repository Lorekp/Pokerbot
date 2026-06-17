"""Live cross-check (native vs the actual Python oracle on fresh random inputs) + micro-benchmark.

    POKERBOT_BETTING_ABSTRACTION=1 python tools/aivat_native/bench_and_crosscheck.py
"""

from __future__ import annotations

import os
import struct
import time

import numpy as np

os.environ.setdefault("POKERBOT_BETTING_ABSTRACTION", "1")

from pokerbot.eval import aivat as oracle
from pokerbot.eval import aivat_native as nat


def bits(x):
    return struct.pack("<d", float(x))


def crosscheck(n=20000):
    rng = np.random.default_rng(20260609)
    mism = 0

    # decision_correction on random rows of varied length
    for _ in range(n):
        k = int(rng.integers(2, 12))
        p = rng.random(k)
        if rng.random() < 0.1:
            p[:] = 0.0  # exercise the s<=0 guard
        bv = rng.standard_normal(k) * 9.0
        ti = int(rng.integers(-1, k + 1))
        a = oracle.decision_correction(p, bv, ti)
        b = nat.decision_correction(p, bv, ti)
        if bits(a) != bits(b):
            mism += 1
            if mism <= 3:
                print("decision mismatch", a, b, "k", k, "ti", ti)
    print(f"decision_correction: {n} random cases, mismatches={mism}")

    # chance_reduce across pairwise-trap sizes + small enumeration sizes
    mism2 = 0
    sizes = [0, 1, 7, 37, 47, 64, 127, 128, 129, 200, 257, 1000, 4096]
    per = max(1, n // len(sizes) // 5)
    for sz in sizes:
        for _ in range(per * 5):
            realized = float(rng.standard_normal() * 13)
            vals = (rng.standard_normal(sz) * 13).tolist()
            expected = float(np.mean(vals)) if vals else float(realized)
            a = float(realized) - expected
            b = nat.chance_reduce(realized, vals)
            if bits(a) != bits(b):
                mism2 += 1
                if mism2 <= 3:
                    print("chance_reduce mismatch sz", sz, a, b)
    print(f"chance_reduce: {len(sizes)*per*5} random cases, mismatches={mism2}")

    # accumulator summary on random series of varied length
    mism3 = 0
    for _ in range(200):
        H = int(rng.integers(0, 800))
        oacc = oracle.AIVATAccumulator()
        nacc = nat.AIVATAccumulator()
        for _ in range(H):
            raw = float(rng.standard_normal() * 20)
            nc = int(rng.integers(0, 4))
            corr = rng.standard_normal(nc).tolist() if nc else None
            nch = int(rng.integers(0, 3))
            ch = rng.standard_normal(nch).tolist() if nch else None
            oacc.add_hand(raw, corr, ch)
            nacc.add_hand(raw, corr, ch)
        os_, ns_ = oacc.summary(), nacc.summary()
        for kk in ("raw_mean_bb", "adj_mean_bb", "raw_std", "adj_std", "variance_reduction"):
            if bits(os_[kk]) != bits(ns_[kk]):
                mism3 += 1
                if mism3 <= 3:
                    print("accumulator mismatch", kk, os_[kk], ns_[kk], "H", H)
    print(f"accumulator: 200 random series, mismatches={mism3}")
    return mism + mism2 + mism3


def bench():
    rng = np.random.default_rng(7)
    print("\n=== MICRO-BENCHMARK (native core vs python oracle) ===")

    # chance_reduce over 47-element enumeration (the real turn/river case), many reveals.
    N = 200000
    vals_list = [(float(rng.standard_normal()), (rng.standard_normal(47)).tolist()) for _ in range(2000)]

    def run_py():
        out = 0.0
        for realized, vals in vals_list:
            v = np.asarray(vals, dtype=np.float64)
            expected = float(np.mean(v)) if v.size else realized
            out += realized - expected
        return out

    def run_nat():
        out = 0.0
        for realized, vals in vals_list:
            out += nat.chance_reduce(realized, vals)
        return out

    reps = 30
    t0 = time.perf_counter()
    for _ in range(reps):
        run_py()
    tpy = (time.perf_counter() - t0) / reps
    t0 = time.perf_counter()
    for _ in range(reps):
        run_nat()
    tnat = (time.perf_counter() - t0) / reps
    print(f"chance_reduce x{len(vals_list)} (47-elt): python={tpy*1e3:.3f} ms  native={tnat*1e3:.3f} ms  "
          f"speedup={tpy/tnat:.2f}x")

    # accumulator summary on a large series (numpy std vs native)
    raw = (rng.standard_normal(50000) * 20)
    adj = raw - rng.standard_normal(50000)

    def acc_py():
        r = np.asarray(raw); a = np.asarray(adj)
        rstd = float(r.std(ddof=1)); astd = float(a.std(ddof=1))
        return (float(r.mean()), float(a.mean()), rstd, astd,
                1 - (astd/rstd)**2 if rstd > 0 else 0.0)

    mod = nat.load_native()

    def acc_nat():
        return mod.accumulator_summary(raw.tolist(), adj.tolist())

    reps = 200
    t0 = time.perf_counter()
    for _ in range(reps):
        acc_py()
    apy = (time.perf_counter() - t0) / reps
    t0 = time.perf_counter()
    for _ in range(reps):
        acc_nat()
    anat = (time.perf_counter() - t0) / reps
    print(f"accumulator_summary (50k hands): python={apy*1e3:.3f} ms  native={anat*1e3:.3f} ms  "
          f"speedup={apy/anat:.2f}x  (native incl. tolist() marshaling)")

    # Pure compute (exclude marshaling): time the native np_sum vs np.sum on a big array.
    big = (rng.standard_normal(100000)).tolist()
    bigarr = np.asarray(big)
    reps = 300
    t0 = time.perf_counter()
    for _ in range(reps):
        float(np.sum(bigarr))
    spy = (time.perf_counter() - t0) / reps
    t0 = time.perf_counter()
    for _ in range(reps):
        mod.np_sum(big)
    snat = (time.perf_counter() - t0) / reps
    print(f"pairwise sum (100k): np.sum={spy*1e3:.3f} ms  native np_sum={snat*1e3:.3f} ms "
          f"(native pays list->Vec marshaling; np.sum is C+SIMD on a contiguous array)")


if __name__ == "__main__":
    print("backend:", "native" if nat.native_available() else "python-fallback")
    total = crosscheck()
    print(f"\nTOTAL MISMATCHES vs live oracle: {total}")
    assert total == 0, "BIT-IDENTITY FAILURE"
    bench()
    print("\nALL CROSS-CHECKS BIT-IDENTICAL.")
