"""TEMP diagnostic: marginal-level OLD-vs-NEW vs self-noise (variance vs bias).

Compares the 4-max BR marginal pipeline OLD (pure-Python board RNG) vs NEW (C++ mt19937 batch)
through marginal_sums, at increasing deal count N, alongside each path's OWN MC self-noise
(two independent deal seeds). If OLD-vs-NEW (same deals, different showdown-RNG stream) is the
same order as the self-noise and shrinks with N, the residual is pure MC variance, not bias.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.stdout.reconfigure(line_buffering=True)

import numpy as np

import aof_cfr.fp_4max_vec as fp
from aof_cfr.profiler import base_policy_rows
from aof_cfr.query import load_cfr_base


def evs(triple):
    num_push, num_fold, den = triple
    out = {}
    for k in den:
        d = den[k]
        safe = np.where(d > 0, d, 1.0)
        out[k] = (num_push[k] / safe, num_fold[k] / safe, d)
    return out


def maxdiff(a, b, min_den):
    md = 0.0
    cnt = 0
    # also track the SIGNED mean diff to detect a persistent offset (bias signature)
    sdiff = []
    for k in a:
        epa, efa, da = a[k]
        epb, efb, db = b[k]
        m = (da >= min_den) & (db >= min_den)
        if not m.any():
            continue
        md = max(md, float(np.abs(epa - epb)[m].max()), float(np.abs(efa - efb)[m].max()))
        cnt += int(m.sum())
        sdiff.append((epa - epb)[m])
        sdiff.append((efa - efb)[m])
    mean_signed = float(np.concatenate(sdiff).mean()) if sdiff else 0.0
    return md, cnt, mean_signed


def run(depth, min_den, ns):
    import time
    load_cfr_base()
    rows = base_policy_rows(4, depth)
    print(f"depth {depth}  min_den {min_den}", flush=True)
    # OLDvNEW = same deals, different board RNG (the validator's quantity)
    # OLDvOLD / NEWvNEW = each path's own MC self-noise (two independent deal seeds)
    # ON_mean = signed mean OLD-NEW diff (a persistent nonzero => bias signature)
    print(f"{'N':>8} {'OLDvNEW':>9} {'OLDvOLD':>9} {'NEWvNEW':>9} {'ON_mean':>9} {'cnt':>7} {'sec':>6}",
          flush=True)
    for N in ns:
        t0 = time.time()
        fp._SHARE_CACHE.clear(); fp._DISABLE_BATCH = True
        old1 = evs(fp.marginal_sums(rows, depth, N, 12345))
        fp._SHARE_CACHE.clear(); fp._DISABLE_BATCH = True
        old2 = evs(fp.marginal_sums(rows, depth, N, 67890))
        fp._SHARE_CACHE.clear(); fp._DISABLE_BATCH = False
        new1 = evs(fp.marginal_sums(rows, depth, N, 12345))
        fp._SHARE_CACHE.clear(); fp._DISABLE_BATCH = False
        new2 = evs(fp.marginal_sums(rows, depth, N, 67890))
        on, c, on_mean = maxdiff(old1, new1, min_den)
        oo, _, _ = maxdiff(old1, old2, min_den)
        nn, _, _ = maxdiff(new1, new2, min_den)
        print(f"{N:>8} {on:>9.4f} {oo:>9.4f} {nn:>9.4f} {on_mean:>9.5f} {c:>7} {time.time()-t0:>6.0f}",
              flush=True)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--depth", type=float, default=8.0)
    ap.add_argument("--min-den", type=float, default=40.0)
    ap.add_argument("--ns", type=int, nargs="+", default=[10000, 40000, 160000])
    a = ap.parse_args()
    run(a.depth, a.min_den, a.ns)
