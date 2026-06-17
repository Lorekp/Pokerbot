"""Validate the batched native showdown kernel at the MARGINAL level (the bar that protects the
equilibrium). Runs the full ``marginal_sums`` pipeline OLD (pure-Python loop) vs NEW (batched native)
over the same deals/depths and asks the only question that matters: **is the OLD-vs-NEW residual MC
variance, or a real bias in the C++ kernel?**

OLD and NEW use the SAME OMPEval evaluator + SAME _N_BOARDS + SAME per-showdown seed scheme; they
differ only in the board RNG stream (Python random.Random vs C++ mt19937). So OLD-vs-NEW (run on the
SAME deals) is two independent MC estimates of the *same* per-cell EV, and any disagreement is pure
showdown-MC noise UNLESS the kernel is biased.

The acceptance bar is **self-calibrating to that MC noise**, not a flat tolerance. A flat 0.05 bb bar
is wrong here: ``den`` is a reach-weight sum, not a distinct-showdown count, so even at den>=40 the
deep fold-prefixed / 4-way nodes are dominated by a handful of distinct showdowns and carry large,
heavy-tailed per-cell noise (NEW-vs-NEW self-noise at den=40 is ~1 bb, not 0.05). So we MEASURE each
path's own self-noise (two independent deal seeds) and require:

  (1) max |ev_OLD - ev_NEW| (same deals)  <=  max(abs_floor, factor * self_noise)
        — the cross-RNG diff is no larger than a path's intrinsic MC self-noise => variance, not bias.
  (2) |signed mean(ev_OLD - ev_NEW)|  <=  bias_floor
        — a *persistent* offset that does not average out is the signature of a real bias. MC variance
          is mean-zero, so the signed mean over all decided cells must be ~0.
  (3) best-response ACTION per bucket (push iff ev_push>ev_fold) identical except at near-indifference
        cells (|ev_push - ev_fold| < indiff_bb) — flips inside the noise band are expected and benign.

The Python (OLD) path is ~10-23 ms/deal, so keep --deals modest; NEW is ~0.5 ms/deal.

  python scripts/aof/validate_showdown_batch.py [--deals 12000] [--depths 6 8] [--min-den 40]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

import aof_cfr.fp_4max_vec as fp
from aof_cfr.fp_4max_vec import marginal_sums
from aof_cfr.profiler import base_policy_rows
from aof_cfr.query import load_cfr_base


def _evs(num_push, num_fold, den, min_den):
    """{(seat,hist): (ev_push, ev_fold, mask)} for buckets with den>=min_den."""
    out = {}
    for k in den:
        d = den[k]
        m = d >= min_den
        safe = np.where(d > 0, d, 1.0)
        out[k] = (num_push[k] / safe, num_fold[k] / safe, m)
    return out


def _max_and_signed(a, b):
    """max |a-b| over the shared mask, and the signed mean diff (bias signature), over both
    ev_push and ev_fold. ``a``/``b`` are _evs dicts."""
    mx = 0.0
    pieces = []
    for k in a:
        epa, efa, ma = a[k]
        epb, efb, mb = b[k]
        m = ma & mb
        if not m.any():
            continue
        mx = max(mx, float(np.abs(epa - epb)[m].max()), float(np.abs(efa - efb)[m].max()))
        pieces.append((epa - epb)[m])
        pieces.append((efa - efb)[m])
    signed = float(np.concatenate(pieces).mean()) if pieces else 0.0
    n = int(sum(p.size for p in pieces))
    return mx, signed, n


def _action_flips(a, b, indiff_bb):
    """BR-action flips on cells decided (|gap|>=indiff) in BOTH, over the shared den mask."""
    flips = 0
    decided = 0
    for k in a:
        epa, efa, ma = a[k]
        epb, efb, mb = b[k]
        m = ma & mb
        if not m.any():
            continue
        gap_o = (epa - efa)[m]
        gap_n = (epb - efb)[m]
        dec = (np.abs(gap_o) >= indiff_bb) & (np.abs(gap_n) >= indiff_bb)
        decided += int(dec.sum())
        flips += int(((np.sign(gap_o) != np.sign(gap_n)) & dec).sum())
    return flips, decided


def run(deals: int, depths, min_den: float, factor: float, abs_floor: float, bias_floor: float,
        indiff_bb: float, seed: int) -> bool:
    load_cfr_base()
    ok = True
    print(f"validating batched showdown (self-noise-referenced): deals={deals} depths={list(depths)} "
          f"min_den={min_den} factor={factor} abs_floor={abs_floor} bias_floor={bias_floor} "
          f"indiff={indiff_bb}", flush=True)
    for depth in depths:
        rows = base_policy_rows(4, depth)
        # CRITICAL: the Python loop and the batched path share exploit_mc._SHARE_CACHE (same object).
        # Clear it between EVERY run or a later run reads back an earlier run's cached shares → a false
        # 0.0 diff. Clearing forces each run to compute its own showdown shares with its own RNG.
        fp._SHARE_CACHE.clear(); fp._DISABLE_BATCH = True
        old = _evs(*marginal_sums(rows, depth, deals, seed), min_den)            # OLD @ seed
        fp._SHARE_CACHE.clear(); fp._DISABLE_BATCH = False
        new = _evs(*marginal_sums(rows, depth, deals, seed), min_den)            # NEW @ seed (same deals)
        fp._SHARE_CACHE.clear(); fp._DISABLE_BATCH = False
        new2 = _evs(*marginal_sums(rows, depth, deals, seed + 982_451), min_den)  # NEW @ other deals

        # cross-RNG diff (same deals) vs the NEW path's own self-noise (independent deals)
        cross_max, cross_signed, ncells = _max_and_signed(old, new)
        self_max, _self_signed, _ = _max_and_signed(new, new2)
        # The action-flip "decided" band MUST scale with the MC noise floor: a bucket whose ev_push/
        # ev_fold gap is below the estimator's own self-noise is not statistically decided at this N,
        # so a sign flip there is noise, not disagreement. flip_band -> indiff_bb as self_max -> 0
        # (high N), so this stays strict where it can and only widens where the sample is too thin.
        flip_band = max(indiff_bb, 0.5 * self_max)
        flips, decided = _action_flips(old, new, flip_band)

        bar = max(abs_floor, factor * self_max)
        cells_ok = ncells > 0   # an empty comparison must NOT pass vacuously (raise --deals)
        var_ok = cross_max <= bar
        bias_ok = abs(cross_signed) <= bias_floor
        flip_ok = flips == 0
        passed = cells_ok and var_ok and bias_ok and flip_ok
        ok = ok and passed
        print(f"  depth {depth:>4}: cross max|ev diff| = {cross_max:.4f} bb   "
              f"NEW self-noise = {self_max:.4f} bb   bar(={factor}x) = {bar:.4f}   "
              f"signed mean = {cross_signed:+.5f} bb (<= {bias_floor})   "
              f"BR flips = {flips}/{decided} (band={flip_band:.3f})   cells={ncells}   "
              f"-> {'PASS' if passed else 'FAIL'}"
              + ("" if cells_ok else "  [NO CELLS >= min_den — raise --deals]")
              + ("" if var_ok else "  [VARIANCE BAR]")
              + ("" if bias_ok else "  [BIAS: persistent offset!]")
              + ("" if flip_ok else "  [ACTION FLIP]"),
              flush=True)
    print("RESULT:", "PASS" if ok else "FAIL", flush=True)
    return ok


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--deals", type=int, default=12000)
    ap.add_argument("--depths", type=float, nargs="+", default=[6.0, 8.0])
    ap.add_argument("--min-den", type=float, default=40.0)
    ap.add_argument("--factor", type=float, default=1.5,
                    help="cross-RNG diff must be <= factor * (NEW path's own MC self-noise)")
    ap.add_argument("--abs-floor", type=float, default=0.05,
                    help="absolute floor for the variance bar (covers tiny self-noise at high den)")
    ap.add_argument("--bias-floor", type=float, default=0.01,
                    help="max |signed mean diff|; a persistent nonzero offset is a bias signature")
    ap.add_argument("--indiff-bb", type=float, default=0.05,
                    help="near-indifference band excluded from the action-flip check")
    ap.add_argument("--seed", type=int, default=12345)
    args = ap.parse_args()
    ok = run(args.deals, args.depths, args.min_den, args.factor, args.abs_floor, args.bias_floor,
             args.indiff_bb, args.seed)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
