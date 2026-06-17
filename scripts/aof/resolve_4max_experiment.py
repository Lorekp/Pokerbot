"""4-max base re-solve experiment (d=8 first): does more FP convergence cut out-of-sample
exploitability below the shipped ~28 mbb/g?

The shipped base used solve_4max_fp(max_rounds=30, br_samples=4000) — drastically
under-converged. Here we:
  1. Measure the SHIPPED base out-of-sample on fresh held-out eval batches (3 seeds).
  2. Re-solve with the vectorized FP solver at high rounds, on (a) the existing on-disk
     8k batch (rounds-only, free) and (b) a larger freshly-precomputed train batch.
  3. Measure each re-solved policy on the SAME held-out eval batches.

Out-of-sample is the honest metric: training and eval deals are disjoint, so a policy that
merely overfits its training batch shows no eval-gap improvement.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np

from aof_cfr.fp_4max_vec import (
    _infoset_index,
    legal_nash_gap_4max,
    precompute_payoffs,
    rows_to_table,
    sample_deals,
    solve_4max_fp_vec,
)
from aof_cfr.profiler import base_policy_rows

DEPTH = 8.0
_LOG = Path("runs/aof_audit/resolve_4max_d8.log")


def emit(s: str) -> None:
    print(s, flush=True)
    _LOG.parent.mkdir(parents=True, exist_ok=True)
    with _LOG.open("a", encoding="utf-8") as fh:
        fh.write(s + "\n")


def gap_of_rows_on_batch(rows, buckets, payoffs) -> float:
    """Out-of-sample legal NashConv (mbb/g) of a policy on a held-out (buckets,payoffs) batch."""
    pol = rows_to_table(rows, DEPTH)
    infosets = _infoset_index(buckets)
    return legal_nash_gap_4max(pol, buckets, payoffs, infosets)[0] * 1000.0


def main() -> None:
    import aof_cfr.query as q
    q.load_cfr_base()

    emit("=" * 74)
    emit(f"4-MAX RE-SOLVE EXPERIMENT @ d={DEPTH}bb  (out-of-sample, honest)")
    emit("=" * 74)

    # ---- held-out eval batches (disjoint from any training batch) ----
    eval_seeds = [9001, 9002, 9003]
    eval_n = 30000
    emit(f"\nBuilding {len(eval_seeds)} held-out eval batches @ {eval_n} deals ...")
    eval_batches = []
    for s in eval_seeds:
        t = time.time()
        b, h = sample_deals(eval_n, s)
        p = precompute_payoffs(h, DEPTH)
        eval_batches.append((b, p))
        emit(f"  eval seed {s}: precompute {time.time()-t:.0f}s")

    def report(label, rows):
        vals = [gap_of_rows_on_batch(rows, b, p) for (b, p) in eval_batches]
        emit(f"  {label:<34} out-of-sample gap: "
             f"mean={np.mean(vals):7.3f}  per-seed={[round(v,2) for v in vals]} mbb/g")
        return float(np.mean(vals))

    # ---- 1. shipped base ----
    emit("\n[1] SHIPPED base (solve_4max_fp, ~30 rounds @ 4k):")
    shipped_rows = base_policy_rows(4, DEPTH)
    base_gap = report("shipped", shipped_rows)

    # ---- 2a. re-solve: rounds-only on existing on-disk 8k batch ----
    disk = Path(f"data/aof/cfr_base/mc_batch_4p_d{int(DEPTH)}_n8000_s12345.npz")
    if disk.is_file():
        emit(f"\n[2a] RE-SOLVE rounds-only on existing {disk.name} (8k deals):")
        dat = np.load(disk)
        b8, p8 = dat["buckets"], dat["payoffs"]
        for rounds in (200, 1000, 3000):
            t = time.time()
            pol = solve_4max_fp_vec(buckets=b8, payoffs=p8, max_rounds=rounds, stack_bb=DEPTH)
            emit(f"  rounds={rounds:>4}  in-sample={pol['nash_conv_bb']*1000:6.3f}  "
                 f"({time.time()-t:.0f}s)")
            report(f"resolved 8k x{rounds}r", pol["rows"])
    else:
        emit(f"\n[2a] SKIP: {disk} not found")

    # ---- 2b. re-solve: larger fresh train batch ----
    for train_n, rounds in ((40000, 1500),):
        emit(f"\n[2b] RE-SOLVE on fresh {train_n}-deal train batch, {rounds} rounds:")
        t = time.time()
        bt, ht = sample_deals(train_n, 42)
        pt = precompute_payoffs(ht, DEPTH)
        emit(f"  train precompute {time.time()-t:.0f}s")
        t = time.time()
        pol = solve_4max_fp_vec(buckets=bt, payoffs=pt, max_rounds=rounds, stack_bb=DEPTH)
        emit(f"  solve {time.time()-t:.0f}s  in-sample={pol['nash_conv_bb']*1000:.3f} mbb/g")
        report(f"resolved {train_n//1000}k x{rounds}r", pol["rows"])

    emit("\nDONE.")


if __name__ == "__main__":
    main()
