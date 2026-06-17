"""Precompute the EXACT (unbiased) 4-max legal BR marginal EVs at high deal count, in parallel.

WHY: the live 4-max marginals were deal-SAMPLED over an 8k batch whose effective per-bucket sample
collapses to ~1 deal at deep fold-prefixed nodes (BB facing two jams), giving EVs that contradict
the equilibrium and caused real superhuman misplays. The analytic 3-max reuse fixes the variance but
is biased ~0.3bb (it drops the folded UTG hand). The TRUE-game estimator (``_br_marginals`` over real
4-hand deals) is UNBIASED — its only error is sampling variance, which → 0 with deal count. The
per-deal payoff precompute is embarrassingly parallel and ``_br_marginals`` returns ADDITIVE sums, so
we fan out many independent deal chunks across all cores, sum, and divide once.

OUTPUT: ``data/aof/cfr_base/marginals_4p_d{depth}.npz`` per depth (a tiny dict: per (seat, history)
the (169,) ev_push / ev_fold + the per-bucket effective den for quality auditing), stamped with the
game model + showdown fee so a stale-model file is rejected at load. The live engine loads these
exact marginals (``fp_4max_vec.load_highN_marginals``); ``route_4max`` adds the analytic substitution
delta on top so the exploit BR stays unbiased AND range-responsive.

USAGE (on the 64 vCPU box):
  python scripts/aof/precompute_4max_marginals.py --depths 4 6 8 10 12 15 20 \
      --total-deals 25000000 --workers 62
A ~25M-deal run drives even the deepest reachable node's std well under ~0.1bb. Smaller --total-deals
(e.g. 1_000_000) is fine for a quick local sanity pass. Play-money / own-account evaluation only.
"""

from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from aof_cfr.exploit_router import _SEATS_4MAX
from aof_cfr.fp_4max_vec import accumulate_marginal_sums, marginal_sums

_DIR = Path("data/aof/cfr_base")


def marginals_disk_path(stack_bb: float) -> Path:
    return _DIR / f"marginals_4p_d{round(stack_bb, 3):g}.npz"


def _chunk(args):
    rows, stack, n, seed = args
    return marginal_sums(rows, stack, n, seed)


def precompute_depth(stack_bb: float, total_deals: int, workers: int, base_seed: int,
                     chunk: int) -> dict:
    from aof_cfr.profiler import base_policy_rows
    from aof_cfr.pushfold_payoffs import flat_fee_default, game_model_default
    from aof_cfr.query import load_cfr_base
    load_cfr_base()                          # needed for base_policy_rows (main process)
    rows = base_policy_rows(4, stack_bb)
    n_chunks = max(1, (total_deals + chunk - 1) // chunk)
    sizes = [chunk] * (n_chunks - 1) + [total_deals - chunk * (n_chunks - 1)]
    jobs = [(rows, stack_bb, sizes[i], base_seed + 1 + i) for i in range(n_chunks)]
    t0 = time.time()
    results = []
    if workers <= 1:
        for j in jobs:
            results.append(_chunk(j))
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for r in ex.map(_chunk, jobs):
                results.append(r)
    evs, den_tot = accumulate_marginal_sums(results)
    # serialize: flat arrays keyed "<seat>|<hist>"
    payload = {"game_model": np.int64(game_model_default()),
               "flat_fee": np.float64(flat_fee_default()),
               "total_deals": np.int64(sum(sizes)), "stack_bb": np.float64(stack_bb)}
    for (seat, hist), (ep, ef) in evs.items():
        payload[f"{seat}|{hist}|push"] = ep.astype(np.float64)
        payload[f"{seat}|{hist}|fold"] = ef.astype(np.float64)
    for (si, hist), d in den_tot.items():
        payload[f"{_SEATS_4MAX[si]}|{hist}|den"] = np.asarray(d, dtype=np.float64)
    _DIR.mkdir(parents=True, exist_ok=True)
    out = marginals_disk_path(stack_bb)
    np.savez_compressed(out, **payload)
    dt = time.time() - t0
    min_den = min(float(d.min()) for d in den_tot.values())
    print(f"  d{stack_bb:g}: {sum(sizes):,} deals in {dt:.0f}s ({n_chunks} chunks x{workers}) "
          f"-> {out.name}  worst-node min den={min_den:.0f}  std~{(6.0/max(min_den,1)**0.5):.2f}bb", flush=True)
    return evs


def main() -> None:
    ap = argparse.ArgumentParser(description="Exact high-N 4-max BR marginals (parallel)")
    ap.add_argument("--depths", type=float, nargs="+", default=[4, 6, 8, 10, 12, 15, 20])
    ap.add_argument("--total-deals", type=int, default=1_000_000)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--chunk", type=int, default=20000, help="deals per parallel task")
    ap.add_argument("--base-seed", type=int, default=777)
    args = ap.parse_args()
    print(f"Exact 4-max marginals: depths={args.depths} total_deals={args.total_deals:,} "
          f"workers={args.workers}", flush=True)
    for d in args.depths:
        precompute_depth(float(d), args.total_deals, args.workers, args.base_seed, args.chunk)
    print("done.", flush=True)


if __name__ == "__main__":
    main()
