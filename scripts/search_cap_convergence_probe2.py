#!/usr/bin/env python3
"""Probe 2: is the prod-cap L1 residual REAL under-convergence or Monte-Carlo noise?

The native exploit solve uses a public-state-derived seed (independent of max_iterations), so more
iters = a longer prefix of the SAME sampled-world stream. We must separate two things:
  (1) DETERMINISM: same iters + same salt -> identical served probs (no run-to-run noise)?
  (2) MC NOISE FLOOR: at a FIXED iter count, how much do served probs move when only the salt (the
      MC stream) changes? If the salt-to-salt spread at 1500 iters is ~ the 200-vs-3000 gap, the
      residual is variance, not convergence lag. If the salt-spread SHRINKS with iters while the
      1-salt curve keeps moving toward a DEEPER (6000) ref, it's genuine under-convergence.
  (3) DEEPER REF: L1 of {200,1500,3000} vs a 6000-iter ref, single salt AND salt-averaged, to see
      whether 3000 itself is settled or the whole thing is still drifting.

Env as always (baked at import):
  POKERBOT_BETTING_ABSTRACTION=1 POKERBOT_POSTFLOP_BUCKETS=1 POKERBOT_PREFLOP_LOSSLESS=1 \
  POKERBOT_POSTFLOP_MODE=fast POKERBOT_POSTFLOP_NUM_BUCKETS=200
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE, os.path.dirname(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from native_exploit_equiv import build_root, load_snapshot, make_opponent_model  # noqa: E402
from search_quality_3s_ab import make_solver, _solve_once  # noqa: E402


def solve_grid(snap, built, model, iters, salt):
    """Served probs for every root at one iter count / one salt. (n_roots, 11)."""
    s = make_solver(snap, max_iters=iters, time_budget_ms=10**12,
                    native=True, native_exploit=True, dag=True)
    s.solve(*built[0])
    out = np.zeros((len(built), 11))
    for ri, (root, pbs) in enumerate(built):
        p, _, tag = _solve_once(s, root, pbs, model, salt)
        assert tag == "search_native_exploit", tag
        out[ri, :p.shape[0]] = p
    return out


def main():
    from pokerbot.search import native_solver as ns
    print("native exploit:", ns.native_subgame_exploit_available(), flush=True)

    bp = os.environ.get("BLUEPRINT_PATH", "checkpoints/serve_blueprint_6max.pkl")
    roots_path = os.environ.get("ROOTS_PATH", "artifacts/search_roots.pkl")
    n_roots = int(os.environ.get("SC_ROOTS", "5"))

    import pickle
    snap = load_snapshot(bp)
    roots = pickle.load(open(roots_path, "rb"))[:n_roots]
    built = [build_root(r) for r in roots]
    hero = int(roots[0]["hero_seat"])
    model = make_opponent_model(list(range(6)), hero)
    print(f"infosets={len(snap):,} roots={len(built)}", flush=True)

    salts = [12345, 23456, 34567, 45678, 56789, 67890]

    # (1) DETERMINISM: solve at 200 iters, salt 12345, twice.
    a = solve_grid(snap, built, model, 200, 12345)
    b = solve_grid(snap, built, model, 200, 12345)
    det = float(np.abs(a - b).sum(axis=1).mean())
    print(f"\n(1) DETERMINISM 200it/salt12345 x2: mean L1 between two runs = {det:.6f} "
          f"({'DETERMINISTIC' if det < 1e-9 else 'NON-DETERMINISTIC'})", flush=True)

    # (2) MC NOISE FLOOR at fixed iters across salts + salt-averaged strategy.
    # Also (3) build a deep 6000 ref (salt-averaged) as the anchor.
    iter_pts = [200, 800, 1500, 3000, 6000]
    per_iter = {}  # iters -> (n_salts, n_roots, 11)
    t_all = time.perf_counter()
    for it in iter_pts:
        arr = np.zeros((len(salts), len(built), 11))
        t0 = time.perf_counter()
        for si, sa in enumerate(salts):
            arr[si] = solve_grid(snap, built, model, it, sa)
        per_iter[it] = arr
        print(f"  iters={it:>5}: {len(salts)} salts done in {time.perf_counter()-t0:6.1f}s", flush=True)
    print(f"  (all solves {time.perf_counter()-t_all:.1f}s)", flush=True)

    # MC noise floor at each iter count = mean over salt-pairs of L1(served_saltA, served_saltB),
    # averaged over roots. (How much a single-salt served strategy jitters purely from the MC stream.)
    print("\n(2) MC NOISE FLOOR: mean pairwise L1 across salts (single-salt jitter) per iter count")
    print(f"{'iters':>6} {'mean pairwise L1':>17} {'salt-avg self-spread':>22}")
    salt_avg = {}
    for it in iter_pts:
        arr = per_iter[it]
        ns_ = len(salts)
        pair = []
        for i in range(ns_):
            for j in range(i + 1, ns_):
                pair.append(np.abs(arr[i] - arr[j]).sum(axis=1).mean())
        mean_pair = float(np.mean(pair))
        salt_avg[it] = arr.mean(axis=0)  # (n_roots, 11) salt-averaged strategy
        # self-spread = mean over salts of L1(single-salt, salt-avg)
        self_spread = float(np.abs(arr - salt_avg[it][None]).sum(axis=2).mean())
        print(f"{it:>6} {mean_pair:>17.4f} {self_spread:>22.4f}", flush=True)

    # (3) DEEPER REF: L1 of salt-averaged {200,800,1500,3000} vs salt-averaged 6000 ref, and
    # single-salt (12345) vs the same 6000 salt-avg ref. Salt-averaging removes the MC noise so the
    # residual here is TRUE convergence lag of the running average.
    ref6k = salt_avg[6000]
    print("\n(3) CONVERGENCE vs 6000-iter SALT-AVERAGED reference (MC noise averaged out)")
    print(f"{'iters':>6} {'saltavg L1->6kref':>18} {'1salt L1->6kref':>17}")
    for it in [200, 800, 1500, 3000]:
        sa_l1 = float(np.abs(salt_avg[it] - ref6k).sum(axis=1).mean())
        one_l1 = float(np.abs(per_iter[it][0] - ref6k).sum(axis=1).mean())
        print(f"{it:>6} {sa_l1:>18.4f} {one_l1:>17.4f}", flush=True)

    # Verdict aids
    floor_1500 = None
    # compute the salt-avg self-spread at 3000 as the "irreducible average jitter" baseline
    print("\nInterpretation aids:")
    p200 = float(np.abs(salt_avg[200] - ref6k).sum(axis=1).mean())
    p1500 = float(np.abs(salt_avg[1500] - ref6k).sum(axis=1).mean())
    p3000 = float(np.abs(salt_avg[3000] - ref6k).sum(axis=1).mean())
    print(f"  salt-avg L1->6kref shrinks 200:{p200:.3f} -> 1500:{p1500:.3f} -> 3000:{p3000:.3f}")
    print(f"  => if this shrinks monotonically & 3000<<200, the residual is REAL convergence lag,")
    print(f"     NOT MC noise (which salt-averaging removed).")

    import json
    summary = {
        "determinism_mean_l1": det,
        "salts": salts,
        "mc_pairwise_l1_by_iter": {},
        "salt_avg_self_spread_by_iter": {},
        "saltavg_l1_to_6kref": {},
        "onesalt_l1_to_6kref": {},
    }
    for it in iter_pts:
        arr = per_iter[it]
        ns_ = len(salts)
        pair = [np.abs(arr[i] - arr[j]).sum(axis=1).mean()
                for i in range(ns_) for j in range(i + 1, ns_)]
        summary["mc_pairwise_l1_by_iter"][str(it)] = float(np.mean(pair))
        summary["salt_avg_self_spread_by_iter"][str(it)] = \
            float(np.abs(arr - salt_avg[it][None]).sum(axis=2).mean())
    for it in [200, 800, 1500, 3000]:
        summary["saltavg_l1_to_6kref"][str(it)] = float(np.abs(salt_avg[it] - ref6k).sum(axis=1).mean())
        summary["onesalt_l1_to_6kref"][str(it)] = float(np.abs(per_iter[it][0] - ref6k).sum(axis=1).mean())
    outp = os.path.join("artifacts", "search_cap_convergence_probe2.json")
    with open(outp, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nwrote {outp}", flush=True)


if __name__ == "__main__":
    main()
