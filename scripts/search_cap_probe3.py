#!/usr/bin/env python3
"""Probe 3: characterize WHAT is moving. Print the actual served prob vectors per root at
{200, 1500, 3000, 6000} single-salt, and the salt-averaged vectors, so we can see whether the L1
motion is major action-mass shifts (real under-search cost) or razor-edge reshuffles."""
from __future__ import annotations
import os, sys
import numpy as np
_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE, os.path.dirname(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from native_exploit_equiv import build_root, load_snapshot, make_opponent_model  # noqa
from search_quality_3s_ab import make_solver, _solve_once  # noqa


def solve_grid(snap, built, model, iters, salt):
    s = make_solver(snap, max_iters=iters, time_budget_ms=10**12,
                    native=True, native_exploit=True, dag=True)
    s.solve(*built[0])
    out = np.zeros((len(built), 11))
    for ri, (root, pbs) in enumerate(built):
        p, _, tag = _solve_once(s, root, pbs, model, salt)
        out[ri, :p.shape[0]] = p
    return out


def main():
    bp = "checkpoints/serve_blueprint_6max.pkl"
    import pickle
    snap = load_snapshot(bp)
    roots = pickle.load(open("artifacts/search_roots.pkl", "rb"))[:5]
    built = [build_root(r) for r in roots]
    hero = int(roots[0]["hero_seat"])
    model = make_opponent_model(list(range(6)), hero)

    salt = 12345
    grids = {it: solve_grid(snap, built, model, it, salt) for it in [200, 1500, 3000, 6000]}
    # argmax stability per root across iter counts
    print("=== single-salt (12345) served prob vectors, rounded, per root ===")
    for ri in range(len(built)):
        print(f"\n-- root {ri} --")
        for it in [200, 1500, 3000, 6000]:
            v = grids[it][ri]
            nz = {a: round(float(v[a]), 3) for a in range(11) if v[a] > 0.005}
            print(f"  {it:>5}it argmax={int(np.argmax(v)):>2}  {nz}")
    # argmax-flip count across the four iter counts (prod-relevant: does the CHOSEN action change?)
    print("\n=== argmax across {200,1500,3000,6000} (does the served ACTION change with iters?) ===")
    flips = 0
    for ri in range(len(built)):
        ams = [int(np.argmax(grids[it][ri])) for it in [200, 1500, 3000, 6000]]
        changed = len(set(ams)) > 1
        flips += changed
        print(f"  root {ri}: argmax {ams} {'FLIPS' if changed else 'stable'}")
    print(f"\n{flips}/{len(built)} roots change served argmax between 200 and 6000 iters (single salt)")


if __name__ == "__main__":
    main()
