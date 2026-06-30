"""Verify + benchmark the gadget-leaf prefix-sharing DAG (POKERBOT_LEAF_GADGET_BATCH).

Loads the captured postflop roots, solves each with the gadget leaf at a fixed seed, BASELINE
(DAG off) vs OPTIMIZED (DAG on), asserts the served action_probs are bit-identical across all
roots, and times both arms. Run from the worktree root with the serve env set.
"""
from __future__ import annotations

import os
import pickle
import sys
import time

# Force THIS checkout's pokerbot to win over any editable-install finder (the repo ships an
# __editable__.pokerbot-*.pth that otherwise resolves `pokerbot` to the MAIN worktree, shadowing a
# sibling worktree). Prepending the repo root (this file's grandparent) to sys.path[0] makes a plain
# `import pokerbot` resolve to the checkout this script lives in — required to bench the local change.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if sys.path[0:1] != [_REPO_ROOT]:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np

from pokerbot.search.pluribus_search import (
    PluribusReBeLConfig,
    PluribusReBeLSolver,
)
from pokerbot.search.pluribus_rollout import ContinuationLeafConfig, set_leaf_gadget_batch
from pokerbot.search import pluribus_rollout as R


class SubgameRoot:
    __slots__ = ("env", "hero_seat", "pbs_vector", "legal_mask", "opponent_profiles")

    def __init__(self, env, hero_seat, pbs_vector, legal_mask, opponent_profiles=None):
        self.env = env
        self.hero_seat = hero_seat
        self.pbs_vector = pbs_vector
        self.legal_mask = legal_mask
        self.opponent_profiles = opponent_profiles


class _PBS:
    def __init__(self, env):
        self.env = env

    def sample_opponent_holes(self, rng):
        from pokerbot.engine.np_engine import SEAT_FOLDED
        env = self.env
        holes = np.asarray(env.holes)
        board = np.asarray(env.board)
        seen = set(int(c) for c in holes.reshape(-1) if int(c) >= 0)
        seen |= set(int(c) for c in board.reshape(-1) if int(c) >= 0)
        avail = [c for c in range(52) if c not in seen]
        rng.shuffle(avail)
        out = {}
        ptr = 0
        status = np.asarray(env.seat_status)
        for s in range(holes.shape[0]):
            if int(holes[s, 0]) >= 0:
                continue
            if int(status[s]) == int(SEAT_FOLDED):
                continue
            out[s] = (int(avail[ptr]), int(avail[ptr + 1]))
            ptr += 2
        return out


def build_root(rec):
    env = rec["env"]
    hero = int(rec["hero_seat"])
    root = SubgameRoot(env=env, hero_seat=hero,
                       pbs_vector=np.zeros(1), legal_mask=np.ones(1, dtype=bool))
    return root, _PBS(env)


def load_snapshot(path: str) -> dict:
    obj = pickle.load(open(path, "rb"))
    if isinstance(obj, dict) and "strat_sum" in obj:
        return {bytes(k): np.asarray(v) for k, v in obj["strat_sum"].items()}
    return obj


def make_solver(snap, iters, rollout_samples):
    cfg = PluribusReBeLConfig(
        max_iterations=iters, time_budget_ms=10**9, search_preflop=True,
        validate_snapshot_at_init=False,
        leaf=ContinuationLeafConfig(select="gadget", rollout_samples=rollout_samples),
    )
    return PluribusReBeLSolver(snap, cfg)


def run_arm(roots, snap, iters, rollout_samples, dag_on):
    set_leaf_gadget_batch(dag_on)
    solver = make_solver(snap, iters, rollout_samples)
    built = [build_root(rec) for rec in roots]
    solver.solve(*built[0])  # warm
    probs = []
    t0 = time.perf_counter()
    for root, pbs in built:
        res = solver.solve(root, pbs)
        probs.append(np.asarray(res.action_probs))
    dt = time.perf_counter() - t0
    return probs, dt / len(built) * 1000.0


def main():
    blueprint = os.environ.get("BLUEPRINT_PATH", "checkpoints/serve_blueprint_6max.pkl")
    roots_path = sys.argv[1] if len(sys.argv) > 1 else "artifacts/search_roots.pkl"
    iters = int(os.environ.get("VERIFY_ITERS", "20"))
    rollout = int(os.environ.get("VERIFY_ROLLOUT", "4"))
    n_roots = int(os.environ.get("VERIFY_NROOTS", "60"))

    roots = pickle.load(open(roots_path, "rb"))[:n_roots]
    snap = load_snapshot(blueprint)
    print(f"loaded {len(roots)} roots, snapshot {len(snap)} infosets; iters={iters} rollout={rollout}",
          flush=True)

    base_probs, base_ms = run_arm(roots, snap, iters, rollout, dag_on=False)
    print(f"BASELINE (DAG off):   {base_ms:8.2f} ms/solve", flush=True)
    dag_probs, dag_ms = run_arm(roots, snap, iters, rollout, dag_on=True)
    print(f"OPTIMIZED (DAG on):   {dag_ms:8.2f} ms/solve", flush=True)

    set_leaf_gadget_batch(False)

    n_ident = 0
    max_abs = 0.0
    worst = None
    for idx, (b, d) in enumerate(zip(base_probs, dag_probs)):
        if np.array_equal(b, d):
            n_ident += 1
        else:
            diff = float(np.max(np.abs(b.astype(np.float64) - d.astype(np.float64))))
            if diff > max_abs:
                max_abs = diff
                worst = idx
    print(f"BIT-IDENTICAL: {n_ident}/{len(base_probs)} roots", flush=True)
    if n_ident != len(base_probs):
        print(f"  worst root {worst}: max|diff|={max_abs:.3e}", flush=True)
        b, d = base_probs[worst], dag_probs[worst]
        print(f"  base={np.array2string(b, precision=6)}", flush=True)
        print(f"  dag ={np.array2string(d, precision=6)}", flush=True)
    speedup = base_ms / dag_ms if dag_ms > 0 else float("nan")
    print(f"SPEEDUP: {speedup:.3f}x  ({'FASTER' if speedup > 1.0 else 'NOT FASTER'})", flush=True)
    ok = (n_ident == len(base_probs)) and (speedup > 1.0)
    print(f"RESULT: {'PASS' if ok else 'FAIL'}", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
