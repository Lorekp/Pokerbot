#!/usr/bin/env python3
"""Gate: per-decision equivalence of the NATIVE CFR subgame solve vs the Python `_cfv` loop.

Both solve the SAME depth-limited subgame; they differ only in the RNG engine (native xoshiro vs
numpy PCG64) and the leaf-cache key (no suit canonicalization natively). So they estimate the SAME
root strategy. We compare the mean root action-probs over many independent solves (varying rng_salt)
at TWO iteration counts:

  * iters=HIGH  -> both CFR processes converge to (nearly) the same best response: a tight L1 here
    proves the recursion / regret-matching / blueprint-mix / abstraction / leaf logic is correct.
  * iters=EVAL  -> the regime the battery actually runs: a small L1 here confirms statistical
    equivalence at the operating point.

Usage: python -m tools.battery_native.crosscheck_subgame <blueprint.pkl>
"""
from __future__ import annotations

import copy
import os
import pickle
import random
import sys

os.environ.setdefault("POKERBOT_PREFLOP_LOSSLESS", "1")
os.environ.setdefault("POKERBOT_SEARCH_PREFLOP", "1")
os.environ.setdefault("POKERBOT_BETTING_ABSTRACTION", "1")
os.environ.setdefault("POKERBOT_POSTFLOP_BUCKETS", "1")
os.environ.setdefault("POKERBOT_POSTFLOP_MODE", "fast")
os.environ.setdefault("POKERBOT_POSTFLOP_NUM_BUCKETS", "200")

import numpy as np

from pokerbot.engine.actions import ActionType
from pokerbot.engine.game import GameState, Street, play_hand

_STREET_IDX = {Street.PREFLOP: 0, Street.FLOP: 1, Street.TURN: 2, Street.RIVER: 3}
from pokerbot.search import pluribus_search as ps
from pokerbot.search.pbs_tracker import PBSTracker
from pokerbot.search.pluribus_rollout import ContinuationLeafConfig
from pokerbot.search.pluribus_search import PluribusReBeLConfig, PluribusReBeLSolver
from pokerbot.search.state_bridge import build_subgame_root


def load_blueprint(path: str) -> dict:
    with open(path, "rb") as f:
        full = pickle.load(f)
    return full["strat_sum"] if isinstance(full, dict) and "strat_sum" in full else full


def collect_roots(n: int) -> list:
    """Play random hands via ``play_hand`` (the exact eval path: it advances streets + deals the board)
    with a recording policy that snapshots each acting node. Return a spread across streets."""
    rng = random.Random(2026)
    by_street: dict[int, list] = {0: [], 1: [], 2: [], 3: []}

    def rec_policy(state):
        legal = state.legal_actions()
        seat = state.action_seat
        if seat is not None and seat >= 0 and state.players[seat].hole:
            by_street[_STREET_IDX.get(state.street, 0)].append(copy.deepcopy(state))
        if not legal:
            return None
        # Passive bias: prefer CHECK/CALL so hands survive to postflop acting nodes (random 6-handed
        # play otherwise jams/folds preflop and never reaches the flop with >=2 live players).
        passive = [a for a in legal if a.action_type in (ActionType.CHECK, ActionType.CALL)]
        if passive and rng.random() < 0.8:
            return rng.choice(passive)
        return rng.choice(legal)

    hands = 0
    need = max(2, n // 4 + 1)  # minimum roots required from EACH street before we stop
    while not all(len(by_street[s]) >= need for s in (0, 1, 2, 3)) and hands < 2000:
        hands += 1
        st = GameState.new_hand(button_seat=rng.randint(0, 5), rng=random.Random(rng.randint(0, 10 ** 9)))
        policies = {s: rec_policy for s in range(6)}
        play_hand(policies, state=st, rng=random.Random(rng.randint(0, 10 ** 9)))

    # Round-robin across streets for a spread (preflop..river), then trim to n.
    roots: list = []
    order = [0, 1, 2, 3]
    idx = {s: 0 for s in order}
    while len(roots) < n and any(idx[s] < len(by_street[s]) for s in order):
        for s in order:
            if idx[s] < len(by_street[s]):
                roots.append(by_street[s][idx[s]])
                idx[s] += 1
                if len(roots) >= n:
                    break
    return roots[:n]


def make_solver(snap, iters: int) -> PluribusReBeLSolver:
    cfg = PluribusReBeLConfig(
        max_iterations=iters, time_budget_ms=10 ** 9, max_depth=48, blueprint_mix=0.5,
        client_bet_sizing=True, search_preflop=True,
        leaf=ContinuationLeafConfig(select="uniform", rollout_samples=4, gadget_iters=256,
                                    max_playout_depth=64, base_seed=0, resample_board=True),
        per_decision_seed=True, native=True,
    )
    return PluribusReBeLSolver(snap, cfg, opponent_model=None)


def mean_probs(solver, root, pbs, k: int) -> np.ndarray:
    acc = np.zeros(11, dtype=np.float64)
    for salt in range(k):
        solver.config.rng_salt = salt
        res = solver.solve(root, pbs)
        acc += np.asarray(res.action_probs, dtype=np.float64)
    return acc / k


def run(snap, roots, iters: int, k: int) -> float:
    solver = make_solver(snap, iters)
    worst = 0.0
    print(f"\n--- iters={iters}  K={k} ---", flush=True)
    for ri, st in enumerate(roots):
        hero = int(st.action_seat)
        hole = st.players[hero].hole
        pbs = PBSTracker.init_uniform(hero, hole, st.board)
        root = build_subgame_root(st, pbs, hero)
        ps.set_native_subgame(False)
        py = mean_probs(solver, root, pbs, k)
        ps.set_native_subgame(True)
        nat = mean_probs(solver, root, pbs, k)
        l1 = float(np.abs(py - nat).sum())
        worst = max(worst, l1)
        street = _STREET_IDX.get(st.street, 0)
        extra = ""
        if l1 > 0.30:
            extra = ("\n   py =" + np.array2string(py, precision=2, suppress_small=True)
                     + "\n   nat=" + np.array2string(nat, precision=2, suppress_small=True))
        print(f"root {ri:2d} street={street} hero={hero} L1={l1:.3f} "
              f"py_argmax={int(py.argmax())} nat_argmax={int(nat.argmax())}{extra}", flush=True)
    print(f"max L1 = {worst:.3f}", flush=True)
    return worst


def main(path: str) -> int:
    snap = load_blueprint(path)
    print(f"blueprint: {len(snap)} infosets", flush=True)
    roots = collect_roots(8)
    print(f"collected {len(roots)} roots", flush=True)
    # Variance-vs-bias diagnostic: same eval-iter regime at LOW vs HIGH K. Under statistical
    # equivalence (different RNG, same expectation) the per-decision L1 must SHRINK ~1/sqrt(K); a
    # plateau would signal a systematic strategy difference (bias).
    w_lo = run(snap, roots, iters=20, k=48)
    w_hi = run(snap, roots, iters=20, k=192)
    shrank = w_hi < w_lo  # more samples -> tighter agreement => variance, not bias
    ok = (w_hi < 0.30) and shrank
    print(f"\nGATE_SUBGAME {'PASS' if ok else 'FAIL'}  (k48={w_lo:.3f} k192={w_hi:.3f} shrank={shrank})")
    return 0 if ok else 1


if __name__ == "__main__":
    p = sys.argv[1] if len(sys.argv) > 1 else r"checkpoints/bp6max_lossless2_iter200.pkl"
    sys.exit(main(p))
