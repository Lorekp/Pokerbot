#!/usr/bin/env python3
"""GATE (Stage-1 river_overbet, offtree_resolve_design.md §8.5d): native/Python statistical
equivalence with the knob ON at RIVER first-in / check-to roots.

Reuses the EXISTING statistical-equivalence harness (tools/battery_native/crosscheck_subgame.py:
same solver config, same mean-over-rng_salt L1 comparison, same shrink-with-K variance-vs-bias
diagnostic) but the roots are river check-to-hero nodes (engine-generated via play_hand — game-flow
states only) and every solve runs with config.river_overbet=True, so BOTH paths solve the WIDENED
root menu. Band gate, not byte: max L1 at k192 < 0.30 AND L1 shrinks from k48 to k192. Additionally
asserts both paths put nonzero mean mass on the un-masked bucket (index BET_BUCKET_START+6) at
every root — the knob must be live inside the harness, not just in the unit probe.

Usage: python scripts/crosscheck_river_overbet.py [blueprint.pkl]
"""
from __future__ import annotations

import copy
import os
import random
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

# The EXISTING harness's env regime (fast made-hand keying — the certified native fast path); its
# module-level setdefaults run at import, before any other pokerbot import.
from tools.battery_native.crosscheck_subgame import (  # noqa: E402
    load_blueprint, make_solver, mean_probs,
)

import numpy as np  # noqa: E402

from pokerbot.engine.actions import ActionType  # noqa: E402
from pokerbot.engine.game import GameState, Street, play_hand  # noqa: E402
from pokerbot.engine.jax_engine import BET_BUCKET_START  # noqa: E402
from pokerbot.engine.np_engine import legal_action_mask_np, to_numpy_state  # noqa: E402
from pokerbot.search import pluribus_search as ps  # noqa: E402
from pokerbot.search.pbs_tracker import PBSTracker  # noqa: E402
from pokerbot.search.state_bridge import build_subgame_root  # noqa: E402

OVERBET_IDX = BET_BUCKET_START + 6


def collect_river_roots(n: int) -> list:
    """River check-to-hero (to_call==0) nodes with a fully-legal 2x-pot bucket, from real deals."""
    rng = random.Random(20260703)
    captured: list = []

    def rec_policy(state):
        legal = state.legal_actions()
        seat = state.action_seat
        if (seat is not None and seat >= 0 and state.street == Street.RIVER
                and state.players[seat].hole):
            captured.append((copy.deepcopy(state), int(seat)))
        if not legal:
            return None
        passive = [a for a in legal if a.action_type in (ActionType.CHECK, ActionType.CALL)]
        if passive and rng.random() < 0.85:
            return rng.choice(passive)
        return rng.choice(legal)

    roots = []
    for _ in range(800):
        if len(roots) >= n:
            break
        captured.clear()
        st0 = GameState.new_hand(button_seat=rng.randint(0, 5),
                                 rng=random.Random(rng.randint(0, 10 ** 9)))
        play_hand({s: rec_policy for s in range(6)},
                  state=st0, rng=random.Random(rng.randint(0, 10 ** 9)))
        for st, hero in captured:
            if len(roots) >= n:
                break
            pbs = PBSTracker.init_uniform(hero, st.players[hero].hole, st.board)
            root = build_subgame_root(st, pbs, hero)
            env = to_numpy_state(root.env)
            to_call = int(env.current_bet) - int(np.asarray(env.committed_street)[hero])
            if to_call > 0:
                continue
            if not bool(legal_action_mask_np(env, hero).astype(bool)[OVERBET_IDX]):
                continue
            roots.append(st)
    return roots


def run(snap, roots, iters: int, k: int):
    solver = make_solver(snap, iters)
    solver.config.river_overbet = True  # the knob under test — ON for BOTH paths
    worst = 0.0
    min_ob_py, min_ob_nat = 1.0, 1.0
    print(f"\n--- river_overbet=ON  iters={iters}  K={k} ---", flush=True)
    for ri, st in enumerate(roots):
        hero = int(st.action_seat)
        pbs = PBSTracker.init_uniform(hero, st.players[hero].hole, st.board)
        root = build_subgame_root(st, pbs, hero)
        ps.set_native_subgame(False)
        py = mean_probs(solver, root, pbs, k)
        ps.set_native_subgame(True)
        nat = mean_probs(solver, root, pbs, k)
        l1 = float(np.abs(py - nat).sum())
        worst = max(worst, l1)
        min_ob_py = min(min_ob_py, float(py[OVERBET_IDX]))
        min_ob_nat = min(min_ob_nat, float(nat[OVERBET_IDX]))
        print(f"root {ri:2d} hero={hero} L1={l1:.3f} "
              f"ob_py={py[OVERBET_IDX]:.4f} ob_nat={nat[OVERBET_IDX]:.4f} "
              f"py_argmax={int(py.argmax())} nat_argmax={int(nat.argmax())}", flush=True)
    return worst, min_ob_py, min_ob_nat


def main(path: str) -> int:
    snap = load_blueprint(path)
    print(f"blueprint: {len(snap)} infosets", flush=True)
    roots = collect_river_roots(5)
    print(f"collected {len(roots)} river check-to roots", flush=True)
    if len(roots) < 5:
        print("GATE_RIVER_OVERBET_XCHECK FAIL (fewer than 5 river roots)")
        return 1
    w_lo, ob_py_lo, ob_nat_lo = run(snap, roots, iters=20, k=48)
    w_hi, ob_py_hi, ob_nat_hi = run(snap, roots, iters=20, k=192)
    shrank = w_hi < w_lo
    knob_live = min(ob_py_hi, ob_nat_hi) > 0.0
    ok = (w_hi < 0.30) and shrank and knob_live
    print(f"\nGATE_RIVER_OVERBET_XCHECK {'PASS' if ok else 'FAIL'}  "
          f"(k48={w_lo:.3f} k192={w_hi:.3f} shrank={shrank} "
          f"min_overbet_mass py={ob_py_hi:.4f} nat={ob_nat_hi:.4f})")
    return 0 if ok else 1


if __name__ == "__main__":
    p = sys.argv[1] if len(sys.argv) > 1 else os.path.join(REPO, "checkpoints", "serve_blueprint_6max.pkl")
    sys.exit(main(p))
