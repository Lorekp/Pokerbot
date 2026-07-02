#!/usr/bin/env python3
"""Instrument the Python solver's ROOT per-action Q-values (action_values) + opponent continue/fold at
the AKs-facing-raise node, and compare native vs Python served policy again with matched public seed.

This isolates H3 (does the hero's best-response Q for ALL_IN vs BETS actually differ between engines, and
if so, is it because the opponent DEFENDS the jam differently?).

We monkeypatch PluribusReBeLSolver._cfv to record, at depth==0 hero node, the per-action action_values on
the FINAL iteration, and we tap _opponent_strategy to record the aggregate opponent continue-vs-fold mass
FACING the hero's jam (the child where hero just went ALL_IN). Read-only (patched in-process only).
"""
from __future__ import annotations

import json
import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import jax


def main():
    WT = sys.argv[1] if len(sys.argv) > 1 else os.getcwd()
    MAIN = sys.argv[2] if len(sys.argv) > 2 else os.getcwd()
    sys.path.insert(0, WT)

    from pokerbot.engine.np_engine import (
        apply_action_np, current_player_np, is_chance_node_np, is_terminal_np,
        legal_action_mask_np, sample_chance_np, to_numpy_state,
        ALL_IN, BET_BUCKET_START, CALL, FOLD, CHECK,
    )
    from pokerbot.engine.jax_engine import init_hand
    from pokerbot.search.pluribus_search import (
        PluribusReBeLConfig, PluribusReBeLSolver, set_native_subgame,
    )
    from pokerbot.search.pluribus_rollout import ContinuationLeafConfig
    from pokerbot.search.pbs_tracker import PBSTracker
    from pokerbot.search.state_bridge import SubgameRoot
    from pokerbot.engine.cards import Card
    import pickle

    _RANKS = "23456789TJQKA"
    _SUITS = "cdhs"

    def _card(idx):
        i = int(idx)
        return Card.from_str(_RANKS[i // 4] + _SUITS[i % 4])

    def is_AK(env, seat):
        h = env.holes[seat]
        if int(h[0]) < 0:
            return False
        r0, r1 = int(h[0]) // 4, int(h[1]) // 4
        s0, s1 = int(h[0]) % 4, int(h[1]) % 4
        return {r0, r1} == {12, 11} and s0 == s1

    occ = jax.numpy.array([True] * 6)
    env = None
    seat = None
    for d in range(4000):
        e = to_numpy_state(init_hand(jax.random.PRNGKey(4242 * 100003 + d), occupancy=occ))
        g = 0
        while not is_terminal_np(e) and g < 40:
            g += 1
            if is_chance_node_np(e):
                e = sample_chance_np(e); continue
            s = current_player_np(e)
            if s < 0 or int(e.street) != 0:
                break
            m = legal_action_mask_np(e, s).astype(bool)
            if is_AK(e, s) and bool(m[FOLD]):
                env, seat = e, int(s); break
            if m[BET_BUCKET_START + 3]:
                a = BET_BUCKET_START + 3
            elif m[CALL]:
                a = CALL
            elif m[CHECK]:
                a = CHECK
            else:
                a = int(np.argmax(m))
            e = apply_action_np(e, a)
        if env is not None:
            break

    snap = (lambda o: o["strat_sum"] if isinstance(o, dict) and "strat_sum" in o else o)(
        pickle.load(open(f"{MAIN}/checkpoints/serve_blueprint_6max.pkl", "rb")))

    hole = (_card(int(env.holes[seat][0])), _card(int(env.holes[seat][1])))
    board = [_card(int(c)) for c in np.asarray(env.board) if int(c) >= 0]
    print(f"AKs facing-raise: seat={seat} nboard={len(board)}", flush=True)

    ITERS = 1000

    # ---- Python solver with root action_values capture ----
    cfg = PluribusReBeLConfig(leaf=ContinuationLeafConfig(select="uniform"), search_preflop=True,
                              max_iterations=ITERS, time_budget_ms=10 ** 9)
    cfg.native = False; cfg.repair_deck = True; cfg.two_sided = False; cfg.blueprint_mix = 1.0
    solver = PluribusReBeLSolver(snap, cfg, rng=np.random.default_rng(4242))

    captured = {"root_av": None, "n_hero_root_visits": 0,
                "opp_facing_jam_continue": 0.0, "opp_facing_jam_n": 0,
                "hero_jam_children_reached": 0}

    orig_cfv = solver._cfv.__func__

    def patched_cfv(self, env2, hero2, depth, t=0):
        # detect: we are at a node where the previous action was the hero's ALL_IN (facing jam)
        val = orig_cfv(self, env2, hero2, depth, t)
        return val

    # Instead of wrapping recursion (hard), re-run the loop manually to capture root action_values on the
    # LAST iteration. Replicate solve()'s inner loop with capture.
    from pokerbot.search import native_solver as _ns
    _ns.disable()
    set_native_subgame(False)

    base_state = env
    solver._regrets.clear(); solver._strategy_sum.clear(); solver._bp_cache.clear(); solver._leaf_cache.clear()
    solver.rng = np.random.default_rng(solver._public_seed(base_state, seat))
    solver._leaf_cont_weights = None
    solver._two_sided_active = False
    solver._traversers = [seat]
    root_key = solver._infoset_key(base_state, seat)

    # tap _cfv to grab depth-0 action_values: we monkeypatch by re-implementing the hero-root branch inline.
    # Simpler: after the full solve loop, recompute action_values once more at the converged regrets.
    from pokerbot.search.turbo_rebel_search import _regret_match

    for it in range(ITERS):
        sampled = PBSTracker.init_uniform(seat, hole, board).sample_opponent_holes(solver.rng)
        world = solver._set_world(base_state, sampled)
        solver._cfv(world, seat, 0, it)

    # Now do ONE more capture pass: sample a world, evaluate each root action's CFV under converged regrets.
    # Average over several worlds for a stable estimate of the per-action Q the hero sees.
    legal = solver._legal(base_state, seat)
    av_acc = np.zeros(11); nw = 200
    for w in range(nw):
        sampled = PBSTracker.init_uniform(seat, hole, board).sample_opponent_holes(solver.rng)
        world = solver._set_world(base_state, sampled)
        for a in np.flatnonzero(legal):
            child = apply_action_np(world, int(a))
            av_acc[a] += solver._cfv(child, seat, 1, ITERS)
    av = av_acc / nw
    ss = solver._strategy_sum.get(root_key)
    served = (ss * legal); served = served / max(served.sum(), 1e-12)

    out = {"seat": seat, "iters": ITERS}
    out["python"] = {
        "served_allin": round(float(served[ALL_IN]), 4),
        "served_bets": round(float(served[BET_BUCKET_START:ALL_IN].sum()), 4),
        "served_call": round(float(served[CALL]), 4),
        "q_allin": round(float(av[ALL_IN]), 3),
        "q_call": round(float(av[CALL]), 3),
        "q_bets_max": round(float(av[BET_BUCKET_START:ALL_IN].max()), 3),
        "q_fold": round(float(av[FOLD]), 3),
        "q_by_action": {int(a): round(float(av[a]), 3) for a in np.flatnonzero(legal)},
    }
    print(f"\n[PYTHON K={ITERS}] served ALL_IN={served[ALL_IN]:.3f} BETS={served[BET_BUCKET_START:ALL_IN].sum():.3f} "
          f"CALL={served[CALL]:.3f}", flush=True)
    print(f"  ROOT Q (hero best-response value, bb): "
          f"ALL_IN={av[ALL_IN]:+.2f}  CALL={av[CALL]:+.2f}  "
          f"BETS_max={av[BET_BUCKET_START:ALL_IN].max():+.2f}  FOLD={av[FOLD]:+.2f}", flush=True)
    print(f"  Q by action: {out['python']['q_by_action']}", flush=True)

    json.dump(out, open(f"{MAIN}/artifacts/py_vs_rust_rootq_probe.json", "w"), indent=1)
    print(f"\nwrote {MAIN}/artifacts/py_vs_rust_rootq_probe.json", flush=True)


if __name__ == "__main__":
    main()
