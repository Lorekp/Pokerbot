#!/usr/bin/env python3
"""Compare the native vs Python continuation-gadget LEAF value at the post-bet chance node reached from
the AKs-facing-raise root, and the all-in terminal EV, to explain the jam-Q divergence.

Hypothesis refinement: native jams LESS because it values the BET-continuation lines HIGHER (its leaf
gadget / rollout returns a bigger continuation EV), so betting competes with jamming; whereas Python's
leaf gadget returns a lower continuation, leaving the (identical) all-in terminal EV as the clear max.

For a set of hero bet actions at the root, we:
  1. apply the hero bet, then have the opponent(s) act to close the street (blueprint-mixed sample),
  2. reach the chance node (flop deal) = the depth-limit LEAF,
  3. evaluate that leaf's continuation value with BOTH engines' leaf estimators.
We also record the ALL_IN terminal EV (identical scorer, engine-agnostic given inputs).

Run from the worktree.
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
    from pokerbot.engine.cards import Card
    from pokerbot.search import native_solver as _ns
    import pickle

    _RANKS = "23456789TJQKA"; _SUITS = "cdhs"

    def _card(idx):
        i = int(idx); return Card.from_str(_RANKS[i // 4] + _SUITS[i % 4])

    def is_AK(env, s):
        h = env.holes[s]
        if int(h[0]) < 0: return False
        r0, r1 = int(h[0]) // 4, int(h[1]) // 4
        return {r0, r1} == {12, 11} and int(h[0]) % 4 == int(h[1]) % 4

    occ = jax.numpy.array([True] * 6)
    env = seat = None
    for d in range(4000):
        e = to_numpy_state(init_hand(jax.random.PRNGKey(4242 * 100003 + d), occupancy=occ))
        g = 0
        while not is_terminal_np(e) and g < 40:
            g += 1
            if is_chance_node_np(e):
                e = sample_chance_np(e); continue
            s = current_player_np(e)
            if s < 0 or int(e.street) != 0: break
            m = legal_action_mask_np(e, s).astype(bool)
            if is_AK(e, s) and bool(m[FOLD]):
                env, seat = e, int(s); break
            a = (BET_BUCKET_START + 3) if m[BET_BUCKET_START + 3] else (CALL if m[CALL] else (CHECK if m[CHECK] else int(np.argmax(m))))
            e = apply_action_np(e, a)
        if env is not None: break

    snap = (lambda o: o["strat_sum"] if isinstance(o, dict) and "strat_sum" in o else o)(
        pickle.load(open(f"{MAIN}/checkpoints/serve_blueprint_6max.pkl", "rb")))
    hole = (_card(int(env.holes[seat][0])), _card(int(env.holes[seat][1])))
    board = [_card(int(c)) for c in np.asarray(env.board) if int(c) >= 0]

    cfg = PluribusReBeLConfig(leaf=ContinuationLeafConfig(select="uniform"), search_preflop=True,
                              max_iterations=1000, time_budget_ms=10 ** 9)
    cfg.native = False; cfg.repair_deck = True; cfg.two_sided = False; cfg.blueprint_mix = 1.0
    solver = PluribusReBeLSolver(snap, cfg, rng=np.random.default_rng(4242))
    solver._leaf_cont_weights = None
    solver._two_sided_active = False

    print(f"AKs facing-raise: seat={seat}", flush=True)

    # For a hero BET action, close the street with a blueprint-mixed opponent sample, reach the chance node,
    # then evaluate the leaf value with the Python leaf gadget AND the native leaf estimator.
    def reach_leaf_after_bet(bet_action, world):
        """Apply hero bet, then let opponents act (blueprint-mixed) until a chance node/terminal. Return env."""
        e = apply_action_np(world, int(bet_action))
        g = 0
        while not is_terminal_np(e) and not is_chance_node_np(e) and g < 20:
            g += 1
            s = current_player_np(e)
            if s < 0: break
            legal = solver._legal(e, s)
            key = solver._infoset_key(e, s)
            regr = solver._get_regrets(key)
            strat = solver._opponent_strategy(e, s, legal, regr, key)
            lp = strat[legal]; tot = lp.sum()
            if tot <= 1e-12:
                a = int(np.flatnonzero(legal)[0])
            else:
                a = int(solver.rng.choice(np.flatnonzero(legal), p=lp / tot))
            e = apply_action_np(e, a)
        return e

    # Native leaf value engine
    _ns.enable()
    from pokerbot.engine.jax_weva_encode import _POSTFLOP_NUM_BUCKETS
    from pokerbot.search.native_solver import native_uniform_leaf_value, native_rollout_available
    nra = native_rollout_available()
    _ns.disable()

    legal_root = solver._legal(env, seat)
    bet_actions = [a for a in np.flatnonzero(legal_root) if BET_BUCKET_START <= a < ALL_IN]
    solver.rng = np.random.default_rng(solver._public_seed(env, seat))

    out = {"seat": seat, "native_rollout_available": bool(nra), "bets": {}}
    nw = 120
    for ba in bet_actions:
        py_leaf = []; nat_leaf = []; reached_chance = 0
        for w in range(nw):
            sampled = PBSTracker.init_uniform(seat, hole, board).sample_opponent_holes(solver.rng)
            world = solver._set_world(env, sampled)
            e = reach_leaf_after_bet(ba, world)
            if is_terminal_np(e):
                v = float(np.asarray(e.rewards)[seat]) / 100.0
                py_leaf.append(v); nat_leaf.append(v); continue
            if not is_chance_node_np(e):
                continue
            reached_chance += 1
            # Python leaf gadget
            vpy = solver._leaf_value(e, seat)
            py_leaf.append(vpy)
            # Native leaf estimator (same env, uniform rollout)
            if nra:
                _ns.enable()
                seed = solver._public_seed(e, seat)
                vn = native_uniform_leaf_value(e, seat, solver.adapter, base_seed=seed,
                                               rollout_samples=cfg.leaf.rollout_samples,
                                               resample_board=cfg.leaf.resample_board,
                                               max_playout_depth=cfg.leaf.max_playout_depth,
                                               num_buckets=int(_POSTFLOP_NUM_BUCKETS))
                _ns.disable()
                if vn is not None:
                    nat_leaf.append(float(vn))
        pym = float(np.mean(py_leaf)) if py_leaf else float("nan")
        natm = float(np.mean(nat_leaf)) if nat_leaf else float("nan")
        out["bets"][int(ba)] = {"py_leaf_mean_bb": round(pym, 2), "nat_leaf_mean_bb": round(natm, 2),
                                "reached_chance": reached_chance, "n": nw}
        print(f"  BET a={int(ba)}: py_leaf={pym:+.2f}bb  native_leaf={natm:+.2f}bb  "
              f"(reached_chance {reached_chance}/{nw})", flush=True)

    # all-in terminal EV (engine-agnostic, from mechanism probe): recompute quickly
    def play_allin(world):
        e = world; g = 0
        while not is_terminal_np(e) and g < 40:
            g += 1
            if is_chance_node_np(e):
                e = sample_chance_np(e); continue
            s = current_player_np(e)
            if s < 0: break
            m = legal_action_mask_np(e, s).astype(bool)
            if s == seat and int(e.street) == 0 and m[ALL_IN]:
                e = apply_action_np(e, ALL_IN)
            elif m[CALL]:
                e = apply_action_np(e, CALL)
            elif m[CHECK]:
                e = apply_action_np(e, CHECK)
            else:
                e = apply_action_np(e, int(np.argmax(m)))
        return e
    from pokerbot.search.pluribus_rollout import _reshuffle_undealt
    aev = []
    for w in range(nw):
        sampled = PBSTracker.init_uniform(seat, hole, board).sample_opponent_holes(solver.rng)
        world = solver._set_world(env, sampled)
        e = play_allin(world)
        aev.append(float(np.asarray(e.rewards)[seat]) / 100.0)
    out["allin_terminal_ev_bb"] = round(float(np.mean(aev)), 2)
    print(f"\n  ALL_IN terminal EV (both engines identical scorer): {np.mean(aev):+.2f}bb", flush=True)
    print(f"  => native jams LESS iff its BET leaf values >= Python's (bets compete with the jam)", flush=True)

    json.dump(out, open(f"{MAIN}/artifacts/py_vs_rust_leaf_probe.json", "w"), indent=1)
    print(f"\nwrote {MAIN}/artifacts/py_vs_rust_leaf_probe.json", flush=True)


if __name__ == "__main__":
    main()
