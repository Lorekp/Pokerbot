#!/usr/bin/env python3
"""Directly compare the OPPONENT response distribution facing the hero's jam, native vs Python.

After the hero (AKs, seat) jams at the root, the NEXT live opponent must act. In one-sided CFR the
opponent samples ONE action from opponent_strategy = (1-mix)*regret_match + mix*blueprint over the
abstracted-legal set. With blueprint_mix=1.0 this is PURELY the blueprint policy at that facing-jam
infoset. If native and Python compute a DIFFERENT opponent continue-vs-fold facing the jam, the hero
best-responds to a different defense -> different jam Q -> different fixed point (H1/H3).

We measure, over many sampled opponent-hole worlds, the aggregate P(opponent CALLS/RAISES the jam) vs
P(FOLDS) using:
  - Python: solver._opponent_strategy at the post-jam node (blueprint-mixed, mix=1.0)
  - Native: the blueprint policy via the SAME adapter (blueprint_mix=1.0 => opponent_strategy == blueprint)
Both should be the SAME blueprint lookup. Any difference is the mechanism.

Also: does the abstracted-legal mask facing an all-in differ (FOLD/CALL only vs also re-jam)? And does the
opponent even reach the blueprint (fingerprint HIT) or fall to uniform?

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
        PluribusReBeLConfig, PluribusReBeLSolver,
    )
    from pokerbot.search.pluribus_rollout import ContinuationLeafConfig
    from pokerbot.search.pbs_tracker import PBSTracker
    from pokerbot.engine.cards import Card
    import pickle

    _RANKS = "23456789TJQKA"; _SUITS = "cdhs"
    def _card(idx):
        i = int(idx); return Card.from_str(_RANKS[i // 4] + _SUITS[i % 4])
    def is_AK(env, s):
        h = env.holes[s]
        if int(h[0]) < 0: return False
        return {int(h[0]) // 4, int(h[1]) // 4} == {12, 11} and int(h[0]) % 4 == int(h[1]) % 4

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
    solver._leaf_cont_weights = None; solver._two_sided_active = False

    print(f"AKs facing-raise: seat={seat}", flush=True)

    # Hero jams; find the next live opponent to act; measure their blueprint-mixed response.
    jam_env = apply_action_np(env, ALL_IN)
    # opponent facing the jam
    opp = current_player_np(jam_env)
    print(f"  after hero ALL_IN: next actor = seat {opp}", flush=True)
    if opp is None or opp < 0:
        print("  jam ended action (all others folded already) -> terminal", flush=True)

    rng = np.random.default_rng(solver._public_seed(env, seat))
    nw = 2000
    # aggregate opponent action distribution over sampled opponent-hole worlds
    acc = np.zeros(11)
    fold_mass = 0.0; continue_mass = 0.0; bp_hits = 0
    legal_seen = None
    for w in range(nw):
        sampled = PBSTracker.init_uniform(seat, hole, board).sample_opponent_holes(rng)
        world = solver._set_world(env, sampled)
        e = apply_action_np(world, ALL_IN)
        if is_terminal_np(e) or is_chance_node_np(e):
            continue
        o = current_player_np(e)
        if o is None or o < 0:
            continue
        legal = solver._legal(e, o)
        legal_seen = legal.copy()
        key = solver._infoset_key(e, o)
        regr = solver._get_regrets(key)
        strat = solver._opponent_strategy(e, o, legal, regr, key)
        # check blueprint hit
        bp = solver._cached_blueprint(e, o, key)
        if bp.sum() > 1e-12:
            bp_hits += 1
        acc += strat
        fold_mass += float(strat[FOLD])
        continue_mass += float(strat[CALL] + strat[BET_BUCKET_START:].sum())
    acc /= max((acc.sum()), 1e-12)
    out = {
        "seat": seat, "opp_seat": int(opp) if (opp is not None and opp >= 0) else -1,
        "opp_legal_facing_jam": [int(x) for x in np.flatnonzero(legal_seen)] if legal_seen is not None else [],
        "opp_mean_fold": round(fold_mass / nw, 4),
        "opp_mean_continue": round(continue_mass / nw, 4),
        "bp_hit_rate": round(bp_hits / nw, 4),
        "opp_action_dist": {int(a): round(float(acc[a]), 4) for a in range(11) if acc[a] > 1e-4},
    }
    print(f"  opponent facing jam: mean FOLD={out['opp_mean_fold']:.3f}  "
          f"CONTINUE={out['opp_mean_continue']:.3f}  bp_hit={out['bp_hit_rate']:.3f}", flush=True)
    print(f"  opp legal actions facing jam: {out['opp_legal_facing_jam']}", flush=True)
    print(f"  opp action dist: {out['opp_action_dist']}", flush=True)
    print(f"\n  NOTE: with blueprint_mix=1.0 both engines' opponent policy IS this blueprint lookup, so if"
          f"\n  native and Python agree here, the opponent DEFENSE is identical -> divergence is NOT the"
          f"\n  opponent range but the hero-side CFR trajectory/averaging (H3).", flush=True)

    json.dump(out, open(f"{MAIN}/artifacts/py_vs_rust_opp_probe.json", "w"), indent=1)
    print(f"\nwrote {MAIN}/artifacts/py_vs_rust_opp_probe.json", flush=True)


if __name__ == "__main__":
    main()
