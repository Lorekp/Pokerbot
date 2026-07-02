#!/usr/bin/env python3
"""FINAL: quantify the served-policy VARIANCE across solver seeds for Python-shared-stream (production),
Python-separate-stream (coupling broken), and native, at the AKs node. Native's win = LOWER-VARIANCE,
less-over-aggressive one-sided CFR; Python's shared-stream RNG coupling inflates variance + mean jam.
"""
from __future__ import annotations
import json, os, sys
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import numpy as np, jax


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

    _RANKS = "23456789TJQKA"; _SUITS = "cdhs"
    def _card(i): return Card.from_str(_RANKS[int(i) // 4] + _SUITS[int(i) % 4])
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
    K = 400
    SEEDS = list(range(1, 25))

    def py_run(seed, shared):
        cfg = PluribusReBeLConfig(leaf=ContinuationLeafConfig(select="uniform"), search_preflop=True,
                                  max_iterations=K, time_budget_ms=10 ** 9)
        cfg.native = False; cfg.repair_deck = True; cfg.two_sided = False; cfg.blueprint_mix = 1.0
        s = PluribusReBeLSolver(snap, cfg, rng=np.random.default_rng(4242))
        set_native_subgame(False)
        s._leaf_cont_weights = None; s._two_sided_active = False; s._traversers = [seat]
        s._regrets.clear(); s._strategy_sum.clear(); s._bp_cache.clear(); s._leaf_cache.clear()
        s.rng = np.random.default_rng(seed)
        hole_rng = s.rng if shared else np.random.default_rng(seed ^ 0x9E3779B9)
        rk = s._infoset_key(env, seat)
        for it in range(K):
            sampled = PBSTracker.init_uniform(seat, hole, board).sample_opponent_holes(hole_rng)
            world = s._set_world(env, sampled)
            s._cfv(world, seat, 0, it)
        legal = s._legal(env, seat)
        ss = s._strategy_sum.get(rk, np.zeros(11))
        served = (ss * legal); served = served / max(served.sum(), 1e-12)
        return float(served[ALL_IN]), float(served[BET_BUCKET_START:].sum())

    def nat_run(seed):
        cfg = PluribusReBeLConfig(leaf=ContinuationLeafConfig(select="uniform"), search_preflop=True,
                                  max_iterations=K, time_budget_ms=10 ** 9)
        cfg.native = True; cfg.repair_deck = True; cfg.two_sided = False; cfg.blueprint_mix = 1.0
        cfg.rng_salt = seed
        s = PluribusReBeLSolver(snap, cfg, rng=np.random.default_rng(seed))
        set_native_subgame(True)
        s.rng = np.random.default_rng(seed)
        pbs = PBSTracker.init_uniform(seat, hole, board)
        legal = legal_action_mask_np(env, seat).astype(bool)
        root = SubgameRoot(env=env, hero_seat=seat, pbs_vector=pbs.get_public_belief_state(), legal_mask=legal)
        r = s.solve(root, pbs)
        p = np.asarray(r.action_probs)
        return float(p[ALL_IN]), float(p[BET_BUCKET_START:].sum())

    pysh = np.array([py_run(sd, True) for sd in SEEDS])
    pysep = np.array([py_run(sd, False) for sd in SEEDS])
    nat = np.array([nat_run(sd) for sd in SEEDS])
    def summ(a):
        return {"allin_mean": round(float(a[:, 0].mean()), 3), "allin_std": round(float(a[:, 0].std()), 3),
                "agg_mean": round(float(a[:, 1].mean()), 3), "agg_std": round(float(a[:, 1].std()), 3)}
    out = {"seat": seat, "K": K, "n_seeds": len(SEEDS),
           "python_shared": summ(pysh), "python_separate": summ(pysep), "native": summ(nat)}
    print(f"AKs node, {len(SEEDS)} seeds, K={K}:", flush=True)
    for k in ("python_shared", "python_separate", "native"):
        v = out[k]
        print(f"  {k:<16} ALL_IN {v['allin_mean']:.3f} ±{v['allin_std']:.3f}   "
              f"AGG {v['agg_mean']:.3f} ±{v['agg_std']:.3f}", flush=True)
    json.dump(out, open(f"{MAIN}/artifacts/py_vs_rust_variance_final.json", "w"), indent=1)
    print(f"\nwrote {MAIN}/artifacts/py_vs_rust_variance_final.json", flush=True)


if __name__ == "__main__":
    main()
