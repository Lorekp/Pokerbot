#!/usr/bin/env python3
"""Pin down the one-sided CFR-dynamics difference: compare, at the AKs node, native vs Python for
   (a) linear_averaging ON vs OFF, (b) served strategy_sum-average vs FINAL regret-match policy.

If native's LOW jam comes from its strategy_sum but its FINAL regret-match would ALSO jam high, the
difference is in AVERAGING. If native's regrets themselves never favor the jam, it's the regret update /
traversal. We can read Python's internals directly; for native we compare served (strat_sum-average) at
linear ON/OFF and infer.

Also: run Python with a SINGLE fixed opponent-hole world per iteration but MANY iterations to see how the
served jam evolves, and print the per-action regret + strategy_sum at the root for Python (the ground
truth we can inspect).

Run from the worktree.
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
    from pokerbot.search.turbo_rebel_search import _regret_match
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
    print(f"AKs facing-raise: seat={seat}", flush=True)

    K = 1000
    out = {"seat": seat, "K": K}

    # ---- Native served at linear ON vs OFF ----
    def native_served(linear):
        cfg = PluribusReBeLConfig(leaf=ContinuationLeafConfig(select="uniform"), search_preflop=True,
                                  max_iterations=K, time_budget_ms=10 ** 9, linear_averaging=linear)
        cfg.native = True; cfg.repair_deck = True; cfg.two_sided = False; cfg.blueprint_mix = 1.0
        s = PluribusReBeLSolver(snap, cfg, rng=np.random.default_rng(4242))
        set_native_subgame(True)
        s.rng = np.random.default_rng(4242)
        pbs = PBSTracker.init_uniform(seat, hole, board)
        legal = legal_action_mask_np(env, seat).astype(bool)
        root = SubgameRoot(env=env, hero_seat=seat, pbs_vector=pbs.get_public_belief_state(), legal_mask=legal)
        r = s.solve(root, pbs)
        return np.asarray(r.action_probs, dtype=np.float64)

    for lin in (True, False):
        p = native_served(lin)
        out[f"native_linear_{lin}"] = {"allin": round(float(p[ALL_IN]), 4),
                                        "bets": round(float(p[BET_BUCKET_START:ALL_IN].sum()), 4)}
        print(f"  NATIVE linear={lin}: ALL_IN={p[ALL_IN]:.3f} BETS={p[BET_BUCKET_START:ALL_IN].sum():.3f}", flush=True)

    # ---- Python: run the loop manually, inspect ROOT regrets + strategy_sum + served-average vs final-RM ----
    def python_internals(linear):
        cfg = PluribusReBeLConfig(leaf=ContinuationLeafConfig(select="uniform"), search_preflop=True,
                                  max_iterations=K, time_budget_ms=10 ** 9, linear_averaging=linear)
        cfg.native = False; cfg.repair_deck = True; cfg.two_sided = False; cfg.blueprint_mix = 1.0
        s = PluribusReBeLSolver(snap, cfg, rng=np.random.default_rng(4242))
        set_native_subgame(False)
        s._leaf_cont_weights = None; s._two_sided_active = False; s._traversers = [seat]
        s._regrets.clear(); s._strategy_sum.clear(); s._bp_cache.clear(); s._leaf_cache.clear()
        s.rng = np.random.default_rng(s._public_seed(env, seat))
        rk = s._infoset_key(env, seat)
        for it in range(K):
            sampled = PBSTracker.init_uniform(seat, hole, board).sample_opponent_holes(s.rng)
            world = s._set_world(env, sampled)
            s._cfv(world, seat, 0, it)
        legal = s._legal(env, seat)
        regr = s._regrets.get(rk, np.zeros(11))
        ss = s._strategy_sum.get(rk, np.zeros(11))
        served = (ss * legal); served = served / max(served.sum(), 1e-12)
        final_rm = _regret_match(regr, legal.astype(np.float64))
        return served, final_rm, regr, legal

    for lin in (True, False):
        served, final_rm, regr, legal = python_internals(lin)
        out[f"python_linear_{lin}"] = {
            "served_allin": round(float(served[ALL_IN]), 4),
            "served_bets": round(float(served[BET_BUCKET_START:ALL_IN].sum()), 4),
            "final_rm_allin": round(float(final_rm[ALL_IN]), 4),
            "final_rm_bets": round(float(final_rm[BET_BUCKET_START:ALL_IN].sum()), 4),
            "root_regret": {int(a): round(float(regr[a]), 2) for a in np.flatnonzero(legal)},
        }
        print(f"  PYTHON linear={lin}: served ALL_IN={served[ALL_IN]:.3f}  "
              f"final-regret-match ALL_IN={final_rm[ALL_IN]:.3f}  "
              f"root_regret[ALL_IN]={regr[ALL_IN]:.1f} max_bet_regret={regr[BET_BUCKET_START:ALL_IN].max():.1f}", flush=True)

    json.dump(out, open(f"{MAIN}/artifacts/py_vs_rust_cfr_dynamics.json", "w"), indent=1)
    print(f"\nwrote {MAIN}/artifacts/py_vs_rust_cfr_dynamics.json", flush=True)


if __name__ == "__main__":
    main()
