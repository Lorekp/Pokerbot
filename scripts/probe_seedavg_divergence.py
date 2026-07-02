#!/usr/bin/env python3
"""Is native's lower aggression a SYSTEMATIC convergence difference or per-seed BASIN noise?

For a handful of preflop nodes, run BOTH engines at MANY solver seeds each (rng_salt varied) and compare
the SEED-AVERAGED served aggression (bets+jam) and ALL_IN. If, averaged over seeds, native still aggresses
less, it's a real bias; if the seed-averages converge, the per-node divergence is high-variance basin noise
and native only wins at the H2H's particular (production) seeds.

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
    from pokerbot.engine.cards import Card
    import pickle

    _RANKS = "23456789TJQKA"; _SUITS = "cdhs"
    def _card(i): return Card.from_str(_RANKS[int(i) // 4] + _SUITS[int(i) % 4])

    snap = (lambda o: o["strat_sum"] if isinstance(o, dict) and "strat_sum" in o else o)(
        pickle.load(open(f"{MAIN}/checkpoints/serve_blueprint_6max.pkl", "rb")))
    occ = jax.numpy.array([True] * 6)
    K = 400
    SEEDS = [1, 2, 3, 4, 5, 6, 7, 8]

    def make(native):
        cfg = PluribusReBeLConfig(leaf=ContinuationLeafConfig(select="uniform"), search_preflop=True,
                                  max_iterations=K, time_budget_ms=10 ** 9)
        cfg.native = native; cfg.repair_deck = True; cfg.two_sided = False; cfg.blueprint_mix = 1.0
        return PluribusReBeLSolver(snap, cfg, rng=np.random.default_rng(4242))
    s_py, s_rust = make(False), make(True)

    def served(solver, native, env, seat, salt):
        set_native_subgame(bool(native))
        solver.config.rng_salt = salt
        solver.rng = np.random.default_rng(salt)
        hole = (_card(int(env.holes[seat][0])), _card(int(env.holes[seat][1])))
        board = [_card(int(c)) for c in np.asarray(env.board) if int(c) >= 0]
        pbs = PBSTracker.init_uniform(seat, hole, board)
        legal = legal_action_mask_np(env, seat).astype(bool)
        root = SubgameRoot(env=env, hero_seat=seat, pbs_vector=pbs.get_public_belief_state(), legal_mask=legal)
        r = solver.solve(root, pbs)
        return np.asarray(r.action_probs, dtype=np.float64)

    # gather nodes
    nodes = []
    d = 0
    while len(nodes) < 12 and d < 1500:
        d += 1
        e = to_numpy_state(init_hand(jax.random.PRNGKey(777 * 100003 + d), occupancy=occ))
        g = 0; node = None
        while not is_terminal_np(e) and g < 40:
            g += 1
            if is_chance_node_np(e):
                e = sample_chance_np(e); continue
            s = current_player_np(e)
            if s < 0 or int(e.street) != 0: break
            m = legal_action_mask_np(e, s).astype(bool)
            if int(m.sum()) > 1:
                node = (e, int(s)); break
            e = apply_action_np(e, int(np.argmax(m)))
        if node is not None:
            nodes.append(node)

    rows = []
    for (env, seat) in nodes:
        agp = []; agr = []; aip = []; air = []
        for sd in SEEDS:
            p_py = served(s_py, False, env, seat, sd)
            p_rust = served(s_rust, True, env, seat, sd)
            agp.append(float(p_py[BET_BUCKET_START:].sum())); agr.append(float(p_rust[BET_BUCKET_START:].sum()))
            aip.append(float(p_py[ALL_IN])); air.append(float(p_rust[ALL_IN]))
        rows.append({
            "agg_py_seedavg": round(float(np.mean(agp)), 3), "agg_rust_seedavg": round(float(np.mean(agr)), 3),
            "agg_py_std": round(float(np.std(agp)), 3), "agg_rust_std": round(float(np.std(agr)), 3),
            "allin_py_seedavg": round(float(np.mean(aip)), 3), "allin_rust_seedavg": round(float(np.mean(air)), 3),
        })
        print(f"  node seat{seat}: AGG py={rows[-1]['agg_py_seedavg']:.3f}(±{rows[-1]['agg_py_std']:.2f}) "
              f"rust={rows[-1]['agg_rust_seedavg']:.3f}(±{rows[-1]['agg_rust_std']:.2f})  "
              f"ALLIN py={rows[-1]['allin_py_seedavg']:.3f} rust={rows[-1]['allin_rust_seedavg']:.3f}", flush=True)

    agp = np.array([r["agg_py_seedavg"] for r in rows]); agr = np.array([r["agg_rust_seedavg"] for r in rows])
    aip = np.array([r["allin_py_seedavg"] for r in rows]); air = np.array([r["allin_rust_seedavg"] for r in rows])
    out = {
        "n_nodes": len(rows), "K": K, "n_seeds": len(SEEDS),
        "mean_agg_py_seedavg": round(float(agp.mean()), 4), "mean_agg_rust_seedavg": round(float(agr.mean()), 4),
        "mean_allin_py_seedavg": round(float(aip.mean()), 4), "mean_allin_rust_seedavg": round(float(air.mean()), 4),
        "frac_nodes_rust_lower_agg_seedavg": round(float((agr < agp - 0.02).mean()), 3),
    }
    print(f"\n  SEED-AVERAGED over {len(SEEDS)} seeds, {len(rows)} nodes:", flush=True)
    print(f"    mean AGG: py={out['mean_agg_py_seedavg']:.3f}  rust={out['mean_agg_rust_seedavg']:.3f}", flush=True)
    print(f"    mean ALL_IN: py={out['mean_allin_py_seedavg']:.3f}  rust={out['mean_allin_rust_seedavg']:.3f}", flush=True)
    print(f"    frac nodes rust still LESS aggressive (seed-avg): {out['frac_nodes_rust_lower_agg_seedavg']:.2f}", flush=True)
    json.dump({"summary": out, "rows": rows}, open(f"{MAIN}/artifacts/py_vs_rust_seedavg.json", "w"), indent=1)
    print(f"\nwrote {MAIN}/artifacts/py_vs_rust_seedavg.json", flush=True)


if __name__ == "__main__":
    main()
