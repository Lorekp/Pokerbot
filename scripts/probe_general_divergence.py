#!/usr/bin/env python3
"""Is the native-vs-Python jam divergence AKs-specific or GENERAL? And is the root legal mask identical?

Scans many random preflop hero decisions (any hole), solves each with native and Python (repair=ON,
two=OFF, K=400), and records the served ALL_IN + total-aggression difference. Reports the aggregate:
does native systematically jam/aggress LESS across the board? Also asserts the root abstracted-legal mask
is identical between engines (rules out an abstraction-mask mismatch).

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

    def make(native):
        cfg = PluribusReBeLConfig(leaf=ContinuationLeafConfig(select="uniform"), search_preflop=True,
                                  max_iterations=K, time_budget_ms=10 ** 9)
        cfg.native = native; cfg.repair_deck = True; cfg.two_sided = False; cfg.blueprint_mix = 1.0
        return PluribusReBeLSolver(snap, cfg, rng=np.random.default_rng(4242))

    s_py, s_rust = make(False), make(True)

    def served(solver, native, env, seat):
        set_native_subgame(bool(native))
        solver.rng = np.random.default_rng(4242)
        hole = (_card(int(env.holes[seat][0])), _card(int(env.holes[seat][1])))
        board = [_card(int(c)) for c in np.asarray(env.board) if int(c) >= 0]
        pbs = PBSTracker.init_uniform(seat, hole, board)
        legal = legal_action_mask_np(env, seat).astype(bool)
        root = SubgameRoot(env=env, hero_seat=seat, pbs_vector=pbs.get_public_belief_state(), legal_mask=legal)
        r = solver.solve(root, pbs)
        return np.asarray(r.action_probs, dtype=np.float64)

    # collect preflop nodes: first hero decision of each hand (varied holes/positions)
    rows = []
    mask_mismatch = 0
    n = 0
    d = 0
    while n < 60 and d < 2000:
        d += 1
        e = to_numpy_state(init_hand(jax.random.PRNGKey(999 * 100003 + d), occupancy=occ))
        g = 0
        node = None
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
        if node is None:
            continue
        env, seat = node
        # abstraction mask identity
        lp = s_py._legal(env, seat); lr = s_rust._legal(env, seat)
        if not np.array_equal(lp, lr):
            mask_mismatch += 1
        p_py = served(s_py, False, env, seat)
        p_rust = served(s_rust, True, env, seat)
        agg_py = float(p_py[BET_BUCKET_START:].sum())   # bets + all-in
        agg_rust = float(p_rust[BET_BUCKET_START:].sum())
        rows.append({
            "allin_py": round(float(p_py[ALL_IN]), 3), "allin_rust": round(float(p_rust[ALL_IN]), 3),
            "agg_py": round(agg_py, 3), "agg_rust": round(agg_rust, 3),
            "facing": bool(legal_action_mask_np(env, seat).astype(bool)[FOLD]),
        })
        n += 1

    aip = np.array([r["allin_py"] for r in rows]); air = np.array([r["allin_rust"] for r in rows])
    agp = np.array([r["agg_py"] for r in rows]); agr = np.array([r["agg_rust"] for r in rows])
    out = {
        "n_nodes": len(rows), "K": K, "mask_mismatch": mask_mismatch,
        "mean_allin_py": round(float(aip.mean()), 4), "mean_allin_rust": round(float(air.mean()), 4),
        "mean_agg_py": round(float(agp.mean()), 4), "mean_agg_rust": round(float(agr.mean()), 4),
        "frac_rust_lower_allin": round(float((air < aip - 0.02).mean()), 3),
        "frac_rust_higher_allin": round(float((air > aip + 0.02).mean()), 3),
        "frac_rust_lower_agg": round(float((agr < agp - 0.02).mean()), 3),
    }
    print(f"n={out['n_nodes']} nodes  mask_mismatch={mask_mismatch}", flush=True)
    print(f"  mean ALL_IN: py={out['mean_allin_py']:.3f}  rust={out['mean_allin_rust']:.3f}", flush=True)
    print(f"  mean AGGRESSION(bets+jam): py={out['mean_agg_py']:.3f}  rust={out['mean_agg_rust']:.3f}", flush=True)
    print(f"  frac nodes rust jams LOWER: {out['frac_rust_lower_allin']:.2f}  "
          f"HIGHER: {out['frac_rust_higher_allin']:.2f}", flush=True)
    print(f"  frac nodes rust LESS aggressive overall: {out['frac_rust_lower_agg']:.2f}", flush=True)
    json.dump({"summary": out, "rows": rows}, open(f"{MAIN}/artifacts/py_vs_rust_general_probe.json", "w"), indent=1)
    print(f"\nwrote {MAIN}/artifacts/py_vs_rust_general_probe.json", flush=True)


if __name__ == "__main__":
    main()
