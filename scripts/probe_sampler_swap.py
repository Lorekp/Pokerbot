#!/usr/bin/env python3
"""DECISIVE H1 test: run the PYTHON solver but replace its opponent-hole sampler with a port of NATIVE's
`deal_opponent_holes` (single Fisher-Yates over the unseen pool, sequential seat deal). If Python's jam
Q/served then COLLAPSES toward native's low value, the mechanism is the opponent-hole SAMPLING DIVERSITY
(H1). If it stays high, sampling is not the cause and the divergence is elsewhere (RNG/traversal).

We also directly compare, over N draws, the marginal opponent-combo distribution of the two samplers
(are they both uniform? does one concentrate?), and whether per-iteration av[ALL_IN] shifts.

Read-only: we only swap the sampler used to build worlds inside a manual copy of the Python solve loop.

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
        ALL_IN, BET_BUCKET_START, CALL, FOLD, CHECK, SEAT_FOLDED,
    )
    from pokerbot.engine.jax_engine import init_hand
    from pokerbot.search.pluribus_search import (
        PluribusReBeLConfig, PluribusReBeLSolver, set_native_subgame,
    )
    from pokerbot.search.pluribus_rollout import ContinuationLeafConfig
    from pokerbot.search.pbs_tracker import PBSTracker
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

    hole_idx = (int(env.holes[seat][0]), int(env.holes[seat][1]))
    board_idx = [int(c) for c in np.asarray(env.board) if int(c) >= 0]
    known0 = set(hole_idx) | set(board_idx)

    # Native-style dealer: single Fisher-Yates over unseen, deal sequentially to non-hero seats.
    def native_style_sample(rng):
        unseen = [c for c in range(52) if c not in known0]
        rng.shuffle(unseen)
        out = {}
        p = 0
        for s in range(6):
            if s == seat:
                continue
            out[s] = (unseen[p], unseen[p + 1]); p += 2
        return out

    K = 1000
    set_native_subgame(False)

    def python_loop(sampler_kind):
        cfg = PluribusReBeLConfig(leaf=ContinuationLeafConfig(select="uniform"), search_preflop=True,
                                  max_iterations=K, time_budget_ms=10 ** 9)
        cfg.native = False; cfg.repair_deck = True; cfg.two_sided = False; cfg.blueprint_mix = 1.0
        s = PluribusReBeLSolver(snap, cfg, rng=np.random.default_rng(4242))
        s._leaf_cont_weights = None; s._two_sided_active = False; s._traversers = [seat]
        s._regrets.clear(); s._strategy_sum.clear(); s._bp_cache.clear(); s._leaf_cache.clear()
        s.rng = np.random.default_rng(s._public_seed(env, seat))
        rk = s._infoset_key(env, seat)
        for it in range(K):
            if sampler_kind == "python":
                sampled = PBSTracker.init_uniform(seat, hole, board).sample_opponent_holes(s.rng)
            else:  # native-style
                sampled = native_style_sample(s.rng)
            world = s._set_world(env, sampled)
            s._cfv(world, seat, 0, it)
        legal = s._legal(env, seat)
        ss = s._strategy_sum.get(rk, np.zeros(11))
        served = (ss * legal); served = served / max(served.sum(), 1e-12)
        regr = s._regrets.get(rk, np.zeros(11))
        return served, regr, legal

    out = {"seat": seat, "K": K}
    for kind in ("python", "native_style"):
        served, regr, legal = python_loop(kind)
        out[f"python_with_{kind}_sampler"] = {
            "served_allin": round(float(served[ALL_IN]), 4),
            "served_bets": round(float(served[BET_BUCKET_START:ALL_IN].sum()), 4),
            "root_regret_allin": round(float(regr[ALL_IN]), 1),
            "root_regret_bet_max": round(float(regr[BET_BUCKET_START:ALL_IN].max()), 1),
        }
        print(f"  PYTHON solver + {kind:>12} sampler: served ALL_IN={served[ALL_IN]:.3f}  "
              f"BETS={served[BET_BUCKET_START:ALL_IN].sum():.3f}  "
              f"regret[ALL_IN]={regr[ALL_IN]:.0f} max_bet_regret={regr[BET_BUCKET_START:ALL_IN].max():.0f}", flush=True)

    # marginal opp-combo distribution comparison (are both uniform?)
    from collections import Counter
    def marginals(sampler, n=20000):
        cnt = Counter()
        rng = np.random.default_rng(123)
        for _ in range(n):
            if sampler == "python":
                sh = PBSTracker.init_uniform(seat, hole, board).sample_opponent_holes(rng)
            else:
                sh = native_style_sample(rng)
            for s2, (c0, c1) in sh.items():
                cnt[(min(c0, c1), max(c0, c1))] += 1
        return cnt
    mp = marginals("python"); mn = marginals("native_style")
    # compare distribution over combos: total variation-ish (top concentration)
    def stats(c):
        v = np.array(list(c.values()), dtype=float); v = v / v.sum()
        return {"n_distinct": len(c), "max_p": round(float(v.max()), 5), "entropy": round(float(-(v * np.log(v + 1e-12)).sum()), 3)}
    out["marginal_python"] = stats(mp)
    out["marginal_native_style"] = stats(mn)
    print(f"\n  marginal opp-combo: python {out['marginal_python']}  native {out['marginal_native_style']}", flush=True)

    json.dump(out, open(f"{MAIN}/artifacts/py_vs_rust_sampler_swap.json", "w"), indent=1)
    print(f"\nwrote {MAIN}/artifacts/py_vs_rust_sampler_swap.json", flush=True)


if __name__ == "__main__":
    main()
