#!/usr/bin/env python3
"""Isolate WHICH part of the sampler swap flips Python's jam: (A) the hole JOINT distribution itself, or
(B) RNG-stream desync (the sampler consumes a different number of draws, shifting the downstream
opponent-ACTION sampling stream).

We run the Python solve loop three ways, all with a SEPARATE, fixed rng for the hole sampler (so the
solve's own opponent-action rng stream is IDENTICAL across arms) and a matched hole-sampler seed:
  1. python-sampler   (PBSTracker.sample_opponent_holes)
  2. native-style      (single Fisher-Yates, deal all 5 non-hero seats)
  3. native-style-LIVE (single Fisher-Yates, deal only NON-FOLDED non-hero seats; folded seats left as-is)

If (2)/(3) still collapse the jam vs (1) with the action-rng held fixed, the cause is the hole JOINT
distribution (H1 proper). If holding the action-rng fixed makes them agree, the cause was RNG desync.

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
    hole_idx = (int(env.holes[seat][0]), int(env.holes[seat][1]))
    board_idx = [int(c) for c in np.asarray(env.board) if int(c) >= 0]
    known0 = set(hole_idx) | set(board_idx)
    status = np.asarray(env.seat_status)
    live_opps = [s for s in range(6) if s != seat and int(status[s]) != SEAT_FOLDED]
    folded_opps = [s for s in range(6) if s != seat and int(status[s]) == SEAT_FOLDED]
    print(f"AKs facing-raise: seat={seat}  live_opps={live_opps}  folded_opps={folded_opps}", flush=True)

    def native_all(rng):
        unseen = [c for c in range(52) if c not in known0]; rng.shuffle(unseen)
        out = {}; p = 0
        for s in range(6):
            if s == seat: continue
            out[s] = (unseen[p], unseen[p + 1]); p += 2
        return out

    def native_live_only(rng):
        unseen = [c for c in range(52) if c not in known0]; rng.shuffle(unseen)
        out = {}; p = 0
        for s in range(6):
            if s == seat or int(status[s]) == SEAT_FOLDED: continue
            out[s] = (unseen[p], unseen[p + 1]); p += 2
        return out

    def python_sampler(rng):
        return PBSTracker.init_uniform(seat, hole, board).sample_opponent_holes(rng)

    K = 1000
    set_native_subgame(False)

    def run(sampler):
        cfg = PluribusReBeLConfig(leaf=ContinuationLeafConfig(select="uniform"), search_preflop=True,
                                  max_iterations=K, time_budget_ms=10 ** 9)
        cfg.native = False; cfg.repair_deck = True; cfg.two_sided = False; cfg.blueprint_mix = 1.0
        s = PluribusReBeLSolver(snap, cfg, rng=np.random.default_rng(4242))
        s._leaf_cont_weights = None; s._two_sided_active = False; s._traversers = [seat]
        s._regrets.clear(); s._strategy_sum.clear(); s._bp_cache.clear(); s._leaf_cache.clear()
        # SEPARATE fixed action-rng (the solve's own stream) held identical across arms:
        s.rng = np.random.default_rng(s._public_seed(env, seat))
        hole_rng = np.random.default_rng(20260701)  # matched hole-sampler seed across arms
        rk = s._infoset_key(env, seat)
        for it in range(K):
            sampled = sampler(hole_rng)
            world = s._set_world(env, sampled)
            s._cfv(world, seat, 0, it)
        legal = s._legal(env, seat)
        ss = s._strategy_sum.get(rk, np.zeros(11))
        served = (ss * legal); served = served / max(served.sum(), 1e-12)
        regr = s._regrets.get(rk, np.zeros(11))
        return float(served[ALL_IN]), float(served[BET_BUCKET_START:ALL_IN].sum()), float(regr[ALL_IN]), float(regr[BET_BUCKET_START:ALL_IN].max())

    out = {"seat": seat, "K": K, "live_opps": live_opps, "folded_opps": folded_opps}
    for name, samp in [("python", python_sampler), ("native_all", native_all), ("native_live_only", native_live_only)]:
        ai, bets, rai, rbet = run(samp)
        out[name] = {"allin": round(ai, 4), "bets": round(bets, 4), "regret_allin": round(rai, 1), "regret_bet_max": round(rbet, 1)}
        print(f"  {name:>17}: ALL_IN={ai:.3f} BETS={bets:.3f}  regret[ALL_IN]={rai:.0f} max_bet_regret={rbet:.0f}", flush=True)

    # KEY DIFF: does python sample holes to FOLDED seats and does native's dealing-to-folded matter?
    # Count how often python assigns holes to folded seats (it does; _opponent_seats includes folded).
    json.dump(out, open(f"{MAIN}/artifacts/py_vs_rust_sampler_isolate.json", "w"), indent=1)
    print(f"\nwrote {MAIN}/artifacts/py_vs_rust_sampler_isolate.json", flush=True)


if __name__ == "__main__":
    main()
