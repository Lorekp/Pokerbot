#!/usr/bin/env python3
"""Confirm the RNG-STREAM COUPLING mechanism is SYSTEMATIC (not seed luck) and characterize it.

Runs the PYTHON production solve loop at the AKs node across MANY hole-sampler/action seeds under two
regimes:
  A) SHARED stream (production): one rng drives BOTH sample_opponent_holes and the opponent-action
     rng.choice (== both engines' real design).
  B) SEPARATE streams: a dedicated rng for hole sampling, a separate fixed rng for the solve's actions.

For each we report the served ALL_IN over seeds (mean, spread). If SHARED is reliably HIGH (~0.7) with
low spread, the coupling induces a SYSTEMATIC over-jam in Python; if SEPARATE is reliably LOW (~0.03),
that's the unbiased best-response. Native's shared-stream draw count differs, landing it near the
SEPARATE (low) value -- i.e. native accidentally avoids Python's coupling bias.

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

    K = 600
    set_native_subgame(False)

    def run(seed, shared):
        cfg = PluribusReBeLConfig(leaf=ContinuationLeafConfig(select="uniform"), search_preflop=True,
                                  max_iterations=K, time_budget_ms=10 ** 9)
        cfg.native = False; cfg.repair_deck = True; cfg.two_sided = False; cfg.blueprint_mix = 1.0
        s = PluribusReBeLSolver(snap, cfg, rng=np.random.default_rng(4242))
        s._leaf_cont_weights = None; s._two_sided_active = False; s._traversers = [seat]
        s._regrets.clear(); s._strategy_sum.clear(); s._bp_cache.clear(); s._leaf_cache.clear()
        s.rng = np.random.default_rng(seed)
        hole_rng = s.rng if shared else np.random.default_rng(seed ^ 0x5DEECE66D)
        rk = s._infoset_key(env, seat)
        for it in range(K):
            sampled = PBSTracker.init_uniform(seat, hole, board).sample_opponent_holes(hole_rng)
            world = s._set_world(env, sampled)
            s._cfv(world, seat, 0, it)
        legal = s._legal(env, seat)
        ss = s._strategy_sum.get(rk, np.zeros(11))
        served = (ss * legal); served = served / max(served.sum(), 1e-12)
        return float(served[ALL_IN])

    seeds = [11, 22, 33, 44, 55, 66, 77, 88]
    shared_vals = [run(sd, True) for sd in seeds]
    sep_vals = [run(sd, False) for sd in seeds]
    # native production for reference
    def native(seed):
        cfg = PluribusReBeLConfig(leaf=ContinuationLeafConfig(select="uniform"), search_preflop=True,
                                  max_iterations=K, time_budget_ms=10 ** 9)
        cfg.native = True; cfg.repair_deck = True; cfg.two_sided = False; cfg.blueprint_mix = 1.0
        s = PluribusReBeLSolver(snap, cfg, rng=np.random.default_rng(seed))
        set_native_subgame(True)
        s.rng = np.random.default_rng(seed)
        from pokerbot.search.state_bridge import SubgameRoot
        pbs = PBSTracker.init_uniform(seat, hole, board)
        legal = legal_action_mask_np(env, seat).astype(bool)
        root = SubgameRoot(env=env, hero_seat=seat, pbs_vector=pbs.get_public_belief_state(), legal_mask=legal)
        # vary the public seed input via rng_salt so native's internal public_seed changes
        s.config.rng_salt = seed
        r = s.solve(root, pbs)
        return float(np.asarray(r.action_probs)[ALL_IN])
    nat_vals = [native(sd) for sd in seeds]

    out = {
        "seat": seat, "K": K, "seeds": seeds,
        "python_shared_stream_allin": [round(v, 3) for v in shared_vals],
        "python_separate_stream_allin": [round(v, 3) for v in sep_vals],
        "native_allin": [round(v, 3) for v in nat_vals],
        "mean_shared": round(float(np.mean(shared_vals)), 3),
        "mean_separate": round(float(np.mean(sep_vals)), 3),
        "mean_native": round(float(np.mean(nat_vals)), 3),
    }
    print(f"\n  PYTHON shared-stream  ALL_IN: mean={out['mean_shared']:.3f}  vals={out['python_shared_stream_allin']}", flush=True)
    print(f"  PYTHON separate-stream ALL_IN: mean={out['mean_separate']:.3f}  vals={out['python_separate_stream_allin']}", flush=True)
    print(f"  NATIVE (production)    ALL_IN: mean={out['mean_native']:.3f}  vals={out['native_allin']}", flush=True)
    json.dump(out, open(f"{MAIN}/artifacts/py_vs_rust_rng_coupling.json", "w"), indent=1)
    print(f"\nwrote {MAIN}/artifacts/py_vs_rust_rng_coupling.json", flush=True)


if __name__ == "__main__":
    main()
