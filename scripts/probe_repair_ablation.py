#!/usr/bin/env python3
"""Ablation: native vs Python served ALL_IN at the AKs node across repair_deck ON/OFF and two_sided ON/OFF.

Discriminates the mechanism:
  - If Python(repair=OFF) jams HIGH and Python(repair=ON) jams LOWER, the repair fix reduces Python's
    over-jam (the known stale-deck bug).
  - If native jams LOW at BOTH repair settings while Python jams HIGH at both, the divergence is NOT the
    deck at all -> a genuine CFR-trajectory/RNG difference.
  - two_sided=ON should collapse both engines' jam (opponents learn to defend) -> the principled fix.

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

    def solve(native, repair, two, iters):
        cfg = PluribusReBeLConfig(leaf=ContinuationLeafConfig(select="uniform"), search_preflop=True,
                                  max_iterations=iters, time_budget_ms=10 ** 9)
        cfg.native = native; cfg.repair_deck = repair; cfg.two_sided = two; cfg.blueprint_mix = 1.0
        s = PluribusReBeLSolver(snap, cfg, rng=np.random.default_rng(4242))
        set_native_subgame(bool(native))
        s.rng = np.random.default_rng(4242)
        pbs = PBSTracker.init_uniform(seat, hole, board)
        legal = legal_action_mask_np(env, seat).astype(bool)
        root = SubgameRoot(env=env, hero_seat=seat, pbs_vector=pbs.get_public_belief_state(), legal_mask=legal)
        r = s.solve(root, pbs)
        p = np.asarray(r.action_probs, dtype=np.float64)
        return float(p[ALL_IN]), float(p[BET_BUCKET_START:ALL_IN].sum()), float(p[CALL]), r.metadata.get("served")

    K = 1000
    out = {"seat": seat, "iters": K, "grid": {}}
    print(f"\n{'config':<28}{'ALL_IN':>8}{'BETS':>8}{'CALL':>8}  served", flush=True)
    for native in (False, True):
        for repair in (False, True):
            for two in (False, True):
                ai, bets, call, srv = solve(native, repair, two, K)
                name = f"{'RUST' if native else 'PY  '} repair={int(repair)} two={int(two)}"
                out["grid"][name] = {"allin": round(ai, 4), "bets": round(bets, 4), "call": round(call, 4), "served": srv}
                print(f"{name:<28}{ai:>8.3f}{bets:>8.3f}{call:>8.3f}  {srv}", flush=True)

    json.dump(out, open(f"{MAIN}/artifacts/py_vs_rust_repair_ablation.json", "w"), indent=1)
    print(f"\nwrote {MAIN}/artifacts/py_vs_rust_repair_ablation.json", flush=True)


if __name__ == "__main__":
    main()
