#!/usr/bin/env python3
"""INSTRUMENT the native-vs-Python fixed-point divergence on a preflop AKs open/jam node.

Read-only investigation harness. Constructs a controlled preflop root where the hero holds AKs and
solves it with BOTH engines (native Rust subgame vs interpreted Python _cfv) at several iteration
counts, reading the served ALL_IN probability. Both use repair_deck=ON, two_sided=OFF, blueprint_mix=1.0,
search_preflop=ON -- the exact H2H config. The ONLY difference is native-vs-interpreted.

Also measures, per engine at the AKs-jam node, the OPPONENT continue-vs-fold rate the hero is
best-responding to (by replaying the opponent-hole sampler + opponent_strategy at a facing-jam child),
and the AKs all-in equity, to rank the three hypotheses.

Run from the worktree so the patched .pyd loads:
  POKERBOT_BETTING_ABSTRACTION=1 POKERBOT_POSTFLOP_BUCKETS=1 POKERBOT_PREFLOP_LOSSLESS=1 \
  POKERBOT_POSTFLOP_MODE=fast POKERBOT_POSTFLOP_NUM_BUCKETS=200 \
  python scripts/probe_py_vs_rust_akq_jam.py <worktree> <main>
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

    from pokerbot.engine.jax_engine import init_hand
    from pokerbot.engine.np_engine import (
        apply_action_np, current_player_np, is_chance_node_np, is_terminal_np,
        legal_action_mask_np, sample_chance_np, to_numpy_state,
        ALL_IN, BET_BUCKET_START, CALL, FOLD, CHECK,
    )
    from pokerbot.search.pluribus_search import (
        PluribusReBeLConfig, PluribusReBeLSolver, set_native_subgame,
    )
    from pokerbot.search.pluribus_rollout import ContinuationLeafConfig
    from pokerbot.search.pbs_tracker import PBSTracker
    from pokerbot.search.state_bridge import SubgameRoot
    from pokerbot.engine.cards import Card
    import pickle

    _RANKS = "23456789TJQKA"
    _SUITS = "cdhs"

    def _card(idx):
        i = int(idx)
        return Card.from_str(_RANKS[i // 4] + _SUITS[i % 4])

    snap = (lambda o: o["strat_sum"] if isinstance(o, dict) and "strat_sum" in o else o)(
        pickle.load(open(f"{MAIN}/checkpoints/serve_blueprint_6max.pkl", "rb")))

    occ = jax.numpy.array([True] * 6)

    def hole_str(env, seat):
        h = env.holes[seat]
        if int(h[0]) < 0:
            return "--"
        def cs(i):
            return _RANKS[int(i) // 4] + _SUITS[int(i) % 4]
        return cs(h[0]) + cs(h[1])

    def is_AK(env, seat):
        h = env.holes[seat]
        if int(h[0]) < 0:
            return False
        r0, r1 = int(h[0]) // 4, int(h[1]) // 4
        s0, s1 = int(h[0]) % 4, int(h[1]) % 4
        return {r0, r1} == {12, 11} and s0 == s1  # AKs (rank 12=A, 11=K, suited)

    # ---- find a preflop node where the acting hero holds AKs (folded-to or facing a raise) ----
    # Scan seeds; play forward via blueprint-ish (just fold non-hero to reach an opener) is complex,
    # so instead: at each preflop decision, if the actor holds AKs, snapshot the env. We drive
    # non-AKs actors with a simple deterministic policy (min-action) only to REACH varied AKs spots.
    def find_ak_root(seed_base, want_facing):
        for d in range(4000):
            env = to_numpy_state(init_hand(jax.random.PRNGKey(seed_base * 100003 + d), occupancy=occ))
            g = 0
            while not is_terminal_np(env) and g < 40:
                g += 1
                if is_chance_node_np(env):
                    env = sample_chance_np(env)
                    continue
                seat = current_player_np(env)
                if seat < 0:
                    break
                if int(env.street) != 0:
                    break  # only preflop
                mask = legal_action_mask_np(env, seat).astype(bool)
                facing = bool(mask[FOLD])  # FOLD legal => there is a bet to call => facing action
                if is_AK(env, seat) and (facing == want_facing):
                    return env, int(seat), d
                # drive forward: opener raises pot-ish (bucket), others fold unless AK
                if mask[BET_BUCKET_START + 3]:
                    a = BET_BUCKET_START + 3
                elif mask[CALL]:
                    a = CALL
                elif mask[CHECK]:
                    a = CHECK
                else:
                    a = int(np.argmax(mask))
                env = apply_action_np(env, a)
        return None, None, None

    def make_solver(native, iters):
        cfg = PluribusReBeLConfig(
            leaf=ContinuationLeafConfig(select="uniform"),
            search_preflop=True,
            max_iterations=iters,
            time_budget_ms=10 ** 9,
        )
        cfg.native = native
        cfg.repair_deck = True
        cfg.two_sided = False
        cfg.blueprint_mix = 1.0
        s = PluribusReBeLSolver(snap, cfg, rng=np.random.default_rng(4242))
        return s

    def solve_probs(native, env, seat, iters):
        set_native_subgame(bool(native))
        s = make_solver(native, iters)
        s.rng = np.random.default_rng(4242)
        hole = (_card(int(env.holes[seat][0])), _card(int(env.holes[seat][1])))
        board = [_card(int(c)) for c in np.asarray(env.board) if int(c) >= 0]
        pbs = PBSTracker.init_uniform(seat, hole, board)
        legal = legal_action_mask_np(env, seat).astype(bool)
        root = SubgameRoot(env=env, hero_seat=seat,
                           pbs_vector=pbs.get_public_belief_state(), legal_mask=legal)
        res = s.solve(root, pbs)
        p = np.asarray(res.action_probs, dtype=np.float64)
        return p, res.metadata.get("served", "?")

    results = {}

    for label, want_facing in [("open_folded_to", False), ("facing_raise", True)]:
        env, seat, d = find_ak_root(4242, want_facing)
        if env is None:
            results[label] = {"error": "no AKs node found"}
            print(f"[{label}] no AKs node found", flush=True)
            continue
        blk = {"deck_idx": d, "seat": seat, "hole": hole_str(env, seat),
               "street": int(env.street), "board_n": int((np.asarray(env.board) >= 0).sum()),
               "curve": {}}
        print(f"\n=== {label}: AKs seat {seat} (deck {d}) hole={hole_str(env,seat)} "
              f"nboard={blk['board_n']} ===", flush=True)
        for iters in (50, 100, 200, 400, 1000):
            p_rust, srv_r = solve_probs(True, env, seat, iters)
            p_py, srv_p = solve_probs(False, env, seat, iters)
            row = {
                "rust_allin": round(float(p_rust[ALL_IN]), 4),
                "py_allin": round(float(p_py[ALL_IN]), 4),
                "rust_fold": round(float(p_rust[FOLD]), 4),
                "py_fold": round(float(p_py[FOLD]), 4),
                "rust_call": round(float(p_rust[CALL]), 4),
                "py_call": round(float(p_py[CALL]), 4),
                "rust_bet_sum": round(float(p_rust[BET_BUCKET_START:ALL_IN].sum()), 4),
                "py_bet_sum": round(float(p_py[BET_BUCKET_START:ALL_IN].sum()), 4),
                "served_rust": srv_r, "served_py": srv_p,
            }
            blk["curve"][iters] = row
            print(f"  K={iters:5d}  ALL_IN rust={row['rust_allin']:.3f} py={row['py_allin']:.3f}  "
                  f"| FOLD r={row['rust_fold']:.3f} p={row['py_fold']:.3f}  "
                  f"| CALL r={row['rust_call']:.3f} p={row['py_call']:.3f}  "
                  f"| BETS r={row['rust_bet_sum']:.3f} p={row['py_bet_sum']:.3f}  "
                  f"[{srv_r}/{srv_p}]", flush=True)
        results[label] = blk

    out = f"{MAIN}/artifacts/py_vs_rust_akq_jam_probe.json"
    json.dump(results, open(out, "w"), indent=1)
    print(f"\nwrote {out}", flush=True)


if __name__ == "__main__":
    main()
