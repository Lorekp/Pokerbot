#!/usr/bin/env python3
"""FIXED-DECK 6-max self-play DECISION-DIVERGENCE: does the 160k engine play differently with more search?

~100 static deck seeds (same cards every time), 6 copies of the 160k blueprint+search engine at a table.
For each deck we play a REFERENCE hand where every seat acts by the CONVERGED search (K=5000, SAMPLED with a
fixed per-node salt so hands reach the flop/turn/river where search matters), recording every non-trivial
decision node (state + acting seat). Then at each recorded node we RE-SOLVE at each iter count K in
{200,2000,3000,4000,5000} and compare the served policy to the K=5000 reference:
  * argmax_flip  -- fraction of decisions whose CHOSEN (modal) action differs from the converged solve
  * mean_TV      -- mean total-variation distance of the served action distribution vs K=5000
broken out by street. Static deck => the ONLY variable across K is search depth (run-2-at-200 ==
run-2-at-3000 on the same cards). NATIVE GTO subgame (no opponent model), deterministic per (K, node).

  POKERBOT_BETTING_ABSTRACTION=1 POKERBOT_POSTFLOP_BUCKETS=1 POKERBOT_PREFLOP_LOSSLESS=1 \
  POKERBOT_POSTFLOP_NUM_BUCKETS=200 python scripts/search_divergence_selfplay.py
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax  # noqa: E402

from pokerbot.engine.jax_engine import init_hand  # noqa: E402
from pokerbot.engine.np_engine import (  # noqa: E402
    apply_action_np, current_player_np, is_chance_node_np, is_terminal_np,
    legal_action_mask_np, sample_chance_np, to_numpy_state,
)
from pokerbot.search.pluribus_rollout import ContinuationLeafConfig  # noqa: E402
from pokerbot.search.pluribus_search import (  # noqa: E402
    PluribusReBeLConfig, PluribusReBeLSolver, set_native_subgame,
)

sys.path.insert(0, "scripts")
from native_exploit_equiv import build_root, load_snapshot  # noqa: E402

_STREET = {0: "preflop", 3: "flop", 4: "turn", 5: "river"}


def _street(env) -> str:
    return _STREET.get(int((np.asarray(env.board) >= 0).sum()), "?")


def _mk_solver(snap, K):
    return PluribusReBeLSolver(snap, PluribusReBeLConfig(
        max_iterations=K, time_budget_ms=10**12, search_preflop=True,
        validate_snapshot_at_init=False, native=True,
        leaf=ContinuationLeafConfig(select="uniform", rollout_samples=4)))


def _policy(solver, env, seat):
    root, pbs = build_root({"env": env, "hero_seat": int(seat)})
    p = np.asarray(solver.solve(root, pbs).action_probs, dtype=np.float64)
    return p


def main():
    bp = os.environ.get("BLUEPRINT_PATH", "checkpoints/serve_blueprint_6max.pkl")
    n_decks = int(os.environ.get("SD_DECKS", "100"))
    Ks = [int(x) for x in os.environ.get("SD_KS", "200,2000,3000,4000,5000").split(",")]
    REF = max(Ks)
    base = int(os.environ.get("SD_SEED", "777"))
    out = os.environ.get("SD_OUT", "artifacts/search_divergence_selfplay.json")

    set_native_subgame(True)
    snap = load_snapshot(bp)
    occ = jax.numpy.array([True] * 6)
    solvers = {K: _mk_solver(snap, K) for K in Ks}
    print(f"blueprint={bp} infosets={len(snap):,} decks={n_decks} Ks={Ks} ref=K{REF} (native GTO self-play)",
          flush=True)

    agg = {K: {} for K in Ks}            # K -> street -> [n_flip, n_nodes, sum_tv]
    n_nodes = 0
    solves = 0
    postflop_hands = 0
    t0 = time.perf_counter()

    for d in range(n_decks):
        key = jax.random.PRNGKey(base * 100003 + d)
        env = to_numpy_state(init_hand(key, occupancy=occ))
        nodes = []
        guard = 0
        reached_flop = False
        while not is_terminal_np(env) and guard < 400:
            guard += 1
            if is_chance_node_np(env):          # betting closed -> deal the next street (flop/turn/river)
                env = sample_chance_np(env)
                continue
            seat = current_player_np(env)
            if seat < 0:
                break
            mask = legal_action_mask_np(env, seat).astype(bool)
            if int(mask.sum()) <= 1:
                env = apply_action_np(env, int(np.argmax(mask)))
                continue
            st = _street(env)
            if st != "preflop":
                reached_flop = True
            pref = _policy(solvers[REF], env, seat); solves += 1
            nodes.append((env, int(seat), mask.copy(), st, pref))
            # SAMPLE the reference action (fixed per-node salt) so the hand plays realistically to showdown
            pn = np.clip(pref, 0, None); pn = pn / pn.sum() if pn.sum() > 0 else mask / mask.sum()
            salt = base * 1_000_003 + d * 9973 + len(nodes)
            a = int(np.random.default_rng(salt).choice(len(pn), p=pn))
            env = apply_action_np(env, a)
        if reached_flop:
            postflop_hands += 1
        # re-solve each recorded node at each K, compare argmax + TV to REF
        for (env_n, seat_n, mask_n, st, pref) in nodes:
            ref_arg = int(np.argmax(np.where(mask_n, pref, -1.0)))
            for K in Ks:
                pk = pref if K == REF else _policy(solvers[K], env_n, seat_n)
                if K != REF:
                    solves += 1
                k_arg = int(np.argmax(np.where(mask_n, pk, -1.0)))
                s = agg[K].setdefault(st, [0, 0, 0.0])
                s[0] += int(k_arg != ref_arg); s[1] += 1; s[2] += 0.5 * float(np.abs(pk - pref).sum())
            n_nodes += 1
        el = time.perf_counter() - t0
        eta = el / (d + 1) * (n_decks - d - 1)
        print(f"  deck {d+1}/{n_decks}: {len(nodes)} nodes ({'postflop' if reached_flop else 'preflop-only'}) "
              f"| cum {n_nodes} nodes, {postflop_hands} postflop hands, {solves} solves "
              f"| {el/60:.1f}min, ETA {eta/60:.1f}min", flush=True)
        json.dump({"blueprint": bp, "decks_done": d + 1, "Ks": Ks, "ref": REF, "n_nodes": n_nodes,
                   "postflop_hands": postflop_hands, "agg": {str(K): agg[K] for K in Ks}},
                  open(out, "w"), indent=1)

    print(f"\n=== DECISION DIVERGENCE vs K={REF} (fixed-deck self-play, {n_nodes} nodes, "
          f"{postflop_hands}/{n_decks} hands reached postflop) ===", flush=True)
    print(f"{'K':>6} {'argmax-flip %':>14} {'mean TV':>9}   (how often / how much the play changes vs converged)")
    for K in Ks:
        f = sum(v[0] for v in agg[K].values()); n = sum(v[1] for v in agg[K].values())
        tv = sum(v[2] for v in agg[K].values())
        print(f"{K:>6} {100.0*f/max(n,1):>13.1f}% {tv/max(n,1):>9.3f}", flush=True)
    streets = ["preflop", "flop", "turn", "river"]
    print(f"\nargmax-flip % by street:\n{'K':>6}  " + "  ".join(f"{s:>8}" for s in streets))
    for K in Ks:
        cells = [(f"{100.0*agg[K][s][0]/max(agg[K][s][1],1):>7.1f}%" if s in agg[K] else f"{'--':>8}")
                 for s in streets]
        print(f"{K:>6}  " + "  ".join(cells), flush=True)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
