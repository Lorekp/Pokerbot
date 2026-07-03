#!/usr/bin/env python3
"""STAGE-1 BATTERY DIAGNOSTIC: how often does the engine_ab_h2h game loop actually reach a
river_overbet-ARMABLE node (river decision, to_call==0, 2x-pot engine-legal, >1 legal action)?

Replays engine_ab_h2h's EXACT game loop (same init_hand keys, same arm-independent action salts,
same forced-node handling) with the DEPLOYED baseline policy on ALL six seats. Because the real
A-{} vs B-{river_overbet} run at the same seed produced ZERO divergent decks, the realized
trajectories of that battery are identical to this A-vs-A replay — so the counts here are the
EXACT armed-node exposure of the measured battery. Rot 0 only by default (rot 1 mirrors the seat
assignment, same trajectory statistics).

Usage: python scripts/probe_river_node_frequency.py --decks 200 --seed 9931
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

# Import engine_ab_h2h as a module: its module-level env preamble (EMD-K160 regime) runs BEFORE
# any pokerbot import, exactly like the real battery.
_spec = importlib.util.spec_from_file_location(
    "eab", os.path.join(REPO, "scripts", "engine_ab_h2h.py"))
eab = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(eab)

import numpy as np  # noqa: E402
import jax  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--decks", type=int, default=200)
    ap.add_argument("--seed", type=int, default=9931)
    ap.add_argument("--out", default=os.path.join(REPO, "artifacts", "river_node_frequency.json"))
    args = ap.parse_args()

    import pickle
    from pokerbot.engine.jax_engine import init_hand
    from pokerbot.engine.np_engine import (
        apply_action_np, current_player_np, is_chance_node_np, is_terminal_np,
        legal_action_mask_np, sample_chance_np, to_numpy_state,
    )
    from pokerbot.search.pluribus_search import set_native_subgame
    from pokerbot.search import native_solver as _ns

    if not _ns.native_subgame_available():
        _ns.enable()

    raw = pickle.load(open(os.path.join(REPO, "checkpoints", "serve_blueprint_6max.pkl"), "rb"))
    snap = raw["strat_sum"] if isinstance(raw, dict) and "strat_sum" in raw else raw
    pf = eab._build_policy({}, snap, args.seed, REPO)

    occ = jax.numpy.array([True] * 6)
    base_seed = args.seed
    stats = {"decks": 0, "decision_nodes": [0, 0, 0, 0], "forced_nodes": [0, 0, 0, 0],
             "river_checkto_armable": 0, "river_checkto_decks": [],
             "flopdecks": 0, "riverdecks": 0}

    for d in range(args.decks):
        env = to_numpy_state(init_hand(jax.random.PRNGKey(base_seed * 100003 + d), occupancy=occ))
        g = 0
        saw_flop = False
        saw_river_decision = False
        while not is_terminal_np(env) and g < 400:
            g += 1
            if is_chance_node_np(env):
                env = sample_chance_np(env)
                continue
            seat = current_player_np(env)
            if seat < 0:
                break
            street = int(env.street)
            if street > 0:
                saw_flop = True
            mask = legal_action_mask_np(env, seat).astype(bool)
            if int(mask.sum()) <= 1:
                stats["forced_nodes"][street] += 1
                a = int(np.argmax(mask))
            else:
                stats["decision_nodes"][street] += 1
                if street == 3:
                    saw_river_decision = True
                    to_call = int(env.current_bet) - int(np.asarray(env.committed_street)[seat])
                    if to_call == 0 and bool(mask[9]):
                        stats["river_checkto_armable"] += 1
                        stats["river_checkto_decks"].append(d)
                salt = base_seed * 7 + d * 131 + g
                set_native_subgame(True)
                p = np.asarray(pf(env, int(seat)), dtype=np.float64)
                pc = np.clip(p, 0, None)
                s = pc.sum()
                pc = (mask.astype(np.float64) / mask.sum()) if s <= 0 else pc / s
                a = int(np.random.default_rng(salt).choice(len(pc), p=pc))
            env = apply_action_np(env, a)
        stats["decks"] += 1
        stats["flopdecks"] += int(saw_flop)
        stats["riverdecks"] += int(saw_river_decision)
        if (d + 1) % 25 == 0:
            print(f"  deck {d + 1}/{args.decks}: decision_nodes={stats['decision_nodes']} "
                  f"river_checkto_armable={stats['river_checkto_armable']} "
                  f"flopdecks={stats['flopdecks']} riverdecks={stats['riverdecks']}", flush=True)

    print(json.dumps({k: v for k, v in stats.items() if k != "river_checkto_decks"}, indent=1))
    with open(args.out, "w") as f:
        json.dump(stats, f, indent=1)
    print(f"wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
