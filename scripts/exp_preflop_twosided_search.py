#!/usr/bin/env python3
"""EXPERIMENT: does TWO-SIDED preflop search (with repair_deck) OPEN AA at serve time, or still over-jam?

The blueprint limps AA preflop and is served RAW (search_preflop=False); the deployed serve searches
POSTFLOP only, so it structurally cannot touch the preflop shape. Preflop search has been OFF because it
"over-jams" -- but that over-jam has TWO independent causes that have NEVER been fixed together:
  * stale-deck bug (repair_deck): all-in lines replayed ONE frozen board -> BR to a single runout
    (pluribus_search.py:80-88).
  * one-sided over-aggression (two_sided): hero one-sidedly BRs a frozen field (pluribus_search.py:89-95).
The "over-jams even two-sided" note (live_advisor_bridge.py:264) predates the repair_deck combination.

This rigs hero to AA at RFI (folded-to-hero) nodes and serves under 4 configs, reporting the served
action distribution. Verdict per config: OPEN (raise mass high) = serve-time fix; JAM (all-in mass high)
= over-jam persists; LIMP (call mass high) = trap persists.

  python scripts/exp_preflop_twosided_search.py [n_per_pos=6] [iters=400] [positions=UTG,CO,BTN]
"""
from __future__ import annotations

import json
import os
import sys
import time

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

os.environ.setdefault("JAX_PLATFORMS", "cpu")
_FROZEN = os.path.join(REPO, "data", "abstraction", "emd_frozen_flop_turn_160_r400.emdfrzn")
for _k, _v in {
    "POKERBOT_BETTING_ABSTRACTION": "1", "POKERBOT_POSTFLOP_BUCKETS": "1",
    "POKERBOT_POSTFLOP_NUM_BUCKETS": "160", "POKERBOT_POSTFLOP_MODE": "emd",
    "POKERBOT_POSTFLOP_EMD_FROZEN": _FROZEN, "POKERBOT_EMD_FROZEN_MIRROR_FALLBACK": "1",
    "POKERBOT_PREFLOP_LOSSLESS": "1",
}.items():
    os.environ.setdefault(_k, _v)
os.environ.setdefault("POKERBOT_NATIVE_SUBGAME", "1")  # deployed; repair_deck falls back to python per-solve

import numpy as np  # noqa: E402

FOLD, CHECK, CALL = 0, 1, 2
BET_BUCKET_START, ALL_IN = 3, 10


def seat_of(button: int, pos: str) -> int:
    return (button + {"BTN": 0, "SB": 1, "BB": 2, "UTG": 3, "HJ": 4, "CO": 5}[pos]) % 6


def preflop_order(button: int) -> list[int]:
    return [(button + 3 + i) % 6 for i in range(6)]


def rig_hero_aces(holes: np.ndarray, hero: int) -> np.ndarray:
    """Swap within the holes array so hero holds Ac,Ad. Preflop: board undealt, aces live only in the 12
    hole slots, so an in-array swap preserves the 12-card multiset (NO duplicate cards introduced)."""
    holes = holes.copy()
    want = [48, 49]  # rank A=12 -> 12*4+suit ; Ac=48, Ad=49

    def find(card: int):
        for s in range(holes.shape[0]):
            for sl in range(2):
                if int(holes[s, sl]) == card:
                    return (s, sl)
        return None

    for i, card in enumerate(want):
        loc = find(card)
        if loc == (hero, i):
            continue
        if loc is None:
            holes[hero, i] = card
            continue
        tmp = int(holes[hero, i])
        holes[hero, i] = int(holes[loc[0], loc[1]])
        holes[loc[0], loc[1]] = tmp
    return holes


def main() -> int:
    import pickle

    import jax
    from pokerbot.engine.jax_engine import init_hand
    from pokerbot.engine.np_engine import (
        apply_action_np, current_player_np, is_terminal_np, legal_action_mask_np, to_numpy_state,
    )
    from pokerbot.eval.solver_exploitability import make_search_policy_fn

    n_per_pos = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    iters = int(sys.argv[2]) if len(sys.argv) > 2 else 400
    positions = (sys.argv[3].split(",") if len(sys.argv) > 3 else ["UTG", "CO", "BTN"])
    leaf = sys.argv[4] if len(sys.argv) > 4 else "uniform"  # continuation leaf: uniform | gadget

    ckpt = os.path.join(REPO, "checkpoints", "serve_blueprint_6max.pkl")
    raw = pickle.load(open(ckpt, "rb"))
    snap = raw["strat_sum"] if isinstance(raw, dict) and "strat_sum" in raw else raw
    print(f"blueprint: {len(snap)} infosets | iters={iters} n_per_pos={n_per_pos} pos={positions}", flush=True)

    CFGS = {
        "BP_blueprint":    dict(search_preflop=False),
        "PF_1side":        dict(search_preflop=True, two_sided=False, repair_deck=False, iterations=iters),
        "PF_2side":        dict(search_preflop=True, two_sided=True, repair_deck=False, iterations=iters),
        "PF_2side_repair": dict(search_preflop=True, two_sided=True, repair_deck=True, iterations=iters),
    }
    policies = {name: make_search_policy_fn(snap, leaf_select=leaf, seed=0,
                                            time_budget_ms=10 ** 9, **kw)
                for name, kw in CFGS.items()}
    print(f"continuation leaf = {leaf}", flush=True)

    # acc[cfg][pos] -> list of 11-prob rows
    acc: dict[str, dict[str, list]] = {c: {p: [] for p in positions} for c in CFGS}
    occ = jax.numpy.array([True] * 6)
    t0 = time.time()

    d = 0
    need = {p: n_per_pos for p in positions}
    while any(need[p] > 0 for p in positions):
        env0 = to_numpy_state(init_hand(jax.random.PRNGKey(910001 + d), occupancy=occ))
        d += 1
        btn = int(env0.button)
        order = preflop_order(btn)
        for pos in positions:
            if need[pos] <= 0:
                continue
            pi = ["UTG", "HJ", "CO", "BTN", "SB"].index(pos)
            e = env0
            ok = True
            for s in order[:pi]:
                if is_terminal_np(e) or current_player_np(e) != s:
                    ok = False
                    break
                e = apply_action_np(e, FOLD)
            if not ok or is_terminal_np(e):
                continue
            hero = seat_of(btn, pos)
            if current_player_np(e) != hero:
                continue
            e = e._replace(holes=rig_hero_aces(np.asarray(e.holes), hero))
            legal = legal_action_mask_np(e, hero)
            for name, pol in policies.items():
                probs = np.asarray(pol(e, hero), dtype=np.float64)
                acc[name][pos].append([round(float(x), 6) for x in probs])
            need[pos] -= 1
            print(f"  {pos}: sample {n_per_pos - need[pos]}/{n_per_pos} "
                  f"(deal {d}, {time.time() - t0:.0f}s) legal_raises="
                  f"{[a for a in range(BET_BUCKET_START, ALL_IN + 1) if legal[a]]}", flush=True)

    def summarize(rows):
        arr = np.asarray(rows)
        m = arr.mean(axis=0)
        return {
            "n": len(rows),
            "fold": round(float(m[FOLD]), 3),
            "limp_call": round(float(m[CALL]), 3),
            "open_raise": round(float(m[BET_BUCKET_START:ALL_IN].sum()), 3),
            "all_in": round(float(m[ALL_IN]), 3),
            "raise_by_bucket": [round(float(m[a]), 3) for a in range(BET_BUCKET_START, ALL_IN + 1)],
            "probs": [round(float(x), 4) for x in m],
        }

    out = {"meta": {"hand": "AA", "iters": iters, "n_per_pos": n_per_pos, "positions": positions,
                    "configs": {k: {kk: vv for kk, vv in v.items()} for k, v in CFGS.items()},
                    "action_layout": "0 fold,1 check,2 call(=limp at RFI),3..9 pot-frac bets,10 all-in"},
           "results": {}}
    print("\n==== AA served action distribution (RFI, folded-to-hero) ====", flush=True)
    print(f"{'config':17s} {'pos':4s} {'fold':>6s} {'limp':>6s} {'open':>6s} {'jam':>6s}   verdict", flush=True)
    for name in CFGS:
        out["results"][name] = {}
        for pos in positions:
            if not acc[name][pos]:
                continue
            s = summarize(acc[name][pos])
            out["results"][name][pos] = s
            top = max([("fold", s["fold"]), ("limp", s["limp_call"]),
                       ("open", s["open_raise"]), ("jam", s["all_in"])], key=lambda x: x[1])
            print(f"{name:17s} {pos:4s} {s['fold']:6.2f} {s['limp_call']:6.2f} "
                  f"{s['open_raise']:6.2f} {s['all_in']:6.2f}   -> {top[0].upper()}", flush=True)

    path = os.path.join(REPO, "artifacts", "exp_preflop_twosided_search.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    print(f"\nwrote {path} ({time.time() - t0:.0f}s)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
