#!/usr/bin/env python3
"""ANCHOR the blueprint's preflop RFI to a published GTO chart (user 2026-07-09: the CFR preflop is a
degenerate limp/jam equilibrium of the abstracted game; the value-bootstrap can't install open-raising, so
OVERRIDE the RFI rows with a known-sound open/fold chart and keep search for postflop).

For every RFI node (folded-to-hero at each position x all 169 hands) we compute the EXACT serve fingerprint
(same env + fingerprint_fast_emd as preflop_strategy_audit.py, which matched the deployed keyspace) and set
the served row to [FOLD = 1-f, OPEN@bucket7 = f] where f = 4-source consensus open frequency (no limp, no
jam). Deployed POSTFLOP rows + non-RFI preflop rows are retained. Output: a serve snapshot pkl.

Deployed serve_blueprint_6max.pkl is READ-ONLY. Output -> artifacts/serve_blueprint_anchored.pkl.

  python scripts/anchor_preflop.py [--out artifacts/serve_blueprint_anchored.pkl] [--deals 6]
"""
import argparse
import json
import os
import pickle

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
EMD = os.path.join(REPO, "data", "abstraction", "emd_frozen_flop_turn_160_r400.emdfrzn")
for _k, _v in {
    "JAX_PLATFORMS": "cpu", "POKERBOT_BETTING_ABSTRACTION": "1", "POKERBOT_POSTFLOP_BUCKETS": "1",
    "POKERBOT_POSTFLOP_NUM_BUCKETS": "160", "POKERBOT_POSTFLOP_MODE": "emd",
    "POKERBOT_POSTFLOP_EMD_FROZEN": EMD, "POKERBOT_EMD_FROZEN_MIRROR_FALLBACK": "1",
    "POKERBOT_PREFLOP_LOSSLESS": "1",
}.items():
    os.environ.setdefault(_k, _v)

import numpy as np  # noqa: E402
import jax  # noqa: E402
from pokerbot.engine.jax_engine import init_hand  # noqa: E402
from pokerbot.engine.np_engine import (  # noqa: E402
    apply_action_np, current_player_np, is_terminal_np, to_numpy_state,
)
from pokerbot.search import native_solver as ns  # noqa: E402
import battery_native as bn  # noqa: E402

FOLD, CHECK, CALL, OPEN, ALL_IN = 0, 1, 2, 7, 10   # OPEN=bucket 7 (the deployed/audit standard open)
RANKS = "23456789TJQKA"
POSITIONS = ["UTG", "HJ", "CO", "BTN", "SB"]
SEAT_OFF = {"BTN": 0, "SB": 1, "BB": 2, "UTG": 3, "HJ": 4, "CO": 5}


def all_169():
    out = []
    for i in range(12, -1, -1):
        for j in range(12, -1, -1):
            if i == j:
                out.append((RANKS[i] * 2, (i * 4 + 0, i * 4 + 1)))
            elif i > j:
                out.append((RANKS[i] + RANKS[j] + "s", (i * 4 + 0, j * 4 + 0)))
            else:
                out.append((RANKS[j] + RANKS[i] + "o", (j * 4 + 0, i * 4 + 1)))
    seen, uniq = set(), []
    for lab, cc in out:
        if lab not in seen:
            seen.add(lab)
            uniq.append((lab, cc))
    return uniq


def seat_of(btn, pos):
    return (btn + SEAT_OFF[pos]) % 6


def preflop_order(btn):
    return [(btn + 3 + i) % 6 for i in range(6)]


def rig_hero(holes, hero, cards):
    holes = holes.copy()

    def find(card):
        for s in range(holes.shape[0]):
            for sl in range(2):
                if int(holes[s, sl]) == card:
                    return (s, sl)
        return None
    for i, card in enumerate(cards):
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


def rfi_node(env0, btn, pos):
    pi = POSITIONS.index(pos)
    e = env0
    for s in preflop_order(btn)[:pi]:
        if is_terminal_np(e) or current_player_np(e) != s:
            return None, None
        e = apply_action_np(e, FOLD)
    hero = seat_of(btn, pos)
    if is_terminal_np(e) or current_player_np(e) != hero:
        return None, None
    return e, hero


def key_of(e, hero):
    flat = ns._to_flat(e).tobytes()
    return bytes(bn.fingerprint_fast_emd(flat, int(hero), 160, [], EMD, True))


def consensus_rfi(chart):
    """position -> {hand: mean open-freq over the 4 sources (absent = 0)}."""
    out = {}
    for pos, srcmap in chart["rfi"].items():
        srcs = list(srcmap.keys())
        hands = set()
        for s in srcs:
            hands |= set(srcmap[s].keys())
        m = {}
        for h in hands:
            vals = [float(srcmap[s].get(h, 0.0)) for s in srcs]
            m[h] = sum(vals) / len(vals)
        out[pos] = m
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chart", default=os.path.join(REPO, "artifacts", "published_preflop_ranges.json"))
    ap.add_argument("--out", default=os.path.join(REPO, "artifacts", "serve_blueprint_anchored.pkl"))
    ap.add_argument("--deals", type=int, default=6, help="button rotations to cover all RFI-node fingerprints")
    args = ap.parse_args()

    chart = json.load(open(args.chart))
    cons = consensus_rfi(chart)
    dep_path = os.path.join(REPO, "checkpoints", "serve_blueprint_6max.pkl")
    raw = pickle.load(open(dep_path, "rb"))
    wrapped = isinstance(raw, dict) and "strat_sum" in raw
    dep = raw["strat_sum"] if wrapped else raw
    dep = {bytes(k): np.asarray(v, dtype=np.float64) for k, v in dep.items()}
    print(f"deployed: {len(dep):,} rows", flush=True)

    hands = all_169()
    occ = jax.numpy.array([True] * 6)
    anchored = {}
    n_nodes, n_newkey = 0, 0
    for pos in POSITIONS:
        m = cons[pos]
        for lab, cards in hands:
            f = float(m.get(lab, 0.0))            # consensus open frequency (0 if absent)
            row = np.zeros(11, dtype=np.float64)
            row[FOLD] = 1.0 - f
            row[OPEN] = f
            for d in range(args.deals):
                env0 = to_numpy_state(init_hand(jax.random.PRNGKey(730000 + d), occupancy=occ))
                btn = int(env0.button)
                e, hero = rfi_node(env0, btn, pos)
                if e is None:
                    continue
                e = e._replace(holes=rig_hero(np.asarray(e.holes), hero, cards))
                k = key_of(e, hero)
                if k not in dep:
                    n_newkey += 1
                anchored[k] = row.copy()
                n_nodes += 1

    print(f"anchored RFI keys: {len(anchored):,}  (nodes visited {n_nodes:,}; keys not in deployed: {n_newkey:,})",
          flush=True)
    merged = dict(dep)
    merged.update(anchored)
    print(f"merged: {len(merged):,} rows (deployed {len(dep):,} + {len(merged)-len(dep):,} new)", flush=True)

    # ---- validate: re-lookup each RFI node in `merged` -> must return the anchored (open/fold) shape ----
    print("\n=== RFI shape AFTER anchor (open%/limp%/fold% modal over 169) ===", flush=True)
    for pos in POSITIONS:
        opn = limp = jam = fold = 0
        for lab, cards in hands:
            env0 = to_numpy_state(init_hand(jax.random.PRNGKey(730000), occupancy=occ))
            btn = int(env0.button)
            e, hero = rfi_node(env0, btn, pos)
            if e is None:
                continue
            e = e._replace(holes=rig_hero(np.asarray(e.holes), hero, cards))
            r = merged.get(key_of(e, hero))
            if r is None:
                continue
            s = float(r.sum())
            p = r / s if s > 0 else r
            a = int(np.argmax([p[FOLD], p[CALL], p[3:ALL_IN].sum(), p[ALL_IN]]))
            if a == 0:
                fold += 1
            elif a == 1:
                limp += 1
            elif a == 2:
                opn += 1
            else:
                jam += 1
        n = fold + limp + opn + jam
        print(f"  {pos:3s}: open={100*opn/n:5.1f}%  limp={100*limp/n:4.1f}%  jam={100*jam/n:4.1f}%  "
              f"fold={100*fold/n:5.1f}%  (n={n})", flush=True)

    out_obj = {"strat_sum": merged} if wrapped else merged
    with open(args.out, "wb") as f:
        pickle.dump(out_obj, f)
    print(f"\nwrote {args.out}", flush=True)
    print("Reference (sound, class-count basis higher than combo%): premiums 100% open, limp 0, jam 0.", flush=True)


if __name__ == "__main__":
    main()
