#!/usr/bin/env python3
"""Facing-an-open (3-bet) shape for the DEPLOYED blueprint. For a set of (opener, hero) spots, the opener
makes a standard raise, folds around to hero, and we read hero's served response distribution across all
169 hand classes: FOLD / FLAT(call) / 3BET(raise) / JAM. Shows whether the blueprint has a real 3-bet
range or collapses to flat/fold (another whole-tree dimension beyond RFI).

  python scripts/vs_open_audit.py
"""
import os
import sys

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
import pickle  # noqa: E402
from pokerbot.engine.jax_engine import init_hand  # noqa: E402
from pokerbot.engine.np_engine import (  # noqa: E402
    apply_action_np, current_player_np, is_terminal_np, legal_action_mask_np, to_numpy_state,
)
from pokerbot.search import native_solver as ns  # noqa: E402
import battery_native as bn  # noqa: E402

FOLD, CHECK, CALL, ALL_IN = 0, 1, 2, 10
OPEN = 7  # opener's standard raise bucket (matches make_aa_open_override)
RANKS = "23456789TJQKA"
ORDER_POS = ["UTG", "HJ", "CO", "BTN", "SB", "BB"]  # preflop acting order
# spots: (opener_pos, hero_pos) with opener before hero
SPOTS = [("CO", "BTN"), ("BTN", "BB"), ("BTN", "SB"), ("UTG", "BB"), ("CO", "SB")]


def all_169():
    out, seen = [], set()
    for i in range(12, -1, -1):
        for j in range(12, -1, -1):
            if i == j:
                lab, cc = RANKS[i] * 2, (i * 4 + 0, i * 4 + 1)
            elif i > j:
                lab, cc = RANKS[i] + RANKS[j] + "s", (i * 4 + 0, j * 4 + 0)
            else:
                lab, cc = RANKS[j] + RANKS[i] + "o", (j * 4 + 0, i * 4 + 1)
            if lab not in seen:
                seen.add(lab)
                out.append((lab, cc))
    return out


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


def lookup(snap, e, hero):
    flat = ns._to_flat(e).tobytes()
    key = bytes(bn.fingerprint_fast_emd(flat, hero, 160, [], EMD, True))
    row = snap.get(key)
    if row is None:
        return None
    s = float(row.sum())
    return (row / s) if s > 0 else None


def build_vs_open(env0, btn, opener_pos, hero_pos):
    """Walk preflop: fold before opener, opener raises OPEN, fold between opener and hero. Returns
    (env_at_hero, hero_seat) or (None, None) if the line is not reachable as constructed."""
    off = {"BTN": 0, "SB": 1, "BB": 2, "UTG": 3, "HJ": 4, "CO": 5}
    order = [(btn + 3 + i) % 6 for i in range(6)]  # UTG..BB seats
    oi, hi = ORDER_POS.index(opener_pos), ORDER_POS.index(hero_pos)
    hero = (btn + off[hero_pos]) % 6
    e = env0
    for k in range(hi):  # everyone acting before hero
        seat = order[k]
        if is_terminal_np(e) or current_player_np(e) != seat:
            return None, None
        act = OPEN if k == oi else FOLD
        legal = legal_action_mask_np(e, seat)
        if k == oi and not legal[OPEN]:
            act = next((a for a in range(3, 11) if legal[a]), CALL)  # fallback raise bucket
        e = apply_action_np(e, act)
    if is_terminal_np(e) or current_player_np(e) != hero:
        return None, None
    return e, hero


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=None,
                    help="ADDITIVE: audit this NLCFR001 checkpoint's rows instead of the deployed pkl.")
    args = ap.parse_args()
    if args.checkpoint:
        import sys as _sys
        _sys.path.insert(0, os.path.join(REPO, "scripts"))
        from preflop_strategy_audit import _load_nlcfr_strat
        snap = {bytes(k): np.asarray(v, dtype=np.float64)
                for k, v in _load_nlcfr_strat(args.checkpoint).items()}
        print(f"checkpoint {args.checkpoint}: {len(snap):,} infosets\n", flush=True)
    else:
        dep = os.path.join(REPO, "checkpoints", "serve_blueprint_6max.pkl")
        raw = pickle.load(open(dep, "rb"))
        snap = raw["strat_sum"] if isinstance(raw, dict) and "strat_sum" in raw else raw
        snap = {bytes(k): np.asarray(v, dtype=np.float64) for k, v in snap.items()}
        print(f"deployed: {len(snap):,} infosets\n", flush=True)

    hands = all_169()
    occ = jax.numpy.array([True] * 6)
    print(f"{'spot':16s} | {'avg fold':>8s} {'avg flat':>8s} {'avg 3bet':>8s} {'avg jam':>8s} | "
          f"{'#fold':>5s} {'#flat':>5s} {'#3bet':>5s} {'#jam':>5s}  cov", flush=True)
    for opener, hero_pos in SPOTS:
        M, covn = [], 0
        threebet_hands = []
        for lab, cards in hands:
            acc, n = np.zeros(4), 0
            for d in range(3):
                env0 = to_numpy_state(init_hand(jax.random.PRNGKey(640000 + d), occupancy=occ))
                btn = int(env0.button)
                e, hero = build_vs_open(env0, btn, opener, hero_pos)
                if e is None:
                    continue
                e = e._replace(holes=rig_hero(np.asarray(e.holes), hero, cards))
                p = lookup(snap, e, hero)
                if p is None:
                    continue
                acc += np.array([float(p[FOLD]), float(p[CALL]), float(p[3:ALL_IN].sum()), float(p[ALL_IN])])
                n += 1
            if n == 0:
                continue
            v = acc / n
            M.append(v)
            covn += 1
            if np.argmax(v) in (2, 3):  # 3bet or jam modal
                threebet_hands.append(lab)
        if not M:
            print(f"{opener}->{hero_pos:11s} | (spot not reachable as constructed)", flush=True)
            continue
        M = np.array(M)
        avg = M.mean(axis=0)
        modal = M.argmax(axis=1)
        cnt = [int((modal == k).sum()) for k in range(4)]
        print(f"{opener+'->'+hero_pos:16s} | {avg[0]:8.3f} {avg[1]:8.3f} {avg[2]:8.3f} {avg[3]:8.3f} | "
              f"{cnt[0]:5d} {cnt[1]:5d} {cnt[2]:5d} {cnt[3]:5d}  {covn}/169", flush=True)
        print(f"   3bet/jam-modal ({len(threebet_hands)}): {', '.join(sorted(threebet_hands))}", flush=True)
    print("\nReference (sound): a 3-bettor has a real polarized/linear 3-bet range (~6-12% of hands 3bet, "
          "premiums 3bet/4bet for value, some flats). FLAT-heavy w/ near-zero 3bet = exploitable (never "
          "punished for opening; opener realizes equity freely).", flush=True)


if __name__ == "__main__":
    main()
