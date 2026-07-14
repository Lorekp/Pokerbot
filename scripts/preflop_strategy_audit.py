#!/usr/bin/env python3
"""Full-preflop-tree strategy dump for the DEPLOYED blueprint: for every one of the 169 hand classes at
every position, look up the served action distribution at (a) the RFI node (folded to hero) and (b) facing
a single open (hero in the blinds / behind). Summarizes the OPEN / LIMP / FOLD / JAM shape per position so
we can see mis-shapes across the WHOLE range -- not just one hand.

Deterministic + zero game-variance: it reads the strategy table directly (same env + fingerprint that
matched the deployed key space exactly in make_aa_open_override.py).

  POKERBOT_* env (deployed EMD-K160) set BELOW before importing pokerbot/battery_native.
  python scripts/preflop_strategy_audit.py [--out artifacts/_preflop_audit.json]
"""
import argparse
import json
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
RANKS = "23456789TJQKA"  # index 0..12 -> card = rank_idx*4 + suit (Ac=48)
POSITIONS = ["UTG", "HJ", "CO", "BTN", "SB"]  # RFI positions (folded-to-hero)
SEAT_OFF = {"BTN": 0, "SB": 1, "BB": 2, "UTG": 3, "HJ": 4, "CO": 5}


def all_169():
    """Yield (label, (card_a, card_b)) for every hand class. Pairs c/d; suited both c; offsuit c/d."""
    out = []
    for i in range(12, -1, -1):
        for j in range(12, -1, -1):
            if i == j:
                out.append((RANKS[i] * 2, (i * 4 + 0, i * 4 + 1)))                 # pair
            elif i > j:
                out.append((RANKS[i] + RANKS[j] + "s", (i * 4 + 0, j * 4 + 0)))    # suited
            else:
                out.append((RANKS[j] + RANKS[i] + "o", (j * 4 + 0, i * 4 + 1)))    # offsuit
    # de-dup (i>j suited, j>i offsuit already covers all 169 exactly once)
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
    """Place hero's two hole cards = ``cards`` by in-array swap (preflop: 12 hole slots, board undealt, so
    a swap preserves the multiset -> no duplicate card)."""
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


def rfi_node(env0, btn, pos):
    """Fold everyone before ``pos`` so it is folded-to-hero. Returns (env_at_hero, hero) or (None, None)."""
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


def classify(p):
    """Return (fold, limp, open, jam) mass from an 11-action prob row (RFI semantics: 2=limp/call)."""
    fold = float(p[FOLD])
    limp = float(p[CALL])
    jam = float(p[ALL_IN])
    opn = float(p[3:ALL_IN].sum())  # pot-fraction raises 3..9
    return fold, limp, opn, jam


def _load_nlcfr_strat(path):
    """Parse an NLCFR001 checkpoint's raw strat_sum map ({[u8;16] -> [f64;11]}). Returns the RAW cumulative
    strategy rows (lookup() normalizes + treats all-zero/absent as a miss), so the audit reads an alternate
    (e.g. value-leaf re-solved) snapshot exactly like the deployed pkl. Additive: only used by --checkpoint."""
    import struct
    with open(path, "rb") as f:
        buf = f.read()
    if buf[:8] != b"NLCFR001":
        raise ValueError(f"{path}: bad magic {buf[:8]!r}")
    p = 8
    (_iters,) = struct.unpack_from("<Q", buf, p)
    p += 8

    def _read_map(pp):
        (n,) = struct.unpack_from("<Q", buf, pp)
        pp += 8
        out = {}
        for _ in range(n):
            key = bytes(buf[pp:pp + 16])
            pp += 16
            row = struct.unpack_from("<11d", buf, pp)
            pp += 11 * 8
            out[key] = np.asarray(row, dtype=np.float64)
        return out, pp

    _cum, p = _read_map(p)     # cum_regret (unused here)
    strat, p = _read_map(p)    # strat_sum
    return strat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(REPO, "artifacts", "_preflop_audit.json"))
    ap.add_argument("--deals", type=int, default=3, help="button deals to average the lookup over")
    ap.add_argument("--checkpoint", default=None,
                    help="ADDITIVE: audit this NLCFR001 checkpoint's preflop rows instead of the deployed "
                         "pkl (e.g. a value-leaf re-solve). Unset => deployed serve_blueprint_6max.pkl.")
    ap.add_argument("--pkl", default=None,
                    help="ADDITIVE: audit this serve-snapshot PKL (e.g. the RFI-anchored blueprint) instead "
                         "of the deployed pkl. Mutually exclusive with --checkpoint.")
    args = ap.parse_args()

    if args.pkl:
        raw = pickle.load(open(args.pkl, "rb"))
        s = raw["strat_sum"] if isinstance(raw, dict) and "strat_sum" in raw else raw
        snap = {bytes(k): np.asarray(v, dtype=np.float64) for k, v in s.items()}
        print(f"pkl {args.pkl}: {len(snap):,} infosets", flush=True)
    elif args.checkpoint:
        snap = {bytes(k): np.asarray(v, dtype=np.float64) for k, v in _load_nlcfr_strat(args.checkpoint).items()}
        print(f"checkpoint {args.checkpoint}: {len(snap):,} infosets", flush=True)
    else:
        dep = os.path.join(REPO, "checkpoints", "serve_blueprint_6max.pkl")
        raw = pickle.load(open(dep, "rb"))
        snap = raw["strat_sum"] if isinstance(raw, dict) and "strat_sum" in raw else raw
        snap = {bytes(k): np.asarray(v, dtype=np.float64) for k, v in snap.items()}
        print(f"deployed: {len(snap):,} infosets", flush=True)

    hands = all_169()
    occ = jax.numpy.array([True] * 6)
    # per position: hand -> averaged (fold,limp,open,jam) + coverage count
    result = {pos: {} for pos in POSITIONS}
    miss = {pos: 0 for pos in POSITIONS}
    for pos in POSITIONS:
        for lab, cards in hands:
            accs, ncov = np.zeros(4), 0
            for d in range(args.deals):
                env0 = to_numpy_state(init_hand(jax.random.PRNGKey(730000 + d), occupancy=occ))
                btn = int(env0.button)
                e, hero = rfi_node(env0, btn, pos)
                if e is None:
                    continue
                e = e._replace(holes=rig_hero(np.asarray(e.holes), hero, cards))
                p = lookup(snap, e, hero)
                if p is None:
                    continue
                accs += np.asarray(classify(p))
                ncov += 1
            if ncov == 0:
                miss[pos] += 1
                continue
            result[pos][lab] = (accs / ncov).tolist()

    # Summaries per position: open% (open+jam modal), limp% (limp modal), and the open/limp hand lists.
    print(f"\n{'pos':4s} {'open%':>6s} {'limp%':>6s} {'fold%':>6s}   (of 169 classes, modal action)", flush=True)
    summary = {}
    for pos in POSITIONS:
        rows = result[pos]
        n = len(rows)
        opn = [lab for lab, v in rows.items() if (v[2] + v[3]) == max(v)]
        limp = [lab for lab, v in rows.items() if v[1] == max(v) and v[1] > v[2] + v[3]]
        fold = [lab for lab, v in rows.items() if v[0] == max(v)]
        openpct = 100.0 * len(opn) / n if n else 0.0
        limppct = 100.0 * len(limp) / n if n else 0.0
        foldpct = 100.0 * len(fold) / n if n else 0.0
        summary[pos] = {"n_covered": n, "open_pct": openpct, "limp_pct": limppct,
                        "fold_pct": foldpct, "open_hands": sorted(opn), "limp_hands": sorted(limp)}
        print(f"{pos:4s} {openpct:6.1f} {limppct:6.1f} {foldpct:6.1f}   "
              f"covered={n}/169 miss={miss[pos]}", flush=True)
        if limp:
            print(f"       LIMP-modal hands ({len(limp)}): {', '.join(sorted(limp))}", flush=True)

    # Reference: solid 6-max first-in open frequencies (no limping in a sound RFI strategy).
    print("\nReference (sound 6-max RFI, %VPIP=%PFR, ZERO limp): "
          "UTG ~16  HJ ~20  CO ~26  BTN ~44  SB ~40 (limp 0)", flush=True)
    print("A sound blueprint should have limp% ~ 0 at every position and open% near the reference.", flush=True)

    with open(args.out, "w") as f:
        json.dump({"summary": summary, "per_hand": result, "miss": miss}, f, indent=1)
    print(f"\nwrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
