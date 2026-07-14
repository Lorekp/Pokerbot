#!/usr/bin/env python3
"""Anchor FACING-OPEN (fold/call/3bet) + VS-3BET (fold/call/4bet) on top of the RFI-anchored pkl. Removes
the deployed jam-shove-vs-open degeneracy AND makes the CALLER + 3bettor ranges inside the postflop search
sound (range_reach_full reads these rows). Standard published ranges, position-tiered; SB/BB split with a
WIDE BB defense (fixes the tight-blind-defense steal-exploitability hint). NO jam.

  python scripts/anchor_faceopen.py [--in artifacts/serve_blueprint_anchored.pkl]
                                    [--out artifacts/serve_blueprint_anchored3.pkl]
"""
import argparse
import os
import pickle
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(REPO, "scripts"))
from vs_open_audit import (  # noqa: E402
    ns, bn, EMD, all_169, rig_hero, build_vs_open, ORDER_POS,
    init_hand, to_numpy_state, np, jax,
)
from pokerbot.engine.np_engine import (  # noqa: E402
    apply_action_np, current_player_np, is_terminal_np, legal_action_mask_np,
)

FOLD, CALL, RAISE_B = 0, 2, 7    # RAISE_B = raise bucket 7 (open/3bet/4bet); no jam
RANKS = "23456789TJQKA"
SEAT_OFF = {"BTN": 0, "SB": 1, "BB": 2, "UTG": 3, "HJ": 4, "CO": 5}


def _ri(c):
    return RANKS.index(c)


def expand(spec):
    out = set()
    for tok in spec.replace(" ", "").split(","):
        if not tok:
            continue
        if len(tok) >= 2 and tok[0] == tok[1]:
            lo = _ri(tok[0])
            hi = 12 if tok.endswith("+") else lo
            for i in range(lo, hi + 1):
                out.add(RANKS[i] * 2)
        elif tok.endswith("s+") or tok.endswith("o+"):
            hc, lc, suit = tok[0], tok[1], tok[2]
            h, low = _ri(hc), _ri(lc)
            for j in range(low, h):
                out.add(f"{RANKS[h]}{RANKS[j]}{suit}")
        else:
            out.add(tok)
    return out


TB_RANGES = {   # 3bet (value + wheel-ace bluffs), tiered by OPENER
    "EP": expand("QQ+, AKs, AKo, AQs, A5s, A4s"),
    "MP": expand("JJ+, AKs, AKo, AQs, AJs, KQs, A5s, A4s, A3s"),
    "LP": expand("TT+, AJs+, KQs, AQo, AKo, A5s, A4s, A3s, A2s"),
}
CALL_RANGES = {   # flat; BB defends WIDE (closing, discounted), SB moderate, IP standard
    "IP": expand("22-JJ, ATs+, KTs+, QTs+, JTs, T9s, 98s, 87s, AQo, AJo, KQo"),
    "SB": expand("22-TT, A2s+, K9s+, Q9s+, JTs, T9s, 98s, AJo+, KQo"),
    "BB": expand("22-JJ, A2s+, K3s+, Q6s+, J7s+, T7s+, 96s+, 86s+, 75s+, 64s+, 54s, "
                 "A7o+, K9o+, Q9o+, J9o+, T9o, 98o"),
}
VS3BET_4BET = expand("AA, KK, AKs, A5s")               # 4bet for value + one bluff
VS3BET_CALL = expand("QQ, JJ, TT, AKo, AQs, KQs")      # flat-defend vs a 3bet
TIER = {"UTG": "EP", "HJ": "EP", "CO": "MP", "BTN": "LP", "SB": "LP"}


def seat_of(btn, pos):
    return (btn + SEAT_OFF[pos]) % 6


def key_of(e, hero):
    flat = ns._to_flat(e).tobytes()
    return bytes(bn.fingerprint_fast_emd(flat, int(hero), 160, [], EMD, True))


def _raise_or(e, seat):
    legal = np.asarray(legal_action_mask_np(e, seat), dtype=bool)
    if legal[RAISE_B]:
        return RAISE_B
    for a in range(3, 10):
        if legal[a]:
            return a
    return CALL if legal[CALL] else FOLD


def build_vs_3bet(env0, btn, opener, threebettor):
    """opener opens, threebettor (later) 3bets, folds around -> opener faces the 3bet. Returns (env, opener_seat)."""
    order = [(btn + 3 + i) % 6 for i in range(6)]
    op, tb = seat_of(btn, opener), seat_of(btn, threebettor)
    if order.index(tb) <= order.index(op):
        return None, None
    e = env0
    opened = False
    for _ in range(30):
        if is_terminal_np(e):
            return None, None
        cp = current_player_np(e)
        if cp < 0:
            return None, None
        if cp == op:
            if not opened:
                e = apply_action_np(e, _raise_or(e, cp))
                opened = True
                continue
            return e, cp                       # opener facing the 3bet -> decision node
        if cp == tb and opened:
            e = apply_action_np(e, _raise_or(e, cp))   # the 3bet
            continue
        e = apply_action_np(e, FOLD)
    return None, None


def facing_open_matchups():
    out = []
    for oi, opener in enumerate(ORDER_POS[:-1]):
        for hero_pos in ORDER_POS[oi + 1:]:
            out.append((opener, hero_pos))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default=os.path.join(REPO, "artifacts", "serve_blueprint_anchored.pkl"))
    ap.add_argument("--out", default=os.path.join(REPO, "artifacts", "serve_blueprint_anchored3.pkl"))
    ap.add_argument("--deals", type=int, default=6)
    args = ap.parse_args()

    raw = pickle.load(open(args.inp, "rb"))
    wrapped = isinstance(raw, dict) and "strat_sum" in raw
    snap = raw["strat_sum"] if wrapped else raw
    snap = {bytes(k): np.asarray(v, dtype=np.float64) for k, v in snap.items()}
    print(f"in ({os.path.basename(args.inp)}): {len(snap):,} rows", flush=True)

    hands = all_169()
    occ = jax.numpy.array([True] * 6)
    n_set, n_new = 0, 0

    def apply_node(builder, args_tuple, raise_set, call_set, cntbox):
        nonlocal n_set, n_new
        for lab, cards in hands:
            idx = 0 if lab in raise_set else (1 if lab in call_set else 2)
            bucket = (RAISE_B, CALL, FOLD)[idx]
            row = np.zeros(11, dtype=np.float64)
            row[bucket] = 1.0
            cntbox[idx] += 1
            for d in range(args.deals):
                env0 = to_numpy_state(init_hand(jax.random.PRNGKey(640000 + d), occupancy=occ))
                btn = int(env0.button)
                e, hero = builder(env0, btn, *args_tuple)
                if e is None:
                    continue
                e = e._replace(holes=rig_hero(np.asarray(e.holes), hero, cards))
                k = key_of(e, hero)
                if k not in snap:
                    n_new += 1
                snap[k] = row.copy()
                n_set += 1

    print("\n=== FACING-OPEN (3bet/call/fold) ===", flush=True)
    for opener, hero_pos in facing_open_matchups():
        tier = TIER[opener]
        ctx = "BB" if hero_pos == "BB" else ("SB" if hero_pos == "SB" else "IP")
        t3 = TB_RANGES[tier]
        tc = CALL_RANGES[ctx] - t3
        cnt = [0, 0, 0]
        apply_node(build_vs_open, (opener, hero_pos), t3, tc, cnt)
        n = sum(cnt) // (169 // 169)  # cnt already per-169
        print(f"  {opener:3s}->{hero_pos:3s}[{ctx}]: 3bet={cnt[0]:3d} call={cnt[1]:3d} fold={cnt[2]:3d} "
              f"({100*cnt[0]/169:.0f}/{100*cnt[1]/169:.0f}/{100*cnt[2]/169:.0f}%)", flush=True)

    print("\n=== VS-3BET (4bet/call/fold, opener facing a 3bet) ===", flush=True)
    for oi, opener in enumerate(ORDER_POS[:-1]):
        for tb in ORDER_POS[oi + 1:]:
            cnt = [0, 0, 0]
            apply_node(build_vs_3bet, (opener, tb), VS3BET_4BET, VS3BET_CALL, cnt)
            if cnt[0] + cnt[1] > 0:
                print(f"  {opener:3s} vs {tb:3s} 3bet: 4bet={cnt[0]:3d} call={cnt[1]:3d} fold={cnt[2]:3d}", flush=True)

    print(f"\nrows set: {n_set:,} (keys not previously present: {n_new:,})", flush=True)
    out_obj = {"strat_sum": snap} if wrapped else snap
    with open(args.out, "wb") as f:
        pickle.dump(out_obj, f)
    print(f"wrote {args.out} ({len(snap):,} rows)", flush=True)


if __name__ == "__main__":
    main()
