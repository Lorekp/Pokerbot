#!/usr/bin/env python3
"""Preflop mispricing diagnosis: direct-key extraction of AVG vs CURRENT preflop policy
from NLCFR001 checkpoints at exact canonical nodes (no pokerbot imports, no env traps).

Key facts this mirrors EXACTLY (verified against source 2026-07-03):
  * fingerprint.rs: preflop key = blake2b16(b"w" + LE_i32(pos_enum*169 + class) + b"S" + betting6)
    betting6 = [street, pot_b, tocall_b, spr_b, is_button, facing_agg]
    POT_EDGES=(3,6,12,25,50,100,200) bb; TOCALL_EDGES=(.01,.34,.67,1.01,2.01) of to_call/pot;
    SPR_EDGES=(1,2.5,5,10,20) of eff_stack/pot. facing_agg=0 at RFI (core.rs init last_aggressor=-1).
  * core.rs: 100bb stacks (10_000 chips), SB 50 / BB 100, no ante/rake.
    RFI raise-to amounts (keep-list buckets 4,5,6 = 1.0/1.5/2.0 pot): 2.5 / 3.25 / 4bb (+ 100bb jam).
  * trainer.rs: strat_sum accumulation ss[a] += t * sigma_t(a)  (linear averaging)
    -> avg strategy = row-normalize(strat_sum); window current policy = row-normalize(ss2 - ss1).
  * CFR+ cum_regret rows are positive-floored; current policy = regret-match(max(0, r)).

Nodes (6-max, all seats 100bb):
  RFI UTG/HJ/CO/BTN (pot 1.5bb, to_call 1bb -> tocall_b 2), RFI SB (to_call .5bb -> tocall_b 1).
  vs-OPEN (a 2.5bb open folded to hero; pot 4bb -> pot_b 1, tocall_b 2, facing_agg 1):
     BB (pos 5), SB (pos 4), BTN (pos 3, is_button=1). NOTE the trainer key CANNOT distinguish the
     opener's position or open size (2.5-4bb all alias to the same 6 bytes) -- one row serves all.
  vs-3BET (hero opened 2.5, got 3-bet to ~7.5-9.5bb: pot_b 2, tocall_b 3, facing_agg 1): BTN.

Usage:
  python scripts/blueprint2/preflop_root_diag.py --out artifacts/x.json ckpt1.nlcfr [ckpt2.nlcfr ...]
  --deltas : also emit window current policy between consecutive checkpoints (sorted by iters)
  --hands  : comma list of extra hand classes to detail (default premium+consensus-open probes)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, ".."))
from f32_regret_strategy_delta import load_checkpoint  # noqa: E402

NUM_ACTIONS = 11
FOLD, CHECK, CALL, ALL_IN = 0, 1, 2, 10
ACTION_NAMES = ["FOLD", "CHECK", "CALL", "B1/3p", "B1/2p", "B.66p",
                "B.75p", "B1.0p", "B1.5p", "B2.0p", "ALLIN"]
RAISE_SET = (7, 8, 9)          # keep-list (4,5,6) -> actions 7/8/9 = 1.0/1.5/2.0 pot
POS = {"UTG": 0, "HJ": 1, "CO": 2, "BTN": 3, "SB": 4, "BB": 5}
_RANKS = "23456789TJQKA"


def class_name(cls: int) -> str:
    if cls < 13:
        return _RANKS[cls] * 2
    tri = (cls - 13) % 78
    suited = cls < 91
    # invert tri = hr*(hr-1)/2 + lr
    hr = 1
    while (hr + 1) * hr // 2 <= tri:
        hr += 1
    lr = tri - hr * (hr - 1) // 2
    return _RANKS[hr] + _RANKS[lr] + ("s" if suited else "o")


CLASS_BY_NAME = {class_name(c): c for c in range(169)}


def combos(cls: int) -> int:
    return 6 if cls < 13 else (4 if cls < 91 else 12)


HIST_KEY = False  # set by --hist-key: POKERBOT_PREFLOP_HIST_KEY=1 experiment key space
FULL_HIST = False  # set by --full-hist: POKERBOT_FULLHIST_KEY=1 (blueprint-v4 Stage 0) key space


def key_for(pos_enum: int, cls: int, betting: bytes, hist: bytes = b"\x00\x00\x00",
            fullhist: bytes = b"") -> bytes:
    msg = b"w" + struct.pack("<i", pos_enum * 169 + cls)
    if HIST_KEY:
        msg += b"H" + hist
    if FULL_HIST:
        # blueprint-v4 Stage-0 A segment: street byte + raw engine action-id bytes (the exact
        # fingerprint.rs::push_full_hist_bytes payload; the b"S"+betting6 tail is kept).
        msg += b"A" + fullhist
    msg += b"S" + betting
    return hashlib.blake2b(msg, digest_size=16).digest()


# ---- canonical nodes: (label, pos_enum, betting6, legal action set, hist3) ---------------------
# hist3 = [n_calls_before_first_raise, n_raises, first_raiser_pos_enum + 1] (0 = no raiser).
# Ignored (all nodes alias) unless --hist-key.
def nodes():
    H0 = bytes((0, 0, 0))
    out = []
    # RFI: street0, pot 1.5bb->pot_b0, spr 99/1.5=66->b5, facing_agg 0.
    for name in ("UTG", "HJ", "CO", "BTN"):
        b = bytes((0, 0, 2, 5, 1 if name == "BTN" else 0, 0))
        out.append((f"rfi_{name}", POS[name], b, (FOLD, CALL) + RAISE_SET + (ALL_IN,), H0))
    # SB RFI: to_call .5/1.5=0.333 -> tocall_b 1.
    out.append(("rfi_SB", POS["SB"], bytes((0, 0, 1, 5, 0, 0)), (FOLD, CALL) + RAISE_SET + (ALL_IN,), H0))
    # BTN after ONE limper (same 6-byte betting key as rfi_BTN; distinct only under --hist-key).
    out.append(("BTN_after_limp", POS["BTN"], bytes((0, 0, 2, 5, 1, 0)),
                (FOLD, CALL) + RAISE_SET + (ALL_IN,), bytes((1, 0, 0))))
    # vs a single 2.5bb open, folded to hero: pot 4bb->pot_b1, tocall_b2, spr 97.5/4=24->b5, agg1.
    out.append(("vsopen_BB", POS["BB"], bytes((0, 1, 2, 5, 0, 1)), (FOLD, CALL) + RAISE_SET + (ALL_IN,),
                bytes((0, 1, POS["BTN"] + 1))))
    out.append(("vsopen_BB_vsUTG", POS["BB"], bytes((0, 1, 2, 5, 0, 1)),
                (FOLD, CALL) + RAISE_SET + (ALL_IN,), bytes((0, 1, POS["UTG"] + 1))))
    out.append(("vsopen_SB", POS["SB"], bytes((0, 1, 2, 5, 0, 1)), (FOLD, CALL) + RAISE_SET + (ALL_IN,),
                bytes((0, 1, POS["BTN"] + 1))))
    out.append(("vsopen_BTN", POS["BTN"], bytes((0, 1, 2, 5, 1, 1)), (FOLD, CALL) + RAISE_SET + (ALL_IN,),
                bytes((0, 1, POS["CO"] + 1))))
    # BTN opened 2.5, BB 3-bet to 7.5bb: pot 50+250+750=10.5bb->pot_b2; to_call 500/1050=.476->b2;
    # spr 92.5/10.5=8.8->b3; also the (tocall_b3, spr 4/5) variants inspect_preflop_nodes used.
    out.append(("vs3bet_BTN_a", POS["BTN"], bytes((0, 2, 2, 3, 1, 1)), (FOLD, CALL) + RAISE_SET + (ALL_IN,),
                bytes((0, 2, POS["BTN"] + 1))))
    out.append(("vs3bet_BTN_b", POS["BTN"], bytes((0, 2, 3, 4, 1, 1)), (FOLD, CALL) + RAISE_SET + (ALL_IN,),
                bytes((0, 2, POS["BTN"] + 1))))
    return out


# ---- full-hist nodes (blueprint-v4 Stage 0): ENGINE-REPLAYED, exact by construction -----------
# The bp3 lesson (kill-gate negative-control pattern): wrong-key-space extraction silently reads 0
# rows. Under FULL action-history keying the node key needs the exact (street byte + action-id
# bytes) payload AND the exact betting6 of the replayed state, so instead of hand-computing them we
# REPLAY each canonical action sequence through the pure-Python engine (game-flow states only) and
# read betting6/hist off the resulting state. Imports pokerbot lazily (only in --full-hist mode).
# Node sequences use engine action ids: FOLD=0, CALL=2, sized raises 7/8/9 (keep-list buckets
# 4/5/6 = 1.0/1.5/2.0 pot -> RFI raise-to 2.5/3.25/4bb), ALL_IN=10. button=0 => seats:
# 0=BTN 1=SB 2=BB 3=UTG 4=HJ 5=CO; preflop first actor = UTG (seat 3).
FULLHIST_NODE_SEQS = [
    # (label, hero position name, action sequence from hand start)
    ("rfi_UTG", "UTG", []),
    ("rfi_HJ", "HJ", [0]),
    ("rfi_CO", "CO", [0, 0]),
    ("rfi_BTN", "BTN", [0, 0, 0]),
    ("rfi_SB", "SB", [0, 0, 0, 0]),
    # BTN after exactly one limper (three distinct rows under full-hist keying):
    ("BTN_after_limp_UTG", "BTN", [2, 0, 0]),
    ("BTN_after_limp_HJ", "BTN", [0, 2, 0]),
    ("BTN_after_limp_CO", "BTN", [0, 0, 2]),
    # vs a single sized open, folded to hero (per-size rows under full-hist keying):
    ("vsopen_BB", "BB", [0, 0, 0, 7, 0]),          # BTN opens 2.5bb
    ("vsopen_BB_325", "BB", [0, 0, 0, 8, 0]),      # BTN opens 3.25bb
    ("vsopen_BB_400", "BB", [0, 0, 0, 9, 0]),      # BTN opens 4.0bb
    ("vsopen_BB_vsUTG", "BB", [7, 0, 0, 0, 0]),    # UTG opens 2.5bb
    ("vsopen_SB", "SB", [0, 0, 0, 7]),             # BTN opens 2.5bb, SB to act
    ("vsopen_BTN", "BTN", [0, 0, 7]),              # CO opens 2.5bb, BTN to act
    # hero (BTN) opened 2.5bb, SB folds, BB 3-bets (two 3-bet sizes):
    ("vs3bet_BTN_3b7", "BTN", [0, 0, 0, 7, 0, 7]),
    ("vs3bet_BTN_3b8", "BTN", [0, 0, 0, 7, 0, 8]),
]


def nodes_fullhist():
    """Replay each canonical sequence -> (label, pos_enum, betting6, legal, hist3, fullhist)."""
    import jax
    import numpy as np
    from pokerbot.engine.jax_engine import init_hand
    from pokerbot.engine.np_engine import (
        apply_action_np, is_chance_node_np, legal_action_mask_np, sample_chance_np,
        to_numpy_state,
    )

    pot_edges = (3.0, 6.0, 12.0, 25.0, 50.0, 100.0, 200.0)
    tocall_edges = (0.01, 0.34, 0.67, 1.01, 2.01)
    spr_edges = (1.0, 2.5, 5.0, 10.0, 20.0)

    def bucket(x, edges):
        b = 0
        for e in edges:
            if x >= e:
                b += 1
            else:
                break
        return b

    def betting6(env, seat):
        committed = np.asarray(env.committed_street)
        seat_status = np.asarray(env.seat_status)
        stacks = np.asarray(env.stacks)
        pot = int(np.asarray(env.pot_dead)) + int(committed.sum())
        to_call = max(0, int(np.asarray(env.current_bet)) - int(committed[seat]))
        live = [int(stacks[s]) for s in range(6) if int(seat_status[s]) not in (0, 1)]
        eff = (min(live) if live else 0) / 100.0
        pot_bb = pot / 100.0
        last_agg = int(np.asarray(env.last_aggressor))
        return bytes((
            int(np.asarray(env.street)) & 7,
            bucket(pot_bb, pot_edges),
            bucket(to_call / max(pot, 1), tocall_edges),
            bucket(eff / max(pot_bb, 1e-6), spr_edges),
            1 if seat == int(np.asarray(env.button)) else 0,
            1 if (last_agg >= 0 and last_agg != seat) else 0,
        ))

    pos_by_offset = (3, 4, 5, 0, 1, 2)
    allowed_pre = {0, 1, 2, 7, 8, 9, 10}  # keep-list (4,5,6) trainer mask over engine ids
    out = []
    for label, pos_name, seq in FULLHIST_NODE_SEQS:
        env = to_numpy_state(init_hand(jax.random.PRNGKey(0), button=0))
        for a in seq:
            while is_chance_node_np(env):
                env = sample_chance_np(env)
            env = apply_action_np(env, int(a))
        while is_chance_node_np(env):
            env = sample_chance_np(env)
        seat = int(env.action_seat)
        pe = pos_by_offset[(seat - int(np.asarray(env.button))) % 6]
        assert pe == POS[pos_name], (
            f"{label}: replay landed on pos {pe}, expected {pos_name}={POS[pos_name]}")
        assert int(np.asarray(env.street)) == 0, f"{label}: replay left preflop"
        hl = max(0, min(int(np.asarray(env.hist_len)), 64))
        act = np.asarray(env.hist_action)
        assert [int(act[i]) for i in range(hl)] == [int(a) for a in seq], (
            f"{label}: recorded hist {[int(act[i]) for i in range(hl)]} != applied {seq}")
        fullhist = bytes([int(np.asarray(env.street)) & 0xFF]
                         + [int(act[i]) & 0xFF for i in range(hl)])
        mask = legal_action_mask_np(env, seat)
        legal = tuple(a for a in np.flatnonzero(mask) if int(a) in allowed_pre)
        out.append((label, pe, betting6(env, seat), tuple(int(a) for a in legal),
                    bytes((0, 0, 0)), fullhist))
    return out


def renorm(row, legal):
    tot = sum(row[a] for a in legal)
    if tot <= 0.0:
        return {a: 1.0 / len(legal) for a in legal}
    return {a: row[a] / tot for a in legal}


def regret_match(row, legal):
    pos = {a: max(0.0, row[a]) for a in legal}
    return renorm([pos.get(a, 0.0) if a in legal else 0.0 for a in range(NUM_ACTIONS)], legal)


def agg(policy_by_class, legal):
    """Combo-weighted aggregate over classes present. Returns summary dict."""
    tot_w = 0.0
    acc = {a: 0.0 for a in legal}
    for cls, pol in policy_by_class.items():
        w = combos(cls)
        tot_w += w
        for a in legal:
            acc[a] += w * pol[a]
    if tot_w == 0:
        return None
    m = {a: acc[a] / tot_w for a in legal}
    raise_f = sum(m.get(a, 0.0) for a in RAISE_SET)
    return {
        "n_classes": len(policy_by_class),
        "fold": round(m.get(FOLD, 0.0), 4),
        "limp_call": round(m.get(CALL, 0.0), 4),
        "open_or_raise": round(raise_f + m.get(ALL_IN, 0.0), 4),
        "raise_sized": round(raise_f, 4),
        "allin": round(m.get(ALL_IN, 0.0), 4),
        "by_action": {ACTION_NAMES[a]: round(m[a], 4) for a in legal},
    }


def extract(ck, node_list, detail_classes):
    """Per checkpoint: per node -> avg + current aggregates and per-hand detail."""
    strat, cum = ck["strat"], ck["cum"]
    out = {}
    for label, pe, betting, legal, hist, fullhist in node_list:
        avg_by, cur_by = {}, {}
        detail = {}
        for cls in range(169):
            k = key_for(pe, cls, betting, hist, fullhist)
            srow = strat.get(k)
            crow = cum.get(k)
            if srow is None and crow is None:
                continue
            if srow is not None:
                avg_by[cls] = renorm(list(srow), legal)
            if crow is not None:
                cur_by[cls] = regret_match(list(crow), legal)
            if cls in detail_classes:
                d = {}
                if srow is not None:
                    d["strat_sum_raw"] = [round(srow[a], 3) for a in legal]
                    d["avg"] = {ACTION_NAMES[a]: round(avg_by[cls][a], 4) for a in legal}
                if crow is not None:
                    d["cum_regret"] = [round(crow[a], 1) for a in legal]
                    d["current"] = {ACTION_NAMES[a]: round(cur_by[cls][a], 4) for a in legal}
                d["legal_order"] = [ACTION_NAMES[a] for a in legal]
                detail[class_name(cls)] = d
        out[label] = {
            "avg": agg(avg_by, legal),
            "current_regret_matched": agg(cur_by, legal),
            "detail": detail,
        }
    return out


def window_delta(ck1, ck2, node_list):
    """Row-normalized strat_sum delta = (t-weighted) CURRENT policy averaged over (t1, t2]."""
    out = {}
    for label, pe, betting, legal, hist, fullhist in node_list:
        by = {}
        for cls in range(169):
            k = key_for(pe, cls, betting, hist, fullhist)
            r1 = ck1["strat"].get(k)
            r2 = ck2["strat"].get(k)
            if r2 is None:
                continue
            d = [max(0.0, r2[a] - (r1[a] if r1 else 0.0)) for a in range(NUM_ACTIONS)]
            by[cls] = renorm(d, legal)
        out[label] = agg(by, legal)
    return out


DEFAULT_DETAIL = "AA,KK,AKs,AKo,QQ,K9s,A5o,76s,J9o,T9o,K9o,A2s,72o,QTo,98s"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoints", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--deltas", action="store_true")
    ap.add_argument("--hands", default=DEFAULT_DETAIL)
    ap.add_argument("--no-detail", action="store_true")
    ap.add_argument("--hist-key", action="store_true",
                    help="checkpoint was trained with POKERBOT_PREFLOP_HIST_KEY=1")
    ap.add_argument("--full-hist", action="store_true",
                    help="checkpoint was trained with POKERBOT_FULLHIST_KEY=1 (blueprint-v4 "
                         "Stage 0); nodes are engine-replayed exact-history keys")
    args = ap.parse_args()
    global HIST_KEY, FULL_HIST
    HIST_KEY = bool(args.hist_key)
    FULL_HIST = bool(args.full_hist)
    assert not (HIST_KEY and FULL_HIST), "--hist-key and --full-hist are mutually exclusive"

    detail = set() if args.no_detail else {
        CLASS_BY_NAME[h] for h in args.hands.split(",") if h in CLASS_BY_NAME}
    if FULL_HIST:
        node_list = nodes_fullhist()
    else:
        node_list = [(label, pe, b, legal, h, b"") for label, pe, b, legal, h in nodes()]
    results = {"meta": {
        "nodes": {label: {"pos": pe, "betting6": list(b), "hist3": list(h),
                          "fullhist": list(fh),
                          "legal": [ACTION_NAMES[a] for a in legal]}
                  for label, pe, b, legal, h, fh in node_list},
        "hist_key": HIST_KEY,
        "full_hist": FULL_HIST,
        "open_sizes_bb_at_rfi": {"B1.0p": 2.5, "B1.5p": 3.25, "B2.0p": 4.0, "ALLIN": 100.0,
                                 "SB": [2.0, 2.75, 3.5]},
        "current=regret-matched cum_regret (CFR+ positive floor)": True,
        "avg=row-normalized strat_sum (linear averaging, weight t)": True,
    }, "checkpoints": []}

    loaded = []
    for path in args.checkpoints:
        ck = load_checkpoint(path)
        loaded.append((path, ck))
        print(f"[diag] {path}: iters={ck['iters_done']} strat_rows={len(ck['strat'])}", flush=True)
        results["checkpoints"].append({
            "path": path,
            "iters": ck["iters_done"],
            "nodes": extract(ck, node_list, detail),
        })

    if args.deltas and len(loaded) > 1:
        loaded.sort(key=lambda pc: pc[1]["iters_done"])
        results["windows"] = []
        for (p1, c1), (p2, c2) in zip(loaded, loaded[1:]):
            results["windows"].append({
                "from_iters": c1["iters_done"], "to_iters": c2["iters_done"],
                "nodes": window_delta(c1, c2, node_list),
            })

    with open(args.out, "w") as f:
        json.dump(results, f, indent=1)
    print(f"[diag] wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
