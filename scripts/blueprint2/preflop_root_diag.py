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


def key_for(pos_enum: int, cls: int, betting: bytes, hist: bytes = b"\x00\x00\x00") -> bytes:
    msg = b"w" + struct.pack("<i", pos_enum * 169 + cls)
    if HIST_KEY:
        msg += b"H" + hist
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
    for label, pe, betting, legal, hist in node_list:
        avg_by, cur_by = {}, {}
        detail = {}
        for cls in range(169):
            k = key_for(pe, cls, betting, hist)
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
    for label, pe, betting, legal, hist in node_list:
        by = {}
        for cls in range(169):
            k = key_for(pe, cls, betting, hist)
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
    args = ap.parse_args()
    global HIST_KEY
    HIST_KEY = bool(args.hist_key)

    detail = set() if args.no_detail else {
        CLASS_BY_NAME[h] for h in args.hands.split(",") if h in CLASS_BY_NAME}
    node_list = nodes()
    results = {"meta": {
        "nodes": {label: {"pos": pe, "betting6": list(b), "hist3": list(h),
                          "legal": [ACTION_NAMES[a] for a in legal]}
                  for label, pe, b, legal, h in node_list},
        "hist_key": HIST_KEY,
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
