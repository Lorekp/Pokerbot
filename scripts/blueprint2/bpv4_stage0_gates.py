#!/usr/bin/env python3
"""blueprint-v4 Stage-0 gate evaluator (design doc §6.1, pre-registered — G1-G4, G6 + ladder + M1).

Per arm: walks the checkpoint ladder (iter_*.nlcfr), extracts the canonical preflop nodes with the
arm's key space (summary | fullhist — engine-replayed exact keys via preflop_root_diag), and
evaluates the pre-registered gates:

  G1 monotonicity : UTG < HJ < CO < BTN open%, each step >= -0.5pt tolerance, at 30k AND 40k
  G2 BTN open     : >= 20% at 40k; decline 30k->40k >= -3pt; window-current(35k->40k)/avg >= 0.75
  G3 AA-BTN-limp  : <= 30% at 40k AND non-rising 30k->40k
  G4 BB fold vs BTN open : <= 70% (fullhist: weighted over the arm's own BTN open-size mix)
  G6 jam sanity   : BTN RFI all-in share of open <= 10% at 40k
  M1 (measurement): per-checkpoint strat/cum row counts + blob bytes (the Stage-3 sizing curve)

Checkpoint blobs can be multi-GB under full-hist keying, so rows are STREAMED and only the ~5k
wanted node keys are kept (never a full-table dict). Negative control (the bp3 kill-gate pattern):
extraction in the WRONG key space must read 0 rows — asserted per checkpoint.

Usage:
  python scripts/blueprint2/bpv4_stage0_gates.py --arm H200 \
      --dir runs/native_cfr/bpv4_stage0/H200 --key-mode fullhist \
      --out artifacts/bpv4_stage0_gates_H200.json

PRE-MENU (arm-3): pass --pre-menu arm3 (or env POKERBOT_PRE_MENU=arm3) for an arm trained with the
WIDENED preflop raise menu, engine ids 5/6/7/8/9 = 0.66/0.75/1.0/1.5/2.0 pot (trainer env
POKERBOT_PRE_KEEP="2,3,4,5,6"; RFI raise-to 2.0/2.12/2.5/3.25/4.0bb). This (a) counts ids 5/6 as
opens in every open% / G1 / G2 / G6 aggregate, (b) adds the two BB-response nodes vs the NEW BTN
open sizes (vsopen_BB_200 / vsopen_BB_212), and (c) extends the G4 open-size weighting across all
5 sizes (weighted by the arm's own BTN open-size mix, identical construction to the 3-size form).
DEFAULT IS BYTE-IDENTICAL to the pre-arm-3 instrument (arm-2b gate JSON reproducible with
defaults); off-default runs stamp "pre_menu" into the output JSON so evidence is self-describing
(absence of the field == default 3-size menu).
"""
from __future__ import annotations

import argparse
import json
import mmap
import os
import struct
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, ".."))
import preflop_root_diag as prd  # noqa: E402

NUM_ACTIONS = 11
ROW = 16 + NUM_ACTIONS * 8
FOLD, CHECK, CALL, ALL_IN = 0, 1, 2, 10
RAISES = (7, 8, 9)  # default 3-size menu; the LIVE menu is prd.RAISE_SET (prd.set_pre_menu)
DETAIL_HANDS = ("AA", "KK", "QQ", "AKs", "AKo", "K9s", "A5o", "76s", "72o", "T9s")

# G4 BB-response node -> the BTN open-size action label that weights it (the arm's own open mix).
# Order matters for byte-identical default output: the default 3 sizes keep their historical order.
BB_SIZE_WEIGHTS_DEFAULT = (("vsopen_BB", "B1.0p"), ("vsopen_BB_325", "B1.5p"),
                           ("vsopen_BB_400", "B2.0p"))
BB_SIZE_WEIGHTS_ARM3 = (("vsopen_BB", "B1.0p"), ("vsopen_BB_325", "B1.5p"),
                        ("vsopen_BB_400", "B2.0p"), ("vsopen_BB_200", "B.66p"),
                        ("vsopen_BB_212", "B.75p"))


def bb_size_weights():
    """The active G4 (BB-vs-BTN-open) size-weighting table, per the selected pre-menu."""
    return BB_SIZE_WEIGHTS_ARM3 if prd.PRE_MENU == "arm3" else BB_SIZE_WEIGHTS_DEFAULT


def scan_checkpoint(path: str, wanted: set[bytes]):
    """Stream an NLCFR001 blob; return (iters, n_cum, n_strat, cum_hits, strat_hits)."""
    hits_cum: dict[bytes, tuple] = {}
    hits_strat: dict[bytes, tuple] = {}
    with open(path, "rb") as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        try:
            assert mm[:8] == b"NLCFR001", f"{path}: bad magic"
            (iters,) = struct.unpack_from("<Q", mm, 8)
            p = 16
            counts = []
            for hits in (hits_cum, hits_strat):
                (n,) = struct.unpack_from("<Q", mm, p)
                p += 8
                counts.append(n)
                end = p + n * ROW
                q = p
                while q < end:
                    key = mm[q:q + 16]
                    if key in wanted:
                        hits[bytes(key)] = struct.unpack_from("<11d", mm, q + 16)
                    q += ROW
                p = end
        finally:
            mm.close()
    return iters, counts[0], counts[1], hits_cum, hits_strat


def build_nodes(key_mode: str):
    if key_mode == "fullhist":
        prd.FULL_HIST, prd.HIST_KEY = True, False
        return prd.nodes_fullhist()
    prd.FULL_HIST, prd.HIST_KEY = False, False
    return [(label, pe, b, legal, h, b"") for label, pe, b, legal, h in prd.nodes()]


def wrong_space_nodes(key_mode: str):
    """The OPPOSITE key space's node list (negative control — must read 0 rows)."""
    other = "summary" if key_mode == "fullhist" else "fullhist"
    nl = build_nodes(other)
    # restore the real mode's globals for the actual extraction
    build_nodes(key_mode)
    return other, nl


def node_keys(node_list, key_mode: str):
    prd.FULL_HIST = key_mode == "fullhist"
    prd.HIST_KEY = False
    keys = {}
    for label, pe, betting, legal, hist, fullhist in node_list:
        for cls in range(169):
            keys[(label, cls)] = prd.key_for(pe, cls, betting, hist, fullhist)
    return keys


def extract_nodes(node_list, keys, cum_hits, strat_hits, detail_cls):
    out = {}
    for label, pe, betting, legal, hist, fullhist in node_list:
        avg_by, cur_by = {}, {}
        detail = {}
        for cls in range(169):
            k = keys[(label, cls)]
            srow = strat_hits.get(k)
            crow = cum_hits.get(k)
            if srow is None and crow is None:
                continue
            if srow is not None:
                avg_by[cls] = prd.renorm(list(srow), legal)
            if crow is not None:
                cur_by[cls] = prd.regret_match(list(crow), legal)
            if cls in detail_cls:
                d = {}
                if srow is not None:
                    d["avg"] = {prd.ACTION_NAMES[a]: round(avg_by[cls][a], 4) for a in legal}
                    d["strat_mass"] = round(sum(srow), 1)
                if crow is not None:
                    d["current"] = {prd.ACTION_NAMES[a]: round(cur_by[cls][a], 4) for a in legal}
                detail[prd.class_name(cls)] = d
        out[label] = {
            "avg": prd.agg(avg_by, legal),
            "current": prd.agg(cur_by, legal),
            "detail": detail,
            "classes_present": len(avg_by) or len(cur_by),
        }
    return out


def window_agg(node_list, keys, strat_lo, strat_hi):
    """Row-normalized strat_sum delta agg per node between two checkpoints (window current)."""
    out = {}
    for label, pe, betting, legal, hist, fullhist in node_list:
        by = {}
        for cls in range(169):
            k = keys[(label, cls)]
            r2 = strat_hi.get(k)
            if r2 is None:
                continue
            r1 = strat_lo.get(k)
            d = [max(0.0, r2[a] - (r1[a] if r1 else 0.0)) for a in range(NUM_ACTIONS)]
            by[cls] = prd.renorm(d, legal)
        out[label] = prd.agg(by, legal)
    return out


def open_pct(node_agg) -> float | None:
    a = node_agg.get("avg")
    return None if a is None else 100.0 * a["open_or_raise"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--dir", required=True)
    ap.add_argument("--key-mode", choices=("summary", "fullhist"), required=True)
    ap.add_argument("--pre-menu", choices=("default", "arm3"),
                    default=os.environ.get("POKERBOT_PRE_MENU", "default"),
                    help="preflop raise menu of the arm's trainer: 'default' = 3-size ids 7/8/9 "
                         "(byte-identical historical instrument) | 'arm3' = widened 5-size ids "
                         "5/6/7/8/9 (POKERBOT_PRE_KEEP=2..6). Env POKERBOT_PRE_MENU sets the "
                         "default.")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    prd.set_pre_menu(args.pre_menu)

    ckpts = []
    for name in os.listdir(args.dir):
        if name.startswith("iter_") and name.endswith(".nlcfr"):
            ckpts.append((int(name[5:-6]), os.path.join(args.dir, name)))
    ckpts.sort()
    if not ckpts:
        print(f"FATAL: no iter_*.nlcfr in {args.dir}")
        return 2

    node_list = build_nodes(args.key_mode)
    keys = node_keys(node_list, args.key_mode)
    wanted = set(keys.values())
    neg_name, neg_nodes = wrong_space_nodes(args.key_mode)
    prd.FULL_HIST = neg_name == "fullhist"
    neg_keys = node_keys(neg_nodes, neg_name)
    prd.FULL_HIST = args.key_mode == "fullhist"
    wanted |= set(neg_keys.values())
    detail_cls = {prd.CLASS_BY_NAME[h] for h in DETAIL_HANDS}

    rfi = ["rfi_UTG", "rfi_HJ", "rfi_CO", "rfi_BTN"]
    ladder = {}
    m1 = []
    strat_by_iter = {}
    for iters, path in ckpts:
        t0 = time.time()
        it, n_cum, n_strat, cum_hits, strat_hits = scan_checkpoint(path, wanted)
        assert it == iters, (it, iters)
        neg_present = sum(1 for k in set(neg_keys.values()) if k in strat_hits or k in cum_hits)
        nodes = extract_nodes(node_list, keys, cum_hits, strat_hits, detail_cls)
        strat_by_iter[iters] = {k: v for k, v in strat_hits.items()}
        m1.append({"iters": iters, "cum_rows": n_cum, "strat_rows": n_strat,
                   "blob_bytes": os.path.getsize(path)})
        btn = nodes["rfi_BTN"]
        aa = btn["detail"].get("AA", {}).get("avg", {})
        aa_limp = aa.get("CALL")
        opens = {n: open_pct(nodes[n]) for n in rfi}
        # BB fold vs BTN open (fullhist: weight per-size rows by the arm's own BTN open-size mix;
        # arm3 pre-menu extends the table to all 5 open sizes, identical construction)
        if args.key_mode == "fullhist":
            mix = btn["avg"]["by_action"] if btn["avg"] else {}
            w = {node: mix.get(lbl, 0.0) for node, lbl in bb_size_weights()}
            tot = sum(w.values())
            bb_fold = None
            per_size = {}
            if tot > 0:
                acc = 0.0
                used = 0.0
                for n, wt in w.items():
                    a = nodes[n]["avg"]
                    if a is not None:
                        per_size[n] = a["fold"]
                        acc += wt * a["fold"]
                        used += wt
                bb_fold = acc / used if used > 0 else None
        else:
            a = nodes["vsopen_BB"]["avg"]
            bb_fold = None if a is None else a["fold"]
            per_size = {}
        jam_share = None
        if btn["avg"] and btn["avg"]["open_or_raise"] > 0:
            jam_share = btn["avg"]["allin"] / btn["avg"]["open_or_raise"]
        ladder[iters] = {
            "open_pct": opens, "aa_btn_limp": aa_limp, "bb_fold_vs_btn_open": bb_fold,
            "bb_fold_per_size": per_size, "btn_jam_share_of_open": jam_share,
            "neg_control_rows": neg_present,
            "nodes": nodes,
        }
        print(f"[{args.arm}] iter {iters}: rows(strat)={n_strat:,} "
              f"BTN open={opens['rfi_BTN'] if opens['rfi_BTN'] is None else round(opens['rfi_BTN'],1)}% "
              f"opens={ {k.split('_')[1]: (None if v is None else round(v,1)) for k,v in opens.items()} } "
              f"AA-limp={None if aa_limp is None else round(100*aa_limp,1)}% "
              f"BBfold={None if bb_fold is None else round(100*bb_fold,1)}% "
              f"jam={None if jam_share is None else round(100*jam_share,1)}% "
              f"neg_ctrl={neg_present} ({time.time()-t0:.0f}s)", flush=True)
        assert neg_present == 0, (
            f"NEGATIVE CONTROL FAILED: {neg_present} rows present under the {neg_name} key space")

    # window-current ratio (35k -> 40k) for G2's third clause
    win_ratio = None
    win = None
    if 35000 in strat_by_iter and 40000 in strat_by_iter and 40000 in ladder:
        win = window_agg(node_list, keys, strat_by_iter[35000], strat_by_iter[40000])
        wa = win.get("rfi_BTN")
        avg40 = ladder[40000]["open_pct"]["rfi_BTN"]
        if wa is not None and avg40:
            win_ratio = (100.0 * wa["open_or_raise"]) / avg40

    def g(iters):
        return ladder.get(iters, {})

    def mono_ok(iters):
        o = g(iters).get("open_pct", {})
        if any(o.get(n) is None for n in rfi):
            return None
        return all(o[rfi[i + 1]] - o[rfi[i]] >= -0.5 for i in range(3))

    gates = {}
    l30, l40 = g(30000), g(40000)
    has30, has40 = bool(l30), bool(l40)
    if has30 and has40:
        gates["G1_monotonicity"] = {
            "pass": bool(mono_ok(30000)) and bool(mono_ok(40000)),
            "mono_30k": mono_ok(30000), "mono_40k": mono_ok(40000),
            "opens_30k": l30["open_pct"], "opens_40k": l40["open_pct"],
        }
        btn40 = l40["open_pct"]["rfi_BTN"]
        btn30 = l30["open_pct"]["rfi_BTN"]
        g2 = {
            "btn_open_40k_pct": btn40,
            "decline_30k_40k_pt": None if (btn40 is None or btn30 is None) else btn40 - btn30,
            "window_ratio_35k_40k": win_ratio,
        }
        g2["pass"] = bool(
            btn40 is not None and btn40 >= 20.0
            and g2["decline_30k_40k_pt"] is not None and g2["decline_30k_40k_pt"] >= -3.0
            and (win_ratio is None or win_ratio >= 0.75))
        gates["G2_btn_open"] = g2
        aa40, aa30 = l40["aa_btn_limp"], l30["aa_btn_limp"]
        gates["G3_aa_btn_limp"] = {
            "aa_limp_40k": aa40, "aa_limp_30k": aa30,
            "pass": bool(aa40 is not None and aa40 <= 0.30
                         and (aa30 is None or aa40 <= aa30 + 1e-9)),
        }
        bf = l40["bb_fold_vs_btn_open"]
        gates["G4_bb_fold_vs_btn_open"] = {
            "fold_40k": bf, "per_size_40k": l40["bb_fold_per_size"],
            "pass": bool(bf is not None and bf <= 0.70),
        }
        js = l40["btn_jam_share_of_open"]
        gates["G6_jam_share"] = {"jam_share_40k": js,
                                 "pass": bool(js is not None and js <= 0.10)}
    result = {
        "arm": args.arm, "key_mode": args.key_mode, "dir": args.dir,
        "gates": gates,
        "gates_evaluated_at": [30000, 40000] if (has30 and has40) else "MISSING CHECKPOINTS",
        "ladder": {str(k): {kk: vv for kk, vv in v.items() if kk != "nodes"}
                   for k, v in ladder.items()},
        "detail_40k": {n: ladder[40000]["nodes"][n]["detail"] for n in
                       ("rfi_BTN", "rfi_UTG")} if has40 else None,
        "window_35k_40k_rfi_BTN": (win or {}).get("rfi_BTN") if win else None,
        "m1_rows": m1,
    }
    if args.pre_menu != "default":
        # stamped ONLY off-default: the default JSON stays byte-identical to the pre-arm-3
        # instrument (absence of "pre_menu" == default 3-size menu).
        result["pre_menu"] = args.pre_menu
        result["open_raise_ids"] = list(prd.RAISE_SET)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=1)
    print(f"[{args.arm}] gates: " + "  ".join(
        f"{k}={'PASS' if v.get('pass') else 'FAIL'}" for k, v in gates.items()))
    print(f"[{args.arm}] wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
