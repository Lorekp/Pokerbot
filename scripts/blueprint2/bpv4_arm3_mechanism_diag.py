#!/usr/bin/env python3
"""Arm-3 limp-trap MECHANISM diagnostic (zero-download, ranged) -- WHY does AA-BTN-limp survive?

Extends the amendment sec 23 limp-response diagnostic (_k1_limp_response_diag.py) to the arm-3 15k
blob via the ranged reader (bpv4_ranged_scan) -- the 499.8 GiB checkpoint never touches disk.

Discriminates two mechanisms for the ~45% converged AA-BTN-limp (amendment sec 28):
  H1  PREFLOP GEOMETRY : the blinds OVER-fold to opens (73%) AND UNDER-punish limps (rarely
      iso-raise, mostly complete/check). Then limping is geometrically best: it keeps the
      73%-over-folder in the pot AND is not punished -- opening just wins the blinds. If so the
      defect is the preflop RESPONSE STRUCTURE (a degenerate abstract-game equilibrium), fixable
      preflop, NOT a postflop-abstraction equity-realization deficit.
  H2  POSTFLOP-DRIVEN  : the blinds DO punish limps adequately (iso-raise a lot), yet BTN still
      limps -- so the limp value comes from trained POSTFLOP continuation EV, pointing at the
      postflop abstraction (EMD-K160). (Confirmed/refuted separately by the check-call EV probe.)

Reads, under the arm's exact fullhist+M-RR+arm3-menu key space:
  * the four RFI open nodes + the five vsopen_BB response nodes (BB fold-vs-open, the shape asymmetry
    baseline) -- from the standard node list;
  * NEW limp-response nodes (blinds facing a BTN limp):
      vslimp_SB          SB   [0,0,0,2]      BTN limps, SB to act        -> SB punish/complete/fold
      vslimp_BB_SBfold   BB   [0,0,0,2,0]    ...SB folds, BB to act      -> BB punish/check/fold
      vslimp_BB_SBcall   BB   [0,0,0,2,2]    ...SB completes, BB to act  -> BB punish/check
  The key comparison: BB fold% vs a BTN OPEN (~73%) versus SB/BB raise% (punish) vs a BTN LIMP.
  A large asymmetry (fold-a-lot-to-open, under-punish-limp) IS the H1 geometry.

Usage:
  python scripts/blueprint2/bpv4_arm3_mechanism_diag.py            # arm-3 latest (15000)
  python scripts/blueprint2/bpv4_arm3_mechanism_diag.py --iter 12500
"""
from __future__ import annotations

import argparse
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, ".."))

import bpv4_stage0_gates as g0            # noqa: E402
import bpv4_kill_gate as kg               # noqa: E402
import bpv4_ranged_scan as rs             # noqa: E402
import preflop_root_diag as prd           # noqa: E402

DETAIL_HANDS = ("AA", "KK", "QQ", "AKs", "AKo", "K9s", "A5o", "76s", "72o")

# Blinds' RESPONSE to a BTN limp (engine ids: FOLD=0 CHECK=1 CALL=2, raises 5/6/7/8/9, ALLIN=10;
# button=0, UTG acts first). A limp = CALL(2) at the folded-to BTN. Same construction the sec-23
# arm-2b diagnostic used -- these sequences are legal under any preflop menu (folds + a call only).
LIMP_SEQS = [
    ("vslimp_SB", "SB", [0, 0, 0, 2]),
    ("vslimp_BB_SBfold", "BB", [0, 0, 0, 2, 0]),
    ("vslimp_BB_SBcall", "BB", [0, 0, 0, 2, 2]),
    # BTN's play when its limp gets ISO-RAISED (SB folds, BB iso-raises size 7/8). If the limp range
    # FIGHTS BACK (3-bets/jams) rather than folding, iso-raising is unprofitable -> punishment is
    # suppressed -> the limp equilibrium locks (the sec-23 arm-2b signal: limp range 3-bet 50.6%).
    ("limp_vs_iso7_BTN", "BTN", [0, 0, 0, 2, 0, 7]),
    ("limp_vs_iso8_BTN", "BTN", [0, 0, 0, 2, 0, 8]),
]
LIMP_LABELS = [s[0] for s in LIMP_SEQS]


def _mix(node_agg):
    """(raise_total, fold, call/check, by_action) for a node's avg + current."""
    out = {}
    for kind in ("avg", "current"):
        a = node_agg.get(kind)
        if a is None:
            out[kind] = None
            continue
        by = a.get("by_action", {})
        # CALL/CHECK share = whatever is not fold and not an open/raise
        cc = max(0.0, 1.0 - a["fold"] - a["open_or_raise"])
        out[kind] = {"raise": a["open_or_raise"], "fold": a["fold"], "call_check": cc,
                     "allin": a.get("allin", 0.0), "by_action": by}
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bucket", default=kg.DEFAULT_BUCKET)
    ap.add_argument("--prefix", default=kg.DEFAULT_PREFIX)
    ap.add_argument("--run-subdir", default="bpv4_emd160_arm3")
    ap.add_argument("--region", default=kg.DEFAULT_REGION)
    ap.add_argument("--iter", type=int, default=None, help="ladder point (default: latest)")
    ap.add_argument("--pre-menu", choices=("default", "arm3"),
                    default=os.environ.get("POKERBOT_PRE_MENU", "arm3"))
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = args.out or os.path.join(REPO, "artifacts", f"bpv4_arm3_mechanism_diag_{args.run_subdir}.json")
    prd.set_pre_menu(args.pre_menu)

    # Extend the canonical fullhist sequence table with the limp-response nodes BEFORE node build
    # (they are engine-replayed + legality-asserted exactly like the registered nodes).
    prd.FULLHIST_NODE_SEQS = list(prd.FULLHIST_NODE_SEQS) + LIMP_SEQS

    node_list = g0.build_nodes("fullhist")            # standard nodes + the 3 limp-response nodes
    keys = g0.node_keys(node_list, "fullhist")
    neg_name, neg_nodes = g0.wrong_space_nodes("fullhist")
    neg_keys = g0.node_keys(neg_nodes, neg_name)
    g0.build_nodes("fullhist")
    prd.set_pre_menu(args.pre_menu)
    g0.build_nodes("fullhist")
    detail_cls = {prd.CLASS_BY_NAME[h] for h in DETAIL_HANDS}
    wanted = set(keys.values()) | set(neg_keys.values())

    # --- resolve latest ladder point + ranged scan ---------------------------------------------------
    import boto3  # noqa: PLC0415
    try:
        import urllib3
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    except Exception:
        pass
    s3 = boto3.client("s3", region_name=args.region, verify=False)
    ladder = kg._list_ladder(s3, args.bucket, f"{args.prefix}/{args.run_subdir}")
    if not ladder:
        print("ERROR: empty ladder"); return 2
    target = args.iter if args.iter is not None else ladder[-1][0]
    match = [(i, k) for i, k in ladder if i == target]
    if not match:
        print(f"ERROR: iter {target} not in ladder {[i for i, _ in ladder]}"); return 2
    it_want, key = match[0]
    print(f"[mech] ranged-scanning iter {it_want}  (s3://{args.bucket}/{key})", flush=True)
    reader = rs.S3RangedReader(args.bucket, key, args.region)
    it, n_cum, n_strat, cum_hits, strat_hits = rs.scan_ranged(reader, wanted, workers=args.workers)
    neg_present = sum(1 for k in set(neg_keys.values()) if k in strat_hits or k in cum_hits)
    print(f"[mech] iter {it}: cum_rows={n_cum:,} strat_rows={n_strat:,} neg_ctrl={neg_present} "
          f"[{reader.n_get:,} GETs, {reader.n_bytes/1e6:.0f} MB]", flush=True)
    if neg_present != 0:
        print(f"ABORT: neg control {neg_present} != 0 (wrong key space)"); return 2

    nodes = g0.extract_nodes(node_list, keys, cum_hits, strat_hits, detail_cls)

    # --- OPEN side (the shape asymmetry baseline) ----------------------------------------------------
    rfi = ["rfi_UTG", "rfi_HJ", "rfi_CO", "rfi_BTN"]
    opens = {n: g0.open_pct(nodes[n]) for n in rfi}
    bb_fold, per_size = kg._bb_fold_weighted(nodes)
    aa_limp_avg = kg._aa_limp(nodes)
    print("\n=== OPEN side (baseline) ===")
    print("  opens: " + "  ".join(f"{n.split('_')[1]}={'NA' if opens[n] is None else f'{opens[n]:.1f}%'}"
                                   for n in rfi))
    print(f"  AA-BTN-limp (avg) = {'NA' if aa_limp_avg is None else f'{aa_limp_avg*100:.1f}%'}")
    print(f"  BB fold vs BTN OPEN (weighted) = {'NA' if bb_fold is None else f'{bb_fold*100:.1f}%'}  "
          f"[per size: {{ {', '.join(f'{k}={v*100:.0f}%' for k,v in per_size.items())} }}]")

    # --- LIMP side (the mechanism) -------------------------------------------------------------------
    print("\n=== LIMP-RESPONSE side (do the blinds punish a BTN limp?) ===")
    limp = {}
    for lbl in LIMP_LABELS:
        m = _mix(nodes.get(lbl, {}))
        limp[lbl] = {"mix": m, "detail": nodes.get(lbl, {}).get("detail", {}),
                     "classes_present": nodes.get(lbl, {}).get("classes_present")}
        a = m.get("avg")
        if a is None:
            print(f"  {lbl:18s}: NO ROWS (node unvisited?)"); continue
        print(f"  {lbl:18s}: PUNISH(raise)={a['raise']*100:5.1f}%  fold={a['fold']*100:5.1f}%  "
              f"call/check={a['call_check']*100:5.1f}%  allin={a['allin']*100:.1f}%  "
              f"(classes={limp[lbl]['classes_present']})")

    # Aggregate blind punish rate vs a limp: SB raise + (weighted BB raise after SB fold/complete).
    sb = limp["vslimp_SB"]["mix"].get("avg")
    punish_note = ""
    if sb is not None:
        sb_raise = sb["raise"]
        punish_note = (f"SB iso-raises a BTN limp {sb_raise*100:.1f}% of the time"
                       f" (vs BB folding {bb_fold*100:.0f}% to a BTN OPEN)")
        print(f"\n  ASYMMETRY: {punish_note}")

    print("\n=== READING ===")
    print("  H1 (preflop geometry) if: blinds UNDER-punish the limp (low SB/BB raise%, high")
    print("     call/check%) while over-folding to opens -> limp keeps the over-folder in, unpunished.")
    print("  H2 (postflop-driven)  if: blinds DO iso-raise limps a lot yet BTN still limps ~45%.")

    with open(out, "w", encoding="utf-8") as f:
        json.dump({
            "iters": it, "neg_control_rows": neg_present, "pre_menu": args.pre_menu,
            "open_side": {"opens_pct": opens, "aa_btn_limp_avg": aa_limp_avg,
                          "bb_fold_vs_open_weighted": bb_fold, "bb_fold_per_size": per_size},
            "limp_response": limp,
            "ranged_gets": reader.n_get, "ranged_mb": round(reader.n_bytes / 1e6, 1),
        }, f, indent=2, default=lambda o: None)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
