#!/usr/bin/env python3
"""Run the bpv4 mid-run kill gate on a HUGE checkpoint via RANGED reads -- no full download.

Identical instrument to bpv4_kill_gate.py (same nodes, keys, neg-control, pre-menu guard, and the
same `evaluate`), but the rows are pulled by ranged binary search (bpv4_ranged_scan.scan_ranged)
instead of downloading the 400-500 GB blob and mmap-scanning it. This is the ONLY way to read the
arm-3 iter_015000 shape: that checkpoint is 499.8 GiB and does not fit on a 283 GB drive.

It ALSO prints the CURRENT-iterate (regret-matched) AA-BTN-limp alongside the gate-canonical AVERAGE
one: the current iterate is the leading indicator of whether the widened arm-3 menu is pulling the
AA limp down (the average lags). The pre-registered VERDICT is unchanged -- it is the average, via
the shared `evaluate`.

Usage (arm-3 latest = iter 15000):
  python scripts/blueprint2/bpv4_ranged_gate.py --run-subdir bpv4_emd160_arm3 --pre-menu arm3
  python scripts/blueprint2/bpv4_ranged_gate.py --iter 12500 --no-prior      # pin a ladder point
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
import bpv4_kill_gate as kg               # noqa: E402  (reuse the trusted instrument's pure fns)
import bpv4_ranged_scan as rs             # noqa: E402
import preflop_root_diag as prd           # noqa: E402


def _extract_ranged(reader, node_list, keys, neg_keys, detail_cls, arm3_probe, workers):
    """Mirror of bpv4_kill_gate._extract but sourcing rows from a ranged reader."""
    probe_keys = set()
    for lst in arm3_probe.values():
        probe_keys |= set(lst)
    wanted = set(keys.values()) | set(neg_keys.values()) | probe_keys
    it, n_cum, n_strat, cum_hits, strat_hits = rs.scan_ranged(reader, wanted, workers=workers)
    nodes = g0.extract_nodes(node_list, keys, cum_hits, strat_hits, detail_cls)
    neg_present = sum(1 for k in set(neg_keys.values()) if k in strat_hits or k in cum_hits)
    menu = kg._menu_read(arm3_probe, cum_hits, strat_hits)
    m1 = {"iters": it, "cum_rows": n_cum, "strat_rows": n_strat, "blob_bytes": reader.size()}
    return it, nodes, m1, neg_present, menu


def _current_aa_limp(nodes):
    """The CURRENT-iterate (regret-matched) AA-BTN-limp share -- the leading indicator."""
    aa = nodes["rfi_BTN"]["detail"].get("AA", {}).get("current", {})
    return aa.get("CALL")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bucket", default=kg.DEFAULT_BUCKET)
    ap.add_argument("--prefix", default=kg.DEFAULT_PREFIX)
    ap.add_argument("--run-subdir", default="bpv4_emd160_arm3")
    ap.add_argument("--region", default=kg.DEFAULT_REGION)
    ap.add_argument("--iter", type=int, default=None, help="pin a ladder point (default: latest)")
    ap.add_argument("--prior-iter", type=int, default=None,
                    help="trend baseline (default: the preceding ladder point)")
    ap.add_argument("--no-prior", action="store_true", help="skip the trend baseline read")
    ap.add_argument("--pre-menu", choices=("default", "arm3"),
                    default=os.environ.get("POKERBOT_PRE_MENU", "arm3"),
                    help="arm-3 is a widened-menu run -> default here is 'arm3' (unlike the kill gate)")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--mono-tol-pt", type=float, default=0.5)
    ap.add_argument("--btn-floor-pct", type=float, default=20.0)
    ap.add_argument("--aa-limp-cap", type=float, default=0.30)
    ap.add_argument("--k1a-aa-cap", type=float, default=0.50)
    ap.add_argument("--k1-hard-iters", type=int, default=20000)
    ap.add_argument("--bb-fold-cap", type=float, default=0.70)
    ap.add_argument("--jam-cap", type=float, default=0.10)
    ap.add_argument("--decline-floor-pt", type=float, default=-3.0)
    ap.add_argument("--trend-hard-iters", type=int, default=30000)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = args.out or os.path.join(REPO, "artifacts", f"bpv4_ranged_gate_{args.run_subdir}.json")
    prd.set_pre_menu(args.pre_menu)

    # --- nodes / keys / neg-control / arm-3 probe (identical construction to bpv4_kill_gate.main) ---
    node_list = g0.build_nodes("fullhist")
    keys = g0.node_keys(node_list, "fullhist")
    neg_name, neg_nodes = g0.wrong_space_nodes("fullhist")
    neg_keys = g0.node_keys(neg_nodes, neg_name)
    g0.build_nodes("fullhist")
    arm3_probe = kg._arm3_probe_keys()
    prd.set_pre_menu(args.pre_menu)
    g0.build_nodes("fullhist")
    detail_cls = {prd.CLASS_BY_NAME[h] for h in kg.DETAIL_HANDS}

    # --- resolve the ladder point(s) on S3 -----------------------------------------------------------
    import boto3  # noqa: PLC0415
    try:
        import urllib3
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    except Exception:
        pass
    s3 = boto3.client("s3", region_name=args.region, verify=False)
    run_prefix = f"{args.prefix}/{args.run_subdir}"
    ladder = kg._list_ladder(s3, args.bucket, run_prefix)
    if not ladder:
        print(f"ERROR: no iter_XXXXXX.nlcfr ladder under s3://{args.bucket}/{run_prefix}/checkpoints/")
        return 2
    print(f"[ranged-gate] ladder: {[i for i, _ in ladder]}")
    if args.iter is not None:
        match = [(i, k) for i, k in ladder if i == args.iter]
        if not match:
            print(f"ERROR: iter {args.iter} not in ladder {[i for i, _ in ladder]}")
            return 2
        idx = ladder.index(match[0])
    else:
        idx = len(ladder) - 1
    latest_iters, latest_key = ladder[idx]

    prior_key = prior_iters = None
    if not args.no_prior:
        if args.prior_iter is not None:
            pm = [(i, k) for i, k in ladder if i == args.prior_iter]
            if pm:
                prior_iters, prior_key = pm[0]
        elif idx > 0:
            prior_iters, prior_key = ladder[idx - 1]

    # --- ranged extraction ---------------------------------------------------------------------------
    print(f"[ranged-gate] latest = iter {latest_iters}  (s3://{args.bucket}/{latest_key})")
    latest_reader = rs.S3RangedReader(args.bucket, latest_key, args.region)
    latest_it, latest_nodes, latest_m1, latest_neg, latest_menu = _extract_ranged(
        latest_reader, node_list, keys, neg_keys, detail_cls, arm3_probe, args.workers)
    print(f"[ranged-gate] iter {latest_it}: strat_rows={latest_m1['strat_rows']:,} "
          f"cum_rows={latest_m1['cum_rows']:,} blob={latest_m1['blob_bytes']/1e9:.1f}GB "
          f"neg_ctrl_rows={latest_neg} arm3_menu_present={latest_menu['arm3_present']} "
          f"(btn_id56_mass={latest_menu['btn_id56_mass']:.3g} bb200_rows={latest_menu['bb200_rows']} "
          f"bb212_rows={latest_menu['bb212_rows']})", flush=True)

    prior_nodes = prior_m1 = None
    if prior_key:
        print(f"[ranged-gate] prior  = iter {prior_iters}  (trend baseline)")
        prior_reader = rs.S3RangedReader(args.bucket, prior_key, args.region)
        prior_it, prior_nodes, prior_m1, prior_neg, _pm = _extract_ranged(
            prior_reader, node_list, keys, neg_keys, detail_cls, arm3_probe, args.workers)
        if prior_neg != 0:
            print(f"ABORT: prior negative control read {prior_neg} rows under {neg_name} -- key space "
                  "wrong; not emitting a verdict.")
            return 2

    # --- negative control + pre-menu consistency guard (identical to bpv4_kill_gate.main) ------------
    if latest_neg != 0:
        print(f"ABORT: negative control read {latest_neg} rows under the {neg_name} key space -- the "
              "checkpoint is NOT in the fullhist key space. Not emitting a verdict.")
        return 2
    blob_is_arm3 = latest_menu["arm3_present"]
    flag_is_arm3 = args.pre_menu == "arm3"
    if blob_is_arm3 != flag_is_arm3:
        print(f"ABORT (pre-menu mismatch): blob arm3={blob_is_arm3} but --pre-menu={args.pre_menu} "
              f"(btn_id56_mass={latest_menu['btn_id56_mass']:.4g}, bb200={latest_menu['bb200_rows']}, "
              f"bb212={latest_menu['bb212_rows']}). Not emitting a verdict.")
        return 2

    trend_hard = int(latest_it) >= args.trend_hard_iters
    k1a_regime = int(latest_it) < args.k1_hard_iters
    regime = "K1a (EARLY-KILL: sole hard clause AA-limp)" if k1a_regime else "K1 (BINDING all-clause)"
    aa_cap_applied = args.k1a_aa_cap if k1a_regime else args.aa_limp_cap
    print(f"\n=== bpv4 {args.run_subdir} RANGED MID-RUN GATE @ iters={latest_it}  [{regime}, "
          f"AA cap {aa_cap_applied*100:.0f}%]"
          + (f" (trend vs {prior_iters})" if prior_m1 else "") + " ===")
    ok, lines, gdetail = kg.evaluate(
        latest_nodes, prior_nodes, mono_tol_pt=args.mono_tol_pt, btn_floor_pct=args.btn_floor_pct,
        aa_cap=args.aa_limp_cap, bb_fold_cap=args.bb_fold_cap, jam_cap=args.jam_cap,
        decline_floor_pt=args.decline_floor_pt, trend_hard=trend_hard,
        k1a_regime=k1a_regime, k1a_aa_cap=args.k1a_aa_cap)
    for ln in lines:
        print("  " + ln)

    # Leading-indicator extras (NOT part of the pre-registered verdict).
    aa_avg = kg._aa_limp(latest_nodes)
    aa_cur = _current_aa_limp(latest_nodes)
    aa_avg_prev = kg._aa_limp(prior_nodes) if prior_nodes else None
    aa_cur_prev = _current_aa_limp(prior_nodes) if prior_nodes else None
    print(f"  [extra] AA-BTN-limp AVERAGE  (gate quantity): "
          f"{'NA' if aa_avg is None else f'{aa_avg*100:.1f}%'}"
          + ("" if aa_avg_prev is None else f"  (prior {aa_avg_prev*100:.1f}%)"))
    print(f"  [extra] AA-BTN-limp CURRENT  (leading ind.) : "
          f"{'NA' if aa_cur is None else f'{aa_cur*100:.1f}%'}"
          + ("" if aa_cur_prev is None else f"  (prior {aa_cur_prev*100:.1f}%)"))
    print(f"  [extra] arm-2b kill signature was AA-limp 61.8% @20k; K1a EARLY-KILL cap is 50%.")
    print(f"  M1: strat_rows={latest_m1['strat_rows']:,} cum_rows={latest_m1['cum_rows']:,} "
          f"blob={latest_m1['blob_bytes']/1e9:.2f}GB  neg_control_rows={latest_neg} (must be 0)  "
          f"[ranged reads: {latest_reader.n_get:,} GETs, {latest_reader.n_bytes/1e6:.0f} MB]")

    verdict = "PASS" if ok else "KILL"
    print(f"\nVERDICT: {verdict}  [regime {regime}; AA cap {aa_cap_applied*100:.0f}%]  "
          "(pre-registered = AVERAGE strategy)")

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump({
            "verdict": verdict, "run_subdir": args.run_subdir, "iters": latest_it,
            "prior_iters": prior_iters, "regime": "K1a" if k1a_regime else "K1",
            "k1a_regime": k1a_regime, "aa_cap_applied": aa_cap_applied, "trend_hard": trend_hard,
            "gates": lines, "gate_detail": gdetail, "m1_latest": latest_m1,
            "neg_control_rows": latest_neg, "menu_read": latest_menu,
            "aa_limp_average": aa_avg, "aa_limp_current": aa_cur,
            "aa_limp_average_prior": aa_avg_prev, "aa_limp_current_prior": aa_cur_prev,
            "ranged_gets": latest_reader.n_get, "ranged_mb": round(latest_reader.n_bytes / 1e6, 1),
            "pre_menu": args.pre_menu, "open_raise_ids": list(prd.RAISE_SET),
        }, f, indent=2)
    print(f"wrote {out}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
