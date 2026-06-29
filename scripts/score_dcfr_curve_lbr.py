"""LOCAL scorer for the DCFR-vs-CFR+ exploitability-vs-compute curve. Reads the streamed checkpoints
(blueprint_{arm}_{milestone}.pkl + manifest.json) the box produced, runs PAIRED-CRN native LBR on every
milestone of EACH arm, and emits the curve + the convergence-speed verdict.

This is the measurement I kept deferring: exploitability per unit of compute, the axis that actually
defines blueprint strength -- not deals/s. For each milestone (= equal deals for both arms) we get
LBR(cfrplus) and LBR(dcfr); the paired diff (dcfr - cfrplus) uses the SAME deck seeds so deck variance
cancels. Two readouts:
  * convergence speed (the algorithmic lever): at how many fewer DEALS does DCFR reach CFR+'s plateau LBR.
  * total speed: same, but in WALL-CLOCK (DCFR is also cheaper per deal -- pruning shrinks the deep tail).

Hardened per the pre-launch validity audit:
  B1 each arm is scored across ITS OWN reached milestones (not only milestones present in both), and the
     "plateau/speedup" verdict is WITHHELD unless CFR+ actually reached the final planned milestone -- so a
     spot-kill mid-DCFR cannot mislabel a low, non-converged milestone as "CFR+'s plateau."
  W1 the always-fold calibration anchor GATES: a bad anchor withholds the flip recommendation.
  W2 a CFR+ last-two-milestone slope line is printed so "plateau" is earned, not assumed.
  W4 deals-speedup is reported as the clean algorithmic number; wall-speedup carries a run-order caveat.

  python scripts/score_dcfr_curve_lbr.py artifacts/dcfrcurve_<ts>
"""
from __future__ import annotations

import json
import os
import pickle
import sys
import time

os.environ.setdefault("POKERBOT_PREFLOP_LOSSLESS", "1")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.environ.get("PYD_DIR", HERE))   # staged battery_native.pyd (LBR backend)
sys.path.insert(1, ROOT)                              # pokerbot package (repo root, not the script dir)

import numpy as np  # noqa: E402

RUN_DIR = sys.argv[1] if len(sys.argv) > 1 else "artifacts/dcfrcurve"
SEEDS = int(os.environ.get("CURVE_SEEDS", "12"))
HANDS = int(os.environ.get("CURVE_HANDS", "6000"))
BASE_SEED = 0x5EEDBAB0


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def ci95(vals):
    v = np.asarray(vals, dtype=np.float64)
    m = float(v.mean())
    se = float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else 0.0
    return m, se, m - 1.96 * se, m + 1.96 * se


def load_snap(path):
    with open(path, "rb") as f:
        obj = pickle.load(f)
    return {bytes(k): np.asarray(v, dtype=np.float64) for k, v in obj.items()}


def main():
    manifest = json.load(open(os.path.join(RUN_DIR, "manifest.json")))
    num_buckets = int(manifest["num_buckets"])
    planned = sorted(int(x) for x in manifest["milestones"])
    target = max(planned)
    os.environ.setdefault("POKERBOT_POSTFLOP_NUM_BUCKETS", str(num_buckets))  # LBR made-hand buckets == train

    from pokerbot.eval.lbr6 import LBR6Config  # noqa: PLC0415
    from pokerbot.eval import lbr_native as L  # noqa: PLC0415
    if not L.available():
        log("FATAL native battery_native LBR not available")
        return 1

    cps = {(rec["arm"], int(rec["milestone"])): rec for rec in manifest["checkpoints"]}
    log(f"=== DCFR curve scoring (made-hand K={num_buckets}, {SEEDS} seeds x {HANDS} hands) ===")
    log(f"planned milestones: {planned} (target={target})  present: {sorted(cps)}")

    seeds = [BASE_SEED ^ i for i in range(SEEDS)]
    assert len(set(seeds)) == SEEDS

    # calibration anchor (blueprint-independent) -- GATES the verdict (W1)
    calib = LBR6Config(hands=HANDS, seed=0, n_seats=6)
    rf = L.lbr6_exploitability_native(None, calib, policy="fold")
    calib_ok = 110 <= rf.bb_per_100 <= 145
    log(f"calibration always-fold = {rf.bb_per_100:+.1f} bb/100 ({'OK' if calib_ok else 'OFF -- harness suspect'})")

    def lbr_vals(snap):
        out = []
        for s in seeds:
            cfg = LBR6Config(hands=HANDS, seed=s, n_seats=6)
            r = L.lbr6_exploitability_native(snap, cfg, policy="blueprint", depth_edges=[])
            out.append(float(r.bb_per_100))
        return out

    # B1: score EVERY checkpoint of EACH arm across its own reached milestones (not only the intersection)
    scored = {}  # (arm, milestone) -> dict
    for (arm, m), rec in sorted(cps.items()):
        snap = load_snap(os.path.join(RUN_DIR, rec["file"]))
        vals = lbr_vals(snap)
        mn, _se, lo, hi = ci95(vals)
        scored[(arm, m)] = {"mean": mn, "ci95": [lo, hi], "vals": vals, "wall_s": rec["wall_s"],
                            "infosets": rec["infosets"], "deals": rec["cum_deals"]}
        log(f"  {arm} @ {m} (deals={rec['cum_deals']/1e6:.2f}M): LBR {mn:+.1f} [{lo:+.0f},{hi:+.0f}] "
            f"infosets={rec['infosets']:,} wall={rec['wall_s']:.0f}s")

    cfrplus_ms = sorted(m for (a, m) in scored if a == "cfrplus")
    dcfr_ms = sorted(m for (a, m) in scored if a == "dcfr")
    both = sorted(set(cfrplus_ms) & set(dcfr_ms))
    if not cfrplus_ms or not dcfr_ms:
        log("an arm has zero scored checkpoints -- cannot build the curve")
        return 1

    # paired rows (same seeds -> deck variance cancels) where both arms reached the milestone
    rows = []
    for m in both:
        c, d = scored[("cfrplus", m)], scored[("dcfr", m)]
        diff = [dv - cv for dv, cv in zip(d["vals"], c["vals"])]
        pm, _pse, plo, phi = ci95(diff)
        rows.append({"milestone": m, "deals": c["deals"],
                     "cfrplus": {k: c[k] for k in ("mean", "ci95", "wall_s", "infosets")},
                     "dcfr": {k: d[k] for k in ("mean", "ci95", "wall_s", "infosets")},
                     "paired_diff": {"mean": pm, "ci95": [plo, phi]}})

    # --- completeness gate (B1): trust "plateau" only if CFR+ reached the planned target ---
    cfrplus_complete = max(cfrplus_ms) >= target
    dcfr_complete = max(dcfr_ms) >= target
    plat_m = max(cfrplus_ms)
    plateau = scored[("cfrplus", plat_m)]
    cfrplus_plateau = plateau["mean"]
    cfrplus_plateau_deals = plateau["deals"]
    cfrplus_plateau_wall = plateau["wall_s"]

    # CFR+ last-two-milestone slope (W2): is the "plateau" actually flat?
    slope_note = ""
    if len(cfrplus_ms) >= 2:
        m_prev = cfrplus_ms[-2]
        d_lbr = cfrplus_plateau - scored[("cfrplus", m_prev)]["mean"]
        slope_note = (f"CFR+ slope last leg ({m_prev}->{plat_m}): {d_lbr:+.1f} bb/100 "
                      f"({'still improving' if d_lbr < -2 else 'flat'})")

    # --- DCFR reach: first DCFR milestone whose LBR mean <= CFR+'s plateau ---
    dcfr_reach_m = next((m for m in dcfr_ms if scored[("dcfr", m)]["mean"] <= cfrplus_plateau), None)
    speedup_deals = speedup_wall = None
    if dcfr_reach_m is not None:
        dr = scored[("dcfr", dcfr_reach_m)]
        speedup_deals = cfrplus_plateau_deals / dr["deals"]
        speedup_wall = cfrplus_plateau_wall / dr["wall_s"]

    # --- plateau quality guardrail (DCFR has no 6-max Nash guarantee) ---
    final_paired = rows[-1]["paired_diff"] if rows else None
    if final_paired is None:
        quality = "no overlapping milestone -- cannot pair"
    elif final_paired["ci95"][0] > 0:
        quality = "DCFR WORSE at top milestone (paired CI > 0) -- speed traded for exploitability"
    elif final_paired["ci95"][1] < 0:
        quality = "DCFR BETTER at top milestone (paired CI < 0)"
    else:
        quality = "quality-equivalent at top milestone (paired CI crosses 0)"

    # --- verdict (gated on calibration W1 + completeness B1) ---
    if not calib_ok:
        verdict = f"HARNESS SUSPECT (always-fold {rf.bb_per_100:+.1f} outside [110,145]) -- verdict WITHHELD"
    elif not cfrplus_complete:
        verdict = (f"PARTIAL -- CFR+ only reached {plat_m}/{target} iters; its plateau is not established, "
                   f"speedup WITHHELD. Curve below is still valid as far as it goes.")
    elif speedup_deals is None:
        verdict = (f"DCFR did NOT reach CFR+'s plateau LBR ({cfrplus_plateau:+.1f}) within measured "
                   f"milestones -- NOT a convergence win on this abstraction; {quality}")
    elif speedup_deals >= 1.5 and final_paired and final_paired["ci95"][0] <= 0:
        verdict = (f"DCFR+pruning WINS: reaches CFR+'s plateau LBR in {speedup_deals:.1f}x fewer deals "
                   f"(~{speedup_wall:.1f}x less wall) with no plateau-quality regression -> flip default ON")
    elif speedup_deals >= 1.5:
        verdict = (f"DCFR+pruning FASTER ({speedup_deals:.1f}x fewer deals) but {quality} -> keep gated; "
                   f"revisit prune_threshold/explore_frac")
    else:
        verdict = (f"DCFR+pruning NOT a clear convergence win (speedup_deals={speedup_deals:.2f}); "
                   f"{quality} -> keep gated-OFF, look at PCS/vector-CFR")

    print("\n" + "=" * 94)
    print("DCFR-vs-CFR+ EXPLOITABILITY-vs-COMPUTE CURVE  (LBR bb/100; lower = less exploitable = stronger)")
    print("=" * 94)
    print(f"  calibration always-fold = {rf.bb_per_100:+.1f} ({'OK' if calib_ok else 'OFF'})   "
          f"made-hand K={num_buckets}  {SEEDS} seeds x {HANDS} hands   "
          f"complete: cfrplus={cfrplus_complete} dcfr={dcfr_complete}")
    print(f"  {'deals':>9} | {'CFR+ LBR':>21} | {'DCFR LBR':>21} | {'paired(d-c)':>16} | wall c/d")
    # print the union of milestones; pair where possible, else show the single arm present
    for m in sorted(set(cfrplus_ms) | set(dcfr_ms)):
        c = scored.get(("cfrplus", m))
        d = scored.get(("dcfr", m))
        deals = (c or d)["deals"]
        cstr = f"{c['mean']:+7.1f} [{c['ci95'][0]:+.0f},{c['ci95'][1]:+.0f}]" if c else " " * 21
        dstr = f"{d['mean']:+7.1f} [{d['ci95'][0]:+.0f},{d['ci95'][1]:+.0f}]" if d else " " * 21
        if c and d:
            pd = next(r["paired_diff"] for r in rows if r["milestone"] == m)
            pstr = f"{pd['mean']:+6.1f}[{pd['ci95'][0]:+.0f},{pd['ci95'][1]:+.0f}]"
        else:
            pstr = " " * 16
        wstr = f"{c['wall_s']:.0f}/{d['wall_s']:.0f}s" if (c and d) else (
            f"{c['wall_s']:.0f}/-s" if c else f"-/{d['wall_s']:.0f}s")
        print(f"  {deals/1e6:7.2f}M | {cstr:>21} | {dstr:>21} | {pstr:>16} | {wstr}")
    print("-" * 94)
    if slope_note:
        print(f"  {slope_note}")
    print(f"  CFR+ plateau LBR  : {cfrplus_plateau:+.1f} bb/100 @ {cfrplus_plateau_deals/1e6:.2f}M deals "
          f"/ {cfrplus_plateau_wall:.0f}s  (milestone {plat_m}/{target}, complete={cfrplus_complete})")
    if dcfr_reach_m is not None:
        print(f"  DCFR reaches it   : @ {scored[('dcfr',dcfr_reach_m)]['deals']/1e6:.2f}M deals "
              f"/ {scored[('dcfr',dcfr_reach_m)]['wall_s']:.0f}s (milestone {dcfr_reach_m}) "
              f"-> {speedup_deals:.1f}x fewer deals, ~{speedup_wall:.1f}x less wall")
        print("  (deals-speedup is the clean algorithmic number; wall-speedup also folds in DCFR's per-deal "
              "pruning savings AND any run-order drift -- W4)")
    else:
        print("  DCFR reaches it   : NOT within the measured milestones")
    print(f"  plateau quality   : {quality}")
    print(f"  VERDICT           : {verdict}")
    print("=" * 94)

    out = {"num_buckets": num_buckets, "seeds": SEEDS, "hands": HANDS,
           "planned_milestones": planned, "target": target,
           "calibration_fold_bb": rf.bb_per_100, "calibration_ok": calib_ok,
           "cfrplus_complete": cfrplus_complete, "dcfr_complete": dcfr_complete,
           "rows": rows,
           "per_arm": {f"{a}_{m}": scored[(a, m)] for (a, m) in scored},
           "cfrplus_plateau_lbr": cfrplus_plateau, "cfrplus_plateau_milestone": plat_m,
           "cfrplus_plateau_deals": cfrplus_plateau_deals, "cfrplus_plateau_wall_s": cfrplus_plateau_wall,
           "slope_note": slope_note,
           "dcfr_reach_milestone": dcfr_reach_m,
           "speedup_deals": speedup_deals, "speedup_wall": speedup_wall,
           "plateau_quality": quality, "verdict": verdict}
    jpath = os.path.join(RUN_DIR, "dcfr_curve_verdict.json")
    with open(jpath, "w") as f:
        json.dump(out, f, indent=2)
    log(f"wrote {jpath}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
