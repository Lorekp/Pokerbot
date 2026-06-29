#!/usr/bin/env python3
"""f32_regret QUALITY GATE (local preliminary) — CRN-paired LBR exploitability A/B.

This is the SOUND quality metric for the f32_regret lever (unlike per-infoset strategy-L1, which measures
benign chaotic re-equilibration in a no-unique-equilibrium game; see f32_regret_strategy_delta.py). It
runs the fully-native rayon LBR6 on the f64-baseline and f32_regret blueprints at MATCHED CRN seeds and
reports the paired difference in exploitability (bb/100). f32 passes iff the paired difference's 95% CI
straddles 0 (no statistically-significant increase in exploitability).

PIPELINE (all local, free): NLCFR001 blob -> snapshot pkl -> native LBR (validate_blueprint_native) per
arm at identical seeds -> paired t-CI on per-seed exploitability.

  python scripts/f32_regret_lbr_ab.py \
    --f64-bin .../ab_f64.bin --f32-bin .../ab_f32.bin --hands 5000 --seeds 20 --value-only

NOTE: this corroborates the chaotic-re-equilibration-is-benign axis at a few-thousand iters (where regret
magnitudes are tiny and f32 is trivially precise). The HIGH-MAGNITUDE axis (regret ~2e7 at the production
T=62000, where f32 ULP ~= 1-2) rests on the numerical-headroom analysis + the full cloud T=62000 LBR
ship-gate; do NOT flip the f32_regret default ON from this preliminary alone.
"""
import argparse
import json
import math
import os
import statistics as st
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def to_snapshot(bin_path, pkl_path):
    subprocess.run(
        [sys.executable, os.path.join(HERE, "nlcfr_to_snapshot.py"), bin_path, pkl_path],
        check=True,
        cwd=HERE,  # nlcfr_to_snapshot imports f32_regret_strategy_delta (same dir)
    )


def run_lbr(pkl_path, hands, seeds, value_only, out_json):
    cmd = [
        sys.executable,
        os.path.join(ROOT, "scripts", "eval", "validate_blueprint_native.py"),
        "--checkpoint", pkl_path,
        "--hands", str(hands),
        "--seeds", str(seeds),
        "--no-calibrate",
        "--output", out_json,
    ]
    if value_only:
        cmd.append("--value-only")
    env = dict(os.environ, POKERBOT_PREFLOP_LOSSLESS="1")
    subprocess.run(cmd, check=True, cwd=ROOT, env=env)
    with open(out_json) as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--f64-bin", required=True, help="f64-baseline NLCFR001 blob")
    ap.add_argument("--f32-bin", required=True, help="f32_regret NLCFR001 blob")
    ap.add_argument("--hands", type=int, default=5000)
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--value-only", action="store_true")
    args = ap.parse_args()
    # Resolve inputs to absolute paths NOW (relative to the invocation cwd) — the conversion + validation
    # subprocesses run with cwd=scripts/ and cwd=ROOT respectively, so relative paths would break.
    args.f64_bin = os.path.abspath(args.f64_bin)
    args.f32_bin = os.path.abspath(args.f32_bin)

    with tempfile.TemporaryDirectory() as td:
        f64_pkl, f32_pkl = os.path.join(td, "f64.pkl"), os.path.join(td, "f32.pkl")
        to_snapshot(args.f64_bin, f64_pkl)
        to_snapshot(args.f32_bin, f32_pkl)
        print("=== f64 baseline LBR ===")
        a = run_lbr(f64_pkl, args.hands, args.seeds, args.value_only, os.path.join(td, "f64.json"))
        print("=== f32_regret LBR ===")
        b = run_lbr(f32_pkl, args.hands, args.seeds, args.value_only, os.path.join(td, "f32.json"))

    fa, fb = a["per_seed"], b["per_seed"]
    assert len(fa) == len(fb), "seed count mismatch — arms not CRN-paired"
    d = [x - y for x, y in zip(fb, fa)]  # f32 - f64 (positive = f32 MORE exploitable = worse)
    n = len(d)
    mean = st.mean(d)
    sd = st.pstdev(d) * math.sqrt(n / (n - 1)) if n > 1 else 0.0
    se = sd / math.sqrt(n) if n else 0.0
    # two-sided t_.975 for small df (table; falls back to 1.96 for large n)
    tcrit = {4: 2.776, 8: 2.365, 12: 2.201, 16: 2.131, 20: 2.093, 24: 2.069, 30: 2.045}.get(
        n, 1.96
    )
    lo, hi = mean - tcrit * se, mean + tcrit * se
    straddles = lo < 0 < hi

    print("\n================ f32_regret LBR quality gate (local preliminary) ================")
    print(f"  f64 exploitability = {a['bb_per_100']:+.1f} bb/100  CI{[round(c,1) for c in a['ci95']]}")
    print(f"  f32 exploitability = {b['bb_per_100']:+.1f} bb/100  CI{[round(c,1) for c in b['ci95']]}")
    print(f"  CRN-paired diff (f32 - f64) = {mean:+.2f} bb/100  se={se:.2f}  t={mean/se if se else 0:+.3f}")
    print(f"  95% CI for the difference   = [{lo:+.1f}, {hi:+.1f}] bb/100  (n={n} paired seeds)")
    print(
        f"  VERDICT: {'PASS — no significant exploitability difference (f32 quality-equivalent)' if straddles else 'FAIL — f32 significantly changes exploitability; investigate'}"
    )
    print("  (lower exploitability = better; positive diff = f32 nominally worse but must be non-sig)")
    sys.exit(0 if straddles else 1)


if __name__ == "__main__":
    main()
