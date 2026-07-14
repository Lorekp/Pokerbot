#!/usr/bin/env python3
"""blueprint-v4 STRENGTH GATE -- the missing ACCEPTANCE criterion (so no arm passes on SHAPE alone).

The Stage-0 kill battery (bpv4_kill_gate.py: G1-G6 + K1a + K-COV) is ALL preflop-shape + on-policy
coverage. Nothing in it measures whether a trained arm is actually a STRONGER, deployable blueprint than
the one we serve. This gate closes that hole with the two trustworthy 6-max strength signals, pre-registered:

  (1) bb/100 vs the DEPLOYED EMD-K160 incumbent (checkpoints/serve_blueprint_6max.pkl), seat-rotated
      duplicate + CRN-paired vs the fixed CoinPoker-mirror pool, with a paired 95% CI. This is the
      money signal (eval-rotated-duplicate / 6max-quality-signal-research: rank by mbb/game vs a fixed
      field, never a bare point estimate). A presum arm keys POSTFLOP on the preflop pot-type SUMMARY --
      a DIFFERENT key space than the non-presum incumbent -- and the encoder bakes ONE key space per
      process, so the two CANNOT be served in one process. We therefore drive rank_cross_abstraction.py,
      which runs EACH arm in its OWN subprocess env vs the SAME pool on the SAME deck seeds (CRN-paired).

  (2) LBR exploitability bound (lbr6, native ~700 h/s with a Python analytic fallback), each arm served
      in its own key-space env. LBR is a ONE-SIDED certificate, NOT a 3+-player ranking comparator (no
      Nash in multiway; our LBR RISES as the average sharpens), so it is used only as "arm not markedly
      MORE exploitable than the incumbent + margin", never to rank the two.

PRE-REGISTERED VERDICT (set before the run):
  * bb/100 gate  KILL  iff the arm's paired (arm - incumbent) 95% CI is ENTIRELY BELOW 0 (arm provably
                       weaker on money vs the realistic field). PROMOTABLE iff entirely ABOVE 0.
  * LBR gate     KILL  iff arm_lbr_bb100 > incumbent_lbr_bb100 + --lbr-margin.
  * OVERALL      KILL  on either gate; else PASS. PROMOTABLE = bb/100 provably stronger AND LBR not worse.

SERVABLE-INPUT CONTRACT (RAM):  both --arm and --incumbent must be SERVABLE snapshots the harness can load
into a dict -- a stamped .pkl (e.g. the deployed 7 MB EMD-K160) or a compact .nlcfr. A RAW presum training
checkpoint is NOT servable here: its strat section is preflop-only under snapshot_avg (postflop lives in
the .nlsnap) AND a full presum table is ~145M rows (>20 GB as a dict -> OOM on a 31 GB box). Produce the
arm's SERVABLE first with snap_avg (tools/battery_native/examples/snap_avg.exe <out.nlcfr> <ckpt.nlcfr>
<snap_*.nlsnap ...>) -- that is ALSO the deployed object (the snap_avg average, not a single iterate), which
is the correct thing to gate on.

  # self-check (proves the harness end-to-end; delta must be ~0, LBR equal):
  python scripts/blueprint2/bpv4_strength_gate.py --self-check --decks 200 --lbr-hands 400

  # a real presum arm (servable produced by snap_avg), vs the deployed incumbent:
  python scripts/blueprint2/bpv4_strength_gate.py \
      --arm artifacts/bpv4_presum_1_servable.nlcfr --arm-presum-key --arm-mrr \
      --arm-emd-frozen data/abstraction/emd_frozen_flop_turn_160_r400.emdfrzn \
      --decks 600 --lbr-hands 800 --out artifacts/strength_gate_presum1.json
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile

_HERE = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS = os.path.dirname(_HERE)
_ROOT = os.path.dirname(_SCRIPTS)
for _p in (_HERE, _SCRIPTS, _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

_DEFAULT_INCUMBENT = os.path.join(_ROOT, "checkpoints", "serve_blueprint_6max.pkl")
_DEFAULT_EMD = os.path.join(_ROOT, "data", "abstraction", "emd_frozen_flop_turn_160_r400.emdfrzn")
_XRANK = os.path.join(_SCRIPTS, "rank_cross_abstraction.py")


# =================================================================================================
# LBR arm (internal --lbr-arm mode): env is ALREADY set by the parent; import pokerbot -> bake key space.
# =================================================================================================
def _lbr_arm_main(args) -> int:
    from rank_cross_abstraction import _load  # no pokerbot import -> safe here

    snap = _load(args.blueprint)
    from pokerbot.eval.lbr6 import LBR6Config

    cfg = LBR6Config(hands=int(args.lbr_hands), seed=int(args.seed), n_seats=6, track_ranges=True)
    emd = os.path.abspath(args.emd_frozen) if args.emd_frozen else None
    bb = None
    method = None
    err = None
    try:
        from pokerbot.eval.lbr_native import lbr6_exploitability_native

        res = lbr6_exploitability_native(snap, cfg, policy="blueprint", emd_frozen_path=emd)
        bb = float(res.bb_per_100)
        method = "native"
    except Exception as e:  # native pyd missing / no presum support -> key-space-correct Python fallback
        err = f"{type(e).__name__}: {e}"
        from pokerbot.eval.lbr import make_blueprint_policy
        from pokerbot.eval.lbr6 import lbr6_exploitability

        pol = make_blueprint_policy(snap)
        res = lbr6_exploitability(pol, cfg)
        bb = float(getattr(res, "bb_per_100", None) if not isinstance(res, dict) else res["bb_per_100"])
        method = "python(native_failed)"
    out = {"label": args.label, "lbr_bb_per_100": bb, "hands": int(args.lbr_hands),
           "infosets": len(snap), "method": method, "native_error": err}
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f)
    print(f"[lbr] {args.label}: exploitability {bb:+.1f} bb/100 ({method}; {len(snap):,} infosets)", flush=True)
    return 0


def _run_lbr(blueprint, *, label, mode, num_buckets, emd_frozen, river_ochs,
             preflop_hist_key, fullhist_key, presum_key, mrr, lbr_hands, seed) -> dict:
    from rank_cross_abstraction import _arm_env

    out = tempfile.mktemp(suffix=f"_lbr_{label}.json")
    cmd = [sys.executable, os.path.abspath(__file__), "--lbr-arm", "--blueprint", blueprint,
           "--label", label, "--out", out, "--lbr-hands", str(lbr_hands), "--seed", str(seed)]
    if mode == "emd" and emd_frozen:
        cmd += ["--emd-frozen", emd_frozen]
    env = _arm_env(os.environ, mode=mode, num_buckets=num_buckets, emd_frozen=emd_frozen,
                   workers=1, river_ochs=river_ochs, preflop_hist_key=preflop_hist_key,
                   fullhist_key=fullhist_key, presum_key=presum_key, mrr=mrr)
    print(f"[gate] LBR {label} ({mode}/K{num_buckets}) ...", flush=True)
    r = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if r.returncode != 0:
        sys.stderr.write((r.stdout[-2000:] or "") + "\n" + (r.stderr[-2000:] or "") + "\n")
        raise SystemExit(f"LBR arm {label} failed (rc={r.returncode})")
    for line in r.stdout.splitlines():
        if line.startswith("[lbr]"):
            print("   " + line, flush=True)
    with open(out, encoding="utf-8") as f:
        data = json.load(f)
    os.remove(out)
    return data


# =================================================================================================
# bb/100 vs incumbent -- delegate to rank_cross_abstraction (subprocess-per-arm, own env, CRN-paired).
# =================================================================================================
def _run_bb100(args, out_json) -> dict:
    cmd = [sys.executable, _XRANK,
           "--a", args.incumbent, "--a-mode", args.incumbent_mode,
           "--a-num-buckets", str(args.incumbent_num_buckets), "--a-emd-frozen", args.incumbent_emd_frozen,
           "--b", args.arm, "--b-mode", args.arm_mode,
           "--b-num-buckets", str(args.arm_num_buckets), "--b-emd-frozen", args.arm_emd_frozen,
           "--decks", str(args.decks), "--seed", str(args.seed), "--workers", str(args.workers),
           "--label-a", "deployed-K160", "--label-b", "bpv4-arm", "--out-json", out_json]
    if args.arm_presum_key:
        cmd += ["--b-presum-key"]
    if args.arm_fullhist_key:
        cmd += ["--b-fullhist-key"]
    if args.arm_mrr:
        cmd += ["--b-mrr"]
    if args.arm_river_ochs:
        cmd += ["--b-river-ochs", args.arm_river_ochs]
    # the incumbent is the deployed EMD-K160 (no fullhist/presum/mrr) -> no --a-* key-space flags.
    print(f"[gate] bb/100 vs pool: rank_cross_abstraction ({args.decks} decks) ...", flush=True)
    r = subprocess.run(cmd)
    if r.returncode != 0:
        raise SystemExit(f"bb/100 comparison failed (rc={r.returncode})")
    with open(out_json, encoding="utf-8") as f:
        return json.load(f)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # internal LBR arm
    ap.add_argument("--lbr-arm", action="store_true", help="(internal) run one LBR arm in this process")
    ap.add_argument("--blueprint"); ap.add_argument("--label", default=""); ap.add_argument("--out")
    ap.add_argument("--emd-frozen", default="")
    # arm (candidate)
    ap.add_argument("--arm", help="candidate SERVABLE blueprint (.pkl stamped, or compact .nlcfr)")
    ap.add_argument("--arm-mode", default="emd", choices=("fast", "emd", "ehs"))
    ap.add_argument("--arm-num-buckets", type=int, default=160)
    ap.add_argument("--arm-emd-frozen", default=_DEFAULT_EMD)
    ap.add_argument("--arm-river-ochs", default="")
    ap.add_argument("--arm-preflop-hist-key", action="store_true")
    ap.add_argument("--arm-fullhist-key", action="store_true", help="arm trained with POKERBOT_FULLHIST_KEY=1")
    ap.add_argument("--arm-presum-key", action="store_true", help="arm trained with POKERBOT_PRESUM_KEY=1 (implies fullhist)")
    ap.add_argument("--arm-mrr", action="store_true", help="arm trained with POKERBOT_MRR=1")
    # incumbent (deployed EMD-K160 by default)
    ap.add_argument("--incumbent", default=_DEFAULT_INCUMBENT)
    ap.add_argument("--incumbent-mode", default="emd", choices=("fast", "emd", "ehs"))
    ap.add_argument("--incumbent-num-buckets", type=int, default=160)
    ap.add_argument("--incumbent-emd-frozen", default=_DEFAULT_EMD)
    # run knobs
    ap.add_argument("--decks", type=int, default=600, help="distinct deals per arm vs the pool (each = 6 rotations)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--lbr-hands", type=int, default=800)
    ap.add_argument("--lbr-margin", type=float, default=25.0,
                    help="KILL if arm LBR exceeds incumbent LBR by more than this (bb/100)")
    ap.add_argument("--self-check", action="store_true",
                    help="arm := incumbent (EMD-K160); paired bb/100 must be ~0 and LBR equal -- proves wiring")
    args = ap.parse_args()

    if args.lbr_arm:
        return _lbr_arm_main(args)

    if args.self_check:
        args.arm = args.incumbent
        args.arm_mode = args.incumbent_mode
        args.arm_num_buckets = args.incumbent_num_buckets
        args.arm_emd_frozen = args.incumbent_emd_frozen
        args.arm_presum_key = args.arm_fullhist_key = args.arm_mrr = args.arm_preflop_hist_key = False
        args.arm_river_ochs = ""
    if not args.arm:
        ap.error("--arm is required (or use --self-check)")

    print(f"[gate] STRENGTH GATE  arm={args.arm}  incumbent={args.incumbent}", flush=True)
    print(f"[gate]   decks={args.decks} lbr_hands={args.lbr_hands} seed={args.seed} "
          f"lbr_margin={args.lbr_margin}", flush=True)

    # (1) bb/100 vs the fixed pool, paired (arm - incumbent).
    bb_json = args.out + ".bb100.json" if args.out else tempfile.mktemp(suffix="_bb100.json")
    bb = _run_bb100(args, bb_json)
    comb = bb["combined"]
    delta = float(comb["delta_bb_per_100"]); ci = float(comb["ci95"])
    lo, hi = delta - ci, delta + ci
    bb_weaker = hi < 0.0            # arm 95% CI entirely below the incumbent
    bb_stronger = lo > 0.0         # arm 95% CI entirely above the incumbent

    # (2) LBR exploitability, each arm in its own key-space env.
    lbr_inc = _run_lbr(args.incumbent, label="deployed-K160", mode=args.incumbent_mode,
                       num_buckets=args.incumbent_num_buckets, emd_frozen=args.incumbent_emd_frozen,
                       river_ochs="", preflop_hist_key=False, fullhist_key=False, presum_key=False,
                       mrr=False, lbr_hands=args.lbr_hands, seed=args.seed)
    lbr_arm = _run_lbr(args.arm, label="bpv4-arm", mode=args.arm_mode,
                       num_buckets=args.arm_num_buckets, emd_frozen=args.arm_emd_frozen,
                       river_ochs=args.arm_river_ochs, preflop_hist_key=args.arm_preflop_hist_key,
                       fullhist_key=args.arm_fullhist_key or args.arm_presum_key,
                       presum_key=args.arm_presum_key, mrr=args.arm_mrr,
                       lbr_hands=args.lbr_hands, seed=args.seed)
    lbr_a = float(lbr_arm["lbr_bb_per_100"]); lbr_i = float(lbr_inc["lbr_bb_per_100"])
    lbr_worse = lbr_a > lbr_i + args.lbr_margin

    # (3) pre-registered verdict.
    bb_gate = "KILL" if bb_weaker else "PASS"
    lbr_gate = "KILL" if lbr_worse else "PASS"
    overall = "KILL" if (bb_weaker or lbr_worse) else "PASS"
    promotable = bool(bb_stronger and not lbr_worse)

    result = {
        "arm": args.arm, "incumbent": args.incumbent,
        "decks": args.decks, "lbr_hands": args.lbr_hands, "seed": args.seed,
        "bb100_vs_pool": {"delta_bb_per_100": delta, "ci95": ci, "lo": lo, "hi": hi,
                          "a_incumbent": comb["a"], "b_arm": comb["b"], "n": comb["n"],
                          "arm_coverage": bb.get("b_coverage"), "incumbent_coverage": bb.get("a_coverage"),
                          "verdict": bb["verdict"], "per_archetype": bb.get("per_archetype")},
        "lbr": {"arm_bb_per_100": lbr_a, "incumbent_bb_per_100": lbr_i, "margin": args.lbr_margin,
                "arm_method": lbr_arm["method"], "incumbent_method": lbr_inc["method"]},
        "gates": {"bb100": bb_gate, "lbr": lbr_gate},
        "verdict": overall, "promotable": promotable,
    }
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)

    print("\n[gate] ================= STRENGTH GATE VERDICT =================", flush=True)
    print(f"[gate] bb/100 vs pool (arm - incumbent): {delta:+.2f} +- {ci:.2f}  "
          f"[95% CI {lo:+.1f}..{hi:+.1f}]  -> {bb_gate}"
          f"{'  (PROMOTABLE)' if bb_stronger else ''}", flush=True)
    print(f"[gate]   incumbent {comb['a']:+.1f} | arm {comb['b']:+.1f} bb/100 vs pool  "
          f"(arm cov {(bb.get('b_coverage') or 0)*100:.0f}%)", flush=True)
    print(f"[gate] LBR exploitability: arm {lbr_a:+.1f} vs incumbent {lbr_i:+.1f} "
          f"(+{args.lbr_margin:.0f} margin) -> {lbr_gate}", flush=True)
    print(f"[gate] OVERALL: {overall}"
          f"{'  |  PROMOTABLE (arm >= deployed on money, LBR not worse)' if promotable else ''}", flush=True)
    if args.out:
        print(f"[gate] wrote {args.out}", flush=True)
    # non-zero exit on KILL so the babysitter/battery can branch on it.
    return 0 if overall == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
