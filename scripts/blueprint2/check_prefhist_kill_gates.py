#!/usr/bin/env python3
"""Mid-run KILL GATE for the bp3_prefhist_1 retrain (preflop history-key blueprint-v3).

Downloads the latest S3 ladder checkpoint (plus the previous ladder point for the trend check),
runs the Leg-A direct-key preflop extraction (scripts/blueprint2/preflop_root_diag.py --hist-key --
the extractor that reproduced the committed audit numbers to <=0.5pt), and evaluates the
PRE-REGISTERED acceptance gates from artifacts/preflop_mispricing_diagnosis.md section 6.4,
calibrated to the local 20k A/B (NOT to chart consensus):

  G1  positional monotonicity NORMAL: UTG <= HJ <= CO <= BTN opens (tolerance --mono-tol,
      default 0.5pt) and BTN strictly wider than UTG -- no inversion anywhere.
  G2  BTN open >= 15%           (local hist arm: 23.9% @20k; deployed pathology: 4-9%).
  G3  AA-BTN-limp <= 60%        AND falling vs the prior ladder point. TREND CALIBRATION NOTE:
      the E5 POSITIVE CONTROL itself rises 21.6% @5k -> 53.7% @20k (early transient) while being
      the validated-good arm, so a hard trend-kill at the 10-20k mid-run point would kill the
      positive control. The <=60% cap is therefore HARD at every check; the falling-trend clause
      is ADVISORY (WARN) before --trend-hard-iters (default 30000) and HARD from there on (the
      trap-limp share must recede with convergence, like every prior run's jam share).
  G4  BB-vs-BTN-open fold <= 75% (local: 69.6% @20k; deployed pathology: 82-90%).

Prints PASS/KILL with every number. KILL => terminate the box (~$5 sunk at 10-20k iters):
  aws ec2 terminate-instances --instance-ids <iid> --region eu-central-1
(and stop the babysitter). Field strength is still decided by the FINAL 8k-deck CRN gate -- this
gate only prevents paying for a run that already reproduces the aliasing pathology.

Usage (default = latest bp3_prefhist_1 ladder point on S3):
  python scripts/blueprint2/check_prefhist_kill_gates.py
  python scripts/blueprint2/check_prefhist_kill_gates.py --iter 20000       # pin a ladder point
  python scripts/blueprint2/check_prefhist_kill_gates.py --checkpoint a.nlcfr --prior-checkpoint b.nlcfr
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))

DEFAULT_BUCKET = "video-processing-6789"
DEFAULT_PREFIX = "pokerbot/prod6000/native_cfr"
DEFAULT_RUN = "bp3_prefhist_1"
DEFAULT_REGION = "eu-central-1"


def _list_ladder(s3, bucket: str, run_prefix: str) -> list[tuple[int, str]]:
    """Sorted [(iters, key)] for the versioned iter_XXXXXX.nlcfr ladder checkpoints."""
    out: list[tuple[int, str]] = []
    token = None
    while True:
        kw = {"Bucket": bucket, "Prefix": f"{run_prefix}/checkpoints/"}
        if token:
            kw["ContinuationToken"] = token
        resp = s3.list_objects_v2(**kw)
        for obj in resp.get("Contents", []):
            m = re.search(r"iter_(\d+)\.nlcfr$", obj["Key"])
            if m:
                out.append((int(m.group(1)), obj["Key"]))
        if not resp.get("IsTruncated"):
            break
        token = resp.get("NextContinuationToken")
    return sorted(out)


def _download(s3, bucket: str, key: str, dst: str) -> str:
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if not os.path.exists(dst):
        print(f"[kill-gate] downloading s3://{bucket}/{key} -> {dst}", flush=True)
        s3.download_file(bucket, key, dst)
    else:
        print(f"[kill-gate] reusing cached {dst}", flush=True)
    return dst


def _run_diag(checkpoints: list[str], out_json: str) -> dict:
    """Leg-A direct-key extraction with the bp3 hist-key key space. The extractor builds the H bytes
    itself (--hist-key); POKERBOT_PREFLOP_HIST_KEY=1 is ALSO exported so any env-sensitive codepath
    in the extraction subprocess matches the bp3 serve regime."""
    env = dict(os.environ)
    env["POKERBOT_PREFLOP_HIST_KEY"] = "1"
    cmd = [sys.executable, os.path.join(_HERE, "preflop_root_diag.py"),
           "--hist-key", "--out", out_json] + checkpoints
    print(f"[kill-gate] extracting: {' '.join(cmd)}", flush=True)
    proc = subprocess.run(cmd, env=env, cwd=REPO, capture_output=True, text=True, timeout=1800)
    if proc.returncode != 0:
        print(proc.stdout)
        print(proc.stderr, file=sys.stderr)
        raise RuntimeError(f"preflop_root_diag failed rc={proc.returncode}")
    with open(out_json, encoding="utf-8") as f:
        return json.load(f)


def _node(ck: dict, label: str) -> dict:
    n = ck["nodes"].get(label) or {}
    return n.get("avg") or {}


def _aa_limp(ck: dict) -> float | None:
    d = ((ck["nodes"].get("rfi_BTN") or {}).get("detail") or {}).get("AA") or {}
    avg = d.get("avg") or {}
    return avg.get("CALL")


def evaluate(latest: dict, prior: dict | None, *, mono_tol: float, btn_floor: float,
             aa_cap: float, bb_fold_cap: float, trend_hard: bool) -> tuple[bool, list[str]]:
    lines: list[str] = []
    ok = True

    opens = {}
    for name in ("UTG", "HJ", "CO", "BTN", "SB"):
        opens[name] = float(_node(latest, f"rfi_{name}").get("open_or_raise", float("nan")))
    seq = [opens[n] for n in ("UTG", "HJ", "CO", "BTN")]
    lines.append("opens (avg, open_or_raise): "
                 + "  ".join(f"{n}={opens[n]*100:.1f}%" for n in ("UTG", "HJ", "CO", "BTN", "SB")))

    # G1 monotonicity: no earlier position may open more than a later one beyond mono_tol.
    inversions = []
    names = ("UTG", "HJ", "CO", "BTN")
    for i in range(4):
        for j in range(i + 1, 4):
            if seq[j] < seq[i] - mono_tol:
                inversions.append(f"{names[i]}({seq[i]*100:.1f}%) > {names[j]}({seq[j]*100:.1f}%)")
    g1 = not inversions and seq[3] > seq[0]
    ok &= g1
    lines.append(f"G1 monotonicity NORMAL (tol {mono_tol*100:.1f}pt, BTN>UTG): "
                 f"{'PASS' if g1 else 'FAIL'}" + (f"  inversions: {inversions}" if inversions else ""))

    # G2 BTN open floor.
    g2 = opens["BTN"] >= btn_floor
    ok &= g2
    lines.append(f"G2 BTN open {opens['BTN']*100:.1f}% >= {btn_floor*100:.0f}%: {'PASS' if g2 else 'FAIL'}")

    # G3 AA-BTN limp cap + falling trend.
    aa = _aa_limp(latest)
    if aa is None:
        ok = False
        lines.append("G3 AA-BTN-limp: FAIL (AA row missing at rfi_BTN -- checkpoint too early or "
                     "wrong key space; verify the run exports POKERBOT_PREFLOP_HIST_KEY=1)")
    else:
        g3a = aa <= aa_cap
        trend_txt = "no prior ladder point (trend not checked)"
        falling = True
        if prior is not None:
            aa_prev = _aa_limp(prior)
            if aa_prev is not None:
                falling = aa < aa_prev + 0.005  # falling (not rising) across the ladder
                trend_txt = f"prior {aa_prev*100:.1f}% -> latest {aa*100:.1f}% ({'falling/flat' if falling else 'RISING'})"
        # The cap is HARD everywhere; the trend is HARD only from --trend-hard-iters (see module
        # docstring: the E5 positive control RISES 21.6%->53.7% over 5k->20k, so an early hard
        # trend-kill would kill the validated-good arm).
        g3 = g3a and (falling or not trend_hard)
        ok &= g3
        status = "PASS" if g3 else "FAIL"
        if g3 and not falling:
            status = "PASS (trend WARN: rising, advisory before trend-hard-iters)"
        lines.append(f"G3 AA-BTN-limp {aa*100:.1f}% <= {aa_cap*100:.0f}% and falling"
                     f"{' (HARD trend)' if trend_hard else ' (advisory trend)'}: {status}  [{trend_txt}]")

    # G4 BB-vs-BTN-open fold cap.
    bb = _node(latest, "vsopen_BB")
    fold = float(bb.get("fold", float("nan")))
    g4 = fold <= bb_fold_cap
    ok &= g4
    lines.append(f"G4 BB-vs-BTN-open fold {fold*100:.1f}% <= {bb_fold_cap*100:.0f}%: "
                 f"{'PASS' if g4 else 'FAIL'}")

    # Informational (risk note in the diagnosis): BTN jam share of the opening range.
    allin = float(_node(latest, "rfi_BTN").get("allin", float("nan")))
    lines.append(f"info: BTN RFI all-in share {allin*100:.1f}% (expect it to recede with convergence); "
                 f"BB-vs-UTG fold {float(_node(latest, 'vsopen_BB_vsUTG').get('fold', float('nan')))*100:.1f}% "
                 "(distinct row under the hist key)")
    return ok, lines


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bucket", default=DEFAULT_BUCKET)
    ap.add_argument("--prefix", default=DEFAULT_PREFIX)
    ap.add_argument("--run-subdir", default=DEFAULT_RUN)
    ap.add_argument("--region", default=DEFAULT_REGION)
    ap.add_argument("--iter", type=int, default=None,
                    help="pin a specific ladder point (default: the latest iter_XXXXXX.nlcfr)")
    ap.add_argument("--checkpoint", default=None,
                    help="local .nlcfr override (skips S3 entirely)")
    ap.add_argument("--prior-checkpoint", default=None,
                    help="local prior-ladder-point .nlcfr for the AA-limp trend check")
    ap.add_argument("--mono-tol", type=float, default=0.005,
                    help="monotonicity tolerance as a fraction (default 0.005 = 0.5pt)")
    ap.add_argument("--btn-floor", type=float, default=0.15)
    ap.add_argument("--aa-limp-cap", type=float, default=0.60)
    ap.add_argument("--bb-fold-cap", type=float, default=0.75)
    ap.add_argument("--trend-hard-iters", type=int, default=30000,
                    help="the AA-limp falling-trend clause is ADVISORY (WARN) below this many iters "
                         "and HARD from it (the E5 positive control rises 21.6%%->53.7%% over "
                         "5k->20k, so an early hard trend-kill would kill the validated-good arm)")
    ap.add_argument("--cache-dir", default=os.path.join(REPO, "artifacts", "bp3_ladder"))
    ap.add_argument("--out", default=os.path.join(REPO, "artifacts", "bp3_prefhist_kill_gate.json"))
    args = ap.parse_args()

    latest_path, prior_path = args.checkpoint, args.prior_checkpoint
    latest_iters = prior_iters = None
    if latest_path is None:
        import boto3  # noqa: PLC0415
        s3 = boto3.client("s3", region_name=args.region)
        run_prefix = f"{args.prefix}/{args.run_subdir}"
        ladder = _list_ladder(s3, args.bucket, run_prefix)
        if not ladder:
            print(f"ERROR: no iter_XXXXXX.nlcfr ladder checkpoints under "
                  f"s3://{args.bucket}/{run_prefix}/checkpoints/ (run not started / too early?)")
            return 2
        print(f"[kill-gate] ladder: {[i for i, _ in ladder]}")
        if args.iter is not None:
            match = [(i, k) for i, k in ladder if i == args.iter]
            if not match:
                print(f"ERROR: iter {args.iter} not in ladder {[i for i, _ in ladder]}")
                return 2
            idx = ladder.index(match[0])
        else:
            idx = len(ladder) - 1
        latest_iters, latest_key = ladder[idx]
        latest_path = _download(s3, args.bucket, latest_key,
                                os.path.join(args.cache_dir, f"{args.run_subdir}_iter{latest_iters}.nlcfr"))
        if idx > 0 and prior_path is None:
            # Trend baseline "across the ladder": the latest point at <= HALF the current iters
            # (adjacent 500-iter points are too close to carry a trend signal); falls back to the
            # immediately preceding point early in the run.
            half = [t for t in ladder[:idx] if t[0] <= latest_iters // 2]
            prior_iters, prior_key = (half[-1] if half else ladder[idx - 1])
            prior_path = _download(s3, args.bucket, prior_key,
                                   os.path.join(args.cache_dir, f"{args.run_subdir}_iter{prior_iters}.nlcfr"))

    os.makedirs(args.cache_dir, exist_ok=True)
    diag_out = os.path.join(args.cache_dir, "prefhist_kill_gate_diag.json")
    ckpts = ([prior_path] if prior_path else []) + [latest_path]
    diag = _run_diag(ckpts, diag_out)
    by_iters = sorted(diag["checkpoints"], key=lambda c: c["iters"])
    latest_ck = by_iters[-1]
    prior_ck = by_iters[-2] if len(by_iters) > 1 else None

    print(f"\n=== bp3_prefhist_1 MID-RUN KILL GATE @ iters={latest_ck['iters']}"
          + (f" (trend vs {prior_ck['iters']})" if prior_ck else "") + " ===")
    ok, lines = evaluate(latest_ck, prior_ck, mono_tol=args.mono_tol, btn_floor=args.btn_floor,
                         aa_cap=args.aa_limp_cap, bb_fold_cap=args.bb_fold_cap,
                         trend_hard=int(latest_ck["iters"]) >= args.trend_hard_iters)
    for ln in lines:
        print("  " + ln)
    verdict = "PASS" if ok else "KILL"
    print(f"\nVERDICT: {verdict}")
    if not ok:
        print("KILL the run now (~$5 sunk): terminate the box + stop the babysitter; the aliasing "
              "pathology reproduced despite the hist key -- do not fund it to the backstop.")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"verdict": verdict, "iters": latest_ck["iters"],
                   "prior_iters": prior_ck["iters"] if prior_ck else None,
                   "gates": lines, "diag_json": diag_out,
                   "params": {"mono_tol": args.mono_tol, "btn_floor": args.btn_floor,
                              "aa_limp_cap": args.aa_limp_cap, "bb_fold_cap": args.bb_fold_cap}},
                  f, indent=2)
    print(f"wrote {args.out}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
