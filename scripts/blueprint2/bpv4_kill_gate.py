#!/usr/bin/env python3
"""Mid-run KILL GATE for a blueprint-v4 Stage-0 cloud FULLHIST arm (design v2 sec 6.3).

Schedule: run at ~20k AND ~30k iters (the bp3 lesson: the 30k check would have saved the run;
cost at risk ~$5 / ~$12). Downloads the latest bpv4 S3 ladder checkpoint (plus the immediately
preceding ladder point for the trend check), STREAM-extracts the canonical preflop nodes with the
FULLHIST key space (preflop_root_diag.py --full-hist node/key machinery, run memory-safe via
bpv4_stage0_gates.scan_checkpoint -- NEVER load_checkpoint, a fullhist blob is 30-55 GB), and
evaluates the pre-registered gates calibrated at the mid-run point:

  G1  monotonicity  : UTG <= HJ <= CO <= BTN open%, each step >= -0.5pt tolerance, BTN > UTG.
  G2  BTN open      : >= 20% (HARD at every check). At >= --trend-hard-iters (default 30k) the
                      TREND form is HARD too: decline prior->latest >= -3pt.
  G3  AA-BTN-limp   : <= 30% (HARD). At >= --trend-hard-iters: non-rising prior->latest (HARD).
  G4  BB-vs-BTN-open fold : <= 70% (fullhist: weighted over the arm's own BTN open-size mix,
                      mirroring bpv4_stage0_gates.py).
  G6  jam sanity    : BTN RFI all-in share of open <= 10%.

Below --trend-hard-iters (the 20k check) the G2/G3 trend clauses are ADVISORY (WARN only) -- the
E5-class positive controls slide early, so an early hard trend-kill would kill a validated-good arm
(same discipline as check_prefhist_kill_gates.py). The hard caps G1/G2-level/G3-level/G4/G6 apply at
every check.

NEGATIVE CONTROL (bp3 kill-gate pattern): the OPPOSITE (summary) key space must read 0 rows from a
fullhist checkpoint -- a nonzero count means the extraction key space is wrong / the run is not
training fullhist, so the gate output cannot be trusted: ABORT (rc 2), do not emit a verdict.

Prints PASS/KILL with every number. KILL => terminate the box + stop the babysitter:
  aws ec2 terminate-instances --instance-ids <iid> --region eu-central-1
Field strength is still decided by the FINAL Stage-0 gates (bpv4_stage0_gates.py @30k/40k) + the G5
probe; this gate only prevents paying a fullhist arm to the backstop once it already reproduces the
limp-trap pathology.

Usage (default = latest bpv4_h200 ladder point on S3):
  python scripts/blueprint2/bpv4_kill_gate.py
  python scripts/blueprint2/bpv4_kill_gate.py --iter 20000            # pin a ladder point
  python scripts/blueprint2/bpv4_kill_gate.py --run-subdir bpv4_h1000
  python scripts/blueprint2/bpv4_kill_gate.py --checkpoint latest.nlcfr --prior-checkpoint prev.nlcfr
  python scripts/blueprint2/bpv4_kill_gate.py --iter 20000 --no-prior # skip the trend-baseline download

ROBUST BIG-PULL ROUTE (recommended on this box -- a fullhist checkpoint is 30-55 GB, and python's
urllib3 cannot verify the SSL-inspecting proxy while the AWS CLI can): download with the CLI (it has
a live progress bar + a working cert bundle), then point the gate at the local file:
  aws s3 cp s3://video-processing-6789/pokerbot/prod6000/native_cfr/bpv4_h200/checkpoints/iter_020000.nlcfr \
      artifacts/bpv4_ladder/bpv4_h200_iter20000.nlcfr --region eu-central-1
  python scripts/blueprint2/bpv4_kill_gate.py --checkpoint artifacts/bpv4_ladder/bpv4_h200_iter20000.nlcfr

PRE-MENU (arm-3): for an arm trained with the WIDENED preflop raise menu (engine ids 5/6/7/8/9 =
0.66/0.75/1.0/1.5/2.0 pot, trainer env POKERBOT_PRE_KEEP="2,3,4,5,6"), pass --pre-menu arm3 (or
env POKERBOT_PRE_MENU=arm3). Ids 5/6 then count as opens in G1/G2/G6, the two BB-response nodes
vs the NEW BTN open sizes (vsopen_BB_200 / vsopen_BB_212) are added, and the G4 weighting spans
all 5 sizes (weighted by the arm's own BTN open-size mix, identical construction). DEFAULT IS
BYTE-IDENTICAL to the pre-arm-3 instrument; off-default runs stamp "pre_menu" +
"open_raise_ids" into params so evidence files are self-describing (absence == default 3-size).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import threading
import time
import warnings

# This box reaches S3 through an SSL-inspecting proxy: python's urllib3 cannot verify the re-signed
# chain, so boto3 is used with verify=False (the repo convention -- see poll_dcfr_curve.py /
# poll_hogthru.py) and the resulting per-request InsecureRequestWarning is silenced (a 51 GB download
# otherwise emits dozens of them). The AWS CLI ships its own working CA bundle, so `aws s3 cp` stays
# fully TLS-verified -- prefer it for the big pull (see the ROBUST BIG-PULL ROUTE in the docstring).
try:
    import urllib3  # noqa: E402

    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
except Exception:  # pragma: no cover - urllib3 is always present via botocore
    warnings.filterwarnings("ignore", message="Unverified HTTPS request")

_HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, ".."))

# The FULLHIST node/key machinery (preflop_root_diag --full-hist) + the memory-safe streaming
# extractor, reused verbatim from the final Stage-0 gate evaluator (single source of truth).
import bpv4_stage0_gates as g0  # noqa: E402
import preflop_root_diag as prd  # noqa: E402

DEFAULT_BUCKET = "video-processing-6789"
DEFAULT_PREFIX = "pokerbot/prod6000/native_cfr"
DEFAULT_RUN = "bpv4_h200"
DEFAULT_REGION = "eu-central-1"

DETAIL_HANDS = ("AA", "KK", "QQ", "AKs", "K9s", "72o")


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


class _Progress:
    """Thread-safe, throttled S3 download progress line (s3transfer calls it from N worker threads)."""

    def __init__(self, total: int, every_s: float = 2.0):
        self._total = max(1, int(total))
        self._done = 0
        self._every = every_s
        self._t0 = time.time()
        self._last = 0.0
        self._lock = threading.Lock()

    def __call__(self, n: int) -> None:
        with self._lock:
            self._done += n
            now = time.time()
            if now - self._last < self._every and self._done < self._total:
                return
            self._last = now
            el = max(1e-6, now - self._t0)
            rate = self._done / el
            pct = 100.0 * self._done / self._total
            eta = (self._total - self._done) / rate if rate > 0 else 0.0
            sys.stdout.write(
                f"\r[kill-gate]   {self._done / 1e9:6.2f}/{self._total / 1e9:.2f} GB "
                f"({pct:5.1f}%)  {rate / 1e6:6.1f} MB/s  ETA {eta / 60:5.1f} min   ")
            sys.stdout.flush()


def _download(s3, bucket: str, key: str, dst: str) -> str:
    """Download one checkpoint with a live progress line; size-verified cache reuse.

    Writes to a .part temp then atomically renames, so an interrupted download never masquerades as a
    complete cache hit (the old check trusted any existing file, so a Ctrl-C mid-pull left a corrupt
    'cache' that the next run would have scanned)."""
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    total = int(s3.head_object(Bucket=bucket, Key=key)["ContentLength"])
    if os.path.exists(dst) and os.path.getsize(dst) == total:
        print(f"[kill-gate] reusing cached {dst} ({total / 1e9:.1f} GB)", flush=True)
        return dst
    free = shutil.disk_usage(os.path.dirname(dst) or ".").free
    if free < total * 1.1:
        print(f"[kill-gate] WARNING: only {free / 1e9:.1f} GB free on the cache drive for a "
              f"{total / 1e9:.1f} GB checkpoint -- the download may fail; free space or pass "
              f"--cache-dir on a bigger drive.", flush=True)
    tmp = dst + ".part"
    print(f"[kill-gate] downloading s3://{bucket}/{key} -> {dst} ({total / 1e9:.1f} GB)", flush=True)
    s3.download_file(bucket, key, tmp, Callback=_Progress(total))
    os.replace(tmp, dst)
    print(f"\n[kill-gate] downloaded {dst}", flush=True)
    return dst


def _extract(path: str, node_list, keys, neg_keys, detail_cls):
    """Memory-safe streaming extraction of the fullhist nodes off one checkpoint.

    Returns (iters, nodes_dict, m1_dict, neg_present). NEVER loads the full table into a dict --
    scan_checkpoint mmaps the blob and keeps only the ~few-thousand wanted node keys."""
    wanted = set(keys.values()) | set(neg_keys.values())
    it, n_cum, n_strat, cum_hits, strat_hits = g0.scan_checkpoint(path, wanted)
    nodes = g0.extract_nodes(node_list, keys, cum_hits, strat_hits, detail_cls)
    neg_present = sum(1 for k in set(neg_keys.values()) if k in strat_hits or k in cum_hits)
    m1 = {"iters": it, "cum_rows": n_cum, "strat_rows": n_strat,
          "blob_bytes": os.path.getsize(path)}
    return it, nodes, m1, neg_present


def _btn_open(nodes) -> float | None:
    return g0.open_pct(nodes["rfi_BTN"])  # already *100


def _aa_limp(nodes) -> float | None:
    aa = nodes["rfi_BTN"]["detail"].get("AA", {}).get("avg", {})
    return aa.get("CALL")


def _bb_fold_weighted(nodes) -> tuple[float | None, dict]:
    """BB fold vs BTN open, weighted by the arm's own BTN open-size mix (mirrors bpv4_stage0_gates;
    under --pre-menu arm3 the g0.bb_size_weights() table spans all 5 open sizes)."""
    btn = nodes["rfi_BTN"]["avg"]
    mix = btn["by_action"] if btn else {}
    w = {node: mix.get(lbl, 0.0) for node, lbl in g0.bb_size_weights()}
    per_size = {}
    acc = used = 0.0
    for n, wt in w.items():
        a = nodes.get(n, {}).get("avg")
        if a is not None:
            per_size[n] = a["fold"]
            acc += wt * a["fold"]
            used += wt
    return (acc / used if used > 0 else None), per_size


def _jam_share(nodes) -> float | None:
    a = nodes["rfi_BTN"]["avg"]
    if a and a["open_or_raise"] > 0:
        return a["allin"] / a["open_or_raise"]
    return None


def evaluate(latest, prior, *, mono_tol_pt: float, btn_floor_pct: float, aa_cap: float,
             bb_fold_cap: float, jam_cap: float, decline_floor_pt: float,
             trend_hard: bool) -> tuple[bool, list[str], dict]:
    lines: list[str] = []
    detail: dict = {}
    ok = True
    rfi = ["rfi_UTG", "rfi_HJ", "rfi_CO", "rfi_BTN"]

    opens = {n: g0.open_pct(latest[n]) for n in rfi}
    detail["opens_pct"] = opens
    lines.append("opens (avg open_or_raise): " + "  ".join(
        f"{n.split('_')[1]}={'NA' if opens[n] is None else f'{opens[n]:.1f}%'}" for n in rfi))

    # G1 monotonicity.
    if any(opens[n] is None for n in rfi):
        g1 = False
        lines.append("G1 monotonicity: FAIL (an RFI node read 0 rows -- checkpoint too early or "
                     "wrong key space; verify the run exports POKERBOT_FULLHIST_KEY=1)")
    else:
        seq = [opens[n] for n in rfi]
        steps_ok = all(seq[i + 1] - seq[i] >= -mono_tol_pt for i in range(3))
        g1 = steps_ok and seq[3] > seq[0]
        lines.append(f"G1 monotonicity (each step >= -{mono_tol_pt:.1f}pt, BTN>UTG): "
                     f"{'PASS' if g1 else 'FAIL'}")
    ok &= g1
    detail["G1"] = g1

    # G2 BTN open floor (+ trend when hard).
    btn = opens["rfi_BTN"]
    btn_prev = g0.open_pct(prior["rfi_BTN"]) if prior else None
    decline = None if (btn is None or btn_prev is None) else btn - btn_prev
    g2_level = btn is not None and btn >= btn_floor_pct
    g2_trend_ok = decline is None or decline >= decline_floor_pt
    g2 = g2_level and (g2_trend_ok or not trend_hard)
    ok &= g2
    tnote = ("no prior ladder point" if btn_prev is None else
             f"prior {btn_prev:.1f}% -> latest {btn:.1f}% (decline {decline:+.1f}pt)")
    trend_status = ("HARD" if trend_hard else "advisory") + (
        "" if g2_trend_ok else " -- TREND BREACH"
        + ("" if trend_hard else " (WARN, advisory below trend-hard-iters)"))
    lines.append(f"G2 BTN open {'NA' if btn is None else f'{btn:.1f}%'} >= {btn_floor_pct:.0f}% "
                 f"AND decline >= {decline_floor_pt:+.0f}pt [{trend_status}]: "
                 f"{'PASS' if g2 else 'FAIL'}  [{tnote}]")
    detail["G2"] = {"btn_open_pct": btn, "btn_open_prev_pct": btn_prev, "decline_pt": decline,
                    "level_ok": g2_level, "trend_ok": g2_trend_ok, "trend_hard": trend_hard,
                    "pass": g2}

    # G3 AA-BTN-limp cap (+ non-rising trend when hard).
    aa = _aa_limp(latest)
    aa_prev = _aa_limp(prior) if prior else None
    if aa is None:
        g3 = False
        lines.append("G3 AA-BTN-limp: FAIL (AA row missing at rfi_BTN -- checkpoint too early / "
                     "wrong key space)")
    else:
        g3_level = aa <= aa_cap
        g3_trend_ok = aa_prev is None or aa <= aa_prev + 1e-9  # non-rising
        g3 = g3_level and (g3_trend_ok or not trend_hard)
        tnote3 = ("no prior ladder point" if aa_prev is None else
                  f"prior {aa_prev*100:.1f}% -> latest {aa*100:.1f}% "
                  f"({'non-rising' if g3_trend_ok else 'RISING'})")
        ts3 = ("HARD" if trend_hard else "advisory") + (
            "" if g3_trend_ok else " -- RISING"
            + ("" if trend_hard else " (WARN, advisory)"))
        lines.append(f"G3 AA-BTN-limp {aa*100:.1f}% <= {aa_cap*100:.0f}% AND non-rising [{ts3}]: "
                     f"{'PASS' if g3 else 'FAIL'}  [{tnote3}]")
        detail["G3"] = {"aa_limp": aa, "aa_limp_prev": aa_prev, "level_ok": g3_level,
                        "trend_ok": g3_trend_ok, "trend_hard": trend_hard, "pass": g3}
    ok &= g3

    # G4 BB fold vs BTN open (weighted).
    bb_fold, per_size = _bb_fold_weighted(latest)
    g4 = bb_fold is not None and bb_fold <= bb_fold_cap
    ok &= g4
    lines.append(f"G4 BB-vs-BTN-open fold {'NA' if bb_fold is None else f'{bb_fold*100:.1f}%'} "
                 f"<= {bb_fold_cap*100:.0f}% (open-size-weighted): {'PASS' if g4 else 'FAIL'}  "
                 f"[per-size folds: {{ {', '.join(f'{k}={v*100:.1f}%' for k, v in per_size.items())} }}]")
    detail["G4"] = {"bb_fold_weighted": bb_fold, "per_size": per_size, "pass": g4}

    # G6 jam sanity.
    js = _jam_share(latest)
    g6 = js is not None and js <= jam_cap
    ok &= g6
    lines.append(f"G6 BTN jam share of open {'NA' if js is None else f'{js*100:.1f}%'} "
                 f"<= {jam_cap*100:.0f}%: {'PASS' if g6 else 'FAIL'}")
    detail["G6"] = {"jam_share": js, "pass": g6}

    return ok, lines, detail


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bucket", default=DEFAULT_BUCKET)
    ap.add_argument("--prefix", default=DEFAULT_PREFIX)
    ap.add_argument("--run-subdir", default=DEFAULT_RUN,
                    help="S3 run subdir (bpv4_h200 | bpv4_h1000 | bpv4_hd200 | bpv4_he200)")
    ap.add_argument("--region", default=DEFAULT_REGION)
    ap.add_argument("--iter", type=int, default=None,
                    help="pin a ladder point (default: the latest iter_XXXXXX.nlcfr)")
    ap.add_argument("--checkpoint", default=None, help="local .nlcfr override (skips S3)")
    ap.add_argument("--prior-checkpoint", default=None,
                    help="local prior-ladder-point .nlcfr for the G2/G3 trend check")
    ap.add_argument("--no-prior", action="store_true",
                    help="skip downloading the preceding ladder point (the G2/G3 trend baseline). "
                         "At the 20k check the trend is advisory anyway, so this saves a ~28-55 GB "
                         "download; the hard caps G1/G2-level/G3-level/G4/G6 still fully evaluate.")
    ap.add_argument("--mono-tol-pt", type=float, default=0.5)
    ap.add_argument("--btn-floor-pct", type=float, default=20.0)
    ap.add_argument("--aa-limp-cap", type=float, default=0.30)
    ap.add_argument("--bb-fold-cap", type=float, default=0.70)
    ap.add_argument("--jam-cap", type=float, default=0.10)
    ap.add_argument("--decline-floor-pt", type=float, default=-3.0,
                    help="G2 trend: BTN open decline prior->latest must be >= this (pt)")
    ap.add_argument("--trend-hard-iters", type=int, default=30000,
                    help="G2/G3 trend clauses are ADVISORY below this and HARD from it (design 6.3: "
                         "trend forms added AT the 30k check; early positive-control arms slide)")
    ap.add_argument("--pre-menu", choices=("default", "arm3"),
                    default=os.environ.get("POKERBOT_PRE_MENU", "default"),
                    help="preflop raise menu of the arm's trainer: 'default' = 3-size ids 7/8/9 "
                         "(byte-identical historical instrument) | 'arm3' = widened 5-size ids "
                         "5/6/7/8/9 (POKERBOT_PRE_KEEP=2..6). Env POKERBOT_PRE_MENU sets the "
                         "default.")
    ap.add_argument("--cache-dir", default=os.path.join(REPO, "artifacts", "bpv4_ladder"))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = args.out or os.path.join(REPO, "artifacts", f"bpv4_kill_gate_{args.run_subdir}.json")
    prd.set_pre_menu(args.pre_menu)

    # --- build the fullhist node list + keys ONCE (engine-replayed, exact) -------------------------
    node_list = g0.build_nodes("fullhist")
    keys = g0.node_keys(node_list, "fullhist")
    neg_name, neg_nodes = g0.wrong_space_nodes("fullhist")
    neg_keys = g0.node_keys(neg_nodes, neg_name)
    g0.build_nodes("fullhist")  # restore prd globals to the real mode after neg-space build
    detail_cls = {prd.CLASS_BY_NAME[h] for h in DETAIL_HANDS}

    latest_path, prior_path = args.checkpoint, args.prior_checkpoint
    latest_iters = prior_iters = None
    if latest_path is None:
        import boto3  # noqa: PLC0415
        s3 = boto3.client("s3", region_name=args.region, verify=False)
        run_prefix = f"{args.prefix}/{args.run_subdir}"
        ladder = _list_ladder(s3, args.bucket, run_prefix)
        if not ladder:
            print(f"ERROR: no iter_XXXXXX.nlcfr ladder under "
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
        if idx > 0 and prior_path is None and not args.no_prior:
            # Trend baseline = the immediately preceding ladder point (checkpoint cadence is 10k, so
            # adjacent points already carry a trend signal, unlike the bp3 500-iter cadence).
            prior_iters, prior_key = ladder[idx - 1]
            prior_path = _download(s3, args.bucket, prior_key,
                                   os.path.join(args.cache_dir, f"{args.run_subdir}_iter{prior_iters}.nlcfr"))

    os.makedirs(args.cache_dir, exist_ok=True)

    # --- extract (streaming, memory-safe) ----------------------------------------------------------
    latest_it, latest_nodes, latest_m1, latest_neg = _extract(
        latest_path, node_list, keys, neg_keys, detail_cls)
    print(f"[kill-gate] latest iter {latest_it}: strat_rows={latest_m1['strat_rows']:,} "
          f"cum_rows={latest_m1['cum_rows']:,} blob={latest_m1['blob_bytes']/1e9:.1f}GB "
          f"neg_ctrl_rows={latest_neg}", flush=True)
    prior_nodes = None
    prior_m1 = None
    if prior_path:
        prior_it, prior_nodes, prior_m1, prior_neg = _extract(
            prior_path, node_list, keys, neg_keys, detail_cls)
        print(f"[kill-gate] prior  iter {prior_it}: strat_rows={prior_m1['strat_rows']:,} "
              f"neg_ctrl_rows={prior_neg}", flush=True)
        if prior_neg != 0:
            print(f"ABORT: prior checkpoint negative control read {prior_neg} rows under the "
                  f"{neg_name} key space -- extraction key space is wrong; not emitting a verdict.")
            return 2

    if latest_neg != 0:
        print(f"ABORT: negative control read {latest_neg} rows under the {neg_name} key space -- "
              f"the checkpoint is NOT in the fullhist key space (run not training fullhist, or a "
              f"key/extractor bug). The gate numbers cannot be trusted; not emitting a verdict.")
        return 2

    trend_hard = int(latest_it) >= args.trend_hard_iters
    print(f"\n=== bpv4 {args.run_subdir} MID-RUN KILL GATE @ iters={latest_it}"
          + (f" (trend vs {prior_m1['iters']}, {'HARD' if trend_hard else 'advisory'})"
             if prior_m1 else "") + " ===")
    ok, lines, gdetail = evaluate(
        latest_nodes, prior_nodes, mono_tol_pt=args.mono_tol_pt, btn_floor_pct=args.btn_floor_pct,
        aa_cap=args.aa_limp_cap, bb_fold_cap=args.bb_fold_cap, jam_cap=args.jam_cap,
        decline_floor_pt=args.decline_floor_pt, trend_hard=trend_hard)
    for ln in lines:
        print("  " + ln)
    print(f"  M1: strat_rows={latest_m1['strat_rows']:,} cum_rows={latest_m1['cum_rows']:,} "
          f"blob={latest_m1['blob_bytes']/1e9:.2f}GB  neg_control_rows={latest_neg} (must be 0)")

    verdict = "PASS" if ok else "KILL"
    print(f"\nVERDICT: {verdict}")
    if not ok:
        print("KILL the run now: terminate the box (aws ec2 terminate-instances --instance-ids "
              "<iid> --region " + args.region + ") + stop the babysitter. The limp-trap pathology "
              "reproduced under fullhist -- do not fund it to the backstop.")

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump({
            "verdict": verdict, "run_subdir": args.run_subdir, "iters": latest_it,
            "prior_iters": prior_m1["iters"] if prior_m1 else None,
            "trend_hard": trend_hard, "gates": lines, "gate_detail": gdetail,
            "m1_latest": latest_m1, "m1_prior": prior_m1,
            "neg_control_rows": latest_neg,
            "params": {"mono_tol_pt": args.mono_tol_pt, "btn_floor_pct": args.btn_floor_pct,
                       "aa_limp_cap": args.aa_limp_cap, "bb_fold_cap": args.bb_fold_cap,
                       "jam_cap": args.jam_cap, "decline_floor_pt": args.decline_floor_pt,
                       "trend_hard_iters": args.trend_hard_iters,
                       # stamped ONLY off-default so the default JSON stays byte-identical to
                       # the pre-arm-3 instrument (absence == default 3-size menu).
                       **({"pre_menu": args.pre_menu,
                           "open_raise_ids": list(prd.RAISE_SET)}
                          if args.pre_menu != "default" else {})},
        }, f, indent=2)
    print(f"wrote {out}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
