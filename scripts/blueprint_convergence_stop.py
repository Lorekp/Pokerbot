#!/usr/bin/env python3
"""Convergence-based STOP signal for the native CFR+ 6-max blueprint trainer.

WHY
---
The trainer (scripts/launch_native_cfr_training_spot.py) runs to a *fixed* --iters
(or a fixed wall-clock cap in the babysitter). That wastes compute: the average
strategy stops moving long before the iter/time cap. This helper turns "stop at a
fixed iter" into "stop AT CONVERGENCE", with the iter/time caps kept only as backstops.

WHAT IT MEASURES
----------------
Given an S3 run prefix, it lists the *versioned* checkpoints
(``{prefix}/checkpoints/iter_NNNNNN.nlcfr``), takes the two latest, deserializes each
with ``battery_native.NlheCFRTrainer(...).load_bytes(...)``, snapshots the average
strategy (``{[u8;16] -> [f64;11]}``), row-normalizes each policy over its support, and
computes the **mean L1 strategy-delta over the SHARED infosets** between the two
consecutive checkpoints. This is exactly the metric used in the retroactive iter62000
study (artifacts/blueprint_convergence_iter62000.md) where the bp6max_pluribus run
demonstrably plateaued.

This is ABSTRACTION-AGNOSTIC: ``load_bytes``/``snapshot`` read the tables saved inside
the blob, so the constructor's (num_buckets, depth_edges, n_deals) do not affect the
result. The same helper therefore works for the better-blueprint run regardless of which
richer --num-buckets / --depth-edges the abstraction sweep settles on.

CONVERGENCE DECISION
--------------------
The helper itself is STATELESS about history except via a tiny JSON state file
(``--state-file``) that records the consecutive count of "below threshold" rounds. It
emits one of:
    CONVERGED   : the latest strategy-delta is <= --threshold for >= --k consecutive
                  checkpoint-rounds (a sustained plateau, not a one-off dip).
    CONTINUING  : not yet plateaued (delta above threshold, or streak < k).
    UNKNOWN     : fewer than 2 versioned checkpoints exist yet (can't diff) -> never stop.

Exit code: 0 = CONVERGED, 1 = CONTINUING, 2 = UNKNOWN/insufficient-data, 3 = error.
The babysitter reads the printed ``STATUS=...`` line (robust) and/or the exit code.

THRESHOLD + K (derived from the iter62000 curve)
------------------------------------------------
On bp6max_pluribus (num_buckets=200, depth_edges=[]), the mean-L1 strategy-delta fell
0.314 (4k) -> 0.018 (50k) -> 0.014 (62k), the infoset table saturated by ~iter 48k, and
the delta's per-step relative drop flattened to ~3-4%/step from ~iter 50k. The 1/t
running-average diagnostic (t*delta) flattened toward a constant ~870-890 over the same
window, confirming the late movement is the mechanical running-average artifact rather
than genuine strategy change. So:
    --threshold 0.018  (the iter-50k plateau floor of THAT run)
    --k 3              (3 consecutive checkpoints below = ~6k iters of no real movement)
gives a stop at ~iter 54k on that run vs the 62k/64k it actually ran -> ~15% saved, with
no meaningful strategy difference.

NOTE on richer abstractions: a larger infoset table converges to a SLIGHTLY higher L1
floor (more thinly-visited rows move more per visit). The threshold is therefore exposed
as a flag; if the better-blueprint run's early curve floors higher, bump --threshold to
that run's observed plateau (the babysitter passes it through). The K-consecutive rule
and the 1/t-aware interpretation are abstraction-independent.

FIXED-SEED LBR TREND (optional secondary diagnostic, --lbr-trend)
-----------------------------------------------------------------
A CRN (common-random-numbers) idea: evaluate native LBR exploitability on a FIXED seed
set each round. Because the seeds/deals are shared across checkpoints, the deal noise
CANCELS in the checkpoint-to-checkpoint DELTA, so the *trend* is low-variance even though
the absolute CI is wide. In a 2p/HU zero-sum game "exploitability stopped DECREASING" would
be the gold-standard convergence signal.

**It is NOT a reliable convergence signal for the 6-MAX blueprint**, and this tool does
NOT gate the stop on it. Measured on bp6max_pluribus (fixed 500-800 deals, CRN):
LBR exploitability RISES as the blueprint converges (-196 bb/100 @2k -> +135 @40k ->
+160-260 @50k-62k) and then JITTERS in a band rather than plateauing. This is expected:
6-max has no Nash-exploitability guarantee (memory: 6max-lbr-exploitability,
native-lbr-validators), a self-play CFR blueprint converges toward equilibrium, not toward
1-ply-LBR robustness, and a sharper blueprint exposes MORE exploitable near-pure actions to
a clairvoyant-belief 1-ply LBR. The CRN delta cancels deal noise but not this structural
jitter. So for 6-max the trustworthy convergence signal is the STRATEGY-DELTA plateau above;
the LBR trend is logged only as context (and as the headline absolute-exploitability number
for a final report). --lbr-trend runs it and prints LBR=... but never flips STATUS.

(For a future HU/2p tabular run the LBR-decreasing signal WOULD be meaningful; this tool
leaves a clean seam — `--lbr-trend --lbr-gate` could then AND it into the stop. Not wired,
deliberately, because it is wrong for the NB=600 6-max run this targets.)
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Ensure the local battery_native build is importable. The convergence-stop runs LOCALLY (in the
# babysitter's polling loop), not on the box, so it needs the Windows .pyd on sys.path. We probe the
# canonical release dirs before declaring the script unrunnable -- the build is deterministic and
# idempotent so missing means "not built yet" (cargo build --release --features python at
# tools/battery_native/), not "build failed".
#
# CRITICAL: prefer the crate that matches the babysitter's --crate-src. The default launch uses
# runs/bp_crate_emd/battery_native (the ISOLATED EMD crate) because it has the NlheCFRTrainer class
# this script needs. The main tools/battery_native crate (the search/eval surface) replaced
# NlheCFRTrainer with SubgameSolver -- importing the wrong crate produces "module has no attribute
# 'NlheCFRTrainer'" on every poll. The EMD-first probe order makes the right choice when both exist.
_BATTERY_NATIVE_CANDIDATES = [
    ROOT / "runs" / "bp_crate_emd" / "battery_native" / "target" / "release",
    ROOT / "runs" / "bp_crate_frozen" / "battery_native" / "target" / "release",
    ROOT / "tools" / "battery_native" / "target" / "release",
]
for _cand in _BATTERY_NATIVE_CANDIDATES:
    if _cand.is_dir():
        _cand_str = str(_cand)
        if _cand_str not in sys.path:
            sys.path.insert(0, _cand_str)
        # On Windows the binary is .pyd; on Linux .so. The import probe below covers both.
        break

_ITER_RE = re.compile(r"/iter_(\d+)\.nlcfr$")


def _eprint(*a):
    print(*a, file=sys.stderr, flush=True)


def list_versioned_checkpoints(s3, bucket: str, run_prefix: str):
    """Return [(iters_done:int, key:str), ...] sorted ascending by iters_done.

    Only the versioned ``iter_NNNNNN.nlcfr`` objects (NOT ``latest.nlcfr``, which is a
    moving pointer we can't diff against a *prior* one reliably).
    """
    ckpt_prefix = f"{run_prefix}/checkpoints/"
    out = []
    token = None
    while True:
        kw = dict(Bucket=bucket, Prefix=ckpt_prefix)
        if token:
            kw["ContinuationToken"] = token
        resp = s3.list_objects_v2(**kw)
        for obj in resp.get("Contents", []):
            m = _ITER_RE.search(obj["Key"])
            if m:
                out.append((int(m.group(1)), obj["Key"]))
        if resp.get("IsTruncated"):
            token = resp.get("NextContinuationToken")
        else:
            break
    out.sort(key=lambda t: t[0])
    return out


def _download(s3, bucket: str, key: str) -> bytes:
    obj = s3.get_object(Bucket=bucket, Key=key)
    return obj["Body"].read()


def _norm_row(row):
    s = sum(row)
    if s > 0:
        return [v / s for v in row]
    return [0.0] * len(row)


def snapshot_from_blob(blob: bytes, num_buckets: int, depth_edges, n_deals: int):
    """Deserialize a .nlcfr blob and return (iters_done, {key16 -> [f64;11]}).

    The constructor args are irrelevant to deserialization (the blob carries its own
    tables) but the binding requires them; defaults match the deployed serve abstraction.
    """
    import battery_native as bn  # local import: only needed on the Windows .pyd host

    t = bn.NlheCFRTrainer(num_buckets, list(depth_edges), n_deals)
    if not t.load_bytes(blob):
        raise ValueError("NlheCFRTrainer.load_bytes returned False (corrupt/foreign blob)")
    return t.iters_done(), t.snapshot()


def _fixed_seed_lbr(snapshot: dict, args) -> float:
    """CRN fixed-seed native LBR exploitability (bb/100) of `snapshot`. Diagnostic only.

    Uses pokerbot.eval.lbr_native (the native rayon harness). The FIXED (args.lbr_seed, args.lbr_hands)
    means the same deals/seeds are replayed every round, so the round-to-round delta is CRN-clean even
    though the absolute CI is wide. Raises if the native harness is unavailable (caller swallows it).
    """
    from pokerbot.eval.lbr_native import lbr6_exploitability_native, available  # noqa: PLC0415
    from pokerbot.eval.lbr6 import LBR6Config  # noqa: PLC0415

    if not available():
        raise RuntimeError("native Lbr6Harness unavailable (build battery_native --features python)")
    cfg = LBR6Config(hands=int(args.lbr_hands), seed=int(args.lbr_seed), n_seats=int(args.lbr_n_seats))
    # depth_edges passed through so a depth-bucketed table still keys correctly; [] for NB=600.
    res = lbr6_exploitability_native(snapshot, cfg, policy="blueprint",
                                     depth_edges=list(args.depth_edges) or None)
    return float(res.bb_per_100)


def strategy_delta(snap_new: dict, snap_old: dict):
    """Mean L1 distance between the two normalized 11-action policies over SHARED infosets.

    Returns (mean_l1, shared_count, max_l1, frac_rows_changed).
    """
    shared = 0
    tot = 0.0
    mx = 0.0
    changed = 0
    for k, cur in snap_new.items():
        prev = snap_old.get(k)
        if prev is None:
            continue
        shared += 1
        d = 0.0
        a = _norm_row(cur)
        b = _norm_row(prev)
        for x, y in zip(a, b):
            d += abs(x - y)
        tot += d
        if d > mx:
            mx = d
        if d > 1e-9:
            changed += 1
    mean = (tot / shared) if shared else 0.0
    frac = (changed / shared) if shared else 0.0
    return mean, shared, mx, frac


def _load_state(path: str) -> dict:
    if not path or not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:  # noqa: BLE001
        return {}


def _save_state(path: str, state: dict):
    if not path:
        return
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)
    os.replace(tmp, path)


def evaluate(args) -> int:
    import boto3  # local import: keeps `--help`/`py_compile` dependency-free

    s3 = boto3.client("s3", region_name=args.region)
    run_prefix = args.run_prefix.rstrip("/")

    ckpts = list_versioned_checkpoints(s3, args.bucket, run_prefix)
    if len(ckpts) < 2:
        have = ckpts[-1][0] if ckpts else None
        print(f"STATUS=UNKNOWN reason=need>=2-versioned-checkpoints have={len(ckpts)} "
              f"latest_iter={have}")
        return 2

    (it_old, key_old), (it_new, key_new) = ckpts[-2], ckpts[-1]

    # State carries the consecutive below-threshold streak AND the last pair we scored,
    # so a babysitter re-poll on the SAME latest checkpoint does not double-count the streak.
    state = _load_state(args.state_file)
    pair_id = f"{it_old}->{it_new}"
    if state.get("last_pair") == pair_id and not args.force:
        streak = int(state.get("streak", 0))
        delta = state.get("last_delta")
        converged = streak >= args.k
        status = "CONVERGED" if converged else "CONTINUING"
        print(f"STATUS={status} reason=already-scored-this-pair pair={pair_id} "
              f"delta={delta} streak={streak}/{args.k} threshold={args.threshold}")
        return 0 if converged else 1

    t0 = time.time()
    blob_old = _download(s3, args.bucket, key_old)
    blob_new = _download(s3, args.bucket, key_new)
    iters_old, snap_old = snapshot_from_blob(blob_old, args.num_buckets, args.depth_edges, args.n_deals)
    iters_new, snap_new = snapshot_from_blob(blob_new, args.num_buckets, args.depth_edges, args.n_deals)
    mean_l1, shared, max_l1, frac = strategy_delta(snap_new, snap_old)
    n_new_infosets = len(snap_new) - len(snap_old)
    dt = time.time() - t0

    below = mean_l1 <= args.threshold
    prev_streak = int(state.get("streak", 0))
    streak = (prev_streak + 1) if below else 0
    converged = streak >= args.k

    # Optional secondary diagnostic: CRN fixed-seed LBR exploitability of the LATEST snapshot. Logged
    # only; does NOT touch `converged`/STATUS (it rises then jitters for 6-max -- see docstring).
    lbr_bb100 = None
    lbr_crn_delta = None
    if args.lbr_trend:
        try:
            lbr_bb100 = _fixed_seed_lbr(snap_new, args)
            prev_lbr = state.get("last_lbr_bb100")
            if prev_lbr is not None:
                lbr_crn_delta = lbr_bb100 - prev_lbr
        except Exception as e:  # noqa: BLE001
            _eprint(f"WARN: --lbr-trend failed (diagnostic only, ignored): {type(e).__name__}: {e}")

    state.update({
        "last_pair": pair_id,
        "last_delta": mean_l1,
        "streak": streak,
        "threshold": args.threshold,
        "k": args.k,
        "iters_new": iters_new,
        "n_infosets": len(snap_new),
        "updated": time.time(),
    })
    if lbr_bb100 is not None:
        state["last_lbr_bb100"] = lbr_bb100
    _save_state(args.state_file, state)

    status = "CONVERGED" if converged else "CONTINUING"
    lbr_txt = ""
    if lbr_bb100 is not None:
        dtxt = f"{lbr_crn_delta:+.2f}" if lbr_crn_delta is not None else "n/a"
        lbr_txt = f" LBR={lbr_bb100:+.1f} LBR_crn_delta={dtxt} (DIAGNOSTIC-ONLY-NOT-A-STOP-SIGNAL)"
    # Machine-readable single line (the babysitter greps STATUS=) + human detail.
    print(f"STATUS={status} run={run_prefix} pair={pair_id} "
          f"delta={mean_l1:.6f} threshold={args.threshold} below={below} "
          f"streak={streak}/{args.k} shared={shared} new_infosets={n_new_infosets} "
          f"max_l1={max_l1:.4f} frac_moved={frac:.3f} n_infosets={len(snap_new)} "
          f"eval_s={dt:.1f}{lbr_txt}")
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump({
                "status": status, "delta": mean_l1, "threshold": args.threshold,
                "streak": streak, "k": args.k, "shared": shared,
                "iter_old": iters_old, "iter_new": iters_new,
                "new_infosets": n_new_infosets, "n_infosets": len(snap_new),
                "max_l1": max_l1, "frac_moved": frac,
                "lbr_bb100": lbr_bb100, "lbr_crn_delta": lbr_crn_delta,
            }, f, indent=2)
    return 0 if converged else 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Convergence-based STOP signal for the native CFR+ blueprint trainer.")
    p.add_argument("--run-prefix", required=True,
                   help="S3 key prefix of the run, e.g. "
                        "pokerbot/prod6000/native_cfr/bp6max_pluribus")
    p.add_argument("--bucket", default="video-processing-6789")
    p.add_argument("--region", default="eu-central-1")
    p.add_argument("--threshold", type=float, default=0.018,
                   help="mean-L1 strategy-delta plateau floor (derived from iter62000 = 0.018; "
                        "bump for a richer abstraction whose early curve floors higher)")
    p.add_argument("--k", type=int, default=3,
                   help="consecutive below-threshold checkpoint-rounds required to declare CONVERGED")
    p.add_argument("--state-file", default="",
                   help="JSON file persisting the below-threshold streak across polls "
                        "(REQUIRED for the K-consecutive rule to work across babysitter rounds)")
    p.add_argument("--json-out", default="", help="optional path to write a JSON result blob")
    p.add_argument("--num-buckets", type=int, default=200,
                   help="constructor arg only; irrelevant to deserialization (blob carries its tables)")
    p.add_argument("--depth-edges", type=float, nargs="*", default=[],
                   help="constructor arg only; irrelevant to deserialization")
    p.add_argument("--n-deals", type=int, default=1, help="constructor arg only; irrelevant")
    p.add_argument("--force", action="store_true",
                   help="re-score even if the latest pair was already scored (testing)")
    # --- optional secondary diagnostic: fixed-seed (CRN) LBR exploitability trend ---
    # Logged for context only; NEVER gates the stop (see the LBR section of the docstring: it RISES
    # then jitters for 6-max, so it is not a convergence signal here). Requires the native Lbr6Harness.
    p.add_argument("--lbr-trend", action="store_true",
                   help="ALSO compute a CRN fixed-seed native LBR exploitability of the latest checkpoint "
                        "and print LBR=...,LBR_crn_delta=... (diagnostic only; does NOT change STATUS)")
    p.add_argument("--lbr-hands", type=int, default=500, help="fixed-seed deal count for --lbr-trend")
    p.add_argument("--lbr-seed", type=int, default=20260624,
                   help="FIXED seed for --lbr-trend (must be constant across rounds for the CRN delta to cancel deal noise)")
    p.add_argument("--lbr-n-seats", type=int, default=6, help="table size for --lbr-trend")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return evaluate(args)
    except Exception as e:  # noqa: BLE001
        _eprint(f"ERROR: {type(e).__name__}: {e}")
        print(f"STATUS=ERROR reason={type(e).__name__}")
        return 3


if __name__ == "__main__":
    sys.exit(main())
