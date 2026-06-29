"""On-box DCFR-vs-CFR+ LBR-vs-COMPUTE *curve* training.

Trains TWO arms to a matched deal budget, snapshotting the average-strategy blueprint at log-spaced
ITERATION milestones, and STREAMING each checkpoint to S3 the instant it is produced. The streaming is
the whole point of this rewrite: hogthru_1782747943 wrapped the run in a single `timeout 2700` and only
uploaded OUT_DIR at the very end, so the rc=124 kill dropped the summary. Here every checkpoint is an
independent S3 object + an updated manifest, so a hang/spot-kill/timeout loses at most the in-flight
milestone, never the curve.

Both arms are IDENTICAL except the algorithm:
  - cfrplus : dcfr=False  -> shipped CFR+ positive-regret floor; prune_threshold=-inf (RBP hook inert).
  - dcfr    : dcfr=True   -> signed-regret DCFR discount (alpha=1.5,beta=0,gamma=2) + ACTIVE Regret-Based
              Pruning (dcfr_prune_threshold, ~explore_frac explore-iters). The Pluribus recipe.
Same num_buckets, same n_deals, same default base seed, work_steal ON for BOTH (the banked load-imbalance
fix; we are NOT measuring work_steal here, we hold it fixed so the only variable is the algorithm).

Made-hand abstraction (emd_frozen=None) so the native Lbr6Harness can score the blueprints LOCALLY
(the JAX root-dealer stays off the box). LBR is scored afterwards by scripts/score_dcfr_curve_lbr.py over
every checkpoint of both arms -> the exploitability-vs-compute curve and the convergence-speed verdict.
"""
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.environ.get("PYD_DIR", "."))
import battery_native as bn  # noqa: E402

NUM_BUCKETS = int(os.environ.get("DCFR_NUM_BUCKETS", "200"))
N_DEALS = int(os.environ.get("DCFR_N_DEALS", "1024"))
MILESTONES = sorted(int(x) for x in os.environ.get(
    "DCFR_MILESTONES", "1000,2000,4000,8000,16000,32000,50000").split(",") if x.strip())
PRUNE_THRESHOLD = float(os.environ.get("DCFR_PRUNE_THRESHOLD", "-300000.0"))
EXPLORE_FRAC = float(os.environ.get("DCFR_EXPLORE_FRAC", "0.05"))
HOG_CAP = int(os.environ.get("DCFR_HOG_CAP", str(1 << 21)))
OUT_DIR = os.environ.get("OUT_DIR", "/var/log/dcfrcurve")
S3_DEST = os.environ.get("S3_DEST", "").rstrip("/")          # s3://bucket/.../dcfrcurve  ("" => no upload)
REGION = os.environ.get("REGION", "eu-central-1")

ARMS = [("cfrplus", False), ("dcfr", True)]
TOTAL = max(MILESTONES)


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def s3_cp(local, name):
    """Stream one artifact to S3 immediately (best-effort, bounded). No-op if S3_DEST unset (local smoke)."""
    if not S3_DEST:
        return
    dst = f"{S3_DEST}/{name}"
    try:
        r = subprocess.run(["aws", "s3", "cp", local, dst, "--region", REGION],
                           capture_output=True, text=True, timeout=180)
        if r.returncode != 0:
            log(f"WARN upload {name} rc={r.returncode}: {r.stderr.strip()[:160]}")
    except Exception as e:  # noqa: BLE001
        log(f"WARN upload {name} raised {e!r}")


def make_trainer(dcfr):
    # positional ctor (PyO3 signature): num_buckets, depth_edges, n_deals, n_seats, prune_threshold,
    # fast_merge, emd_frozen_path, batch_deals, hogwild, hogwild_capacity, work_steal, dcfr,
    # dcfr_prune_threshold, dcfr_explore_frac
    return bn.NlheCFRTrainer(NUM_BUCKETS, [], N_DEALS, 6, float("-inf"), False, None, 0,
                             False, HOG_CAP, True, dcfr, PRUNE_THRESHOLD, EXPLORE_FRAC)


def train_arm(arm, dcfr, manifest):
    t = make_trainer(dcfr)
    assert t.is_hogwild() is False, "hogwild must be OFF for the curve cert"
    log(f"=== ARM {arm} (dcfr={dcfr}) start: milestones={MILESTONES} n_deals={N_DEALS} "
        f"num_buckets={NUM_BUCKETS} work_steal=ON prune={PRUNE_THRESHOLD} explore={EXPLORE_FRAC} ===")
    t0 = time.time()
    for m in MILESTONES:
        if t.iters_done() >= m:
            continue
        t.train(m - t.iters_done())
        wall = time.time() - t0
        done = t.iters_done()
        cum_deals = done * N_DEALS
        ips = done / wall if wall else 0.0
        # snapshot -> pickle -> stream to S3 immediately
        snap = {bytes(k): list(v) for k, v in t.snapshot().items()}
        fn = f"blueprint_{arm}_{m}.pkl"
        path = os.path.join(OUT_DIR, fn)
        import pickle
        with open(path, "wb") as f:
            pickle.dump(snap, f, protocol=4)
        rec = {"arm": arm, "dcfr": dcfr, "milestone": int(done), "cum_deals": int(cum_deals),
               "wall_s": round(wall, 1), "infosets": len(snap), "deals_per_s": round(ips * N_DEALS, 1),
               "file": fn, "num_buckets": NUM_BUCKETS, "n_deals": N_DEALS}
        manifest["checkpoints"].append(rec)
        mpath = os.path.join(OUT_DIR, "manifest.json")
        with open(mpath, "w") as f:
            json.dump(manifest, f, indent=2)
        s3_cp(path, fn)
        s3_cp(mpath, "manifest.json")
        eta = (TOTAL - done) / ips if ips else 0.0
        log(f"ARM {arm} milestone {done:,}/{TOTAL:,} ({100*done/TOTAL:4.0f}%) "
            f"deals={cum_deals/1e6:6.2f}M {ips*N_DEALS/1e3:7.0f}K deals/s infosets={len(snap):,} "
            f"wall={wall:5.0f}s ETA={eta:5.0f}s -> streamed {fn}")
    log(f"=== ARM {arm} DONE wall={time.time()-t0:.0f}s infosets={t.n_infosets():,} ===")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    manifest = {"num_buckets": NUM_BUCKETS, "n_deals": N_DEALS, "milestones": MILESTONES,
                "prune_threshold": PRUNE_THRESHOLD, "explore_frac": EXPLORE_FRAC,
                "rayon_num_threads": os.environ.get("RAYON_NUM_THREADS", "(default)"),
                "checkpoints": []}
    log(f"=== DCFR-vs-CFR+ CURVE training (made-hand K={NUM_BUCKETS}); LBR scored LOCALLY ===")
    log(f"milestones={MILESTONES} -> deals {[m*N_DEALS for m in MILESTONES]}")
    for arm, dcfr in ARMS:
        train_arm(arm, dcfr, manifest)
    with open(os.path.join(OUT_DIR, "CURVE_TRAIN_DONE"), "w") as f:
        f.write("ok")
    s3_cp(os.path.join(OUT_DIR, "manifest.json"), "manifest.json")
    s3_cp(os.path.join(OUT_DIR, "CURVE_TRAIN_DONE"), "CURVE_TRAIN_DONE")
    log("CURVE_TRAIN_DONE -- both arms' checkpoints streamed; score paired LBR locally")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
