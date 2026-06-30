"""Cloud-side QUALITY training: train BOTH made-hand arms (fork-merge, hogwild) to a matched deal budget
and save each blueprint's average-strategy snapshot as a pickle, for LOCAL paired-LBR scoring.

We do the TRAINING on the box (where hogwild is fast — many cores) but score LBR LOCALLY (the native
Lbr6Harness root-dealer needs JAX, which we keep OFF the box). The snapshot is dict[bytes16 -> list[f64]]
(the normalized average strategy = exactly what validate_blueprint_native.py's raw-snapshot branch wants),
pickled with the stdlib — no numpy/JAX/pokerbot on the box.

Made-hand abstraction (num_buckets, emd_frozen=None) so the blueprint is abstraction-matched to the native
LBR (POKERBOT_POSTFLOP_NUM_BUCKETS). Both arms: SAME num_buckets, SAME n_deals, SAME deal budget, SAME
base seed — only the hogwild toggle differs. A core-saturating n_deals (default 1024) is used so BOTH arms
train fast at high core count (n_deals is the MC batch, not the abstraction — identical for both arms, so
the LBR comparison stays apples-to-apples)."""
import os, sys, time, pickle

sys.path.insert(0, os.environ.get("PYD_DIR", "."))
import battery_native as bn

NUM_BUCKETS = int(os.environ.get("QUAL_NUM_BUCKETS", "200"))
N_DEALS = int(os.environ.get("QUAL_N_DEALS", "1024"))
ITERS = int(os.environ.get("QUAL_ITERS", "12000"))          # 12000 x 1024 ~= 12.3M deals (~converged)
HOG_CAP = int(os.environ.get("HOG_CAP", str(1 << 21)))
OUT_DIR = os.environ.get("OUT_DIR", "/var/log/hogwild")
CKPT = max(1000, ITERS // 10)


def _log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def train(hogwild):
    tag = "hog" if hogwild else "fork"
    t = bn.NlheCFRTrainer(NUM_BUCKETS, [], N_DEALS, 6, float("-inf"), False, None, 0, hogwild, HOG_CAP)
    assert t.is_hogwild() == hogwild, f"toggle wrong arm={tag}"
    _log(f"QUAL {tag} start: ITERS={ITERS:,} deals={ITERS*N_DEALS:,} num_buckets={NUM_BUCKETS} "
         f"n_deals={N_DEALS} hogwild={hogwild}")
    t0 = time.time()
    while t.iters_done() < ITERS:
        nxt = min(t.iters_done() + CKPT, ITERS)
        t.train(nxt - t.iters_done())
        el = time.time() - t0
        done = t.iters_done()
        ips = done / el if el else 0.0
        eta = (ITERS - done) / ips if ips else 0.0
        _log(f"QUAL {tag} {done:,}/{ITERS:,} ({100*done/ITERS:4.0f}%) "
             f"{ips*N_DEALS/1e3:7.0f}K deals/s infosets={t.n_infosets():,} ETA={eta:5.0f}s")
    snap = {bytes(k): list(v) for k, v in t.snapshot().items()}
    p = os.path.join(OUT_DIR, f"blueprint_{tag}.pkl")
    with open(p, "wb") as f:
        pickle.dump(snap, f, protocol=4)
    _log(f"QUAL {tag} saved {p} infosets={len(snap):,} wall={time.time()-t0:.0f}s")
    return p


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    _log(f"=== Hogwild QUALITY training (made-hand K={NUM_BUCKETS}); LBR scored LOCALLY ===")
    for hw in (False, True):
        train(hw)
    open(os.path.join(OUT_DIR, "QUAL_TRAIN_DONE"), "w").write("ok")
    _log("QUAL_TRAIN_DONE — both blueprints saved; run paired LBR locally")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
