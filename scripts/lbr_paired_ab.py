"""LOCAL quality A/B: does the Hogwild lock-free path produce a WORSE (more LBR-exploitable) 6-max
blueprint than the fork-merge path, at a matched MC sample budget? Mirrors the f32-regret cert
([[f32-regret-cfr-rows-2026-06-29]]): made-hand abstraction (the only one native Lbr6Harness supports),
paired CRN LBR (same decks per seed for BOTH arms -> deck variance cancels -> tight paired-diff CI).

Both arms: same num_buckets, same n_deals, same deal budget, same training base seed. The only diff is
POKERBOT_HOGWILD on/off (the runtime toggle into the shared lock-free table). The native LBR uses the
SAME made-hand bucketing (POKERBOT_POSTFLOP_NUM_BUCKETS), so the blueprint is abstraction-matched.

Verdict on the PAIRED difference d_i = hog_bb_i - fork_bb_i (higher bb/100 = MORE exploitable = WORSE):
  paired 95% CI crosses 0      -> QUALITY-EQUIVALENT (PASS)
  paired CI entirely > 0       -> hogwild MORE exploitable (FAIL)
  paired CI entirely < 0       -> hogwild LESS exploitable (PASS, bonus)
"""
from __future__ import annotations

import os, sys, time, json

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)   # the hogwild-enabled battery_native.pyd staged here wins over any repo-root copy
sys.path.insert(1, ROOT)   # pokerbot package

os.environ.setdefault("POKERBOT_PREFLOP_LOSSLESS", "1")
NUM_BUCKETS = int(os.environ.get("QAB_NUM_BUCKETS", "200"))
os.environ.setdefault("POKERBOT_POSTFLOP_NUM_BUCKETS", str(NUM_BUCKETS))  # LBR made-hand buckets == train

import numpy as np
import battery_native as bn
from pokerbot.eval.lbr6 import LBR6Config
from pokerbot.eval import lbr_native as L

N_DEALS = int(os.environ.get("QAB_N_DEALS", "200"))
ITERS = int(os.environ.get("QAB_ITERS", "25000"))      # deal budget = ITERS * N_DEALS
HANDS = int(os.environ.get("QAB_HANDS", "8000"))       # LBR hands per seed
SEEDS = int(os.environ.get("QAB_SEEDS", "16"))         # paired seeds
HOG_CAP = int(os.environ.get("QAB_HOG_CAP", str(1 << 21)))
TRAIN_SEED = int(os.environ.get("QAB_TRAIN_SEED", "12345"))
CKPT_EVERY = max(1000, ITERS // 8)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def train_arm(hogwild: bool):
    tag = "HOG " if hogwild else "FORK"
    # made-hand abstraction (emd_frozen_path=None), depth_edges=[] (non-depth). hogwild flag selects arm.
    t = bn.NlheCFRTrainer(NUM_BUCKETS, [], N_DEALS, 6, float("-inf"), False, None, 0, hogwild, HOG_CAP)
    assert t.is_hogwild() == hogwild, f"arm toggle wrong: is_hogwild={t.is_hogwild()} expected {hogwild}"
    try:
        t.seed(TRAIN_SEED)  # if the trainer exposes a seed setter; else base_seed default (same for both)
    except Exception:
        pass
    log(f"{tag} train start: ITERS={ITERS:,} (deals={ITERS*N_DEALS:,}) num_buckets={NUM_BUCKETS} "
        f"n_deals={N_DEALS} hogwild={hogwild} cap={HOG_CAP:,}")
    t0 = time.time()
    while t.iters_done() < ITERS:
        nxt = min(t.iters_done() + CKPT_EVERY, ITERS)
        t.train(nxt - t.iters_done())
        el = time.time() - t0
        done = t.iters_done()
        ips = done / el if el else 0.0
        eta = (ITERS - done) / ips if ips else float("nan")
        log(f"{tag} progress {done:,}/{ITERS:,} ({100*done/ITERS:4.1f}%)  "
            f"{ips*N_DEALS/1e3:6.1f}K deals/s  {ips:5.1f} it/s  infosets={t.n_infosets():,}  ETA={eta:5.0f}s")
    wall = time.time() - t0
    snap = {bytes(k): np.asarray(v, dtype=np.float64) for k, v in t.snapshot().items()}
    log(f"{tag} done: wall={wall:.1f}s infosets={len(snap):,} "
        f"({ITERS*N_DEALS/wall/1e3:.1f}K deals/s)")
    return snap, wall


def lbr_pooled(snap, seeds):
    """Return per-seed bb/100 list for this blueprint over the given (paired) seed list."""
    vals = []
    for s in seeds:
        cfg = LBR6Config(hands=HANDS, seed=s, n_seats=6)
        r = L.lbr6_exploitability_native(snap, cfg, policy="blueprint", depth_edges=[])
        vals.append(float(r.bb_per_100))
    return vals


def ci95(vals):
    v = np.asarray(vals, dtype=np.float64)
    m = float(v.mean())
    se = float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else 0.0
    return m, se, m - 1.96 * se, m + 1.96 * se


def main() -> int:
    if not L.available():
        log("FATAL native battery_native LBR not available")
        return 1
    log(f"=== Hogwild vs fork-merge QUALITY A/B (made-hand K={NUM_BUCKETS}, paired LBR) ===")

    # LOAD-MODE: score pre-trained blueprints (e.g. trained on the 128-core box, where hogwild is fast,
    # then downloaded). Skips local training entirely — only the paired LBR runs (fast, needs JAX locally).
    LOAD_FORK = os.environ.get("QAB_LOAD_FORK")
    LOAD_HOG = os.environ.get("QAB_LOAD_HOG")
    if LOAD_FORK and LOAD_HOG:
        import pickle
        fork_snap = {bytes(k): np.asarray(v, np.float64)
                     for k, v in pickle.load(open(LOAD_FORK, "rb")).items()}
        hog_snap = {bytes(k): np.asarray(v, np.float64)
                    for k, v in pickle.load(open(LOAD_HOG, "rb")).items()}
        fork_wall = hog_wall = 0.0
        log(f"LOADED blueprints: fork={LOAD_FORK} ({len(fork_snap):,})  hog={LOAD_HOG} ({len(hog_snap):,})")
    else:
        fork_snap, fork_wall = train_arm(False)
        hog_snap, hog_wall = train_arm(True)

    # paired CRN seeds (same decks for BOTH arms) — the validator's documented pattern
    BASE_SEED = 0x5EEDBAB0
    seeds = [BASE_SEED ^ i for i in range(SEEDS)]
    assert len(set(seeds)) == SEEDS

    # one calibration anchor (blueprint-independent): always-fold must hit ~ +128 bb/100
    calib = LBR6Config(hands=HANDS, seed=0, n_seats=6)
    rf = L.lbr6_exploitability_native(None, calib, policy="fold")
    log(f"calibration always-fold = {rf.bb_per_100:+.1f} bb/100 "
        f"({'OK' if 110 <= rf.bb_per_100 <= 145 else 'OFF — harness suspect'})")

    log(f"LBR fork-merge arm ({SEEDS} seeds x {HANDS} = {SEEDS*HANDS:,} hands)...")
    fork_vals = lbr_pooled(fork_snap, seeds)
    log(f"LBR hogwild arm ({SEEDS} seeds x {HANDS} = {SEEDS*HANDS:,} hands)...")
    hog_vals = lbr_pooled(hog_snap, seeds)

    fm_m, fm_se, fm_lo, fm_hi = ci95(fork_vals)
    hg_m, hg_se, hg_lo, hg_hi = ci95(hog_vals)
    diff = [h - f for h, f in zip(hog_vals, fork_vals)]   # paired: hog - fork
    d_m, d_se, d_lo, d_hi = ci95(diff)

    if d_lo > 0:
        verdict = "FAIL — hogwild MORE exploitable (paired CI > 0)"
    elif d_hi < 0:
        verdict = "PASS (bonus) — hogwild LESS exploitable (paired CI < 0)"
    else:
        verdict = "PASS — quality-equivalent (paired CI crosses 0)"

    print("\n" + "=" * 74)
    print("QUALITY A/B RESULT  (LBR exploitability: higher bb/100 = MORE exploitable = WORSE)")
    print("=" * 74)
    print(f"  budget        : {ITERS:,} iters x {N_DEALS} = {ITERS*N_DEALS:,} deals/arm   made-hand K={NUM_BUCKETS}")
    print(f"  fork-merge    : {fm_m:+7.1f} bb/100   95% CI [{fm_lo:+.1f}, {fm_hi:+.1f}]   infosets={len(fork_snap):,}")
    print(f"  hogwild       : {hg_m:+7.1f} bb/100   95% CI [{hg_lo:+.1f}, {hg_hi:+.1f}]   infosets={len(hog_snap):,}")
    print(f"  PAIRED diff   : {d_m:+7.1f} bb/100   95% CI [{d_lo:+.1f}, {d_hi:+.1f}]   (hog - fork)")
    print(f"  per-seed diff : {[round(x,1) for x in diff]}")
    print(f"  VERDICT       : {verdict}")
    print("=" * 74)

    out = {
        "iters": ITERS, "n_deals": N_DEALS, "deals_per_arm": ITERS * N_DEALS, "num_buckets": NUM_BUCKETS,
        "hands": HANDS, "seeds": SEEDS, "train_wall": {"fork": fork_wall, "hog": hog_wall},
        "fork": {"mean": fm_m, "ci95": [fm_lo, fm_hi], "vals": fork_vals, "infosets": len(fork_snap)},
        "hog": {"mean": hg_m, "ci95": [hg_lo, hg_hi], "vals": hog_vals, "infosets": len(hog_snap)},
        "paired_diff": {"mean": d_m, "ci95": [d_lo, d_hi], "vals": diff},
        "verdict": verdict,
    }
    open(os.path.join(HERE, "quality_ab_result.json"), "w").write(json.dumps(out, indent=2))
    log("wrote quality_ab_result.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
