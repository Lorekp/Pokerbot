#!/usr/bin/env python3
"""Local driver for the native Rust CFR+ blueprint trainer (NlheCFRTrainer, tools/battery_native).

Runs the trainer on the local box (CPU rayon, no GPU), checkpoints periodically, and writes the final
blueprint as a ``TabularBlueprintAdapter``-ready pkl (``dict[bytes16 -> np.float32[11]]``). This is the
SAME trainer the AWS-spot launcher (``launch_native_cfr_training_spot.py``) runs — local is for a fast
proof / a smaller blueprint; spot c6i.32xlarge is for the full 5000-iter production run.

  python scripts/train_native_bp_local.py --iters 2000 --n-deals 200 --num-buckets 200 \
      --out checkpoints/native_bp_local

The trainer accumulates a persistent table across iterations, so checkpoints resume cleanly.
"""
from __future__ import annotations

import argparse
import os
import pickle
import time

import numpy as np


def _load_ext():
    """Import the native extension DIRECTLY as ``battery_native`` — NOT via ``native_solver`` (which pulls
    in jax -> ~55-83 XLA CPU threads that can RARELY deadlock the trainer's rayon + first GIL-release).
    The trainer is pure Rust and needs no jax; the direct import keeps the thread count minimal. Fall back
    to ``native_solver._try_load`` only if the direct import misses (e.g. a non-standard build location)."""
    try:
        import battery_native as bn  # noqa: PLC0415
        return bn
    except ImportError:
        from pokerbot.search import native_solver as ns  # noqa: PLC0415
        return ns._try_load()


def _fmt_eta(eta_s: float | None) -> str:
    """ASCII compact ETA (--/s/m/h)."""
    if eta_s is None:
        return "--"
    eta_s = int(eta_s)
    if eta_s < 90:
        return f"~{eta_s}s"
    if eta_s < 5400:
        return f"~{eta_s // 60}m"
    return f"~{eta_s / 3600:.1f}h"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=2000)
    ap.add_argument("--n-deals", type=int, default=200, help="deals fanned out per CFR+ iteration")
    ap.add_argument("--num-buckets", type=int, default=200, help="postflop bucket count (MUST match serve)")
    ap.add_argument("--n-seats", type=int, default=6,
                    help="table width: 6 = deployed full-ring blueprint (default), 7 = MTT tournament "
                         "anchor. 7 keys distinctly via the fingerprint's generic seat encoding; there "
                         "is no served blueprint for other widths (they fall back to 6 in the trainer).")
    ap.add_argument("--ckpt-every", type=int, default=500)
    ap.add_argument("--log-every", type=int, default=50,
                    help="print a progress line (iters done/target, iters/sec, ETA) every N iters. "
                         "Infoset count is only read at checkpoint lines (cheap n_infosets() if the "
                         "extension exposes it, else snapshot()) to keep the per-log step light.")
    ap.add_argument("--out", default="checkpoints/native_bp_local")
    a = ap.parse_args()

    ext = _load_ext()
    if ext is None or not hasattr(ext, "NlheCFRTrainer"):
        raise SystemExit("native NlheCFRTrainer not built (cargo build --release --features python)")

    os.makedirs(a.out, exist_ok=True)
    # n_seats is a newer 4th ctor arg (default 6). Pass it when requested; fall back to the 3-arg form on
    # an older extension (which is 6-seat only) so a 6-max run still works against a pre-n_seats build.
    if a.n_seats != 6:
        t = ext.NlheCFRTrainer(a.num_buckets, [], a.n_deals, a.n_seats)
    else:
        try:
            t = ext.NlheCFRTrainer(a.num_buckets, [], a.n_deals, a.n_seats)
        except TypeError:
            t = ext.NlheCFRTrainer(a.num_buckets, [], a.n_deals)
    print(f"trainer: n_seats={getattr(t, 'n_seats', lambda: a.n_seats)()} num_buckets={a.num_buckets} "
          f"n_deals={a.n_deals}", flush=True)
    # Cheap live infoset count if this build exposes it (strat_sum.len()); else fall back to the
    # expensive snapshot() at checkpoints only. A train running on an OLDER extension simply won't have
    # n_infosets() -> hasattr guards it. Never call it on the per-log (~50 iter) line.
    _has_cheap_count = hasattr(t, "n_infosets")

    def _infoset_count() -> int:
        if _has_cheap_count:
            try:
                return int(t.n_infosets())
            except Exception:  # noqa: BLE001 — count is diagnostic only
                return -1
        return len(t.snapshot())  # expensive; checkpoint lines only

    log_every = max(1, a.log_every)
    ckpt_every = max(1, a.ckpt_every)
    t0 = time.time()
    done = 0
    while done < a.iters:
        # Step in log_every-sized chunks so we can emit a progress line every ~log_every iters, but never
        # step PAST the next ckpt_every boundary -> checkpoints always land exactly on a ckpt_every multiple
        # even when log_every does not evenly divide ckpt_every.
        next_ckpt = ((done // ckpt_every) + 1) * ckpt_every
        step = min(log_every, a.iters - done, next_ckpt - done)
        t.train(step)
        done += step
        elapsed = time.time() - t0
        ips = done / elapsed if elapsed > 0 else 0.0
        eta = ((a.iters - done) / ips) if ips > 0 else None
        is_ckpt = (done % ckpt_every == 0) or (done >= a.iters)
        if is_ckpt:
            _tag = "" if a.n_seats == 6 else f"_{a.n_seats}p"
            with open(os.path.join(a.out, f"native_bp{_tag}_iter{done}.nlcfr"), "wb") as f:
                f.write(t.save_bytes())
            print(f"[{int(elapsed):5d}s] CKPT iters={done}/{a.iters}  ({ips:.2f} it/s)  "
                  f"ETA {_fmt_eta(eta)}  infosets={_infoset_count()}", flush=True)
        elif done % log_every == 0 or done >= a.iters:
            # Light progress line (no infoset count — that read is reserved for checkpoint lines).
            print(f"[{int(elapsed):5d}s] iters={done}/{a.iters}  ({ips:.2f} it/s)  "
                  f"ETA {_fmt_eta(eta)}", flush=True)

    # Final blueprint in the TabularBlueprintAdapter input shape: dict[bytes16 -> np.float32[11]].
    snap = {bytes(k): np.asarray(v, dtype=np.float32) for k, v in t.snapshot().items()}
    # Seat-tag the 7-max (and any non-6) blueprint so it never clobbers the deployed 6-max final.pkl.
    tag = "" if a.n_seats == 6 else f"_{a.n_seats}p"
    out_pkl = os.path.join(a.out, f"native_bp_final{tag}.pkl")
    with open(out_pkl, "wb") as f:
        pickle.dump(snap, f)
    print(f"DONE: {len(snap)} infosets in {int(time.time() - t0)}s -> {out_pkl}", flush=True)


if __name__ == "__main__":
    main()
