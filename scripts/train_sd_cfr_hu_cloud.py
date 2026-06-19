#!/usr/bin/env python3
"""
Single-machine SD-CFR HU training for EC2 spot: S3 checkpoint/resume + spot-interrupt + parallelism.

- Resumes from the latest S3 checkpoint on startup (so a spot interruption loses at most one
  in-flight iteration; the resume is bit-exact — validated by run_sd_cfr_hu_resume_compounding).
- Polls the IMDSv2 spot-termination notice in a background thread; on notice (or SIGTERM) it
  checkpoints the current state to S3 and exits cleanly.
- Runs each iteration's shards in parallel fork processes when --workers > 1 (Linux), else the
  single-threaded reference. Per-iteration RNG is seeded seed+n-1 so resume reproduces exactly.

Local dry-run: --local-only writes checkpoints to a local dir (no AWS) for validation.
"""

from __future__ import annotations

import argparse
import os
import signal
import threading
import time
import urllib.request
from pathlib import Path


def _flag_truthy(val: str | None) -> bool:
    return (val or "").lower() not in ("", "0", "false", "no")

import numpy as np

from pokerbot.training.sd_cfr_nlhe import SDCFRNLHEConfig, SDCFRNLHETrainer
from pokerbot.training.sd_cfr_nlhe_cloud import (
    LocalDirObjectStore,
    S3ObjectStore,
    checkpoint_from_store,
    checkpoint_to_store,
    latest_checkpoint_iteration,
)

_INTERRUPT = threading.Event()


def _spot_interrupt_watcher() -> None:
    """Set the interrupt flag when EC2 signals a spot termination (IMDSv2)."""
    base = "http://169.254.169.254/latest"
    while not _INTERRUPT.is_set():
        try:
            tok_req = urllib.request.Request(
                base + "/api/token",
                method="PUT",
                headers={"X-aws-ec2-metadata-token-ttl-seconds": "60"},
            )
            token = urllib.request.urlopen(tok_req, timeout=2).read().decode()
            req = urllib.request.Request(
                base + "/meta-data/spot/termination-time",
                headers={"X-aws-ec2-metadata-token": token},
            )
            resp = urllib.request.urlopen(req, timeout=2)
            if resp.status == 200:
                print("SPOT TERMINATION NOTICE — will checkpoint and exit.", flush=True)
                _INTERRUPT.set()
                return
        except Exception:
            pass  # 404 (no notice) / not-on-EC2 / transient — keep polling
        time.sleep(5)


def _run_hogwild_persistent(args, cfg, trainer, store, start: int, workers: int) -> None:
    """Persistent-worker hogwild: spawn the shared table + workers ONCE; workers traverse continuously;
    the parent snapshots the live table + checkpoints every --checkpoint-interval-sec (no per-round
    respawn/reseed/recompile). On resume the table is seeded ONCE from the latest checkpoint."""
    from collections import defaultdict

    from pokerbot.engine.jax_engine import NUM_ACTIONS as _NA
    from pokerbot.training.sd_cfr_nlhe_hogwild import start_hogwild_session

    interval = float(args.checkpoint_interval_sec)
    is_dcfr = getattr(args, "cfr_variant", "cfr_plus") == "dcfr"
    use_baseline = bool(getattr(args, "vr_baseline", False))
    session = start_hogwild_session(
        args.hogwild_capacity, max(workers, 1),
        occupancy=cfg.occupancy, client_bet_sizing=args.client_bet_sizing,
        cfr_plus=(not is_dcfr),  # DCFR keeps signed regret + periodic discount; else CFR+ floors
        seed=cfg.seed, init_strat=dict(trainer._strat_sum) or None,
        init_regret=trainer._cum_regret or None,
        use_baseline=use_baseline,
    )
    print(f"Hogwild PERSISTENT session: {max(workers, 1)} continuous workers, snapshot+checkpoint "
          f"every {interval:.0f}s (no per-round respawn). variant="
          f"{'dcfr' if is_dcfr else 'cfr_plus'} vr_baseline={use_baseline}.", flush=True)
    try:
        for n in range(start + 1, args.iterations + 1):
            t0 = time.perf_counter()
            while (time.perf_counter() - t0) < interval and not _INTERRUPT.is_set():
                time.sleep(5)
                if session.workers_alive() == 0:
                    print("ERROR: all hogwild workers died — aborting.", flush=True)
                    return
            res = session.snapshot()  # racy live read (accepted hogwild noise)
            trainer._strat_sum = defaultdict(lambda: np.zeros(_NA, dtype=np.float64), res["strat"])
            trainer._cum_regret = dict(res["regret"])
            trainer._completed_iterations = n
            wall = time.perf_counter() - t0
            print(f"iter {n}/{args.iterations} loss=0.000000 wall={wall:.1f}s "
                  f"traversals={session.total_traversals()} infosets={res['n_infosets']} "
                  f"load={res['load_factor']:.3f} workers_alive={session.workers_alive()}", flush=True)
            checkpoint_to_store(trainer, store, n)
            print(f"  checkpoint -> iter {n}", flush=True)
            if is_dcfr:
                # DCFR: discount the live accumulated regret/strategy in place, treating each snapshot
                # tick as iteration n (the lock-free analog of per-iteration DCFR). Racy = hogwild noise.
                session.table.apply_dcfr_discount(n)
            if _INTERRUPT.is_set():
                print("Interrupted — checkpoint saved, exiting cleanly.", flush=True)
                return
    finally:
        session.stop()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--iterations", type=int, default=200)
    p.add_argument("--traversals", type=int, default=90)
    p.add_argument("--deal-sweeps", type=int, default=6)
    p.add_argument("--workers", type=int, default=0, help="parallel workers (>1 => parallel)")
    p.add_argument("--executor", choices=["pool", "subproc", "fork"], default="pool",
                   help="parallel executor; pool=persistent workers+dynamic queue (default, best), "
                        "subproc=per-iter processes, fork=deadlocks under JAX")
    p.add_argument("--trainer", choices=["pool", "hogwild"], default="pool",
                   help="traversal trainer: 'pool' = bulk-sync per-iteration (default, bit-identical "
                        "reference); 'hogwild' = shared-memory lock-free continuous CFR (H5, near-linear "
                        "many-core scaling). Hogwild ignores --executor.")
    p.add_argument("--hogwild-capacity", type=int, default=1 << 22,
                   help="hogwild shared-table slot capacity (RuntimeError-on-full is the under-size canary).")
    p.add_argument("--checkpoint-interval-sec", type=float, default=180.0,
                   help="hogwild PERSISTENT mode: seconds between live-table snapshots/checkpoints "
                        "(workers run continuously; a reclaim loses <= this much). --iterations caps "
                        "the number of checkpoints.")
    p.add_argument("--num-shards", type=int, default=0, help="shards/iter (default: =workers)")
    p.add_argument("--seed", type=int, default=2)
    p.add_argument("--max-depth", type=int, default=64)
    p.add_argument("--bet-buckets", default=None,
                   help="action abstraction: comma-separated indices into "
                        "(1/3,0.50,0.66,0.75,1.0,1.5,2.0) the traversal may bet, e.g. '1,4' = "
                        "half-pot+pot (fold/check/call/all-in always kept). Omit = exact (all 7).")
    p.add_argument("--client-bet-sizing", action="store_true",
                   help="CoinPoker client-matched STREET-AWARE sizing (preflop 2x/2.75x/3.5x, postflop "
                        "1/3,1/2,3/4 pot + all-in); overrides --bet-buckets.")
    p.add_argument("--occupancy", choices=["hu", "6max"], default="hu",
                   help="table layout: 'hu' = heads-up BTN/BB (default), '6max' = full 6-player ring "
                        "(the Pluribus foundation target). Drives init_hand occupancy + live-seat loop.")
    p.add_argument("--cfr-variant", choices=["cfr_plus", "dcfr"], default="cfr_plus",
                   help="regret scheme: cfr_plus (default, floor-at-0) or dcfr (Discounted CFR a1.5/b0/g2, "
                        "signed regret + periodic discount per snapshot tick). hogwild only.")
    p.add_argument("--vr-baseline", action="store_true",
                   help="VR-MCCFR per-action baseline control variate (Schmid 2019) — variance reduction. "
                        "hogwild only; allocates a 3rd table array.")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--checkpoint-every", type=int, default=1)
    p.add_argument("--sft-checkpoint", default="checkpoints/policy_sft_prod6000.pt")
    p.add_argument("--no-sft-init", action="store_true")
    p.add_argument("--s3-bucket", default=None)
    p.add_argument("--s3-prefix", default="pokerbot/sd_cfr_hu")
    p.add_argument("--region", default="eu-central-1")
    p.add_argument("--local-only", default=None, help="local checkpoint dir instead of S3 (dry-run)")
    args = p.parse_args()

    # Depth-retrain startup guard: if the depth KEY is on, the deal randomizer MUST be configured. The
    # sampler fails SAFE (None -> fixed 100bb) on a missing/malformed range, which is right interactively but
    # WRONG for an unattended run — it would silently train the whole table at 100bb with the depth byte on
    # (everything in one bucket), discovered only at the post-run coverage gate. Fail LOUD instead.
    if _flag_truthy(os.environ.get("POKERBOT_DEPTH_BUCKETS")):
        raw = os.environ.get("POKERBOT_TRAIN_DEPTH_BB", "").strip()
        ok = False
        try:
            _lo, _hi = (float(x) for x in raw.split(","))
            ok = _hi > _lo > 0
        except (ValueError, TypeError):
            ok = False
        if not ok:
            raise SystemExit(
                "POKERBOT_DEPTH_BUCKETS is set but POKERBOT_TRAIN_DEPTH_BB is unset/malformed "
                f"({raw!r}); refusing to train a depth-keyed table with fixed-100bb deals. "
                "Set POKERBOT_TRAIN_DEPTH_BB=lo,hi (e.g. 30,200).")
        print(f"Depth retrain: POKERBOT_DEPTH_BUCKETS=1, deals log-U over {raw}bb (shared per deal).",
              flush=True)

    # C1 short-handed coverage guard: a malformed POKERBOT_TRAIN_OCCUPANCY_MIX would silently disable the
    # occupancy randomization (fixed full ring) on an unattended run — fail LOUD instead.
    _occ_raw = os.environ.get("POKERBOT_TRAIN_OCCUPANCY_MIX", "").strip()
    if _occ_raw:
        _ok = False
        try:
            _lo, _hi = (int(x) for x in _occ_raw.split(","))
            _ok = 2 <= _lo <= _hi <= 6
        except (ValueError, TypeError):
            _ok = False
        if not _ok:
            raise SystemExit(
                f"POKERBOT_TRAIN_OCCUPANCY_MIX is malformed ({_occ_raw!r}); expected LO,HI with "
                "2 <= LO <= HI <= 6 (e.g. 2,6). Refusing to silently train fixed-ring on a short-handed run.")
        print(f"Short-handed coverage: per-deal live-seat count over {_occ_raw} (weighted toward full).",
              flush=True)

    if args.local_only:
        store = LocalDirObjectStore(args.local_only)
        print(f"Checkpoint store: local {args.local_only}", flush=True)
    else:
        if not args.s3_bucket:
            raise SystemExit("--s3-bucket required (or use --local-only)")
        store = S3ObjectStore(args.s3_bucket, args.s3_prefix, region=args.region)
        print(f"Checkpoint store: s3://{args.s3_bucket}/{args.s3_prefix}", flush=True)

    bet_buckets = (
        tuple(int(x) for x in str(args.bet_buckets).split(",") if x.strip() != "")
        if args.bet_buckets else None
    )
    occupancy = (
        (True, True, True, True, True, True)
        if args.occupancy == "6max"
        else (False, False, False, True, False, True)  # HU BTN/BB (SDCFRNLHEConfig default)
    )
    cfg = SDCFRNLHEConfig(
        iterations=args.iterations,
        traversals_per_iter=args.traversals,
        deal_sweeps=args.deal_sweeps,
        seed=args.seed,
        lr=args.lr,
        max_traverse_depth=args.max_depth,
        sft_checkpoint=None if args.no_sft_init else args.sft_checkpoint,
        bet_bucket_keep=bet_buckets,
        client_bet_sizing=args.client_bet_sizing,
        occupancy=occupancy,
    )
    print(f"Occupancy: {args.occupancy} -> {occupancy}", flush=True)
    if bet_buckets is not None:
        print(f"Action abstraction ON: bet buckets {bet_buckets} "
              f"(+fold/check/call/all-in) — bounded tables, full-depth feasible.", flush=True)
    trainer = SDCFRNLHETrainer(cfg)

    start = latest_checkpoint_iteration(store)
    if start > 0:
        checkpoint_from_store(trainer, store, start)
        print(f"Resumed from S3 checkpoint iter {start}", flush=True)
    else:
        print("No prior checkpoint — fresh start (SFT-initialized).", flush=True)

    # Spot-interrupt + SIGTERM -> graceful checkpoint+exit.
    threading.Thread(target=_spot_interrupt_watcher, daemon=True).start()
    signal.signal(signal.SIGTERM, lambda *_: _INTERRUPT.set())

    workers = max(0, args.workers)
    num_shards = args.num_shards or max(workers, 1)
    # Executors (all bit-identical to single-threaded; the traversal strategy is the frozen net):
    #   pool    (default) — PERSISTENT spawn workers + dynamic per-unit queue: JAX init ONCE, a heavy
    #                       unit doesn't bottleneck a static shard. Best efficiency.
    #   subproc           — independent per-iteration subprocesses (fresh JAX each). Simpler; ~2-3x.
    #   fork              — fork-COW; DEADLOCKS under JAX on this instance (kept for completeness).
    run_parallel = None  # for fork/subproc (per-iteration callable)
    pool = None          # for the persistent pool
    hogwild = args.trainer == "hogwild"
    if hogwild:
        print(f"Hogwild traversal: {max(workers, 1)} shared-table workers/round, "
              f"capacity={args.hogwild_capacity}, continuous (no per-iter barrier).", flush=True)
    elif workers > 1:
        if args.executor == "fork":
            from pokerbot.training.sd_cfr_nlhe_parallel import run_iteration_parallel as run_parallel
            print(f"Parallel traversal: {workers} FORK workers, {num_shards} shards/iter", flush=True)
        elif args.executor == "subproc":
            from pokerbot.training.sd_cfr_nlhe_subproc import run_iteration_subproc as run_parallel
            print(f"Parallel traversal: {workers} SUBPROC workers, {num_shards} shards/iter", flush=True)
        else:  # pool
            from pokerbot.training.sd_cfr_nlhe_pool import PersistentPool
            pool = PersistentPool(workers)
            print(f"Parallel traversal: {workers} POOL workers (persistent, dynamic queue)", flush=True)

    if hogwild:
        _run_hogwild_persistent(args, cfg, trainer, store, start, workers)
        print("Training complete.", flush=True)
        return

    try:
        for n in range(start + 1, args.iterations + 1):
            t0 = time.perf_counter()
            np.random.seed(cfg.seed + n - 1)
            if pool is not None:
                loss = pool.run_iteration(trainer, float(n))
            elif run_parallel is not None:
                loss = run_parallel(trainer, float(n), num_shards, num_workers=workers)
            else:
                loss = trainer.train_iteration(float(n))
            trainer._completed_iterations = n
            wall = time.perf_counter() - t0
            coll = getattr(trainer, "_iter_collisions", 0)
            print(f"iter {n}/{args.iterations} loss={loss:.6f} wall={wall:.1f}s collisions={coll}", flush=True)
            if coll:
                print(f"  WARNING: {coll} cumulative tabular-key collisions metered (distinct infosets "
                      "sharing a key) — blueprint integrity at risk; investigate before trusting.", flush=True)

            if n % max(1, args.checkpoint_every) == 0 or _INTERRUPT.is_set() or n == args.iterations:
                checkpoint_to_store(trainer, store, n)
                print(f"  checkpoint -> iter {n}", flush=True)
            if _INTERRUPT.is_set():
                print("Interrupted — checkpoint saved, exiting cleanly.", flush=True)
                return
    finally:
        if pool is not None:
            pool.close()

    print("Training complete.", flush=True)


if __name__ == "__main__":
    main()
