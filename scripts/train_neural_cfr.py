#!/usr/bin/env python3
"""Train VR-DeepPDCFR+ on the JAX-vectorized 6-max NLHE environment.

SATURATION: pass --vcpus N (and optionally --sat-topology / --sat-intra-op) to size the XLA Eigen
thread pool + CPU device count BEFORE jax is imported, so the box's cores are fully used with NO
oversubscription. The measured-best topology on JAX-CPU is a SINGLE device with a multithreaded Eigen
pool (intra_op ~= vcpus); pmap-over-host-CPU-devices ("fine") was benchmarked 6x SLOWER for the large
NLHE traversal because it single-threads each device and kills Eigen matmul vectorization. See
scripts/neural_cfr/saturate_env.py and the scaling curve in scripts/neural_cfr/scaling_curve.py.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# --- saturation env MUST be configured before ANY jax import -----------------------------------------
# Pre-parse only the saturation flags (argparse proper runs later for the rest); set XLA_FLAGS + thread
# caps in os.environ, THEN import the jax-dependent training code.
_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument("--vcpus", type=int, default=None)
_pre.add_argument("--sat-topology", default="single", choices=["single", "coarse", "fine"])
_pre.add_argument("--sat-intra-op", type=int, default=None)
_pre.add_argument("--sat-devices", type=int, default=None)
_known, _ = _pre.parse_known_args()
if _known.vcpus is not None or _known.sat_intra_op is not None or _known.sat_devices is not None:
    try:
        from scripts.neural_cfr.saturate_env import configure
    except ModuleNotFoundError:
        # running as a bare file (not -m): add repo root to sys.path so scripts.* resolves.
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from scripts.neural_cfr.saturate_env import configure
    _sat = configure(vcpus=_known.vcpus, topology=_known.sat_topology,
                     n_devices=_known.sat_devices, intra_op=_known.sat_intra_op)
    print(f"[saturate] {_sat.describe()}", file=sys.stderr, flush=True)

from dataclasses import replace  # noqa: E402

from pokerbot.encoding.weva_encoder import assert_production_weva_index  # noqa: E402
from pokerbot.training.neural_cfr import (  # noqa: E402
    VRDeepPDCFRConfig,
    VRDeepPDCFRPlusTrainer,
    load_trainer_checkpoint,
    save_trainer_checkpoint,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="VR-DeepPDCFR+ training")
    # saturation flags (already consumed pre-import; re-declared so --help shows them and argparse
    # does not error on them).
    parser.add_argument("--vcpus", type=int, default=None,
                        help="size XLA threads/devices for an N-vCPU box (set before jax import)")
    parser.add_argument("--sat-topology", default="single", choices=["single", "coarse", "fine"],
                        help="single=1 device w/ N-thread Eigen (measured best); coarse/fine experimental")
    parser.add_argument("--sat-intra-op", type=int, default=None, help="override Eigen intra-op threads")
    parser.add_argument("--sat-devices", type=int, default=None, help="override XLA host device count")
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--traversals", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--buffer-capacity", type=int, default=100_000)
    parser.add_argument("--alpha", type=float, default=2.3)
    parser.add_argument("--gamma", type=float, default=2.0)
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--game", default="nlhe", choices=["nlhe", "nlhe-hu", "leduc"],
        help="which game to train. nlhe=6-max (NOTE: the VR persp_sign baseline is exact only for "
             "2-player zero-sum, so 6-max is biased); nlhe-hu=heads-up NLHE (CORRECT estimator, the "
             "intended GPU target); leduc=toy gate.",
    )
    parser.add_argument("--hidden-dim", type=int, default=None, help="MLP width (default config 256)")
    parser.add_argument("--train-steps", type=int, default=None,
                        help="SGD steps/net/iter (uniform). GPU makes more steps cheap; 1 underfits.")
    parser.add_argument("--avg-steps", type=int, default=None,
                        help="override SGD steps for the SERVED average-strategy net (ref ~5000)")
    parser.add_argument("--hist-steps", type=int, default=None,
                        help="override SGD steps for the HistoryValueNet baseline")
    parser.add_argument("--checkpoint-every", type=int, default=0,
                        help="atomically save a resumable checkpoint every N iters (overnight safety)")
    parser.add_argument(
        "--traversal-workers", type=int, default=0,
        help="independent worker PROCESSES that generate traversals data-parallel (the saturation "
             "lever: a single JAX-CPU process is ~1-2 cores on the scan-heavy traversal regardless of "
             "thread count). Set ~= vcpus-1. 0 = in-process single path.",
    )
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/history_value_net.pkl",
        help="Output path for HistoryValueNet + blueprint params",
    )
    parser.add_argument(
        "--resume",
        default=None,
        help="Resume from full checkpoint (history_value_net_full.pkl or base path)",
    )
    parser.add_argument(
        "--allow-dev-weva",
        action="store_true",
        help="Skip production WEVA gate (dev smoke only)",
    )
    args = parser.parse_args()

    # The WEVA index gates only the 6-max tabular-feature path. Neural CFR encodes dense features
    # (jax_engine.encode_info_state -- card ranks/one-hots/stacks, no WEVA), so nlhe-hu and leduc
    # never need it.
    if args.game == "nlhe" and not args.allow_dev_weva:
        assert_production_weva_index(Path("data/weva"), context="Phase 2 neural CFR / HistoryValueNet training")

    # Build the env spec for the chosen game (None => trainer defaults to 6-max nlhe).
    env = None
    if args.game == "nlhe-hu":
        from pokerbot.training.cfr_env import nlhe_hu_env_spec
        env = nlhe_hu_env_spec()
    elif args.game == "leduc":
        from pokerbot.training.cfr_env import leduc_env_spec
        env = leduc_env_spec()

    # The multiprocess traversal pool (mp_traversal) hardcodes env_name="nlhe" (6-max reset), so it
    # cannot generate heads-up occupancy. On a single GPU the saturation lever is irrelevant anyway
    # (use the in-process path); fail loudly rather than silently train 6-max under a HU label.
    if args.game == "nlhe-hu" and args.traversal_workers:
        parser.error("--game nlhe-hu does not support --traversal-workers>0 (mp pool is 6-max only); "
                     "use the in-process GPU path (--traversal-workers 0).")

    # Optional config overrides (None => keep the dataclass/loaded default).
    overrides = {k: v for k, v in {
        "hidden_dim": args.hidden_dim,
        "train_steps_per_iter": args.train_steps,
        "avg_strategy_steps_per_iter": args.avg_steps,
        "history_value_steps_per_iter": args.hist_steps,
    }.items() if v is not None}

    if args.resume:
        trainer = load_trainer_checkpoint(args.resume)
        if env is not None:
            trainer.env = env  # restore the chosen game (load defaults to 6-max nlhe; HU dims match)
        start = trainer._start_iteration
        cfg = replace(trainer.config, total_iterations=start + args.iterations,
                      traversal_workers=args.traversal_workers, **overrides)
        trainer.config = cfg
        # re-init the pool on resume if requested (the loaded trainer has workers=0 by default).
        if args.traversal_workers and getattr(trainer, "_trav_pool", None) is None:
            from pokerbot.training.mp_traversal import TraversalPool
            trainer._trav_pool = TraversalPool(
                num_workers=args.traversal_workers, env_name="nlhe",
                hidden_dim=cfg.hidden_dim, num_actions=trainer.env.num_actions,
                reward_scale=cfg.reward_scale, alpha=cfg.alpha, epsilon=cfg.epsilon)
        print(f"Resuming from iteration {start}, training to {cfg.total_iterations} (game={args.game})")
    else:
        cfg = VRDeepPDCFRConfig(
            total_iterations=args.iterations,
            traversals_per_iter=args.traversals,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            buffer_capacity=args.buffer_capacity,
            alpha=args.alpha,
            gamma=args.gamma,
            epsilon=args.epsilon,
            seed=args.seed,
            traversal_workers=args.traversal_workers,
            **overrides,
        )
        trainer = VRDeepPDCFRPlusTrainer(cfg, env=env)
        print(f"Training game={args.game}  env.num_players={trainer.env.num_players}  "
              f"hidden={cfg.hidden_dim}  steps(train/avg/hist)="
              f"{cfg.train_steps_per_iter}/{cfg.avg_strategy_steps_per_iter}/"
              f"{cfg.history_value_steps_per_iter}")

    try:
        trainer.train(log_every=max(1, args.iterations // 50),
                      checkpoint_every=args.checkpoint_every, checkpoint_path=args.checkpoint)
    finally:
        trainer.close()  # release the worker pool
    full_out = save_trainer_checkpoint(trainer, args.checkpoint)
    print(f"Completed VR-DeepPDCFR+ through iteration {trainer._iteration}.")
    print(f"Saved inference checkpoint: {args.checkpoint}")
    print(f"Saved full trainer checkpoint: {full_out}")


if __name__ == "__main__":
    main()
