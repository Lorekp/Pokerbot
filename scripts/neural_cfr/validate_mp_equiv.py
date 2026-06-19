"""Validate the multiprocess traversal path is STATISTICALLY EQUIVALENT to single-process.

Trains the Leduc gate twice at a matched budget -- once in-process (workers=0), once with W traversal
worker processes -- and prints both exploitability numbers. They will not be bit-identical (distinct RNG
substreams by design) but must land in the same ballpark and well below the uniform ceiling, confirming
the worker sharding did not change the estimator.

Run:  python -m scripts.neural_cfr.validate_mp_equiv --workers 3 --iters 120 --traversals 192
"""

from __future__ import annotations

import argparse
import time


def _train_and_eval(workers: int, iters: int, traversals: int, batch: int, hidden: int,
                    train_steps: int, samples: int, seed: int) -> dict:
    from pokerbot.training.cfr_env import leduc_env_spec
    from pokerbot.training.leduc_gate import LEDUC_REWARD_SCALE, build_leduc_policy_fn
    from pokerbot.training.leduc_tabular_cfr import infoset_exploitability_mbb_from_policy
    from pokerbot.training.neural_cfr import (
        AverageStrategyNet,
        VRDeepPDCFRConfig,
        VRDeepPDCFRPlusTrainer,
    )

    env = leduc_env_spec()
    cfg = VRDeepPDCFRConfig(
        total_iterations=iters, traversals_per_iter=traversals, batch_size=batch,
        hidden_dim=hidden, buffer_capacity=100_000, reward_scale=LEDUC_REWARD_SCALE,
        seed=seed, train_steps_per_iter=train_steps, traversal_workers=workers,
    )
    trainer = VRDeepPDCFRPlusTrainer(cfg, env=env)
    t0 = time.perf_counter()
    for t in range(1, iters + 1):
        trainer.train_iteration(t)
    dt = time.perf_counter() - t0
    avg_module = AverageStrategyNet(hidden, env.num_actions)
    policy_fn = build_leduc_policy_fn(avg_module, trainer.avg_state.params, env)
    expl, *_ = infoset_exploitability_mbb_from_policy(policy_fn, samples=samples)
    trainer.close()
    return {"workers": workers, "exploitability_mbb": float(expl), "train_sec": dt}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--iters", type=int, default=120)
    ap.add_argument("--traversals", type=int, default=192)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--train-steps", type=int, default=1)
    ap.add_argument("--samples", type=int, default=90)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    single = _train_and_eval(0, args.iters, args.traversals, args.batch, args.hidden,
                             args.train_steps, args.samples, args.seed)
    multi = _train_and_eval(args.workers, args.iters, args.traversals, args.batch, args.hidden,
                            args.train_steps, args.samples, args.seed)
    print(f"single-process : exploitability = {single['exploitability_mbb']:8.1f} mbb/g  "
          f"({single['train_sec']:.1f}s)")
    print(f"mp ({args.workers:2d} workers): exploitability = {multi['exploitability_mbb']:8.1f} mbb/g  "
          f"({multi['train_sec']:.1f}s)")
    print(f"delta          : {abs(single['exploitability_mbb']-multi['exploitability_mbb']):8.1f} mbb/g")


if __name__ == "__main__":
    main()
