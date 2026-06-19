"""Leduc budget-vs-exploitability sweep for VR-DeepPDCFR+ (GPU convergence study).

Trains the REAL neural trainer on Leduc at several (iters, train_steps[, avg_steps]) budgets and
reports each run's exact infoset-BR exploitability (mbb/g) against the once-computed tabular floor.
The tabular floor is strategy-independent and disk-cached, so the sweep only pays the neural cost.

This answers: as the budget rises (cheap on GPU), how LOW does Leduc exploitability go -- does it
approach the ~0.30 mbb tabular floor when budget is no longer the constraint? And does boosting the
SERVED average-strategy net's step count specifically (point 4) help beyond the uniform knob?

Run inside repo root:
  PYTHONPATH=. ~/jaxgpu-venv/bin/python scripts/neural_cfr/leduc_budget_sweep.py \
      --points 200:8 200:64 800:64 1500:64 --avg-mult 1
  PYTHONPATH=. ~/jaxgpu-venv/bin/python scripts/neural_cfr/leduc_budget_sweep.py \
      --points 800:8 --avg-steps 8 32 128   # avg-net-steps study at fixed base budget
"""

from __future__ import annotations

import argparse
import json
import time

from pokerbot.training.cfr_env import leduc_env_spec
from pokerbot.training.leduc_gate import LEDUC_REWARD_SCALE, build_leduc_policy_fn
from pokerbot.training.leduc_tabular_cfr import (
    equilibrium_reference_exploitability_mbb,
    infoset_exploitability_mbb_from_policy,
)
from pokerbot.training.neural_cfr import (
    AverageStrategyNet,
    VRDeepPDCFRConfig,
    VRDeepPDCFRPlusTrainer,
)


def train_and_measure(iters, train_steps, traversals, hidden, batch, samples, seed,
                      avg_steps=None, hv_steps=None):
    env = leduc_env_spec()
    cfg = VRDeepPDCFRConfig(
        total_iterations=iters, traversals_per_iter=traversals, batch_size=batch,
        hidden_dim=hidden, reward_scale=LEDUC_REWARD_SCALE, seed=seed,
        train_steps_per_iter=train_steps,
        avg_strategy_steps_per_iter=avg_steps,
        history_value_steps_per_iter=hv_steps,
    )
    trainer = VRDeepPDCFRPlusTrainer(cfg, env=env)
    t0 = time.perf_counter()
    for t in range(1, iters + 1):
        trainer.train_iteration(t)
    train_s = time.perf_counter() - t0
    avg_module = AverageStrategyNet(hidden, env.num_actions)
    policy_fn = build_leduc_policy_fn(avg_module, trainer.avg_state.params, env)
    expl, *_ = infoset_exploitability_mbb_from_policy(policy_fn, samples=samples)
    trainer.close()
    return float(expl), train_s


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--points", nargs="+", default=["200:8", "200:64"],
                   help="iters:train_steps points, e.g. 200:8 800:64 1500:64")
    p.add_argument("--avg-steps", type=int, nargs="*", default=None,
                   help="if set, sweep avg-strategy-net steps at the FIRST point (per-net study)")
    p.add_argument("--traversals", type=int, default=256)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--samples", type=int, default=90)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--reference-iters", type=int, default=2000)
    args = p.parse_args()

    import jax
    print(f"JAX backend={jax.default_backend()} devices={jax.devices()}", flush=True)
    print("computing tabular floor (cached after first run)...", flush=True)
    floor = equilibrium_reference_exploitability_mbb(iterations=args.reference_iters)
    print(f"tabular floor = {floor:.3f} mbb/g\n", flush=True)

    rows = []
    if args.avg_steps:
        # Per-net study: fix the base budget at the first point, vary avg-net steps only.
        iters, base_steps = (int(x) for x in args.points[0].split(":"))
        print(f"=== avg-strategy-net step study @ {iters} iter, base train_steps={base_steps} ===",
              flush=True)
        for a in args.avg_steps:
            expl, secs = train_and_measure(
                iters, base_steps, args.traversals, args.hidden, args.batch,
                args.samples, args.seed, avg_steps=a)
            ratio = expl / floor if floor > 0 else float("inf")
            rows.append({"iters": iters, "train_steps": base_steps, "avg_steps": a,
                         "exploit_mbb": round(expl, 1), "ratio": round(ratio, 1),
                         "train_s": round(secs, 1)})
            print(f"avg_steps={a:4d}: {expl:8.1f} mbb/g  ({ratio:6.1f}x floor)  "
                  f"[{secs:.1f}s train]", flush=True)
    else:
        for pt in args.points:
            iters, steps = (int(x) for x in pt.split(":"))
            expl, secs = train_and_measure(
                iters, steps, args.traversals, args.hidden, args.batch, args.samples, args.seed)
            ratio = expl / floor if floor > 0 else float("inf")
            rows.append({"iters": iters, "train_steps": steps, "avg_steps": steps,
                         "exploit_mbb": round(expl, 1), "ratio": round(ratio, 1),
                         "train_s": round(secs, 1)})
            print(f"iters={iters:5d} steps={steps:4d}: {expl:8.1f} mbb/g  "
                  f"({ratio:6.1f}x floor)  [{secs:.1f}s train]", flush=True)

    print("\n" + json.dumps({"floor_mbb": round(floor, 3), "rows": rows}))


if __name__ == "__main__":
    main()
