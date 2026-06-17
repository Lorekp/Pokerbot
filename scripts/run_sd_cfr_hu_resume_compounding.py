#!/usr/bin/env python3
"""
Step-5 gate: resume must NOT compound drift.

An interrupted convergence run is checkpointed-and-reloaded at *every* barrier (worst case:
a fresh TrainState is rebuilt each iteration). We run that interrupted arm in lockstep with an
uninterrupted arm to ~10 iters and prove the two stay equivalent:

  - CFR tables (_cum_regret, _strat_sum) bit-identical EVERY iteration (regrets are the policy
    authority; any nonzero diff here is a real resume bug),
  - per-iteration loss exact,
  - any params max-diff stays FLAT (does not trend upward) across iterations.

A single-iteration check cannot detect compounding by construction; this spans iterations and
prints the per-iter (max_regret_diff, max_strat_diff, max_param_diff, loss_delta) series as the
artifact. Flat -> resume is safe for an interruptible run. Growing -> stop, real resume bug.

This is a LOCAL smoke-scale harness (cheap, deterministic). It validates the resume *mechanism*,
not strategy quality.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from flax.core import unfreeze

from pokerbot.training.sd_cfr_nlhe import (
    SDCFRNLHEConfig,
    SDCFRNLHETrainer,
    load_nlhe_checkpoint,
    save_nlhe_checkpoint,
)
from pokerbot.training.sd_cfr_nlhe_distributed import compare_trainer_states

ROOT = Path(__file__).resolve().parents[1]

# Bit-identical regrets are required. Params may carry tiny XLA re-init noise after a fresh
# TrainState; the gate is that this noise stays FLAT, not that it is zero.
LOSS_TOL = 1e-5
PARAM_FLAT_CEIL = 2e-4  # absolute ceiling for a "flat" param diff (handoff noted ~1.7e-4)
PARAM_GROWTH_FACTOR = 4.0  # last-iter diff must not exceed factor * first-iter diff (+ ceil)


def _max_param_diff(a: SDCFRNLHETrainer, b: SDCFRNLHETrainer) -> float:
    pa, pb = unfreeze(a.state.params), unfreeze(b.state.params)
    worst = 0.0

    def _walk(x, y) -> None:
        nonlocal worst
        if isinstance(x, dict):
            for k in x:
                _walk(x[k], y[k])
        else:
            worst = max(worst, float(np.max(np.abs(np.asarray(x) - np.asarray(y)))))

    _walk(pa, pb)
    return worst


def _cfg(seed: int, traversals: int, depth: int, batch: int) -> SDCFRNLHEConfig:
    return SDCFRNLHEConfig(
        iterations=1,
        traversals_per_iter=traversals,
        deal_sweeps=1,
        train_batch_size=batch,
        sft_checkpoint=None,
        seed=seed,
        max_traverse_depth=depth,
    )


def main() -> None:
    p = argparse.ArgumentParser(description="Resume-compounding gate (Step 5)")
    p.add_argument("--iters", type=int, default=10)
    p.add_argument("--traversals", type=int, default=4)
    p.add_argument("--depth", type=int, default=16)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--work-dir", default="logs/sd_cfr_hu_resume_compounding_work")
    p.add_argument("--output", default="logs/sd_cfr_hu_resume_compounding.json")
    args = p.parse_args()

    cfg = _cfg(args.seed, args.traversals, args.depth, args.batch)
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    uninterrupted = SDCFRNLHETrainer(cfg)
    interrupted = SDCFRNLHETrainer(cfg)

    rows: list[dict] = []
    print(
        f"Resume-compounding gate: {args.iters} iters, traversals={args.traversals} "
        f"depth={args.depth} batch={args.batch} seed={args.seed}",
        flush=True,
    )
    print(f"  {'iter':>4} {'loss_delta':>12} {'regret_diff':>12} {'strat_diff':>12} {'param_diff':>12}", flush=True)

    for n in range(1, args.iters + 1):
        w = float(n)

        # Uninterrupted arm: one iteration with the per-iter seed.
        np.random.seed(cfg.seed + n - 1)
        loss_ref = uninterrupted.train_iteration(w)

        # Interrupted arm: same iteration, then checkpoint + reload into a FRESH trainer
        # (simulating a worker restart at this barrier).
        np.random.seed(cfg.seed + n - 1)
        loss_res = interrupted.train_iteration(w)
        interrupted._completed_iterations = n
        ckpt = work_dir / f"resume_iter_{n:04d}_seed{cfg.seed}.pkl"
        save_nlhe_checkpoint(interrupted, ckpt)
        reloaded = SDCFRNLHETrainer(cfg)
        restored_iter = load_nlhe_checkpoint(reloaded, ckpt)
        assert restored_iter == n, (restored_iter, n)
        assert reloaded._completed_iterations == n  # loader sets it (B.1)
        interrupted = reloaded

        cmp = compare_trainer_states(uninterrupted, interrupted)
        param_diff = _max_param_diff(uninterrupted, interrupted)
        loss_delta = abs(loss_ref - loss_res)
        row = {
            "iter": n,
            "loss_ref": loss_ref,
            "loss_res": loss_res,
            "loss_delta": loss_delta,
            "max_cum_regret_diff": cmp["max_cum_regret_diff"],
            "max_strat_sum_diff": cmp["max_strat_sum_diff"],
            "max_param_diff": param_diff,
            "regret_bit_identical": cmp["cum_regret_bit_identical"],
            "strat_bit_identical": cmp["strat_sum_bit_identical"],
            "reservoir_match": cmp["reservoir_len_match"] and cmp["reservoir_insert_match"],
        }
        rows.append(row)
        print(
            f"  {n:>4} {loss_delta:>12.2e} {cmp['max_cum_regret_diff']:>12.2e} "
            f"{cmp['max_strat_sum_diff']:>12.2e} {param_diff:>12.2e}",
            flush=True,
        )

    # ---- Gate evaluation (do not relax) ----
    regret_ok = all(r["regret_bit_identical"] for r in rows)
    strat_ok = all(r["strat_bit_identical"] for r in rows)
    reservoir_ok = all(r["reservoir_match"] for r in rows)
    loss_ok = all(r["loss_delta"] <= LOSS_TOL for r in rows)

    param_series = [r["max_param_diff"] for r in rows]
    first, last, worst = param_series[0], param_series[-1], max(param_series)
    growth_ceiling = max(PARAM_FLAT_CEIL, PARAM_GROWTH_FACTOR * first)
    param_flat = (worst <= PARAM_FLAT_CEIL) or (last <= growth_ceiling)

    passed = regret_ok and strat_ok and reservoir_ok and loss_ok and param_flat
    result = {
        "gate": "resume_does_not_compound",
        "passed": passed,
        "iters": args.iters,
        "config": {
            "traversals": args.traversals, "depth": args.depth,
            "batch": args.batch, "seed": args.seed,
        },
        "checks": {
            "regret_bit_identical_all_iters": regret_ok,
            "strat_bit_identical_all_iters": strat_ok,
            "reservoir_match_all_iters": reservoir_ok,
            "loss_exact_all_iters": loss_ok,
            "param_diff_flat": param_flat,
        },
        "param_diff_series": param_series,
        "param_diff_first": first,
        "param_diff_last": last,
        "param_diff_worst": worst,
        "rows": rows,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["checks"], indent=2), flush=True)
    print(
        f"param diff: first={first:.2e} last={last:.2e} worst={worst:.2e} "
        f"flat={param_flat} -> resume {'SAFE' if passed else 'UNSAFE'}",
        flush=True,
    )
    print(f"Wrote {out}", flush=True)
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
