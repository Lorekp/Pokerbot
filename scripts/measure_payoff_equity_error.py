#!/usr/bin/env python3
"""Measure combo-exact vs class-mean vs MC payoff equity error on HU showdowns."""

from __future__ import annotations

import argparse
import json
import random
import statistics
import time
from pathlib import Path

import numpy as np

from aof_equity import equity_hu_exact, warmup
from aof_equity_grids import (
    _equity_two_combos_exact,
    load_hu_equity_grid,
)
from aof_equity_showdown import cards_to_ids
from aof_rl.encode import _hand_label
from aof_rl.env import ACTION_ALL_IN, ACTION_FOLD, step_hand
from aof_rl.payoff import configure_payoff
from aof_rl.reward_noise import reward_noise_floor_bb, reward_noise_floor_definition
from aof_rl.spot_forcing import reset_hand_for_curriculum
from aof_rl.curriculum import CurriculumConfig, CurriculumScheduler, CurriculumStage
from pokerbot.solver.hu_preflop_dcfr import _build_169_buckets


def _hu_stage() -> CurriculumStage:
    return CurriculumStage(
        "hu_only", (2,), "open_or_facing", flat_rake_bb=0.0, updates=999
    )


def _collect_hu_showdowns(n_target: int, seed: int, stack_bb: float) -> list[dict]:
    rng = random.Random(seed)
    sched = CurriculumScheduler(CurriculumConfig(stages=[_hu_stage()]))
    stage = sched.stage_for_update(0)
    out: list[dict] = []
    hands = 0
    while len(out) < n_target and hands < n_target * 50:
        hands += 1
        state, _ = reset_hand_for_curriculum(rng, stage, stack_bb=stack_bb)
        guard = 0
        while not state.terminal:
            guard += 1
            seat = state.action_seat
            if seat is None:
                break
            action = ACTION_ALL_IN if rng.random() < 0.85 else ACTION_FOLD
            state = step_hand(state, action, payoff_samples=4000, payoff_seed=0)
        alive = [s for s in state.seats if s not in state.folded]
        if len(alive) == 2:
            a, b = alive[0], alive[1]
            out.append(
                {
                    "hero": state.holes[a],
                    "villain": state.holes[b],
                    "pot_bb": float(sum(state.invested.values())),
                }
            )
    return out


def _delta_ev_bb(eq_err: float, pot_bb: float) -> float:
    return abs(eq_err) * pot_bb


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("-n", type=int, default=4000, help="HU showdown samples")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--noise-k", type=float, default=0.25)
    p.add_argument("--batch-hands", type=int, default=128)
    p.add_argument("--payoff-samples", type=int, default=1200)
    p.add_argument(
        "--grid",
        type=Path,
        default=Path("data/aof/hu_equity_169.npz"),
    )
    p.add_argument(
        "-o",
        type=Path,
        default=Path("artifacts/audit/payoff_equity_error.json"),
    )
    p.add_argument("--stack-bb", type=float, default=8.0)
    args = p.parse_args()

    warmup()
    configure_payoff(payoff_hu_mode="exact_enum")
    t0 = time.perf_counter()
    show = _collect_hu_showdowns(args.n, args.seed, args.stack_bb)
    if not show:
        raise SystemExit("No HU showdowns collected")

    labels, _ = _build_169_buckets()
    grid = None
    if args.grid.is_file():
        grid = load_hu_equity_grid(args.grid)
    else:
        print(f"Warning: class grid missing at {args.grid}", flush=True)

    d_class: list[float] = []
    d_mc: list[float] = []
    returns_proxy: list[float] = []

    for row in show:
        hero = row["hero"]
        vill = row["villain"]
        pot = row["pot_bb"]
        h0, h1, v0, v1 = cards_to_ids(hero, vill)
        exact = float(_equity_two_combos_exact(h0, h1, v0, v1))
        mc = float(equity_hu_exact(hero, vill, args.payoff_samples, args.seed))
        if grid is not None:
            i = labels.index(_hand_label(hero))
            j = labels.index(_hand_label(vill))
            cls = float(grid[i, j])
            d_class.append(_delta_ev_bb(cls - exact, pot))
        d_mc.append(_delta_ev_bb(mc - exact, pot))
        share_a = exact * pot
        returns_proxy.append(share_a - pot * 0.5)

    floor_bb = reward_noise_floor_bb(
        returns_proxy, batch_hands=args.batch_hands, noise_k=args.noise_k
    )

    def stats(xs: list[float]) -> dict[str, float]:
        if not xs:
            return {}
        xs_s = sorted(xs)
        n = len(xs_s)

        def pct(p: float) -> float:
            idx = min(n - 1, int(p * n))
            return xs_s[idx]

        return {
            "mean": float(statistics.mean(xs)),
            "p95": pct(0.95),
            "p99": pct(0.99),
            "max": xs_s[-1],
        }

    payload = {
        "n_showdowns": len(show),
        "wall_seconds": time.perf_counter() - t0,
        "reward_noise_floor": reward_noise_floor_definition(
            batch_hands=args.batch_hands, noise_k=args.noise_k
        ),
        "reward_noise_floor_bb": floor_bb,
        "class_mean_vs_exact_delta_ev_bb": stats(d_class),
        "mc_vs_exact_delta_ev_bb": stats(d_mc),
        "class_mean_below_floor": (
            stats(d_class).get("p95", float("inf")) < floor_bb if d_class else None
        ),
        "mc_p95_below_floor": (
            stats(d_mc).get("p95", float("inf")) < floor_bb if d_mc else None
        ),
    }
    args.o.parent.mkdir(parents=True, exist_ok=True)
    args.o.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"Wrote {args.o}")


if __name__ == "__main__":
    main()
