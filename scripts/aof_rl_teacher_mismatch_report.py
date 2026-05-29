#!/usr/bin/env python3
"""Report RL spot kinds and teacher paths over synthetic rollouts."""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from aof_ranges import active_seats
from aof_rl.env import ACTION_ALL_IN, ACTION_FOLD, reset_hand, step_hand
from aof_rl.spot_diagnostics import (
    SpotDiagnostics,
    classify_decision_spot,
    record_decision,
)
from aof_rl.teacher import NashTeacherCache

NUM_HANDS_DEFAULT = 200
SEED = 42


def _rollout(
    rng: random.Random,
    *,
    table_size: int,
    teacher_version: int,
    samples: int,
    payoff_samples: int,
) -> tuple[SpotDiagnostics, Counter[str]]:
    diag = SpotDiagnostics()
    terminals: Counter[str] = Counter()
    cache = NashTeacherCache(teacher_version=teacher_version, use_fast=True)
    state = reset_hand(rng, table_size=table_size, stack_bb=8.0)
    guard = 0
    while not state.terminal:
        guard += 1
        if guard > 32:
            break
        seat = state.action_seat
        if seat is None:
            break
        kind = classify_decision_spot(state, seat)
        result = cache.get_with_path(
            state, seat, samples=samples, seed=SEED + state.hand_id
        )
        record_decision(
            diag,
            state,
            seat,
            teacher_path=result.path,
            teacher_version=teacher_version,
        )
        action = result.teacher.action_index
        state = step_hand(
            state, action, payoff_samples=payoff_samples, payoff_seed=SEED + guard
        )
    if state.terminal:
        alive = [s for s in state.seats if s not in state.folded]
        if len(alive) == 3:
            terminals["terminal_3way"] += 1
        elif len(alive) == 4:
            terminals["terminal_4way"] += 1
        else:
            terminals["terminal_other"] += 1
    return diag, terminals


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hands", type=int, default=NUM_HANDS_DEFAULT)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--teacher-version", type=int, default=1)
    parser.add_argument("--obs-version", type=int, default=1)
    parser.add_argument("--teacher-samples", type=int, default=None)
    parser.add_argument("--payoff-samples", type=int, default=None)
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("artifacts/aof_rl_teacher_mismatch_report.json"),
    )
    args = parser.parse_args()
    rng = random.Random(args.seed)
    if args.teacher_samples is not None:
        teacher_samples = args.teacher_samples
    elif args.hands <= 40:
        teacher_samples = 120
    else:
        teacher_samples = 800
    if args.payoff_samples is not None:
        payoff_samples = args.payoff_samples
    elif args.hands <= 40:
        payoff_samples = 150
    else:
        payoff_samples = 400

    spot_counts: Counter[str] = Counter()
    path_counts: Counter[str] = Counter()
    terminal_counts: Counter[str] = Counter()

    for _ in range(args.hands):
        table_size = rng.choice((2, 3, 4))
        diag, terms = _rollout(
            rng,
            table_size=table_size,
            teacher_version=args.teacher_version,
            samples=teacher_samples,
            payoff_samples=payoff_samples,
        )
        spot_counts.update(diag.counts)
        path_counts.update(diag.teacher_paths)
        terminal_counts.update(terms)

    payload = {
        "schema_version": 1,
        "obs_version": args.obs_version,
        "teacher_version": args.teacher_version,
        "num_hands": args.hands,
        "spot_kind_counts": dict(spot_counts),
        "teacher_path_counts": dict(path_counts),
        "terminal_counts": dict(terminal_counts),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
