"""Timing helpers for HU showdown benchmarks."""

from __future__ import annotations

import json
import statistics
import time
from pathlib import Path
from typing import Any, Callable

from aof_equity import warmup
from aof_equity_grids import _card_str_to_id, _equity_two_combos_exact

N_CANONICAL_KEYS = 97_513
RUNOUTS_PER_KEY = 1_712_304
EVAL7_PER_KEY = 2 * RUNOUTS_PER_KEY
DENOM2 = 3_424_608


def warmup_equity() -> None:
    warmup()
    h0 = _card_str_to_id("Ac")
    h1 = _card_str_to_id("Ad")
    v0 = _card_str_to_id("Kh")
    v1 = _card_str_to_id("Kd")
    _equity_two_combos_exact(h0, h1, v0, v1)


def bench_callable(
    fn: Callable[[], None],
    *,
    n_calls: int,
    n_warmup: int = 3,
    n_repeat: int = 2,
    label: str = "bench",
) -> dict[str, Any]:
    for _ in range(n_warmup):
        fn()

    repeats: list[dict[str, float]] = []
    for _ in range(n_repeat):
        t0 = time.perf_counter()
        for _ in range(n_calls):
            fn()
        wall_s = time.perf_counter() - t0
        repeats.append(
            {
                "wall_s": wall_s,
                "s_per_call": wall_s / n_calls,
                "calls_per_s": n_calls / wall_s,
            }
        )

    s_per_call = statistics.mean(r["s_per_call"] for r in repeats)
    calls_per_s = statistics.mean(r["calls_per_s"] for r in repeats)
    wall_mean = statistics.mean(r["wall_s"] for r in repeats)
    s_std = (
        statistics.stdev(r["s_per_call"] for r in repeats) if len(repeats) > 1 else 0.0
    )

    return {
        "label": label,
        "n_calls": n_calls,
        "n_repeat": n_repeat,
        "wall_s": wall_mean,
        "s_per_call": s_per_call,
        "ns_per_call": s_per_call * 1e9,
        "calls_per_s": calls_per_s,
        "s_per_call_std": s_std,
        "repeats": repeats,
    }


def projected_full_build_h(s_per_key: float) -> float:
    return s_per_key * N_CANONICAL_KEYS / 3600.0


def summarize_repeats_discard_first(
    s_per_key_values: list[float],
    *,
    discard_first: int = 1,
) -> dict[str, float]:
    """Drop the first repeat(s) before computing mean/std (cache warm-up)."""
    if len(s_per_key_values) <= discard_first:
        used = s_per_key_values
    else:
        used = s_per_key_values[discard_first:]
    s_mean = statistics.mean(used)
    s_std = statistics.stdev(used) if len(used) > 1 else 0.0
    rel_std_pct = 100.0 * s_std / s_mean if s_mean > 0 else 0.0
    return {
        "s_per_key": s_mean,
        "s_per_key_std": s_std,
        "s_per_key_rel_std_pct": rel_std_pct,
        "n_repeats_used": len(used),
        "n_repeats_discarded": min(discard_first, len(s_per_key_values)),
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
