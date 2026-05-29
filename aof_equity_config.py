"""Central MC/exact budgets and adaptive multiway equity API."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np

EquityMethod = Literal["exact", "mc"]


@dataclass(frozen=True)
class EquityRuntimeConfig:
    advisor_samples: int = 20_000
    facing2_samples: int = 20_000
    facing3_samples: int = 20_000
    behind_samples: int = 20_000
    open_multiway_samples: int = 20_000
    rl_teacher_samples: int = 2_000
    rl_payoff_samples: int = 4_000
    exact_pair_cap: int = 256
    exact_product_cap_3way: int = 256
    exact_product_cap_4way: int = 128
    max_combos_per_range_advisor: int = 96
    seed: int | None = None


@dataclass(frozen=True)
class EquityResult:
    equity: float
    method: EquityMethod
    samples_used: int
    valid_combo_products: int | None
    seed: int | None
    stdev_estimate: float | None
    warning: str | None


_DEFAULT_CONFIG = EquityRuntimeConfig()


def default_equity_config() -> EquityRuntimeConfig:
    return _DEFAULT_CONFIG


def equity_vs_ranges_adaptive(
    hero_key: str,
    villain_ranges: Sequence[str],
    samples: int,
    config: EquityRuntimeConfig | None = None,
    *,
    seed: int,
    rng: np.random.Generator | None = None,
) -> EquityResult:
    """
    Hero equity vs one or more villain ranges (independent range sampling).

    Dispatches to HU / 3-way / 4-way kernels in ``aof_equity``.
    """
    from aof_equity import (
        equity_vs_four_ranges_adaptive,
        equity_vs_range,
        equity_vs_three_ranges_adaptive,
        equity_vs_two_ranges_adaptive,
    )

    cfg = config or _DEFAULT_CONFIG
    n = len(villain_ranges)
    max_c = cfg.max_combos_per_range_advisor

    if n == 0:
        return EquityResult(0.5, "mc", 0, None, seed, None, "no villain ranges")
    if n == 1:
        eq = equity_vs_range(
            hero_key,
            villain_ranges[0],
            samples,
            seed,
            max_c,
        )
        return EquityResult(eq, "mc", samples, None, seed, None, None)
    if n == 2:
        eq, meta = equity_vs_two_ranges_adaptive(
            hero_key,
            villain_ranges[0],
            villain_ranges[1],
            samples,
            seed,
            max_combos=max_c,
            return_meta=True,
        )
        return EquityResult(
            eq,
            meta["method"],
            meta["samples_used"],
            meta.get("valid_products"),
            seed,
            None,
            meta.get("warning"),
        )
    if n == 3:
        eq, meta = equity_vs_three_ranges_adaptive(
            hero_key,
            list(villain_ranges),
            samples,
            seed,
            max_combos=max_c,
            config=cfg,
            return_meta=True,
        )
        return EquityResult(
            eq,
            meta["method"],
            meta["samples_used"],
            meta.get("valid_products"),
            seed,
            None,
            meta.get("warning"),
        )
    if n >= 4:
        eq, meta = equity_vs_three_ranges_adaptive(
            hero_key,
            list(villain_ranges[:3]),
            samples,
            seed,
            max_combos=max_c,
            config=cfg,
            return_meta=True,
        )
        return EquityResult(
            eq,
            meta["method"],
            meta["samples_used"],
            meta.get("valid_products"),
            seed,
            None,
            "4+ villain ranges: capped to 3 (4-way pot max)",
        )
    raise RuntimeError("unreachable")


def aof_strict_ranges() -> bool:
    return os.environ.get("AOF_STRICT_RANGES", "").strip() in ("1", "true", "yes")
