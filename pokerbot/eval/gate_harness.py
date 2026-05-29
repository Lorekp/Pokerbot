"""Shared gate harness: fixed opponent pool, relative exploitability + bb100 metrics."""

from __future__ import annotations

import random
from typing import Callable

from pokerbot.eval.bots import BOT_REGISTRY
from pokerbot.eval.exploitability import (
    SIGNED_EXPLOITER_EV_METRIC_NOTE,
    estimate_exploitability,
    require_exploitability_harness_sanity,
)
from pokerbot.eval.head_to_head import run_head_to_head

# Canonical static pool — NOT SCAPT, NOT league sampling
GATE_OPPONENT_POOL: tuple[str, ...] = ("nit", "lag", "calling_station", "tag", "human_like")

# Hard precondition seed/hands for reference-policy ordering (independent of gate hand count).
_SANITY_HANDS = 400


def run_gate_eval(
    bot_factory: Callable[[], object],
    *,
    hands: int,
    seed: int,
    opponent_pool: tuple[str, ...] = GATE_OPPONENT_POOL,
) -> dict:
    """
    Run relative exploitability + bb100_vs_pool on a fixed static opponent distribution.

    Hard-fails if exploitability harness sanity precondition does not pass.
    """
    harness_sanity = require_exploitability_harness_sanity(hands=_SANITY_HANDS, seed=seed)

    bot = bot_factory()
    exploit = estimate_exploitability(bot, hands=hands, seed=seed)

    pool_results: dict[str, float] = {}
    for name in opponent_pool:
        if name not in BOT_REGISTRY:
            raise ValueError(f"Unknown gate opponent {name!r}; registered: {sorted(BOT_REGISTRY)}")
        opp = BOT_REGISTRY[name](random.Random(seed))
        hero_fn = lambda s, b=bot: b(s)  # noqa: E731
        opp_fn = lambda s, o=opp: o(s)  # noqa: E731
        r = run_head_to_head(hero_fn, opp_fn, hands=max(hands // len(opponent_pool), 1), seed=seed)
        pool_results[name] = r.bb_per_100

    bb100 = sum(pool_results.values()) / max(len(pool_results), 1)
    return {
        "exploiter_ev_bb_per_hand": exploit.exploiter_ev_bb_per_hand,
        "signed_exploiter_ev_metric_note": SIGNED_EXPLOITER_EV_METRIC_NOTE,
        "br_ev_bb_per_hand": exploit.br_ev_bb_per_hand,
        "bb100_vs_pool": bb100,
        "bb100_by_opponent": pool_results,
        "opponent_pool": list(opponent_pool),
        "seed": seed,
        "hands": hands,
        "harness_sanity_precondition": harness_sanity,
    }
