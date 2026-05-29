"""Backward-compatible re-exports; prefer rollouts.py and paired_eval.py."""

from __future__ import annotations

from aof_rl.eval.paired_eval import mean_payoff_per_hand, paired_hand_delta
from aof_rl.eval.rollouts import (
    DecisionRecord,
    HandOutcome,
    play_deal_symmetric,
    play_deal_with_deviator,
)

__all__ = [
    "DecisionRecord",
    "HandOutcome",
    "mean_payoff_per_hand",
    "paired_hand_delta",
    "play_deal_symmetric",
    "play_deal_with_deviator",
]
