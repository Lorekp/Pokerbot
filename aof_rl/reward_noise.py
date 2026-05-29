"""Reward-noise floor for payoff abstraction accept/reject gates."""

from __future__ import annotations

import math
from typing import Sequence


def reward_noise_floor_bb(
    returns_per_hand: Sequence[float],
    *,
    batch_hands: int,
    noise_k: float = 0.25,
) -> float:
    """
    Per-update advantage-estimation noise scale in bb.

    reward_noise_floor_bb = k * std(returns) / sqrt(batch_hands)

    Abstraction error (p95 |ΔEV_bb|) should be below this to be negligible vs PPO.
    """
    if batch_hands <= 0:
        raise ValueError("batch_hands must be positive")
    if not returns_per_hand:
        return 0.0
    n = len(returns_per_hand)
    mean = sum(returns_per_hand) / n
    var = sum((r - mean) ** 2 for r in returns_per_hand) / max(n - 1, 1)
    std = math.sqrt(var) if var > 0 else 0.0
    return float(noise_k * std / math.sqrt(batch_hands))


def reward_noise_floor_definition(
    *,
    batch_hands: int,
    noise_k: float = 0.25,
) -> dict[str, object]:
    """JSON-serializable definition block for audit artifacts."""
    return {
        "formula": "k * std(returns_per_hand_in_update) / sqrt(batch_hands)",
        "batch_hands": batch_hands,
        "noise_k": noise_k,
    }
