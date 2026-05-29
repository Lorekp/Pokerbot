"""Paired comparison helpers for eval and exploit harnesses."""

from __future__ import annotations

from aof_rl.eval.rollouts import HandOutcome


def mean_payoff_per_hand(outcomes: list[HandOutcome]) -> float:
    if not outcomes:
        return 0.0
    total = sum(sum(o.payoffs.values()) for o in outcomes)
    n_obs = sum(len(o.payoffs) for o in outcomes)
    return total / max(n_obs, 1)


def paired_hand_delta(
    outcome_a: HandOutcome,
    outcome_b: HandOutcome,
) -> float:
    """Mean seat payoff under A minus mean under B for the same deal."""
    seats = set(outcome_a.payoffs) | set(outcome_b.payoffs)
    if not seats:
        return 0.0
    return sum(
        outcome_a.payoffs.get(s, 0.0) - outcome_b.payoffs.get(s, 0.0) for s in seats
    ) / len(seats)
