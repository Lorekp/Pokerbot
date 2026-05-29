"""Symmetric multi-seat rollouts on fixed deals (shared by eval and exploit)."""

from __future__ import annotations

from dataclasses import dataclass

from aof_rl.env import reset_hand_from_holes, step_hand
from aof_rl.eval.deal_stream import DealSpec
from aof_rl.eval.policy_adapter import EvalPolicy
from aof_rl.teacher import NashTeacherCache


@dataclass
class DecisionRecord:
    hand_index: int
    table_size: int
    seat: str
    action: int
    teacher_action: int | None
    teacher_edge_bb: float | None
    q_bb: tuple[float, float] | None
    teacher_q_bb: tuple[float, float] | None
    uncertainty_bb: float | None


@dataclass
class HandOutcome:
    hand_index: int
    table_size: int
    payoffs: dict[str, float]


def play_deal_symmetric(
    deal: DealSpec,
    policy: EvalPolicy,
    cache: NashTeacherCache,
    *,
    teacher_samples: int,
    payoff_samples: int,
    eval_seed: int,
    stack_bb: float = 8.0,
) -> tuple[HandOutcome, list[DecisionRecord]]:
    """All seats use the same policy; record every decision."""
    state = reset_hand_from_holes(
        deal.seats,
        deal.holes,
        table_size=deal.table_size,
        stack_bb=stack_bb,
        hand_id=deal.hand_index,
    )
    records: list[DecisionRecord] = []
    guard = 0
    while not state.terminal:
        guard += 1
        if guard > 32:
            raise RuntimeError("Hand guard exceeded")
        seat = state.action_seat
        if seat is None:
            break
        decision_seed = eval_seed + deal.hand_index * 1000 + guard
        cache.get_with_path(
            state, seat, samples=teacher_samples, seed=decision_seed
        )
        d = policy.act(state, seat, cache, decision_seed=decision_seed)
        records.append(
            DecisionRecord(
                hand_index=deal.hand_index,
                table_size=deal.table_size,
                seat=seat,
                action=d.action,
                teacher_action=d.teacher_action,
                teacher_edge_bb=d.teacher_edge_bb,
                q_bb=d.q_bb,
                teacher_q_bb=d.teacher_q_bb,
                uncertainty_bb=d.uncertainty_bb,
            )
        )
        payoff_seed = deal.payoff_base_seed + guard
        state = step_hand(
            state,
            d.action,
            payoff_samples=payoff_samples,
            payoff_seed=payoff_seed,
        )
    outcome = HandOutcome(
        hand_index=deal.hand_index,
        table_size=deal.table_size,
        payoffs=dict(state.payoffs),
    )
    return outcome, records


def play_deal_with_deviator(
    deal: DealSpec,
    policy: EvalPolicy,
    deviator_seat: str,
    br_policy: EvalPolicy,
    cache: NashTeacherCache,
    *,
    teacher_samples: int,
    payoff_samples: int,
    eval_seed: int,
    stack_bb: float = 8.0,
) -> HandOutcome:
    """Frozen opponents use ``policy``; ``deviator_seat`` uses ``br_policy``."""
    state = reset_hand_from_holes(
        deal.seats,
        deal.holes,
        table_size=deal.table_size,
        stack_bb=stack_bb,
        hand_id=deal.hand_index,
    )
    guard = 0
    while not state.terminal:
        guard += 1
        if guard > 32:
            raise RuntimeError("Hand guard exceeded")
        seat = state.action_seat
        if seat is None:
            break
        decision_seed = eval_seed + deal.hand_index * 1000 + guard
        actor = br_policy if seat == deviator_seat else policy
        d = actor.act(state, seat, cache, decision_seed=decision_seed)
        payoff_seed = deal.payoff_base_seed + guard
        state = step_hand(
            state,
            d.action,
            payoff_samples=payoff_samples,
            payoff_seed=payoff_seed,
        )
    return HandOutcome(
        hand_index=deal.hand_index,
        table_size=deal.table_size,
        payoffs=dict(state.payoffs),
    )
