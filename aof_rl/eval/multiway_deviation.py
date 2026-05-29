"""Unilateral deviation gain (oracle BR) vs frozen policy in 3/4-max."""

from __future__ import annotations

from aof_rl.eval.deal_stream import DealStream
from aof_rl.eval.eval_schema import SeatDeviationGain
from aof_rl.eval.policy_adapter import EvalPolicy, OracleEvalPolicy
from aof_rl.eval.rollouts import play_deal_symmetric, play_deal_with_deviator
from aof_rl.teacher import NashTeacherCache


def compute_multiway_deviation(
    policy: EvalPolicy,
    deal_stream: DealStream,
    cache: NashTeacherCache,
    *,
    table_sizes: tuple[int, ...] = (3, 4),
    teacher_samples: int = 800,
    payoff_samples: int = 1200,
    eval_seed: int = 0,
) -> dict[str, dict[str, SeatDeviationGain]]:
    """
    Per-seat unilateral deviation gain using Nash oracle as BR.

    deviation_gain[s] = E[u_s | oracle deviates at s] - E[u_s | policy at s].
    """
    br = OracleEvalPolicy(name="oracle_br")
    out: dict[str, dict[str, SeatDeviationGain]] = {}

    for ts in table_sizes:
        if ts not in (3, 4):
            continue
        sub = DealStream(
            hands=deal_stream.hands,
            base_seed=deal_stream.base_seed,
            table_sizes=(ts,),
            stack_bb=deal_stream.stack_bb,
        )
        seats_seen: set[str] = set()
        for deal in sub.iter_deals():
            seats_seen.update(deal.seats)

        seat_gains: dict[str, list[float]] = {s: [] for s in seats_seen}
        for deal in sub.iter_deals():
            base, _ = play_deal_symmetric(
                deal,
                policy,
                cache,
                teacher_samples=teacher_samples,
                payoff_samples=payoff_samples,
                eval_seed=eval_seed,
                stack_bb=deal_stream.stack_bb,
            )
            for seat in deal.seats:
                deviated = play_deal_with_deviator(
                    deal,
                    policy,
                    seat,
                    br,
                    cache,
                    teacher_samples=teacher_samples,
                    payoff_samples=payoff_samples,
                    eval_seed=eval_seed,
                    stack_bb=deal_stream.stack_bb,
                )
                gain = deviated.payoffs.get(seat, 0.0) - base.payoffs.get(seat, 0.0)
                seat_gains[seat].append(gain)

        out[str(ts)] = {
            seat: SeatDeviationGain(
                deviation_gain_bb_per_hand=sum(vals) / max(len(vals), 1),
                hands=len(vals),
            )
            for seat, vals in seat_gains.items()
        }

    return out
