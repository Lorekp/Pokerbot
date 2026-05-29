"""Aggregate metrics, confidence intervals, calibration buckets."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from aof_rl.eval.eval_schema import (
    THIN_EDGE_BUCKET_LABELS,
    CalibrationBucket,
    SliceMetrics,
)
from aof_rl.eval.rollouts import DecisionRecord, HandOutcome

THIN_EDGE_EDGES = (0.0, 0.1, 0.25, 0.5, float("inf"))


def ci95(samples: np.ndarray) -> tuple[float, float]:
    if samples.size == 0:
        return (0.0, 0.0)
    if samples.size == 1:
        v = float(samples[0])
        return (v, v)
    lo, hi = np.percentile(samples, [2.5, 97.5])
    return (float(lo), float(hi))


@dataclass
class MetricsAccumulator:
    """Collect per (hand, seat) payoffs and per-decision teacher stats."""

    seat_payoffs: dict[str, list[float]] = field(default_factory=dict)
    table_payoffs: dict[int, list[float]] = field(default_factory=dict)
    all_payoffs: list[float] = field(default_factory=list)
    decisions: list[DecisionRecord] = field(default_factory=list)
    paired_deltas: list[float] = field(default_factory=list)
    seed_means: list[float] = field(default_factory=list)

    def add_hand(self, outcome: HandOutcome) -> None:
        for seat, bb in outcome.payoffs.items():
            self.seat_payoffs.setdefault(seat, []).append(bb)
            self.all_payoffs.append(bb)
        ts = outcome.table_size
        self.table_payoffs.setdefault(ts, []).append(
            sum(outcome.payoffs.values()) / max(len(outcome.payoffs), 1)
        )

    def add_decisions(self, records: list[DecisionRecord]) -> None:
        self.decisions.extend(records)

    def add_paired_delta(self, delta: float) -> None:
        self.paired_deltas.append(delta)

    def add_seed_mean(self, mean_bb: float) -> None:
        self.seed_means.append(mean_bb)

    def _disagreement_rate(self, records: list[DecisionRecord]) -> float | None:
        with_teacher = [
            r
            for r in records
            if r.teacher_action is not None and r.teacher_action >= 0
        ]
        if not with_teacher:
            return None
        disagree = sum(1 for r in with_teacher if r.action != r.teacher_action)
        return disagree / len(with_teacher)

    def _calibration_buckets(self) -> list[CalibrationBucket]:
        buckets: list[list[DecisionRecord]] = [[] for _ in THIN_EDGE_BUCKET_LABELS]
        for r in self.decisions:
            if r.teacher_edge_bb is None:
                continue
            edge = abs(r.teacher_edge_bb)
            for i in range(len(THIN_EDGE_BUCKET_LABELS)):
                lo, hi = THIN_EDGE_EDGES[i], THIN_EDGE_EDGES[i + 1]
                if edge >= lo and edge < hi:
                    buckets[i].append(r)
                    break
        out: list[CalibrationBucket] = []
        for label, recs in zip(THIN_EDGE_BUCKET_LABELS, buckets):
            if not recs:
                out.append(
                    CalibrationBucket(
                        label=label,
                        count=0,
                        disagreement_rate=0.0,
                        mean_q_error_bb=0.0,
                        mean_uncertainty_bb=0.0,
                    )
                )
                continue
            disagree = sum(
                1
                for r in recs
                if r.teacher_action is not None and r.action != r.teacher_action
            )
            q_errs: list[float] = []
            uncs: list[float] = []
            for r in recs:
                if r.q_bb is not None and r.teacher_q_bb is not None:
                    chosen = r.q_bb[r.action]
                    t_chosen = r.teacher_q_bb[r.action]
                    q_errs.append(abs(chosen - t_chosen))
                if r.uncertainty_bb is not None:
                    uncs.append(r.uncertainty_bb)
            out.append(
                CalibrationBucket(
                    label=label,
                    count=len(recs),
                    disagreement_rate=disagree / len(recs),
                    mean_q_error_bb=float(np.mean(q_errs)) if q_errs else 0.0,
                    mean_uncertainty_bb=float(np.mean(uncs)) if uncs else 0.0,
                )
            )
        return out

    def _slice_from_payoffs(
        self,
        payoffs: list[float],
        decisions: list[DecisionRecord],
    ) -> SliceMetrics:
        arr = np.asarray(payoffs, dtype=np.float64)
        n = int(arr.size)
        mean = float(arr.mean()) if n else 0.0
        if self.seed_means:
            ci = ci95(np.asarray(self.seed_means, dtype=np.float64))
        elif n > 1:
            ci = ci95(arr)
        else:
            ci = (mean, mean)
        return SliceMetrics(
            hands=n,
            bb_per_hand=mean,
            bb_per_hand_ci95=ci,
            bb_per_100=mean * 100.0,
            disagreement_rate_vs_teacher=self._disagreement_rate(decisions),
        )

    def aggregate(self) -> tuple[SliceMetrics, dict[str, SliceMetrics], dict[str, SliceMetrics], list[CalibrationBucket]]:
        agg = self._slice_from_payoffs(self.all_payoffs, self.decisions)
        by_seat = {
            seat: self._slice_from_payoffs(pay, [d for d in self.decisions if d.seat == seat])
            for seat, pay in self.seat_payoffs.items()
        }
        by_table = {
            str(ts): self._slice_from_payoffs(
                pay,
                [d for d in self.decisions if d.table_size == ts],
            )
            for ts, pay in self.table_payoffs.items()
        }
        return agg, by_seat, by_table, self._calibration_buckets()

    def paired_stats(self) -> tuple[float, tuple[float, float], int]:
        arr = np.asarray(self.paired_deltas, dtype=np.float64)
        n = int(arr.size)
        if n == 0:
            return 0.0, (0.0, 0.0), 0
        mean = float(arr.mean())
        return mean, ci95(arr), n
