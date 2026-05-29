"""JSON schema v1 for AoF RL evaluation reports."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any


EVAL_SCHEMA_VERSION = 1

THIN_EDGE_BUCKET_LABELS = ("0-0.1bb", "0.1-0.25bb", "0.25-0.5bb", "0.5bb+")


@dataclass
class SliceMetrics:
    hands: int
    bb_per_hand: float
    bb_per_hand_ci95: tuple[float, float]
    bb_per_100: float
    disagreement_rate_vs_teacher: float | None = None


@dataclass
class CalibrationBucket:
    label: str
    count: int
    disagreement_rate: float
    mean_q_error_bb: float
    mean_uncertainty_bb: float


@dataclass
class HuExploitabilityResult:
    mode: str
    exploitability_bb_per_hand: float
    sb_br_gain_bb: float
    bb_br_gain_bb: float
    hands_enumerated: int
    note: str = ""


@dataclass
class SeatDeviationGain:
    deviation_gain_bb_per_hand: float
    hands: int


@dataclass
class PairedComparison:
    policy_a: str
    policy_b: str
    hands: int
    mean_delta_bb_per_hand: float
    delta_bb_per_hand_ci95: tuple[float, float]
    mean_delta_bb_per_100: float


@dataclass
class EvalReport:
    eval_id: str
    policy: dict[str, Any]
    config: dict[str, Any]
    aggregate: SliceMetrics
    by_seat: dict[str, SliceMetrics]
    by_table_size: dict[str, SliceMetrics]
    calibration: dict[str, list[CalibrationBucket]]
    hu_exploitability: HuExploitabilityResult | None = None
    multiway_deviation: dict[str, dict[str, SeatDeviationGain]] = field(
        default_factory=dict
    )
    paired_comparison: PairedComparison | None = None
    generated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    schema_version: int = EVAL_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        def _slice(s: SliceMetrics) -> dict[str, Any]:
            d = asdict(s)
            d["bb_per_hand_ci95"] = list(s.bb_per_hand_ci95)
            return d

        out: dict[str, Any] = {
            "schema_version": self.schema_version,
            "generated_at": self.generated_at,
            "eval_id": self.eval_id,
            "policy": self.policy,
            "config": self.config,
            "aggregate": _slice(self.aggregate),
            "by_seat": {k: _slice(v) for k, v in self.by_seat.items()},
            "by_table_size": {k: _slice(v) for k, v in self.by_table_size.items()},
            "calibration": {
                "thin_edge_buckets": [asdict(b) for b in self.calibration["thin_edge_buckets"]],
            },
            "multiway_deviation": {
                ts: {seat: asdict(g) for seat, g in seats.items()}
                for ts, seats in self.multiway_deviation.items()
            },
        }
        if self.hu_exploitability is not None:
            out["hu_exploitability"] = asdict(self.hu_exploitability)
        if self.paired_comparison is not None:
            pc = asdict(self.paired_comparison)
            pc["delta_bb_per_hand_ci95"] = list(self.paired_comparison.delta_bb_per_hand_ci95)
            out["paired_comparison"] = pc
        return out


def validate_report(data: dict[str, Any]) -> None:
    """Raise AssertionError if required keys are missing."""
    from aof_validation.snapshot_schema import check_aof_eval_report

    check_aof_eval_report(data)
