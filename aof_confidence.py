"""Structured advisor confidence and approximation metadata."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any

MARGINAL_EDGE_BB = 0.30  # keep in sync with aof.MARGINAL_EDGE_BB


TIER_USER_LABELS: dict[str, str] = {
    "solver_hu": "HU solver-backed",
    "static_profile_single": "static-profile, not Nash",
    "static_profile_multiway_true_equity": "static-profile multiway equity",
    "approx_call_behind": "call-with-behind approximation",
    "approx_facing_two_conditional": "conditional 3-way MC (Facing 2)",
    "approx_facing_three_conditional": "conditional 4-way MC (Facing 3)",
    "approx_facing_three_fallback": "4-way MC with range fallback (Facing 3)",
    "approx_fallback_static": "fallback static range",
    "unsupported": "unsupported spot",
}


class ApproximationTier(str, Enum):
    SOLVER_HU = "solver_hu"
    STATIC_PROFILE_HU_OR_SINGLE = "static_profile_single"
    STATIC_PROFILE_MULTIWAY_TRUE_EQUITY = "static_profile_multiway_true_equity"
    APPROX_CALL_BEHIND = "approx_call_behind"
    APPROX_FACING_TWO_CONDITIONAL = "approx_facing_two_conditional"
    APPROX_FACING_THREE_CONDITIONAL = "approx_facing_three_conditional"
    APPROX_FACING_THREE_FALLBACK = "approx_facing_three_fallback"
    APPROX_FALLBACK_STATIC = "approx_fallback_static"
    UNSUPPORTED = "unsupported"


@dataclass(frozen=True)
class AdvisorConfidence:
    edge_bb: float
    marginal: bool
    equity_method: str
    samples_used: int | None
    mc_stdev_estimate: float | None
    range_provenance: str
    fallback_used: bool
    approximation_tier: str
    range_sensitivity_flag: bool | None = None
    warning: str | None = None
    rake_configured: bool = False
    side_pots_supported: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def tier_from_detail_prefix(prefix: str) -> ApproximationTier:
    if prefix.startswith("hu-solver:"):
        return ApproximationTier.SOLVER_HU
    if prefix.startswith("4way-exact+conditional:"):
        return ApproximationTier.APPROX_FACING_THREE_CONDITIONAL
    if prefix.startswith("4way-mc+conditional:"):
        return ApproximationTier.APPROX_FACING_THREE_CONDITIONAL
    if prefix.startswith("4way-exact+fallback:"):
        return ApproximationTier.APPROX_FACING_THREE_FALLBACK
    if prefix.startswith("4way-mc+fallback:"):
        return ApproximationTier.APPROX_FACING_THREE_FALLBACK
    if prefix.startswith("3way-mc+conditional:"):
        return ApproximationTier.APPROX_FACING_TWO_CONDITIONAL
    if prefix.startswith("approx fallback static 2-shover:"):
        return ApproximationTier.APPROX_FALLBACK_STATIC
    if prefix.startswith("approx fallback:") or prefix.startswith("approx fallback"):
        return ApproximationTier.APPROX_FALLBACK_STATIC
    if prefix.startswith("approx:"):
        return ApproximationTier.APPROX_CALL_BEHIND
    if prefix.startswith("static-profile:"):
        return ApproximationTier.STATIC_PROFILE_HU_OR_SINGLE
    return ApproximationTier.UNSUPPORTED


def build_confidence(
    *,
    action: str,
    ev_primary: float,
    ev_alt: float,
    detail_prefix: str,
    equity_method: str = "mc",
    samples_used: int | None = None,
    mc_stdev_estimate: float | None = None,
    range_provenance: str = "unknown",
    fallback_used: bool = False,
    range_sensitivity_flag: bool | None = None,
    rake_configured: bool = False,
    side_pots_supported: bool = False,
    explicit_warning: str | None = None,
) -> AdvisorConfidence:
    edge = abs(
        (ev_primary if action == "ALL IN" else ev_alt)
        - (ev_alt if action == "ALL IN" else ev_primary)
    )
    tier = tier_from_detail_prefix(detail_prefix)
    marginal = edge < MARGINAL_EDGE_BB
    warnings: list[str] = []
    if marginal:
        warnings.append(f"thin edge ({edge:.2f}bb < {MARGINAL_EDGE_BB}bb)")
    if fallback_used:
        warnings.append("fallback static range used")
    if tier in (
        ApproximationTier.STATIC_PROFILE_HU_OR_SINGLE,
        ApproximationTier.STATIC_PROFILE_MULTIWAY_TRUE_EQUITY,
    ) and range_provenance in ("hand_tuned", "static_profile"):
        warnings.append("static-profile, not Nash")
    if tier == ApproximationTier.APPROX_CALL_BEHIND:
        warnings.append("call-with-behind approximation")
    if tier == ApproximationTier.APPROX_FACING_TWO_CONDITIONAL:
        warnings.append("conditional 3-way MC (Facing 2)")
    if tier == ApproximationTier.APPROX_FACING_THREE_CONDITIONAL:
        warnings.append("conditional 4-way MC (Facing 3)")
    if tier == ApproximationTier.APPROX_FACING_THREE_FALLBACK:
        warnings.append("Facing 3 third-shover fallback range used")
    if tier == ApproximationTier.APPROX_FALLBACK_STATIC:
        warnings.append("Facing 2 fallback static range")
    if mc_stdev_estimate is not None and edge > 0 and mc_stdev_estimate > edge:
        warnings.append("MC noise large vs edge")
    if rake_configured:
        warnings.append("rake-adjusted chip EV")
    if not side_pots_supported:
        pass  # informational only when unequal stacks attempted
    if explicit_warning:
        warnings.insert(0, explicit_warning)
    warning = "; ".join(warnings) if warnings else None
    return AdvisorConfidence(
        edge_bb=edge,
        marginal=marginal,
        equity_method=equity_method,
        samples_used=samples_used,
        mc_stdev_estimate=mc_stdev_estimate,
        range_provenance=range_provenance,
        fallback_used=fallback_used,
        approximation_tier=tier.value,
        range_sensitivity_flag=range_sensitivity_flag,
        warning=warning,
        rake_configured=rake_configured,
        side_pots_supported=side_pots_supported,
    )
