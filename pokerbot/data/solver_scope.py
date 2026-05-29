"""Label validity scope checks for postflop-solver SFT rows."""

from __future__ import annotations

from typing import Any

POSTFLOP_SCOPE_COLUMNS = (
    "oop_range",
    "ip_range",
    "bet_sizes",
    "solver_converged",
    "spot_class",
)


def _missing_columns(row: dict[str, Any], columns: tuple[str, ...]) -> list[str]:
    return [c for c in columns if c not in row or row[c] is None]


def validate_postflop_solver_row_scope(
    row: dict[str, Any],
    *,
    require_converged: bool = True,
    expected_oop_range: str | None = None,
    expected_ip_range: str | None = None,
    expected_bet_sizes: str | None = None,
    allowed_spot_classes: frozenset[str] | None = None,
    source: str = "row",
) -> None:
    """Reject rows that must not supervise SFT (unconverged or scope mismatch)."""
    missing = _missing_columns(row, POSTFLOP_SCOPE_COLUMNS)
    if missing:
        raise ValueError(
            f"{source}: postflop_solver row missing scope columns {missing}. "
            "Narrow-range labels require oop_range/ip_range/bet_sizes metadata."
        )

    if require_converged and not bool(row["solver_converged"]):
        raise ValueError(
            f"{source}: solver_converged=false — unconverged DCFR snapshots are not GTO labels."
        )

    if allowed_spot_classes is not None and str(row["spot_class"]) not in allowed_spot_classes:
        raise ValueError(
            f"{source}: spot_class={row['spot_class']!r} not in allowed {sorted(allowed_spot_classes)!r}"
        )

    if expected_oop_range is not None and str(row["oop_range"]) != expected_oop_range:
        raise ValueError(
            f"{source}: oop_range {row['oop_range']!r} != expected {expected_oop_range!r}. "
            "Applying a narrow-range label to a wider-range spot is invalid supervision."
        )

    if expected_ip_range is not None and str(row["ip_range"]) != expected_ip_range:
        raise ValueError(
            f"{source}: ip_range {row['ip_range']!r} != expected {expected_ip_range!r}. "
            "Applying a narrow-range label to a wider-range spot is invalid supervision."
        )

    if expected_bet_sizes is not None and str(row["bet_sizes"]) != expected_bet_sizes:
        raise ValueError(
            f"{source}: bet_sizes {row['bet_sizes']!r} != expected {expected_bet_sizes!r}."
        )


def scope_summary(row: dict[str, Any]) -> str:
    return (
        f"spot_class={row.get('spot_class')} "
        f"OOP={row.get('oop_range')} IP={row.get('ip_range')} "
        f"bet_tree={row.get('bet_sizes')}"
    )
