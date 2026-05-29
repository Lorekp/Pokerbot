"""Equity test helpers with rich failure messages."""

from __future__ import annotations

import pytest

from aof_equity_config import EquityResult


def assert_equity_close(
    actual: float,
    expected: float,
    tolerance: float,
    *,
    method: str,
    samples: int | None = None,
    seed: int | None = None,
    note: str = "",
    valid_combo_products: int | None = None,
) -> None:
    """Assert equity within tolerance; include MC/exact metadata on failure."""
    if abs(actual - expected) <= tolerance:
        return
    parts = [
        f"equity mismatch: got {actual:.6f} expected {expected:.6f} ± {tolerance}",
        f"method={method!r}",
    ]
    if samples is not None:
        parts.append(f"samples={samples}")
    if seed is not None:
        parts.append(f"seed={seed}")
    if valid_combo_products is not None:
        parts.append(f"valid_combo_products={valid_combo_products}")
    if note:
        parts.append(f"note={note}")
    pytest.fail("; ".join(parts))


def assert_equity_result_close(
    result: EquityResult,
    expected: float,
    tolerance: float,
    *,
    note: str = "",
    expect_method: str | None = None,
) -> None:
    if expect_method is not None and result.method != expect_method:
        pytest.fail(
            f"expected method {expect_method!r}, got {result.method!r}; "
            f"equity={result.equity:.6f} seed={result.seed}"
        )
    assert_equity_close(
        result.equity,
        expected,
        tolerance,
        method=result.method,
        samples=result.samples_used,
        seed=result.seed,
        note=note,
        valid_combo_products=result.valid_combo_products,
    )
