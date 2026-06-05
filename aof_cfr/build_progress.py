"""Shared progress + ETA logging for long-running tensor/build jobs."""

from __future__ import annotations

import time


def fmt_duration(seconds: float) -> str:
    seconds = max(0.0, seconds)
    if seconds >= 3600:
        return f"{seconds / 3600:.1f}h"
    if seconds >= 60:
        return f"{seconds / 60:.1f}m"
    return f"{seconds:.0f}s"


def log_build_progress(
    current: int,
    total: int,
    started: float,
    *,
    label: str = "progress",
    unit: str = "items",
    min_interval_s: float = 0.0,
    _last_log: list[float] | None = None,
) -> None:
    """Print one progress line with elapsed wall time and ETA."""
    if _last_log is not None and min_interval_s > 0:
        now = time.monotonic()
        if current not in (1, total) and now - _last_log[0] < min_interval_s:
            return
        _last_log[0] = now

    elapsed = time.monotonic() - started
    rate = current / elapsed if elapsed > 0 else 0.0
    remaining = max(0, total - current)
    eta = remaining / rate if rate > 0 else 0.0
    pct = 100.0 * current / total if total else 100.0
    print(
        f"  {label} {current}/{total} ({pct:.0f}%) "
        f"elapsed={fmt_duration(elapsed)} eta={fmt_duration(eta)} "
        f"({rate:.2f} {unit}/s)",
        flush=True,
    )
