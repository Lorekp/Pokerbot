"""Wall-clock timing accumulator for AoF league training."""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator


@dataclass
class BucketStats:
    seconds: float = 0.0
    calls: int = 0

    def add(self, sec: float, *, calls: int = 1) -> None:
        self.seconds += sec
        self.calls += calls


@dataclass
class TimingAccumulator:
    """Flat bucket timings; labels are dotted paths."""

    buckets: dict[str, BucketStats] = field(default_factory=dict)
    wall_start: float | None = None
    wall_end: float | None = None
    counters: dict[str, int] = field(default_factory=dict)
    update_snapshots: list[dict[str, Any]] = field(default_factory=list)
    spot_snapshots: list[dict[str, Any]] = field(default_factory=list)
    _baseline_buckets: dict[str, BucketStats] = field(default_factory=dict, repr=False)
    _update_wall_start: float | None = field(default=None, repr=False)
    _spot_wall_start: float | None = field(default=None, repr=False)
    _current_update_idx: int | None = field(default=None, repr=False)
    _current_spot_idx: int | None = field(default=None, repr=False)
    _current_stage: str | None = field(default=None, repr=False)
    _current_spot_key: str | None = field(default=None, repr=False)

    def start_wall(self) -> None:
        self.wall_start = time.perf_counter()

    def stop_wall(self) -> float:
        if self.wall_start is None:
            self.wall_start = time.perf_counter()
        self.wall_end = time.perf_counter()
        return self.total_wall_seconds()

    def total_wall_seconds(self) -> float:
        if self.wall_start is None:
            return 0.0
        end = self.wall_end if self.wall_end is not None else time.perf_counter()
        return end - self.wall_start

    def add(self, name: str, sec: float, *, calls: int = 1) -> None:
        b = self.buckets.setdefault(name, BucketStats())
        b.add(sec, calls=calls)

    def inc_counter(self, name: str, n: int = 1) -> None:
        self.counters[name] = self.counters.get(name, 0) + n

    @contextmanager
    def measure(self, name: str) -> Iterator[None]:
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.add(name, time.perf_counter() - t0)

    def merge(self, other: TimingAccumulator) -> None:
        for name, stats in other.buckets.items():
            b = self.buckets.setdefault(name, BucketStats())
            b.seconds += stats.seconds
            b.calls += stats.calls
        for k, v in other.counters.items():
            self.counters[k] = self.counters.get(k, 0) + v
        self.update_snapshots.extend(other.update_snapshots)
        self.spot_snapshots.extend(other.spot_snapshots)

    def _copy_buckets(self) -> dict[str, BucketStats]:
        return {
            name: BucketStats(stats.seconds, stats.calls)
            for name, stats in self.buckets.items()
        }

    def begin_update(self, update_idx: int, stage_name: str) -> None:
        self._baseline_buckets = self._copy_buckets()
        self._update_wall_start = time.perf_counter()
        self._current_update_idx = update_idx
        self._current_stage = stage_name

    def end_update(self) -> None:
        if self._update_wall_start is None:
            return
        wall = time.perf_counter() - self._update_wall_start
        buckets_delta: dict[str, dict[str, float | int]] = {}
        all_names = set(self.buckets) | set(self._baseline_buckets)
        for name in all_names:
            cur = self.buckets.get(name, BucketStats())
            base = self._baseline_buckets.get(name, BucketStats())
            ds = cur.seconds - base.seconds
            dc = cur.calls - base.calls
            if ds > 0.0 or dc > 0:
                buckets_delta[name] = {
                    "seconds": round(ds, 4),
                    "calls": dc,
                }
        top_buckets = sorted(
            buckets_delta.items(),
            key=lambda item: float(item[1]["seconds"]),
            reverse=True,
        )[:5]
        self.update_snapshots.append(
            {
                "update": self._current_update_idx,
                "stage": self._current_stage,
                "wall_seconds": round(wall, 4),
                "top_buckets": [
                    {"name": name, **stats} for name, stats in top_buckets
                ],
                "buckets": buckets_delta,
            }
        )
        self._update_wall_start = None
        self._current_update_idx = None
        self._current_stage = None
        self._baseline_buckets = {}

    def begin_spot(self, spot_idx: int, spot_key: str, stage_name: str) -> None:
        self._baseline_buckets = self._copy_buckets()
        self._spot_wall_start = time.perf_counter()
        self._current_spot_idx = spot_idx
        self._current_spot_key = spot_key
        self._current_stage = stage_name

    def end_spot(self) -> None:
        if self._spot_wall_start is None:
            return
        wall = time.perf_counter() - self._spot_wall_start
        buckets_delta: dict[str, dict[str, float | int]] = {}
        all_names = set(self.buckets) | set(self._baseline_buckets)
        for name in all_names:
            cur = self.buckets.get(name, BucketStats())
            base = self._baseline_buckets.get(name, BucketStats())
            ds = cur.seconds - base.seconds
            dc = cur.calls - base.calls
            if ds > 0.0 or dc > 0:
                buckets_delta[name] = {
                    "seconds": round(ds, 4),
                    "calls": dc,
                }
        top_buckets = sorted(
            buckets_delta.items(),
            key=lambda item: float(item[1]["seconds"]),
            reverse=True,
        )[:5]
        self.spot_snapshots.append(
            {
                "spot": self._current_spot_idx,
                "spot_key": self._current_spot_key,
                "stage": self._current_stage,
                "wall_seconds": round(wall, 4),
                "top_buckets": [
                    {"name": name, **stats} for name, stats in top_buckets
                ],
                "buckets": buckets_delta,
            }
        )
        self._spot_wall_start = None
        self._current_spot_idx = None
        self._current_spot_key = None
        self._current_stage = None
        self._baseline_buckets = {}

    def sorted_report(self, *, wall_total: float | None = None) -> list[dict[str, Any]]:
        bucket_sum = sum(b.seconds for b in self.buckets.values()) or 1e-9
        total = wall_total if wall_total is not None and wall_total > 0 else bucket_sum
        rows: list[dict[str, Any]] = []
        for rank, (name, stats) in enumerate(
            sorted(self.buckets.items(), key=lambda x: x[1].seconds, reverse=True),
            start=1,
        ):
            rows.append(
                {
                    "rank": rank,
                    "name": name,
                    "seconds": round(stats.seconds, 4),
                    "pct": round(100.0 * stats.seconds / total, 2),
                    "calls": stats.calls,
                    "sec_per_call": round(
                        stats.seconds / max(stats.calls, 1), 6
                    ),
                }
            )
        return rows
