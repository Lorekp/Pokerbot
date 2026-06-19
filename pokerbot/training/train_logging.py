"""Lightweight ETA progress logging for CFR training loops (no device syncs in the hot path).

Used by the Leduc gate and any training driver to print periodic progress with a rolling ms/iter and
an ETA. Kept in its own module so it adds zero coupling to the trainer and can be dropped into
``VRDeepPDCFRPlusTrainer.train`` or a cloud driver with one line.
"""

from __future__ import annotations

import time
from typing import Callable


def fmt_duration(seconds: float) -> str:
    """Human ``1h02m03s`` / ``4m05s`` / ``9s`` from a seconds count."""
    s = int(max(0.0, seconds))
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m{sec:02d}s"
    if m:
        return f"{m}m{sec:02d}s"
    return f"{sec}s"


class ProgressLogger:
    """Periodic ``iter X/T | ms/iter | elapsed | ETA`` logger.

    ``every`` = log cadence in iterations (0 disables). ``extra`` is an optional cheap callable
    returning a trailing string (e.g. buffer sizes) -- keep it sync-free / device-cheap so logging
    never stalls training. ``label`` prefixes each line.
    """

    def __init__(self, total: int, *, start_iter: int = 0, every: int = 0, label: str = "train",
                 extra: Callable[[], str] | None = None, clock: Callable[[], float] = time.time) -> None:
        self.total = total
        self.start_iter = start_iter
        self.every = every
        self.label = label
        self.extra = extra
        self._clock = clock
        self._t0: float | None = None

    def begin(self) -> "ProgressLogger":
        self._t0 = self._clock()
        return self

    def line(self, it: int) -> str | None:
        """The log line for iteration ``it`` (or None if this iter is not a logging tick)."""
        if not self.every or self._t0 is None:
            return None
        done = it - self.start_iter
        is_last = it >= self.total
        if done <= 0 or (it % self.every != 0 and not is_last):
            return None
        elapsed = self._clock() - self._t0
        total = self.total - self.start_iter
        per = elapsed / max(done, 1)
        eta = per * (total - done)
        msg = (f"[{self.label}] iter {it}/{self.total}  {per * 1000:.0f} ms/iter  "
               f"elapsed {fmt_duration(elapsed)}  ETA {fmt_duration(eta)}")
        if self.extra is not None:
            try:
                msg += "  " + self.extra()
            except Exception:  # noqa: BLE001 -- logging must never break training
                pass
        return msg

    def step(self, it: int) -> None:
        """Print the progress line if ``it`` is a logging tick."""
        line = self.line(it)
        if line is not None:
            print(line, flush=True)
