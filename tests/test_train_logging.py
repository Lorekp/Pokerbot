"""Unit tests for the ETA progress logger (no JAX, deterministic fake clock)."""

from __future__ import annotations

from pokerbot.training.train_logging import ProgressLogger, fmt_duration


def test_fmt_duration():
    assert fmt_duration(9) == "9s"
    assert fmt_duration(125) == "2m05s"
    assert fmt_duration(3723) == "1h02m03s"
    assert fmt_duration(-5) == "0s"


def test_progress_logger_eta_math():
    # clock: begin() reads 100.0; line(5) reads 110.0 -> elapsed 10s over 5 iters = 2 s/iter.
    ticks = iter([100.0, 110.0])
    log = ProgressLogger(10, every=5, label="t", clock=lambda: next(ticks)).begin()
    line = log.line(5)
    assert line is not None
    assert "iter 5/10" in line
    assert "2000 ms/iter" in line          # 2.0 s/iter
    assert "elapsed 10s" in line
    assert "ETA 10s" in line               # 2 s/iter * (10-5) remaining


def test_progress_logger_skips_non_ticks_and_disabled():
    log = ProgressLogger(10, every=5, clock=lambda: 0.0).begin()
    assert log.line(3) is None             # not a multiple of 5, not the last iter
    assert log.line(10) is not None        # final iter always logs
    off = ProgressLogger(10, every=0, clock=lambda: 0.0).begin()
    assert off.line(5) is None             # every=0 disables logging entirely


def test_extra_callable_appended_and_failsafe():
    log = ProgressLogger(2, every=1, clock=iter([0.0, 1.0, 2.0]).__next__,
                         extra=lambda: "buf=42").begin()
    assert "buf=42" in log.line(1)
    # a throwing extra must never break logging
    boom = ProgressLogger(2, every=1, clock=iter([0.0, 1.0]).__next__,
                          extra=lambda: (_ for _ in ()).throw(RuntimeError("x"))).begin()
    assert boom.line(1) is not None        # line still produced despite extra() raising
