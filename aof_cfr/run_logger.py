"""Timestamped file + stdout logging for long Phase A build jobs."""

from __future__ import annotations

import atexit
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

_DEFAULT_LOG_DIR = Path("data/aof/cfr_base/logs")
_active_log: Path | None = None
_run_start = time.monotonic()


class _Tee:
    """Mirror stdout/stderr to a log file with line buffering."""

    def __init__(self, stream, log_path: Path, *, header: str) -> None:
        self._stream = stream
        self._name = getattr(stream, "name", "stream")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self._file = log_path.open("a", encoding="utf-8", buffering=1)
        self._file.write(header)
        self._file.flush()

    def write(self, data: str) -> int:
        if not data:
            return 0
        self._stream.write(data)
        self._file.write(data)
        if "\n" in data:
            self.flush()
        return len(data)

    def flush(self) -> None:
        self._stream.flush()
        self._file.flush()

    def isatty(self) -> bool:
        return self._stream.isatty()

    def close(self) -> None:
        self._file.close()


def init_run_log(
    name: str,
    *,
    log_dir: Path | str = _DEFAULT_LOG_DIR,
    also_stderr: bool = True,
) -> Path:
    """Attach tee loggers; returns log file path. Idempotent per process."""
    global _active_log
    if _active_log is not None:
        return _active_log

    log_dir = Path(log_dir)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = log_dir / f"{name}_{stamp}.log"
    header = (
        f"# run={name} pid={__import__('os').getpid()} "
        f"started={datetime.now(timezone.utc).isoformat()}\n"
    )
    sys.stdout = _Tee(sys.stdout, path, header=header)
    if also_stderr:
        sys.stderr = _Tee(sys.stderr, path, header="")
    _active_log = path
    atexit.register(log_run_footer, name=name)
    print(f"[run_logger] logging to {path}", flush=True)
    return path


def log_run_footer(*, name: str = "run") -> None:
    print(
        f"[run_logger] {name} finished elapsed={fmt_elapsed()} "
        f"at {datetime.now(timezone.utc).isoformat()}",
        flush=True,
    )


def fmt_elapsed(start: float | None = None) -> str:
    dt = time.monotonic() - (start if start is not None else _run_start)
    if dt >= 3600:
        return f"{dt / 3600:.2f}h"
    if dt >= 60:
        return f"{dt / 60:.1f}m"
    return f"{dt:.0f}s"


def log_step(msg: str) -> None:
    print(f"[{fmt_elapsed()}] {msg}", flush=True)
