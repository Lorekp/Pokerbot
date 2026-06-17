"""Persistent JSONL worker process for postflop_solver_cli."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO

from pokerbot.solver.postflop_solver_client import (
    DEFAULT_CLI_PATH,
    SolverSolveResult,
    parse_solve_response,
)

PROTOCOL_VERSION = 1
REQUEST_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class WorkerError(Exception):
    code: str
    message: str
    request_id: str | None = None

    def __str__(self) -> str:
        rid = self.request_id or "unknown"
        return f"worker error [{rid}] {self.code}: {self.message}"


class SolverTimeoutError(WorkerError):
    """Raised when the worker subprocess does not respond within timeout_sec."""


def _readline_with_timeout(stream: TextIO, timeout_sec: float) -> str | None:
    """Read one line from stream, or None on timeout. Cross-platform."""
    if timeout_sec <= 0:
        return stream.readline()

    if sys.platform != "win32":
        import select

        ready, _, _ = select.select([stream], [], [], timeout_sec)
        if not ready:
            return None
        return stream.readline()

    result: list[str | None] = [None]
    error: list[BaseException | None] = [None]

    def _read() -> None:
        try:
            result[0] = stream.readline()
        except OSError as exc:
            error[0] = exc

    thread = threading.Thread(target=_read, daemon=True)
    thread.start()
    thread.join(timeout_sec)
    if thread.is_alive():
        return None
    if error[0] is not None:
        raise error[0]
    return result[0]


class PersistentSolverWorker:
    """Long-lived postflop_solver_cli worker (--protocol jsonl)."""

    def __init__(
        self,
        *,
        cli_path: Path | None = None,
        solver_threads_per_worker: int | None = None,
        disable_worker_cache: bool = False,
        worker_cache_size: int | None = None,
    ) -> None:
        self._cli_path = cli_path or DEFAULT_CLI_PATH
        self._solver_threads_per_worker = solver_threads_per_worker
        self._disable_worker_cache = disable_worker_cache
        self._worker_cache_size = worker_cache_size
        if not self._cli_path.is_file():
            raise FileNotFoundError(
                f"postflop_solver_cli not found at {self._cli_path}. "
                "Build with: cd external/postflop_solver_cli && cargo build --release"
            )
        self._proc: subprocess.Popen[str] | None = None
        self._lock = threading.Lock()
        self._closed = False
        self._start_process()

    def _start_process(self) -> None:
        env = os.environ.copy()
        if self._solver_threads_per_worker is not None:
            env["RAYON_NUM_THREADS"] = str(self._solver_threads_per_worker)
        cmd = [str(self._cli_path), "worker", "--protocol", "jsonl"]
        if self._disable_worker_cache:
            cmd.append("--no-cache")
        elif self._worker_cache_size is not None:
            cmd.extend(["--cache-size", str(self._worker_cache_size)])
        self._proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            env=env,
        )

    def _kill_process(self) -> None:
        proc = self._proc
        if proc is None:
            return
        if proc.poll() is None:
            proc.kill()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass

    def _respawn(self) -> None:
        self._kill_process()
        self._start_process()

    def solve(
        self,
        request: dict[str, Any],
        *,
        request_id: str,
        timeout_sec: float = 3600.0,
    ) -> SolverSolveResult:
        if self._closed:
            raise RuntimeError("worker already closed")
        if self._proc is None or self._proc.stdin is None or self._proc.stdout is None:
            raise RuntimeError("worker process missing stdio pipes")

        envelope = {
            "protocol_version": PROTOCOL_VERSION,
            "request_schema_version": REQUEST_SCHEMA_VERSION,
            "request_id": request_id,
            "request": request,
        }
        line = json.dumps(envelope, separators=(",", ":"), sort_keys=True)

        with self._lock:
            try:
                self._proc.stdin.write(line + "\n")
                self._proc.stdin.flush()
            except OSError as exc:
                raise RuntimeError(f"worker stdin write failed: {exc}") from exc

            try:
                resp_line = _readline_with_timeout(self._proc.stdout, timeout_sec)
            except OSError as exc:
                raise RuntimeError(f"worker stdout read failed: {exc}") from exc

            if resp_line is None:
                self._respawn()
                raise SolverTimeoutError(
                    "timeout",
                    f"no worker response within {timeout_sec}s",
                    request_id=request_id,
                )

        if not resp_line:
            code = self._proc.poll()
            stderr_tail = ""
            if self._proc.stderr is not None:
                stderr_tail = self._proc.stderr.read(500)
            self._respawn()
            raise RuntimeError(
                f"worker exited unexpectedly (code={code}); stderr={stderr_tail[:500]!r}"
            )

        try:
            resp = json.loads(resp_line.strip())
        except json.JSONDecodeError as exc:
            self._respawn()
            raise RuntimeError(f"invalid worker json: {exc}; line={resp_line[:200]!r}") from exc

        if resp.get("protocol_version") != PROTOCOL_VERSION:
            raise WorkerError(
                "protocol_version",
                f"unexpected protocol_version {resp.get('protocol_version')}",
                request_id=str(resp.get("request_id", request_id)),
            )

        if not resp.get("ok"):
            err = resp.get("error") or {}
            raise WorkerError(
                str(err.get("code", "worker_failure")),
                str(err.get("message", "unknown worker error")),
                request_id=str(resp.get("request_id", request_id)),
            )

        result = resp.get("result")
        if not isinstance(result, dict):
            raise WorkerError(
                "missing_result",
                "worker ok=true but result missing",
                request_id=str(resp.get("request_id", request_id)),
            )
        return parse_solve_response(result)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        proc = self._proc
        if proc is not None and proc.stdin is not None:
            try:
                proc.stdin.write("shutdown\n")
                proc.stdin.flush()
            except OSError:
                pass
        if proc is not None:
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self._kill_process()

    def __enter__(self) -> PersistentSolverWorker:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


def default_worker(cli_path: Path | None = None) -> PersistentSolverWorker:
    return PersistentSolverWorker(cli_path=cli_path)
