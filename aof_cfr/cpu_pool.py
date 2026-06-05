"""CPU pool helpers — avoid BLAS/Numba oversubscription in multiprocess builds."""

from __future__ import annotations

import os


def logical_cpus() -> int:
    return os.cpu_count() or 4


def parallel_thread_budget(*, reserve: int = 0) -> int:
    return max(1, logical_cpus() - max(0, reserve))


def describe_parallel_layout(workers: int, omp_per_worker: int) -> dict[str, int | str | bool]:
    """Summarize a multiprocess + OpenMP layout and flag oversubscription."""
    logical = logical_cpus()
    total = int(workers) * int(omp_per_worker)
    return {
        "workers": int(workers),
        "omp_per_worker": int(omp_per_worker),
        "total_compute_threads": total,
        "logical_cpus": logical,
        "oversubscribed": total > logical,
        "utilization_pct": round(100.0 * min(total, logical) / logical, 1),
    }


def assert_no_oversubscription(workers: int, omp_per_worker: int, *, label: str = "build") -> None:
    info = describe_parallel_layout(workers, omp_per_worker)
    if info["oversubscribed"]:
        raise RuntimeError(
            f"{label}: oversubscription {info['total_compute_threads']} threads "
            f"on {info['logical_cpus']} logical CPUs "
            f"(workers={workers} x omp={omp_per_worker})"
        )


def pin_compute_threads(n: int = 1) -> None:
    """Pin BLAS/OpenMP/Numba to n threads (call once per process)."""
    s = str(max(1, int(n)))
    for key in (
        "OMP_NUM_THREADS",
        "OMP_THREAD_LIMIT",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "NUMBA_NUM_THREADS",
    ):
        os.environ[key] = s


def default_pool_workers(*, reserve: int = 2, cap: int | None = None) -> int:
    """
    Worker count tuned for CPU-bound Python pools on typical desktop CPUs.

    Uses logical//2 as a physical-core heuristic (6 on a 6c/12t Ryzen) so we
    do not launch one process per hardware thread and oversubscribe BLAS/Numba.
    """
    logical = logical_cpus()
    physical = max(1, logical // 2)
    workers = max(1, min(physical, logical - max(0, reserve)))
    if cap is not None:
        workers = min(workers, cap)
    return workers


def pool_layout(
    workers: int | None = None,
    *,
    reserve: int = 2,
    native_trio: bool = False,
    openmp_threads: int | None = None,
    row_parallel: bool = False,
) -> dict[str, int]:
    """Summary for logging: chosen workers and thread pins."""
    if native_trio and row_parallel:
        w = workers if workers is not None else native_row_pool_workers(reserve=reserve)
        omp = openmp_threads if openmp_threads is not None else omp_threads_per_row_worker(w)
        return {
            "workers": w,
            "threads_per_worker": omp,
            "logical_cpus": logical_cpus(),
            "backend": "native_row_parallel",
        }
    if native_trio:
        omp = openmp_threads if openmp_threads is not None else native_openmp_threads()
        return {
            "workers": 1,
            "threads_per_worker": omp,
            "logical_cpus": logical_cpus(),
            "backend": "native_openmp",
        }
    w = workers if workers is not None else default_pool_workers(reserve=reserve)
    return {
        "workers": w,
        "threads_per_worker": 1,
        "logical_cpus": logical_cpus(),
        "backend": "python_pool",
    }


def native_openmp_threads(*, reserve: int = 1) -> int:
    """OpenMP thread count for in-process native trio builds (use all logical CPUs)."""
    return parallel_thread_budget(reserve=reserve)


def native_row_pool_workers(*, reserve: int = 0, cap: int | None = None) -> int:
    """Process count for row-parallel native tensor builds (one row per worker)."""
    logical = logical_cpus()
    physical = max(1, logical // 2)
    workers = max(1, min(physical, logical - max(0, reserve)))
    if cap is not None:
        workers = min(workers, cap)
    return workers


def omp_threads_per_row_worker(row_workers: int, *, reserve: int = 0) -> int:
    """OpenMP threads per row worker so total threads stay within logical CPU count."""
    return max(1, parallel_thread_budget(reserve=reserve) // max(1, int(row_workers)))


def tensor_build_workers(
    *,
    native_trio: bool = False,
    workers: int | None = None,
    row_parallel: bool = False,
) -> int:
    if workers is not None:
        return workers
    if native_trio and row_parallel:
        return native_row_pool_workers()
    if native_trio:
        return 1
    return default_pool_workers(reserve=2)


_STALE_BUILD_PATTERNS = (
    "build_3way_eq_tensor_npz",
    "build_pair_eq_rows_parallel",
    "phase_a_pair_eq_table_validate",
    "phase_a_gate_derisk",
    "phase_a_spot_check_run",
    "post_hoist_validate_row0",
    "meso_trio_native",
    "profile_trio_row0",
    "multiprocessing-fork",
    "multiprocessing.spawn",
    "spawn_main",
)

# Long-running pair-eq table builds are prerequisites — do not auto-kill by default.
_PAIR_EQ_BUILD_PATTERN = "build_trio_pair_eq_table"


def kill_stale_tensor_builds(*, verbose: bool = True, include_pair_eq: bool = False) -> int:
    """Stop orphan tensor-build parents/workers before starting a new build."""
    if os.name != "nt":
        return 0
    import subprocess

    self_pid = os.getpid()
    patterns = list(_STALE_BUILD_PATTERNS)
    if include_pair_eq:
        patterns.append(_PAIR_EQ_BUILD_PATTERN)
    pattern = "|".join(patterns)
    ps = (
        f"$self = {self_pid}; "
        f"$pats = '{pattern}'; "
        "$killed = @(); "
        "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
        "Where-Object { $_.ProcessId -ne $self -and $_.CommandLine -match $pats } | "
        "ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue; $killed += $_.ProcessId }; "
        "$killed -join ','"
    )
    proc = subprocess.run(
        ["powershell", "-NoProfile", "-Command", ps],
        capture_output=True,
        text=True,
        check=False,
    )
    raw = (proc.stdout or "").strip()
    if not raw:
        return 0
    ids = [int(x) for x in raw.split(",") if x.strip().isdigit()]
    if verbose and ids:
        print(f"Killed stale tensor-build processes: {ids}", flush=True)
    return len(ids)
