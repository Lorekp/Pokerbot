#!/usr/bin/env python3
"""Shadow-build parity: consecutive native rebuilds and optional full-universe NPZ diff."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

from aof_equity import warmup
from aof_equity_showdown import N_CANONICAL_KEYS, build_showdown_table_v2, load_canonical_keys


def _score2_sha256(score2: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(score2, dtype=np.uint32).tobytes()).hexdigest()


def _load_npz_score2(path: Path) -> np.ndarray:
    data = np.load(path, allow_pickle=False)
    keys = np.asarray(data["keys"], dtype=np.uint64)
    score2 = np.asarray(data["score2"], dtype=np.uint32)
    if len(keys) != N_CANONICAL_KEYS or len(score2) != N_CANONICAL_KEYS:
        raise SystemExit(
            f"{path}: expected {N_CANONICAL_KEYS} keys, got keys={len(keys)} score2={len(score2)}"
        )
    return score2


def _native_build(threads: int) -> tuple[np.ndarray, str, float]:
    keys = load_canonical_keys()
    t0 = time.perf_counter()
    _, score2, _ = build_showdown_table_v2(keys, threads=threads)
    elapsed = time.perf_counter() - t0
    score2 = np.asarray(score2, dtype=np.uint32)
    return score2, _score2_sha256(score2), elapsed


def _first_mismatch(
    a: np.ndarray, b: np.ndarray, *, label_a: str, label_b: str
) -> dict[str, object] | None:
    if a.shape != b.shape:
        return {"reason": "shape", label_a: int(a.shape[0]), label_b: int(b.shape[0])}
    diff = a != b
    if not np.any(diff):
        return None
    idx = int(np.argmax(diff))
    return {
        "index": idx,
        label_a: int(a[idx]),
        label_b: int(b[idx]),
        "n_mismatch": int(np.count_nonzero(diff)),
    }


def run_consecutive(n: int, threads: int) -> dict[str, object]:
    runs: list[dict[str, object]] = []
    score2_arrays: list[np.ndarray] = []
    for i in range(n):
        score2, sha, elapsed = _native_build(threads)
        score2_arrays.append(score2)
        runs.append(
            {
                "run": i + 1,
                "score2_sha256": sha,
                "build_seconds": round(elapsed, 3),
                "n_keys": int(len(score2)),
            }
        )
    baseline_sha = runs[0]["score2_sha256"]
    agreed = all(r["score2_sha256"] == baseline_sha for r in runs)
    mismatch: dict[str, object] | None = None
    if not agreed:
        for i in range(1, n):
            mm = _first_mismatch(
                score2_arrays[0],
                score2_arrays[i],
                label_a="baseline_score2",
                label_b=f"run{i + 1}_score2",
            )
            if mm:
                mismatch = {"run": i + 1, **mm}
                break
    return {
        "consecutive_runs": n,
        "agreed": agreed,
        "score2_sha256": baseline_sha,
        "mismatch": mismatch,
        "runs": runs,
    }


def run_against_npz(npz_path: Path, threads: int) -> dict[str, object]:
    oracle = _load_npz_score2(npz_path)
    oracle_sha = _score2_sha256(oracle)
    score2, native_sha, elapsed = _native_build(threads)
    mm = _first_mismatch(oracle, score2, label_a="oracle_score2", label_b="native_score2")
    return {
        "oracle_npz": str(npz_path),
        "oracle_score2_sha256": oracle_sha,
        "native_score2_sha256": native_sha,
        "build_seconds": round(elapsed, 3),
        "agreed": mm is None and oracle_sha == native_sha,
        "mismatch": mm,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--consecutive",
        type=int,
        default=0,
        help="Run N native full builds and require identical score2 (default: 3 if no --against)",
    )
    p.add_argument(
        "--against",
        type=Path,
        default=None,
        help="Compare native build score2 to this NPZ (full universe)",
    )
    p.add_argument("--threads", type=int, default=4)
    p.add_argument(
        "--report",
        type=Path,
        default=Path("artifacts/audit/hu_showdown_shadow_report.json"),
    )
    args = p.parse_args()

    warmup()

    report: dict[str, object] = {"threads": args.threads}
    ok = True

    n_consecutive = args.consecutive
    if n_consecutive <= 0 and args.against is None:
        n_consecutive = 3
    if n_consecutive > 0:
        consec = run_consecutive(n_consecutive, args.threads)
        report["consecutive"] = consec
        ok = ok and bool(consec["agreed"])

    if args.against is not None:
        if not args.against.is_file():
            raise SystemExit(f"Oracle NPZ not found: {args.against}")
        against = run_against_npz(args.against, args.threads)
        report["against_npz"] = against
        ok = ok and bool(against["agreed"])

    report["ok"] = ok
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
