#!/usr/bin/env python3
"""Macro-tier native HU showdown benchmarks (hu_table_build CLI end-to-end)."""

from __future__ import annotations

import argparse
import subprocess
import time
from pathlib import Path

from bench._keys import fixtures_dir
from bench._timing import N_CANONICAL_KEYS, projected_full_build_h, write_json

CLI_DIR = Path(__file__).resolve().parents[1] / "tools" / "ompeval_cli"
CLI_BIN = CLI_DIR / "build" / "Release" / "hu_table_build.exe"


def _run_cli(keys_path: Path, out_path: Path, threads: int) -> dict:
    cmd = [
        str(CLI_BIN),
        "--keys",
        str(keys_path),
        "--out",
        str(out_path),
        "--threads",
        str(threads),
        "--format",
        "raw",
    ]
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    wall = time.perf_counter() - t0
    if proc.returncode != 0:
        raise RuntimeError(
            f"hu_table_build failed rc={proc.returncode}\nstdout={proc.stdout}\nstderr={proc.stderr}"
        )
    n_keys = len(load_keys_count(keys_path))
    s_per_key = wall / n_keys
    return {
        "wall_s": wall,
        "n_keys": n_keys,
        "threads": threads,
        "s_per_key": s_per_key,
        "projected_full_build_h": projected_full_build_h(s_per_key),
        "projected_full_build_min": projected_full_build_h(s_per_key) * 60.0,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }


def load_keys_count(keys_path: Path) -> list[int]:
    import numpy as np

    if keys_path.suffix == ".npy":
        arr = np.load(keys_path)
        return list(arr)
    data = keys_path.read_bytes()
    n = len(data) // 8
    return list(range(n))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fixture", choices=["smoke", "stratified"], default="stratified")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--json-out", type=Path, default=None)
    args = p.parse_args()

    if not CLI_BIN.is_file():
        raise SystemExit(f"CLI missing: {CLI_BIN} (run tools/ompeval_cli/build.ps1)")

    keys_path = fixtures_dir() / f"{args.fixture}_keys.uint64.npy"
    out_path = CLI_DIR / "build" / f"{args.fixture}_scores.bin"
    result = _run_cli(keys_path, out_path, threads=args.threads)
    result["fixture"] = args.fixture
    result["n_canonical_keys"] = N_CANONICAL_KEYS
    result["cli"] = str(CLI_BIN)

    print(f"fixture={args.fixture} n_keys={result['n_keys']} threads={args.threads}")
    print(f"s/key={result['s_per_key']:.4f}")
    print(f"projected_full_build_min={result['projected_full_build_min']:.1f}")

    if args.json_out is not None:
        payload = dict(result)
        payload.pop("stdout", None)
        payload.pop("stderr", None)
        write_json(args.json_out, payload)


if __name__ == "__main__":
    main()
