#!/usr/bin/env python3
"""Micro-tier postflop benchmark: subprocess spawn + IPC overhead."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from bench.postflop._timing import DEFAULT_RUST_CLI, resolve_cli, spawn_only_overhead, write_json


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cli-path", type=Path, default=None)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--json-out", type=Path, default=None)
    args = p.parse_args()

    cli = resolve_cli(args.cli_path)
    if not cli.is_file():
        raise SystemExit(f"CLI not found: {cli}")

    result = spawn_only_overhead(cli, repeats=args.repeats)
    print(f"cli={cli}")
    print(f"spawn_mean_sec={result['spawn_mean_sec']:.4f} over {args.repeats} repeats")

    if args.json_out:
        write_json(args.json_out, result)
        print(f"Wrote {args.json_out}")


if __name__ == "__main__":
    main()
