#!/usr/bin/env python3
"""Micro-tier HU showdown benchmarks (single eval7 / single exact key)."""

from __future__ import annotations

import argparse
from pathlib import Path

from aof_equity_grids import _card_str_to_id, _equity_two_combos_exact, _eval7
from bench._timing import bench_callable, warmup_equity, write_json


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--eval7-n", type=int, default=50_000)
    p.add_argument("--key-n", type=int, default=10)
    p.add_argument("--json-out", type=Path, default=None)
    args = p.parse_args()

    warmup_equity()

    h0 = _card_str_to_id("Ac")
    h1 = _card_str_to_id("Ad")
    v0 = _card_str_to_id("Kh")
    v1 = _card_str_to_id("Kd")

    eval7_res = bench_callable(
        lambda: _eval7(0, 1, 2, 3, 4, 5, 6),
        n_calls=args.eval7_n,
        label="eval7_fixed",
    )
    key_res = bench_callable(
        lambda: _equity_two_combos_exact(h0, h1, v0, v1),
        n_calls=args.key_n,
        label="exact_one_key",
    )

    print("NOTE: micro eval7 rate != build eval rate (see meso tier for ETA)")
    print(
        f"eval7_fixed: {eval7_res['calls_per_s']:,.0f} evals/s "
        f"({eval7_res['ns_per_call']:,.0f} ns/call)"
    )
    print(
        f"exact_one_key: {key_res['s_per_call']:.4f} s/key "
        f"({key_res['calls_per_s']:.2f} keys/s)"
    )

    payload = {
        "note": "micro eval7 rate != build eval rate; use meso s/key for ETA",
        "eval7_fixed": eval7_res,
        "exact_one_key": key_res,
    }
    if args.json_out:
        write_json(args.json_out, payload)
        print(f"Wrote {args.json_out}")


if __name__ == "__main__":
    main()
