#!/usr/bin/env python3
"""Plot native thread-scaling curve from meso_native scaling JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--json-in",
        type=Path,
        default=Path("docs/native_rewrite/baseline/meso_native_scaling_run.json"),
    )
    p.add_argument(
        "--png-out",
        type=Path,
        default=Path("docs/native_rewrite/baseline/native_scaling.png"),
    )
    args = p.parse_args()

    payload = json.loads(args.json_in.read_text(encoding="utf-8"))
    rows = payload["scaling"]
    threads = [row["threads"] for row in rows]
    s_per_key = [row["s_per_key"] for row in rows]
    mins = [row["projected_full_build_min"] for row in rows]

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping plot")
        return

    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax1.plot(threads, s_per_key, marker="o", label="s/key")
    ax1.set_xlabel("threads")
    ax1.set_ylabel("seconds / key")
    ax1.set_title("Native build_scores scaling (stratified 4096 keys)")
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(threads, mins, marker="s", color="tab:orange", label="projected full build (min)")
    ax2.set_ylabel("projected full build (min)")

    args.png_out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.png_out, dpi=150)
    print(f"wrote {args.png_out}")


if __name__ == "__main__":
    main()
