#!/usr/bin/env python3
"""Summarize BR/L1 checkpoint trajectory from a gate run log."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("log", type=Path)
    p.add_argument("--gate-mbb", type=float, default=100.0)
    args = p.parse_args()
    text = args.log.read_text(encoding="utf-8", errors="replace")
    pat = re.compile(
        r"Checkpoint eval iter (\d+): infoset BR=([\d.]+) mbb "
        r"\(gate [\d.]+ [^)]+\) L1_vs_tabular_mean=([\d.]+)"
    )
    rows = [(int(m[0]), float(m[1]), float(m[2])) for m in pat.findall(text)]
    if not rows:
        print("No checkpoint eval lines found.")
        return
    print(f"Gate threshold: {args.gate_mbb:.0f} mbb\n")
    print("iter\tBR_mbb\tL1_mean\tBR_delta\tL1_delta")
    prev_br, prev_l1 = None, None
    first_under = None
    for it, br, l1 in rows:
        dbr = "" if prev_br is None else f"{br - prev_br:+.1f}"
        dl1 = "" if prev_l1 is None else f"{l1 - prev_l1:+.4f}"
        if first_under is None and br <= args.gate_mbb:
            first_under = it
        print(f"{it}\t{br:.2f}\t{l1:.4f}\t{dbr}\t{dl1}")
        prev_br, prev_l1 = br, l1
    last_it, last_br, last_l1 = rows[-1]
    print(f"\nFinal: iter {last_it} BR={last_br:.2f} L1={last_l1:.4f}")
    if first_under:
        print(f"First checkpoint at/below gate: iter {first_under}")
    else:
        print(f"Never reached gate ({args.gate_mbb:.0f} mbb) in logged checkpoints")
    if len(rows) >= 3:
        late = rows[-3:]
        br_slope = (late[-1][1] - late[0][1]) / (late[-1][0] - late[0][0])
        l1_slope = (late[-1][2] - late[0][2]) / (late[-1][0] - late[0][0])
        print(f"Late BR slope (last 3 ckpts): {br_slope:.3f} mbb/iter")
        print(f"Late L1 slope (last 3 ckpts): {l1_slope:.6f}/iter")


if __name__ == "__main__":
    main()
