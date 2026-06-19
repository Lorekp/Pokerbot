#!/usr/bin/env python3
"""Battery validation of the bluff-suppression overlay (pokerbot/opponent/value_sizing.py).

Same CRN-paired isolation as validate_value_sizing.py: run_archetype does an exploit-ON vs exploit-OFF
A/B and returns the uplift in bb/hand. The bluff-suppression overlay only changes the exploit-ON arm, so
its marginal EV is isolated by running the SAME battery twice -- once with POKERBOT_BLUFF_SUPPRESS on,
once off -- and differencing the uplift (the exploit-OFF arm is byte-identical):

    suppress_gain(bb/100) = 100 * [ uplift(BLUFF_SUPPRESS=1) - uplift(BLUFF_SUPPRESS=0) ]

POKERBOT_VALUE_SIZING is held at its default (on) in BOTH arms, so value-sizing cancels and only the
bluff term is measured. Target = the calling-station (where cutting -EV bluffs should help / not hurt).
Safety = lag (must not get worse -- a folding LAG is where over-suppression would cost us). Pooled.
"""
from __future__ import annotations

import argparse
import math
import os
import statistics as st


def _pool(archetype: str, hands: int, seeds: list[int], suppress: bool) -> list[float]:
    os.environ["POKERBOT_BLUFF_SUPPRESS"] = "1" if suppress else "0"
    from pokerbot.eval.exploitation_battery import run_archetype
    out = []
    for s in seeds:
        r = run_archetype(archetype, hands=hands, seed=s, progress_every=10**9)  # progress off
        out.append(r["uplift_bb_per_hand"] * 100.0)  # -> bb/100
    return out


def _ci(xs: list[float]) -> tuple[float, float]:
    if len(xs) < 2:
        return (xs[0] if xs else 0.0, 0.0)
    m = st.mean(xs)
    half = 1.96 * st.pstdev(xs) / math.sqrt(len(xs))
    return m, half


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--hands", type=int, default=1200)
    ap.add_argument("--seeds", type=int, default=24)
    ap.add_argument("--archetypes", default="calling_station,counter_exploiter,lag")
    args = ap.parse_args()
    seeds = list(range(args.seeds))
    lines: list[str] = []

    def emit(s: str) -> None:
        print(s, flush=True)
        lines.append(s)

    emit(f"bluff-suppression battery A/B | hands={args.hands} seeds={args.seeds}\n")
    emit(f"{'archetype':<18}{'uplift OFF':>16}{'uplift ON':>16}{'suppress gain':>16}{'verdict':>10}")
    emit("-" * 76)
    for arch in args.archetypes.split(","):
        off = _pool(arch, args.hands, seeds, suppress=False)
        on = _pool(arch, args.hands, seeds, suppress=True)
        m_off, h_off = _ci(off)
        m_on, h_on = _ci(on)
        delta = [a - b for a, b in zip(on, off)]  # CRN-paired: same seed, only the overlay differs
        m_d, h_d = _ci(delta)
        if m_d - h_d > 0:
            verdict = "GAIN+"
        elif m_d + h_d < 0:
            verdict = "LOSS-"
        else:
            verdict = "~flat"
        emit(f"{arch:<18}{m_off:>9.1f}+/-{h_off:<5.1f}{m_on:>9.1f}+/-{h_on:<5.1f}"
             f"{m_d:>9.1f}+/-{h_d:<5.1f}{verdict:>10}")
    emit("\n(bb/100, CRN-paired vs exploit-OFF; 'suppress gain' = ON-minus-OFF uplift = overlay marginal EV)")
    from pathlib import Path
    Path("runs").mkdir(exist_ok=True)
    Path("runs/bluff_suppress_ab.txt").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
