#!/usr/bin/env python3
"""Battery validation of the river thin-value expansion (pokerbot/opponent/value_sizing.py).

This overlay is the ONE with real value-cut risk (betting marginally-ahead hands into a range that may
beat us), so it is default OFF and gated on a server-HUD weak-showdown read. The standard derived-stat
battery never produces server HUD, so -- unlike value_sizing -- we must WARM-START the villain profile
with a HUD (wsd < 0.50) for the overlay to fire at all. We seed a station-shaped profile carrying that
HUD and run the same CRN-paired A/B vs the calling_station bot, whose actual wide calling range resolves
real showdowns -- so a value-cut (betting thin into hands that beat us) shows up as a NEGATIVE gain.

    thin_value_gain(bb/100) = 100 * [ uplift(RIVER_THIN_VALUE=1) - uplift(RIVER_THIN_VALUE=0) ]

PASS to enable live = gain vs calling_station with CI excluding 0 AND no loss vs lag (value-cut guard).
"""
from __future__ import annotations

import argparse
import math
import os
import statistics as st


def _seeded_station():
    """A confirmed calling-station villain profile carrying a weak-showdown server HUD (so the
    river thin-value overlay's value-cut guard is satisfied and it actually fires)."""
    from pokerbot.opponent.stats import OpponentStats
    s = OpponentStats(hands=60, vpip=46, pfr=6, aggression_bets_raises=6, aggression_calls=34,
                      cbet_faced=30, fold_to_cbet=7, saw_flop=30)
    s.hud = {"wsd": 43, "wtsd": 40, "foldToCbet": 27, "fta": 29, "vpip": 58, "pfr": 9}
    return s


def _pool(archetype: str, hands: int, seeds: list[int], thin: bool) -> list[float]:
    os.environ["POKERBOT_RIVER_THIN_VALUE"] = "1" if thin else "0"
    from pokerbot.eval.exploitation_battery import run_archetype
    out = []
    for s in seeds:
        r = run_archetype(archetype, hands=hands, seed=s, progress_every=10**9,
                          initial_stats=_seeded_station())
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
    ap.add_argument("--hands", type=int, default=1500)
    ap.add_argument("--seeds", type=int, default=24)
    ap.add_argument("--archetypes", default="calling_station,lag")
    args = ap.parse_args()
    seeds = list(range(args.seeds))
    lines: list[str] = []

    def emit(s: str) -> None:
        print(s, flush=True)
        lines.append(s)

    emit(f"river thin-value battery A/B | hands={args.hands} seeds={args.seeds} (HUD-seeded villain)\n")
    emit(f"{'archetype':<18}{'uplift OFF':>16}{'uplift ON':>16}{'thinvalue gain':>16}{'verdict':>10}")
    emit("-" * 78)
    for arch in args.archetypes.split(","):
        off = _pool(arch, args.hands, seeds, thin=False)
        on = _pool(arch, args.hands, seeds, thin=True)
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
    emit("\n(bb/100, CRN-paired vs exploit-OFF; 'thinvalue gain' = ON-minus-OFF uplift = overlay marginal EV)")
    from pathlib import Path
    Path("runs").mkdir(exist_ok=True)
    Path("runs/river_thin_value_ab.txt").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
