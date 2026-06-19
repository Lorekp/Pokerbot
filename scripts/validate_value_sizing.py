#!/usr/bin/env python3
"""Battery validation of the HUD-driven value-SIZING overlay (pokerbot/opponent/value_sizing.py).

run_archetype already does a CRN-paired exploit-ON vs exploit-OFF A/B and returns the uplift in
bb/hand. The value-sizing overlay ONLY changes the exploit-ON arm, so its marginal EV is isolated by
running the SAME battery twice — once with POKERBOT_VALUE_SIZING on, once off — and differencing the
uplift (the exploit-OFF arm is byte-identical, so the difference is purely the sizing overlay):

    sizing_gain(bb/100) = 100 · [ uplift(SIZING=1) − uplift(SIZING=0) ]

Target = the calling-station (where over-betting value should print). Safety = counter_exploiter /
lag (the overlay must NOT make us worse there — value-cut / counter-exploit guard). Pooled over seeds.
"""
from __future__ import annotations

import argparse
import math
import os
import statistics as st


def _run_one(arch: str, hands: int, seed: int, sizing: bool) -> float:
    """One CRN A/B leg: uplift (bb/100) for one (archetype, seed, sizing) cell. POKERBOT_VALUE_SIZING is
    read live (value_sizing.py), so setting it per call is safe even inside a reused worker process.
    seed_hud=True: give the villain the server HUD the live coach would have, so the HUD-gated value-sizing
    overlay FIRES in the sim exactly as it does live (without it the WSD value-cut guard fails closed and
    the overlay is a no-op -> the A/B would read ~0 for the wrong reason)."""
    os.environ["POKERBOT_VALUE_SIZING"] = "1" if sizing else "0"
    from pokerbot.eval.exploitation_battery import run_archetype
    r = run_archetype(arch, hands=hands, seed=seed, progress_every=10**9, seed_hud=True)
    return r["uplift_bb_per_hand"] * 100.0  # -> bb/100


def _task(spec: tuple[str, int, int, bool]) -> tuple[str, int, bool, float]:
    """Picklable ProcessPool worker: returns (archetype, seed, sizing, uplift_bb100)."""
    arch, hands, seed, sizing = spec
    return arch, seed, sizing, _run_one(arch, hands, seed, sizing)


def _pool(archetype: str, hands: int, seeds: list[int], sizing: bool) -> list[float]:
    return [_run_one(archetype, hands, s, sizing) for s in seeds]


def _pool_parallel(archetypes: list[str], hands: int, seeds: list[int],
                   workers: int) -> dict[tuple[str, bool], list[float]]:
    """Run every (archetype, seed, sizing) cell across `workers` processes. Returns seed-ordered uplift
    lists keyed by (archetype, sizing) so the caller can form CRN-paired ON-minus-OFF deltas."""
    from concurrent.futures import ProcessPoolExecutor
    specs = [(a, hands, s, sz) for a in archetypes for s in seeds for sz in (False, True)]
    acc: dict[tuple[str, bool], dict[int, float]] = {(a, sz): {} for a in archetypes for sz in (False, True)}
    done = 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for arch, seed, sizing, val in ex.map(_task, specs):
            acc[(arch, sizing)][seed] = val
            done += 1
            print(f"  [{done}/{len(specs)}] {arch} seed={seed} sizing={int(sizing)} -> {val:8.2f} bb/100",
                  flush=True)
    return {k: [v[s] for s in seeds] for k, v in acc.items()}


def _ci(xs: list[float]) -> tuple[float, float]:
    if len(xs) < 2:
        return (xs[0] if xs else 0.0, 0.0)
    m = st.mean(xs)
    half = 1.96 * st.pstdev(xs) / math.sqrt(len(xs))
    return m, half


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--hands", type=int, default=1000)
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--archetypes", default="calling_station,counter_exploiter,lag")
    ap.add_argument("--workers", type=int, default=1,
                    help="parallel processes across (archetype, seed, sizing) cells (1 = serial)")
    args = ap.parse_args()
    seeds = list(range(args.seeds))
    archetypes = args.archetypes.split(",")
    lines: list[str] = []

    def emit(s: str) -> None:
        print(s, flush=True)
        lines.append(s)

    grid: dict[tuple[str, bool], list[float]] = {}
    if args.workers > 1:
        emit(f"(running {len(archetypes) * args.seeds * 2} cells across {args.workers} workers)")
        grid = _pool_parallel(archetypes, args.hands, seeds, args.workers)

    emit(f"value-sizing battery A/B | hands={args.hands} seeds={args.seeds}\n")
    emit(f"{'archetype':<18}{'uplift OFF':>16}{'uplift ON':>16}{'sizing gain':>16}{'verdict':>10}")
    emit("-" * 76)
    for arch in archetypes:
        off = grid[(arch, False)] if grid else _pool(arch, args.hands, seeds, sizing=False)
        on = grid[(arch, True)] if grid else _pool(arch, args.hands, seeds, sizing=True)
        m_off, h_off = _ci(off)
        m_on, h_on = _ci(on)
        delta = [a - b for a, b in zip(on, off)]  # CRN-paired: same seed, only the overlay differs
        m_d, h_d = _ci(delta)
        # verdict: target wants gain>0; safety archetypes want no significant loss
        if m_d - h_d > 0:
            verdict = "GAIN+"
        elif m_d + h_d < 0:
            verdict = "LOSS-"
        else:
            verdict = "~flat"
        emit(f"{arch:<18}{m_off:>9.1f}+/-{h_off:<5.1f}{m_on:>9.1f}+/-{h_on:<5.1f}"
             f"{m_d:>9.1f}+/-{h_d:<5.1f}{verdict:>10}")
    emit("\n(bb/100, CRN-paired vs exploit-OFF; 'sizing gain' = ON-minus-OFF uplift = overlay marginal EV)")
    from pathlib import Path
    Path("runs").mkdir(exist_ok=True)
    Path("runs/value_sizing_ab.txt").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
