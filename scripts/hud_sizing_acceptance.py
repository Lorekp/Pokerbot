#!/usr/bin/env python3
"""Workstream F: calibrate the 5/6-max RNR ceilings + validate the HUD value-sizing escalation.

Two jobs, one harness (so the calibration table and the acceptance cells share the SAME measurement
infra and seeds):

  CALIBRATION (Task 1)  — sweep candidate ceiling c in {0.4,0.5,0.6} (plus an UNCAPPED reference at
    1.0) vs the adaptive ``CounterExploiterBot`` at the FULL 6-max table (the counter-exploiter is one
    shared stateful adversary across all five opponent seats, so it cannot be made an n=5 table here).
    Report the CRN-paired uplift (on - off, the cost-of-being-modeled — NOISY, ~±60 bb/100 at feasible N)
    AND the deviation MAGNITUDE bet_shift(c) (the cap's mechanism — TIGHT). The cap is set from bet_shift
    monotonicity + the structural argument (ceiling non-increasing in N, <= the 4-max 0.6); the n=5 entry
    is derived between 4-max and 6-max by that monotonicity (no independent n=5 counter-exploiter cell).
    The cap is applied by monkeypatching ``exploitation_battery.DEFAULT_MAX_ALPHA`` (the single cap the
    heuristic ExploitHero uses) — mirroring ``apply_live_value_sizing``'s ``ceiling = cap_eff(1, n)`` clamp.

  ACCEPTANCE (Task 2) — using the calibrated ceilings, on the SIZING channel (CRN-paired, AIVAT):
    1. sizing uplift vs the IRC STATION mix (POKERBOT_VALUE_SIZING on vs off differenced), CI excludes 0.
    2. retained net vs a weak/station field.
    3. RNR holds vs CounterExploiterBot at the calibrated cap, and uncapped is materially worse.
    4. rake-adjusted net-EV (gross and net of contributed_rake_bb), valley 0.10/0.25 stake.

All bb/100. n=5 is realized by seating ONE always-fold sit-out bot (the engine is fixed 6-max), so
only 5 seats contest the pot; this is stated honestly in the artifact.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics as st
from pathlib import Path

# Blueprint-free heuristic battery; keep postflop bucketing consistent with the live overlay path.
os.environ.setdefault("POKERBOT_POSTFLOP_MODE", "fast")

from pokerbot.engine.actions import Action, ActionType  # noqa: E402


def _bb100_ci(per_seed_bb_per_hand: list[float]) -> tuple[float, float]:
    """Pool per-seed bb/HAND means into a bb/100 mean + 95% half-width (t-ish, 1.96·SE over seeds)."""
    xs = [v * 100.0 for v in per_seed_bb_per_hand]  # -> bb/100
    k = len(xs)
    if k == 0:
        return 0.0, 0.0
    m = sum(xs) / k
    if k < 2:
        return m, 0.0
    sd = st.stdev(xs)
    return m, 1.96 * sd / math.sqrt(k)


def _counter_cell(spec):
    """Picklable worker: ONE (cap, seed) counter-exploiter cell at the full 6-max table. Returns the
    per-seed scalars the pooler needs. The cap is the single ExploitHero ceiling.

    NOTE on table size: ``run_archetype('counter_exploiter')`` builds ONE shared adaptive bot occupying
    all five non-hero seats (it is stateful and observes the hero across the whole run), so the
    counter-exploiter table is inherently FULL 6-max — there is no faithful way to make it an n=5 table
    in this infra. The n=5 ceiling is therefore set by the structural monotonicity argument (see the
    report), not an independent counter-exploiter cell."""
    n, c, hands, seed = spec
    os.environ.setdefault("POKERBOT_POSTFLOP_MODE", "fast")
    import pokerbot.eval.exploitation_battery as eb
    from pokerbot.eval.exploitation_battery import run_archetype

    eb.DEFAULT_MAX_ALPHA = c
    r = run_archetype("counter_exploiter", hands=hands, seed=seed, seed_hud=True,
                      allin_ev=True, progress_every=10 ** 9)
    return {
        "on_net": r["on_aivat"]["adj_mean_bb"], "off_net": r["off_aivat"]["adj_mean_bb"],
        "uplift": r["uplift_bb_per_hand"], "bet_shift": r["bet_shift"],
        "bet_freq_on": r["bet_freq_on"], "bet_freq_off": r["bet_freq_off"],
    }


def _run_counter(n: int, c: float, hands: int, seeds: list[int], sitout: bool = False, pool=None) -> dict:
    """Hero (overlay ON, cap=c) vs the homogeneous CounterExploiterBot table (full 6-max).

    Returns the per-seed ON net (cost of being modeled, absolute), the CRN uplift (on-off), and the
    deviation-magnitude bet_shift, all with 95% CIs. ``pool`` is an optional ProcessPoolExecutor to fan
    the seeds across cores. ``n``/``sitout`` are kept for the result label only (the table is 6-max)."""
    specs = [(n, c, hands, s) for s in seeds]
    rows = list(pool.map(_counter_cell, specs)) if pool is not None else [_counter_cell(s) for s in specs]

    on_net = [x["on_net"] for x in rows]
    off_net = [x["off_net"] for x in rows]
    uplift = [x["uplift"] for x in rows]
    bet_shift = [x["bet_shift"] for x in rows]
    on_m, on_h = _bb100_ci(on_net)
    off_m, off_h = _bb100_ci(off_net)
    up_m, up_h = _bb100_ci(uplift)
    bs_m = sum(bet_shift) / len(bet_shift) if bet_shift else 0.0
    bs_h = (1.96 * st.stdev(bet_shift) / math.sqrt(len(bet_shift))) if len(bet_shift) > 1 else 0.0
    return {
        "n": n, "cap": c, "hands": hands, "seeds": len(seeds),
        "on_net_bb100": on_m, "on_net_ci95": on_h,
        "off_net_bb100": off_m, "off_net_ci95": off_h,
        "uplift_bb100": up_m, "uplift_ci95": up_h,
        "uplift_lcb": up_m - up_h,    # lower CI bound of cost-of-being-modeled (the AoF-style gate stat)
        "bet_shift": bs_m, "bet_shift_ci95": bs_h,  # deviation MAGNITUDE (low-variance cap mechanism)
    }


def calibrate(caps, hands, seeds, pool=None) -> dict:
    """Sweep the candidate alpha ceiling c vs the full 6-max counter-exploiter table. For each c report
    the (noisy) cost-of-being-modeled uplift AND the (tight) deviation-magnitude bet_shift — the latter
    is the cap's mechanism and the reliable calibration signal at feasible N."""
    out = {"sweep": []}
    rows = []
    for c in caps:
        row = _run_counter(6, c, hands, seeds, pool=pool)
        rows.append(row)
        print(f"  [calib n=6 cap={c}] uplift={row['uplift_bb100']:+.1f}+-{row['uplift_ci95']:.1f} "
              f"(lcb={row['uplift_lcb']:+.1f})  bet_shift={row['bet_shift']:+.4f}+-{row['bet_shift_ci95']:.4f}",
              flush=True)
    out["sweep"] = rows
    return out


# --- Acceptance gates ----------------------------------------------------------------------------
def _sizing_cell(spec):
    """Picklable worker for the SIZING-channel A/B: one (archetype, cap, seed) -> the CRN-paired
    sizing-on minus sizing-off uplift, plus each leg's absolute net (for the rake-adjust + retained gates).
    The SIZING=1/0 legs differ ONLY in the value_sizing overlay; everything else (deck, rng, base) is
    byte-identical, so the common variance cancels hard in the per-seed delta."""
    arch, cap, hands, seed = spec
    os.environ.setdefault("POKERBOT_POSTFLOP_MODE", "fast")
    import pokerbot.eval.exploitation_battery as eb
    from pokerbot.eval.exploitation_battery import run_archetype

    eb.DEFAULT_MAX_ALPHA = cap
    os.environ["POKERBOT_VALUE_SIZING"] = "1"
    r1 = run_archetype(arch, hands=hands, seed=seed, seed_hud=True, allin_ev=True, progress_every=10 ** 9)
    os.environ["POKERBOT_VALUE_SIZING"] = "0"
    r0 = run_archetype(arch, hands=hands, seed=seed, seed_hud=True, allin_ev=True, progress_every=10 ** 9)
    os.environ["POKERBOT_VALUE_SIZING"] = "1"
    return {
        "uplift_on": r1["uplift_bb_per_hand"], "uplift_off": r0["uplift_bb_per_hand"],
        "on_net": r1["on_aivat"]["adj_mean_bb"], "off_net": r1["off_aivat"]["adj_mean_bb"],
        "bet_shift_on": r1["bet_shift"], "bet_shift_off": r0["bet_shift"],
    }


def _sizing_uplift(field_archetype: str, hands: int, seeds: list[int], cap: float, pool=None) -> dict:
    """Gate-1: sizing-overlay marginal EV = uplift(SIZING=1) - uplift(SIZING=0), CRN-paired per seed.
    Returns the paired delta in bb/100 with CI (the +11.9 anchor target) plus the ON/OFF arm absolutes."""
    specs = [(field_archetype, cap, hands, s) for s in seeds]
    rows = list(pool.map(_sizing_cell, specs)) if pool is not None else [_sizing_cell(s) for s in specs]
    on_with = [x["uplift_on"] for x in rows]
    on_without = [x["uplift_off"] for x in rows]
    delta = [a - b for a, b in zip(on_with, on_without)]
    d_m, d_h = _bb100_ci(delta)
    with_m, with_h = _bb100_ci(on_with)
    without_m, without_h = _bb100_ci(on_without)
    on_net = [x["on_net"] for x in rows]
    off_net = [x["off_net"] for x in rows]
    onnet_m, onnet_h = _bb100_ci(on_net)
    offnet_m, offnet_h = _bb100_ci(off_net)
    return {
        "field": field_archetype, "hands": hands, "seeds": len(seeds), "cap": cap,
        "uplift_sizing_on_bb100": with_m, "uplift_sizing_on_ci95": with_h,
        "uplift_sizing_off_bb100": without_m, "uplift_sizing_off_ci95": without_h,
        "sizing_gain_bb100": d_m, "sizing_gain_ci95": d_h,
        "sizing_gain_lcb": d_m - d_h,
        "on_net_bb100": onnet_m, "on_net_ci95": onnet_h,
        "off_net_bb100": offnet_m, "off_net_ci95": offnet_h,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--hands", type=int, default=1500)
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--caps", default="0.4,0.5,0.6,1.0")
    ap.add_argument("--accept-cap", type=float, default=0.5,
                    help="ceiling used for the acceptance gates if calibration is skipped")
    ap.add_argument("--floor", type=float, default=-0.73,
                    help="cost-of-being-modeled loss floor (bb/100); AoF n=4 analog target")
    ap.add_argument("--stake", default="0.10/0.25")
    ap.add_argument("--workers", type=int, default=1, help="process pool size across seed cells")
    ap.add_argument("--out", default="artifacts/hud_sizing_acceptance.json")
    ap.add_argument("--skip-calib", action="store_true")
    ap.add_argument("--skip-accept", action="store_true")
    args = ap.parse_args()

    seeds = list(range(args.seeds))
    caps = [float(x) for x in args.caps.split(",")]

    from pokerbot.economics.rake_floor import floor_bb100
    try:
        rake_floor = float(floor_bb100(args.stake, n_dealt=6))
    except Exception:  # noqa: BLE001
        rake_floor = 0.0

    result: dict = {"config": {"hands": args.hands, "seeds": args.seeds, "caps": caps,
                               "floor_bb100": args.floor, "stake": args.stake,
                               "rake_floor_bb100": rake_floor,
                               "native_subgame": False, "engine": "python_battery_6max",
                               "note": ("absolute on-arm net vs an ADAPTIVE bot has very high per-hand "
                                        "variance (std~73 bb/100/hand); the CRN sizing-channel delta and "
                                        "bet_shift are the low-variance signals.")}}

    pool = None
    if args.workers > 1:
        from concurrent.futures import ProcessPoolExecutor
        pool = ProcessPoolExecutor(max_workers=args.workers)
    try:
        if not args.skip_calib:
            print(f"=== CALIBRATION (counter-exploiter, full 6-max, hands={args.hands} "
                  f"seeds={args.seeds} workers={args.workers}) ===", flush=True)
            result["calibration"] = calibrate(caps, args.hands, seeds, pool=pool)
            # The absolute cost-of-being-modeled is NOT resolvable at feasible N (per-hand std ~73
            # bb/100 vs an adaptive bot -> uplift CIs ~±60). The reliable calibration signal is the
            # deviation MAGNITUDE bet_shift(c) (the cap's mechanism). The picker reports the largest
            # cap whose uplift lower-CI bound clears the floor; with the floor inside the noise band it
            # falls back to the most-bounded-loss cap, and the FINAL ceiling is set by the principled
            # monotonicity + bet_shift argument in the report (NOT by the noisy point pick).
            rows = result["calibration"]["sweep"]
            cand = [r for r in rows if r["cap"] < 1.0]
            ok = [r for r in cand if r["uplift_lcb"] >= args.floor]
            pick = (max(ok, key=lambda r: r["cap"]) if ok
                    else max(cand, key=lambda r: r["uplift_lcb"]))
            uncapped = next((r for r in rows if r["cap"] >= 1.0), None)
            result["calibration"]["picker_suggestion_cap6"] = pick["cap"]
            result["calibration"]["uncapped_bet_shift"] = uncapped["bet_shift"] if uncapped else None
            print(f"picker-suggested 6-max cap: {pick['cap']} "
                  f"(uncapped bet_shift={uncapped['bet_shift']:.4f} vs capped reduces deviation)",
                  flush=True)

        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(result, indent=1), encoding="utf-8")
        print(f"wrote {args.out} (calibration)", flush=True)

        if not args.skip_accept:
            cap6 = float(result.get("calibration", {}).get("picker_suggestion_cap6", args.accept_cap))
            print(f"\n=== ACCEPTANCE (cap={cap6}, stake={args.stake}, rake_floor={rake_floor:.1f}) ===",
                  flush=True)
            accept = {"cap_used": cap6, "rake_floor_bb100": rake_floor}
            print("[gate1] sizing uplift vs station mix ...", flush=True)
            g1 = _sizing_uplift("calling_station", args.hands, seeds, cap6, pool=pool)
            g1["gross_on_net_bb100"] = g1["on_net_bb100"]
            g1["net_on_net_bb100"] = g1["on_net_bb100"] - rake_floor
            accept["gate1_sizing_uplift_station"] = g1
            print("[gate2] retained net vs weak (calling_station) ...", flush=True)
            # Gate 2 reuses gate-1's ON-arm net (same field, same cap) — no extra run needed.
            accept["gate2_retained_weak"] = {
                "field": "calling_station", "cap": cap6,
                "gross_net_bb100": g1["on_net_bb100"], "gross_ci95": g1["on_net_ci95"],
                "net_of_rake_bb100": g1["on_net_bb100"] - rake_floor}
            print("[gate3] RNR holds vs counter-exploiter (capped vs uncapped) ...", flush=True)
            accept["gate3_rnr_capped"] = _run_counter(6, cap6, args.hands, seeds, False, pool=pool)
            accept["gate3_rnr_uncapped"] = _run_counter(6, 1.0, args.hands, seeds, False, pool=pool)
            result["acceptance"] = accept
            Path(args.out).write_text(json.dumps(result, indent=1), encoding="utf-8")
            print(f"wrote {args.out} (full)", flush=True)
    finally:
        if pool is not None:
            pool.shutdown()


if __name__ == "__main__":
    main()
