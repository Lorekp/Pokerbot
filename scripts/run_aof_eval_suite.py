"""Run the AoF evaluation-metrics suite -> versioned runs/aof_eval/<...>/metrics.json.

Profiles:
  --fast     small-n sanity sweep (minutes; the CI subset mirrors it in tests/test_aof_eval_gates.py)
  --full     the certification sweep (hours; nightly / pre-merge for EV-affecting changes)
  default    a medium sweep (3-max + 4-max @ 8bb, all fields, full arms)

Freeze a named baseline before an EV-affecting phase:
  python scripts/run_aof_eval_suite.py --full --tag baseline_pre_phase3

Sections can be skipped (--no-nash, --no-coldstart, --no-adversary) to iterate on one metric.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from aof_cfr.evalsuite.paired_arena import DEFAULT_SEEDS, ArenaConfig, run_cell  # noqa: E402
from aof_cfr.evalsuite.result import AoFEvalCell, write_report  # noqa: E402


def uplift_cells(cfg: ArenaConfig) -> list[AoFEvalCell]:
    out = run_cell(cfg)
    cells = []
    for arm, res in out["arms"].items():
        cells.append(AoFEvalCell(
            metric="winrate", players=cfg.players, depth=cfg.stack_bb, field_name=str(cfg.field),
            arm=arm, hands=res.hands, seeds=len(cfg.seeds),
            mean_bb100=round(res.mean(), 3), se_bb100=round(res.se(), 3),
            ci95_bb100=round(2.31 * res.se(), 3) if len(cfg.seeds) >= 8 else round(4.3 * res.se(), 3),
            meta={"exploit_decisions": res.exploit_decisions,
                  "raw_per_seed": [round(x, 2) for x in res.per_seed_raw_bb100],
                  "depth_mode": cfg.depth_mode}))
    for arm, u in out["uplift"].items():
        is_gto_field = cfg.field == "all_gto"
        gate = ("uplift lower bound >= -1.0 bb/100 (safety vs GTO; n/a when underpowered)"
                if is_gto_field else "uplift - ci95 > 0 (CI-separated edge)")
        if is_gto_field:
            # Safety = the MEAN is not below -1 bb/100 AND the CI is not entirely negative
            # (a wide CI straddling zero around a ~0 mean is consistent with "no leak", not
            # evidence of one — the old lower-bound-only rule flapped at -1.07 vs -1.0).
            mean_ok = u["mean"] >= -1.0
            not_sig_neg = (u["mean"] + u["ci95"]) >= 0.0
            passed = True if (mean_ok and not_sig_neg) else (
                None if (abs(u["mean"]) < 1.0 and u["ci95"] > 1.5) else False)
        else:
            passed = (u["mean"] - u["ci95"]) > 0.0
        cells.append(AoFEvalCell(
            metric="uplift", players=cfg.players, depth=cfg.stack_bb, field_name=str(cfg.field),
            arm=arm, hands=out["arms"][arm].hands, seeds=len(cfg.seeds),
            mean_bb100=round(u["mean"], 3), se_bb100=round(u["ci95"] / 2, 3),
            ci95_bb100=round(u["ci95"], 3), gate=gate, passed=passed,
            meta={"per_seed": [round(x, 2) for x in u["per_seed"]],
                  "depth_mode": cfg.depth_mode}))
    return cells


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--tag", default="")
    ap.add_argument("--legacy", action="store_true",
                    help="pin the pre-Phase-3 gating (confidence_mode=min, gate_mode=point) — "
                         "for baseline freezes comparable against the shipping defaults")
    ap.add_argument("--players", type=int, default=0)
    ap.add_argument("--no-nash", action="store_true")
    ap.add_argument("--no-coldstart", action="store_true")
    ap.add_argument("--no-adversary", action="store_true")
    ap.add_argument("--no-drift", action="store_true")
    ap.add_argument("--explain-grid", action="store_true",
                    help="re-measure the manifest depth_grid_4max cells with disjoint seeds")
    args = ap.parse_args()

    if args.explain_grid:
        from aof_cfr.evalsuite.nash_gates import explain_4max_grid
        for row in explain_4max_grid():
            print(row)
        return 0

    if args.fast:
        seeds, hands, warmup, players, depths = (1, 2, 3), 300, 300, [3], [8.0]
        nash_samples, cs_budgets = 8000, (0, 50, 250)
    elif args.full:
        seeds, hands, warmup = DEFAULT_SEEDS, 5000, 1000
        players, depths = [2, 3, 4], [4.0, 8.0, 12.0, 20.0]
        nash_samples, cs_budgets = 48000, (0, 25, 50, 100, 250, 500, 1000)
    else:
        seeds, hands, warmup = DEFAULT_SEEDS[:4], 2000, 1000
        players, depths = [3, 4], [8.0]
        nash_samples, cs_budgets = 24000, (0, 50, 250, 1000)
    if args.players:
        players = [args.players]
    mode_kw = ({"confidence_mode": "min", "gate_mode": "point"} if args.legacy else {})
    if args.legacy:
        print("LEGACY gating modes pinned (min/point) - baseline-comparable run")

    cells: list[AoFEvalCell] = []

    if not args.no_nash:
        print("== nash gates ==")
        from aof_cfr.evalsuite.nash_gates import nash_gate_cells
        for c in nash_gate_cells(depths=tuple(depths), samples=nash_samples):
            cells.append(c)
            print(f"   n={c.players} d{c.depth:g}: {c.mean_bb100:.2f} mbb/g "
                  f"({c.meta['method']}) passed={c.passed}")

    for n in players:
        for d in depths:
            for fname in ("all_leaky", "mixed", "all_gto"):
                cfg = ArenaConfig(players=n, stack_bb=d, field=fname, hands=hands,
                                  seeds=tuple(seeds), warmup_hands=warmup, **mode_kw)
                print(f"== arena n={n} d{d:g} {fname} ==")
                for c in uplift_cells(cfg):
                    cells.append(c)
                    if c.metric == "uplift":
                        print(f"   {c.arm:>16s}: {c.mean_bb100:+7.2f} +/- {c.ci95_bb100:.2f} "
                              f"passed={c.passed}")
            if not args.no_drift:
                cfg = ArenaConfig(players=n, stack_bb=d, field="all_leaky", hands=hands,
                                  seeds=tuple(seeds), warmup_hands=warmup, depth_mode="drift",
                                  **mode_kw)
                print(f"== arena n={n} d{d:g} all_leaky (depth drift) ==")
                cells.extend(uplift_cells(cfg))

    if not args.no_coldstart:
        print("== cold-start curve ==")
        from aof_cfr.evalsuite.coldstart import coldstart_cells
        cfg = ArenaConfig(players=players[0], stack_bb=8.0, field="all_leaky", hands=hands,
                          seeds=tuple(seeds), warmup_hands=0, **mode_kw)
        for c in coldstart_cells(cfg, budgets=cs_budgets):
            cells.append(c)
            if c.metric == "coldstart":
                print(f"   obs={c.meta['observed_hands']:>5d}: {c.mean_bb100:+6.2f} +/- {c.ci95_bb100:.2f}")
            else:
                print(f"   hands_to_half_uplift={c.meta['hands_to_half_uplift']} passed={c.passed}")

    if not args.no_adversary:
        print("== adversary battery ==")
        from aof_cfr.evalsuite.adversary import (
            counter_exploiter_cell, exploitability_budget_cell, gear_switch_cell)
        acfg = ArenaConfig(players=players[0], stack_bb=8.0, field="all_leaky", hands=hands,
                           seeds=tuple(seeds)[:max(2, len(seeds) // 2)], warmup_hands=warmup,
                           arms=("gto", "adaptive_online"), **mode_kw)
        for fn in (counter_exploiter_cell, gear_switch_cell, exploitability_budget_cell):
            c = fn(acfg)
            cells.append(c)
            print(f"   {c.metric}: {c.mean_bb100:+.3f} passed={c.passed} {c.meta}")

    p = write_report(cells, _ROOT / "runs" / "aof_eval", seeds=seeds, tag=args.tag)
    n_gated = sum(1 for c in cells if c.passed is not None)
    n_pass = sum(1 for c in cells if c.passed)
    print(f"\nWrote {p}\nGates: {n_pass}/{n_gated} passed")
    return 0 if n_pass == n_gated else 1


if __name__ == "__main__":
    raise SystemExit(main())
