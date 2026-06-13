"""Three-way comparison of AoF eval artifacts: baseline (legacy gating, v1 base) ->
post_phase3_5_6 (new gating, v1 base) -> post_phase4 (new gating, v2 base).

Run:  python scripts/compare_aof_eval_runs.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
TAGS = tuple(sys.argv[1:]) or ("baseline_pre_phase3", "post_phase3_5_6", "post_phase4")


def cells(tag):
    d = json.loads((_ROOT / "runs" / "aof_eval" / tag / "metrics.json").read_text(encoding="utf-8"))
    out = {}
    for c in d["cells"]:
        k = (c["metric"], c["players"], c["field_name"], c["arm"],
             c["meta"].get("depth_mode", "fixed"), c["meta"].get("observed_hands"))
        out[k] = c
    return d["header"], out


def fmt(c):
    return f"{c['mean_bb100']:+7.2f} +/- {c['ci95_bb100']:5.2f}" if c else "      n/a      "


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass
    runs = [cells(t) for t in TAGS]
    print("game_model_version:", " -> ".join(str(h.get("game_model_version")) for h, _ in runs))
    print("gates all_passed: ", " -> ".join(
        str(json.loads((_ROOT / "runs" / "aof_eval" / t / "metrics.json")
                       .read_text(encoding="utf-8"))["gates"]["all_passed"]) for t in TAGS))

    for arm in ("adaptive_online", "shift_frozen"):
        print(f"\n== uplift vs gto, arm={arm} (bb/100, baseline -> phase3/5/6 -> phase4) ==")
        keys = sorted({k for _, cc in runs for k in cc
                       if k[0] == "uplift" and k[3] == arm}, key=str)
        for k in keys:
            vals = [cc.get(k) for _, cc in runs]
            print(f"  {k[1]}-max {k[2]:>12s} {k[4]:>5s}: " + "  ->  ".join(fmt(v) for v in vals))

    print("\n== cold-start curve (adaptive uplift @ prior obs) ==")
    keys = sorted({k for _, cc in runs for k in cc if k[0] == "coldstart"},
                  key=lambda k: (k[5] is None, k[5]))
    for k in keys:
        vals = [cc.get(k) for _, cc in runs]
        print(f"  obs={str(k[5]):>5s}: " + "  ->  ".join(fmt(v) for v in vals))

    print("\n== safety battery ==")
    for metric in ("vs_counter_exploiter", "gear_switch", "exploitability_budget"):
        keys = sorted({k for _, cc in runs for k in cc if k[0] == metric}, key=str)
        for k in keys:
            vals = [cc.get(k) for _, cc in runs]
            line = "  ->  ".join((fmt(v) + f" [{'PASS' if v['passed'] else ('n/a' if v['passed'] is None else 'FAIL')}]")
                                 if v else "n/a" for v in vals)
            print(f"  {metric}: {line}")

    print("\n== nash gates (mbb/g) ==")
    keys = sorted({k for _, cc in runs for k in cc if k[0] == "nash_gap"}, key=str)
    for k in keys:
        vals = [cc.get(k) for _, cc in runs]
        print(f"  n={k[1]}: " + "  ->  ".join(f"{v['mean_bb100']:7.3f}" if v else "  n/a " for v in vals))

    print("\n== absolute winrate, adaptive arm, all_leaky fixed (gross / net of 12bb/100 fee) ==")
    for (h, cc), tag in zip(runs, TAGS):
        for k in sorted([k for k in cc if k[0] == "winrate" and k[3] == "adaptive_online"
                         and k[2] == "all_leaky" and k[4] == "fixed"], key=str):
            c = cc[k]
            net = c.get("net_bb100", c["mean_bb100"] - 12.0)
            print(f"  {tag:>20s} {k[1]}-max: gross {c['mean_bb100']:+7.2f}  net {net:+7.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
