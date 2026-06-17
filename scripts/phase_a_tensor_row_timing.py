#!/usr/bin/env python3
"""Analyze per-row tensor build timing vs bucket combo profile."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from aof_cfr.buckets import COMBOS_169, LABELS_169
from aof_cfr.trio_payoff_tensor import EXACT_TRIPLE_COMBO_MAX, N_BUCKETS, row_timing_manifest_path

_ROW_DONE_RE = re.compile(
    r"bucket row (\d+) cols: done 28561 triples in ([0-9.]+)s \(([0-9.]+) triples/s\)"
)


def _combo_count(i: int, j: int, k: int) -> int:
    n = 0
    for c0, c1 in COMBOS_169[i]:
        for c2, c3 in COMBOS_169[j]:
            for c4, c5 in COMBOS_169[k]:
                if len({c0, c1, c2, c3, c4, c5}) == 6:
                    n += 1
    return n


def row_combo_profile(btn_row: int) -> dict:
    counts: list[int] = []
    exact = 0
    for j in range(N_BUCKETS):
        for k in range(N_BUCKETS):
            c = _combo_count(btn_row, j, k)
            counts.append(c)
            if 0 < c <= EXACT_TRIPLE_COMBO_MAX:
                exact += 1
    arr = counts
    mean_c = sum(arr) / len(arr)
    return {
        "row": btn_row,
        "label": LABELS_169[btn_row],
        "btn_combos": len(COMBOS_169[btn_row]),
        "mean_combo_count": round(mean_c, 2),
        "max_combo_count": max(arr),
        "exact_enum_triples": exact,
        "exact_enum_frac": round(exact / len(arr), 4),
        "zero_collision_triples": sum(1 for c in arr if c == 0),
    }


def parse_log_row_timings(log_path: Path) -> list[dict]:
    rows: dict[int, dict] = {}
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = _ROW_DONE_RE.search(line)
        if not m:
            continue
        row = int(m.group(1))
        elapsed = float(m.group(2))
        tps = float(m.group(3))
        rows[row] = {
            "row": row,
            "label": LABELS_169[row],
            "elapsed_s": round(elapsed, 2),
            "elapsed_min": round(elapsed / 60, 2),
            "triples_per_s": round(tps, 2),
            "source": str(log_path),
        }
    return [rows[r] for r in sorted(rows)]


def merge_timings(*sources: list[dict]) -> list[dict]:
    by_row: dict[int, dict] = {}
    for src in sources:
        for entry in src:
            by_row[int(entry["row"])] = entry
    return [by_row[r] for r in sorted(by_row)]


def analyze(timings: list[dict]) -> dict:
    rows_needed = sorted({int(t["row"]) for t in timings})
    profiles = {r: row_combo_profile(r) for r in rows_needed}
    merged: list[dict] = []
    elapsed = [t["elapsed_s"] for t in timings]
    for t in timings:
        row = int(t["row"])
        p = profiles[row]
        merged.append({**t, **p})

    idx_corr = _pearson([t["row"] for t in merged], elapsed)
    mean_combo_corr = _pearson([p["mean_combo_count"] for p in merged], elapsed)
    exact_frac_corr = _pearson([p["exact_enum_frac"] for p in merged], elapsed)

    fast = [t for t in merged if t["elapsed_s"] < 200]
    slow = [t for t in merged if t["elapsed_s"] > 400]
    step_rows = []
    for i in range(1, len(merged)):
        prev, cur = merged[i - 1], merged[i]
        if cur["elapsed_s"] > prev["elapsed_s"] * 1.5:
            step_rows.append({"from_row": prev["row"], "to_row": cur["row"], "ratio": round(cur["elapsed_s"] / prev["elapsed_s"], 2)})

    if idx_corr > 0.7 and mean_combo_corr < 0.5:
        verdict = "accumulation_suspect"
        verdict_note = (
            "Row time correlates with row index more than combo profile — "
            "investigate growing in-process state (not native pair-eq table)."
        )
    elif mean_combo_corr > 0.5 or exact_frac_corr > 0.4:
        verdict = "bucket_weight"
        verdict_note = (
            "Row time tracks combo/collision profile (exact-enum fraction, mean combo count), "
            "not monotonic index drift."
        )
    else:
        verdict = "mixed"
        verdict_note = "Timing step changes align with bucket profile shifts; weak index-only drift."

    return {
        "n_rows": len(merged),
        "elapsed_s": {
            "min": min(elapsed),
            "max": max(elapsed),
            "mean": round(sum(elapsed) / len(elapsed), 2),
        },
        "correlation": {
            "row_index_vs_elapsed": round(idx_corr, 3),
            "mean_combo_count_vs_elapsed": round(mean_combo_corr, 3),
            "exact_enum_frac_vs_elapsed": round(exact_frac_corr, 3),
        },
        "step_changes": step_rows[:10],
        "fast_rows_sample": fast[:5],
        "slow_rows_sample": slow[:5],
        "verdict": verdict,
        "verdict_note": verdict_note,
        "rows": merged,
    }


def _pearson(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    if n < 2:
        return 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den_x = sum((x - mx) ** 2 for x in xs) ** 0.5
    den_y = sum((y - my) ** 2 for y in ys) ** 0.5
    if den_x == 0 or den_y == 0:
        return 0.0
    return num / (den_x * den_y)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--log",
        action="append",
        default=[],
        help="build_3way_eq_tensor log(s) to parse (repeatable)",
    )
    ap.add_argument(
        "--timing-json",
        default="",
        help="row_timing_bpc*.json from checkpoint dir",
    )
    ap.add_argument(
        "--out",
        default="data/aof/cfr_base/phase_a_tensor_row_timing.json",
    )
    args = ap.parse_args()

    sources: list[list[dict]] = []
    for log in args.log:
        p = Path(log)
        if p.is_file():
            sources.append(parse_log_row_timings(p))
    if args.timing_json:
        tpath = Path(args.timing_json)
        if tpath.is_file():
            sources.append(json.loads(tpath.read_text(encoding="utf-8")))
    if not sources:
        default_log = Path("data/aof/cfr_base/logs/build_3way_eq_tensor_20260603_030826.log")
        if default_log.is_file():
            sources.append(parse_log_row_timings(default_log))

    timings = merge_timings(*sources)
    if not timings:
        raise SystemExit("No row timings found")

    report = analyze(timings)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}, indent=2))
    print(f"Wrote {out} ({len(report['rows'])} rows)")


if __name__ == "__main__":
    main()
