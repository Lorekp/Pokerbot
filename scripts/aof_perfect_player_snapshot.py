#!/usr/bin/env python3
"""Generate snapshot: chart-pool default vs archived tight profile."""

from __future__ import annotations

import copy
import sys
from dataclasses import dataclass
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

SPOTS: list[tuple[str, bool, str]] = [
    ("AA UTG 8bb", False, "4-max open premium"),
    ("AJo BTN 8bb", False, "4-max open steal"),
    ("J8o BTN 8bb", False, "4-max open marginal"),
    ("77 SB 8bb", False, "4-max open SB"),
    ("22 UTG 8bb", False, "4-max open trash"),
    ("KQs BTN 8bb", False, "4-max open suited"),
    ("T9o CO 8bb", False, "4-max open weak (CO→BTN)"),
    ("TT BTN 3 8bb", False, "3-way open"),
    ("AKo SB hu 8bb", False, "HU open button"),
    ("J6o SB hu 8bb", False, "HU open marginal"),
    ("72o SB hu 8bb", False, "HU open trash"),
    ("AA BB vs BTN 8bb", True, "4-max facing clean"),
    ("AJo BB vs BTN 8bb", True, "4-max facing marginal"),
    ("A5s BB vs BTN 8bb", True, "4-max facing suited"),
    ("99 SB vs BTN 8bb", True, "4-max facing medium"),
    ("KQo BB vs BTN 8bb", True, "4-max facing broadway"),
    ("72o BB vs UTG 8bb", True, "4-max facing trash"),
    ("AJo SB vs BTN behind 1", True, "4-max facing +1 behind"),
    ("Ks4d BB vs UTG,BTN 8bb", True, "Facing 2"),
    ("Ks4d BB vs UTG,BTN,SB 8bb", True, "Facing 3"),
    ("AA BB vs SB hu 8bb", True, "HU facing"),
    ("TT BB vs SB hu 8bb", True, "HU facing medium"),
    ("AJo BB vs SB hu 8bb", True, "HU facing marginal"),
]

SAMPLES = 12_000
SEED = 42


@dataclass(frozen=True)
class Row:
    spot: str
    note: str
    chart_action: str
    chart_detail: str
    chart_ev: float | None
    tight_action: str
    tight_detail: str
    tight_ev: float | None

    @property
    def changed(self) -> bool:
        return self.chart_action != self.tight_action

    @property
    def ev_delta(self) -> float | None:
        if self.chart_ev is None or self.tight_ev is None:
            return None
        return self.tight_ev - self.chart_ev


def _apply_tables(
    call_pct: dict,
    call_rng: dict,
    shove_rng: dict,
    *,
    second_jam: dict | None = None,
    third_jam: dict | None = None,
) -> None:
    import aof
    import aof_ranges as base

    base.CALL_PCT = call_pct
    base.CALL_RANGE_STR = call_rng
    base.SHOVE_RANGE_STR = shove_rng
    if second_jam is not None:
        base.SECOND_JAM_RANGE_STR = second_jam
    if third_jam is not None:
        base.THIRD_JAM_RANGE_STR = third_jam


def _install_profile(name: str, saved: dict) -> None:
    import aof_ranges_chart as chart
    import aof_ranges_tight as tight

    if name == "chart":
        _apply_tables(
            copy.deepcopy(chart.CALL_PCT),
            copy.deepcopy(chart.CALL_RANGE_STR),
            copy.deepcopy(chart.SHOVE_RANGE_STR),
            second_jam=copy.deepcopy(chart.SECOND_JAM_RANGE_STR),
            third_jam=copy.deepcopy(chart.THIRD_JAM_RANGE_STR),
        )
        return
    if name == "tight":
        _apply_tables(
            copy.deepcopy(tight.CALL_PCT),
            copy.deepcopy(tight.CALL_RANGE_STR),
            copy.deepcopy(tight.SHOVE_RANGE_STR),
        )
        return
    raise ValueError(name)


def _eval_all() -> list[Row]:
    from aof import evaluate_spot
    from aof_equity import warmup
    import aof_ranges as base

    saved = {
        "CALL_PCT": copy.deepcopy(base.CALL_PCT),
        "CALL_RANGE_STR": copy.deepcopy(base.CALL_RANGE_STR),
        "SHOVE_RANGE_STR": copy.deepcopy(base.SHOVE_RANGE_STR),
        "SECOND_JAM_RANGE_STR": copy.deepcopy(base.SECOND_JAM_RANGE_STR),
        "THIRD_JAM_RANGE_STR": copy.deepcopy(base.THIRD_JAM_RANGE_STR),
    }

    warmup()
    rows: list[Row] = []

    for spot, force_call, note in SPOTS:
        _install_profile("chart", saved)
        ca, cd = evaluate_spot(
            spot, nash=True, force_call=force_call, samples=SAMPLES, seed=SEED
        )
        _install_profile("tight", saved)
        ta, td = evaluate_spot(
            spot, nash=True, force_call=force_call, samples=SAMPLES, seed=SEED
        )
        rows.append(
            Row(
                spot,
                note,
                ca,
                cd,
                _primary_ev(cd, ca),
                ta,
                td,
                _primary_ev(td, ta),
            )
        )

    _install_profile("chart", saved)
    return rows


def _primary_ev(detail: str, action: str) -> float | None:
    key = "callEV=" if "callEV=" in detail else "shoveEV="
    if key not in detail:
        return None
    try:
        return float(detail.split(key)[1].split("bb")[0])
    except ValueError:
        return None


def _short_detail(detail: str) -> str:
    if "callEV=" in detail:
        part = detail.split("callEV=")[1]
        ev = part.split("bb")[0]
        if "eq " in detail:
            eq = detail.split("eq ")[1].split("%")[0]
            return f"eq {eq}% EV {ev}bb"
    if "shoveEV=" in detail:
        part = detail.split("shoveEV=")[1]
        ev = part.split("bb")[0]
        return f"shoveEV {ev}bb"
    return detail[:48]


def _markdown(rows: list[Row]) -> str:
    changed = [r for r in rows if r.changed]
    lines = [
        "# AoF advisor: chart pool (default) vs tight archive",
        "",
        "Snapshot generated by `scripts/aof_perfect_player_snapshot.py`.",
        "**Default** uses loose chart pool in `aof_ranges.py` / `aof_ranges_chart.py`.",
        "**Tight** column uses archived `aof_ranges_tight.py` (former default).",
        "",
        "## What changed between columns",
        "",
        "| Aspect | **Chart (default)** | **Tight archive** |",
        "|--------|---------------------|-------------------|",
        "| Call vs your open | Looser `CALL_*` + higher `CALL_PCT` | Tighter → more folds |",
        "| Shove when you face them | Wide `SHOVE_*` | Tighter shoves |",
        "| Facing 2/3 | `SECOND_JAM` / `THIRD_JAM` charts | Continue-vs-open proxies |",
        "",
        f"Settings: `{SAMPLES}` MC samples, seed `{SEED}`, 8bb effective.",
        "",
        "## Summary",
        "",
        f"- **{len(changed)} / {len(rows)}** spots flipped action (ALL IN ↔ FOLD).",
        "",
        "## Spot-by-spot",
        "",
        "| Spot | Note | Chart | Tight | Action Δ | EV Δ (tight−chart bb) |",
        "|------|------|-------|-------|----------|------------------------|",
    ]
    for r in rows:
        act = "**flip**" if r.changed else "same"
        ev_d = f"{r.ev_delta:+.2f}" if r.ev_delta is not None else "—"
        lines.append(
            f"| `{r.spot}` | {r.note} | **{r.chart_action}** | **{r.tight_action}** | {act} | {ev_d} |"
        )
    lines.extend(["", "## Detail lines", ""])
    for r in rows:
        lines.append(f"### `{r.spot}`")
        lines.append(
            f"- **Chart:** {r.chart_action} — `{_short_detail(r.chart_detail)}`"
        )
        lines.append(f"  - `{r.chart_detail}`")
        lines.append(
            f"- **Tight:** {r.tight_action} — `{_short_detail(r.tight_detail)}`"
        )
        lines.append(f"  - `{r.tight_detail}`")
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    print("Running chart default vs tight archive snapshot...")
    rows = _eval_all()
    md = _markdown(rows)
    out = _ROOT / "docs" / "aof-perfect-vs-chart-snapshot.md"
    out.write_text(md, encoding="utf-8")
    print(f"Wrote {out}")
    print(f"Actions changed: {sum(1 for r in rows if r.changed)} / {len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
