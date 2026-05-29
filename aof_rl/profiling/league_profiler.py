"""Global profiler hook for PPO league training."""



from __future__ import annotations



import cProfile

import io

import json

import pstats

from dataclasses import dataclass, field

from pathlib import Path

from typing import Any



from aof_rl.profiling.timer import TimingAccumulator



_ACTIVE: TimingAccumulator | None = None

_LAST: TimingAccumulator | None = None

_CPROFILE_UPDATE: int | None = None

_CPROFILE: cProfile.Profile | None = None





def set_profiler(acc: TimingAccumulator | None) -> None:

    global _ACTIVE

    _ACTIVE = acc





def get_profiler() -> TimingAccumulator | None:

    return _ACTIVE





def store_last_timings(acc: TimingAccumulator) -> None:

    global _LAST

    _LAST = acc





def get_last_timings() -> TimingAccumulator | None:

    return _LAST





def set_cprofile_update(n: int | None) -> None:

    global _CPROFILE_UPDATE, _CPROFILE

    _CPROFILE_UPDATE = n

    _CPROFILE = None





def cprofile_update_index() -> int | None:

    return _CPROFILE_UPDATE


def set_cprofile_spot(n: int | None) -> None:
    set_cprofile_update(n)


def cprofile_spot_index() -> int | None:
    return cprofile_update_index()





def start_cprofile() -> None:

    global _CPROFILE

    _CPROFILE = cProfile.Profile()

    _CPROFILE.enable()





def stop_cprofile() -> None:

    global _CPROFILE

    if _CPROFILE is not None:

        _CPROFILE.disable()





def get_cprofile_stats(*, top_n: int = 40) -> list[dict[str, Any]]:

    if _CPROFILE is None:

        return []

    ps = pstats.Stats(_CPROFILE)

    rows: list[dict[str, Any]] = []

    for (filename, lineno, funcname), (_cc, nc, tt, ct, _callers) in ps.stats.items():

        rows.append(

            {

                "function": f"{filename}:{lineno}({funcname})",

                "calls": nc,

                "cumtime": round(ct, 6),

                "tottime": round(tt, 6),

            }

        )

    rows.sort(key=lambda row: row["cumtime"], reverse=True)

    return rows[:top_n]





def write_cprofile_text(path: Path) -> None:

    if _CPROFILE is None:

        return

    stream = io.StringIO()

    ps = pstats.Stats(_CPROFILE, stream=stream)

    ps.sort_stats(pstats.SortKey.CUMULATIVE)

    ps.print_stats()

    path.parent.mkdir(parents=True, exist_ok=True)

    path.write_text(stream.getvalue(), encoding="utf-8")





@dataclass

class LeagueProfilerReport:

    timings: TimingAccumulator = field(default_factory=TimingAccumulator)

    updates: int = 0

    transitions: int = 0

    hands: int = 0

    metadata: dict[str, Any] = field(default_factory=dict)

    cprofile_top: list[dict[str, Any]] = field(default_factory=list)



    def to_dict(self) -> dict[str, Any]:

        wall = self.timings.total_wall_seconds()

        return {

            "total_wall_seconds": round(wall, 3),

            "updates": self.updates,

            "hands": self.hands,

            "transitions": self.transitions,

            "sec_per_update": round(wall / max(self.updates, 1), 3),

            "metadata": self.metadata,

            "counters": dict(self.timings.counters),

            "buckets": self.timings.sorted_report(

                wall_total=self.timings.total_wall_seconds()

            ),

            "per_update": list(self.timings.update_snapshots),

            "cprofile_top": list(self.cprofile_top),

        }



    def write_json(self, path: Path) -> None:

        path.parent.mkdir(parents=True, exist_ok=True)

        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")



    def write_markdown(self, path: Path) -> None:

        d = self.to_dict()

        lines = [

            "# AoF league GPU profile",

            "",

            f"- **Wall time**: {d['total_wall_seconds']} s",

            f"- **Updates**: {d['updates']}",

            f"- **Hands**: {d['hands']}",

            f"- **Transitions**: {d['transitions']}",

            f"- **Sec/update**: {d['sec_per_update']}",

            "",

        ]

        meta = d.get("metadata") or {}

        if meta:

            lines.append("## Run metadata")

            lines.append("")

            for k, v in meta.items():

                lines.append(f"- **{k}**: {v}")

            lines.append("")



        counters = d.get("counters") or {}

        if counters:

            lines.append("## Counters")

            lines.append("")

            for k in sorted(counters):

                lines.append(f"- `{k}`: {counters[k]}")

            lines.append("")



        per_update = d.get("per_update") or []

        if per_update:

            lines.extend(

                [

                    "## Per-update breakdown",

                    "",

                    "| Update | Stage | Wall (s) | Top buckets |",

                    "|--------|-------|----------|-------------|",

                ]

            )

            for row in per_update:

                top = ", ".join(

                    f"`{b['name']}` {b['seconds']}s"

                    for b in row.get("top_buckets", [])[:3]

                )

                lines.append(

                    f"| {row.get('update')} | {row.get('stage')} | "

                    f"{row.get('wall_seconds')} | {top or '-'} |"

                )

            lines.append("")



        cprofile_top = d.get("cprofile_top") or []

        if cprofile_top:

            lines.extend(

                [

                    "## cProfile top functions (by cumtime)",

                    "",

                    "| Rank | Function | Calls | Cumtime (s) | Tottime (s) |",

                    "|------|----------|-------|-------------|-------------|",

                ]

            )

            for rank, row in enumerate(cprofile_top, start=1):

                lines.append(

                    f"| {rank} | `{row['function']}` | {row['calls']} | "

                    f"{row['cumtime']} | {row['tottime']} |"

                )

            lines.append("")



        lines.extend(

            [

                "## Bottlenecks (sorted by wall time)",

                "",

                "| Rank | Bucket | Seconds | % | Calls | Sec/call |",

                "|------|--------|---------|---|-------|----------|",

            ]

        )

        for row in d["buckets"]:

            lines.append(

                f"| {row['rank']} | `{row['name']}` | {row['seconds']} | "

                f"{row['pct']} | {row['calls']} | {row['sec_per_call']} |"

            )

        lines.append("")

        path.parent.mkdir(parents=True, exist_ok=True)

        path.write_text("\n".join(lines), encoding="utf-8")


