#!/usr/bin/env python3
"""Audit static-profile range coverage and sensitivity linkage."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from aof_ranges import (
    CALL_RANGE_STR,
    SECOND_JAM_RANGE_STR,
    SHOVE_RANGE_STR,
    STACK_DEPTHS,
    THIRD_JAM_RANGE_STR,
    active_seats,
    lookup_range,
    range_provenance,
)
from aof_ranges import RangeIntent


def _audit_coverage() -> dict[str, Any]:
    report: dict[str, Any] = {"by_table_size": {}, "missing_keys": [], "fallback_reachable": []}
    for ts in (2, 3, 4):
        seats = active_seats(ts)
        depth_report: dict[str, Any] = {}
        for depth in STACK_DEPTHS:
            open_keys: list[str] = []
            call_keys: list[str] = []
            for pos in seats:
                try:
                    r = lookup_range(
                        RangeIntent.OPEN_SHOVE, pos, float(depth), table_size=ts
                    )
                    open_keys.append(pos)
                    if r.is_fallback:
                        report["fallback_reachable"].append(
                            {"table_size": ts, "depth": depth, "seat": pos, "intent": "open"}
                        )
                except Exception as exc:
                    report["missing_keys"].append(
                        {"table_size": ts, "depth": depth, "seat": pos, "error": str(exc)}
                    )
                for opener in seats:
                    if opener == pos:
                        continue
                    try:
                        lookup_range(
                            RangeIntent.CALL_OPEN_SHOVE,
                            pos,
                            float(depth),
                            table_size=ts,
                            first_shover=opener,
                        )
                        call_keys.append(f"{pos}|{opener}")
                    except Exception:
                        pass
            depth_report[str(depth)] = {
                "open_positions": open_keys,
                "call_pair_count": len(call_keys),
            }
        depth_report["provenance_hu_sb"] = (
            range_provenance("BB", "SB", 8.0, table_size=ts, shove=False)
            if ts == 2
            else "n/a"
        )
        report["by_table_size"][str(ts)] = depth_report
    report["shove_depth_keys"] = {str(k): list(v.keys()) for k, v in SHOVE_RANGE_STR.items()}
    report["call_depth_keys"] = {str(k): len(v) for k, v in CALL_RANGE_STR.items()}
    report["multijam"] = _audit_multijam()
    return report


def _facing_two_pairs(table_size: int) -> set[tuple[str, str]]:
    """(second, first) where two villains jam before a later hero seat."""
    order = active_seats(table_size)
    pairs: set[tuple[str, str]] = set()
    for hi in range(2, len(order)):
        for i in range(hi):
            for j in range(i + 1, hi):
                pairs.add((order[j], order[i]))
    return pairs


def _facing_three_triples(table_size: int) -> set[tuple[str, str, str]]:
    order = active_seats(table_size)
    triples: set[tuple[str, str, str]] = set()
    for hi in range(3, len(order)):
        for i in range(hi):
            for j in range(i + 1, hi):
                for k in range(j + 1, hi):
                    triples.add((order[k], order[i], order[j]))
    return triples


def _audit_multijam() -> dict[str, Any]:
    depth = 8
    missing_second: list[dict[str, Any]] = []
    missing_third: list[dict[str, Any]] = []
    sj = SECOND_JAM_RANGE_STR.get(depth, {})
    tj = THIRD_JAM_RANGE_STR.get(depth, {})
    for ts in (3, 4):
        for pair in sorted(_facing_two_pairs(ts)):
            if pair not in sj:
                missing_second.append({"table_size": ts, "pair": list(pair)})
        for triple in sorted(_facing_three_triples(ts)):
            if triple not in tj:
                missing_third.append({"table_size": ts, "triple": list(triple)})
    return {
        "second_jam_keys_8bb": len(sj),
        "third_jam_keys_8bb": len(tj),
        "missing_second_jam": missing_second,
        "missing_third_jam": missing_third,
    }


def _md_from_report(data: dict[str, Any]) -> str:
    lines = [
        "# AoF static range audit",
        "",
        f"_Generated {data.get('generated_at', '')}_",
        "",
        "This report documents **hand-tuned static profiles** — not Nash equilibrium.",
        "",
        "## Coverage by table size",
        "",
    ]
    for ts, block in data.get("by_table_size", {}).items():
        lines.append(f"### {ts}-max")
        for depth, info in block.items():
            if depth == "provenance_hu_sb":
                lines.append(f"- HU BB vs SB provenance sample: `{info}`")
                continue
            if not isinstance(info, dict):
                continue
            lines.append(
                f"- depth {depth}bb: open seats {info.get('open_positions')}, "
                f"call pairs {info.get('call_pair_count')}"
            )
        lines.append("")
    missing = data.get("missing_keys", [])
    lines.append(f"## Missing / error keys: {len(missing)}")
    if missing:
        lines.append(f"```json\n{json.dumps(missing[:20], indent=2)}\n```\n")
    flips = data.get("range_sensitivity_flips")
    if flips is not None:
        lines.append(f"## Range sensitivity action flips (from snapshot): **{flips}**")
    mj = data.get("multijam", {})
    if mj:
        lines.append("## Multi-jam chart coverage (8bb)")
        lines.append(
            f"- SECOND_JAM keys: {mj.get('second_jam_keys_8bb', 0)}; "
            f"missing: {len(mj.get('missing_second_jam', []))}"
        )
        lines.append(
            f"- THIRD_JAM keys: {mj.get('third_jam_keys_8bb', 0)}; "
            f"missing: {len(mj.get('missing_third_jam', []))}"
        )
        if mj.get("missing_second_jam") or mj.get("missing_third_jam"):
            lines.append("```json")
            lines.append(json.dumps(mj, indent=2)[:2000])
            lines.append("```")
    lines.append("")
    lines.append("## Warning")
    lines.append("")
    lines.append("3/4-max ranges remain static-profile unless replaced by validated solver data.")
    return "\n".join(lines)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--sensitivity-json",
        type=Path,
        default=Path("artifacts/aof_range_sensitivity_snapshot.json"),
    )
    p.add_argument(
        "-o",
        "--json-out",
        type=Path,
        default=Path("artifacts/aof_static_range_audit.json"),
    )
    p.add_argument(
        "--md-out",
        type=Path,
        default=Path("artifacts/aof_static_range_audit.md"),
    )
    args = p.parse_args()

    payload: dict[str, Any] = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "static_profile_only": True,
        "not_solver_backed_34max": True,
    }
    payload.update(_audit_coverage())
    if args.sensitivity_json.is_file():
        sens = json.loads(args.sensitivity_json.read_text(encoding="utf-8"))
        spots = sens.get("spots", [])
        payload["range_sensitivity_flips"] = sum(1 for s in spots if s.get("action_flip"))
        payload["largest_sensitivity_deltas"] = sorted(
            spots,
            key=lambda s: abs(float(s.get("delta_bb", 0))),
            reverse=True,
        )[:5]

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    args.md_out.write_text(_md_from_report(payload), encoding="utf-8")
    print(f"Wrote {args.json_out} and {args.md_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
