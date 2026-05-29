#!/usr/bin/env python3
"""Summarize AoF validation JSON artifacts into a human-readable report."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

DEFAULT_ARTIFACTS: dict[str, Path] = {
    "multiway_equity": Path("artifacts/aof_multiway_equity_snapshot.json"),
    "open_multiway": Path("artifacts/aof_open_multiway_snapshot.json"),
    "call_behind": Path("artifacts/aof_call_behind_snapshot.json"),
    "range_sensitivity": Path("artifacts/aof_range_sensitivity_snapshot.json"),
    "teacher_mismatch_v1": Path("artifacts/aof_rl_teacher_mismatch_v1.json"),
    "teacher_mismatch_v2": Path("artifacts/aof_rl_teacher_mismatch_v2.json"),
    "policy_comparison": Path("artifacts/aof_policy_version_comparison.json"),
    "eval_v1": Path("artifacts/eval_v1.json"),
    "eval_v2": Path("artifacts/eval_v2.json"),
}


def _load(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _top_n(rows: list[dict], key: str, n: int = 5) -> list[dict]:
    return sorted(rows, key=lambda r: abs(float(r.get(key, 0))), reverse=True)[:n]


def _md_table(rows: list[dict], cols: tuple[str, ...]) -> str:
    if not rows:
        return "_none_\n"
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for r in rows:
        lines.append("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |")
    return "\n".join(lines) + "\n"


def build_summary(paths: dict[str, Path]) -> str:
    parts: list[str] = [
        "# AoF validation summary",
        "",
        "_Auto-generated. Does not claim equilibrium or Nash quality._",
        "",
        "## Reliability note",
        "",
        "- **HU solver-backed** spots use `solver_hu_dcfr` ranges when present.",
        "- **Static-profile** 3/4-max ranges are hand-tuned, not solver-backed.",
        "- **Approximation** paths (call-behind, Facing 2 conditional/fallback) are labeled explicitly.",
        "",
    ]

    mw = _load(paths["multiway_equity"])
    if mw:
        spots = mw.get("spots", [])
        top = _top_n(spots, "abs_delta")
        parts += ["## Largest exact vs MC equity deltas", ""]
        parts.append(_md_table(top, ("name", "exact_equity", "mc_equity", "abs_delta", "equity_method")))
        parts.append("")

    om = _load(paths["open_multiway"])
    if om:
        spots = om.get("spots", [])
        top = _top_n(spots, "delta_bb")
        flips = [s for s in spots if s.get("action_flip")]
        parts += ["## Open multiway: largest shortcut vs current EV delta", ""]
        parts.append(_md_table(top, ("spot", "new_action", "delta_bb", "action_flip")))
        parts.append(f"\nAction flips: **{len(flips)}** / {len(spots)}\n")

    cb = _load(paths["call_behind"])
    if cb:
        spots = cb.get("spots", [])
        top = _top_n(spots, "delta_bb")
        parts += ["## Call-with-behind: largest legacy vs current delta", ""]
        parts.append(_md_table(top, ("spot", "delta_bb")))
        parts.append("")

    rs = _load(paths["range_sensitivity"])
    if rs:
        spots = rs.get("spots", [])
        flips = [s for s in spots if s.get("action_flip")]
        top = _top_n(spots, "delta_bb")
        parts += ["## Range sensitivity action flips", ""]
        parts.append(f"Flips: **{len(flips)}** / {len(spots)}\n")
        parts.append(_md_table(top, ("spot", "default_action", "chart_action", "delta_bb")))
        parts.append("")

    for label, key in (("v1", "teacher_mismatch_v1"), ("v2", "teacher_mismatch_v2")):
        rep = _load(paths[key])
        if not rep:
            continue
        parts += [f"## Teacher mismatch report ({label})", ""]
        parts.append(
            f"- obs_version={rep.get('obs_version')} teacher_version={rep.get('teacher_version')}"
        )
        parts.append(f"- hands={rep.get('num_hands')}")
        sk = rep.get("spot_kind_counts", {})
        tp = rep.get("teacher_path_counts", {})
        parts.append(f"- spot kinds: `{json.dumps(sk, sort_keys=True)}`")
        parts.append(f"- teacher paths: `{json.dumps(tp, sort_keys=True)}`")
        f2 = tp.get("static_profile_facing_two_missing", 0)
        f2v2 = sum(
            tp.get(k, 0)
            for k in (
                "static_profile_facing_two_conditional",
                "static_profile_facing_two_fallback",
                "static_profile_facing_two",
            )
        )
        if label == "v1" and f2:
            parts.append(
                f"- **v1 Facing 2 missing path count**: {f2} (expected for two-shover-before-hero)"
            )
        if label == "v2" and f2v2:
            parts.append(f"- **v2 Facing 2 advisor paths**: {f2v2}")
        parts.append("")

    cmp = _load(paths["policy_comparison"])
    if cmp:
        parts += ["## Policy version comparison (smoke/benchmark)", ""]
        for pol in cmp.get("policies", []):
            parts.append(
                f"- **{pol.get('name')}**: bb/100={pol.get('bb_per_100'):.2f}, "
                f"teacher agree={pol.get('action_agreement_with_teacher'):.3f}, "
                f"obs={pol.get('obs_version')} teacher={pol.get('teacher_version')}"
            )
            f2 = pol.get("facing_two_decisions", {})
            cb2 = pol.get("call_behind_decisions", {})
            parts.append(
                f"  - Facing 2: n={f2.get('count')} call_rate={f2.get('call_rate')}"
            )
            parts.append(
                f"  - call-behind: n={cb2.get('count')} call_rate={cb2.get('call_rate')}"
            )
        parts.append("")

    for label, key in (("v1 eval", "eval_v1"), ("v2 eval", "eval_v2")):
        ev = _load(paths[key])
        if ev:
            meta = ev.get("checkpoint_meta", {})
            parts.append(
                f"- **{label}**: bb/100={ev.get('bb_per_100'):.2f} "
                f"(obs={meta.get('obs_version')} teacher={meta.get('teacher_version')})"
            )
    parts.append("")
    parts += [
        "## Static-profile warning",
        "",
        "3/4-max open/call ranges remain **hand-tuned static profiles**. "
        "Use `scripts/aof_static_range_audit.py` for coverage gaps.",
        "",
    ]
    return "\n".join(parts)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("artifacts/aof_validation_summary.md"),
    )
    p.add_argument("--append", action="store_true", help="Append to existing summary")
    args = p.parse_args()

    text = build_summary(DEFAULT_ARTIFACTS)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.append and args.output.is_file():
        text = args.output.read_text(encoding="utf-8") + "\n---\n\n" + text
    args.output.write_text(text, encoding="utf-8")
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
