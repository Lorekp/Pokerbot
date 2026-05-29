#!/usr/bin/env python3
"""Analyze per-model mistakes from OpenRouter benchmark JSONL."""
from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _fmt(v) -> str:
    if v is None:
        return "null"
    if isinstance(v, (list, dict)):
        s = json.dumps(v, ensure_ascii=False)
        return s if len(s) < 120 else s[:117] + "..."
    return repr(v)


def _raw_issues(raw_path: Path | None) -> list[str]:
    if raw_path is None or not raw_path.is_file():
        return []
    text = raw_path.read_text(encoding="utf-8", errors="replace")
    issues: list[str] = []
    if text.strip().startswith("```"):
        issues.append("markdown_fence_wrapped_json")
    if not text.strip().startswith("{") and "```json" not in text and "{" in text:
        issues.append("leading_prose_or_noise")
    if text.count("{") > text.count("}"):
        issues.append("truncated_json")
    if re.search(r'"rank"\s*:\s*"10"', text):
        issues.append("used_10_instead_of_T")
    if re.search(r"hole_cards.*board|community.*hero", text, re.I):
        issues.append("possible_hero_board_confusion_in_raw")
    return issues


def analyze(jsonl: Path) -> str:
    rows = [json.loads(l) for l in jsonl.read_text(encoding="utf-8").splitlines() if l.strip()]
    lines: list[str] = []

    by_model: dict[str, list] = defaultdict(list)
    for r in rows:
        by_model[r["model"]].append(r)

    for model in sorted(by_model.keys()):
        lines.append(f"\n## `{model}`\n")
        for r in by_model[model]:
            img = Path(r["image_path"]).name
            sid = "012026" if "012026" in img else "012139"
            cmp_ = r.get("comparison") or {}
            val = r.get("validation") or {}
            lines.append(f"### {sid} (`{img}`)\n")
            lines.append(
                f"- **critical_pass:** {cmp_.get('critical_pass')} | "
                f"**validator:** {'accepted' if val.get('accepted') else 'rejected'} | "
                f"**latency:** {r['latency_ms']/1000:.1f}s"
            )
            u = r.get("usage") or {}
            if u.get("cost") is not None:
                lines.append(f"- **cost:** ${float(u['cost']):.4f}")

            if r.get("provider_error"):
                lines.append(f"- **provider_error:** `{r['provider_error'].get('type')}` — {r['provider_error'].get('message', '')[:200]}")
            if r.get("parse_error"):
                lines.append(f"- **parse_error:** `{r['parse_error']}`")

            raw_p = r.get("raw_response_path")
            if raw_p:
                issues = _raw_issues(Path(raw_p))
                if issues:
                    lines.append(f"- **formatting/raw issues:** {', '.join(issues)}")

            crit = cmp_.get("critical_mismatches") or []
            if crit:
                lines.append("- **Critical mismatches:**")
                for m in crit:
                    lines.append(f"  - `{m['path']}`: expected {_fmt(m['expected'])} -> got {_fmt(m['got'])}")
            else:
                lines.append("- **Critical mismatches:** none")

            soft = cmp_.get("soft_mismatches") or []
            if soft:
                paths = [m["path"] for m in soft]
                lines.append(f"- **Soft mismatches ({len(soft)}):** {', '.join(f'`{p}`' for p in paths)}")

            for w in cmp_.get("fixture_warnings") or []:
                lines.append(f"- **fixture_warning:** {w}")
            for w in val.get("warnings") or []:
                lines.append(f"- **validator_warning:** {w}")
            for e in val.get("errors") or []:
                lines.append(f"- **validator_error:** `{e}`")

            # Parsed snapshot for common failure modes
            pj = r.get("parsed_json") or {}
            hero = pj.get("hero") or {}
            lines.append(
                f"- **parsed snapshot:** street={pj.get('street')!r} pot={((pj.get('pot') or {}).get('amount'))!r} "
                f"hero={hero.get('name')!r} cards={[c.get('code') or (c.get('rank','')+c.get('suit','')) for c in hero.get('hole_cards',[])]} "
                f"board={[c.get('code') or (c.get('rank','')+c.get('suit','')) for c in pj.get('community_cards',[])]} "
                f"actions={[a.get('type') for a in pj.get('visible_actions',[])]} "
                f"pre_action={[t.get('type') for t in pj.get('pre_action_toggles',[])]}"
            )
            lines.append("")

    return "\n".join(lines)


if __name__ == "__main__":
    p = Path(sys.argv[1] if len(sys.argv) > 1 else ROOT / "runs/openrouter_model_battery.jsonl")
    print(analyze(p))
