#!/usr/bin/env python3
import json
import sys
from collections import defaultdict
from pathlib import Path

path = Path(sys.argv[1] if len(sys.argv) > 1 else "runs/openrouter_model_battery.jsonl")
rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

by_model: dict = defaultdict(
    lambda: {"latencies": [], "costs": [], "critical_pass": 0, "total": 0, "errors": []}
)
for r in rows:
    m = r["model"]
    d = by_model[m]
    d["total"] += 1
    d["latencies"].append(r["latency_ms"])
    u = r.get("usage") or {}
    if u.get("cost") is not None:
        d["costs"].append(float(u["cost"]))
    cmp = r.get("comparison_normalized") or r.get("comparison") or {}
    if cmp.get("critical_pass"):
        d["critical_pass"] += 1
    if r.get("provider_error"):
        d["errors"].append(r["provider_error"].get("type", "provider_error"))
    elif r.get("parse_error"):
        d["errors"].append("parse_error")

lines = ["# OpenRouter vision model battery\n", "## Summary by model\n\n"]
prompt_ver = rows[0].get("prompt_version") if rows else None
if prompt_ver:
    lines.append(f"\n**Prompt version:** `{prompt_ver}`\n")

lines.append("| Model | Images | Critical pass | Avg latency (s) | Total cost (USD) | Notes |")
lines.append("|-------|--------|---------------|-----------------|------------------|-------|")
for m in sorted(by_model.keys()):
    d = by_model[m]
    lat = sum(d["latencies"]) / len(d["latencies"]) / 1000 if d["latencies"] else 0
    cost = sum(d["costs"]) if d["costs"] else 0
    notes = ", ".join(sorted(set(d["errors"]))) if d["errors"] else "ok"
    lines.append(
        f"| `{m}` | {d['total']} | {d['critical_pass']}/{d['total']} | {lat:.1f} | ${cost:.4f} | {notes} |"
    )

lines.append("\n## Per run\n")
for r in rows:
    img = Path(r["image_path"]).name
    cmp = r.get("comparison_normalized") or r.get("comparison") or {}
    cp = cmp.get("critical_pass", False)
    cost = (r.get("usage") or {}).get("cost")
    crit = len(cmp.get("critical_mismatches") or [])
    dbg = r.get("derivation_debug") or {}
    dealer_dbg = (
        f", dealer={dbg.get('dealer_physical_zone')} via {dbg.get('dealer_zone_resolution_method')}"
        if dbg
        else ""
    )
    cost_s = f"${float(cost):.4f}" if cost is not None else "n/a"
    lines.append(
        f"- **{r['model']}** / `{img}`: "
        f"{'PASS' if cp else 'FAIL'} ({crit} critical mismatches), "
        f"{r['latency_ms']/1000:.1f}s, {cost_s}{dealer_dbg}"
    )

text = "\n".join(lines) + "\n"
print(text)
Path("runs/openrouter_model_battery_report.md").write_text(text, encoding="utf-8")
