#!/usr/bin/env python3
import re
from datetime import datetime, timezone
from pathlib import Path
from pokerbot.solver.label_completion import CompletionLedger

ROOT = Path(".")
LOG = ROOT / "logs/solver_expansion_chain.log"
TARGET = 6000

ledger = CompletionLedger.load(ROOT / "data/solver/train_prod6000_completed.jsonl")
counts = {}
for k in ledger.completed:
    t = k.split("|", 1)[0]
    counts[t] = counts.get(t, 0) + 1

text = LOG.read_text(encoding="utf-8", errors="replace")
lines = text.splitlines()

# Post-resume rate (latest cutover restart)
marker = "prod6000 cutover restart interleaved 2026-05-29T15:44:08"
idx = text.rfind(marker)
post = lines[text[:idx].count("\n"):] if idx >= 0 else lines

ok_re = re.compile(r"OK converged=True exploit=([\d.]+)% iters=(\d+) time=([\d.]+)s")
solve_re = re.compile(r"^\[(\d+)\] solving (.+)$")
times = []
for ln in post:
    m = ok_re.search(ln)
    if m:
        times.append(float(m.group(3)))

remaining = TARGET - len(ledger.completed)
avg = sum(times) / len(times) if times else 0
rate_h = 3600 / avg if avg > 0 else 0
eta_h = remaining / rate_h if rate_h > 0 else 0

in_flight = None
for ln in reversed(lines):
    m = solve_re.match(ln.strip())
    if m and not any(ok_re.search(x) for x in lines[lines.index(ln)+1:lines.index(ln)+3] if ln in lines):
        pass
for i in range(len(lines) - 1, -1, -1):
    if lines[i].startswith("[") and " solving " in lines[i]:
        in_flight = lines[i]
        break

print(f"completed={len(ledger.completed)}")
print(f"remaining={remaining}")
print(f"pct={100*len(ledger.completed)/TARGET:.1f}%")
print(f"sb={counts.get('sb_vs_bb_flop',0)} btn={counts.get('btn_vs_bb_flop',0)} bunching={counts.get('bunching_postflop',0)}")
print(f"post_resume_ok={len(times)}")
print(f"avg_solve_sec={avg:.1f}")
print(f"rate_per_hour={rate_h:.1f}")
print(f"eta_hours={eta_h:.1f}")
print(f"eta_days={eta_h/24:.1f}")
print(f"in_flight={in_flight}")
