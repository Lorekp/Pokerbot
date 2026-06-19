"""Pool independent HU-LBR runs (each its own seed) into one tight exploitability estimate.

Each run JSON has {hands, ev_bb_per_hand, stderr_bb_per_hand, mbb_per_g}. Runs are independent
(distinct seeds), so the pooled mean is hand-weighted and the pooled variance sums per-run sum-variances:
  pooled_mean = sum(mean_i * h_i) / sum(h_i)
  Var(pooled_mean) = sum(h_i^2 * se_i^2) / (sum h_i)^2     (se_i = std_i/sqrt(h_i))
"""

from __future__ import annotations

import json
import math
import sys

rows = []
tot_h = 0
w_mean = 0.0
var_num = 0.0
for f in sys.argv[1:]:
    try:
        d = json.load(open(f))
    except Exception as e:  # noqa: BLE001
        print(f"  SKIP {f}: {e}")
        continue
    h = int(d["hands"])
    mean = float(d["ev_bb_per_hand"])
    se = float(d["stderr_bb_per_hand"])
    tot_h += h
    w_mean += mean * h
    var_num += (h * se) ** 2
    rows.append((f.split("/")[-1], d["mbb_per_g"], 1.96 * se * 1000.0, h))

if not tot_h:
    print("no runs to pool")
    raise SystemExit(1)

mean = w_mean / tot_h
se_pooled = math.sqrt(var_num) / tot_h
mbb = mean * 1000.0
ci = 1.96 * se_pooled * 1000.0

print("  per-run:")
for name, m, c, h in sorted(rows):
    print(f"    {name:28s} {m:+8.1f} +/- {c:7.1f} mbb/g  ({h} hands)")
print(f"\n  POOLED over {len(rows)} runs, {tot_h} hands:")
print(f"    mbb/g  = {mbb:+.1f}  +/- {ci:.1f} (95% CI)")
print(f"    bb/100 = {mean * 100:+.3f}")
print("  reference: always-fold +750; ACPC 2014-16 +400..1000; DeepStack ~0. lower = less exploitable.")
