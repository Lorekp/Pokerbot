"""Summarize fast_merge prod-shape bench results."""
import statistics
from pathlib import Path
import re

base = Path(r"./artifacts/fast_merge_prod_shape")
configs = ["serial_200", "serial_1024", "fast_1024", "fast_1024_8t"]

results = {}
for label in configs:
    walls, deals, infos = [], [], []
    for i in range(1, 6):
        p = base / f"{label}_{i}.log"
        text = p.read_text().strip()
        m = re.search(r"wall=([\d.]+)s.*?deals/s=(\d+).*?infosets=(\d+)", text)
        walls.append(float(m.group(1)))
        deals.append(int(m.group(2)))
        infos.append(int(m.group(3)))
    results[label] = (walls, deals, infos)
    print(
        f"{label:18s} walls={[round(w, 3) for w in walls]}  "
        f"mean={statistics.mean(walls):.3f}s  stddev={statistics.stdev(walls):.3f}s  "
        f"deals/s_mean={statistics.mean(deals):.0f}  infosets={infos[0]}"
    )

print()
s = statistics.mean(results["serial_1024"][0])
f = statistics.mean(results["fast_1024"][0])
g = statistics.mean(results["serial_200"][0])
e = statistics.mean(results["fast_1024_8t"][0])

print(f"ratio fast_1024   / serial_1024 = {f / s:.3f}x  ({'FASTER' if f < s else 'SLOWER'})")
print(f"ratio fast_1024_8t/ serial_1024 = {e / s:.3f}x  ({'FASTER' if e < s else 'SLOWER'})")
print(f"ratio serial_1024 / serial_200  = {s / g:.3f}x")
print(f"deals/sec serial_200  = {statistics.mean(results['serial_200'][1]):.0f}")
print(f"deals/sec serial_1024 = {statistics.mean(results['serial_1024'][1]):.0f}")
print(f"deals/sec fast_1024   = {statistics.mean(results['fast_1024'][1]):.0f}")
print(f"deals/sec fast_1024_8t= {statistics.mean(results['fast_1024_8t'][1]):.0f}")