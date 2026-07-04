"""Resume the killed twosided_ext_848484 sharded run (CRN-exact).

Each shard restarts at its next unplayed deck index with the same stride=4,
writing to shard{i}b.json. Then merges parts a+b per shard and interleaves
round-robin, matching h2h_sharded.py's merge exactly.
"""
import json
import math
import os
import subprocess
import sys
import time

ROOT = r"."
ART = f"{ROOT}/artifacts/battery_r4"
BASE = f"{ART}/twosided_ext_848484"
STRIDE = 4
PER_SHARD = 750  # 3000 / 4

ARM_A = '{"two_sided": false}'
ARM_B = "{}"
SEED = "848484"

env = dict(os.environ)
env["RAYON_NUM_THREADS"] = "3"
env["POKERBOT_NATIVE_SUBGAME"] = "1"

t0 = time.perf_counter()
procs = []
already = 0
for i in range(STRIDE):
    done = len(json.load(open(f"{BASE}.shard{i}.json"))["per_deck_nets"])
    already += done
    remaining = PER_SHARD - done
    start = i + STRIDE * done
    so = f"{BASE}.shard{i}b.json"
    if remaining <= 0:
        print(f"shard {i}: complete ({done})", flush=True)
        continue
    log = open(f"{BASE}.shard{i}b.log", "w")
    cmd = [sys.executable, "-u", f"{ROOT}/scripts/engine_ab_h2h.py",
           "--arm-a", ARM_A, "--arm-b", ARM_B, "--seed", SEED,
           "--settle-samples", "200", "--label", "twosided_ext_848484_resume",
           "--deck-start", str(start), "--deck-stride", str(STRIDE),
           "--decks", str(remaining), "--out", so]
    procs.append((subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT,
                                   env=env, cwd=ROOT), log, so, i))
    print(f"shard {i}: resume deck {start}, stride {STRIDE}, {remaining} decks -> {so}", flush=True)

total_target = 3000
while procs and any(p.poll() is None for p, _, _, _ in procs):
    time.sleep(20)
    resumed = 0
    for _, _, so, _ in procs:
        try:
            resumed += len(json.load(open(so)).get("per_deck_nets", []))
        except Exception:
            pass
    tot = already + resumed
    el = (time.perf_counter() - t0) / 60
    rate = resumed / max(el, 0.01)
    eta = (total_target - tot) / max(rate, 1.0)
    print(f"  progress {tot}/{total_target} decks | {el:.1f}min ETA {eta:.1f}min", flush=True)

fails = []
for p, log, so, i in procs:
    p.wait()
    log.close()
    if p.returncode != 0:
        fails.append(so)
if fails:
    raise SystemExit(f"resume shard(s) FAILED: {fails}")

# Merge: per-shard concat (part a then part b = deck order within shard),
# then round-robin interleave across shards.
shard_lists, flopdecks, a_total, b_total = [], 0, 0.0, 0.0
for i in range(STRIDE):
    da = json.load(open(f"{BASE}.shard{i}.json"))
    nets = list(da["per_deck_nets"])
    flopdecks += int(da.get("flopdecks", 0))
    a_total += float(da.get("a_total", 0.0))
    b_total += float(da.get("b_total", 0.0))
    pb = f"{BASE}.shard{i}b.json"
    if os.path.exists(pb):
        db = json.load(open(pb))
        nets += list(db["per_deck_nets"])
        flopdecks += int(db.get("flopdecks", 0))
        a_total += float(db.get("a_total", 0.0))
        b_total += float(db.get("b_total", 0.0))
    assert len(nets) == PER_SHARD, f"shard {i}: {len(nets)} != {PER_SHARD}"
    shard_lists.append(nets)

per_deck = []
for j in range(PER_SHARD):
    for sl in shard_lists:
        per_deck.append(sl[j])

m = sum(per_deck) / len(per_deck)
se = (sum((x - m) ** 2 for x in per_deck) / (len(per_deck) - 1)) ** 0.5 / math.sqrt(len(per_deck))
first = json.load(open(f"{BASE}.shard0.json"))
merged = {
    "label": "twosided_ext_848484",
    "arm_a": first.get("arm_a"),
    "arm_b": first.get("arm_b"),
    "decks": len(per_deck),
    "per_deck_nets": per_deck,
    "flopdecks": flopdecks,
    "a_total": a_total,
    "b_total": b_total,
    "a_minus_b_per_deck": m,
    "ci95": [m - 1.96 * se, m + 1.96 * se],
    "wall_s": time.perf_counter() - t0,
    "shards": STRIDE,
    "resumed": True,
}
json.dump(merged, open(f"{BASE}.json", "w"), indent=1)
print(f"MERGED {len(per_deck)} decks: mean/deck {m:+.2f}  CI95 [{m-1.96*se:+.1f}, {m+1.96*se:+.1f}]  "
      f"flopdecks {flopdecks} -> {BASE}.json", flush=True)
