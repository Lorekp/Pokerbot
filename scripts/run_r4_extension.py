#!/usr/bin/env python3
"""Driver: run R4 battery extension (seed 848484, 3000 decks each).

Step 1: Effectiveness re-screens (40 decks, seed 848484) — confirm knobs are LIVE at new seed.
Step 2: Main run 1 — leaf_biased_continuations=True vs {} — 3000 decks, seed 848484.
Step 3: Main run 2 — two_sided=False vs {} — 3000 decks, seed 848484.

All runs: POKERBOT_NATIVE_SUBGAME=1, EMD env set by engine_ab_h2h.py itself (setdefault).
Sharded: 4 shards, rayon=3/shard (matches R4 R4 battery protocol).

Output artifacts:
  artifacts/battery_r4/screen_biased_ext_848484.json
  artifacts/battery_r4/screen_twosided_ext_848484.json
  artifacts/battery_r4/biased_ext_848484.json
  artifacts/battery_r4/twosided_ext_848484.json

Heartbeat logged every 60s during sharded runs.
"""
from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import time

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ARTIFACTS = os.path.join(REPO, "artifacts", "battery_r4")
SHARDED = os.path.join(REPO, "scripts", "h2h_sharded.py")
ENGINE_H2H = os.path.join(REPO, "scripts", "engine_ab_h2h.py")

# Env: native subgame ON. EMD env is set by engine_ab_h2h.py via setdefault — don't override.
BASE_ENV = dict(os.environ)
BASE_ENV["POKERBOT_NATIVE_SUBGAME"] = "1"

SEED = 848484
SHARDS = 4
RAYON = 3


def run_screen(label: str, arm_a: str, arm_b: str, seed: int = SEED, decks: int = 40) -> dict:
    """Run a single-process 40-deck effectiveness screen (not sharded — fast)."""
    out = os.path.join(ARTIFACTS, f"{label}.json")
    cmd = [
        sys.executable, "-u", ENGINE_H2H,
        "--arm-a", arm_a,
        "--arm-b", arm_b,
        "--decks", str(decks),
        "--seed", str(seed),
        "--settle-samples", "200",
        "--label", label,
        "--out", out,
    ]
    log_path = os.path.join(ARTIFACTS, f"{label}.log")
    print(f"\n[SCREEN] {label} — {decks} decks seed {seed}", flush=True)
    t0 = time.perf_counter()
    with open(log_path, "w") as logf:
        proc = subprocess.Popen(cmd, env=BASE_ENV, stdout=logf, stderr=subprocess.STDOUT)
        proc.wait()
    wall = time.perf_counter() - t0
    if proc.returncode != 0:
        raise SystemExit(f"[SCREEN FAILED] {label} rc={proc.returncode} — see {log_path}")
    result = json.load(open(out))
    nonzero = sum(1 for x in result.get("per_deck_nets", []) if abs(x) > 1e-9)
    print(
        f"[SCREEN DONE] {label}: mean/deck={result.get('a_minus_b_per_deck', result.get('ours_minus_them_per_deck', '?')):+.2f}"
        f"  nonzero={nonzero}/{decks}  flopdecks={result.get('flopdecks', '?')}  wall={wall:.0f}s",
        flush=True,
    )
    if nonzero == 0:
        raise SystemExit(f"[DEAD KNOB] {label} — zero nonzero diffs on {decks} decks! STOP. Do not run main run.")
    print(f"  => Knob is LIVE ({nonzero} nonzero diffs). Proceeding to main run.", flush=True)
    return result


def run_sharded(label: str, arm_a: str, arm_b: str, decks: int = 3000,
                seed: int = SEED, shards: int = SHARDS, rayon: int = RAYON) -> dict:
    """Run sharded H2H via h2h_sharded.py with heartbeat logging."""
    out = os.path.join(ARTIFACTS, f"{label}.json")
    cmd = [
        sys.executable, "-u", SHARDED,
        "--shards", str(shards),
        "--rayon-per-shard", str(rayon),
        "--decks", str(decks),
        "--out", out,
        "--",
        ENGINE_H2H,
        "--arm-a", arm_a,
        "--arm-b", arm_b,
        "--seed", str(seed),
        "--settle-samples", "200",
        "--label", label,
    ]
    log_path = os.path.join(ARTIFACTS, f"{label}.driver.log")
    print(
        f"\n[MAIN RUN] {label} — {decks} decks seed {seed} {shards} shards rayon={rayon}",
        flush=True,
    )
    print(f"  log -> {log_path}", flush=True)
    t0 = time.perf_counter()
    logf = open(log_path, "w")
    proc = subprocess.Popen(cmd, env=BASE_ENV, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    # Heartbeat: stream stdout to both log and console; show ETA every 60s.
    buf = []
    last_hb = time.perf_counter()
    while True:
        line = proc.stdout.readline()
        if not line and proc.poll() is not None:
            break
        if line:
            logf.write(line)
            logf.flush()
            buf.append(line)
            # Print progress lines from h2h_sharded.py directly
            if "progress" in line or "MERGED" in line or "shard" in line.lower():
                print(f"  {line.rstrip()}", flush=True)
        now = time.perf_counter()
        if now - last_hb >= 60:
            elapsed = (now - t0) / 60
            # Try to read current merged count from partial shard outputs
            total_done = 0
            base_out = os.path.splitext(out)[0]
            for i in range(shards):
                try:
                    d = json.load(open(f"{base_out}.shard{i}.json"))
                    total_done += len(d.get("per_deck_nets", []))
                except Exception:
                    pass
            eta = (elapsed / max(total_done, 1)) * (decks - total_done) if total_done > 0 else float("nan")
            print(
                f"  [HB {elapsed:.1f}min] {total_done}/{decks} decks done"
                + (f"  ETA ~{eta:.1f}min" if not math.isnan(eta) else ""),
                flush=True,
            )
            last_hb = now

    proc.wait()
    logf.close()
    wall = time.perf_counter() - t0
    if proc.returncode != 0:
        raise SystemExit(f"[MAIN RUN FAILED] {label} rc={proc.returncode} — see {log_path}")

    result = json.load(open(out))
    mean = result.get("a_minus_b_per_deck", result.get("ours_minus_them_per_deck", float("nan")))
    ci = result.get("ci95", [None, None])
    flopdecks = result.get("flopdecks", "?")
    nonzero = sum(1 for x in result.get("per_deck_nets", []) if abs(x) > 1e-9)
    print(
        f"[MAIN DONE] {label}: mean/deck={mean:+.2f}"
        f"  CI95=[{ci[0]:+.1f}, {ci[1]:+.1f}]"
        f"  flopdecks={flopdecks}  nonzero={nonzero}/{decks}"
        f"  wall={wall/60:.1f}min",
        flush=True,
    )
    return result


def pooled_stats(r1: dict, r2: dict) -> tuple[float, float, list[float]]:
    """Combine two independent runs (different seeds): concatenate per-deck diffs, recompute stats."""
    all_diffs = list(r1.get("per_deck_nets", [])) + list(r2.get("per_deck_nets", []))
    n = len(all_diffs)
    m = sum(all_diffs) / n
    se = (sum((x - m) ** 2 for x in all_diffs) / (n - 1)) ** 0.5 / math.sqrt(n)
    ci = [m - 1.96 * se, m + 1.96 * se]
    return m, se, ci


def main():
    os.makedirs(ARTIFACTS, exist_ok=True)
    print(f"=== R4 BATTERY EXTENSION — seed {SEED}, 3000 decks each ===", flush=True)
    print(f"  POKERBOT_NATIVE_SUBGAME={BASE_ENV.get('POKERBOT_NATIVE_SUBGAME')}", flush=True)
    print(f"  EMD env: set by engine_ab_h2h.py setdefaults", flush=True)
    print(f"  Repo: {REPO}", flush=True)
    t_total = time.perf_counter()

    # ------------------------------------------------------------------
    # STEP 1: Effectiveness screens (40 decks, seed 848484)
    # ------------------------------------------------------------------
    print("\n" + "=" * 70, flush=True)
    print("STEP 1: Effectiveness re-screens (40 decks, seed 848484)", flush=True)
    print("=" * 70, flush=True)

    screen_biased = run_screen(
        "screen_biased_ext_848484",
        arm_a='{"leaf_biased_continuations": true}',
        arm_b="{}",
    )

    screen_twosided = run_screen(
        "screen_twosided_ext_848484",
        arm_a='{"two_sided": false}',
        arm_b="{}",
    )

    print("\n[SCREENS PASSED] Both knobs are live at seed 848484.", flush=True)

    # ------------------------------------------------------------------
    # STEP 2: Main run 1 — leaf_biased_continuations
    # ------------------------------------------------------------------
    print("\n" + "=" * 70, flush=True)
    print("STEP 2: Main run 1 — leaf_biased_continuations (3000 decks)", flush=True)
    print("=" * 70, flush=True)

    r_biased_ext = run_sharded(
        "biased_ext_848484",
        arm_a='{"leaf_biased_continuations": true}',
        arm_b="{}",
    )

    # ------------------------------------------------------------------
    # STEP 3: Main run 2 — two_sided=False
    # ------------------------------------------------------------------
    print("\n" + "=" * 70, flush=True)
    print("STEP 3: Main run 2 — two_sided=False (3000 decks)", flush=True)
    print("=" * 70, flush=True)

    r_twosided_ext = run_sharded(
        "twosided_ext_848484",
        arm_a='{"two_sided": false}',
        arm_b="{}",
    )

    # ------------------------------------------------------------------
    # STEP 4: Load the seed-424242 1000-deck runs and compute pooled stats
    # ------------------------------------------------------------------
    print("\n" + "=" * 70, flush=True)
    print("STEP 4: Computing pooled statistics", flush=True)
    print("=" * 70, flush=True)

    r_biased_1k = json.load(open(os.path.join(ARTIFACTS, "biased_leaves_1k.json")))
    r_twosided_1k = json.load(open(os.path.join(ARTIFACTS, "two_sided_off_1k.json")))

    m_b, se_b, ci_b = pooled_stats(r_biased_1k, r_biased_ext)
    m_ts, se_ts, ci_ts = pooled_stats(r_twosided_1k, r_twosided_ext)

    n_biased = len(r_biased_1k.get("per_deck_nets", [])) + len(r_biased_ext.get("per_deck_nets", []))
    n_twosided = len(r_twosided_1k.get("per_deck_nets", [])) + len(r_twosided_ext.get("per_deck_nets", []))

    # Decision rules
    biased_verdict = "ADOPT-RECOMMEND" if ci_b[0] > 0 else "PARK"
    twosided_verdict_detail = ""
    if ci_ts[1] < 0:
        # CI entirely negative: OFF is certifiably worse than ON -> recommend flip live to ON
        twosided_verdict_detail = "CI_HI<0 => two_sided=OFF certifiably WORSE; RECOMMEND flip live to two_sided=True"
    elif ci_ts[0] > 0:
        # CI entirely positive: OFF is certifiably better than ON
        twosided_verdict_detail = "CI_LO>0 => two_sided=OFF certifiably BETTER (unexpected)"
    else:
        # CI includes 0: inconclusive
        twosided_verdict_detail = "CI includes 0 => leave live at two_sided=False; eval-baseline divergence harmless"

    print(f"\n--- POOLED: leaf_biased_continuations ({n_biased} decks) ---", flush=True)
    print(f"  1k run (seed 424242): mean={r_biased_1k['a_minus_b_per_deck']:+.2f}", flush=True)
    print(f"  3k run (seed 848484): mean={r_biased_ext.get('a_minus_b_per_deck', r_biased_ext.get('ours_minus_them_per_deck')):+.2f}", flush=True)
    print(f"  POOLED: mean={m_b:+.2f}  SE={se_b:.2f}  CI95=[{ci_b[0]:+.1f}, {ci_b[1]:+.1f}]", flush=True)
    print(f"  DECISION: {biased_verdict}", flush=True)

    print(f"\n--- POOLED: two_sided=False ({n_twosided} decks) ---", flush=True)
    print(f"  1k run (seed 424242): mean={r_twosided_1k['a_minus_b_per_deck']:+.2f}", flush=True)
    print(f"  3k run (seed 848484): mean={r_twosided_ext.get('a_minus_b_per_deck', r_twosided_ext.get('ours_minus_them_per_deck')):+.2f}", flush=True)
    print(f"  POOLED: mean={m_ts:+.2f}  SE={se_ts:.2f}  CI95=[{ci_ts[0]:+.1f}, {ci_ts[1]:+.1f}]", flush=True)
    print(f"  DECISION: {twosided_verdict_detail}", flush=True)

    # ------------------------------------------------------------------
    # STEP 5: Write verdict markdown
    # ------------------------------------------------------------------
    verdict_path = os.path.join(ARTIFACTS, "extension_verdict.md")
    ext_biased_mean = r_biased_ext.get("a_minus_b_per_deck", r_biased_ext.get("ours_minus_them_per_deck", float("nan")))
    ext_biased_ci = r_biased_ext.get("ci95", [None, None])
    ext_biased_fd = r_biased_ext.get("flopdecks", "?")
    ext_twosided_mean = r_twosided_ext.get("a_minus_b_per_deck", r_twosided_ext.get("ours_minus_them_per_deck", float("nan")))
    ext_twosided_ci = r_twosided_ext.get("ci95", [None, None])
    ext_twosided_fd = r_twosided_ext.get("flopdecks", "?")
    total_wall = (time.perf_counter() - t_total) / 60

    # Screen nonzero counts
    screen_biased_nonzero = sum(1 for x in screen_biased.get("per_deck_nets", []) if abs(x) > 1e-9)
    screen_twosided_nonzero = sum(1 for x in screen_twosided.get("per_deck_nets", []) if abs(x) > 1e-9)
    screen_biased_mean = screen_biased.get("a_minus_b_per_deck", float("nan"))
    screen_twosided_mean = screen_twosided.get("a_minus_b_per_deck", float("nan"))

    biased_nonzero_ext = sum(1 for x in r_biased_ext.get("per_deck_nets", []) if abs(x) > 1e-9)
    twosided_nonzero_ext = sum(1 for x in r_twosided_ext.get("per_deck_nets", []) if abs(x) > 1e-9)

    verdict_md = f"""# R4 Battery Extension Verdict — seed 848484, 3000 decks each

**Date:** 2026-07-02
**Engine:** EMD-K160 (POKERBOT_NATIVE_SUBGAME=1, EMD env via engine_ab_h2h.py setdefaults)
**Base seed (extension):** {SEED}  |  **Decks per run:** 3000 (4 shards, rayon=3/shard)
**Prior run (pooled in):** seed 424242, 1000 decks (artifacts/battery_r4/biased_leaves_1k.json, two_sided_off_1k.json)
**Total wall time:** {total_wall:.1f} min

---

## Effectiveness Re-screens (40 decks, seed 848484)

Both knobs screened LIVE at the new seed before main runs.

| Candidate | mean/deck | nonzero/40 | flopdecks | Status |
|-----------|-----------|------------|-----------|--------|
| `{{"leaf_biased_continuations": true}}` | {screen_biased_mean:+.2f} | {screen_biased_nonzero}/40 | {screen_biased.get('flopdecks', '?')} | LIVE |
| `{{"two_sided": false}}` | {screen_twosided_mean:+.2f} | {screen_twosided_nonzero}/40 | {screen_twosided.get('flopdecks', '?')} | LIVE |

---

## Extension Run Results (3000 decks, seed 848484)

### Run 1: `{{"leaf_biased_continuations": true}}` vs `{{}}`

| Metric | Value |
|--------|-------|
| mean/deck (A−B) | {ext_biased_mean:+.2f} |
| CI95 | [{ext_biased_ci[0]:+.1f}, {ext_biased_ci[1]:+.1f}] |
| flopdecks | {ext_biased_fd} |
| nonzero decks | {biased_nonzero_ext}/3000 |

### Run 2: `{{"two_sided": false}}` vs `{{}}`

| Metric | Value |
|--------|-------|
| mean/deck (A−B) | {ext_twosided_mean:+.2f} |
| CI95 | [{ext_twosided_ci[0]:+.1f}, {ext_twosided_ci[1]:+.1f}] |
| flopdecks | {ext_twosided_fd} |
| nonzero decks | {twosided_nonzero_ext}/3000 |

---

## Pooled Statistics (seed 424242 + seed 848484, independent seeds)

Method: concatenate per-deck diffs across both runs (independent seeds → simple concatenation is
unbiased). Pooled n = n_1 + n_2. SE = std(all_diffs) / sqrt(N).

### C1: `leaf_biased_continuations=True`

| Run | n | mean/deck |
|-----|---|-----------|
| seed 424242 (R4) | 1000 | {r_biased_1k['a_minus_b_per_deck']:+.2f} |
| seed 848484 (ext) | 3000 | {ext_biased_mean:+.2f} |
| **POOLED** | **{n_biased}** | **{m_b:+.2f}** |

- **Pooled SE:** {se_b:.2f}
- **Pooled CI95:** [{ci_b[0]:+.1f}, {ci_b[1]:+.1f}]
- **CI excludes 0:** {"YES (lo > 0)" if ci_b[0] > 0 else "NO (CI includes 0)"}

**Decision (biased_continuations):** {biased_verdict}

{"ADOPT-RECOMMEND: Pooled EMD CI lo > 0. Recommend flipping default to leaf_biased_continuations=True in scaffold baseline + PluribusReBeLConfig (do not apply here; recommend only)." if biased_verdict == "ADOPT-RECOMMEND" else "PARK: Two engines (~+40 lean both), pooled CI still includes 0. Signal is directionally positive but does not certify at the ~4000-deck budget. Stop spending decks on this knob."}

### C2: `two_sided=False` (arm A) vs `two_sided=True` (arm B, deployed default)

Reminder: A negative mean means two_sided=OFF is WORSE than ON (live deviates from deployed baseline by running OFF).

| Run | n | mean/deck |
|-----|---|-----------|
| seed 424242 (R4) | 1000 | {r_twosided_1k['a_minus_b_per_deck']:+.2f} |
| seed 848484 (ext) | 3000 | {ext_twosided_mean:+.2f} |
| **POOLED** | **{n_twosided}** | **{m_ts:+.2f}** |

- **Pooled SE:** {se_ts:.2f}
- **Pooled CI95:** [{ci_ts[0]:+.1f}, {ci_ts[1]:+.1f}]
- **CI excludes 0:** {"YES (hi < 0)" if ci_ts[1] < 0 else ("YES (lo > 0)" if ci_ts[0] > 0 else "NO (CI includes 0)")}

**Decision (two_sided):** {twosided_verdict_detail}

---

## Artifacts

- `screen_biased_ext_848484.json` — effectiveness re-screen biased_continuations (40 decks)
- `screen_twosided_ext_848484.json` — effectiveness re-screen two_sided (40 decks)
- `biased_ext_848484.json` — main run leaf_biased_continuations (3000 decks)
- `twosided_ext_848484.json` — main run two_sided=False (3000 decks)
"""

    with open(verdict_path, "w") as f:
        f.write(verdict_md)
    print(f"\n[VERDICT WRITTEN] {verdict_path}", flush=True)
    print(f"\n=== ALL DONE  Total wall: {total_wall:.1f} min ===", flush=True)


if __name__ == "__main__":
    main()
