#!/usr/bin/env python3
"""Battery driver: run each candidate spec vs a shared baseline sequentially via engine_ab_h2h.py.

Resumable: candidates whose output JSON already exists are skipped.

Usage
-----
  # candidates.json = [{"name": "K150", "arm_spec": {"iterations": 150}}, ...]
  python scripts/engine_ab_battery.py \\
      --baseline '{}' \\
      --candidates candidates.json \\
      --decks 40 \\
      --seed 4242 \\
      --settle-samples 200 \\
      --outdir artifacts/battery_run1

Output
------
  artifacts/battery_run1/<name>.json      — per-candidate H2H result
  artifacts/ab_battery_<timestamp>.md     — ranked summary table (markdown)
  artifacts/ab_battery_<timestamp>.json   — machine-readable summary

Ranking
-------
  ADOPT   — 95% CI entirely above 0   (candidate > baseline)
  REJECT  — 95% CI entirely below 0   (candidate < baseline)
  INCONCLUSIVE — CI straddles 0
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time


def _run_h2h(baseline_spec: str, arm_spec: str, name: str, decks: int, seed: int,
             settle_samples: int, outdir: str, extra_args: list[str]) -> str:
    """Run engine_ab_h2h.py for one candidate; return path to the output JSON."""
    out_path = os.path.join(outdir, f"{name}.json")
    here = os.path.dirname(os.path.abspath(__file__))
    script = os.path.join(here, "engine_ab_h2h.py")
    sharder = os.path.join(here, "h2h_sharded.py")

    # Shard each A/B run across worker processes (CRN-exact; see h2h_sharded.py).
    # Default 6: measured knee on this 6-core box post EMD-native port (240 decks, seed 60321:
    # 4 shards 121s / 6 shards 101s / 8 shards 100s — artifacts/shard_scaling_s{4,6,8}.json).
    # The old 4-shard ceiling was measured when 8s Python solves saturated rayon; stale since 197862a.
    shards = int(os.environ.get("POKERBOT_H2H_SHARDS", "6"))
    cmd = [
        sys.executable, "-u", sharder,
        "--shards", str(shards),
        "--decks", str(decks),
        "--out", out_path,
        "--",
        script,
        "--arm-a", arm_spec,
        "--arm-b", baseline_spec,
        "--seed", str(seed),
        "--settle-samples", str(settle_samples),
        "--label", name,
    ] + extra_args

    print(f"\n{'=' * 70}", flush=True)
    print(f"RUNNING: {name}", flush=True)
    print(f"  cmd: {' '.join(cmd)}", flush=True)
    print(f"{'=' * 70}", flush=True)

    t0 = time.perf_counter()
    proc = subprocess.run(cmd, check=False)
    elapsed = time.perf_counter() - t0

    if proc.returncode != 0:
        print(f"  [battery] WARNING: {name} exited with code {proc.returncode}", flush=True)
    else:
        print(f"  [battery] {name} completed in {elapsed / 60:.1f} min", flush=True)

    return out_path


def _load_result(path: str) -> dict | None:
    """Load a completed H2H result JSON; return None if missing or malformed."""
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def _verdict(lo: float, hi: float) -> str:
    if lo > 0:
        return "ADOPT"
    if hi < 0:
        return "REJECT"
    return "INCONCLUSIVE"


def _format_ci(mean: float, lo: float, hi: float) -> str:
    return f"{mean:+.3f} [{lo:+.3f}, {hi:+.3f}]"


def main():
    parser = argparse.ArgumentParser(
        description="Battery driver: run multiple candidates vs a shared baseline."
    )
    parser.add_argument("--baseline", required=True,
                        help="JSON spec for the baseline arm (B); {} = deployed config")
    parser.add_argument("--candidates", required=True,
                        help="Path to JSON file: list of {name, arm_spec} objects")
    parser.add_argument("--decks", type=int, default=40,
                        help="Number of duplicate decks per H2H run")
    parser.add_argument("--seed", type=int, default=4242, help="Base RNG seed")
    parser.add_argument("--settle-samples", type=int, default=200,
                        help="All-in EV runouts per action (0 = disable)")
    parser.add_argument("--outdir", default="artifacts/ab_battery",
                        help="Directory for per-candidate JSON outputs")
    parser.add_argument("--repo-root", default=os.getcwd(),
                        help="Repo root passed through to engine_ab_h2h.py")
    args, extra = parser.parse_known_args()
    # extra_args passed through to engine_ab_h2h.py (e.g. --repo-root override)
    extra_args = ["--repo-root", args.repo_root] + extra

    # Load candidates list.
    with open(args.candidates) as f:
        candidates = json.load(f)
    if not isinstance(candidates, list):
        sys.exit(f"[battery] ERROR: {args.candidates} must contain a JSON list")

    os.makedirs(args.outdir, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")

    print(f"[battery] {len(candidates)} candidates, decks={args.decks}, seed={args.seed}", flush=True)
    print(f"[battery] baseline spec: {args.baseline}", flush=True)
    print(f"[battery] outdir: {args.outdir}", flush=True)

    results = []
    for entry in candidates:
        name = str(entry["name"])
        arm_spec = json.dumps(entry["arm_spec"]) if isinstance(entry["arm_spec"], dict) else str(entry["arm_spec"])
        out_path = os.path.join(args.outdir, f"{name}.json")

        existing = _load_result(out_path)
        if existing is not None and "a_minus_b_per_deck" in existing:
            print(f"\n[battery] SKIPPING {name} — result already exists at {out_path}", flush=True)
            results.append((name, arm_spec, existing))
            continue

        _run_h2h(
            baseline_spec=args.baseline,
            arm_spec=arm_spec,
            name=name,
            decks=args.decks,
            seed=args.seed,
            settle_samples=args.settle_samples,
            outdir=args.outdir,
            extra_args=extra_args,
        )
        r = _load_result(out_path)
        if r is None:
            print(f"[battery] WARNING: {name} produced no valid output at {out_path}", flush=True)
            r = {"label": name, "a_minus_b_per_deck": float("nan"), "ci95": [float("nan"), float("nan")],
                 "error": "no output"}
        results.append((name, arm_spec, r))

    # --- Ranked summary ---
    # Sort: ADOPT first (highest mean), then INCONCLUSIVE, then REJECT (lowest mean).
    def _sort_key(item):
        _, _, r = item
        mean = r.get("a_minus_b_per_deck", float("nan"))
        lo, hi = r.get("ci95", [float("nan"), float("nan")])
        v = _verdict(lo, hi)
        order = {"ADOPT": 0, "INCONCLUSIVE": 1, "REJECT": 2}.get(v, 3)
        return (order, -mean if not math.isnan(mean) else 0)

    results_sorted = sorted(results, key=_sort_key)

    # Markdown table.
    md_lines = [
        f"# A/B Battery Summary — {ts}",
        "",
        f"baseline spec: `{args.baseline}`  decks={args.decks}  seed={args.seed}",
        "",
        "| Rank | Name | mean/deck (A-B) | 95% CI | Verdict |",
        "|------|------|-----------------|--------|---------|",
    ]
    summary_rows = []
    for rank, (name, arm_spec, r) in enumerate(results_sorted, 1):
        mean = r.get("a_minus_b_per_deck", float("nan"))
        lo, hi = r.get("ci95", [float("nan"), float("nan")])
        v = _verdict(lo, hi) if not (math.isnan(lo) or math.isnan(hi)) else "ERROR"
        ci_str = _format_ci(mean, lo, hi) if not math.isnan(mean) else "n/a"
        md_lines.append(f"| {rank} | `{name}` | {mean:+.3f} | {ci_str} | **{v}** |")
        summary_rows.append({
            "rank": rank,
            "name": name,
            "arm_spec": arm_spec,
            "mean_per_deck": mean,
            "ci95_lo": lo,
            "ci95_hi": hi,
            "verdict": v,
            "result_file": os.path.join(args.outdir, f"{name}.json"),
        })

    md_lines += [
        "",
        "## Legend",
        "- **ADOPT**: 95% CI entirely above 0 — candidate arm (A) beats baseline (B)",
        "- **REJECT**: 95% CI entirely below 0 — baseline beats candidate",
        "- **INCONCLUSIVE**: CI straddles 0 — need more decks",
    ]
    md_text = "\n".join(md_lines)

    md_out = f"artifacts/ab_battery_{ts}.md"
    json_out = f"artifacts/ab_battery_{ts}.json"

    os.makedirs("artifacts", exist_ok=True)
    with open(md_out, "w") as f:
        f.write(md_text)
    with open(json_out, "w") as f:
        json.dump({
            "ts": ts,
            "baseline": args.baseline,
            "decks": args.decks,
            "seed": args.seed,
            "candidates": summary_rows,
        }, f, indent=1)

    print("\n" + md_text, flush=True)
    print(f"\nwrote {md_out}", flush=True)
    print(f"wrote {json_out}", flush=True)


if __name__ == "__main__":
    main()
