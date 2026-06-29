#!/usr/bin/env python3
"""Profile the Rust-native EMD verified-better eval and estimate trend cost.

Two modes:
  run       Execute the same emd -> fast -> combine arms as the pipeline.
  summarize Read existing partial JSONs and produce the same speed/feasibility report.

The `run` mode intentionally passes --require-native. A Python fallback benchmark is worse
than useless here because it looks like progress while measuring the wrong harness.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
EVAL_SCRIPT = ROOT / "scripts" / "eval" / "eval_emd_vs_iter62000.py"
sys.path.insert(0, str(ROOT / "scripts"))
from _emd_selection_gate import build_selection_report  # noqa: E402


def _load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: str | Path, obj: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _fmt_s(seconds: float | None) -> str:
    if seconds is None or not math.isfinite(seconds):
        return "n/a"
    if seconds < 90:
        return f"{seconds:.1f}s"
    return f"{seconds / 60.0:.1f}m"


def _result(partial: dict[str, Any]) -> dict[str, Any]:
    return partial.get("result") or {}


def _hands(partial: dict[str, Any]) -> int:
    r = _result(partial)
    return int(r.get("hands_per_seed", 0)) * len(r.get("seeds") or [])


def _wall(partial: dict[str, Any]) -> float:
    return float(_result(partial).get("wall_s") or 0.0)


def _hands_per_sec(partial: dict[str, Any]) -> float:
    wall = _wall(partial)
    return (_hands(partial) / wall) if wall > 0 else 0.0


def _ci_width(partial: dict[str, Any]) -> float | None:
    ci = _result(partial).get("pooled_ci95")
    if not isinstance(ci, list) or len(ci) != 2:
        return None
    return float(ci[1]) - float(ci[0])


def _paired_delta_stats(emd: dict[str, Any], baseline: dict[str, Any]) -> dict[str, float] | None:
    e = [float(x) for x in (_result(emd).get("per_seed") or [])]
    b = [float(x) for x in (_result(baseline).get("per_seed") or [])]
    if not e or len(e) != len(b):
        return None
    diffs = [bb - ee for ee, bb in zip(e, b)]
    n = len(diffs)
    mean = sum(diffs) / n
    if n < 2:
        return {"n": float(n), "mean": mean, "stderr": 0.0, "ci95_half_width": 0.0}
    var = sum((x - mean) ** 2 for x in diffs) / (n - 1)
    stderr = math.sqrt(var / n)
    return {
        "n": float(n),
        "mean": mean,
        "stderr": stderr,
        "ci95_half_width": 1.96 * stderr,
    }


def _mean_ci(vals: list[float]) -> dict[str, float]:
    n = len(vals)
    mean = sum(vals) / n if n else 0.0
    if n < 2:
        return {"n": float(n), "mean": mean, "stderr": 0.0, "ci95_half_width": 0.0}
    var = sum((x - mean) ** 2 for x in vals) / (n - 1)
    stderr = math.sqrt(var / n)
    return {"n": float(n), "mean": mean, "stderr": stderr, "ci95_half_width": 1.96 * stderr}


def _paired_seat_delta_stats(emd: dict[str, Any], baseline: dict[str, Any]) -> dict[str, dict[str, float]] | None:
    """Return per-seat paired deltas when partials include per_seed_per_seat.

    Delta convention matches the verdict: baseline - emd, positive means the new
    blueprint reduced LBR exploitability for that seat.
    """
    e_rows = _result(emd).get("per_seed_per_seat") or []
    b_rows = _result(baseline).get("per_seed_per_seat") or []
    if not e_rows or len(e_rows) != len(b_rows):
        return None
    seats = sorted({str(k) for row in e_rows + b_rows for k in row.keys()}, key=lambda x: int(x))
    out: dict[str, dict[str, float]] = {}
    for seat in seats:
        diffs = []
        for e, b in zip(e_rows, b_rows):
            if seat in e and seat in b:
                diffs.append(float(b[seat]) - float(e[seat]))
        if diffs:
            out[seat] = _mean_ci(diffs)
    return out or None


def _trend_estimates(candidate_hps: float) -> list[dict[str, Any]]:
    grid = [
        ("tiny heartbeat", 4, 1000),
        ("cheap trend", 8, 1000),
        ("medium trend", 12, 2000),
        ("current full", 24, 4000),
        ("high-confidence", 48, 4000),
        ("selection final", 96, 4000),
    ]
    rows = []
    for label, seeds, hands in grid:
        total = seeds * hands
        sec = total / candidate_hps if candidate_hps > 0 else None
        rows.append({"label": label, "seeds": seeds, "hands": hands, "total_hands": total, "est_seconds": sec})
    return rows


def _summarize(
    emd: dict[str, Any],
    baseline: dict[str, Any] | None,
    verdict: dict[str, Any] | None,
    *,
    selection_tier: str = "cheap",
) -> dict[str, Any]:
    emd_hps = _hands_per_sec(emd)
    out: dict[str, Any] = {
        "emd": {
            "method": _result(emd).get("method"),
            "infosets": _result(emd).get("infosets"),
            "seeds": len(_result(emd).get("seeds") or []),
            "hands_per_seed": _result(emd).get("hands_per_seed"),
            "total_hands": _hands(emd),
            "wall_s": _wall(emd),
            "hands_per_second": emd_hps,
            "pooled_bb_per_100": _result(emd).get("pooled_bb_per_100"),
            "ci95_width_bb_per_100": _ci_width(emd),
        },
        "trend_cost_estimates": _trend_estimates(emd_hps),
    }
    if baseline is not None:
        out["baseline"] = {
            "method": _result(baseline).get("method"),
            "infosets": _result(baseline).get("infosets"),
            "seeds": len(_result(baseline).get("seeds") or []),
            "hands_per_seed": _result(baseline).get("hands_per_seed"),
            "total_hands": _hands(baseline),
            "wall_s": _wall(baseline),
            "hands_per_second": _hands_per_sec(baseline),
            "pooled_bb_per_100": _result(baseline).get("pooled_bb_per_100"),
            "ci95_width_bb_per_100": _ci_width(baseline),
        }
        out["full_gate_core_wall_s"] = _wall(emd) + _wall(baseline)
        out["paired_global_delta"] = _paired_delta_stats(emd, baseline)
        out["paired_seat_delta"] = _paired_seat_delta_stats(emd, baseline)
    if verdict is not None:
        out["verdict"] = {
            "emd_better": verdict.get("emd_better"),
            "mean_blind_gain": verdict.get("mean_blind_gain"),
            "global_regression": verdict.get("global_regression"),
            "anchor_ok": verdict.get("anchor_ok"),
        }
        out["selection"] = build_selection_report(
            verdict,
            tier=selection_tier,
            emd_partial=emd,
            base_partial=baseline,
        )
    return out


def _render_markdown(summary: dict[str, Any]) -> str:
    emd = summary["emd"]
    lines = [
        "# Rust Native EMD Eval Profile",
        "",
        "## Measured Speed",
        "",
        f"- EMD arm: {emd['total_hands']:,} hands in {_fmt_s(float(emd['wall_s']))} "
        f"({float(emd['hands_per_second']):,.0f} hands/s), method `{emd['method']}`.",
    ]
    if "baseline" in summary:
        base = summary["baseline"]
        lines.append(
            f"- Baseline arm: {base['total_hands']:,} hands in {_fmt_s(float(base['wall_s']))} "
            f"({float(base['hands_per_second']):,.0f} hands/s), method `{base['method']}`."
        )
        lines.append(f"- Full two-arm gate core eval: {_fmt_s(float(summary['full_gate_core_wall_s']))}.")
    lines.extend(["", "## Trend Cost", ""])
    for row in summary["trend_cost_estimates"]:
        lines.append(
            f"- {row['label']}: {row['seeds']} seeds x {row['hands']} hands "
            f"= {row['total_hands']:,} hands -> {_fmt_s(row['est_seconds'])} candidate-only."
        )
    if summary.get("paired_global_delta"):
        d = summary["paired_global_delta"]
        lines.extend([
            "",
            "## Signal",
            "",
            f"- Paired global delta over {int(d['n'])} CRN seeds: {d['mean']:+.1f} bb/100 "
            f"+/- {d['ci95_half_width']:.1f} bb/100 (95%).",
        ])
    if summary.get("paired_seat_delta"):
        lines.extend(["", "## Seat Delta Signal", ""])
        for seat, d in summary["paired_seat_delta"].items():
            lines.append(
                f"- Seat {seat}: {d['mean']:+.1f} bb/100 +/- {d['ci95_half_width']:.1f} bb/100 (95%)."
            )
    if summary.get("verdict"):
        v = summary["verdict"]
        lines.extend([
            "",
            "## Verdict Snapshot",
            "",
            f"- emd_better={v.get('emd_better')}  mean_blind_gain={float(v.get('mean_blind_gain') or 0.0):+.1f} "
            f"global_regression={float(v.get('global_regression') or 0.0):+.1f} anchor_ok={v.get('anchor_ok')}",
        ])
    if summary.get("selection"):
        s = summary["selection"]
        lines.extend([
            "",
            "## Selection Report",
            "",
            f"- selection_pass={s.get('selection_pass')}  acceptance_pass={s.get('acceptance_pass')}  "
            f"tier={s.get('tier')}",
            f"- reason_codes={s.get('selection_reason_codes')}",
        ])
    lines.extend([
        "",
        "## Convergence Feasibility",
        "",
        "- Feasible as a checkpoint trend/selection signal: cache the baseline arm, reuse fixed CRN seeds, "
        "and run candidate-only evals at milestones.",
        "- Not safe as the sole convergence proof: 6-max LBR is an operational adversary measurement, not "
        "Nash exploitability, and the pooled confidence intervals are still wide.",
        "- Practical rule: use eval trend for early-abort and model selection, while keeping CFR strategy-delta "
        "convergence logging and a final full verified-better gate.",
    ])
    return "\n".join(lines) + "\n"


def _run_cmd(args: list[str], *, env: dict[str, str], log_path: Path) -> dict[str, Any]:
    t0 = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.run(args, cwd=ROOT, env=env, text=True, stdout=log, stderr=subprocess.STDOUT, check=False)
    wall = time.perf_counter() - t0
    if proc.returncode != 0:
        raise SystemExit(f"command failed ({proc.returncode}); see {log_path}")
    return {"command_wall_s": wall, "log": str(log_path)}


def cmd_run(args: argparse.Namespace) -> int:
    work = Path(args.workdir)
    work.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    py = args.python or sys.executable
    emd_json = work / "emd_partial.json"
    base_json = work / "iter62000_partial.json"
    verdict_json = work / "verdict.json"
    emd_cmd = [
        py, str(EVAL_SCRIPT), "emd",
        "--checkpoint", str(Path(args.emd_checkpoint)),
        "--emd-frozen", str(Path(args.emd_frozen)),
        "--out", str(emd_json),
        "--seeds", str(args.seeds),
        "--hands", str(args.hands),
        "--num-buckets", str(args.num_buckets),
        "--value-only",
        "--native",
        "--require-native",
    ]
    base_cmd = [
        py, str(EVAL_SCRIPT), "fast",
        "--checkpoint", str(Path(args.baseline_checkpoint)),
        "--out", str(base_json),
        "--seeds", str(args.seeds),
        "--hands", str(args.hands),
        "--num-buckets", str(args.baseline_num_buckets),
        "--value-only",
        "--native",
        "--require-native",
    ]
    timings = {
        "emd_command": _run_cmd(emd_cmd, env=env, log_path=work / "emd.log"),
        "baseline_command": _run_cmd(base_cmd, env=env, log_path=work / "iter62000.log"),
    }
    combine_cmd = [
        py, str(EVAL_SCRIPT), "combine",
        "--emd-json", str(emd_json),
        "--iter62000-json", str(base_json),
        "--out", str(verdict_json),
        "--num-buckets", str(args.num_buckets),
        "--emd-frozen", str(Path(args.emd_frozen)),
        "--selection-tier", str(getattr(args, "selection_tier", "cheap")),
    ]
    timings["combine_command"] = _run_cmd(combine_cmd, env=env, log_path=work / "combine.log")
    emd_data = _load_json(emd_json)
    base_data = _load_json(base_json)
    verdict_data = _load_json(verdict_json)
    summary = _summarize(emd_data, base_data, verdict_data, selection_tier=args.selection_tier)
    summary["command_timings"] = timings
    _write_json(work / "profile_summary.json", summary)
    (work / "profile_summary.md").write_text(_render_markdown(summary), encoding="utf-8")
    print(work / "profile_summary.md")
    return 0


def cmd_summarize(args: argparse.Namespace) -> int:
    summary = _summarize(
        _load_json(args.emd_json),
        _load_json(args.iter62000_json) if args.iter62000_json else None,
        _load_json(args.verdict_json) if args.verdict_json else None,
        selection_tier=args.selection_tier,
    )
    if args.out_json:
        _write_json(args.out_json, summary)
    if args.out_md:
        Path(args.out_md).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_md).write_text(_render_markdown(summary), encoding="utf-8")
    print(_render_markdown(summary))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run a full local native eval profile")
    r.add_argument("--emd-checkpoint", required=True)
    r.add_argument("--emd-frozen", required=True)
    r.add_argument("--baseline-checkpoint", required=True)
    r.add_argument("--num-buckets", type=int, required=True)
    r.add_argument("--baseline-num-buckets", type=int, default=200)
    r.add_argument("--seeds", type=int, default=24)
    r.add_argument("--hands", type=int, default=4000)
    r.add_argument("--workdir", default=str(ROOT / "artifacts" / "emd_native_eval_profile"))
    r.add_argument("--python", default=sys.executable)
    r.add_argument("--selection-tier", default="cheap",
                   choices=("heartbeat", "cheap", "medium", "final"))
    r.set_defaults(func=cmd_run)
    s = sub.add_parser("summarize", help="summarize existing partial JSONs")
    s.add_argument("--emd-json", required=True)
    s.add_argument("--iter62000-json")
    s.add_argument("--verdict-json")
    s.add_argument("--out-json")
    s.add_argument("--out-md")
    s.add_argument("--selection-tier", default="cheap",
                   choices=("heartbeat", "cheap", "medium", "final"))
    s.set_defaults(func=cmd_summarize)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
