"""JSON and Markdown export for evaluation reports."""

from __future__ import annotations

import json
from pathlib import Path

from aof_rl.eval.eval_schema import EvalReport


def write_json_report(path: str | Path, report: EvalReport) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")


def write_markdown_report(path: str | Path, report: EvalReport) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = [
        f"# AoF evaluation: {report.eval_id}",
        "",
        f"Generated: {report.generated_at}",
        "",
        "## Aggregate",
        "",
        f"- Hands (seat-observations): {report.aggregate.hands}",
        f"- Mean bb/hand: **{report.aggregate.bb_per_hand:.4f}**",
        f"- 95% CI: [{report.aggregate.bb_per_hand_ci95[0]:.4f}, {report.aggregate.bb_per_hand_ci95[1]:.4f}]",
        f"- bb/100: {report.aggregate.bb_per_100:.2f}",
    ]
    if report.aggregate.disagreement_rate_vs_teacher is not None:
        lines.append(
            f"- Disagreement vs teacher: {report.aggregate.disagreement_rate_vs_teacher:.2%}"
        )
    lines.extend(["", "## Per seat", ""])
    for seat, m in sorted(report.by_seat.items()):
        lines.append(
            f"- **{seat}**: bb/hand={m.bb_per_hand:.4f} "
            f"(CI [{m.bb_per_hand_ci95[0]:.4f}, {m.bb_per_hand_ci95[1]:.4f}]), n={m.hands}"
        )
    lines.extend(["", "## By table size", ""])
    for ts, m in sorted(report.by_table_size.items(), key=lambda x: int(x[0])):
        lines.append(
            f"- **{ts}p**: bb/hand={m.bb_per_hand:.4f}, bb/100={m.bb_per_100:.2f}, n={m.hands}"
        )
    lines.extend(["", "## Thin-edge calibration", ""])
    for b in report.calibration["thin_edge_buckets"]:
        lines.append(
            f"- **{b.label}** (n={b.count}): disagree={b.disagreement_rate:.2%}, "
            f"mean |Q err|={b.mean_q_error_bb:.3f} bb, mean unc={b.mean_uncertainty_bb:.3f} bb"
        )
    if report.hu_exploitability is not None:
        hu = report.hu_exploitability
        lines.extend(
            [
                "",
                "## HU exploitability",
                "",
                f"- Mode: {hu.mode}",
                f"- Exploitability: **{hu.exploitability_bb_per_hand:.4f}** bb/hand",
                f"- SB BR gain: {hu.sb_br_gain_bb:.4f} bb",
                f"- BB BR gain: {hu.bb_br_gain_bb:.4f} bb",
                f"- Pairs/buckets enumerated: {hu.hands_enumerated}",
            ]
        )
        if hu.note:
            lines.append(f"- Note: {hu.note}")
    if report.multiway_deviation:
        lines.extend(["", "## Multiway unilateral deviation (oracle BR)", ""])
        for ts, seats in sorted(report.multiway_deviation.items(), key=lambda x: int(x[0])):
            lines.append(f"### {ts}-max")
            for seat, g in sorted(seats.items()):
                lines.append(
                    f"- **{seat}**: gain={g.deviation_gain_bb_per_hand:.4f} bb/hand (n={g.hands})"
                )
    if report.paired_comparison is not None:
        pc = report.paired_comparison
        lines.extend(
            [
                "",
                "## Paired comparison (CRN)",
                "",
                f"- {pc.policy_a} vs {pc.policy_b} over {pc.hands} hands",
                f"- Mean Δ bb/hand (A−B): **{pc.mean_delta_bb_per_hand:.4f}**",
                f"- 95% CI: [{pc.delta_bb_per_hand_ci95[0]:.4f}, {pc.delta_bb_per_hand_ci95[1]:.4f}]",
                f"- Δ bb/100: {pc.mean_delta_bb_per_100:.2f}",
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
