"""Render a local SVG flamegraph for the native trainer profile counters.

This is a Windows-friendly fallback for the K160 EMD trainer when `cargo
flamegraph` cannot start the OS sampler. It runs the same PyO3
`NlheCFRTrainer.train_profile_json` hook used by the babysitter/microbench and
renders a folded-stack SVG weighted by observed phase wall time plus profile
counters. It is not a sampled call-stack profile; use the emitted cargo
flamegraph command from an elevated shell for that.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.profile_native_trainer_microbench import _load_ext, _summarize  # noqa: E402


def _total(rows: list[dict[str, Any]], key: str) -> float:
    return float(sum(float(r.get(key, 0.0)) for r in rows))


def _run_profile(args: argparse.Namespace) -> dict[str, Any]:
    if args.threads > 0:
        os.environ["RAYON_NUM_THREADS"] = str(args.threads)
    ext = _load_ext()
    trainer = ext.NlheCFRTrainer(
        args.num_buckets,
        [],
        args.deals,
        6,
        float("-inf"),
        False,
        str(args.emd_frozen) if args.emd_frozen else None,
    )
    rows = json.loads(trainer.train_profile_json(args.iterations))
    return {
        "config": {
            "iterations": args.iterations,
            "deals": args.deals,
            "num_buckets": args.num_buckets,
            "emd_frozen": str(args.emd_frozen) if args.emd_frozen else None,
            "threads": args.threads,
        },
        "summary": _summarize(rows),
        "rows": rows,
    }


def _build_folded(payload: dict[str, Any]) -> dict[tuple[str, ...], float]:
    rows = payload["rows"]
    folded: dict[tuple[str, ...], float] = {}

    def add(stack: tuple[str, ...], weight: float) -> None:
        if weight > 0:
            folded[stack] = folded.get(stack, 0.0) + weight

    traverse_us = _total(rows, "traverse_ms") * 1000.0
    merge_us = _total(rows, "merge_ms") * 1000.0
    floor_us = _total(rows, "floor_ms") * 1000.0

    add(("train_iteration_profile", "collect_partials_n_profile", "one_traversal_profile"), traverse_us)
    add(("train_iteration_profile", "merge_partials"), merge_us)
    add(("train_iteration_profile", "floor_cum_regret"), floor_us)

    decision = max(_total(rows, "decision_nodes"), 1.0)
    terminal = _total(rows, "terminal_nodes")
    chance = _total(rows, "chance_nodes")
    hero = _total(rows, "hero_nodes")
    opp = _total(rows, "opponent_nodes")
    denom = max(decision + terminal + chance, 1.0)

    add(
        ("train_iteration_profile", "collect_partials_n_profile", "one_traversal_profile", "decision_nodes"),
        traverse_us * decision / denom,
    )
    add(
        ("train_iteration_profile", "collect_partials_n_profile", "one_traversal_profile", "terminal_return"),
        traverse_us * terminal / denom,
    )
    add(
        ("train_iteration_profile", "collect_partials_n_profile", "one_traversal_profile", "sample_chance"),
        traverse_us * chance / denom,
    )

    decision_weight = traverse_us * decision / denom
    node_denom = max(hero + opp, 1.0)
    add(
        (
            "train_iteration_profile",
            "collect_partials_n_profile",
            "one_traversal_profile",
            "decision_nodes",
            "hero_branch_all_legal",
        ),
        decision_weight * hero / node_denom,
    )
    add(
        (
            "train_iteration_profile",
            "collect_partials_n_profile",
            "one_traversal_profile",
            "decision_nodes",
            "opponent_sample_one",
        ),
        decision_weight * opp / node_denom,
    )

    fp_calls = max(_total(rows, "fingerprint_calls"), 1.0)
    emd_queries = _total(rows, "emd_bucket_queries")
    emd_fp_hits = _total(rows, "emd_fp_memo_hits")
    emd_fp_misses = _total(rows, "emd_fp_memo_misses")
    fp_new = _total(rows, "fp_memo_new_entries")
    fp_weight = decision_weight * min(1.0, fp_calls / decision)
    add(
        (
            "train_iteration_profile",
            "collect_partials_n_profile",
            "one_traversal_profile",
            "decision_nodes",
            "fingerprint_fast_depth_memo_emd",
        ),
        fp_weight,
    )
    if emd_queries > 0:
        add(
            (
                "train_iteration_profile",
                "collect_partials_n_profile",
                "one_traversal_profile",
                "decision_nodes",
                "fingerprint_fast_depth_memo_emd",
                "emd_bucket_memoed",
            ),
            fp_weight * emd_queries / fp_calls,
        )
        add(
            (
                "train_iteration_profile",
                "collect_partials_n_profile",
                "one_traversal_profile",
                "decision_nodes",
                "fingerprint_fast_depth_memo_emd",
                "postflop_bucket_digest_memo_hit",
            ),
            fp_weight * emd_fp_hits / fp_calls,
        )
        add(
            (
                "train_iteration_profile",
                "collect_partials_n_profile",
                "one_traversal_profile",
                "decision_nodes",
                "fingerprint_fast_depth_memo_emd",
                "postflop_bucket_digest_blake2b_miss",
            ),
            fp_weight * emd_fp_misses / fp_calls,
        )
    add(
        (
            "train_iteration_profile",
            "collect_partials_n_profile",
            "one_traversal_profile",
            "decision_nodes",
            "fingerprint_fast_depth_memo_emd",
            "new_fingerprint_memo_entries",
        ),
        fp_weight * fp_new / fp_calls,
    )
    return folded


class Node:
    def __init__(self, name: str) -> None:
        self.name = name
        self.value = 0.0
        self.children: dict[str, Node] = {}


def _tree(folded: dict[tuple[str, ...], float]) -> Node:
    root = Node("root")
    for stack, value in folded.items():
        cur = root
        cur.value += value
        for frame in stack:
            cur = cur.children.setdefault(frame, Node(frame))
            cur.value += value
    return root


def _color(name: str, depth: int) -> str:
    h = abs(hash(name)) % 40
    hue = 18 + h if depth % 2 == 0 else 200 + h
    return f"hsl({hue},70%,62%)"


def _esc(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _render_svg(payload: dict[str, Any], folded: dict[tuple[str, ...], float], out: Path) -> None:
    root = _tree(folded)
    width = 1400
    frame_h = 18
    pad_x = 12
    top = 78
    max_depth = max((len(stack) for stack in folded), default=1)
    height = top + (max_depth + 2) * (frame_h + 2) + 45
    plot_w = width - 2 * pad_x
    total = max(root.value, 1.0)
    rows: list[str] = []

    def draw(node: Node, depth: int, x: float, w: float) -> None:
        if depth > 0:
            y = top + (depth - 1) * (frame_h + 2)
            pct = node.value / total * 100.0
            label = node.name
            if w > 90:
                label = f"{label} ({pct:.1f}%)"
            rows.append(
                f'<g><title>{_esc(node.name)} | {node.value / 1000.0:.3f} ms equiv | {pct:.2f}%</title>'
                f'<rect x="{x:.2f}" y="{y}" width="{max(w, 0.1):.2f}" height="{frame_h}" '
                f'rx="2" ry="2" fill="{_color(node.name, depth)}" stroke="#2b2b2b" stroke-width="0.4"/>'
                f'<text x="{x + 4:.2f}" y="{y + 13}" font-size="11" fill="#111">{_esc(label[:90])}</text></g>'
            )
        child_x = x
        for child in sorted(node.children.values(), key=lambda n: n.value, reverse=True):
            cw = w * child.value / max(node.value, 1e-9)
            if cw >= 0.5:
                draw(child, depth + 1, child_x, cw)
            child_x += cw

    draw(root, 0, pad_x, plot_w)
    summary = payload["summary"]
    subtitle = (
        f"K={payload['config']['num_buckets']} deals={payload['config']['deals']} "
        f"iters={payload['config']['iterations']} threads={summary.get('rayon_threads')} "
        f"nodes/s={summary.get('nodes_per_second', 0):.0f} "
        f"emd_fp_hit={summary.get('emd_fp_memo_hit_rate', 0):.3f}"
    )
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<style>
text {{ font-family: Segoe UI, Arial, sans-serif; }}
.title {{ font-size: 20px; font-weight: 700; }}
.sub {{ font-size: 12px; fill: #444; }}
</style>
<rect x="0" y="0" width="{width}" height="{height}" fill="#fafafa"/>
<text x="{pad_x}" y="28" class="title">Native K160 EMD Trainer Flamegraph</text>
<text x="{pad_x}" y="48" class="sub">{_esc(subtitle)}</text>
<text x="{pad_x}" y="66" class="sub">Counter/phase fallback: cargo-flamegraph OS sampling failed locally with NotAnAdmin.</text>
{''.join(rows)}
</svg>
"""
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(svg, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=25)
    parser.add_argument("--deals", type=int, default=64)
    parser.add_argument("--num-buckets", type=int, default=160)
    parser.add_argument("--emd-frozen", type=Path, default=Path("data/abstraction/emd_frozen_flop_turn_160_r400.emdfrzn"))
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("artifacts/emd_k160_native_trainer_flamegraph.svg"))
    parser.add_argument("--json-out", type=Path, default=Path("artifacts/emd_k160_native_trainer_flamegraph_profile.json"))
    args = parser.parse_args()

    payload = _run_profile(args)
    folded = _build_folded(payload)
    _render_svg(payload, folded, args.out)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(f"wrote {args.out}")
    print(f"wrote {args.json_out}")
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
