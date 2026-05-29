#!/usr/bin/env python3
"""Post-SFT held-out KL diagnostic for prod600 (Task 1)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from pokerbot.data.sft_dataloader import SolverAnnotatedParquetDataset
from pokerbot.models.checkpoint_loader import load_pseudo_siamese_policy


def _row_meta(rec: dict) -> dict:
    raw = json.loads(rec["raw_solver_json"])
    board = raw["solver_request"]["board"]
    hero = raw.get("hero_hole") or []
    return {
        "spot_id": raw.get("spot_id"),
        "spot_class": rec["spot_class"],
        "board": board,
        "board_key": "".join(board),
        "hero_hole": hero,
        "row_key": f"{raw.get('spot_id')}|{''.join(board)}|hero={''.join(hero)}",
    }


def compute_per_row_kl(
    model: torch.nn.Module,
    parquet_path: str | Path,
    device: torch.device | str,
) -> list[dict]:
    device = torch.device(device)
    ds = SolverAnnotatedParquetDataset(parquet_path)
    model.eval()
    rows: list[dict] = []
    with torch.no_grad():
        for i in range(len(ds)):
            s = ds[i]
            b = {k: v.unsqueeze(0).to(device) for k, v in s.items()}
            out = model(
                b["weva_ids"],
                b["actions"],
                b["position"],
                b["legal_mask"],
                cards=b["cards"],
            )
            target = b["solver_policy"] * b["legal_mask"].float()
            target = target / target.sum(dim=-1, keepdim=True).clamp(min=1e-8)
            kl = F.kl_div(
                F.log_softmax(out.logits.float(), dim=-1),
                target,
                reduction="batchmean",
            ).item()
            meta = _row_meta(ds.records[i])
            rows.append({"kl": float(kl), **meta})
    return rows


def _summary(rows: list[dict]) -> dict:
    kls = [r["kl"] for r in rows]
    return {
        "n_rows": len(rows),
        "mean_kl": float(np.mean(kls)) if kls else float("inf"),
        "median_kl": float(np.median(kls)) if kls else float("inf"),
        "max_kl": float(np.max(kls)) if kls else float("inf"),
        "min_kl": float(np.min(kls)) if kls else float("inf"),
        "count_kl_lt_0_10": sum(1 for k in kls if k < 0.10),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="checkpoints/policy_sft_prod600.pt")
    p.add_argument("--heldout-vanilla", default="data/solver/heldout_prod600_vanilla.parquet")
    p.add_argument("--heldout-bunching", default="data/solver/heldout_prod600_bunching.parquet")
    p.add_argument("--pre-sft-json", default="logs/prod600_pre_sft_diagnostic.json")
    p.add_argument("--output", default="logs/prod600_post_sft_holdout_kl.json")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    model = load_pseudo_siamese_policy(args.checkpoint, device=args.device)

    vanilla_rows = compute_per_row_kl(model, args.heldout_vanilla, args.device)
    bunching_rows = compute_per_row_kl(model, args.heldout_bunching, args.device)
    combined_rows = vanilla_rows + bunching_rows

    vanilla_stats = _summary(vanilla_rows)
    bunching_stats = _summary(bunching_rows)
    combined_stats = _summary(combined_rows)
    combined_stats["vanilla_rows"] = vanilla_stats["n_rows"]
    combined_stats["bunching_rows"] = bunching_stats["n_rows"]

    pre = json.loads(Path(args.pre_sft_json).read_text(encoding="utf-8"))
    pre_combined = float(pre["heldout_combined_kl"]["mean_kl"])
    pre_vanilla = float(pre["heldout_vanilla_kl"]["mean_kl"])
    pre_bunching = float(pre["heldout_bunching_kl"]["mean_kl"])

    def _pct_change(post: float, pre: float) -> float | None:
        if pre == 0:
            return None
        return (pre - post) / pre * 100.0

    post_combined = combined_stats["mean_kl"]
    combined_drop_pct = _pct_change(post_combined, pre_combined)

    report = {
        "checkpoint": args.checkpoint,
        "heldout_vanilla_kl": vanilla_stats,
        "heldout_bunching_kl": bunching_stats,
        "heldout_combined_kl": combined_stats,
        "per_row_vanilla": sorted(vanilla_rows, key=lambda r: r["kl"], reverse=True),
        "per_row_bunching": sorted(bunching_rows, key=lambda r: r["kl"], reverse=True),
        "pre_sft_comparison": {
            "pre_combined_mean_kl": pre_combined,
            "post_combined_mean_kl": post_combined,
            "combined_drop_pct": combined_drop_pct,
            "pre_vanilla_mean_kl": pre_vanilla,
            "post_vanilla_mean_kl": vanilla_stats["mean_kl"],
            "vanilla_drop_pct": _pct_change(vanilla_stats["mean_kl"], pre_vanilla),
            "pre_bunching_mean_kl": pre_bunching,
            "post_bunching_mean_kl": bunching_stats["mean_kl"],
            "bunching_drop_pct": _pct_change(bunching_stats["mean_kl"], pre_bunching),
        },
        "decision_thresholds": {
            "meaningful_drop_pct": 30.0,
            "marginal_drop_pct": 10.0,
        },
    }

    if combined_drop_pct is None:
        decision = "ambiguous"
    elif combined_drop_pct >= 30.0:
        decision = "proceed_to_task_2_hypothesis_2_weakened"
    elif combined_drop_pct < 10.0:
        decision = "stop_hypothesis_2_prime_suspect"
    else:
        decision = "marginal_proceed_to_task_2"

    report["decision"] = decision
    report["decision_note"] = (
        f"Combined held-out KL dropped {combined_drop_pct:.1f}% "
        f"({pre_combined:.4f} → {post_combined:.4f}). "
        f"Threshold: ≥30% proceed Task 2; <10% stop."
    )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
