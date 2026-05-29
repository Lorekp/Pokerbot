"""Precondition: SFT Parquet rows must have unique policy inputs per target."""

from __future__ import annotations

from collections import defaultdict
from glob import glob
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from pokerbot.data.sft_dataloader import SolverAnnotatedParquetDataset
from pokerbot.encoding.spot_scope import fuse_position_scope
from pokerbot.engine.actions import ACTION_VOCAB, NUM_ACTIONS

DEFAULT_TARGET_KL_THRESHOLD = 0.05


def _input_fingerprint(rec: dict[str, Any]) -> tuple:
    return (
        int(rec["weva_bucket_id"]),
        np.asarray(rec["cards"], dtype=np.float32).tobytes(),
        np.asarray(rec["actions"], dtype=np.float32).tobytes(),
        np.asarray(rec["position"], dtype=np.float32).tobytes(),
        np.asarray(rec["spot_scope"], dtype=np.float32).tobytes(),
        np.asarray(rec["legal_mask"], dtype=bool).tobytes(),
    )


def _target_kl(p1: np.ndarray, p2: np.ndarray) -> float:
    a = torch.tensor(p1, dtype=torch.float64).clamp(min=1e-8)
    b = torch.tensor(p2, dtype=torch.float64).clamp(min=1e-8)
    a = a / a.sum()
    b = b / b.sum()
    return float(
        F.kl_div(b.log(), a, reduction="batchmean").item()
        + F.kl_div(a.log(), b, reduction="batchmean").item()
    ) / 2.0


def check_sft_input_collisions(
    parquet_glob: str | Path,
    *,
    target_kl_threshold: float = DEFAULT_TARGET_KL_THRESHOLD,
) -> dict[str, Any]:
    """
    Detect rows sharing identical policy inputs but meaningfully different targets.

    Returns summary dict with ``passed`` True when no conflicting collision groups exist.
    """
    pattern = str(parquet_glob)
    paths = sorted(glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No Parquet files match {pattern!r}")

    ds = SolverAnnotatedParquetDataset(parquet_glob)
    groups: dict[tuple, list[dict[str, Any]]] = defaultdict(list)

    for i in range(len(ds)):
        rec = ds.records[i]
        fp = _input_fingerprint(rec)
        policy = np.asarray(rec["solver_policy"], dtype=np.float64)
        groups[fp].append(
            {
                "row": i,
                "spot_class": rec.get("spot_class"),
                "policy": policy,
                "top": ACTION_VOCAB[int(np.argmax(policy))],
            }
        )

    collision_groups = {k: v for k, v in groups.items() if len(v) > 1}
    conflicting_groups: list[dict[str, Any]] = []

    for members in collision_groups.values():
        max_pair_kl = 0.0
        conflicting_pairs: list[dict[str, Any]] = []
        for a, b in combinations(members, 2):
            d = _target_kl(a["policy"], b["policy"])
            max_pair_kl = max(max_pair_kl, d)
            if d > target_kl_threshold:
                conflicting_pairs.append(
                    {
                        "rows": (a["row"], b["row"]),
                        "target_kl": d,
                        "tops": (a["top"], b["top"]),
                    }
                )
        if conflicting_pairs:
            conflicting_groups.append(
                {
                    "size": len(members),
                    "rows": [m["row"] for m in members],
                    "spot_classes": sorted({str(m["spot_class"]) for m in members}),
                    "max_pair_target_kl": max_pair_kl,
                    "conflicting_pairs": conflicting_pairs,
                }
            )

    n_rows = len(ds)
    n_unique = len(groups)
    n_conflicting_rows = sum(len(g["rows"]) for g in conflicting_groups)

    return {
        "passed": len(conflicting_groups) == 0,
        "n_rows": n_rows,
        "n_unique_inputs": n_unique,
        "n_collision_groups": len(collision_groups),
        "n_conflicting_groups": len(conflicting_groups),
        "n_conflicting_rows": n_conflicting_rows,
        "target_kl_threshold": target_kl_threshold,
        "conflicting_groups": conflicting_groups,
        "input_shape": {
            "cards": list(np.asarray(ds.records[0]["cards"]).shape),
            "actions": list(np.asarray(ds.records[0]["actions"]).shape),
            "position_base": list(np.asarray(ds.records[0]["position"]).shape),
            "spot_scope": list(np.asarray(ds.records[0]["spot_scope"]).shape),
            "policy_position": list(
                fuse_position_scope(
                    np.asarray(ds.records[0]["position"]),
                    np.asarray(ds.records[0]["spot_scope"]),
                ).shape
            ),
            "legal_mask": NUM_ACTIONS,
        },
    }


def require_no_sft_input_collisions(
    parquet_glob: str | Path,
    *,
    target_kl_threshold: float = DEFAULT_TARGET_KL_THRESHOLD,
) -> dict[str, Any]:
    """Run collision check; raise SystemExit(1) on conflicting input groups."""
    summary = check_sft_input_collisions(
        parquet_glob, target_kl_threshold=target_kl_threshold
    )
    if not summary["passed"]:
        lines = [
            "SFT input collision precondition FAILED:",
            f"  rows={summary['n_rows']} unique_inputs={summary['n_unique_inputs']} "
            f"conflicting_groups={summary['n_conflicting_groups']} "
            f"(threshold target-target KL > {target_kl_threshold})",
        ]
        for g in summary["conflicting_groups"][:5]:
            lines.append(
                f"  group rows={g['rows']} spot_classes={g['spot_classes']} "
                f"max_kl={g['max_pair_target_kl']:.4f}"
            )
        if len(summary["conflicting_groups"]) > 5:
            lines.append(f"  ... and {len(summary['conflicting_groups']) - 5} more")
        raise SystemExit("\n".join(lines))
    return summary
