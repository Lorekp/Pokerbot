"""Behavioral-cloning Parquet dataset (PokerBench CSV — not solver-verified)."""

from __future__ import annotations

from glob import glob
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from pokerbot.engine.actions import NUM_ACTIONS
from pokerbot.models.policy import PolicyOutput

LABEL_SOURCE_BC = "pokerbench_csv"

BC_REQUIRED_COLUMNS = (
    "weva_bucket_id",
    "actions",
    "position",
    "legal_mask",
    "bc_label",
    "bc_policy",
    "label_source",
)


def _nested_to_array(value: Any, dtype: type) -> np.ndarray:
    if isinstance(value, np.ndarray):
        if value.dtype == object:
            return np.stack([_nested_to_array(x, dtype) for x in value], axis=0)
        return value.astype(dtype)
    if isinstance(value, list):
        if value and isinstance(value[0], (list, np.ndarray)):
            return np.stack([_nested_to_array(x, dtype) for x in value], axis=0)
        return np.array(value, dtype=dtype)
    return np.array(value, dtype=dtype)


def _validate_bc_row(row: dict[str, Any], source: str) -> None:
    for col in BC_REQUIRED_COLUMNS:
        if col not in row or row[col] is None:
            raise ValueError(f"Missing required BC column {col!r} in {source}")
    if row["label_source"] != LABEL_SOURCE_BC:
        raise ValueError(
            f"Invalid label_source {row['label_source']!r} in {source}; "
            f"expected {LABEL_SOURCE_BC!r}"
        )
    if "solver_label" in row or "solver_policy" in row:
        raise ValueError(
            f"BC Parquet must not contain solver_* columns (found in {source})"
        )


def _normalize_bc_record(rec: dict[str, Any]) -> dict[str, Any]:
    legal = _nested_to_array(rec["legal_mask"], bool)
    rec = dict(rec)
    rec["actions"] = _nested_to_array(rec["actions"], np.float32)
    rec["position"] = _nested_to_array(rec["position"], np.float32)
    rec["legal_mask"] = legal
    policy = _nested_to_array(rec["bc_policy"], np.float32).reshape(-1)
    if policy.shape[0] != NUM_ACTIONS:
        raise ValueError(f"bc_policy must have length {NUM_ACTIONS}, got {policy.shape[0]}")
    policy = policy * legal.astype(np.float32)
    total = policy.sum()
    if total <= 0:
        policy = legal.astype(np.float32) / max(legal.sum(), 1)
    else:
        policy = policy / total
    rec["bc_policy"] = policy
    rec["weva_bucket_id"] = int(rec["weva_bucket_id"])
    rec["bc_label"] = int(rec["bc_label"])
    rec["label_source"] = str(rec["label_source"])
    return rec


def make_bc_policy_from_label(label: int, legal_mask: np.ndarray) -> np.ndarray:
    """One-hot BC policy from CSV label (smoke-test fallback only)."""
    legal = legal_mask.astype(bool)
    policy = np.zeros(NUM_ACTIONS, dtype=np.float32)
    if 0 <= label < NUM_ACTIONS and legal[label]:
        policy[label] = 1.0
    elif legal.any():
        policy[legal] = 1.0 / legal.sum()
    else:
        policy[:] = 1.0 / NUM_ACTIONS
    return policy


class BCParquetDataset(Dataset):
    """PokerBench-derived BC loader — explicitly not GTO-verified."""

    def __init__(self, parquet_glob: str | Path) -> None:
        pattern = str(parquet_glob)
        paths = sorted(glob(pattern))
        if not paths:
            raise FileNotFoundError(f"No BC Parquet files match {pattern!r}")

        try:
            import pandas as pd
        except ImportError as exc:
            raise ImportError("pandas required for BCParquetDataset") from exc

        frames = [pd.read_parquet(p) for p in paths]
        df = pd.concat(frames, ignore_index=True)
        self.records: list[dict[str, Any]] = []
        for i, row in df.iterrows():
            raw = row.to_dict()
            if "solver_label" in raw or "solver_policy" in raw:
                raise ValueError(
                    f"BC Parquet must not contain solver_* columns (found in {paths[0]} row {i})"
                )
            rec = _normalize_bc_record(raw)
            _validate_bc_row(rec, source=f"{paths[0]} row {i}")
            self.records.append(rec)

        if not self.records:
            raise ValueError(f"No valid BC rows in Parquet files matching {pattern!r}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        r = self.records[idx]
        return {
            "weva_ids": torch.tensor(r["weva_bucket_id"], dtype=torch.long),
            "actions": torch.from_numpy(r["actions"]),
            "position": torch.from_numpy(r["position"]),
            "legal_mask": torch.from_numpy(r["legal_mask"]),
            "bc_label": torch.tensor(r["bc_label"], dtype=torch.long),
        }


def collate_bc_batch(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {k: torch.stack([b[k] for b in batch]) for k in batch[0]}


def bc_sft_loss(
    out: PolicyOutput,
    bc_label: torch.Tensor,
    legal_mask: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Plain cross-entropy on PokerBench CSV labels (not solver-verified)."""
    ce = F.cross_entropy(out.logits, bc_label)
    with torch.no_grad():
        pred = out.logits.argmax(dim=-1)
        acc = (pred == bc_label).float().mean().item()
    return ce, {"ce_bc": float(ce.item()), "bc_acc": acc}


def make_bc_dataloader(
    parquet_glob: str | Path,
    batch_size: int = 64,
    shuffle: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    ds = BCParquetDataset(parquet_glob)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_bc_batch,
        num_workers=num_workers,
    )
