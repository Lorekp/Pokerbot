"""Tests for solver-annotated SFT dataloader and combined loss."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from pokerbot.data.bc_parquet import BCParquetDataset, bc_sft_loss, collate_bc_batch
from pokerbot.data.sft_dataloader import (
    LABEL_SOURCE_SOLVER,
    SolverAnnotatedParquetDataset,
    collate_sft_batch,
    weva_sft_loss,
)
from pokerbot.engine.actions import NUM_ACTIONS
from pokerbot.models.policy import PseudoSiamesePolicy


def _make_solver_row(weva_id: int = 3, label: int = 2) -> dict:
    policy = np.zeros(NUM_ACTIONS, dtype=np.float32)
    policy[label] = 0.55
    policy[1] = 0.25
    policy[3] = 0.20
    legal = np.zeros(NUM_ACTIONS, dtype=bool)
    legal[label] = True
    legal[1] = True
    legal[3] = True
    policy = policy * legal
    policy /= policy.sum()
    return {
        "weva_bucket_id": weva_id,
        "actions": np.zeros((6, 4, 12), dtype=np.float32).tolist(),
        "position": [1.0, 0, 0, 0, 0, 0],
        "legal_mask": legal.tolist(),
        "solver_label": label,
        "solver_policy": policy.tolist(),
        "label_source": LABEL_SOURCE_SOLVER,
        "stage": "preflop",
    }


def _make_bc_row(weva_id: int = 3, label: int = 2) -> dict:
    legal = np.zeros(NUM_ACTIONS, dtype=bool)
    legal[label] = True
    policy = np.zeros(NUM_ACTIONS, dtype=np.float32)
    policy[label] = 1.0
    return {
        "weva_bucket_id": weva_id,
        "actions": np.zeros((6, 4, 12), dtype=np.float32).tolist(),
        "position": [1.0, 0, 0, 0, 0, 0],
        "legal_mask": legal.tolist(),
        "bc_label": label,
        "bc_policy": policy.tolist(),
        "label_source": "pokerbench_csv",
        "stage": "preflop",
    }


@pytest.fixture
def solver_parquet(tmp_path: Path) -> Path:
    rows = [_make_solver_row(weva_id=1, label=2), _make_solver_row(weva_id=4, label=1)]
    path = tmp_path / "train.parquet"
    pd.DataFrame(rows).to_parquet(path, index=False)
    return path


@pytest.fixture
def bc_parquet(tmp_path: Path) -> Path:
    rows = [_make_bc_row(weva_id=1, label=2), _make_bc_row(weva_id=4, label=1)]
    path = tmp_path / "bc.parquet"
    pd.DataFrame(rows).to_parquet(path, index=False)
    return path


def test_solver_dataset_requires_label_source(solver_parquet: Path):
    ds = SolverAnnotatedParquetDataset(str(solver_parquet))
    assert len(ds) == 2
    item = ds[0]
    assert item["weva_ids"].dtype == torch.long
    assert item["solver_policy"].shape == (NUM_ACTIONS,)


def test_solver_dataset_rejects_bc_rows(tmp_path: Path):
    row = _make_bc_row()
    row["solver_label"] = row.pop("bc_label")
    row["solver_policy"] = row.pop("bc_policy")
    path = tmp_path / "bad.parquet"
    pd.DataFrame([row]).to_parquet(path, index=False)
    with pytest.raises(ValueError, match="pokerbench_csv"):
        SolverAnnotatedParquetDataset(str(path))


def test_solver_dataset_rejects_missing_label_source(tmp_path: Path):
    row = _make_solver_row()
    del row["label_source"]
    path = tmp_path / "bad.parquet"
    pd.DataFrame([row]).to_parquet(path, index=False)
    with pytest.raises(ValueError, match="label_source"):
        SolverAnnotatedParquetDataset(str(path))


def test_bc_dataset_loads(bc_parquet: Path):
    ds = BCParquetDataset(str(bc_parquet))
    assert len(ds) == 2
    assert ds[0]["bc_label"].dtype == torch.long


def test_bc_dataset_rejects_solver_columns(tmp_path: Path):
    row = _make_solver_row()
    path = tmp_path / "bad.parquet"
    pd.DataFrame([row]).to_parquet(path, index=False)
    with pytest.raises(ValueError, match="solver"):
        BCParquetDataset(str(path))


def test_weva_sft_loss_finite():
    model = PseudoSiamesePolicy(hidden=32, num_weva_buckets=16)
    row = _make_solver_row(1, 2)
    batch = collate_sft_batch(
        [
            {
                "weva_ids": torch.tensor(row["weva_bucket_id"], dtype=torch.long),
                "actions": torch.tensor(row["actions"], dtype=torch.float32),
                "position": torch.tensor(row["position"], dtype=torch.float32),
                "legal_mask": torch.tensor(row["legal_mask"], dtype=torch.bool),
                "solver_label": torch.tensor(row["solver_label"], dtype=torch.long),
                "solver_policy": torch.tensor(row["solver_policy"], dtype=torch.float32),
            }
        ]
    )
    out = model(batch["weva_ids"], batch["actions"], batch["position"], batch["legal_mask"])
    loss, stats = weva_sft_loss(
        out,
        batch["solver_label"],
        batch["solver_policy"],
        batch["legal_mask"],
        beta=0.3,
    )
    assert torch.isfinite(loss)
    assert stats["ce_hard"] >= 0


def test_bc_sft_loss_finite():
    model = PseudoSiamesePolicy(hidden=32, num_weva_buckets=16)
    row = _make_bc_row(1, 2)
    batch = collate_bc_batch(
        [
            {
                "weva_ids": torch.tensor(row["weva_bucket_id"], dtype=torch.long),
                "actions": torch.tensor(row["actions"], dtype=torch.float32),
                "position": torch.tensor(row["position"], dtype=torch.float32),
                "legal_mask": torch.tensor(row["legal_mask"], dtype=torch.bool),
                "bc_label": torch.tensor(row["bc_label"], dtype=torch.long),
            }
        ]
    )
    out = model(batch["weva_ids"], batch["actions"], batch["position"], batch["legal_mask"])
    loss, stats = bc_sft_loss(out, batch["bc_label"], batch["legal_mask"])
    assert torch.isfinite(loss)
    assert stats["ce_bc"] >= 0
