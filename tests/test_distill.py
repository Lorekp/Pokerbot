"""Tests for distillation pipeline (no LLM required)."""

import json
from pathlib import Path

import numpy as np
import torch

from pokerbot.training.distill_cache import cache_to_tensors


def test_cache_to_tensors():
    records = [
        {
            "cards": np.zeros((9, 4, 13)).tolist(),
            "actions": np.zeros((6, 4, 12)).tolist(),
            "position": [1, 0, 0, 0, 0, 0],
            "legal_mask": [True] * 11,
            "gto_label": 1,
            "llm_label": 2,
            "agreement": False,
        }
    ]
    t = cache_to_tensors(records)
    assert t["gto_label"].shape == (1,)
    assert t["cards"].shape[0] == 1


def test_distill_dataset(tmp_path: Path):
    from pokerbot.training.distill import DistillDataset

    rec = {
        "cards": np.zeros((9, 4, 13)).tolist(),
        "actions": np.zeros((6, 4, 12)).tolist(),
        "position": [0, 0, 1, 0, 0, 0],
        "legal_mask": [True] * 11,
        "gto_label": 0,
        "llm_label": 1,
        "agreement": False,
    }
    ds = DistillDataset([rec], disagreement_only=True)
    assert len(ds) == 1
    item = ds[0]
    assert item["gto_label"].item() == 0
    assert item["weva_ids"].dtype == torch.long
