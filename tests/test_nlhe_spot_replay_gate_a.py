"""Parquet JAX replay parity and Gate A wiring (no SFT checkpoint required)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from pokerbot.data.sft_dataloader import SolverAnnotatedParquetDataset
from pokerbot.training.nlhe_spot_replay import (
    game_state_from_parquet_record,
    jax_state_from_parquet_record,
    verify_parquet_encode_replay,
)
from pokerbot.training.sd_cfr_nlhe import SDCFRNLHEConfig, SDCFRNLHETrainer
from pokerbot.training.nlhe_gate_a import _row_kl


@pytest.fixture
def vanilla_records():
    ds = SolverAnnotatedParquetDataset("data/solver/heldout_prod600_vanilla.parquet")
    return ds.records


def test_parquet_jax_encode_replay_all_rows(vanilla_records):
    for rec in vanilla_records:
        chk = verify_parquet_encode_replay(rec)
        assert chk["weva_ok"], chk
        assert chk["cards_ok"], chk
        assert chk["actions_ok"], chk


def test_game_state_stack_depth_inference(vanilla_records):
    rec = vanilla_records[0]
    gs, hero = game_state_from_parquet_record(rec)
    raw = json.loads(rec["raw_solver_json"])
    eff = float(raw["solver_request"]["effective_stack_bb"])
    assert abs(gs.players[hero].stack_bb - eff) < 0.01


def test_encode_bundle_from_parquet(vanilla_records):
    from pokerbot.training.nlhe_spot_replay import encode_bundle_from_parquet_record

    rec = vanilla_records[0]
    bundle, _mask = encode_bundle_from_parquet_record(rec)
    assert int(bundle.weva_id) == int(rec["weva_bucket_id"])
    assert np.allclose(bundle.actions, rec["actions"], atol=1e-4)


def test_row_kl_identical_policies():
    mask = np.array([True, True, True] + [False] * 8)
    p = np.array([0.2, 0.5, 0.3] + [0.0] * 8, dtype=np.float64)
    assert _row_kl(p, p, mask) < 1e-6


def test_gate_a_trainer_without_checkpoint(vanilla_records):
    """Untrained random-init trainer runs Gate A path (KL finite)."""
    trainer = SDCFRNLHETrainer(
        SDCFRNLHEConfig(iterations=0, traversals_per_iter=0, sft_checkpoint=None, seed=1)
    )
    rec = vanilla_records[0]
    policy = trainer.eval_strategy_policy_from_parquet(rec)
    mask = np.asarray(rec["legal_mask"], dtype=bool)
    target = np.asarray(rec["solver_policy"], dtype=np.float64)
    kl = _row_kl(policy, target, mask)
    assert np.isfinite(kl)
