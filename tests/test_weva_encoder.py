"""Tests for WEVA encoder and index building."""

import json
from pathlib import Path

import numpy as np
import pytest

from pokerbot.engine.cards import Card, Rank, Suit
from pokerbot.encoding.weva_cfr import (
    RolloutConfig,
    TabularCFRPlus,
    build_action_evs_matrix,
    run_weva_cfr_pipeline,
)
from pokerbot.encoding.weva_encoder import (
    RootInfoSet,
    WEVABuilder,
    WEVAMapper,
    WEVAIndex,
    WEVA_WARMUP_ITERATIONS,
    enumerate_hole_combos,
    expand_root_info_sets,
    is_production_weva_index,
    weva_artifacts_match_config,
)


def test_enumerate_hole_combos_count():
    combos = enumerate_hole_combos()
    assert len(combos) == 1326


def test_tabular_cfr_plus_runs_w10():
    roots = expand_root_info_sets("hole_only")[:8]
    cfg = RolloutConfig(num_rollouts=16, warmup_iterations=WEVA_WARMUP_ITERATIONS, seed=1, cfr_seed=1)
    action_evs, _ = build_action_evs_matrix(roots, cfg)
    cfr = TabularCFRPlus(action_evs, warmup_iterations=WEVA_WARMUP_ITERATIONS, seed=1)
    cfr.run()
    node_evs = cfr.node_evs_under_avg_strategy()
    assert node_evs.shape == (8, cfg.num_depths)


def test_weva_cfr_pipeline_feature_dim():
    roots = expand_root_info_sets("hole_only")[:6]
    cfg = RolloutConfig(num_rollouts=12, warmup_iterations=WEVA_WARMUP_ITERATIONS, seed=2, cfr_seed=2)
    features, calib, _ = run_weva_cfr_pipeline(
        roots, cfg, gamma=0.85, max_nodes=4, feature_mode="hole_only"
    )
    assert features.shape == (6, 4)
    assert "mean_rollout_stderr" in calib


def test_aa_beats_72o_rollout_ordering():
    aa = RootInfoSet(hole=(Card(Rank.ACE, Suit.SPADES), Card(Rank.ACE, Suit.HEARTS)))
    s72 = RootInfoSet(hole=(Card(Rank.SEVEN, Suit.SPADES), Card(Rank.TWO, Suit.HEARTS)))
    cfg = RolloutConfig(num_rollouts=64, seed=42, cfr_seed=42)
    roots = [aa, s72]
    matrix, calib = build_action_evs_matrix(roots, cfg)
    assert matrix[0, 0, 1] > matrix[1, 0, 1]
    assert calib["aa_vs_72o_call_ordering"] == 1.0

    features, _, _ = run_weva_cfr_pipeline(
        roots, cfg, gamma=0.85, max_nodes=4, feature_mode="hole_only"
    )
    index = WEVAIndex(k=2)
    index.fit(features)
    bucket_ids = index.assign(features)
    assert bucket_ids[0] != bucket_ids[1]


def test_weva_index_build_and_lookup(tmp_path: Path):
    builder = WEVABuilder(
        k=8,
        feature_mode="hole_only",
        num_rollouts=16,
        warmup_iterations=WEVA_WARMUP_ITERATIONS,
        max_roots=32,
    )
    builder.save_artifacts(tmp_path)
    assert (tmp_path / "features.npy").exists()
    mapper = WEVAMapper.from_directory(tmp_path)
    combos = enumerate_hole_combos()
    bid = mapper.lookup(combos[0])
    assert 0 <= bid < 8


def test_weva_index_uses_faiss(tmp_path: Path):
    import faiss

    WEVABuilder(k=8, feature_mode="hole_only", num_rollouts=8, max_roots=24).save_artifacts(tmp_path)
    index = WEVAIndex.load(tmp_path)
    assert index._faiss_index is not None
    assert faiss is not None


def test_weva_mapper_uses_cached_features(tmp_path: Path):
    builder = WEVABuilder(k=5, feature_mode="hole_only", num_rollouts=8, max_roots=20)
    builder.save_artifacts(tmp_path)
    mapper = WEVAMapper.from_directory(tmp_path)
    assert mapper.features is not None
    hole = enumerate_hole_combos()[10]
    b1 = mapper.lookup(hole)
    b2 = mapper.lookup(hole)
    assert b1 == b2


def test_weva_build_reproducible_seed(tmp_path: Path):
    b1 = WEVABuilder(
        k=6, feature_mode="hole_only", num_rollouts=24, cfr_seed=7, rollout_seed=7, max_roots=12
    )
    b2 = WEVABuilder(
        k=6, feature_mode="hole_only", num_rollouts=24, cfr_seed=7, rollout_seed=7, max_roots=12
    )
    f1 = b1.build()[3]
    f2 = b2.build()[3]
    np.testing.assert_allclose(f1[:4], f2[:4], rtol=0, atol=1e-5)


def test_weva_build_fingerprint_invalidation(tmp_path: Path):
    builder = WEVABuilder(k=5, feature_mode="hole_only", num_rollouts=8, max_roots=16)
    builder.save_artifacts(tmp_path)
    meta = json.loads((tmp_path / "meta.json").read_text(encoding="utf-8"))
    assert meta["build_fingerprint"] == builder.build_fingerprint()
    assert meta["build_scope"] == "dev"
    assert weva_artifacts_match_config(tmp_path, builder.build_config_dict())

    stale = WEVABuilder(k=5, feature_mode="hole_only", num_rollouts=16, max_roots=16)
    assert not weva_artifacts_match_config(tmp_path, stale.build_config_dict())
    stale.save_artifacts(tmp_path)
    meta2 = json.loads((tmp_path / "meta.json").read_text(encoding="utf-8"))
    assert meta2["build_fingerprint"] == stale.build_fingerprint()
    assert meta2["num_rollouts"] == 16


def test_production_weva_scope(tmp_path: Path):
    dev = WEVABuilder(k=10, feature_mode="hole_only", num_rollouts=8, max_roots=32)
    dev.save_artifacts(tmp_path / "dev")
    assert not is_production_weva_index(tmp_path / "dev")

    prod = WEVABuilder(k=10, feature_mode="hole_only", num_rollouts=8, max_roots=None)
    assert prod.build_scope() == "production"
