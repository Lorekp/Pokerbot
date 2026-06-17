"""
Gate: the cloud data path (serialize -> object store -> deserialize -> merge) must be lossless
and bit-identical to single-threaded. The in-process merge is already validated; this checks the
round-trip the cloud adds does not corrupt fingerprint keys, numpy dtypes, or encode-bundle arrays.
"""

from __future__ import annotations

import numpy as np

from pokerbot.training.sd_cfr_nlhe import SDCFRNLHEConfig, SDCFRNLHETrainer
from pokerbot.training.sd_cfr_nlhe_cloud import (
    LocalDirObjectStore,
    deserialize_shard,
    run_iteration_via_store,
    serialize_shard,
    shard_key,
)
from pokerbot.training.sd_cfr_nlhe_distributed import compare_trainer_states


def _cfg() -> SDCFRNLHEConfig:
    return SDCFRNLHEConfig(
        iterations=1,
        traversals_per_iter=4,
        deal_sweeps=1,
        train_batch_size=64,
        sft_checkpoint=None,
        seed=42,
        max_traverse_depth=16,
    )


def test_shard_serialization_roundtrip_lossless() -> None:
    """A TraversalShardResult with real bundle + numpy deltas must survive serialize/deserialize."""
    from pokerbot.engine.jax_weva_encode import NLHEEncodeBundle
    from pokerbot.encoding.spot_scope import POLICY_POSITION_BASE_DIM, POLICY_POSITION_DIM
    from pokerbot.training.sd_cfr_nlhe import NLHE_NUM_ACTIONS, _NLHERegretVisit
    from pokerbot.training.sd_cfr_nlhe_distributed import TraversalShardResult

    mask = np.array([True, True, True] + [False] * (NLHE_NUM_ACTIONS - 3), dtype=bool)
    bundle = NLHEEncodeBundle(
        weva_id=7,
        cards=np.arange(9 * 4 * 13, dtype=np.float32),
        actions=np.arange(6 * 4 * 12, dtype=np.float32),
        position=np.zeros(POLICY_POSITION_BASE_DIM, dtype=np.float32),
        spot_scope=np.zeros(4, dtype=np.float32),
        policy_position=np.zeros(POLICY_POSITION_DIM, dtype=np.float32),
    )
    key = bundle.fingerprint()
    visit = _NLHERegretVisit(
        opp_reach=1.5,
        util=np.array([1.0, -2.0, 3.0] + [0.0] * (NLHE_NUM_ACTIONS - 3), dtype=np.float64),
        sigma=np.ones(NLHE_NUM_ACTIONS) / NLHE_NUM_ACTIONS,
        mask=mask,
        bundle=bundle,
        opponent_weva_bucket=3,
    )
    result = TraversalShardResult(
        sweep_trav_pairs=[(0, 1)],
        cum_regret_delta={key: np.array([0.5, -0.25, 0.125] + [0.0] * (NLHE_NUM_ACTIONS - 3), dtype=np.float64)},
        strat_sum_delta={key: np.full(NLHE_NUM_ACTIONS, 0.3, dtype=np.float64)},
        pending_regret={key: [visit]},
    )

    back = deserialize_shard(serialize_shard([result]))
    assert len(back) == 1
    r2 = back[0]
    assert r2.sweep_trav_pairs == [(0, 1)]
    assert set(r2.cum_regret_delta) == {key}
    assert np.array_equal(r2.cum_regret_delta[key], result.cum_regret_delta[key])
    assert r2.cum_regret_delta[key].dtype == np.float64
    assert np.array_equal(r2.strat_sum_delta[key], result.strat_sum_delta[key])
    v2 = r2.pending_regret[key][0]
    assert v2.opp_reach == 1.5
    assert v2.opponent_weva_bucket == 3
    assert np.array_equal(v2.util, visit.util)
    assert np.array_equal(v2.mask, visit.mask)
    assert v2.bundle.weva_id == 7
    assert np.array_equal(v2.bundle.cards, bundle.cards)
    assert v2.bundle.fingerprint() == key  # key reconstructs identically


def test_run_iteration_via_store_bit_identical(tmp_path) -> None:
    """A full iteration routed through the object store equals single-threaded, bit for bit."""
    cfg = _cfg()
    ref = SDCFRNLHETrainer(cfg)
    np.random.seed(cfg.seed)
    loss_ref = ref.train_iteration(1.0)

    dist = SDCFRNLHETrainer(cfg)
    np.random.seed(cfg.seed)
    store = LocalDirObjectStore(tmp_path / "store")
    loss_store = run_iteration_via_store(dist, 1.0, num_shards=3, store=store)

    # Shard artifacts actually landed in the store.
    assert store.exists(shard_key(1, 0))
    assert store.exists(shard_key(1, 2))
    assert store.list_keys("iter_0001/")

    assert abs(loss_ref - loss_store) < 1e-5
    cmp = compare_trainer_states(ref, dist)
    assert cmp["cum_regret_bit_identical"], cmp
    assert cmp["strat_sum_bit_identical"], cmp
    assert cmp["params_match"], cmp


def test_store_path_survives_induced_cloud_failures(tmp_path) -> None:
    """Kill→reclaim + out-of-order gather + duplicate delivery, end-to-end through the store,
    must stay bit-identical to single-threaded."""
    from pokerbot.training.sd_cfr_nlhe_cloud import run_iteration_via_store_coordinated

    cfg = _cfg()
    ref = SDCFRNLHETrainer(cfg)
    np.random.seed(cfg.seed)
    loss_ref = ref.train_iteration(1.0)

    dist = SDCFRNLHETrainer(cfg)
    np.random.seed(cfg.seed)
    store = LocalDirObjectStore(tmp_path / "store")
    loss_dist = run_iteration_via_store_coordinated(
        dist,
        1.0,
        num_shards=4,
        store=store,
        killed_shard_ids=frozenset({1}),       # shard 1 dies, then is reclaimed
        shuffle_gather=True,                    # coordinator reads artifacts out of order
        duplicate_shard_ids=frozenset({0}),     # shard 0 delivered twice
    )

    assert abs(loss_ref - loss_dist) < 1e-5
    cmp = compare_trainer_states(ref, dist)
    assert cmp["cum_regret_bit_identical"], cmp
    assert cmp["strat_sum_bit_identical"], cmp
    assert cmp["params_match"], cmp


def test_convergence_via_store_resume_from_store_bit_identical(tmp_path) -> None:
    """Coordinator loop: a run interrupted after iter 1 and resumed from its store checkpoint
    reaches the same iter-2 state, bit-identical, as an uninterrupted store run."""
    from pokerbot.training.sd_cfr_nlhe_cloud import (
        checkpoint_from_store,
        latest_checkpoint_iteration,
        run_convergence_via_store,
    )

    cfg = _cfg()
    uninterrupted = SDCFRNLHETrainer(cfg)
    store_a = LocalDirObjectStore(tmp_path / "a")
    run_convergence_via_store(uninterrupted, num_iters=2, num_shards=3, store=store_a)

    # Interrupt after iter 1 (its barrier checkpoint is in store_b), then resume on a fresh trainer.
    interrupted = SDCFRNLHETrainer(cfg)
    store_b = LocalDirObjectStore(tmp_path / "b")
    run_convergence_via_store(interrupted, num_iters=1, num_shards=3, store=store_b)
    assert latest_checkpoint_iteration(store_b) == 1

    resumed = SDCFRNLHETrainer(cfg)
    assert checkpoint_from_store(resumed, store_b, 1) == 1
    assert resumed._completed_iterations == 1
    run_convergence_via_store(resumed, num_iters=2, num_shards=3, store=store_b, start_iter=2)

    cmp = compare_trainer_states(uninterrupted, resumed)
    assert cmp["cum_regret_bit_identical"], cmp
    assert cmp["strat_sum_bit_identical"], cmp
    assert cmp["params_match"], cmp
