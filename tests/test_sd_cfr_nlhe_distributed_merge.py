"""
Gate: distributed traversal merge must match single-threaded trainer (ground truth).

Comparison target is single-threaded accumulation — NOT inter-worker agreement.
"""

from __future__ import annotations

import numpy as np
import pytest

from pokerbot.training.sd_cfr_nlhe import SDCFRNLHEConfig, SDCFRNLHETrainer
from pokerbot.training.sd_cfr_nlhe_distributed import (
    compare_trainer_states,
    train_iteration_sharded,
)


def _cfg() -> SDCFRNLHEConfig:
    return SDCFRNLHEConfig(
        iterations=1,
        traversals_per_iter=2,
        deal_sweeps=1,
        train_batch_size=64,
        sft_checkpoint=None,
        seed=42,
        max_traverse_depth=16,
    )


@pytest.mark.slow
def test_merge_matches_single_threaded_two_workers() -> None:
    """N iters: sharded 2-worker merge vs single-threaded — params, regrets, strat_sum."""
    cfg = _cfg()
    n_iters = 2
    ref = SDCFRNLHETrainer(cfg)
    dist = SDCFRNLHETrainer(cfg)

    for i in range(n_iters):
        w = float(i + 1)
        np.random.seed(cfg.seed + i)
        loss_ref = ref.train_iteration(w)
        np.random.seed(cfg.seed + i)
        loss_dist = train_iteration_sharded(dist, w, num_shards=2)

        assert loss_ref == pytest.approx(loss_dist, rel=0, abs=1e-5)
        report = compare_trainer_states(ref, dist)
        assert report["params_match"], report
        assert report["cum_regret_match"], report
        assert report["strat_sum_match"], report
        assert report["reservoir_len_match"]
        assert report["reservoir_insert_match"]


def test_merge_pending_regret_belief_weight_not_per_shard_avg() -> None:
    """Two worlds with different opp_reach must merge to reach-weighted target, not mean."""
    from collections import defaultdict

    from pokerbot.engine.jax_weva_encode import NLHEEncodeBundle
    from pokerbot.encoding.spot_scope import POLICY_POSITION_DIM
    from pokerbot.encoding.spot_scope import POLICY_POSITION_BASE_DIM
    from pokerbot.training.sd_cfr_nlhe import _NLHERegretVisit, NLHE_NUM_ACTIONS
    from pokerbot.training.sd_cfr_nlhe_distributed import (
        TraversalShardResult,
        iteration_traversal_plan,
        merge_traversal_shards,
    )

    cfg = SDCFRNLHEConfig(iterations=1, traversals_per_iter=2, deal_sweeps=1, sft_checkpoint=None)
    trainer = SDCFRNLHETrainer(cfg)
    trainer._begin_iteration_regret()
    mask = np.array([True, True, True] + [False] * (NLHE_NUM_ACTIONS - 3), dtype=bool)
    bundle = NLHEEncodeBundle(
        weva_id=0,
        cards=np.zeros(9 * 4 * 13, dtype=np.float32),
        actions=np.zeros(6 * 4 * 12, dtype=np.float32),
        position=np.zeros(POLICY_POSITION_BASE_DIM, dtype=np.float32),
        spot_scope=np.zeros(4, dtype=np.float32),
        policy_position=np.zeros(POLICY_POSITION_DIM, dtype=np.float32),
    )
    key = bundle.fingerprint()
    v_hi = _NLHERegretVisit(
        opp_reach=2.0,
        util=np.zeros(NLHE_NUM_ACTIONS, dtype=np.float64),
        sigma=np.ones(NLHE_NUM_ACTIONS) / NLHE_NUM_ACTIONS,
        mask=mask,
        bundle=bundle,
        opponent_weva_bucket=0,
    )
    v_hi.util[1] = 5.0
    v_lo = _NLHERegretVisit(
        opp_reach=1.0,
        util=np.zeros(NLHE_NUM_ACTIONS, dtype=np.float64),
        sigma=np.ones(NLHE_NUM_ACTIONS) / NLHE_NUM_ACTIONS,
        mask=mask,
        bundle=bundle,
        opponent_weva_bucket=1,
    )
    v_lo.util[1] = -5.0
    plan = iteration_traversal_plan(cfg, 1.0)
    u0 = TraversalShardResult(
        sweep_trav_pairs=[(0, 0)],
        pending_regret={key: [v_hi]},
    )
    u1 = TraversalShardResult(
        sweep_trav_pairs=[(0, 1)],
        pending_regret={key: [v_lo]},
    )
    merge_traversal_shards(trainer, [u0, u1], plan)
    trainer._iter_samples = []
    trainer._flush_pending_regret()
    inst = trainer._iter_samples[0][1]
    assert float(inst[1]) > float(inst[0])


def test_merge_idempotent_under_duplicate_delivery() -> None:
    """At-least-once cloud delivery (slow worker + reclaim) must not double-count regrets."""
    from pokerbot.training.sd_cfr_nlhe import NLHE_NUM_ACTIONS
    from pokerbot.training.sd_cfr_nlhe_distributed import (
        TraversalShardResult,
        iteration_traversal_plan,
        merge_traversal_shards,
    )

    cfg = SDCFRNLHEConfig(iterations=1, traversals_per_iter=2, deal_sweeps=1, sft_checkpoint=None)
    plan = iteration_traversal_plan(cfg, 1.0)
    pairs = list(plan.pairs)
    assert len(pairs) >= 2, pairs
    key = ("info", 0)

    def _unit(pair: tuple[int, int], val: float) -> TraversalShardResult:
        return TraversalShardResult(
            sweep_trav_pairs=[pair],
            cum_regret_delta={key: np.full(NLHE_NUM_ACTIONS, val, dtype=np.float64)},
            strat_sum_delta={key: np.full(NLHE_NUM_ACTIONS, val, dtype=np.float64)},
        )

    once = SDCFRNLHETrainer(cfg)
    once._begin_iteration_regret()
    merge_traversal_shards(once, [_unit(pairs[0], 1.0), _unit(pairs[1], 2.0)], plan)

    twice = SDCFRNLHETrainer(cfg)
    twice._begin_iteration_regret()
    # Each pair delivered twice as distinct result objects (original + reclaim).
    merge_traversal_shards(
        twice,
        [_unit(pairs[0], 1.0), _unit(pairs[1], 2.0), _unit(pairs[0], 1.0), _unit(pairs[1], 2.0)],
        plan,
    )

    assert set(once._cum_regret) == set(twice._cum_regret)
    for k in once._cum_regret:
        assert np.array_equal(once._cum_regret[k], twice._cum_regret[k]), k
        assert np.array_equal(once._strat_sum[k], twice._strat_sum[k]), k
    # And the merged value is the single-delivery sum (1.0 + 2.0), not doubled.
    assert float(twice._cum_regret[key][0]) == 3.0


def test_checkpoint_completed_iterations_autoset(tmp_path) -> None:
    """load_nlhe_checkpoint must SET _completed_iterations on the trainer, not only return it."""
    from pokerbot.training.sd_cfr_nlhe import load_nlhe_checkpoint, save_nlhe_checkpoint

    cfg = _cfg()
    t = SDCFRNLHETrainer(cfg)
    t._completed_iterations = 7
    ckpt = tmp_path / "ckpt.pkl"
    save_nlhe_checkpoint(t, ckpt)

    t2 = SDCFRNLHETrainer(cfg)
    assert t2._completed_iterations == 0
    ret = load_nlhe_checkpoint(t2, ckpt)
    assert ret == 7
    assert t2._completed_iterations == 7  # restored without the caller assigning it


def test_checkpoint_traversal_key_is_authoritative_over_seed(tmp_path) -> None:
    """A resumed trainer must adopt the checkpoint's deal stream, regardless of its config.seed."""
    from pokerbot.training.sd_cfr_nlhe import load_nlhe_checkpoint, save_nlhe_checkpoint

    cfg_a = SDCFRNLHEConfig(
        iterations=1, traversals_per_iter=2, deal_sweeps=1, train_batch_size=64,
        sft_checkpoint=None, seed=11, max_traverse_depth=16,
    )
    t = SDCFRNLHETrainer(cfg_a)
    saved_key = np.asarray(t._traversal_root_key).copy()
    ckpt = tmp_path / "ckpt.pkl"
    save_nlhe_checkpoint(t, ckpt)

    cfg_b = SDCFRNLHEConfig(
        iterations=1, traversals_per_iter=2, deal_sweeps=1, train_batch_size=64,
        sft_checkpoint=None, seed=999, max_traverse_depth=16,
    )
    t2 = SDCFRNLHETrainer(cfg_b)
    assert not np.array_equal(np.asarray(t2._traversal_root_key), saved_key)
    load_nlhe_checkpoint(t2, ckpt)
    assert np.array_equal(np.asarray(t2._traversal_root_key), saved_key)
