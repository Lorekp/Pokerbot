"""Phase 4: SearchAgent, gate harness, checkpoint loader, PBS likelihood."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from pokerbot.engine.game import GameState
from pokerbot.eval.gate_harness import GATE_OPPONENT_POOL, run_gate_eval
from pokerbot.eval.search_agent import SearchAgent
from pokerbot.models.checkpoint_loader import (
    is_legacy_card_encoder_checkpoint,
    load_pseudo_siamese_policy,
)
from pokerbot.models.policy import PseudoSiamesePolicy
from pokerbot.search.pbs_tracker import NUM_BELIEF_COMBOS, PBSTracker
from pokerbot.search.turbo_rebel_search import TurboReBeLConfig, TurboReBeLSolver
from pokerbot.search.leaf_evaluator import MockHistoryValueEvaluator
from pokerbot.search.state_bridge import build_subgame_root


def test_legacy_checkpoint_loader_raises():
    legacy_sd = {"card_encoder.0.net.0.weight": torch.zeros(32, 6, 3, 3)}
    ckpt = {"model": legacy_sd}
    path = Path("checkpoints/_test_legacy_phase4.pt")
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, path)
    try:
        with pytest.raises(ValueError, match="Legacy checkpoint"):
            load_pseudo_siamese_policy(path, "cpu")
    finally:
        path.unlink(missing_ok=True)


def test_is_legacy_detection():
    assert is_legacy_card_encoder_checkpoint({"card_encoder.0.net.0.weight": torch.zeros(1)})
    assert not is_legacy_card_encoder_checkpoint({"weva_embed.weight": torch.zeros(100, 128)})


def test_weva_checkpoint_roundtrip(tmp_path):
    model = PseudoSiamesePolicy(num_weva_buckets=100)
    ckpt_path = tmp_path / "weva_smoke.pt"
    torch.save({"model": model.state_dict(), "num_weva_buckets": 100}, ckpt_path)
    loaded = load_pseudo_siamese_policy(ckpt_path, "cpu")
    assert loaded.num_weva_buckets == 100


def test_combo_likelihood_not_uniform():
    state = GameState.new_hand(button_seat=0)
    state.deal_hole_cards()
    pbs = PBSTracker.init_uniform(0, state.players[0].hole, state.board)
    uniform = pbs.action_likelihood_from_policy(2, np.ones(11) / 11)
    assert uniform.std() < 1e-6

    varied = np.full(NUM_BELIEF_COMBOS, 0.1, dtype=np.float32)
    varied[0] = 0.9
    varied[1] = 0.05
    pbs.update_action(1, 2, policy_likelihood=varied)
    assert pbs.beliefs[1].std() > 0


def test_search_agent_returns_legal_action():
    state = GameState.new_hand(button_seat=3)
    state.deal_hole_cards()
    agent = SearchAgent(use_mock_leaf=True, time_budget_ms=300, device="cpu")
    agent.reset_hand(0)
    while not state.is_terminal() and state.action_seat != 0:
        legal = state.legal_actions()
        state.apply_action(legal[0])
        state.advance_to_next_actor()
    if state.action_seat == 0 and state.legal_actions():
        action = agent(state)
        assert action in state.legal_actions()


def test_gate_opponent_pool_registry():
    from pokerbot.eval.bots import BOT_REGISTRY

    for name in GATE_OPPONENT_POOL:
        assert name in BOT_REGISTRY


def test_run_gate_eval_smoke():
    state = GameState.new_hand(button_seat=0)
    state.deal_hole_cards()

    class PassBot:
        def __call__(self, s):
            return s.legal_actions()[0]

    metrics = run_gate_eval(PassBot, hands=4, seed=0, opponent_pool=("nit", "lag"))
    assert "exploiter_ev_bb_per_hand" in metrics
    assert "signed_exploiter_ev_metric_note" in metrics
    assert "harness_sanity_precondition" in metrics
    assert "bb100_vs_pool" in metrics
    assert metrics["opponent_pool"] == ["nit", "lag"]


def test_batched_leaf_matches_single(tmp_path):
    state = GameState.new_hand(button_seat=3)
    state.deal_hole_cards()
    pbs = PBSTracker.init_uniform(0, state.players[0].hole, state.board)
    root = build_subgame_root(state, pbs, 0)
    ev = MockHistoryValueEvaluator(hidden_dim=32)
    cfg = TurboReBeLConfig(max_iterations=8, ssmig_inner_iters=2, time_budget_ms=500, leaf_batch_size=2)
    solver = TurboReBeLSolver(ev, cfg)
    result = solver.solve(root, pbs)
    assert result.action_probs.sum() == pytest.approx(1.0, rel=1e-3)
    assert result.metadata.get("leaf_eval_count", 0) >= 0


def test_gate_reference_hybrid_fields(tmp_path):
    from pokerbot.eval.gate_reference import promote_baseline, promote_hybrid_baseline, resolve_hybrid_baseline

    hero = PseudoSiamesePolicy(num_weva_buckets=100)
    hero_path = tmp_path / "hero.pt"
    manifest = tmp_path / "gate_reference.json"
    torch.save({"model": hero.state_dict(), "num_weva_buckets": 100}, hero_path)
    promote_baseline(hero_path, baseline_path=tmp_path / "baseline.pt", manifest_path=manifest)

    tf_path = tmp_path / "madt.pt"
    tf_model = torch.nn.Linear(1, 1)
    torch.save({"model": tf_model.state_dict()}, tf_path)
    promote_hybrid_baseline(tf_path, transformer_baseline_path=tmp_path / "tf_baseline.pt", manifest_path=manifest)

    _, tf_resolved, m = resolve_hybrid_baseline(manifest_path=manifest, require=True)
    assert m.get("transformer_sha256")
    assert tf_resolved.exists()
