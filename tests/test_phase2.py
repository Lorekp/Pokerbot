"""Phase 2 integration tests: blueprint bot, distillation, leaf evaluator."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from pokerbot.engine.game import play_hand
from pokerbot.eval.jax_blueprint_bot import JAXBlueprintBot
from pokerbot.search.leaf_evaluator import HistoryValueEvaluator
from pokerbot.training.blueprint_distill import BlueprintDistillConfig, distill_blueprint_to_policy
from pokerbot.training.neural_cfr import VRDeepPDCFRConfig, VRDeepPDCFRPlusTrainer, save_trainer_checkpoint


def _minimal_cfr_checkpoint(tmp_path: Path) -> Path:
    cfg = VRDeepPDCFRConfig(total_iterations=1, traversals_per_iter=2, batch_size=8, buffer_capacity=64)
    trainer = VRDeepPDCFRPlusTrainer(cfg)
    trainer.train_iteration(1)
    ckpt = tmp_path / "history_value_net.pkl"
    save_trainer_checkpoint(trainer, ckpt)
    return ckpt


def test_jax_blueprint_bot_legal_action(tmp_path) -> None:
    import random

    from pokerbot.eval.bots import RandomBot

    ckpt = _minimal_cfr_checkpoint(tmp_path)
    bot = JAXBlueprintBot(ckpt)
    rng = random.Random(0)
    policies = {i: (bot if i == 0 else RandomBot(rng)) for i in range(6)}
    state, _ = play_hand(policies, rng=rng)
    assert state.is_terminal()


def test_distill_loss_finite_on_synthetic(tmp_path, monkeypatch) -> None:
    ckpt = _minimal_cfr_checkpoint(tmp_path)

    def fake_rows(_teacher, _hands, _rng):
        return [
            {
                "weva_id": np.int64(i),
                # cards is required by PseudoSiamesePolicy (encoding v2 added board cards); the real
                # _collect_distill_rows supplies enc["cards"] of SPEC.card_shape = (9, 4, 13).
                "cards": np.zeros((9, 4, 13), dtype=np.float32),
                "actions": np.zeros((6, 4, 12), dtype=np.float32),
                "position": np.zeros(8, dtype=np.float32),   # POLICY_POSITION_DIM (position + spot_scope)
                "legal_mask": np.ones(11, dtype=bool),
                "teacher_probs": np.ones(11, dtype=np.float32) / 11.0,
            }
            for i in range(32)
        ]

    monkeypatch.setattr(
        "pokerbot.training.blueprint_distill._collect_distill_rows",
        fake_rows,
    )
    out = tmp_path / "policy_cfr_blueprint.pt"
    distill_blueprint_to_policy(
        BlueprintDistillConfig(
            cfr_checkpoint=ckpt,
            output_path=out,
            hands=4,
            epochs=1,
            batch_size=16,
            device="cpu",
        )
    )
    assert out.exists()
    payload = torch.load(out, map_location="cpu", weights_only=False)
    assert payload["source"] == "neural_cfr_blueprint"


def test_history_value_evaluator_has_trained_checkpoint(tmp_path) -> None:
    ckpt = _minimal_cfr_checkpoint(tmp_path)
    ev = HistoryValueEvaluator(checkpoint=ckpt)
    assert ev.has_trained_checkpoint
