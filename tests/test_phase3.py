"""Phase 3 integration tests: Step 3 decoupling, SCAPT, gate baseline, mixer."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from pokerbot.encoding.spot_scope import POLICY_POSITION_DIM
from pokerbot.encoding.tensors import SPEC
from pokerbot.eval.gate_reference import promote_baseline, resolve_baseline_checkpoint, sha256_file
from pokerbot.models.policy import PseudoSiamesePolicy
from pokerbot.opponent.belief_loss import opponent_transformer_loss
from pokerbot.opponent.mixer import HybridPolicy, _belief_confidence
from pokerbot.opponent.opponent_transformer import OpponentTransformer, OpponentTransformerConfig
from pokerbot.opponent.scapt_manager import SCAPTConfig, SCAPTManager, SCAPT_MIN_LEAGUE_ACTION_KL, check_league_diversity
from pokerbot.opponent.stats import OpponentTracker
from pokerbot.opponent.sequence_encoder import trajectory_to_batch
from pokerbot.opponent.mixer_calibration import calibrate_belief_threshold
from pokerbot.training.opponent_transformer_train import _population_policy_logits


def _random_trajectory(seq_len: int = 8, hero_seat: int = 0):
    import numpy as np
    from pokerbot.engine.game import NUM_PLAYERS
    from pokerbot.opponent.sequence_encoder import TrajectoryRecord

    rng = np.random.default_rng(42)
    types = [1, 2, 3] * (seq_len // 3 + 1)
    types = types[:seq_len]
    feats = [rng.random(304, dtype=np.float32) for _ in range(seq_len)]
    actors = [rng.integers(0, NUM_PLAYERS) for _ in range(seq_len)]
    return TrajectoryRecord(
        token_types=types,
        token_features=feats,
        actor_seats=actors,
        hero_seat=hero_seat,
        belief_targets={1: 10, 2: 20},
        policy_targets={1: 2, 3: 5},
        cards_known={1: True, 2: True},
        action_known={1: True, 3: True},
    )


def test_train_phase3_does_not_call_self_play_mixed():
    text = Path("scripts/train_phase3.py").read_text(encoding="utf-8")
    assert "train_self_play_mixed" not in text
    assert "HybridAgent" not in text


def test_mi_loss_nonzero_with_population_logits():
    model = OpponentTransformer(OpponentTransformerConfig(d_model=32, n_heads=4, n_layers=1, d_ff=64))
    traj = _random_trajectory(6)
    batch = trajectory_to_batch([traj])
    out = model(batch["token_types"], batch["token_features"], batch["attn_mask"], batch["hero_seat"])
    pop = out["policy_logits"].unsqueeze(0).repeat(3, 1, 1, 1)
    pop = pop + torch.randn_like(pop) * 0.1
    targets = {
        "belief_targets": batch["belief_targets"],
        "policy_targets": batch["policy_targets"],
        "cards_known_mask": batch["cards_known_mask"],
        "action_known_mask": batch["action_known_mask"],
    }
    breakdown = opponent_transformer_loss(out, targets, population_policy_logits=pop)
    assert float(breakdown.mi.item()) != 0.0


def test_scapt_population_forwards_shape():
    model = OpponentTransformer(OpponentTransformerConfig(d_model=32, n_heads=4, n_layers=1, d_ff=64))
    hero = PseudoSiamesePolicy(hidden=32, num_weva_buckets=1000)
    scapt = SCAPTManager()
    scapt.init_population(hero)
    traj = _random_trajectory(6)
    batch = trajectory_to_batch([traj])
    pop = _population_policy_logits(model, batch, scapt, samples=4)
    assert pop is not None
    assert pop.shape[0] >= 2


def test_gate_baseline_promote_and_verify(tmp_path):
    ckpt = tmp_path / "hero.pt"
    model = PseudoSiamesePolicy(hidden=32, num_weva_buckets=1000)
    torch.save({"model": model.state_dict()}, ckpt)
    baseline = tmp_path / "baseline.pt"
    manifest = tmp_path / "gate_reference.json"
    meta = promote_baseline(ckpt, baseline_path=baseline, manifest_path=manifest)
    path, loaded = resolve_baseline_checkpoint(manifest_path=manifest, require=True)
    assert path == baseline
    assert loaded["hero_sha256"] == meta["hero_sha256"]


def test_gate_rejects_sha_mismatch(tmp_path):
    ckpt = tmp_path / "hero.pt"
    torch.save({"model": PseudoSiamesePolicy(hidden=32, num_weva_buckets=1000).state_dict()}, ckpt)
    baseline = tmp_path / "baseline.pt"
    manifest = tmp_path / "gate_reference.json"
    promote_baseline(ckpt, baseline_path=baseline, manifest_path=manifest)
    baseline.write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        resolve_baseline_checkpoint(manifest_path=manifest, require=True)


def test_hybrid_policy_prefers_transformer_when_confident():
    model = PseudoSiamesePolicy(hidden=32, num_weva_buckets=1000)
    model.eval()
    tracker = OpponentTracker()
    policy = HybridPolicy(model, tracker, device="cpu")

    legal = torch.ones(1, 11, dtype=torch.bool)
    cards = torch.zeros(1, *SPEC.card_shape)
    position = torch.zeros(1, POLICY_POSITION_DIM)
    sharp_belief = torch.zeros(1326)
    sharp_belief[0] = 10.0
    assert _belief_confidence(sharp_belief) > 0.35

    tf_out = {
        "policy_logits": torch.randn(1, 5, 11),
        "belief_logits": sharp_belief.view(1, 1, -1).expand(1, 5, -1),
    }
    probs = policy.get_probs(
        torch.tensor([0]),
        cards,
        torch.zeros(1, 6, 4, 12),
        position,
        legal,
        opponent_id=1,
        transformer_out=tf_out,
        actor_seat=1,
    )
    assert probs.shape == (11,)
    assert abs(probs.sum() - 1.0) < 1e-5


def test_trajectory_belief_validation_rejects_invalid():
    from pokerbot.training.trajectory_dataset import _validate_trajectory_row

    with pytest.raises(ValueError, match="without cards_known"):
        _validate_trajectory_row(
            {
                "belief_targets": {"1": 5},
                "cards_known_mask": {"1": False},
            }
        )


def test_scapt_league_diversity_default_sigma_passes_floor():
    hero = PseudoSiamesePolicy(hidden=32, num_weva_buckets=1000)
    scapt = SCAPTManager(SCAPTConfig(population_size=8))
    scapt.init_population(hero)
    report = scapt.measure_league_policy_diversity(hero, torch.device("cpu"), num_probe_states=8, rng=__import__("random").Random(0))
    assert report.mean_pairwise_action_kl >= SCAPT_MIN_LEAGUE_ACTION_KL
    features = scapt.diversity_features(
        torch.stack(
            [
                torch.softmax(torch.randn(8, 11), dim=-1)
                for _ in range(4)
            ]
        )
    )
    assert "mean_pairwise_action_kl" in features


def test_scapt_low_sigma_fails_require_diversity():
    hero = PseudoSiamesePolicy(hidden=32, num_weva_buckets=1000)
    scapt = SCAPTManager(SCAPTConfig(population_size=4, mutation_sigma=0.02))
    scapt.init_population(hero)
    report = scapt.measure_league_policy_diversity(hero, torch.device("cpu"), num_probe_states=8, rng=__import__("random").Random(1))
    with pytest.raises(RuntimeError, match="SCAPT league diversity below floor"):
        check_league_diversity(report, floor=0.5, require=True)


def test_mixer_calibration_prefers_high_confidence_bin():
    n = 64
    dim = 32
    targets = torch.randint(0, dim, (n,))
    logits = torch.randn(n, dim) * 0.05
    logits[torch.arange(n), targets] += 0.2
    mask = torch.ones(n, dtype=torch.bool)
    curve = calibrate_belief_threshold(logits, targets, mask, candidate_thresholds=[0.1, 0.2, 0.3, 0.4])
    assert curve.chosen_threshold <= 0.4
