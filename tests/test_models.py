"""Tests for policy network and opponent modules."""

import numpy as np
import torch

from pokerbot.encoding.spot_scope import encode_spot_scope, fuse_position_scope
from pokerbot.encoding.tensors import SPEC
from pokerbot.models.policy import PseudoSiamesePolicy
from pokerbot.opponent.mixer import exploit_adjustment, safe_mix_policies
from pokerbot.opponent.stats import OpponentTracker


def test_policy_forward():
    model = PseudoSiamesePolicy(hidden=64, num_weva_buckets=128)
    b = 2
    weva_ids = torch.randint(0, 128, (b,), dtype=torch.long)
    cards = torch.randn(b, *SPEC.card_shape)
    actions = torch.randn(b, *SPEC.action_shape)
    pos = np.zeros((b, 6), dtype=np.float32)
    pos[:, 0] = 1.0
    scope = encode_spot_scope("hu_postflop_2p")
    position = torch.from_numpy(
        np.stack([fuse_position_scope(pos[i], scope) for i in range(b)])
    ).float()
    out = model(weva_ids, actions, position, cards=cards)
    assert out.logits.shape == (b, model.num_actions)
    assert out.value.shape == (b,)


def test_safe_mix():
    gto = np.ones(11) / 11
    exploit = np.log(gto)
    mixed = safe_mix_policies(gto, exploit, alpha=0.0)
    assert abs(mixed.sum() - 1.0) < 1e-6


def test_opponent_tracker_basic():
    t = OpponentTracker()
    for _ in range(10):
        t.record_hand(1, vpip=True, pfr=True)
    assert t.get(1).hands == 10
    assert t.confidence(1) == 0.2


def test_opponent_transformer():
    from pokerbot.opponent.opponent_transformer import OpponentTransformer, OpponentTransformerConfig
    from pokerbot.opponent.sequence_encoder import TrajectoryRecord, trajectory_to_batch

    def _traj():
        rng = np.random.default_rng(0)
        return TrajectoryRecord(
            token_types=[1, 2, 3, 1],
            token_features=[rng.random(304, dtype=np.float32) for _ in range(4)],
            actor_seats=[0, 1, 1, 2],
            hero_seat=0,
        )

    model = OpponentTransformer(OpponentTransformerConfig(d_model=64, n_heads=4, n_layers=2, d_ff=128))
    batch = trajectory_to_batch([_traj()])
    out = model(
        batch["token_types"],
        batch["token_features"],
        batch["attn_mask"],
        batch["hero_seat"],
    )
    assert out["belief_logits"].shape == (1, 5, 1326)
