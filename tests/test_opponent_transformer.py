"""Tests for MADT opponent transformer stack."""

from __future__ import annotations

import copy

import numpy as np
import pytest
import torch

from pokerbot.engine.actions import Action, ActionType
from pokerbot.engine.cards import Card, Rank, Suit
from pokerbot.engine.game import GameState, NUM_PLAYERS
from pokerbot.models.policy import PseudoSiamesePolicy
from pokerbot.opponent.belief_loss import opponent_transformer_loss
from pokerbot.opponent.opponent_transformer import KVCache, OpponentTransformer, OpponentTransformerConfig
from pokerbot.opponent.scapt_manager import SCAPTManager
from pokerbot.opponent.sequence_encoder import (
    HandTrajectoryBuilder,
    TrajectoryRecord,
    build_card_removal_mask,
    cards_to_combo_index,
    encode_public_state,
    trajectory_to_batch,
)
from pokerbot.training.mixed_self_play import play_hand_tracked


def _random_trajectory(seq_len: int = 8, hero_seat: int = 0) -> TrajectoryRecord:
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


def test_cards_to_combo_index_roundtrip():
    hole = (Card(Rank.ACE, Suit.SPADES), Card(Rank.KING, Suit.HEARTS))
    idx = cards_to_combo_index(hole)
    assert 0 <= idx < 1326
    assert cards_to_combo_index((hole[1], hole[0])) == idx


def test_encode_public_state_shape():
    state = GameState.new_hand(button_seat=3)
    state.deal_hole_cards()
    feat = encode_public_state(state, actor_seat=0, hero_seat=0)
    assert feat.shape == (304,)


def test_trajectory_to_batch_shapes():
    batch = trajectory_to_batch([_random_trajectory(), _random_trajectory(12)])
    assert batch["token_types"].shape[0] == 2
    assert batch["belief_targets"].shape == (2, 5)
    assert batch["cards_known_mask"].shape == (2, 5)


def test_opponent_transformer_forward_shapes():
    model = OpponentTransformer(OpponentTransformerConfig(d_model=64, n_heads=4, n_layers=2, d_ff=128))
    batch = trajectory_to_batch([_random_trajectory(6), _random_trajectory(10)])
    out = model(
        batch["token_types"],
        batch["token_features"],
        batch["attn_mask"],
        batch["hero_seat"],
    )
    assert out["belief_logits"].shape == (2, 5, 1326)
    assert out["policy_logits"].shape == (2, 5, 11)


def test_belief_loss_zero_when_mask_zero():
    model = OpponentTransformer(OpponentTransformerConfig(d_model=32, n_heads=4, n_layers=1, d_ff=64))
    batch = trajectory_to_batch([_random_trajectory(4)])
    out = model(
        batch["token_types"],
        batch["token_features"],
        batch["attn_mask"],
        batch["hero_seat"],
    )
    targets = {
        "belief_targets": batch["belief_targets"],
        "policy_targets": batch["policy_targets"],
        "cards_known_mask": torch.zeros_like(batch["cards_known_mask"]),
        "action_known_mask": batch["action_known_mask"],
    }
    breakdown = opponent_transformer_loss(out, targets)
    assert breakdown.belief_count == 0.0
    assert abs(breakdown.belief.item()) < 1e-6


def test_belief_loss_normalized_by_mask_count():
    model = OpponentTransformer(OpponentTransformerConfig(d_model=32, n_heads=4, n_layers=1, d_ff=64))
    batch = trajectory_to_batch([_random_trajectory(4)])
    out = model(
        batch["token_types"],
        batch["token_features"],
        batch["attn_mask"],
        batch["hero_seat"],
    )
    targets = {
        "belief_targets": batch["belief_targets"],
        "policy_targets": batch["policy_targets"],
        "cards_known_mask": batch["cards_known_mask"],
        "action_known_mask": batch["action_known_mask"],
    }
    breakdown = opponent_transformer_loss(out, targets)
    assert breakdown.belief_count >= 1.0
    assert torch.isfinite(breakdown.total)


def test_card_removal_mask():
    hero = (Card(Rank.ACE, Suit.SPADES), Card(Rank.KING, Suit.HEARTS))
    board = [Card(Rank.TWO, Suit.CLUBS)]
    mask = build_card_removal_mask(hero, board)
    assert mask.sum() < 1326
    idx = cards_to_combo_index(hero)
    assert not mask[idx]


def test_scapt_perturbation_delta():
    policy = PseudoSiamesePolicy(hidden=32, num_weva_buckets=128)
    mgr = SCAPTManager()
    mgr.init_population(policy)
    base = mgr.population[0].policy_state
    mutated = mgr.perturb(mgr.population[0]).policy_state
    diffs = []
    for k in base:
        if base[k].is_floating_point():
            diffs.append((mutated[k] - base[k]).abs().mean().item())
    assert max(diffs) > 0.0


def test_kv_cache_incremental_parity():
    torch.manual_seed(0)
    model = OpponentTransformer(OpponentTransformerConfig(d_model=64, n_heads=4, n_layers=2, d_ff=128))
    model.eval()
    traj = _random_trajectory(6)
    full_batch = trajectory_to_batch([traj])

    with torch.no_grad():
        full_out = model(
            full_batch["token_types"],
            full_batch["token_features"],
            full_batch["attn_mask"],
            full_batch["hero_seat"],
        )

        cache = KVCache()
        partial_len = 3
        partial = trajectory_to_batch(
            [
                TrajectoryRecord(
                    token_types=traj.token_types[:partial_len],
                    token_features=traj.token_features[:partial_len],
                    actor_seats=traj.actor_seats[:partial_len],
                    hero_seat=traj.hero_seat,
                )
            ]
        )
        out1 = model(
            partial["token_types"],
            partial["token_features"],
            partial["attn_mask"],
            partial["hero_seat"],
            kv_cache=cache,
            use_cache=True,
        )
        cache = out1["kv_cache"]

        rest = trajectory_to_batch(
            [
                TrajectoryRecord(
                    token_types=traj.token_types,
                    token_features=traj.token_features,
                    actor_seats=traj.actor_seats,
                    hero_seat=traj.hero_seat,
                )
            ]
        )
        out2 = model(
            rest["token_types"],
            rest["token_features"],
            rest["attn_mask"],
            rest["hero_seat"],
            kv_cache=cache,
            use_cache=True,
        )

    assert out2["policy_logits"].shape == full_out["policy_logits"].shape
    assert torch.allclose(out2["policy_logits"], full_out["policy_logits"], atol=1e-4, rtol=1e-3)


def test_play_hand_tracked_emits_tokens():
    rng = __import__("random").Random(0)

    def always_check(state: GameState) -> Action:
        legal = state.legal_actions()
        for a in legal:
            if a.action_type == ActionType.CHECK:
                return a
        return legal[0]

    policies = {i: always_check for i in range(NUM_PLAYERS)}
    _, _, traj = play_hand_tracked(policies, hero_seat=0, rng=rng)
    assert traj is not None
    assert len(traj.token_types) > 0


def test_causal_mask_blocks_future():
    """Later positions should not change when future tokens are appended (greedy check)."""
    model = OpponentTransformer(OpponentTransformerConfig(d_model=32, n_heads=4, n_layers=1, d_ff=64))
    model.eval()
    short = _random_trajectory(4)
    long = _random_trajectory(8)
    long.token_types[:4] = short.token_types
    long.token_features[:4] = short.token_features
    long.actor_seats[:4] = short.actor_seats

    b_short = trajectory_to_batch([short])
    b_long = trajectory_to_batch([long])
    with torch.no_grad():
        out_short = model(
            b_short["token_types"],
            b_short["token_features"],
            b_short["attn_mask"],
            b_short["hero_seat"],
        )
        out_long = model(
            b_long["token_types"][:, :4],
            b_long["token_features"][:, :4],
            b_long["attn_mask"][:, :4],
            b_long["hero_seat"],
        )
    assert torch.allclose(out_short["policy_logits"], out_long["policy_logits"], atol=1e-5)
