"""MADT belief-conditioned best-response exploit logits (Phase 7)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

from pokerbot.engine.actions import NUM_ACTIONS
from pokerbot.opponent.sequence_encoder import NUM_BELIEF_COMBOS

if TYPE_CHECKING:
    from pokerbot.opponent.opponent_transformer import OpponentTransformer
    from pokerbot.opponent.sequence_encoder import HandTrajectoryBuilder


def belief_br_exploit_logits(
    belief_probs: np.ndarray,
    opponent_policy_logits: np.ndarray,
    *,
    top_k: int = 32,
) -> np.ndarray:
    """
    Aggregate opponent policy logits weighted by belief mass over combos.

    opponent_policy_logits: (1326, NUM_ACTIONS) or (K, NUM_ACTIONS) aligned with top indices.
    belief_probs: (1326,) prior over opponent hole combos.
    Returns exploit logits (NUM_ACTIONS,).
    """
    belief = np.asarray(belief_probs, dtype=np.float64)
    if opponent_policy_logits.ndim == 2 and opponent_policy_logits.shape[0] == belief.shape[0]:
        legal = np.where(belief > 0)[0]
        if legal.size == 0:
            return np.zeros(NUM_ACTIONS, dtype=np.float64)
        k = min(top_k, legal.size)
        top = legal[np.argsort(-belief[legal])[:k]]
        weights = belief[top]
        weights /= weights.sum() + 1e-12
        logits = opponent_policy_logits[top].astype(np.float64)
        return np.average(logits, axis=0, weights=weights)

    logits = np.asarray(opponent_policy_logits, dtype=np.float64)
    if logits.ndim == 1:
        return logits
    return logits.mean(axis=0)


def belief_br_from_transformer(
    transformer: OpponentTransformer,
    trajectory: HandTrajectoryBuilder,
    belief_logits: torch.Tensor,
    hero_seat: int,
    opp_idx: int,
    device: torch.device,
    *,
    board: list | None = None,
    top_k: int = 32,
) -> np.ndarray:
    """Belief-weighted exploit logits using per-combo MADT policy queries."""
    from pokerbot.opponent.sequence_encoder import (
        build_card_removal_mask,
        combo_index_to_cards,
        trajectory_to_batch,
    )

    belief = torch.softmax(belief_logits[0, opp_idx], dim=-1).cpu().numpy()
    legal = np.where(belief > 1e-8)[0]
    if legal.size == 0:
        return np.zeros(NUM_ACTIONS, dtype=np.float64)

    k = min(top_k, legal.size)
    top = legal[np.argsort(-belief[legal])[:k]]
    weights = belief[top]
    weights /= weights.sum() + 1e-12

    traj = trajectory.finalize()
    batch = trajectory_to_batch([traj], device=device)
    combo_logits = np.zeros((k, NUM_ACTIONS), dtype=np.float64)

    for i, combo_i in enumerate(top):
        hole = combo_index_to_cards(int(combo_i))
        removal = build_card_removal_mask(hole, board or [], device=device).unsqueeze(0)
        with torch.no_grad():
            out = transformer(
                batch["token_types"],
                batch["token_features"],
                batch["attn_mask"],
                batch["hero_seat"],
                card_removal_mask=removal,
            )
        combo_logits[i] = out["policy_logits"][0, opp_idx].cpu().numpy()

    return np.average(combo_logits, axis=0, weights=weights)


def belief_br_from_transformer_output(
    belief_logits: torch.Tensor,
    policy_logits: torch.Tensor,
    opp_idx: int,
    *,
    top_k: int = 32,
    transformer: OpponentTransformer | None = None,
    trajectory: HandTrajectoryBuilder | None = None,
    hero_seat: int = 0,
    device: torch.device | None = None,
    board: list | None = None,
) -> np.ndarray:
    """
    Extract belief-BR exploit logits for one opponent index from transformer outputs.

    When transformer + trajectory are provided, queries per-combo policies on top-K
    belief mass (same pattern as combo_action_likelihood_from_transformer). Otherwise
    falls back to the aggregate policy head logits for that opponent.
    """
    if transformer is not None and trajectory is not None and device is not None:
        return belief_br_from_transformer(
            transformer,
            trajectory,
            belief_logits,
            hero_seat,
            opp_idx,
            device,
            board=board,
            top_k=top_k,
        )

    return policy_logits[0, opp_idx].cpu().numpy().astype(np.float64)
