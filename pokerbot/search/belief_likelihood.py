"""Combo-stratified action likelihoods for PBS Bayesian updates."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

from pokerbot.engine.actions import NUM_ACTIONS
from pokerbot.opponent.sequence_encoder import NUM_BELIEF_COMBOS, combo_index_to_cards

if TYPE_CHECKING:
    from pokerbot.opponent.opponent_transformer import OpponentTransformer
    from pokerbot.opponent.sequence_encoder import HandTrajectoryBuilder


def _uniform_likelihood(action_idx: int, policy_probs: np.ndarray | None) -> np.ndarray:
    if policy_probs is not None:
        p = float(np.clip(policy_probs[action_idx], 1e-6, 1.0))
        return np.full(NUM_BELIEF_COMBOS, p, dtype=np.float32)
    return np.full(NUM_BELIEF_COMBOS, 1.0 / max(NUM_ACTIONS, 1), dtype=np.float32)


def combo_action_likelihood_from_transformer(
    transformer: OpponentTransformer,
    trajectory: HandTrajectoryBuilder,
    hero_seat: int,
    opponent_seat: int,
    action_idx: int,
    combo_mask: np.ndarray,
    prior: np.ndarray,
    device: torch.device,
    *,
    board: list | None = None,
    top_k: int = 64,
    random_tail: int = 16,
    full_1326: bool = False,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Estimate P(a|h) per combo by querying MADT policy head on counterfactual hole cards.

    Evaluates top-K combos by prior mass plus a random tail for coverage.
    """
    rng = rng or np.random.default_rng(0)
    likelihood = _uniform_likelihood(action_idx, None)

    legal_indices = np.where(combo_mask & (prior > 0))[0]
    if legal_indices.size == 0:
        return likelihood

    if full_1326 or legal_indices.size <= top_k + random_tail:
        eval_indices = legal_indices
    else:
        k_top = min(top_k, legal_indices.size)
        top_indices = legal_indices[np.argsort(-prior[legal_indices])[:k_top]]

        tail_n = min(random_tail, max(0, legal_indices.size - k_top))
        if tail_n > 0:
            remaining = np.setdiff1d(legal_indices, top_indices, assume_unique=True)
            if remaining.size > 0:
                tail = rng.choice(remaining, size=min(tail_n, remaining.size), replace=False)
                eval_indices = np.concatenate([top_indices, tail])
            else:
                eval_indices = top_indices
        else:
            eval_indices = top_indices

    if eval_indices.size == 0:
        return likelihood

    from pokerbot.opponent.sequence_encoder import (
        build_card_removal_mask,
        opponent_index_for_seat,
        trajectory_to_batch,
    )

    traj = trajectory.finalize()
    batch = trajectory_to_batch([traj], device=device)
    opp_idx = opponent_index_for_seat(opponent_seat, hero_seat)
    opp_idx = min(max(opp_idx, 0), 4)

    for combo_i in eval_indices:
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
        probs = torch.softmax(out["policy_logits"][0, opp_idx], dim=-1).cpu().numpy()
        likelihood[int(combo_i)] = float(np.clip(probs[action_idx], 1e-6, 1.0))

    return likelihood


def combo_action_likelihood_from_blueprint(
    blueprint_module,
    blueprint_params,
    env,
    actor_seat: int,
    action_idx: int,
    combo_mask: np.ndarray,
    prior: np.ndarray,
    combo_card_indices: list[tuple[int, int]],
    *,
    top_k: int = 32,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Blueprint avg-strategy likelihood per combo (fallback when MADT absent)."""
    import jax.numpy as jnp

    from pokerbot.engine.jax_engine import encode_info_state, legal_action_mask

    rng = rng or np.random.default_rng(0)
    likelihood = _uniform_likelihood(action_idx, None)

    legal_indices = np.where(combo_mask & (prior > 0))[0]
    if legal_indices.size == 0:
        return likelihood

    k = min(top_k, legal_indices.size)
    eval_indices = legal_indices[np.argsort(-prior[legal_indices])[:k]]

    holes = np.asarray(env.holes).copy()
    for combo_i in eval_indices:
        c0, c1 = combo_card_indices[int(combo_i)]
        holes[actor_seat, 0] = c0
        holes[actor_seat, 1] = c1
        env_copy = env._replace(holes=jnp.asarray(holes))
        mask = legal_action_mask(env_copy, jnp.int32(actor_seat))
        info = encode_info_state(env_copy, jnp.int32(actor_seat))
        probs = np.asarray(blueprint_module.apply(blueprint_params, info, mask))
        likelihood[int(combo_i)] = float(np.clip(probs[action_idx], 1e-6, 1.0))

    return likelihood
