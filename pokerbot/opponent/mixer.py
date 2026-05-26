"""Safe GTO/exploit policy mixing (OX-Search inspired)."""

from __future__ import annotations

import numpy as np

from pokerbot.engine.actions import NUM_ACTIONS
from pokerbot.opponent.stats import OpponentStats


def exploit_adjustment(stats: OpponentStats) -> np.ndarray:
    """
    Return logit adjustments for each action index based on opponent archetype.
    Indices: fold, check, call, bet buckets..., all_in
    """
    adj = np.zeros(NUM_ACTIONS, dtype=np.float32)
    arch = stats.archetype()

    if arch == "nit":
        adj[3:9] += 0.5
        adj[0] -= 0.3
    elif arch == "calling_station":
        adj[3:9] -= 0.3
        adj[2] += 0.2
        adj[7:9] += 0.4
    elif arch == "lag":
        adj[0] += 0.3
        adj[2] += 0.2
    elif arch == "tag":
        adj[3:6] += 0.2
    return adj


def safe_mix_policies(
    gto_probs: np.ndarray,
    exploit_logits: np.ndarray,
    alpha: float,
    max_alpha: float = 0.7,
) -> np.ndarray:
    """
    π_final = (1-α)·π_GTO + α·π_exploit
    α gated by confidence; renormalize.
    """
    alpha = float(np.clip(alpha, 0.0, max_alpha))
    exploit_probs = np.exp(exploit_logits - exploit_logits.max())
    exploit_probs /= exploit_probs.sum() + 1e-8
    mixed = (1 - alpha) * gto_probs + alpha * exploit_probs
    mixed = np.clip(mixed, 1e-8, 1.0)
    return mixed / mixed.sum()


def _opponent_seat_index(actor_seat: int, hero_seat: int = 0) -> int:
    """Map absolute seat to opponent index in [0, 4] relative to hero."""
    offset = (actor_seat - hero_seat) % 6
    return offset - 1 if offset > 0 else 0


class HybridPolicy:
    """Combines GTO policy network with MADT opponent-aware adjustments."""

    def __init__(self, model, opponent_tracker, opponent_transformer=None, device="cpu"):
        self.model = model
        self.tracker = opponent_tracker
        self.opponent_transformer = opponent_transformer
        self.device = device

    def get_probs(
        self,
        weva_ids,
        actions,
        position,
        legal_mask,
        opponent_id: int,
        transformer_out: dict | None = None,
        actor_seat: int | None = None,
    ) -> np.ndarray:
        import torch

        with torch.no_grad():
            out = self.model(
                weva_ids.unsqueeze(0) if weva_ids.dim() == 0 else weva_ids,
                actions.unsqueeze(0) if actions.dim() == 1 else actions,
                position.unsqueeze(0) if position.dim() == 1 else position,
                legal_mask.unsqueeze(0) if legal_mask.dim() == 1 else legal_mask,
            )
            gto = out.probs.squeeze(0).cpu().numpy()

        stats = self.tracker.get(opponent_id)
        exploit_logits = np.log(gto + 1e-8) + exploit_adjustment(stats)

        transformer_confidence = 0.0
        if transformer_out is not None and actor_seat is not None:
            policy_logits = transformer_out.get("policy_logits")
            if policy_logits is not None:
                opp_idx = _opponent_seat_index(actor_seat)
                opp_idx = min(max(opp_idx, 0), 4)
                logits = policy_logits[0, opp_idx].cpu().numpy()
                exploit_logits = exploit_logits + 0.5 * logits
                belief_logits = transformer_out.get("belief_logits")
                if belief_logits is not None:
                    belief = torch.softmax(belief_logits[0, opp_idx], dim=-1)
                    entropy = -(belief * torch.log(belief + 1e-8)).sum().item()
                    max_entropy = np.log(belief.shape[-1])
                    transformer_confidence = 1.0 - entropy / max_entropy

        alpha = self.tracker.confidence(opponent_id) * 0.5
        alpha += transformer_confidence * 0.3
        if stats.archetype() == "unknown" and transformer_confidence < 0.1:
            alpha = 0.0

        final = safe_mix_policies(gto, exploit_logits, alpha)
        if legal_mask is not None:
            lm = legal_mask.cpu().numpy().astype(bool)
            if lm.ndim > 1:
                lm = lm.squeeze(0)
            final = final * lm
            final /= final.sum() + 1e-8
        return final
