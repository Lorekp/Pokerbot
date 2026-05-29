"""Safe GTO/exploit policy mixing (OX-Search inspired)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from pokerbot.engine.actions import NUM_ACTIONS
from pokerbot.opponent.stats import OpponentStats

USE_BELIEF_BR = True

# Default until `scripts/calibrate_mixer_threshold.py` writes checkpoints/mixer_threshold.json.
# PROVISIONAL / UNCALIBRATED for live deployment — see docs/phase3-runbook.md § Mixer confidence.
_DEFAULT_BELIEF_CONFIDENCE_THRESHOLD = 0.35


def _load_calibrated_threshold() -> float:
    path = Path("checkpoints/mixer_threshold.json")
    if not path.exists():
        return _DEFAULT_BELIEF_CONFIDENCE_THRESHOLD
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return float(data.get("chosen_threshold", _DEFAULT_BELIEF_CONFIDENCE_THRESHOLD))
    except (OSError, ValueError, TypeError):
        return _DEFAULT_BELIEF_CONFIDENCE_THRESHOLD


BELIEF_CONFIDENCE_THRESHOLD = _load_calibrated_threshold()
TRANSFORMER_LOGIT_WEIGHT = 1.0
STATS_FALLBACK_WEIGHT = 0.2


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


def _opponent_seat_index(actor_seat: int, hero_seat: int) -> int:
    """Map absolute seat to opponent index in [0, 4] relative to hero."""
    from pokerbot.opponent.sequence_encoder import opponent_index_for_seat

    return opponent_index_for_seat(actor_seat, hero_seat)


def _belief_confidence(belief_logits) -> float:
    import torch

    belief = torch.softmax(belief_logits, dim=-1)
    entropy = -(belief * torch.log(belief + 1e-8)).sum().item()
    max_entropy = np.log(belief.shape[-1])
    return float(1.0 - entropy / max(max_entropy, 1e-8))


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
        cards,
        actions,
        position,
        legal_mask,
        opponent_id: int,
        transformer_out: dict | None = None,
        actor_seat: int | None = None,
        hero_seat: int = 0,
        board: list | None = None,
        trajectory=None,
    ) -> np.ndarray:
        import torch

        with torch.no_grad():
            out = self.model(
                weva_ids.unsqueeze(0) if weva_ids.dim() == 0 else weva_ids,
                actions.unsqueeze(0) if actions.dim() == 1 else actions,
                position.unsqueeze(0) if position.dim() == 1 else position,
                legal_mask.unsqueeze(0) if legal_mask.dim() == 1 else legal_mask,
                cards=cards.unsqueeze(0) if cards.dim() == 3 else cards,
            )
            gto = out.probs.squeeze(0).cpu().numpy()

        stats = self.tracker.get(opponent_id)
        transformer_confidence = 0.0
        exploit_logits = np.log(gto + 1e-8)

        if transformer_out is not None and actor_seat is not None:
            policy_logits = transformer_out.get("policy_logits")
            belief_logits = transformer_out.get("belief_logits")
            if policy_logits is not None:
                opp_idx = _opponent_seat_index(actor_seat, hero_seat)
                opp_idx = min(max(opp_idx, 0), 4)
                tf_logits = policy_logits[0, opp_idx].cpu().numpy()
                if belief_logits is not None:
                    transformer_confidence = _belief_confidence(belief_logits[0, opp_idx])
                if transformer_confidence >= BELIEF_CONFIDENCE_THRESHOLD:
                    if USE_BELIEF_BR and belief_logits is not None:
                        from pokerbot.opponent.belief_br import belief_br_from_transformer_output

                        exploit_logits = belief_br_from_transformer_output(
                            belief_logits,
                            policy_logits,
                            opp_idx,
                            transformer=self.opponent_transformer,
                            trajectory=trajectory,
                            hero_seat=hero_seat,
                            device=self.device,
                            board=board,
                        )
                    else:
                        exploit_logits = TRANSFORMER_LOGIT_WEIGHT * tf_logits
                    exploit_logits = exploit_logits + STATS_FALLBACK_WEIGHT * exploit_adjustment(stats)
                else:
                    exploit_logits = exploit_logits + exploit_adjustment(stats)
                    exploit_logits = exploit_logits + 0.35 * tf_logits
        else:
            exploit_logits = exploit_logits + exploit_adjustment(stats)

        if transformer_confidence >= BELIEF_CONFIDENCE_THRESHOLD:
            alpha = min(0.7, 0.35 + transformer_confidence * 0.35)
        else:
            alpha = self.tracker.confidence(opponent_id) * 0.5
            alpha += transformer_confidence * 0.25
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
