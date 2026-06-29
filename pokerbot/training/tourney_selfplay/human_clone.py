"""HumanCloneNet — the skill-conditioned causal-transformer human-action model (Maia-for-poker).

This is the realism ANCHOR ``τ`` of the system: a network that predicts the ACTUAL next human action
from the full-hand causal (state, action, reward) token sequence, conditioned on a per-seat *skill
bin* and a *table skill* scalar. Every league agent's policy is the piKL operator
``softmax((Q + λ·log τ)/(κ+λ))`` over this ``τ`` (see :mod:`pikl`), so the field stays human-shaped
(Cicero / Jacob et al.) while the strength dial ``λ`` pushes it past the data.

DESIGN (component #1 of ``artifacts/tourney_selfplay_design.md``). This is a thin FORK of
:class:`pokerbot.opponent.opponent_transformer.OpponentTransformer`:

  * the token layer (``embed_tokens``), the GPT backbone, the belief head and the policy head are
    **reused verbatim** — we subclass and only override ``_opponent_queries`` to add the
    skill-aware query bias and ``forward`` to thread the skill tensors through.
  * the ONE new mechanism (Maia-2's skill-aware attention, arXiv 2409.20553, adapted) is a learned
    **query bias** added to each opponent's seat query:

        q*_seat = q_seat + W_skill · [ E_skill(skill_bin[seat]) ⊕ E_table(table_skill_bin) ]

    where ``E_skill`` embeds the acting seat's per-seat skill bin (decile of the composite strength
    score) and ``E_table`` embeds the discretized table-skill level. ``W_skill`` projects the
    concatenated (skill ⊕ table-skill) embedding into ``d_model`` so it adds to the seat query. When
    the skill inputs are absent (a plain BC pretrain with no labels) the bias is the learned bias of
    a zero embedding, so the net degrades gracefully to the base OpponentTransformer behavior.

INPUTS / OUTPUTS (mirrors the design table):
  inputs  — causal (state 304d, action 18d, reward 1d) token sequence; per-seat ``skill_bins (B,5)``;
            ``table_skill (B,)``; legal mask
  outputs — per-acting-seat next-action logits (the clone policy ``τ``); aux hole-card belief logits.

The five non-hero query slots are ordered exactly as the base class
(``opponent_seats``/``opponent_index_for_seat``), so the existing
``sequence_encoder.trajectory_to_batch`` / ``clone_loss`` targets line up with no re-indexing.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from pokerbot.engine.game import NUM_PLAYERS
from pokerbot.opponent.opponent_transformer import (
    KVCache,
    OpponentTransformer,
    OpponentTransformerConfig,
)
from pokerbot.opponent.sequence_encoder import TokenType  # noqa: F401  (re-exported for callers)

# Number of skill bins: deciles of the composite strength score (design A1). Bin 0 = weakest fish,
# bin 9 = strongest. A separate "unknown" sentinel maps to the middle bin at encode time.
NUM_SKILL_BINS = 10
# Table-skill is the same decile axis aggregated over the table (mean opponent bin), discretized.
NUM_TABLE_SKILL_BINS = 10


@dataclass
class HumanCloneConfig(OpponentTransformerConfig):
    """Clone config = the base transformer config + the skill-conditioning dims.

    Inherits ``d_model=256 / n_heads=8 / n_layers=6`` etc. from the base so a clone checkpoint is
    architecturally a superset of an OpponentTransformer checkpoint (the skill modules are the only
    extra params).
    """

    num_skill_bins: int = NUM_SKILL_BINS
    num_table_skill_bins: int = NUM_TABLE_SKILL_BINS
    skill_embed_dim: int = 32
    table_skill_embed_dim: int = 16


class HumanCloneNet(OpponentTransformer):
    """Skill-conditioned human-action predictor (the frozen anchor ``τ``).

    Subclasses :class:`OpponentTransformer`; adds ``skill_embed`` / ``table_skill_embed`` and the
    ``skill_proj`` (``W_skill``) that forms the per-seat query bias. ``forward`` accepts the two extra
    skill tensors (both optional) and threads them into ``_opponent_queries``.
    """

    def __init__(self, config: HumanCloneConfig | None = None) -> None:
        cfg = config or HumanCloneConfig()
        super().__init__(cfg)
        self.config: HumanCloneConfig = cfg

        self.skill_embed = nn.Embedding(cfg.num_skill_bins, cfg.skill_embed_dim)
        self.table_skill_embed = nn.Embedding(cfg.num_table_skill_bins, cfg.table_skill_embed_dim)
        # W_skill: [skill ⊕ table_skill] -> d_model. Bias carries the "no skill label" default so the
        # net is well-defined when skill inputs are None (uses bin index 0 -> a learned baseline).
        self.skill_proj = nn.Linear(
            cfg.skill_embed_dim + cfg.table_skill_embed_dim, cfg.d_model
        )

        # Re-init only the new modules (the base __init__ already xavier-init'd everything it owns).
        nn.init.xavier_uniform_(self.skill_proj.weight)
        nn.init.zeros_(self.skill_proj.bias)
        nn.init.normal_(self.skill_embed.weight, std=0.02)
        nn.init.normal_(self.table_skill_embed.weight, std=0.02)

    # -- the ONE overridden mechanism: skill-aware query bias ---------------------------------------
    def _opponent_queries(  # type: ignore[override]
        self,
        h_last: torch.Tensor,
        hero_seat: torch.Tensor,
        skill_bins: torch.Tensor | None = None,
        table_skill: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Five opponent query vectors with the per-seat skill bias added.

        Args mirror the base ``_opponent_queries`` plus:
            skill_bins: (B, NUM_PLAYERS) long — per-ABSOLUTE-seat skill bin (hero seat slot ignored).
            table_skill: (B,) long — discretized table-skill level.

        The query for opponent offset ``o`` (absolute seat ``(hero+o)%N``) is
        ``h_last + seat_emb + role + W_skill·[E_skill(skill_bins[seat]) ⊕ E_table(table_skill)]``.
        """
        b = h_last.shape[0]
        device = h_last.device
        role = self.role_embed(torch.tensor(1, device=device))

        # Table-skill embedding (B, table_dim); default bin 0 when absent.
        if table_skill is None:
            table_idx = torch.zeros(b, dtype=torch.long, device=device)
        else:
            table_idx = table_skill.to(device=device, dtype=torch.long).clamp(
                0, self.config.num_table_skill_bins - 1
            )
        table_emb = self.table_skill_embed(table_idx)  # (B, table_dim)

        opps = []
        for offset in range(1, NUM_PLAYERS):
            seat_ids = (hero_seat + offset) % NUM_PLAYERS  # (B,)
            seat_emb = self.seat_embed(seat_ids)

            if skill_bins is None:
                skill_idx = torch.zeros(b, dtype=torch.long, device=device)
            else:
                # gather this opponent's per-seat skill bin (B,)
                skill_idx = skill_bins.to(device=device, dtype=torch.long).gather(
                    1, seat_ids.view(b, 1)
                ).squeeze(1).clamp(0, self.config.num_skill_bins - 1)
            skill_emb = self.skill_embed(skill_idx)  # (B, skill_dim)

            bias = self.skill_proj(torch.cat([skill_emb, table_emb], dim=-1))  # (B, d_model)
            q = h_last + seat_emb + role + bias
            opps.append(q)
        return torch.stack(opps, dim=1)

    # -- forward = base forward with the skill tensors threaded into the query head ----------------
    def forward(  # type: ignore[override]
        self,
        token_types: torch.Tensor,
        token_features: torch.Tensor,
        attn_mask: torch.Tensor,
        hero_seat: torch.Tensor,
        kv_cache: KVCache | None = None,
        card_removal_mask: torch.Tensor | None = None,
        legal_action_mask: torch.Tensor | None = None,
        use_cache: bool = False,
        *,
        skill_bins: torch.Tensor | None = None,
        table_skill: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Same contract as ``OpponentTransformer.forward`` + ``skill_bins`` / ``table_skill``.

        We re-implement the body (rather than ``super().forward``) only because the base hard-codes the
        non-skill ``_opponent_queries`` call; everything before the query head is reused via the base
        helpers (``embed_tokens``, ``backbone``, ``_build_causal_mask``).
        """
        b, seq_len, _ = token_features.shape
        incremental = kv_cache is not None and kv_cache.seq_len > 0

        if incremental:
            start = kv_cache.seq_len
            token_types = token_types[:, start:]
            token_features = token_features[:, start:]
            attn_mask = attn_mask[:, start:]
            positions = torch.arange(
                start, start + token_types.shape[1], device=token_features.device
            ).unsqueeze(0).expand(b, -1)
        else:
            positions = torch.arange(
                seq_len, device=token_features.device
            ).unsqueeze(0).expand(b, -1)

        x = self.embed_tokens(token_types, token_features, positions)

        if incremental:
            causal = None
            key_padding_mask = None
        else:
            causal = self._build_causal_mask(seq_len, x.device)
            key_padding_mask = ~attn_mask if attn_mask is not None else None

        hidden, new_cache = self.backbone(
            x,
            attn_mask=causal,
            key_padding_mask=key_padding_mask,
            kv_cache=kv_cache,
            use_cache=use_cache or incremental,
        )

        h_last = hidden[:, -1, :]
        queries = self._opponent_queries(
            h_last, hero_seat, skill_bins=skill_bins, table_skill=table_skill
        )

        belief_logits = self.belief_head(queries)
        policy_logits = self.policy_head(queries)

        if card_removal_mask is not None:
            if card_removal_mask.dim() == 2:
                mask = card_removal_mask.unsqueeze(1).expand(-1, 5, -1)
            else:
                mask = card_removal_mask
            belief_logits = belief_logits.masked_fill(~mask, float("-inf"))

        if legal_action_mask is not None:
            policy_logits = policy_logits.masked_fill(~legal_action_mask, float("-inf"))

        out: dict[str, torch.Tensor] = {
            "belief_logits": belief_logits,
            "policy_logits": policy_logits,
            "hidden": hidden,
        }
        if new_cache is not None:
            out["kv_cache"] = new_cache  # type: ignore[assignment]
        return out


def build_clone(config: HumanCloneConfig | None = None) -> HumanCloneNet:
    """Factory mirroring the base-class constructor convention (used by the trainer + league)."""
    return HumanCloneNet(config or HumanCloneConfig())
