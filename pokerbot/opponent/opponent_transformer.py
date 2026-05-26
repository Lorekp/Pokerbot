"""MADT-style autoregressive opponent transformer (GPT decoder + belief/policy heads)."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from pokerbot.engine.actions import NUM_ACTIONS
from pokerbot.engine.game import NUM_PLAYERS
from pokerbot.opponent.sequence_encoder import (
    NUM_BELIEF_COMBOS,
    STATE_FEATURE_DIM,
    TOKEN_FEATURE_DIM,
    TokenType,
    build_card_removal_mask,
)


@dataclass
class OpponentTransformerConfig:
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 6
    d_ff: int = 512
    max_seq_len: int = 256
    num_actions: int = NUM_ACTIONS
    num_belief_combos: int = NUM_BELIEF_COMBOS
    dropout: float = 0.1


@dataclass
class KVCache:
    """Per-layer key/value cache for incremental inference."""

    keys: list[torch.Tensor] = field(default_factory=list)
    values: list[torch.Tensor] = field(default_factory=list)

    def clear(self) -> None:
        self.keys.clear()
        self.values.clear()

    @property
    def seq_len(self) -> int:
        if not self.keys:
            return 0
        return self.keys[0].shape[1]


class CausalDecoderBlock(nn.Module):
    """Single GPT-style decoder block with SDPA."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model,
            n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        key_padding_mask: torch.Tensor | None = None,
        past_k: torch.Tensor | None = None,
        past_v: torch.Tensor | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        h = self.ln1(x)
        q = h
        k = h
        v = h

        if past_k is not None and past_v is not None:
            k = torch.cat([past_k, k], dim=1)
            v = torch.cat([past_v, v], dim=1)

        new_k, new_v = (k, v) if use_cache else (None, None)

        if past_k is not None:
            query_len = q.shape[1]
            full_len = k.shape[1]
            causal = torch.zeros(query_len, full_len, dtype=torch.float32, device=x.device)
            causal[:, -query_len:] = torch.triu(
                torch.ones(query_len, query_len, dtype=torch.float32, device=x.device),
                diagonal=1,
            ) * -1e9
            attn_mask = causal

        attn_out, _ = self.attn(
            q,
            k,
            v,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )

        x = x + attn_out
        x = x + self.ff(self.ln2(x))
        return x, new_k, new_v


class GPTBackbone(nn.Module):
    """Stack of causal decoder blocks."""

    def __init__(self, config: OpponentTransformerConfig) -> None:
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList(
            [
                CausalDecoderBlock(config.d_model, config.n_heads, config.d_ff, config.dropout)
                for _ in range(config.n_layers)
            ]
        )
        self.ln_f = nn.LayerNorm(config.d_model)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        key_padding_mask: torch.Tensor | None = None,
        kv_cache: KVCache | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, KVCache | None]:
        new_cache = KVCache() if use_cache else None
        for i, layer in enumerate(self.layers):
            past_k = kv_cache.keys[i] if kv_cache and i < len(kv_cache.keys) else None
            past_v = kv_cache.values[i] if kv_cache and i < len(kv_cache.values) else None
            x, nk, nv = layer(
                x,
                attn_mask=attn_mask,
                key_padding_mask=key_padding_mask,
                past_k=past_k,
                past_v=past_v,
                use_cache=use_cache,
            )
            if use_cache and new_cache is not None and nk is not None and nv is not None:
                new_cache.keys.append(nk)
                new_cache.values.append(nv)
        return self.ln_f(x), new_cache


class BeliefHead(nn.Module):
    def __init__(self, d_model: int, num_combos: int) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, num_combos)

    def forward(self, q: torch.Tensor, removal_mask: torch.Tensor | None = None) -> torch.Tensor:
        logits = self.proj(q)
        if removal_mask is not None:
            logits = logits.masked_fill(~removal_mask, float("-inf"))
        return logits


class PolicyHead(nn.Module):
    def __init__(self, d_model: int, num_actions: int) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, num_actions)

    def forward(self, q: torch.Tensor, legal_mask: torch.Tensor | None = None) -> torch.Tensor:
        logits = self.proj(q)
        if legal_mask is not None:
            logits = logits.masked_fill(~legal_mask, float("-inf"))
        return logits


class OpponentTransformer(nn.Module):
    """
    Autoregressive opponent model predicting hole-card beliefs and action policies
    for all five non-hero seats simultaneously.
    """

    def __init__(self, config: OpponentTransformerConfig | None = None) -> None:
        super().__init__()
        self.config = config or OpponentTransformerConfig()
        c = self.config

        self.type_embed = nn.Embedding(len(TokenType), c.d_model)
        self.state_proj = nn.Linear(STATE_FEATURE_DIM, c.d_model)
        self.action_proj = nn.Linear(TOKEN_FEATURE_DIM, c.d_model)
        self.reward_proj = nn.Linear(1, c.d_model)
        self.pos_embed = nn.Embedding(c.max_seq_len, c.d_model)

        self.seat_embed = nn.Embedding(NUM_PLAYERS, c.d_model)
        self.role_embed = nn.Embedding(2, c.d_model)  # 0=hero context, 1=opponent query

        self.backbone = GPTBackbone(c)
        self.belief_head = BeliefHead(c.d_model, c.num_belief_combos)
        self.policy_head = PolicyHead(c.d_model, c.num_actions)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def embed_tokens(
        self,
        token_types: torch.Tensor,
        token_features: torch.Tensor,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Fuse type-specific projections into token embeddings."""
        b, seq_len, _ = token_features.shape
        if positions is None:
            positions = torch.arange(seq_len, device=token_types.device).unsqueeze(0).expand(b, -1)

        type_emb = self.type_embed(token_types.clamp(0, len(TokenType) - 1))

        state_mask = token_types == TokenType.STATE
        action_mask = token_types == TokenType.ACTION
        reward_mask = token_types == TokenType.REWARD

        projected = torch.zeros(b, seq_len, self.config.d_model, device=token_features.device)
        if state_mask.any():
            projected[state_mask] = self.state_proj(token_features[state_mask][:, :STATE_FEATURE_DIM])
        if action_mask.any():
            projected[action_mask] = self.action_proj(token_features[action_mask])
        if reward_mask.any():
            projected[reward_mask] = self.reward_proj(
                token_features[reward_mask][:, :1]
            )

        pos = self.pos_embed(positions.clamp(0, self.config.max_seq_len - 1))
        return type_emb + projected + pos

    def _build_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        mask = torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.float32, device=device),
            diagonal=1,
        )
        return mask * -1e9

    def _opponent_queries(self, h_last: torch.Tensor, hero_seat: torch.Tensor) -> torch.Tensor:
        """Form five opponent query vectors from last hidden state."""
        b = h_last.shape[0]
        opps = []
        role = self.role_embed(torch.tensor(1, device=h_last.device))
        for offset in range(1, NUM_PLAYERS):
            seat_ids = (hero_seat + offset) % NUM_PLAYERS
            seat_emb = self.seat_embed(seat_ids)
            q = h_last + seat_emb + role
            opps.append(q)
        return torch.stack(opps, dim=1)

    def forward(
        self,
        token_types: torch.Tensor,
        token_features: torch.Tensor,
        attn_mask: torch.Tensor,
        hero_seat: torch.Tensor,
        kv_cache: KVCache | None = None,
        card_removal_mask: torch.Tensor | None = None,
        legal_action_mask: torch.Tensor | None = None,
        use_cache: bool = False,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            token_types: (B, L) int
            token_features: (B, L, F)
            attn_mask: (B, L) bool — True for valid positions
            hero_seat: (B,) int
            kv_cache: optional incremental cache
            card_removal_mask: (B, 1326) bool — legal combos
            legal_action_mask: (B, 5, 11) bool — optional per-opponent legal actions
        """
        b, seq_len, _ = token_features.shape
        incremental = kv_cache is not None and kv_cache.seq_len > 0

        if incremental:
            start = kv_cache.seq_len
            token_types = token_types[:, start:]
            token_features = token_features[:, start:]
            attn_mask = attn_mask[:, start:]
            positions = torch.arange(start, start + token_types.shape[1], device=token_features.device)
            positions = positions.unsqueeze(0).expand(b, -1)
        else:
            positions = torch.arange(seq_len, device=token_features.device).unsqueeze(0).expand(b, -1)

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

        if incremental and kv_cache is not None:
            full_hidden = hidden
        else:
            full_hidden = hidden

        h_last = full_hidden[:, -1, :]
        queries = self._opponent_queries(h_last, hero_seat)

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
            "hidden": full_hidden,
        }
        if new_cache is not None:
            out["kv_cache"] = new_cache  # type: ignore[assignment]
        return out

    @torch.no_grad()
    def predict(
        self,
        token_types: torch.Tensor,
        token_features: torch.Tensor,
        attn_mask: torch.Tensor,
        hero_seat: torch.Tensor,
        kv_cache: KVCache | None = None,
        hero_hole_indices: tuple[int, int] | None = None,
        board_indices: list[int] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Low-latency inference helper with optional KV cache."""
        self.eval()
        removal = None
        if hero_hole_indices is not None:
            from pokerbot.engine.cards import Card, Rank, Suit

            hero = (
                Card(Rank(hero_hole_indices[0] // 4 + 2), Suit(hero_hole_indices[0] % 4)),
                Card(Rank(hero_hole_indices[1] // 4 + 2), Suit(hero_hole_indices[1] % 4)),
            )
            board = []
            if board_indices:
                for idx in board_indices:
                    board.append(Card(Rank(idx // 4 + 2), Suit(idx % 4)))
            removal = build_card_removal_mask(hero, board, device=token_features.device)
            removal = removal.unsqueeze(0)

        return self.forward(
            token_types,
            token_features,
            attn_mask,
            hero_seat,
            kv_cache=kv_cache,
            card_removal_mask=removal,
            use_cache=True,
        )
