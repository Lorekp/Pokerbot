"""Hybrid agent combining GTO policy + MADT opponent adaptation."""

from __future__ import annotations

from pathlib import Path

import torch

from pokerbot.engine.actions import Action
from pokerbot.engine.game import GameState, Street
from pokerbot.encoding.eval_scope import encode_state_for_policy
from pokerbot.opponent.mixer import HybridPolicy
from pokerbot.opponent.opponent_transformer import KVCache, OpponentTransformer, OpponentTransformerConfig
from pokerbot.opponent.sequence_encoder import HandTrajectoryBuilder, trajectory_to_batch
from pokerbot.opponent.stats import OpponentTracker
from pokerbot.opponent.transformer import HandHistoryBuffer, HandRecord
from pokerbot.training.self_play import _index_to_action, _legal_mask

DEFAULT_TRANSFORMER_CKPT = "checkpoints/opponent_transformer_madt.pt"
FALLBACK_TRANSFORMER_CKPT = "checkpoints/opponent_transformer_pretrained.pt"


class HybridAgent:
    """Production-facing 6-max agent with GTO base + MADT exploit mixing."""

    def __init__(
        self,
        model,
        device: str = "cpu",
        use_transformer: bool = True,
        opponent_transformer_ckpt: str | None = DEFAULT_TRANSFORMER_CKPT,
    ) -> None:
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.model.eval()
        self.tracker = OpponentTracker()
        self.history = HandHistoryBuffer()
        self.transformer = None
        self._trajectory = HandTrajectoryBuilder(hero_seat=0)
        self._kv_cache: KVCache | None = None
        self._last_token_count = 0
        self._hero_seat = 0

        if use_transformer:
            self.transformer = OpponentTransformer(OpponentTransformerConfig()).to(self.device)
            ckpt_path = self._resolve_ckpt(opponent_transformer_ckpt)
            if ckpt_path:
                ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
                self.transformer.load_state_dict(ckpt["model"])
                self.transformer.eval()
                print(f"Loaded MADT opponent transformer from {ckpt_path}")

        self.hybrid = HybridPolicy(
            self.model,
            self.tracker,
            self.transformer,
            device=self.device,
        )
        self.eval_spot_class: str | None = None
        self.require_scope = False

    @staticmethod
    def _resolve_ckpt(path: str | None) -> Path | None:
        for candidate in (path, DEFAULT_TRANSFORMER_CKPT, FALLBACK_TRANSFORMER_CKPT):
            if candidate and Path(candidate).exists():
                return Path(candidate)
        return None

    def reset_hand(self, hero_seat: int) -> None:
        self._hero_seat = hero_seat
        self._trajectory.reset(hero_seat)
        self._kv_cache = KVCache()
        self._last_token_count = 0

    def observe_decision(self, state: GameState, actor_seat: int) -> None:
        self._trajectory.on_decision(state, actor_seat)

    def observe_action(self, state: GameState, seat: int, action: Action) -> None:
        self._trajectory.on_action(state, seat, action)

    def _transformer_outputs(self, state: GameState, hero_seat: int) -> dict | None:
        if self.transformer is None or not self._trajectory.token_types:
            return None

        traj = self._trajectory.finalize()
        batch = trajectory_to_batch([traj], device=self.device)

        incremental = self._kv_cache is not None and self._last_token_count > 0
        if incremental and len(traj.token_types) <= self._last_token_count:
            self._kv_cache = KVCache()
            incremental = False

        hero_hole = state.players[hero_seat].hole
        board = state.board
        removal = None
        if hero_hole:
            from pokerbot.opponent.sequence_encoder import build_card_removal_mask

            removal = build_card_removal_mask(hero_hole, board, device=self.device).unsqueeze(0)

        with torch.no_grad():
            out = self.transformer(
                batch["token_types"],
                batch["token_features"],
                batch["attn_mask"],
                batch["hero_seat"],
                kv_cache=self._kv_cache if incremental else None,
                card_removal_mask=removal,
                use_cache=True,
            )
        if "kv_cache" in out:
            self._kv_cache = out["kv_cache"]
        self._last_token_count = len(traj.token_types)
        return out

    def __call__(self, state: GameState) -> Action:
        seat = state.action_seat
        hero_seat = getattr(self, "_hero_seat", seat)

        if len(self._trajectory.token_types) == 0:
            self._trajectory.init_stacks(state)

        self.observe_decision(state, seat)

        opponent_id = (seat + 1) % 6
        enc = encode_state_for_policy(
            state,
            seat,
            spot_class=self.eval_spot_class,
            require_scope=self.require_scope,
        )
        weva_ids = torch.tensor([enc["weva_id"]], dtype=torch.long, device=self.device)
        cards = torch.from_numpy(enc["cards"]).unsqueeze(0).to(self.device)
        actions = torch.from_numpy(enc["actions"]).unsqueeze(0).to(self.device)
        position = torch.from_numpy(enc["policy_position"]).unsqueeze(0).to(self.device)
        legal = torch.from_numpy(_legal_mask(state, seat)).unsqueeze(0).to(self.device)

        tf_out = self._transformer_outputs(state, hero_seat)
        probs = self.hybrid.get_probs(
            weva_ids,
            cards,
            actions,
            position,
            legal,
            opponent_id,
            transformer_out=tf_out,
            actor_seat=seat,
            hero_seat=hero_seat,
            board=state.board,
            trajectory=self._trajectory,
        )
        idx = int(probs.argmax())
        action = _index_to_action(state, seat, idx)
        self.observe_action(state, seat, action)
        return action

    def record_showdown(self, state: GameState, hero_seat: int, won: bool) -> None:
        self._trajectory.record_showdown(state)
        for p in state.players:
            if p.seat == hero_seat:
                continue
            saw_showdown = state.street == Street.SHOWDOWN and not p.folded
            self.history.add(
                p.seat,
                HandRecord(won=int(won), showdown=int(saw_showdown)),
            )
