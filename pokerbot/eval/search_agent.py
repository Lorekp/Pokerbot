"""Offline SearchAgent for sim harness — mirrors live PBS + TurboReBeL path."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from pokerbot.engine.actions import Action, action_to_index
from pokerbot.engine.game import GameState
from pokerbot.opponent.opponent_transformer import OpponentTransformer, OpponentTransformerConfig
from pokerbot.opponent.sequence_encoder import HandTrajectoryBuilder, trajectory_to_batch
from pokerbot.search.leaf_evaluator import HistoryValueEvaluator, MockHistoryValueEvaluator
from pokerbot.search.pbs_tracker import PBSTracker
from pokerbot.search.state_bridge import build_subgame_root
from pokerbot.search.turbo_rebel_search import TurboReBeLConfig, TurboReBeLSolver
from pokerbot.training.self_play import _index_to_action

DEFAULT_VALUE_CKPT = Path("checkpoints/history_value_net.pkl")
DEFAULT_TRANSFORMER_CKPT = Path("checkpoints/opponent_transformer_madt.pt")


@dataclass
class SearchAgentState:
    pbs: PBSTracker | None = None
    trajectory: HandTrajectoryBuilder = field(default_factory=lambda: HandTrajectoryBuilder(hero_seat=0))
    last_board_len: int = 0
    hero_seat: int = 0
    applied_action_count: int = 0
    last_solve_metadata: dict[str, float] = field(default_factory=dict)


class SearchAgent:
    """GameState sim bot using TurboReBeL + PBS (offline mirror of LiveSearchAdvisor)."""

    def __init__(
        self,
        *,
        value_checkpoint: str | Path | None = str(DEFAULT_VALUE_CKPT),
        transformer_checkpoint: str | Path | None = str(DEFAULT_TRANSFORMER_CKPT),
        device: str = "cpu",
        time_budget_ms: int = 2000,
        max_depth: int = 4,
        use_mock_leaf: bool = False,
        hero_seat: int = 0,
    ) -> None:
        self.device = torch.device(device)
        self.time_budget_ms = time_budget_ms
        self._state = SearchAgentState(hero_seat=hero_seat)
        self._state.trajectory.reset(hero_seat)

        ckpt = Path(value_checkpoint) if value_checkpoint else DEFAULT_VALUE_CKPT
        if use_mock_leaf or not ckpt.exists():
            self.leaf_eval = MockHistoryValueEvaluator()
            self._value_ready = use_mock_leaf
        else:
            self.leaf_eval = HistoryValueEvaluator(ckpt)
            self._value_ready = self.leaf_eval.has_trained_checkpoint

        self.solver = TurboReBeLSolver(
            self.leaf_eval,
            TurboReBeLConfig(time_budget_ms=time_budget_ms, max_depth=max_depth),
        )

        self.transformer: OpponentTransformer | None = None
        tf_path = Path(transformer_checkpoint) if transformer_checkpoint else DEFAULT_TRANSFORMER_CKPT
        if tf_path.exists():
            self.transformer = OpponentTransformer(OpponentTransformerConfig()).to(self.device)
            tf_ckpt = torch.load(tf_path, map_location=self.device, weights_only=False)
            self.transformer.load_state_dict(tf_ckpt["model"])
            self.transformer.eval()

    def reset_hand(self, hero_seat: int) -> None:
        self._state = SearchAgentState(hero_seat=hero_seat)
        self._state.trajectory.reset(hero_seat)

    def _init_pbs(self, state: GameState) -> None:
        hero = self._state.hero_seat
        hole = state.players[hero].hole
        beliefs = self._madt_beliefs(state, hero)
        if beliefs is not None:
            self._state.pbs = PBSTracker.init_from_transformer(beliefs, hero, hole, state.board)
        else:
            self._state.pbs = PBSTracker.init_uniform(hero, hole, state.board)
        self._state.trajectory.reset(hero)
        self._state.trajectory.init_stacks(state)
        self._state.last_board_len = len(state.board)

    def _madt_beliefs(self, state: GameState, hero_seat: int) -> np.ndarray | None:
        if self.transformer is None or not self._state.trajectory.token_types:
            return None
        batch = trajectory_to_batch([self._state.trajectory.finalize()], device=self.device)
        hero_hole = state.players[hero_seat].hole
        removal = None
        if hero_hole:
            from pokerbot.opponent.sequence_encoder import build_card_removal_mask

            removal = build_card_removal_mask(hero_hole, state.board, device=self.device).unsqueeze(0)
        with torch.no_grad():
            out = self.transformer(
                batch["token_types"],
                batch["token_features"],
                batch["attn_mask"],
                batch["hero_seat"],
                card_removal_mask=removal,
            )
        return torch.softmax(out["belief_logits"], dim=-1)[0].cpu().numpy()

    def _catch_up_from_state(self, state: GameState) -> None:
        """Apply opponent/hero actions observed in sim since last hero decision."""
        hero = self._state.hero_seat
        history = list(state.action_history())
        if len(history) < self._state.applied_action_count:
            self._init_pbs(state)
            self._state.applied_action_count = 0

        if self._state.pbs is None:
            self._init_pbs(state)

        pbs = self._state.pbs
        assert pbs is not None
        for i in range(self._state.applied_action_count, len(history)):
            seat, action = history[i]
            if seat != hero:
                idx = action_to_index(action)
                policy_like = self._action_likelihood(state, seat, idx, pbs)
                pbs.update_action(seat, idx, policy_like)
            self._state.trajectory.on_action(state, seat, action)

        self._state.applied_action_count = len(history)

        board_len = len(state.board)
        if board_len > self._state.last_board_len:
            for card in state.board[self._state.last_board_len :]:
                pbs.update_board(card)
            self._state.last_board_len = board_len

    def _sync_pbs(self, state: GameState, seat: int, action: Action | None) -> None:
        self._catch_up_from_state(state)

    def _action_likelihood(
        self,
        state: GameState,
        seat: int,
        action_idx: int,
        pbs: PBSTracker,
    ) -> np.ndarray | None:
        if self.transformer is None:
            return None
        from pokerbot.search.belief_likelihood import combo_action_likelihood_from_transformer

        return combo_action_likelihood_from_transformer(
            self.transformer,
            self._state.trajectory,
            self._state.hero_seat,
            seat,
            action_idx,
            pbs._combo_mask,
            pbs.beliefs[seat],
            self.device,
            board=state.board,
            full_1326=True,
        )

    def __call__(self, state: GameState) -> Action:
        if not self._value_ready:
            raise RuntimeError("SearchAgent requires HistoryValueNet checkpoint or use_mock_leaf=True")

        seat = state.action_seat
        hero = self._state.hero_seat

        if len(self._state.trajectory.token_types) == 0:
            self._init_pbs(state)

        self._state.trajectory.on_decision(state, seat)
        self._sync_pbs(state, seat=None, action=None)

        pbs = self._state.pbs
        assert pbs is not None

        root = build_subgame_root(state, pbs, hero)
        result = self.solver.solve(root, pbs)
        self._state.last_solve_metadata = dict(result.metadata)
        self._state.last_solve_metadata["iterations"] = float(result.iterations)
        self._state.last_solve_metadata["ev"] = result.ev

        idx = int(result.action_probs.argmax())
        action = _index_to_action(state, seat, idx)
        self._state.trajectory.on_action(state, seat, action)
        return action

    @property
    def last_solve_metadata(self) -> dict[str, float]:
        return dict(self._state.last_solve_metadata)
