"""Offline SearchAgent for sim harness — mirrors live PBS + TurboReBeL path."""

from __future__ import annotations

import os
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
from pokerbot.search.pluribus_search import PluribusReBeLConfig, PluribusReBeLSolver
from pokerbot.search.pluribus_rollout import ContinuationLeafConfig
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
        use_pluribus_search: bool = False,
        blueprint_snapshot: dict | None = None,
        pluribus_leaf_select: str = "uniform",
        root_exploit_mixer: bool = False,
        root_mixer_unopened: bool = False,
        aivat_leaf_q: bool = True,
        search_preflop: bool = False,
    ) -> None:
        self.device = torch.device(device)
        self.time_budget_ms = time_budget_ms
        self.use_pluribus_search = use_pluribus_search
        self._state = SearchAgentState(hero_seat=hero_seat)
        self._state.trajectory.reset(hero_seat)

        if use_pluribus_search:
            # Pluribus-faithful path: tabular blueprint + continuation-gadget leaf. The poisoned neural
            # stack (HistoryValueEvaluator / AverageStrategyNet) is NEVER constructed here — the leaf is
            # the continuation gadget, so no value_checkpoint is loaded and the gadget is always "ready".
            self.leaf_eval = None
            self._value_ready = True
            self.solver = PluribusReBeLSolver(
                blueprint_snapshot or {},
                PluribusReBeLConfig(
                    time_budget_ms=time_budget_ms,
                    leaf=ContinuationLeafConfig(select=pluribus_leaf_select),
                    # search_preflop: LBR audit found the base is exploitable ~+180 bb/100 almost
                    # entirely PREFLOP (the blueprint open/3bet ranges are too wide), and that searching
                    # the preflop eliminates it (+180 -> <=0). Deployed paths pass True. Default False
                    # keeps the eval/battery (and the basestrength +9.1 baseline) reproducible.
                    search_preflop=search_preflop,
                    # RNR-restrict the solver BR (#28). Default OFF in eval (byte-identical baselines);
                    # POKERBOT_RNR_SOLVER_EVAL=1 enables the A/B that validates the restriction on the
                    # search path (set_opponent_profiles then feeds per-seat stability). Live is ON via
                    # the bridge's own POKERBOT_RNR_SOLVER gate.
                    rnr_restrict=os.environ.get("POKERBOT_RNR_SOLVER_EVAL", "0") not in ("0", "", "false", "False"),
                ),
            )
        else:
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

        self._decision_rng = None
        self._decision_capture: list | None = None
        # Root-policy exploitation: apply the VALIDATED read->mixer_logits->safe_mix adjustment (the
        # same mechanism the heuristic battery hero proved extracts CI>0 value from leaky archetypes)
        # to the HERO'S OWN post-search policy. The search supplies the GTO base; the mixer supplies
        # the exploitation. This also covers PREFLOP (served from the blueprint, where the solver's
        # opponent-node exploit tail never runs). Off by default; with no profiles set it is inert.
        self.root_exploit_mixer = root_exploit_mixer
        # Re-enable the mixer's UNOPENED branch (tilt our own steals/value-bets from the villain's
        # response read). Its nine-round losses were all measured against the range-free caricature
        # bots; the fair-bot battery arbitrates. Default False = the validated facing-only subset.
        self.root_mixer_unopened = root_mixer_unopened
        self._opponent_profiles: dict | None = None
        # AIVAT per-action baseline source. True (default) = the rigorous continuation-gadget Q-row
        # (``solver._leaf_q``), which re-runs the leaf gadget for every legal hero action at every
        # captured decision — measured at ~44% of the pluribus battery's wall time, entirely OUTSIDE
        # ``solve()``. False = skip it and emit ``bv=None`` so the capturing adapter (ProductionHero)
        # falls back to its cheap crude baseline. The estimator is unbiased for ANY baseline, so this
        # changes ONLY the variance reduction on the SECONDARY per-arm absolute-EV columns
        # (on_aivat/off_aivat) — the headline on-off ``uplift_bb_per_hand`` (raw chip delta), its SE,
        # t, CI, zero_frac and bet_shift are byte-unchanged. Default True preserves the existing path.
        self.aivat_leaf_q = aivat_leaf_q

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

        # TABLE-AGGRESSION KILL-SWITCH: when >= AGG_KILL_COUNT seated villains are confirmed lags, disable
        # the WHOLE exploit (solver opponent-tail AND root mixer) for this decision — the exploit is +EV on
        # every realistic field incl. a 5-tag pro table (+0.99) and loses only vs a lag-dominated table
        # (-0.48). Keys on LOOSE aggression so it fires on lags, not on profitable tags. Re-evaluated each
        # decision; opponent_model is set explicitly so a prior kill never lingers.
        from pokerbot.opponent.root_mixer import _table_too_aggressive

        kill = bool(self._opponent_profiles) and _table_too_aggressive(self._opponent_profiles)
        self.solver.opponent_model = None if kill else self._opponent_profiles

        root = build_subgame_root(state, pbs, hero)
        result = self.solver.solve(root, pbs)
        self._state.last_solve_metadata = dict(result.metadata)
        self._state.last_solve_metadata["iterations"] = float(result.iterations)
        self._state.last_solve_metadata["ev"] = result.ev

        probs = np.asarray(result.action_probs, dtype=float)
        if self.root_exploit_mixer and self._opponent_profiles and not kill:
            probs = self._apply_root_mixer(state, seat, probs)
        if self._decision_rng is not None and probs.sum() > 0:
            w = np.clip(probs, 0.0, None)
            idx = int(self._decision_rng.choices(range(len(w)), weights=w.tolist())[0])
        else:
            idx = int(probs.argmax())
        if self._decision_capture is not None:
            # The leaf value net's Q-row at the decision node = the rigorous per-action AIVAT baseline
            # (value-net AIVAT). Cached in the solver from the just-completed solve. Skipped when
            # ``aivat_leaf_q`` is False (the gadget Q-row is ~44% of the pluribus battery wall time and
            # only feeds the SECONDARY absolute-EV variance reduction): emit bv=None so the capturing
            # adapter falls back to its crude baseline. Headline uplift/SE/CI/zero_frac/bet_shift are
            # unchanged either way (the on-off uplift uses raw chips, not the AIVAT-adjusted columns).
            if self.aivat_leaf_q:
                try:
                    bv = np.asarray(self.solver._leaf_q(root.env), dtype=float).reshape(-1)
                except Exception:  # noqa: BLE001
                    bv = None
            else:
                bv = None
            self._decision_capture.append((probs.copy(), bv, idx))
        action = _index_to_action(state, seat, idx)
        self._state.trajectory.on_action(state, seat, action)
        return action

    def set_opponent_profiles(self, profiles: dict | None) -> None:
        """Wire opponent profiles into BOTH exploitation seams: the solver's opponent-node tail
        (postflop, inside the search) and the root mixer (the hero's own policy, all streets).
        ``None`` disables both — the zero-deviation arm."""
        self._opponent_profiles = profiles
        self.solver.opponent_model = profiles
        # RNR restriction inputs (#28): per-seat stability from each profile's own fast/slow divergence,
        # so a caught adapter collapses the solver's modeled deviation toward CAP_FLOOR. No-op unless
        # config.rnr_restrict is on; N falls back to the model size (_rnr_n_live) when opponent_n_live unset.
        if profiles and getattr(self.solver.config, "rnr_restrict", False):
            from pokerbot.opponent.bayes_exploit import profile_stability
            self.solver.opponent_stability = {
                s: profile_stability(p) for s, p in profiles.items() if p is not None}
        else:
            self.solver.opponent_stability = None

    def _apply_root_mixer(self, state: GameState, seat: int, probs: np.ndarray) -> np.ndarray:
        """Delegate to the shared root mixer (pokerbot.opponent.root_mixer) — the validated
        read->mixer_logits->safe_mix step, extracted verbatim so the eval mirror and the live
        bridge share one implementation. Behavior with ``root_mixer_unopened=False`` (default)
        is the battle-tested facing-only subset, byte-identical to the pre-extraction code."""
        from pokerbot.opponent.root_mixer import apply_root_exploit_mixer

        return apply_root_exploit_mixer(state, seat, probs, self._opponent_profiles,
                                        unopened=self.root_mixer_unopened)

    def enable_decision_capture(self, rng=None) -> None:
        """Capture per-decision ``(action_probs, leaf_q_row, taken_idx)`` for the exploitation battery
        / AIVAT. ``rng`` makes the agent SAMPLE from the search strategy (needed for AIVAT's decision
        control variate to be zero-mean). Off by default ⇒ argmax play unchanged."""
        self._decision_capture = []
        self._decision_rng = rng
        # Arm the solver's ROOT per-action-CFV capture so _leaf_q returns the served-policy-CONSISTENT
        # AIVAT baseline (positive variance reduction). Only when aivat_leaf_q is on (else bv=None and the
        # capturer uses its crude fallback). This also forces the Python CFR loop (the native fast path
        # cannot expose per-action root CFVs) -- acceptable for eval, where AIVAT buys far fewer hands.
        if getattr(self, "solver", None) is not None:
            self.solver._aivat_capture = bool(self.aivat_leaf_q)

    def pop_decisions_raw(self) -> list:
        if self._decision_capture is None:
            return []
        d, self._decision_capture = self._decision_capture, []
        return d

    @property
    def last_solve_metadata(self) -> dict[str, float]:
        return dict(self._state.last_solve_metadata)
