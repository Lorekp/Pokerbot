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


def _hero_hand_strength(state: GameState, seat: int) -> float | None:
    """Cheap hero equity proxy in [0, 1] for value/bluff separation in the exploit mixer.

    Preflop uses the WEVA strength proxy; postflop uses the board-aware heuristic. Best-effort:
    any failure returns None so the mixer falls back to its value-neutral tilt rather than
    breaking the decision.
    """
    try:
        player = state.players[seat]
        if player.hole is None:
            return None
        from pokerbot.encoding.weva_encoder import _preflop_strength
        if state.street == Street.PREFLOP or not state.board:
            return float(_preflop_strength(player.hole))
        from pokerbot.eval.abstraction_bots import _hand_strength
        return float(_hand_strength(state, seat))
    except Exception:
        return None


def _n_prior_aggr(state: GameState, seat: int) -> int | None:
    """Raw per-street aggression count for the exploit context key, computed the SAME way as the live
    search env + offline ingest (train/serve-skew firewall). Preflop the whole action history is the
    preflop sequence ⇒ count its raises (full open/3bet/4bet depth). Postflop ⇒ None so the mixer uses
    its reliable CHECK-legal binary (the context spec caps postflop facing at 1 anyway)."""
    if state.street != Street.PREFLOP:
        return None
    from pokerbot.opponent.context import is_aggressive_type

    n = 0
    for _s, act in state.action_history():
        if is_aggressive_type(getattr(act.action_type, "name", str(act.action_type))):
            n += 1
    return n


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
        self._opponent_id_override: int | None = None
        # Evaluation hooks (off by default ⇒ live behavior unchanged). When ``_decision_rng`` is set the
        # action is SAMPLED from the policy (required for AIVAT's decision control variate to be
        # zero-mean — argmax has no randomization to cancel); when ``_decision_capture`` is a list each
        # decision's (probs, taken_idx) is recorded for the exploitation battery / AIVAT.
        self._decision_rng = None
        self._decision_capture: list | None = None
        # Optional GiftRegister (R4 gift-modulated bounded-ε): banked opponent gifts relax the exploit
        # alpha cap per opponent. None ⇒ the static Pmax cap (unchanged). The live driver credits gifts
        # (e.g. from the search's villain leaf-Q) and the modeled villain's bonus is threaded each call.
        self._gift_register = None
        # Optional {opponent_id -> OpponentStats} loaded from the persistent player DB
        # (player_profiles.py). Consumed in __call__: the modeled villain's profile is passed to
        # HybridPolicy.get_probs (which prefers it over the in-memory tracker). None ⇒ unchanged GTO.
        self._opponent_profiles: dict | None = None
        # Master exploit switch (set_exploit_enabled). False ⇒ pure GTO base (clean A/B OFF arm).
        self._exploit_enabled = True
        # Exploit magnitude (set_exploit_strength): scales the mixer alpha. 1.0 ⇒ full overlay (legacy
        # behavior); 0.0 ⇒ pure GTO base; in-between is the skill-tier exploitation dial (see ADAPT).
        self._exploit_strength = 1.0
        # Optional table-toughness router (pokerbot.opponent.base_router): per-decision, per-villain it
        # may swap the GTO base to the CFR blueprint and damp exploit when the modeled villain reads as
        # a near-equilibrium reg (protect vs strong), while keeping the SFT base + full exploit vs weak
        # fields (the value focus). None ⇒ the validated SFT vehicle, byte-unchanged.
        self._base_router = None

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

    def enable_decision_capture(self, rng=None) -> None:
        """Turn on per-decision capture for evaluation. ``rng`` (a ``random.Random``) makes the agent
        SAMPLE from the policy instead of argmax — needed so AIVAT's decision correction is zero-mean.
        Pass ``rng=None`` to capture probs while keeping argmax. Off by default ⇒ live play unchanged."""
        self._decision_capture = []
        self._decision_rng = rng

    def pop_decisions_raw(self) -> list:
        """Return and clear the captured ``(probs(np.ndarray), taken_idx)`` decisions this hand."""
        if self._decision_capture is None:
            return []
        d, self._decision_capture = self._decision_capture, []
        return d

    def set_opponent_id(self, opponent_id: int | None) -> None:
        """Override which seat the exploit mixer profiles (the primary villain). When None
        the legacy ``(seat+1)%6`` placeholder is used. Set per hero decision by the live
        driver after seeding ``self.tracker`` with the seated players' real stats."""
        self._opponent_id_override = opponent_id

    def set_gift_register(self, register) -> None:
        """Attach a GiftRegister (pokerbot.opponent.safety_budget) for gift-modulated exploitation.
        The live driver credits gifts (the EV a villain surrenders, e.g. from the search leaf-Q); the
        modeled villain's banked bonus relaxes the alpha cap each decision. None ⇒ static Pmax cap."""
        self._gift_register = register

    def set_opponent_profiles(self, profiles: dict | None) -> None:
        """Load ``{opponent_id -> OpponentStats}`` from the persistent player DB
        (`pokerbot.opponent.player_profiles.PlayerProfileStore`) for opponent-conditioned
        exploitation. The modeled villain's profile is passed through to ``HybridPolicy.get_probs``
        each decision (it prefers a DB profile over the in-memory tracker). ``None`` ⇒ unchanged GTO.
        See docs/exploitation-design.md."""
        self._opponent_profiles = profiles

    def set_exploit_enabled(self, enabled: bool) -> None:
        """Master exploit switch. False ⇒ HybridPolicy returns the PURE GTO base (alpha=0, no
        profile/tracker/transformer mixing) — the clean A/B OFF arm / a do-no-harm kill-switch.
        Default True (the mixer is live)."""
        self._exploit_enabled = enabled

    def set_exploit_strength(self, strength: float) -> None:
        """Scale the exploitation magnitude (the mixer alpha) in [0, 1]. 1.0 ⇒ the full validated
        read→mixer overlay (unchanged); 0.0 ⇒ pure GTO base (equivalent to exploit disabled). The
        skill-tier ladder maps ADAPT[tier].exploit_strength here (advanced soft … superhuman full)."""
        self._exploit_strength = float(max(0.0, min(1.0, strength)))

    def set_base_router(self, router) -> None:
        """Attach a ``pokerbot.opponent.base_router.BaseRouter`` (or ``None`` to detach). When set, each
        decision the router inspects the modeled villain's profile and may (a) damp exploit_strength and
        (b) swap the GTO base to the CFR blueprint — protecting against confirmed regs while leaving the
        SFT base + full exploit vs weak fields (the value focus). None ⇒ the validated SFT vehicle."""
        self._base_router = router

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

        opponent_id = self._opponent_id_override if self._opponent_id_override is not None else (seat + 1) % 6
        # Persistent-DB profile for the modeled villain (loaded via set_opponent_profiles, keyed by the
        # same id space as opponent_id). None ⇒ HybridPolicy falls back to its in-memory tracker.
        opponent_profile = (self._opponent_profiles or {}).get(opponent_id)
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

        # Table-toughness routing (task (b)): on a confirmed-reg villain, damp exploit and fall back to
        # the blueprint base (protect vs strong); on weak/unknown villains keep the SFT base + full
        # exploit (the value focus). Only active when a router is attached AND exploitation is on — the
        # clean A/B OFF arm stays the pure SFT GTO base regardless of toughness.
        eff_strength = self._exploit_strength
        base_override = None
        if self._base_router is not None and self._exploit_enabled:
            decision = self._base_router.decide(opponent_profile)
            eff_strength *= decision.exploit_scale
            if decision.use_blueprint:
                base_override = self._base_router.base_probs(state, seat)

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
            opponent_profile=opponent_profile,
            hand_strength=_hero_hand_strength(state, seat),
            n_prior_aggr=_n_prior_aggr(state, seat),
            gift_bonus=(self._gift_register.alpha_bonus(opponent_id) if self._gift_register else 0.0),
            exploit=self._exploit_enabled,
            exploit_strength=eff_strength,
            base_probs_override=base_override,
        )
        if self._decision_rng is not None:
            weights = [max(0.0, float(x)) for x in probs]
            tot = sum(weights)
            idx = (int(self._decision_rng.choices(range(len(probs)), weights=weights)[0])
                   if tot > 0 else int(probs.argmax()))
        else:
            idx = int(probs.argmax())
        if self._decision_capture is not None:
            import numpy as _np

            self._decision_capture.append((_np.asarray(probs, dtype=float).copy(), idx))
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
