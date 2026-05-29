"""Policy inference engine with batching, compile, and AMP."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from aof_rl.encode_v3 import encode_obs_v3
from aof_rl.game_state import AoFEncodeError, AoFGameState
from aof_rl.inference.legal_mask import apply_legal_mask_to_obs, legal_mask_for_game
from aof_rl.inference.manifest import ModelManifest, resolve_manifest, verify_manifest
from aof_rl.inference.ood import OodConfig, OodMonitor, OodResult
from aof_rl.inference.state_io import action_name, game_state_to_hand_state
from aof_rl.obs_v3 import ObsV3
from aof_rl.obs_v3_batch import ObsV3Batch, collate_obs_v3
from aof_rl.policy_v3 import AoFPolicyNetV3, PolicyOutputV3
from aof_rl.registry import load_league_checkpoint
from aof_rl.teacher import NashTeacherCache
from aof_rl.teacher_targets import teacher_action_index, teacher_probs, teacher_q_bb

logger = logging.getLogger(__name__)


@dataclass
class InferenceResult:
    chosen_action: str
    action_logits: list[float]
    q_bb: list[float]
    uncertainty_bb: float
    state_value_bb: float
    action_probs: list[float]
    legal_mask: list[float]
    obs_version: int
    model_id: str
    ood: OodResult
    teacher: dict[str, Any] | None = None


@dataclass
class ServerConfig:
    checkpoint_path: Path
    manifest_path: Path | None = None
    reference_histogram_path: Path | None = None
    ensemble_checkpoint_path: Path | None = None
    device: str = "cpu"
    use_compile: bool = False
    use_amp: bool = False
    allow_side_pots: bool = False
    skip_manifest_verify: bool = False
    teacher_version: int = 2
    model_id: str | None = None


class InferenceEngine:
    def __init__(self, config: ServerConfig) -> None:
        self.config = config
        self.device = torch.device(config.device)
        self._ready = False
        self.manifest: ModelManifest | None = None
        self.manifest_verified = False
        self.meta: dict[str, Any] = {}
        self.model: AoFPolicyNetV3 | None = None
        self.ensemble_model: AoFPolicyNetV3 | None = None
        self.teacher_cache = NashTeacherCache(
            use_fast=True,
            teacher_version=config.teacher_version,
        )
        self.ood_monitor: OodMonitor | None = None
        self.compile_enabled = False
        self.amp_enabled = config.use_amp

    @property
    def ready(self) -> bool:
        return self._ready

    @property
    def model_id(self) -> str:
        if self.config.model_id:
            return self.config.model_id
        if self.manifest is not None:
            return self.manifest.model_id
        return self.config.checkpoint_path.stem

    def load(self) -> None:
        ckpt_path = self.config.checkpoint_path
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")

        manifest = resolve_manifest(ckpt_path, self.config.manifest_path)
        if manifest is not None and not self.config.skip_manifest_verify:
            verify_manifest(ckpt_path, manifest)
            self.manifest_verified = True
        elif manifest is not None:
            logger.warning("manifest present but verification skipped")
        else:
            logger.warning("no manifest found for checkpoint %s", ckpt_path)

        self.manifest = manifest
        self.model, self.meta = load_league_checkpoint(ckpt_path, self.device)
        self.model.eval()

        if self.config.ensemble_checkpoint_path is not None:
            ens_path = self.config.ensemble_checkpoint_path
            if ens_path.is_file():
                self.ensemble_model, _ = load_league_checkpoint(ens_path, self.device)
                self.ensemble_model.eval()
            else:
                logger.warning("ensemble checkpoint not found: %s", ens_path)

        ood_cfg = OodConfig.from_dict(
            (manifest.ood_config if manifest else None) or {}
        )
        ref_path = self.config.reference_histogram_path
        self.ood_monitor = OodMonitor.from_paths(ood_cfg, ref_path)

        if self.config.use_compile and hasattr(torch, "compile"):
            try:
                self.model = torch.compile(self.model)  # type: ignore[assignment]
                if self.ensemble_model is not None:
                    self.ensemble_model = torch.compile(self.ensemble_model)  # type: ignore[assignment]
                self.compile_enabled = True
            except Exception as exc:
                logger.warning("torch.compile failed: %s", exc)

        self._warmup()
        self._ready = True
        logger.info(
            "inference engine ready model_id=%s device=%s compile=%s amp=%s",
            self.model_id,
            self.device,
            self.compile_enabled,
            self.amp_enabled,
        )

    def _warmup(self) -> None:
        assert self.model is not None
        dummy = self._dummy_game_state()
        obs = encode_obs_v3(dummy)
        legal = legal_mask_for_game(dummy)
        obs = apply_legal_mask_to_obs(obs, legal)
        obs_d = ObsV3(
            hero_hand=obs.hero_hand.to(self.device),
            global_feats=obs.global_feats.to(self.device),
            seat_tokens=obs.seat_tokens.to(self.device),
            seat_mask=obs.seat_mask.to(self.device),
            history_tokens=obs.history_tokens.to(self.device),
            history_mask=obs.history_mask.to(self.device),
            legal_mask=legal.to(self.device),
        )
        with torch.inference_mode():
            if self.amp_enabled and self.device.type == "cuda":
                with torch.autocast(device_type="cuda"):
                    self.model.forward(obs_d)
            else:
                self.model.forward(obs_d)

    def _dummy_game_state(self) -> AoFGameState:
        from aof_rl.game_state import SeatState, game_state_from_dict

        return game_state_from_dict(
            {
                "variant": "NLH_AOF",
                "table_size": 2,
                "hero_seat_id": 0,
                "hero_cards": [0, 1],
                "action_history": [],
                "seats": [
                    {
                        "seat_id": 0,
                        "role": "small_blind",
                        "is_hero": True,
                        "state": "pending",
                        "committed_bb": 0.5,
                        "stack_remaining_bb": 7.5,
                        "blind_posted_bb": 0.5,
                        "shover_rank": 0,
                        "eligible_pot_mask": 1,
                    },
                    {
                        "seat_id": 1,
                        "role": "big_blind",
                        "is_hero": False,
                        "state": "pending",
                        "committed_bb": 1.0,
                        "stack_remaining_bb": 7.0,
                        "blind_posted_bb": 1.0,
                        "shover_rank": 0,
                        "eligible_pot_mask": 2,
                    },
                ],
                "effective_stack_bb": 8.0,
                "equal_stacks": True,
                "side_pots_present": False,
            }
        )

    def validate_side_pots(self, game: AoFGameState) -> None:
        if (
            game.side_pots_present
            and not game.equal_stacks
            and not self.config.allow_side_pots
        ):
            raise AoFEncodeError(
                "unsupported side-pot state (enable allow_side_pots to opt in)"
            )

    def _encode_batch(self, games: list[AoFGameState]) -> ObsV3Batch:
        obs_list: list[ObsV3] = []
        for game in games:
            obs = encode_obs_v3(game)
            legal = legal_mask_for_game(game)
            obs = apply_legal_mask_to_obs(obs, legal)
            obs_list.append(
                ObsV3(
                    hero_hand=obs.hero_hand,
                    global_feats=obs.global_feats,
                    seat_tokens=obs.seat_tokens,
                    seat_mask=obs.seat_mask,
                    history_tokens=obs.history_tokens,
                    history_mask=obs.history_mask,
                    legal_mask=legal,
                )
            )
        batch = collate_obs_v3(obs_list)
        return ObsV3Batch(
            hero_hand=batch.hero_hand.to(self.device),
            global_feats=batch.global_feats.to(self.device),
            seat_tokens=batch.seat_tokens.to(self.device),
            seat_mask=batch.seat_mask.to(self.device),
            history_tokens=batch.history_tokens.to(self.device),
            history_mask=batch.history_mask.to(self.device),
            legal_mask=batch.legal_mask.to(self.device),
        )

    def _forward(self, batch: ObsV3Batch) -> PolicyOutputV3:
        assert self.model is not None
        obs = batch.to_obs_v3()
        with torch.inference_mode():
            if self.amp_enabled and self.device.type == "cuda":
                with torch.autocast(device_type="cuda"):
                    return self.model.forward(obs)
            return self.model.forward(obs)

    def _forward_ensemble(self, batch: ObsV3Batch) -> PolicyOutputV3 | None:
        if self.ensemble_model is None:
            return None
        obs = batch.to_obs_v3()
        with torch.inference_mode():
            if self.amp_enabled and self.device.type == "cuda":
                with torch.autocast(device_type="cuda"):
                    return self.ensemble_model.forward(obs)
            return self.ensemble_model.forward(obs)

    def predict_batch(
        self,
        games: list[AoFGameState],
        *,
        include_teacher: bool = False,
        stochastic: bool = False,
        teacher_samples: int = 200,
        teacher_seed: int = 0,
    ) -> list[InferenceResult]:
        if not self._ready or self.model is None:
            raise RuntimeError("engine not ready")
        if not games:
            return []

        for game in games:
            self.validate_side_pots(game)

        batch = self._encode_batch(games)
        out = self._forward(batch)
        ens_out = self._forward_ensemble(batch)

        results: list[InferenceResult] = []
        b = batch.hero_hand.shape[0]
        for i in range(b):
            logits = out.logits[i] if out.logits.dim() > 1 else out.logits
            q = out.q_bb[i] if out.q_bb.dim() > 1 else out.q_bb
            unc = out.uncertainty[i] if out.uncertainty.dim() > 0 else out.uncertainty
            val = out.value[i] if out.value.dim() > 0 else out.value
            probs = out.probs[i] if out.probs.dim() > 1 else out.probs

            if stochastic:
                idx = int(
                    torch.distributions.Categorical(logits=logits.unsqueeze(0))
                    .sample()
                    .item()
                )
            else:
                idx = int(logits.argmax().item())

            legal = batch.legal_mask[i].detach().cpu().tolist()
            game = games[i]
            obs_single = ObsV3(
                hero_hand=batch.hero_hand[i],
                global_feats=batch.global_feats[i],
                seat_tokens=batch.seat_tokens[i],
                seat_mask=batch.seat_mask[i],
                history_tokens=batch.history_tokens[i],
                history_mask=batch.history_mask[i],
                legal_mask=batch.legal_mask[i],
            )

            teacher_block = None
            teacher_action = None
            teacher_obj = None
            if include_teacher:
                hand, hero = game_state_to_hand_state(game)
                tr = self.teacher_cache.get_with_path(
                    hand,
                    hero,
                    samples=teacher_samples,
                    seed=teacher_seed + i,
                )
                t = tr.teacher
                teacher_obj = t
                teacher_action = teacher_action_index(t)
                tq = teacher_q_bb(t)
                teacher_block = {
                    "action": action_name(teacher_action),
                    "q_bb": [float(tq[0]), float(tq[1])],
                    "edge_bb": float(t.edge_bb),
                    "provenance": tr.provenance,
                    "approximation_tier": str(tr.path),
                    "action_probs": teacher_probs(t).tolist(),
                }

            ensemble_action = None
            ensemble_logit_diff = None
            if ens_out is not None:
                e_logits = ens_out.logits[i] if ens_out.logits.dim() > 1 else ens_out.logits
                ensemble_action = int(e_logits.argmax().item())
                ensemble_logit_diff = float(
                    (logits - e_logits).abs().mean().item()
                )

            ood = OodResult(0.0, 0.0, False, None, False)
            if self.ood_monitor is not None:
                ood = self.ood_monitor.score(
                    obs_single,
                    policy_action=idx,
                    policy_probs=probs.detach().cpu().numpy(),
                    uncertainty_bb=float(unc.item()),
                    teacher_action=teacher_action,
                    teacher=teacher_obj,
                    ensemble_action=ensemble_action,
                    ensemble_logit_diff=ensemble_logit_diff,
                )

            results.append(
                InferenceResult(
                    chosen_action=action_name(idx),
                    action_logits=[float(logits[0].item()), float(logits[1].item())],
                    q_bb=[float(q[0].item()), float(q[1].item())],
                    uncertainty_bb=float(unc.item()),
                    state_value_bb=float(val.item()),
                    action_probs=[float(probs[0].item()), float(probs[1].item())],
                    legal_mask=[float(legal[0]), float(legal[1])],
                    obs_version=3,
                    model_id=self.model_id,
                    ood=ood,
                    teacher=teacher_block,
                )
            )
        return results

    def predict_one(
        self,
        game: AoFGameState,
        **kwargs: Any,
    ) -> InferenceResult:
        return self.predict_batch([game], **kwargs)[0]
