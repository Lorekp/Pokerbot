"""Unified policy interface for evaluation (v1/v2/v3, oracle, scripted)."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import torch

from aof_rl.encode import encode_obs
from aof_rl.encode_v3 import encode_obs_v3_from_hand
from aof_rl.env import ACTION_ALL_IN, ACTION_FOLD, AoFHandState
from aof_rl.obs_v3 import ObsV3
from aof_rl.teacher import NashTeacherCache
from aof_rl.teacher_targets import teacher_action_index, teacher_q_bb


@dataclass
class PolicyDecision:
    action: int
    teacher_action: int | None = None
    teacher_edge_bb: float | None = None
    q_bb: tuple[float, float] | None = None
    uncertainty_bb: float | None = None
    teacher_q_bb: tuple[float, float] | None = None


class EvalPolicy(Protocol):
    obs_version: int
    name: str

    def act(
        self,
        state: AoFHandState,
        seat: str,
        cache: NashTeacherCache,
        *,
        decision_seed: int,
    ) -> PolicyDecision: ...

    def action_prob_all_in(
        self,
        state: AoFHandState,
        seat: str,
        cache: NashTeacherCache,
        *,
        decision_seed: int,
    ) -> float:
        """P(ALL_IN) for HU exploitability (deterministic eval uses hard 0/1)."""
        ...


@dataclass
class OracleEvalPolicy:
    """Nash teacher argmax at every seat."""

    name: str = "oracle"
    obs_version: int = 3

    def act(
        self,
        state: AoFHandState,
        seat: str,
        cache: NashTeacherCache,
        *,
        decision_seed: int,
    ) -> PolicyDecision:
        tr = cache.get_with_path(state, seat, samples=2000, seed=decision_seed)
        t = tr.teacher
        act = teacher_action_index(t)
        tq = teacher_q_bb(t)
        return PolicyDecision(
            action=act,
            teacher_action=act,
            teacher_edge_bb=float(t.edge_bb),
            q_bb=(float(tq[0]), float(tq[1])),
            teacher_q_bb=(float(tq[0]), float(tq[1])),
        )

    def action_prob_all_in(
        self,
        state: AoFHandState,
        seat: str,
        cache: NashTeacherCache,
        *,
        decision_seed: int,
    ) -> float:
        d = self.act(state, seat, cache, decision_seed=decision_seed)
        return 1.0 if d.action == ACTION_ALL_IN else 0.0


@dataclass
class ScriptedEvalPolicy:
    call_prob: float = 0.55
    seed: int = 0
    name: str = "scripted"
    obs_version: int = 3

    def act(
        self,
        state: AoFHandState,
        seat: str,
        cache: NashTeacherCache,
        *,
        decision_seed: int,
    ) -> PolicyDecision:
        rng = random.Random(self.seed + decision_seed)
        if state.facing_shove() or state.shover is not None:
            action = ACTION_ALL_IN if rng.random() < self.call_prob else ACTION_FOLD
        else:
            p = min(self.call_prob * 0.85, 0.99)
            action = ACTION_ALL_IN if rng.random() < p else ACTION_FOLD
        return PolicyDecision(action=action)

    def action_prob_all_in(
        self,
        state: AoFHandState,
        seat: str,
        cache: NashTeacherCache,
        *,
        decision_seed: int,
    ) -> float:
        d = self.act(state, seat, cache, decision_seed=decision_seed)
        return 1.0 if d.action == ACTION_ALL_IN else 0.0


@dataclass
class FlatPolicyEval:
    model: object
    device: torch.device
    obs_version: int
    name: str
    deterministic: bool = True

    def act(
        self,
        state: AoFHandState,
        seat: str,
        cache: NashTeacherCache,
        *,
        decision_seed: int,
    ) -> PolicyDecision:
        tr = cache.get_with_path(state, seat, samples=2000, seed=decision_seed)
        t = tr.teacher
        obs = encode_obs(
            state,
            seat,
            t,
            version=self.obs_version,
            teacher_path=tr.path,
            provenance=tr.provenance,
        )
        obs_t = torch.from_numpy(obs).unsqueeze(0).to(self.device)
        with torch.no_grad():
            idx, _ = self.model.act(obs_t, deterministic=self.deterministic)
        tq = teacher_q_bb(t)
        ta = teacher_action_index(t)
        return PolicyDecision(
            action=int(idx),
            teacher_action=ta,
            teacher_edge_bb=float(t.edge_bb),
            teacher_q_bb=(float(tq[0]), float(tq[1])),
        )

    def action_prob_all_in(
        self,
        state: AoFHandState,
        seat: str,
        cache: NashTeacherCache,
        *,
        decision_seed: int,
    ) -> float:
        d = self.act(state, seat, cache, decision_seed=decision_seed)
        return 1.0 if d.action == ACTION_ALL_IN else 0.0


@dataclass
class V3PolicyEval:
    model: object
    device: torch.device
    name: str
    deterministic: bool = True
    obs_version: int = 3
    record_teacher: bool = True

    def act(
        self,
        state: AoFHandState,
        seat: str,
        cache: NashTeacherCache,
        *,
        decision_seed: int,
    ) -> PolicyDecision:
        obs = encode_obs_v3_from_hand(state, seat)
        obs_d = ObsV3(
            hero_hand=obs.hero_hand.to(self.device),
            global_feats=obs.global_feats.to(self.device),
            seat_tokens=obs.seat_tokens.to(self.device),
            seat_mask=obs.seat_mask.to(self.device),
            history_tokens=obs.history_tokens.to(self.device),
            history_mask=obs.history_mask.to(self.device),
            legal_mask=obs.legal_mask.to(self.device),
        )
        with torch.no_grad():
            idx, out = self.model.act(obs_d, deterministic=self.deterministic)
        q = out.q_bb
        if q.dim() > 1:
            q = q[0]
        unc = out.uncertainty
        if unc.dim() > 0:
            unc = unc.squeeze()
        decision = PolicyDecision(
            action=int(idx),
            q_bb=(float(q[0].item()), float(q[1].item())),
            uncertainty_bb=float(unc.item()),
        )
        if self.record_teacher:
            tr = cache.get_with_path(state, seat, samples=2000, seed=decision_seed)
            t = tr.teacher
            tq = teacher_q_bb(t)
            decision.teacher_action = teacher_action_index(t)
            decision.teacher_edge_bb = float(t.edge_bb)
            decision.teacher_q_bb = (float(tq[0]), float(tq[1]))
        return decision

    def action_prob_all_in(
        self,
        state: AoFHandState,
        seat: str,
        cache: NashTeacherCache,
        *,
        decision_seed: int,
    ) -> float:
        obs = encode_obs_v3_from_hand(state, seat)
        obs_d = ObsV3(
            hero_hand=obs.hero_hand.to(self.device),
            global_feats=obs.global_feats.to(self.device),
            seat_tokens=obs.seat_tokens.to(self.device),
            seat_mask=obs.seat_mask.to(self.device),
            history_tokens=obs.history_tokens.to(self.device),
            history_mask=obs.history_mask.to(self.device),
            legal_mask=obs.legal_mask.to(self.device),
        )
        with torch.no_grad():
            out = self.model.forward(obs_d)
        probs = out.probs
        if probs.dim() > 1:
            probs = probs[0]
        return float(probs[ACTION_ALL_IN].item())


def load_eval_policy(
    checkpoint: str | Path | None,
    device: torch.device,
    *,
    mode: str = "checkpoint",
    obs_version: int | None = None,
    teacher_version: int = 2,
    deterministic: bool = True,
    scripted_call_prob: float = 0.55,
) -> tuple[EvalPolicy, dict]:
    """Load policy for evaluation. Returns (policy, metadata)."""
    if mode == "oracle":
        return OracleEvalPolicy(), {"mode": "oracle", "obs_version": 3}

    if mode == "scripted":
        return (
            ScriptedEvalPolicy(call_prob=scripted_call_prob, name="scripted"),
            {"mode": "scripted", "obs_version": 3},
        )

    if not checkpoint:
        raise ValueError("checkpoint required for checkpoint mode")

    path = Path(checkpoint)
    ckpt = torch.load(path, map_location=device, weights_only=False)
    schema = int(ckpt.get("schema_version", 0))

    if schema == 2:
        from aof_rl.registry import load_league_checkpoint

        model, meta = load_league_checkpoint(path, device)
        model.eval()
        name = path.stem
        pol = V3PolicyEval(
            model=model,
            device=device,
            name=name,
            deterministic=deterministic,
        )
        return pol, meta

    from aof_rl.checkpoint import load_policy_from_checkpoint

    model, meta = load_policy_from_checkpoint(
        str(path),
        device,
        obs_version=obs_version,
        teacher_version=teacher_version,
    )
    model.eval()
    o_ver = int(obs_version if obs_version is not None else meta.get("obs_version", 1))
    pol = FlatPolicyEval(
        model=model,
        device=device,
        obs_version=o_ver,
        name=path.stem,
        deterministic=deterministic,
    )
    return pol, meta
