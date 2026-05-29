"""Lockstep multi-hand rollout with batched policy forwards."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

import torch

from aof_rl.curriculum import CurriculumScheduler, CurriculumStage
from aof_rl.encode_v3 import encode_obs_v3_from_hand
from aof_rl.env import ACTION_FOLD, AoFHandState, payoff_step_seed, step_hand
from aof_rl.league import LeagueManager
from aof_rl.obs_v3 import ObsV3
from aof_rl.obs_v3_batch import ObsV3Batch, collate_obs_v3
from aof_rl.policy_v3 import AoFPolicyNetV3, PolicyOutputV3
from aof_rl.profiling.league_profiler import get_profiler
from aof_rl.rollout import (
    NeuralPolicyActor,
    PolicyActor,
    RolloutConfig,
    StepTransition,
    _hero_step_from_output,
    _hero_teacher_lookup,
    _obs_for_transition,
    _rollout_autocast,
    policy_key_for_actor,
)
from aof_rl.spot_forcing import reset_hand_for_curriculum
from aof_rl.teacher import NashTeacherCache


def _prof():
    return get_profiler()


@dataclass
class HandSlot:
    state: AoFHandState
    hero_seat: str
    seat_policies: dict[str, PolicyActor]
    traj: list[StepTransition]
    stage: CurriculumStage
    guard: int = 0


@dataclass
class PendingDecision:
    slot_idx: int
    seat: str
    policy_key: str
    is_hero: bool = False


@dataclass
class BatchOccupancyStats:
    total_forwards: int = 0
    total_rows: int = 0
    by_policy_key: dict[str, list[int]] = field(default_factory=dict)

    def record(self, policy_key: str, batch_size: int) -> None:
        self.total_forwards += 1
        self.total_rows += batch_size
        self.by_policy_key.setdefault(policy_key, []).append(batch_size)

    def mean_batch_size(self) -> float:
        if self.total_forwards == 0:
            return 0.0
        return self.total_rows / self.total_forwards


def _is_neural_key(policy_key: str) -> bool:
    return (
        policy_key.startswith("hero:")
        or policy_key.startswith("neural:")
        or policy_key.startswith("archive:")
        or policy_key.startswith("adversarial_br:")
    )


def _assert_legal_rows(legal_mask: torch.Tensor) -> None:
    if not torch.all(legal_mask.sum(dim=-1) >= 1):
        raise RuntimeError("batched forward has fully masked row(s)")


def _obs_to_device(obs: ObsV3, device: torch.device) -> ObsV3:
    return ObsV3(
        hero_hand=obs.hero_hand.to(device),
        global_feats=obs.global_feats.to(device),
        seat_tokens=obs.seat_tokens.to(device),
        seat_mask=obs.seat_mask.to(device),
        history_tokens=obs.history_tokens.to(device),
        history_mask=obs.history_mask.to(device),
        legal_mask=obs.legal_mask.to(device),
    )


def _pad_obs_batch(
    batch: ObsV3Batch, target: int | None
) -> tuple[ObsV3Batch, int]:
    b = batch.hero_hand.shape[0]
    if target is None or b >= target:
        _assert_legal_rows(batch.legal_mask)
        return batch, b

    def _row_cpu(i: int) -> ObsV3:
        return ObsV3(
            hero_hand=batch.hero_hand[i].detach().cpu(),
            global_feats=batch.global_feats[i].detach().cpu(),
            seat_tokens=batch.seat_tokens[i].detach().cpu(),
            seat_mask=batch.seat_mask[i].detach().cpu(),
            history_tokens=batch.history_tokens[i].detach().cpu(),
            history_mask=batch.history_mask[i].detach().cpu(),
            legal_mask=batch.legal_mask[i].detach().cpu(),
        )

    pad_row = _row_cpu(b - 1)
    rows = [_row_cpu(i) for i in range(b)] + [pad_row] * (target - b)
    padded = collate_obs_v3(rows)
    dev = batch.hero_hand.device
    padded = ObsV3Batch(
        hero_hand=padded.hero_hand.to(dev),
        global_feats=padded.global_feats.to(dev),
        seat_tokens=padded.seat_tokens.to(dev),
        seat_mask=padded.seat_mask.to(dev),
        history_tokens=padded.history_tokens.to(dev),
        history_mask=padded.history_mask.to(dev),
        legal_mask=padded.legal_mask.to(dev),
    )
    _assert_legal_rows(padded.legal_mask)
    return padded, b


def _slice_policy_output(out: PolicyOutputV3, row: int) -> PolicyOutputV3:
    return PolicyOutputV3(
        logits=out.logits[row],
        q_bb=out.q_bb[row],
        value=out.value[row],
        uncertainty=out.uncertainty[row],
        probs=out.probs[row],
        log_probs=out.log_probs[row],
    )


class BatchedHandCollector:
    def __init__(
        self,
        hero_model: AoFPolicyNetV3,
        device: torch.device,
        cache: NashTeacherCache,
        cfg: RolloutConfig,
        league: LeagueManager,
        scheduler: CurriculumScheduler,
        rng: random.Random,
        *,
        n_parallel: int = 64,
        pad_batch_size: int | None = None,
        torch_compile_villain: bool = False,
        torch_compile_hero: bool = False,
    ) -> None:
        self.hero_model = hero_model
        self.device = device
        self.cache = cache
        self.cfg = cfg
        self.league = league
        self.scheduler = scheduler
        self.rng = rng
        self.n_parallel = max(1, n_parallel)
        self.pad_batch_size = pad_batch_size
        self.occupancy = BatchOccupancyStats()
        self._compiled: dict[str, Any] = {}
        self._torch_compile_villain = torch_compile_villain
        self._torch_compile_hero = torch_compile_hero

    def _pad_target(self, policy_key: str) -> int | None:
        if policy_key == "hero:main":
            compile_on = self._torch_compile_hero
        else:
            compile_on = self._torch_compile_villain
        if not compile_on:
            return None
        return self.pad_batch_size or self.n_parallel

    def _model_for_key(
        self, policy_key: str, model: AoFPolicyNetV3, *, compile: bool
    ) -> AoFPolicyNetV3:
        if not compile:
            return model
        if policy_key not in self._compiled:
            try:
                self._compiled[policy_key] = torch.compile(model, mode="reduce-overhead")
            except Exception:
                self._compiled[policy_key] = model
        return self._compiled[policy_key]  # type: ignore[return-value]

    def _resolve_model(
        self, policy_key: str, pending: list[PendingDecision], slots: list[HandSlot | None]
    ) -> AoFPolicyNetV3:
        if policy_key == "hero:main":
            return self._model_for_key(
                policy_key, self.hero_model, compile=self._torch_compile_hero
            )
        for p in pending:
            slot = slots[p.slot_idx]
            if slot is None:
                continue
            actor = slot.seat_policies.get(p.seat)
            if isinstance(actor, NeuralPolicyActor):
                return self._model_for_key(
                    policy_key, actor.model, compile=self._torch_compile_villain
                )
        raise KeyError(f"no neural model for policy_key={policy_key}")

    def _new_slot(self, update: int, stage: CurriculumStage) -> HandSlot:
        prof = _prof()
        if prof is not None:
            with prof.measure("rollout.reset_curriculum"):
                state, hero_seat = reset_hand_for_curriculum(
                    self.rng, stage, stack_bb=self.cfg.stack_bb
                )
        else:
            state, hero_seat = reset_hand_for_curriculum(
                self.rng, stage, stack_bb=self.cfg.stack_bb
            )
        if prof is not None:
            prof.inc_counter(f"table_size.{state.table_size}")
        policies = self.league.sample_seat_policies(
            state.table_size, hero_seat, self.hero_model, self.rng
        )
        self.cfg.flat_rake_bb = self.scheduler.sample_rake_bb(stage, self.rng)
        return HandSlot(
            state=state,
            hero_seat=hero_seat,
            seat_policies=policies,
            traj=[],
            stage=stage,
        )

    def _batched_neural_act(
        self,
        policy_key: str,
        pending: list[PendingDecision],
        slots: list[HandSlot | None],
    ) -> tuple[list[int], list[ObsV3], PolicyOutputV3 | None, torch.Tensor | None, torch.Tensor | None]:
        prof = _prof()
        obs_list: list[ObsV3] = []
        encode_bucket = (
            "rollout.hero_encode"
            if pending[0].is_hero
            else "rollout.villain_encode_batch"
        )
        for p in pending:
            slot = slots[p.slot_idx]
            assert slot is not None
            if prof is not None:
                with prof.measure(encode_bucket):
                    obs_list.append(
                        encode_obs_v3_from_hand(
                            slot.state, p.seat, device=self.device
                        )
                    )
            else:
                obs_list.append(
                    encode_obs_v3_from_hand(slot.state, p.seat, device=self.device)
                )

        batch = collate_obs_v3(obs_list)
        real_b = batch.hero_hand.shape[0]
        self.occupancy.record(policy_key, real_b)
        if prof is not None:
            prof.inc_counter("rollout.batched_forward_rows", real_b)

        padded, real_b = _pad_obs_batch(batch, self._pad_target(policy_key))
        model = self._resolve_model(policy_key, pending, slots)
        fwd_bucket = (
            "rollout.hero_forward_batch"
            if pending[0].is_hero
            else "rollout.villain_forward_batch"
        )
        with _rollout_autocast(self.device, self.cfg.mixed_precision):
            with torch.no_grad():
                if prof is not None:
                    with prof.measure(fwd_bucket):
                        idx_t, lp_t, val_t, out = model.act_batch_with_logprob(
                            padded, deterministic=self.cfg.deterministic_policy
                        )
                else:
                    idx_t, lp_t, val_t, out = model.act_batch_with_logprob(
                        padded, deterministic=self.cfg.deterministic_policy
                    )
        actions = [int(idx_t[i].item()) for i in range(real_b)]
        stored = [
            _obs_for_transition(o, self.device, self.cfg.store_obs_on_cpu)
            for o in obs_list
        ]
        return actions, stored, out, lp_t[:real_b], val_t[:real_b]

    def _apply_hero_batch(
        self,
        pending: list[PendingDecision],
        slots: list[HandSlot | None],
        actions: list[int],
        obs_cpu: list[ObsV3],
        out: PolicyOutputV3 | None,
        log_probs: torch.Tensor | None = None,
        values: torch.Tensor | None = None,
    ) -> dict[tuple[int, str], int]:
        result: dict[tuple[int, str], int] = {}
        assert out is not None
        for i, p in enumerate(pending):
            slot = slots[p.slot_idx]
            assert slot is not None
            t_result = _hero_teacher_lookup(slot.state, p.seat, self.cache, self.cfg)
            row_out = _slice_policy_output(out, i)
            lp = float(log_probs[i].item()) if log_probs is not None else None
            val = float(values[i].item()) if values is not None else None
            step = _hero_step_from_output(
                obs_cpu[i], actions[i], row_out, t_result.teacher, log_prob=lp, value=val
            )
            slot.traj.append(step)
            result[(p.slot_idx, p.seat)] = actions[i]
        return result

    def _apply_villain_batch(
        self,
        pending: list[PendingDecision],
        actions: list[int],
    ) -> dict[tuple[int, str], int]:
        return {(p.slot_idx, p.seat): actions[i] for i, p in enumerate(pending)}


def collect_trajectory_v3_batched(
    hero_model: AoFPolicyNetV3,
    seat_policies: dict[str, PolicyActor],
    device: torch.device,
    hero_seat: str,
    cache: NashTeacherCache,
    cfg: RolloutConfig,
    state: AoFHandState,
    *,
    pad_batch_size: int = 1,
) -> list[StepTransition]:
    """Single-hand batched path for parity with collect_trajectory_v3."""
    collector = BatchedHandCollector(
        hero_model,
        device,
        cache,
        cfg,
        league=None,  # type: ignore[arg-type]
        scheduler=None,  # type: ignore[arg-type]
        rng=random.Random(0),
        n_parallel=1,
        pad_batch_size=pad_batch_size,
    )
    slot = HandSlot(
        state=state,
        hero_seat=hero_seat,
        seat_policies=dict(seat_policies),
        traj=[],
        stage=CurriculumStage(name="t", table_sizes=(state.table_size,), updates=1),
    )
    slots: list[HandSlot | None] = [slot]
    while not slot.state.terminal:
        pending: list[PendingDecision] = []
        seat = slot.state.action_seat
        if seat is None:
            break
        if seat == hero_seat:
            pending.append(PendingDecision(0, seat, "hero:main", is_hero=True))
        else:
            actor = slot.seat_policies.get(seat)
            if actor is not None:
                pending.append(
                    PendingDecision(0, seat, policy_key_for_actor(actor))
                )
        if not pending:
            break
        action_map: dict[tuple[int, str], int] = {}
        by_key: dict[str, list[PendingDecision]] = {}
        for p in pending:
            by_key.setdefault(p.policy_key, []).append(p)
        for policy_key, group in by_key.items():
            if _is_neural_key(policy_key):
                hero_g = [p for p in group if p.is_hero]
                villain_g = [p for p in group if not p.is_hero]
                if hero_g:
                    acts, obs_cpu, out, lp_t, val_t = collector._batched_neural_act(
                        policy_key, hero_g, slots
                    )
                    action_map.update(
                        collector._apply_hero_batch(
                            hero_g, slots, acts, obs_cpu, out, lp_t, val_t
                        )
                    )
                if villain_g:
                    acts, _, _, _, _ = collector._batched_neural_act(
                        policy_key, villain_g, slots
                    )
                    action_map.update(collector._apply_villain_batch(villain_g, acts))
            else:
                for p in group:
                    actor = slot.seat_policies[p.seat]
                    action_map[(0, p.seat)] = actor.act(slot.state, p.seat, cache)
        seat = slot.state.action_seat
        if seat is None:
            break
        action = action_map.get((0, seat), ACTION_FOLD)
        slot.guard += 1
        slot.state = step_hand(
            slot.state,
            action,
            payoff_samples=cfg.payoff_samples,
            payoff_seed=payoff_step_seed(slot.state.hand_id, slot.guard),
        )
    reward = slot.state.payoffs.get(hero_seat, 0.0)
    if slot.traj:
        slot.traj[-1].reward = reward
        slot.traj[-1].done = True
    return slot.traj


def collect_update_rollout_batched(
    hero_model: AoFPolicyNetV3,
    device: torch.device,
    cache: NashTeacherCache,
    cfg: RolloutConfig,
    league: LeagueManager,
    scheduler: CurriculumScheduler,
    rng: random.Random,
    update: int,
    *,
    n_parallel: int = 64,
    total_hands: int = 128,
    pad_batch_size: int | None = None,
    torch_compile_villain: bool = False,
    torch_compile_hero: bool = False,
) -> list[StepTransition]:
    stage = scheduler.stage_for_update(update)
    collector = BatchedHandCollector(
        hero_model,
        device,
        cache,
        cfg,
        league,
        scheduler,
        rng,
        n_parallel=n_parallel,
        pad_batch_size=pad_batch_size,
        torch_compile_villain=torch_compile_villain,
        torch_compile_hero=torch_compile_hero,
    )
    all_traj: list[StepTransition] = []
    hands_started = 0
    hands_completed = 0
    slots: list[HandSlot | None] = [None] * collector.n_parallel

    def start_hand_in_slot(i: int) -> None:
        nonlocal hands_started
        slots[i] = collector._new_slot(update, stage)
        hands_started += 1

    for i in range(min(collector.n_parallel, total_hands)):
        start_hand_in_slot(i)

    loop_prof = _prof()
    batched_ctx = (
        loop_prof.measure("rollout.batched_step_loop") if loop_prof is not None else None
    )
    if batched_ctx is not None:
        batched_ctx.__enter__()
    try:
        while hands_completed < total_hands:
            pending: list[PendingDecision] = []
            for idx, slot in enumerate(slots):
                if slot is None or slot.state.terminal:
                    continue
                seat = slot.state.action_seat
                if seat is None:
                    continue
                if seat == slot.hero_seat:
                    pending.append(
                        PendingDecision(idx, seat, "hero:main", is_hero=True)
                    )
                else:
                    actor = slot.seat_policies.get(seat)
                    if actor is None:
                        continue
                    pending.append(
                        PendingDecision(
                            idx,
                            seat,
                            policy_key_for_actor(actor),
                            is_hero=False,
                        )
                    )

            if not pending:
                break

            action_map: dict[tuple[int, str], int] = {}
            by_key: dict[str, list[PendingDecision]] = {}
            for p in pending:
                by_key.setdefault(p.policy_key, []).append(p)

            for policy_key, group in by_key.items():
                if _is_neural_key(policy_key):
                    hero_g = [p for p in group if p.is_hero]
                    villain_g = [p for p in group if not p.is_hero]
                    if hero_g:
                        acts, obs_cpu, out, lp_t, val_t = collector._batched_neural_act(
                            policy_key, hero_g, slots
                        )
                        action_map.update(
                            collector._apply_hero_batch(
                                hero_g, slots, acts, obs_cpu, out, lp_t, val_t
                            )
                        )
                    if villain_g:
                        acts, _, _, _, _ = collector._batched_neural_act(
                            policy_key, villain_g, slots
                        )
                        action_map.update(collector._apply_villain_batch(villain_g, acts))
                else:
                    prof = _prof()
                    for p in group:
                        slot = slots[p.slot_idx]
                        assert slot is not None
                        actor = slot.seat_policies[p.seat]
                        if prof is not None:
                            with prof.measure("rollout.villain_act"):
                                act = actor.act(slot.state, p.seat, collector.cache)
                        else:
                            act = actor.act(slot.state, p.seat, collector.cache)
                        action_map[(p.slot_idx, p.seat)] = act

            for idx, slot in enumerate(slots):
                if slot is None or slot.state.terminal:
                    continue
                seat = slot.state.action_seat
                if seat is None:
                    continue
                action = action_map.get((idx, seat), ACTION_FOLD)

                slot.guard += 1
                if slot.guard > 32:
                    raise RuntimeError("Hand guard exceeded")
                seed = payoff_step_seed(slot.state.hand_id, slot.guard)
                slot.state = step_hand(
                    slot.state,
                    action,
                    payoff_samples=cfg.payoff_samples,
                    payoff_seed=seed,
                )

                if slot.state.terminal:
                    reward = slot.state.payoffs.get(slot.hero_seat, 0.0)
                    if slot.traj:
                        slot.traj[-1].reward = reward
                        slot.traj[-1].done = True
                    all_traj.extend(slot.traj)
                    hands_completed += 1
                    slots[idx] = None
                    if hands_started < total_hands:
                        start_hand_in_slot(idx)
    finally:
        if batched_ctx is not None:
            batched_ctx.__exit__(None, None, None)

    prof = _prof()
    if prof is not None:
        mean_occ = collector.occupancy.mean_batch_size()
        prof.inc_counter("rollout.batched_mean_batch_size", int(mean_occ * 1000))

    return all_traj
