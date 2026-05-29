"""Solver-guided PPO league training loop (obs v3)."""

from __future__ import annotations

import copy
import random
import time
from pathlib import Path

import torch

from aof_equity import warmup
from aof_ranges import active_seats
from aof_rl.config import TrainExperimentConfig
from aof_rl.curriculum import CurriculumScheduler
from aof_rl.league import LeagueManager
from aof_rl.obs_v3_batch import ObsV3Batch, collate_obs_v3
from aof_rl.policy_v3 import AoFPolicyNetV3
from aof_rl.ppo_v3 import compute_gae, ppo_v3_loss
from aof_rl.registry import resolve_best_checkpoint_path, save_league_checkpoint
from aof_rl.batched_rollout import collect_update_rollout_batched
from aof_rl.payoff import configure_payoff
from aof_rl.rollout import RolloutConfig, collect_hand_v3
from aof_rl.profiling.league_profiler import (
    cprofile_update_index,
    get_profiler,
    set_profiler,
    start_cprofile,
    stop_cprofile,
    store_last_timings,
)
from aof_rl.profiling.timer import TimingAccumulator
from aof_rl.teacher import NashTeacherCache


def _resolve_device(name: str, *, allow_cpu: bool) -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name.startswith("cuda") and not torch.cuda.is_available():
        if allow_cpu:
            return torch.device("cpu")
        raise RuntimeError(
            "CUDA requested but unavailable. Use allow_cpu: true in config."
        )
    return torch.device(name if name != "cuda" or torch.cuda.is_available() else "cpu")


def _train_autocast(device: torch.device, enabled: bool):
    from contextlib import nullcontext

    if enabled and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def _set_deterministic(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass


def _triton_available() -> bool:
    try:
        import triton  # noqa: F401

        return True
    except ImportError:
        return False


def _tensor_to_device(t: torch.Tensor, device: torch.device) -> torch.Tensor:
    if t.device == device:
        return t
    if device.type == "cuda" and t.device.type == "cpu":
        t = t.pin_memory() if not t.is_pinned() else t
        return t.to(device, non_blocking=True)
    return t.to(device)


def _batch_to_device(batch_obs: ObsV3Batch, device: torch.device) -> ObsV3Batch:
    if batch_obs.hero_hand.device == device:
        return batch_obs
    return ObsV3Batch(
        hero_hand=_tensor_to_device(batch_obs.hero_hand, device),
        global_feats=_tensor_to_device(batch_obs.global_feats, device),
        seat_tokens=_tensor_to_device(batch_obs.seat_tokens, device),
        seat_mask=_tensor_to_device(batch_obs.seat_mask, device),
        history_tokens=_tensor_to_device(batch_obs.history_tokens, device),
        history_mask=_tensor_to_device(batch_obs.history_mask, device),
        legal_mask=_tensor_to_device(batch_obs.legal_mask, device),
    )


def _transitions_to_batch(traj: list, device: torch.device):
    batch_obs = collate_obs_v3([t.obs for t in traj])
    batch_obs = _batch_to_device(batch_obs, device)
    batch = {
        "action": torch.tensor([t.action for t in traj], device=device, dtype=torch.long),
        "teacher_action": torch.tensor(
            [t.teacher_action for t in traj], device=device, dtype=torch.long
        ),
        "teacher_q": torch.tensor(
            [t.teacher_q for t in traj], device=device, dtype=torch.float32
        ),
        "teacher_edge": torch.tensor(
            [t.teacher_edge for t in traj], device=device, dtype=torch.float32
        ),
        "old_value": torch.tensor(
            [t.value for t in traj], device=device, dtype=torch.float32
        ),
    }
    old_lp = torch.tensor(
        [t.log_prob for t in traj], device=device, dtype=torch.float32
    )
    rewards = [t.reward for t in traj]
    values = [t.value for t in traj]
    dones = [t.done for t in traj]
    return batch_obs, batch, old_lp, rewards, values, dones


def _training_meta(
    cfg: TrainExperimentConfig,
    tcfg,
    *,
    deterministic: bool,
    wall_seconds: float,
    update: int,
    best_bb_hand: float | None = None,
    checkpoint_kind: str = "final",
) -> dict:
    meta = {
        "algo": tcfg.algo,
        "experiment": cfg.experiment,
        "seed": tcfg.seed,
        "total_updates": tcfg.total_updates,
        "batch_hands": tcfg.batch_hands,
        "lr": tcfg.lr,
        "teacher_bc_coef": tcfg.teacher_bc_coef,
        "q_coef": tcfg.q_coef,
        "deterministic_validation": deterministic,
        "wall_seconds": wall_seconds,
        "completed_update": update,
        "checkpoint_kind": checkpoint_kind,
    }
    if best_bb_hand is not None:
        meta["best_bb_hand"] = best_bb_hand
    return meta


def _maybe_save_best_checkpoint(
    *,
    cfg: TrainExperimentConfig,
    model: AoFPolicyNetV3,
    encoder_cfg,
    deterministic: bool,
    update: int,
    bb_hand: float,
    best_bb_hand: float | None,
    wall_seconds: float,
) -> float | None:
    tcfg = cfg.train
    if not tcfg.save_checkpoint_on_best:
        return best_bb_hand
    threshold = (best_bb_hand or float("-inf")) + tcfg.min_bb_hand_improvement
    if best_bb_hand is not None and bb_hand <= threshold:
        return best_bb_hand
    best_path = resolve_best_checkpoint_path(
        cfg.checkpoint_path, cfg.checkpoint_best_path
    )
    save_league_checkpoint(
        best_path,
        model,
        update=update,
        training=_training_meta(
            cfg,
            tcfg,
            deterministic=deterministic,
            wall_seconds=wall_seconds,
            update=update,
            best_bb_hand=bb_hand,
            checkpoint_kind="best",
        ),
        encoder_cfg=encoder_cfg,
        teacher_version=cfg.teacher_version,
        evaluation={"bb_per_100": None, "paired_eval_hands": 0, "train_bb_hand": bb_hand},
        atomic=True,
    )
    print(
        f"New best bb/hand={bb_hand:.3f} at update {update} -> {best_path}",
        flush=True,
    )
    return bb_hand


def train_ppo_league(cfg: TrainExperimentConfig) -> AoFPolicyNetV3:
    tcfg = cfg.train
    prof: TimingAccumulator | None = None
    if tcfg.profile:
        prof = TimingAccumulator()
        set_profiler(prof)
        prof.start_wall()

    try:
        return _train_ppo_league_impl(cfg, prof)
    finally:
        if tcfg.profile:
            if prof is not None:
                prof.stop_wall()
                store_last_timings(prof)
            set_profiler(None)


def _train_ppo_league_impl(
    cfg: TrainExperimentConfig, prof: TimingAccumulator | None
) -> AoFPolicyNetV3:
    tcfg = cfg.train
    if prof is not None:
        with prof.measure("startup.warmup"):
            warmup()
    else:
        warmup()
    from aof_equity_cache import get_equity_cache

    get_equity_cache().clear()

    deterministic = tcfg.deterministic_validation or cfg.validation.deterministic
    _set_deterministic(tcfg.seed, deterministic)

    device = _resolve_device(tcfg.device, allow_cpu=tcfg.allow_cpu)
    encoder_cfg = cfg.model.to_encoder_config()
    model = AoFPolicyNetV3(encoder_cfg).to(device)

    if tcfg.torch_compile and device.type == "cuda":
        if _triton_available():
            try:
                model = torch.compile(model, mode="reduce-overhead")  # type: ignore[assignment]
            except Exception as exc:
                print(f"torch.compile disabled: {exc}")
        else:
            print("torch.compile disabled: triton not available")

    compile_villain = tcfg.torch_compile_villain and device.type == "cuda" and _triton_available()
    compile_hero = tcfg.torch_compile_hero and device.type == "cuda" and _triton_available()
    if tcfg.torch_compile_villain and not compile_villain:
        print("torch_compile_villain disabled: requires CUDA + triton")
    if tcfg.torch_compile_hero and not compile_hero:
        print("torch_compile_hero disabled: requires CUDA + triton")

    configure_payoff(
        payoff_hu_mode=tcfg.payoff_hu_mode,
        hu_showdown_table_path=tcfg.hu_showdown_table_path,
    )
    rollout_cfg = RolloutConfig(
        stack_bb=tcfg.stack_bb,
        teacher_samples=tcfg.teacher_samples,
        payoff_samples=tcfg.payoff_samples,
        seed=tcfg.seed,
        teacher_version=cfg.teacher_version,
        use_fast_teacher=tcfg.use_fast_teacher and not deterministic,
        teacher_margin_bb=tcfg.teacher_margin_bb,
        teacher_min_samples=tcfg.teacher_min_samples,
        deterministic_policy=deterministic,
        store_obs_on_cpu=tcfg.store_obs_on_cpu,
        mixed_precision=tcfg.mixed_precision,
        payoff_hu_mode=tcfg.payoff_hu_mode,
        hu_showdown_table_path=tcfg.hu_showdown_table_path,
    )
    league = LeagueManager(cfg.league, encoder_cfg, device, rollout_cfg)
    scheduler = CurriculumScheduler(cfg.curriculum)
    ppo_cfg = tcfg.to_ppo_config()
    lut_path = tcfg.teacher_lut_path if tcfg.use_teacher_lut else None
    cache = NashTeacherCache(
        use_fast=rollout_cfg.use_fast_teacher,
        teacher_margin_bb=tcfg.teacher_margin_bb,
        teacher_min_samples=tcfg.teacher_min_samples,
        teacher_version=cfg.teacher_version,
        teacher_lut_path=lut_path,
        teacher_samples=tcfg.teacher_samples,
        stack_bb=tcfg.stack_bb,
    )
    opt = torch.optim.AdamW(model.parameters(), lr=tcfg.lr)
    rng = random.Random(tcfg.seed)
    t_start = time.monotonic()
    best_bb_hand: float | None = None
    best_path = resolve_best_checkpoint_path(
        cfg.checkpoint_path, cfg.checkpoint_best_path
    )
    if tcfg.save_checkpoint_on_best:
        print(f"Best checkpoints: {best_path}", flush=True)
    for update in range(tcfg.total_updates):
        stage = scheduler.stage_for_update(update)
        if not deterministic:
            cache.clear()

        if prof is not None:
            prof.begin_update(update, stage.name)

        do_cprof = prof is not None and cprofile_update_index() == update
        if do_cprof:
            start_cprofile()
        try:
            all_traj = []
            rollout_ctx = prof.measure("update.rollout") if prof is not None else None
            if rollout_ctx is not None:
                rollout_ctx.__enter__()
            try:
                if tcfg.use_batched_rollout:
                    all_traj.extend(
                        collect_update_rollout_batched(
                            model,
                            device,
                            cache,
                            rollout_cfg,
                            league,
                            scheduler,
                            rng,
                            update,
                            n_parallel=tcfg.parallel_hands,
                            total_hands=tcfg.batch_hands,
                            pad_batch_size=(
                                tcfg.parallel_hands
                                if (compile_villain or compile_hero)
                                else None
                            ),
                            torch_compile_villain=compile_villain,
                            torch_compile_hero=compile_hero,
                        )
                    )
                else:
                    hand_ctx = prof.measure("rollout.hand_loop") if prof is not None else None
                    if hand_ctx is not None:
                        hand_ctx.__enter__()
                    try:
                        for _ in range(tcfg.batch_hands):
                            table_size = scheduler.sample_table_size(stage, rng)
                            hero_seat = rng.choice(active_seats(table_size))
                            policies = league.sample_seat_policies(
                                table_size, hero_seat, model, rng
                            )
                            if prof is not None:
                                for _seat, pol in policies.items():
                                    kind = type(pol).__name__
                                    prof.inc_counter(f"opponent.{kind}")
                            rollout_cfg.flat_rake_bb = scheduler.sample_rake_bb(stage, rng)
                            all_traj.extend(
                                collect_hand_v3(
                                    model, policies, device, cache, rollout_cfg, rng, stage
                                )
                            )
                    finally:
                        if hand_ctx is not None:
                            hand_ctx.__exit__(None, None, None)
            finally:
                if rollout_ctx is not None:
                    rollout_ctx.__exit__(None, None, None)

            if not all_traj:
                continue

            if prof is not None:
                prof.inc_counter("hands", tcfg.batch_hands)
                prof.inc_counter("transitions", len(all_traj))

            if prof is not None:
                with prof.measure("update.transitions_to_batch"):
                    batch_obs, batch, old_lp, rewards, values, dones = _transitions_to_batch(
                        all_traj, device
                    )
            else:
                batch_obs, batch, old_lp, rewards, values, dones = _transitions_to_batch(
                    all_traj, device
                )
            if prof is not None:
                with prof.measure("update.ppo.gae"):
                    adv, rets = compute_gae(
                        rewards,
                        values,
                        dones,
                        gamma=tcfg.gamma,
                        lam=tcfg.gae_lambda,
                    )
            else:
                adv, rets = compute_gae(
                    rewards,
                    values,
                    dones,
                    gamma=tcfg.gamma,
                    lam=tcfg.gae_lambda,
                )
            advantages = torch.tensor(adv, device=device, dtype=torch.float32)
            returns = torch.tensor(rets, device=device, dtype=torch.float32)

            model.train()
            stats_last: dict[str, float] = {}
            ppo_ctx = prof.measure("update.ppo.train") if prof is not None else None
            if ppo_ctx is not None:
                ppo_ctx.__enter__()
            try:
                for _ in range(ppo_cfg.ppo_epochs):
                    if prof is not None:
                        with prof.measure("update.ppo.forward"):
                            with _train_autocast(device, tcfg.mixed_precision):
                                loss, stats_last = ppo_v3_loss(
                                    model,
                                    batch_obs,
                                    batch,
                                    old_lp,
                                    advantages,
                                    returns,
                                    ppo_cfg,
                                )
                        opt.zero_grad()
                        with prof.measure("update.ppo.backward"):
                            loss.backward()
                        with prof.measure("update.ppo.optim_step"):
                            torch.nn.utils.clip_grad_norm_(
                                model.parameters(), ppo_cfg.max_grad_norm
                            )
                            opt.step()
                    else:
                        with _train_autocast(device, tcfg.mixed_precision):
                            loss, stats_last = ppo_v3_loss(
                                model, batch_obs, batch, old_lp, advantages, returns, ppo_cfg
                            )
                        opt.zero_grad()
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), ppo_cfg.max_grad_norm
                        )
                        opt.step()
            finally:
                if ppo_ctx is not None:
                    ppo_ctx.__exit__(None, None, None)

            bb_hand = sum(rewards) / max(len(rewards), 1)
            if update % 10 == 0:
                print(
                    f"Update {update} stage={stage.name} bb/hand={bb_hand:.3f} {stats_last}",
                    flush=True,
                )

            best_bb_hand = _maybe_save_best_checkpoint(
                cfg=cfg,
                model=model,
                encoder_cfg=encoder_cfg,
                deterministic=deterministic,
                update=update,
                bb_hand=bb_hand,
                best_bb_hand=best_bb_hand,
                wall_seconds=time.monotonic() - t_start,
            )

            if prof is not None:
                with prof.measure("update.league_snapshot"):
                    league.maybe_snapshot(update, model, bb_hand)
            else:
                league.maybe_snapshot(update, model, bb_hand)

            if (
                cfg.league.adversarial_br_pool
                and update > 0
                and update % cfg.league.br_every_updates == 0
            ):
                from aof_rl.exploit.multiway_br import run_adversarial_br_cycle

                br_ctx = prof.measure("update.adversarial_br") if prof is not None else None
                if br_ctx is not None:
                    br_ctx.__enter__()
                try:
                    registered, gain = run_adversarial_br_cycle(
                        model,
                        encoder_cfg=encoder_cfg,
                        device=device,
                        league_sample_fn=league.sample_seat_policies,
                        cache=cache,
                        rollout_cfg=rollout_cfg,
                        rng=rng,
                        stage=stage,
                        train_steps=cfg.league.br_train_steps,
                        ppo_cfg=ppo_cfg,
                        gain_threshold_bb=cfg.league.br_gain_threshold_bb,
                    )
                    if registered is not None:
                        league.register_adversarial(registered, name=f"br_u{update}")
                finally:
                    if br_ctx is not None:
                        br_ctx.__exit__(None, None, None)

            if prof is not None:
                prof.inc_counter("updates", 1)
        finally:
            if do_cprof:
                stop_cprofile()
            if prof is not None:
                prof.end_update()

    evaluation_meta: dict = {
        "bb_per_100": None,
        "paired_eval_hands": cfg.validation.paired_eval_hands,
    }
    if cfg.validation.paired_eval_hands > 0:
        from aof_rl.eval.harness import AofEvalHarness, EvalConfig

        eval_hands = min(cfg.validation.paired_eval_hands, 500)
        eval_report = AofEvalHarness(
            EvalConfig(
                checkpoint=str(cfg.checkpoint_path),
                eval_id=cfg.experiment,
                hands=eval_hands,
                seeds=cfg.validation.seeds[:3] or [1, 2, 3],
                table_sizes=(2, 3, 4),
                stack_bb=tcfg.stack_bb,
                profile="fast",
                device=str(device),
                deterministic=deterministic,
                teacher_version=cfg.teacher_version,
                compute_hu_exploitability=True,
                hu_br_mode="bucketed_169",
                hu_equity_samples=400,
                compute_multiway=eval_hands >= 50,
            )
        ).run()
        evaluation_meta["bb_per_100"] = eval_report.aggregate.bb_per_100
        if eval_report.hu_exploitability is not None:
            evaluation_meta["hu_exploitability_bb_per_hand"] = (
                eval_report.hu_exploitability.exploitability_bb_per_hand
            )
        if eval_report.aggregate.disagreement_rate_vs_teacher is not None:
            evaluation_meta["disagreement_rate_vs_teacher"] = (
                eval_report.aggregate.disagreement_rate_vs_teacher
            )

    if cfg.validation.br_exploit_hands > 0:
        from aof_rl.exploit.harness import BrExploitConfig, BrExploitHarness

        br_hands = min(cfg.validation.br_exploit_hands, 200)
        br_report = BrExploitHarness(
            BrExploitConfig(
                checkpoint=str(cfg.checkpoint_path),
                eval_id=f"{cfg.experiment}_br",
                hands=br_hands,
                seeds=cfg.validation.seeds[:2] or [1, 2],
                table_sizes=(2, 3, 4),
                stack_bb=tcfg.stack_bb,
                profile="fast",
                device=str(device),
                deterministic=deterministic,
                teacher_version=cfg.teacher_version,
                compute_hu=True,
                hu_mode="bucketed_169",
                hu_equity_samples=300,
                multiway_br=("oracle",),
            )
        ).run()
        br_meta: dict = {}
        if br_report.hu_exploitability:
            br_meta["hu_exploitability_bb_per_hand"] = (
                br_report.hu_exploitability.exploitability_bb_per_hand
            )
        max_mw = 0.0
        for rows in br_report.multiway_deviation_oracle.values():
            for s in rows:
                max_mw = max(max_mw, s.deviation_gain_bb_per_hand)
        if max_mw > 0:
            br_meta["max_multiway_deviation_bb"] = max_mw
        evaluation_meta["br_exploit"] = br_meta

    out = Path(cfg.checkpoint_path)
    final_meta = _training_meta(
        cfg,
        tcfg,
        deterministic=deterministic,
        wall_seconds=time.monotonic() - t_start,
        update=tcfg.total_updates - 1,
        best_bb_hand=best_bb_hand,
        checkpoint_kind="final",
    )
    save_league_checkpoint(
        out,
        model,
        update=tcfg.total_updates,
        training=final_meta,
        encoder_cfg=encoder_cfg,
        teacher_version=cfg.teacher_version,
        evaluation=evaluation_meta,
        atomic=True,
    )
    if best_bb_hand is not None:
        print(
            f"Training done. Final -> {out} | best bb/hand={best_bb_hand:.3f} -> {best_path}",
            flush=True,
        )
    else:
        print(f"Saved checkpoint: {out}", flush=True)
    return model
