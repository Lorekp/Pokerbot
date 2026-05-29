"""Symmetric self-play training for AoF RL."""

from __future__ import annotations

import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from aof_equity import warmup
from aof_ranges import active_seats
from aof_rl.encode import encode_obs, obs_dim
from aof_rl.env import (
    ACTION_FOLD,
    AoFHandState,
    reset_hand,
    step_hand,
)
from aof_rl.policy import AoFPolicyNet
from aof_rl.ppo import PPOConfig, compute_gae, ppo_loss
from aof_rl.checkpoint import enrich_checkpoint_meta
from aof_rl.teacher import NashTeacherCache


@dataclass
class LeagueAgent:
    model: AoFPolicyNet
    elo: float = 1000.0
    name: str = "agent"


@dataclass
class AoFSelfPlayConfig:
    hands_per_update: int = 64
    total_updates: int = 50
    lr: float = 3e-4
    device: str = "cpu"
    checkpoint_dir: Path = Path("checkpoints/aof_rl")
    seed: int = 42
    table_sizes: tuple[int, ...] = (2, 3, 4)
    stack_bb: float = 8.0
    teacher_samples: int = 2000
    payoff_samples: int = 4000
    league_size: int = 0
    league_snapshot_every: int = 10
    lambda_teacher: float = 0.3
    ppo_epochs: int = 4
    max_wall_seconds: float | None = None
    use_fast_teacher: bool = True
    teacher_margin_bb: float = 0.4
    teacher_min_samples: int = 200
    torch_compile: bool = False
    retain_teacher_cache: bool = True
    teacher_cache_max_entries: int = 8192
    rollout_workers: int = 0
    obs_version: int = 1
    teacher_version: int = 1
    checkpoint_path: Path | None = None


def collect_trajectory(
    hero_model: AoFPolicyNet,
    opponent_model: AoFPolicyNet | None,
    device: torch.device,
    hero_seat: str,
    rng: random.Random,
    cache: NashTeacherCache,
    cfg: AoFSelfPlayConfig,
    *,
    ablate_teacher: bool = False,
    table_size: int | None = None,
) -> list[dict]:
    """Collect hero decision steps from one hand."""
    if table_size is None:
        table_size = rng.choice(cfg.table_sizes)
    state = reset_hand(rng, table_size=table_size, stack_bb=cfg.stack_bb)
    traj: list[dict] = []
    opp = opponent_model if opponent_model is not None else hero_model

    def act_for_seat(
        st: AoFHandState,
        s: str,
        model: AoFPolicyNet,
        *,
        record: bool,
    ) -> int:
        if st.terminal or st.action_seat != s:
            return ACTION_FOLD
        t_result = cache.get_with_path(
            st, s, samples=cfg.teacher_samples, seed=cfg.seed + st.hand_id
        )
        teacher = t_result.teacher
        if ablate_teacher:
            from aof import NashTeacher

            teacher = NashTeacher(
                action="FOLD",
                ev_primary=0.0,
                ev_alt=0.0,
                edge_bb=0.0,
                p_fold=0.0,
                facing=st.facing_shove(),
            )
        obs = encode_obs(
            st,
            s,
            teacher,
            version=cfg.obs_version,
            teacher_path=t_result.path,
            provenance=t_result.provenance,
        )
        obs_t = torch.from_numpy(obs).unsqueeze(0).to(device)
        with torch.no_grad():
            out = model(obs_t)
            dist = torch.distributions.Categorical(logits=out.logits)
            idx = int(dist.sample().item())
        if record and s == hero_seat:
            log_probs = torch.log_softmax(out.logits, dim=-1)
            traj.append(
                {
                    "obs": obs,
                    "action": idx,
                    "teacher_action": teacher.action_index,
                    "log_prob": float(log_probs[0, idx].item()),
                    "value": float(out.value.item()),
                    "reward": 0.0,
                    "done": False,
                }
            )
        return idx

    guard = 0
    while not state.terminal:
        guard += 1
        if guard > 32:
            raise RuntimeError("Hand guard exceeded")
        seat = state.action_seat
        if seat is None:
            break
        m = hero_model if seat == hero_seat else opp
        action = act_for_seat(
            state, seat, m, record=(seat == hero_seat)
        )
        state = step_hand(
            state,
            action,
            payoff_samples=cfg.payoff_samples,
            payoff_seed=cfg.seed + state.hand_id * 17 + guard,
        )

    reward = state.payoffs.get(hero_seat, 0.0)
    if traj:
        traj[-1]["reward"] = reward
        traj[-1]["done"] = True
    return traj


def train_self_play(
    model: AoFPolicyNet | None = None,
    config: AoFSelfPlayConfig | None = None,
) -> AoFPolicyNet:
    warmup()
    from aof_equity_cache import get_equity_cache

    get_equity_cache().clear()
    cfg = config or AoFSelfPlayConfig()
    cfg.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(cfg.seed)
    device = torch.device(cfg.device)

    odim = obs_dim(cfg.obs_version)
    if model is None:
        model = AoFPolicyNet(obs_dim=odim).to(device)
    else:
        if model.obs_dim != odim:
            raise ValueError(
                f"Model obs_dim={model.obs_dim} != config obs_dim={odim}"
            )
        model = model.to(device)

    league: list[LeagueAgent] = []
    if cfg.league_size > 0:
        for i in range(cfg.league_size):
            copy_m = AoFPolicyNet().to(device)
            copy_m.load_state_dict(model.state_dict())
            league.append(LeagueAgent(model=copy_m, name=f"agent_{i}"))

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    ppo_cfg = PPOConfig(teacher_coef=cfg.lambda_teacher, ppo_epochs=cfg.ppo_epochs)
    cache = NashTeacherCache(
        use_fast=cfg.use_fast_teacher,
        teacher_margin_bb=cfg.teacher_margin_bb,
        teacher_min_samples=cfg.teacher_min_samples,
        max_entries=cfg.teacher_cache_max_entries,
        teacher_version=cfg.teacher_version,
    )
    if cfg.torch_compile and device.type == "cuda":
        try:
            import triton  # noqa: F401

            model = torch.compile(model, mode="reduce-overhead")  # type: ignore[assignment]
        except Exception as exc:
            print(f"torch.compile disabled ({exc})")
    t_start = time.monotonic()

    for update in range(cfg.total_updates):
        if cfg.max_wall_seconds is not None:
            if time.monotonic() - t_start >= cfg.max_wall_seconds:
                print(f"Wall clock limit reached after {update} updates.", flush=True)
                break
        if not cfg.retain_teacher_cache:
            cache.clear()
        all_traj: list[dict] = []
        if cfg.rollout_workers > 0:
            from aof_rl.parallel_rollout import collect_trajectories_parallel

            all_traj.extend(
                collect_trajectories_parallel(
                    model, cfg, cfg.hands_per_update, rng, workers=cfg.rollout_workers
                )
            )
        else:
            for _ in range(cfg.hands_per_update):
                table_size = rng.choice(cfg.table_sizes)
                hero_seat = rng.choice(active_seats(table_size))
                opp_model = None
                if league:
                    opp_model = rng.choice(league).model
                all_traj.extend(
                    collect_trajectory(
                        model,
                        opp_model,
                        device,
                        hero_seat,
                        rng,
                        cache,
                        cfg,
                        table_size=table_size,
                    )
                )

        if not all_traj:
            continue

        rewards = [t["reward"] for t in all_traj]
        values = [t["value"] for t in all_traj]
        dones = [t["done"] for t in all_traj]
        adv, rets = compute_gae(rewards, values, dones)

        obs = torch.from_numpy(np.stack([t["obs"] for t in all_traj])).float().to(device)
        batch = {
            "obs": obs,
            "action": torch.tensor([t["action"] for t in all_traj], device=device),
            "teacher_action": torch.tensor(
                [t["teacher_action"] for t in all_traj], device=device
            ),
            "old_value": torch.tensor(values, device=device, dtype=torch.float32),
        }
        old_lp = torch.tensor([t["log_prob"] for t in all_traj], device=device)
        advantages = torch.tensor(adv, device=device, dtype=torch.float32)
        returns = torch.tensor(rets, device=device, dtype=torch.float32)

        model.train()
        stats_last: dict[str, float] = {}
        for _ in range(ppo_cfg.ppo_epochs):
            loss, stats_last = ppo_loss(
                model, batch, old_lp, advantages, returns, ppo_cfg
            )
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), ppo_cfg.max_grad_norm)
            opt.step()

        bb_hand = sum(rewards) / max(len(rewards), 1)
        if update % 10 == 0:
            print(f"Update {update}: bb/hand={bb_hand:.3f} {stats_last}", flush=True)

        if (
            cfg.league_size > 0
            and update > 0
            and update % cfg.league_snapshot_every == 0
        ):
            snap = AoFPolicyNet(obs_dim=odim).to(device)
            snap.load_state_dict(model.state_dict())
            league.append(LeagueAgent(model=snap, elo=1000 + bb_hand * 100, name=f"ckpt_{update}"))
            league.sort(key=lambda a: a.elo, reverse=True)
            del league[cfg.league_size :]

    ckpt = cfg.checkpoint_path or (cfg.checkpoint_dir / "policy_selfplay.pt")
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    profile = "smoke" if cfg.total_updates <= 10 else "self_play"
    torch.save(
        {
            "model": model.state_dict(),
            "config": enrich_checkpoint_meta(cfg, training_profile=profile),
            "update": update + 1,
        },
        ckpt,
    )
    return model
