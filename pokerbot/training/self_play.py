"""K-Best league self-play for 6-max NLH."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from pokerbot.engine.actions import Action, ActionType, index_to_bucket, NUM_ACTIONS
from pokerbot.engine.game import GameState, NUM_PLAYERS, play_hand
from pokerbot.encoding.tensors import encode_state
from pokerbot.models.policy import PseudoSiamesePolicy
from pokerbot.training.ppo import PPOConfig, compute_gae, trinal_clip_ppo_loss


@dataclass
class LeagueAgent:
    model: PseudoSiamesePolicy
    elo: float = 1000.0
    name: str = "agent"


@dataclass
class SelfPlayConfig:
    league_size: int = 5
    hands_per_update: int = 64
    total_updates: int = 50
    lr: float = 3e-5
    device: str = "cpu"
    checkpoint_dir: Path = Path("checkpoints")
    seed: int = 42


def _legal_mask(state: GameState, seat: int) -> np.ndarray:
    legal = state.legal_actions(seat)
    from pokerbot.engine.actions import action_to_index

    mask = np.zeros(NUM_ACTIONS, dtype=bool)
    for a in legal:
        mask[action_to_index(a)] = True
    return mask


def _index_to_action(state: GameState, seat: int, index: int) -> Action:
    legal = state.legal_actions(seat)
    from pokerbot.engine.actions import action_to_index

    for a in legal:
        if action_to_index(a) == index:
            return a
    # Map bucket index to nearest legal action
    if index == 0:
        return Action(ActionType.FOLD)
    if index == 1:
        return next((a for a in legal if a.action_type == ActionType.CHECK), legal[0])
    if index == 2:
        return next((a for a in legal if a.action_type == ActionType.CALL), legal[0])
    if index >= NUM_ACTIONS - 1:
        return next((a for a in legal if a.action_type == ActionType.ALL_IN), legal[-1])
    bucket = index_to_bucket(index)
    if bucket is not None:
        atype = ActionType.BET if state.current_bet_bb <= 1e-9 else ActionType.RAISE
        target = state.pot_bb * bucket
        for a in legal:
            if a.action_type in (ActionType.BET, ActionType.RAISE) and abs(a.amount_bb - target) < 1.0:
                return a
    return legal[-1]


def collect_trajectory(
    hero_model: PseudoSiamesePolicy,
    opponent_model: PseudoSiamesePolicy,
    device: torch.device,
    hero_seat: int,
    rng: random.Random,
) -> list[dict]:
    """Collect one hand trajectory from hero perspective."""
    traj: list[dict] = []
    pending: dict | None = None

    def make_policy(model: PseudoSiamesePolicy, seat: int):
        def policy(state: GameState) -> Action:
            nonlocal pending
            enc = encode_state(state, seat)
            weva_ids = torch.tensor([enc["weva_id"]], dtype=torch.long, device=device)
            actions = torch.from_numpy(enc["actions"]).unsqueeze(0).to(device)
            position = torch.from_numpy(enc["position"]).unsqueeze(0).to(device)
            mask = torch.from_numpy(_legal_mask(state, seat)).unsqueeze(0).to(device)
            with torch.no_grad():
                idx, out = model.act(weva_ids, actions, position, mask, deterministic=False)
            if seat == hero_seat:
                log_probs = torch.log_softmax(out.logits, dim=-1)
                pending = {
                    "weva_id": enc["weva_id"],
                    "actions": enc["actions"],
                    "position": enc["position"],
                    "legal_mask": _legal_mask(state, seat),
                    "action": int(idx.item()),
                    "log_prob": float(log_probs[0, idx].item()),
                    "value": float(out.value.item()),
                    "reward": 0.0,
                    "done": False,
                }
                traj.append(pending)
            return _index_to_action(state, seat, int(idx.item()))

        return policy

    policies = {seat: make_policy(hero_model if seat == hero_seat else opponent_model, seat) for seat in range(NUM_PLAYERS)}
    _, deltas = play_hand(policies, rng=rng)
    reward = deltas.get(hero_seat, 0.0)
    if traj:
        traj[-1]["reward"] = reward
        traj[-1]["done"] = True
    return traj


def train_self_play(
    model: PseudoSiamesePolicy | None = None,
    config: SelfPlayConfig | None = None,
) -> PseudoSiamesePolicy:
    config = config or SelfPlayConfig()
    config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(config.seed)
    device = torch.device(config.device)

    if model is None:
        model = PseudoSiamesePolicy().to(device)
    else:
        model = model.to(device)

    league: list[LeagueAgent] = []
    for i in range(config.league_size):
        copy = PseudoSiamesePolicy().to(device)
        copy.load_state_dict(model.state_dict())
        league.append(LeagueAgent(model=copy, name=f"agent_{i}"))

    opt = torch.optim.AdamW(model.parameters(), lr=config.lr)
    ppo_cfg = PPOConfig()

    for update in range(config.total_updates):
        all_traj: list[dict] = []
        for _ in range(config.hands_per_update):
            hero_seat = rng.randrange(NUM_PLAYERS)
            opp = rng.choice(league).model
            all_traj.extend(collect_trajectory(model, opp, device, hero_seat, rng))

        if not all_traj:
            continue

        rewards = [t["reward"] for t in all_traj]
        values = [t["value"] for t in all_traj]
        dones = [t["done"] for t in all_traj]
        adv, rets = compute_gae(rewards, values, dones)

        batch = {
            "weva_ids": torch.tensor([t["weva_id"] for t in all_traj], dtype=torch.long, device=device),
            "actions": torch.from_numpy(np.stack([t["actions"] for t in all_traj])).float().to(device),
            "position": torch.from_numpy(np.stack([t["position"] for t in all_traj])).float().to(device),
            "legal_mask": torch.from_numpy(np.stack([t["legal_mask"] for t in all_traj])).bool().to(device),
            "action": torch.tensor([t["action"] for t in all_traj], device=device),
            "old_value": torch.tensor(values, device=device, dtype=torch.float32),
        }
        old_lp = torch.tensor([t["log_prob"] for t in all_traj], device=device)
        advantages = torch.tensor(adv, device=device, dtype=torch.float32)
        returns = torch.tensor(rets, device=device, dtype=torch.float32)

        model.train()
        for _ in range(ppo_cfg.ppo_epochs):
            loss, stats = trinal_clip_ppo_loss(model, batch, old_lp, advantages, returns, ppo_cfg)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), ppo_cfg.max_grad_norm)
            opt.step()

        # K-best league update: replace worst if main agent wins enough
        bb100 = sum(rewards) / max(len(rewards), 1) * 100
        if update % 10 == 0:
            copy = PseudoSiamesePolicy().to(device)
            copy.load_state_dict(model.state_dict())
            league.append(LeagueAgent(model=copy, elo=1000 + bb100, name=f"ckpt_{update}"))
            league.sort(key=lambda a: a.elo, reverse=True)
            league = league[: config.league_size]

        if update % 10 == 0:
            print(f"Update {update}: bb/hand={bb100:.3f} {stats}")

    ckpt = config.checkpoint_dir / "policy_selfplay.pt"
    torch.save({"model": model.state_dict()}, ckpt)
    return model
