"""Self-play with external league opponents."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from pokerbot.engine.actions import Action, ActionType
from pokerbot.engine.game import GameState, NUM_PLAYERS, Street, play_hand
from pokerbot.encoding.tensors import encode_state
from pokerbot.eval.bots import build_bot
from pokerbot.models.policy import PseudoSiamesePolicy
from pokerbot.opponent.scapt_manager import SCAPTManager
from pokerbot.opponent.sequence_encoder import HandTrajectoryBuilder, TrajectoryRecord
from pokerbot.training.self_play import (
    SelfPlayConfig,
    _index_to_action,
    _legal_mask,
)
from pokerbot.training.ppo import PPOConfig, compute_gae, trinal_clip_ppo_loss
from pokerbot.training.trajectory_dataset import append_trajectories


OpponentFn = Callable[[GameState], Action]


def play_hand_tracked(
    policies: dict[int, OpponentFn],
    hero_seat: int,
    rng: random.Random | None = None,
    builder: HandTrajectoryBuilder | None = None,
) -> tuple[GameState, dict[int, float], TrajectoryRecord | None]:
    """Play one hand while recording causal trajectory tokens for opponent modeling."""
    state = GameState.new_hand(rng=rng)
    state.deal_hole_cards()

    if builder is None:
        builder = HandTrajectoryBuilder(hero_seat=hero_seat)
    else:
        builder.reset(hero_seat)
    builder.init_stacks(state)

    max_actions = 500
    for _ in range(max_actions):
        if state.is_terminal():
            break
        if not state.actors_remaining():
            state.run_out_board()
            break
        seat = state.action_seat
        if state.players[seat].folded or state.players[seat].all_in:
            state.advance_to_next_actor()
            if not state.actors_remaining():
                state.run_out_board()
            continue

        builder.on_decision(state, seat)
        policy = policies.get(seat)
        if policy is None:
            legal = state.legal_actions()
            action = legal[0] if legal else Action(ActionType.FOLD)
        else:
            action = policy(state)
            legal = state.legal_actions()
            if action not in legal:
                action = next(
                    (a for a in legal if a.action_type in (ActionType.CHECK, ActionType.CALL, ActionType.FOLD)),
                    Action(ActionType.FOLD),
                )

        state.apply_action(action)
        builder.on_action(state, seat, action)

        if state.is_terminal():
            break
        state.advance_to_next_actor()
    else:
        raise RuntimeError("Hand exceeded maximum action count")

    if state.street == Street.SHOWDOWN or state.active_player_count() > 1:
        builder.record_showdown(state)

    traj = builder.finalize()
    return state, state.resolve_showdown(), traj


def collect_trajectory_mixed(
    hero_model: PseudoSiamesePolicy,
    opponent_fns: dict[int, OpponentFn],
    device: torch.device,
    hero_seat: int,
    rng: random.Random,
) -> list[dict]:
    traj: list[dict] = []

    def make_cnn_policy(model: PseudoSiamesePolicy, seat: int):
        def policy(state: GameState) -> Action:
            enc = encode_state(state, seat)
            weva_ids = torch.tensor([enc["weva_id"]], dtype=torch.long, device=device)
            actions = torch.from_numpy(enc["actions"]).unsqueeze(0).to(device)
            position = torch.from_numpy(enc["position"]).unsqueeze(0).to(device)
            mask = torch.from_numpy(_legal_mask(state, seat)).unsqueeze(0).to(device)
            with torch.no_grad():
                idx, out = model.act(weva_ids, actions, position, mask, deterministic=False)
            if seat == hero_seat:
                log_probs = torch.log_softmax(out.logits, dim=-1)
                traj.append(
                    {
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
                )
            return _index_to_action(state, seat, int(idx.item()))

        return policy

    policies: dict[int, OpponentFn] = {}
    for seat in range(NUM_PLAYERS):
        if seat == hero_seat:
            policies[seat] = make_cnn_policy(hero_model, seat)
        elif seat in opponent_fns:
            policies[seat] = opponent_fns[seat]
        else:
            policies[seat] = make_cnn_policy(hero_model, seat)

    _, deltas = play_hand(policies, rng=rng)
    reward = deltas.get(hero_seat, 0.0)
    if traj:
        traj[-1]["reward"] = reward
        traj[-1]["done"] = True
    return traj


def collect_opponent_trajectory(
    hero_model: PseudoSiamesePolicy,
    opponent_fns: dict[int, OpponentFn],
    device: torch.device,
    hero_seat: int,
    rng: random.Random,
) -> TrajectoryRecord | None:
    """Play one hand and return MADT trajectory for opponent transformer training."""

    def make_cnn_policy(model: PseudoSiamesePolicy, seat: int):
        def policy(state: GameState) -> Action:
            enc = encode_state(state, seat)
            weva_ids = torch.tensor([enc["weva_id"]], dtype=torch.long, device=device)
            actions = torch.from_numpy(enc["actions"]).unsqueeze(0).to(device)
            position = torch.from_numpy(enc["position"]).unsqueeze(0).to(device)
            mask = torch.from_numpy(_legal_mask(state, seat)).unsqueeze(0).to(device)
            with torch.no_grad():
                idx, _ = model.act(weva_ids, actions, position, mask, deterministic=False)
            return _index_to_action(state, seat, int(idx.item()))

        return policy

    policies: dict[int, OpponentFn] = {}
    for seat in range(NUM_PLAYERS):
        if seat == hero_seat:
            policies[seat] = make_cnn_policy(hero_model, seat)
        elif seat in opponent_fns:
            policies[seat] = opponent_fns[seat]
        else:
            policies[seat] = make_cnn_policy(hero_model, seat)

    _, _, traj = play_hand_tracked(policies, hero_seat, rng=rng)
    return traj


def rollout_opponent_trajectories(
    hero_model: PseudoSiamesePolicy,
    device: torch.device,
    num_hands: int,
    output_dir: Path,
    rng: random.Random | None = None,
    league_opponents: list[str] | None = None,
    scapt: SCAPTManager | None = None,
) -> Path:
    """Run self-play and write trajectory Parquet shards."""
    rng = rng or random.Random()
    league_opponents = league_opponents or ["cnn", "nit", "lag", "donk"]
    trajectories: list[TrajectoryRecord] = []

    for _ in range(num_hands):
        hero_seat = rng.randrange(NUM_PLAYERS)
        if scapt and scapt.population:
            seat_map = scapt.sample_opponent_seats(hero_seat, NUM_PLAYERS, rng)
            opponent_fns: dict[int, OpponentFn] = {}
            for seat, member_id in seat_map.items():
                member_sd = scapt.get_member_policy_state(member_id)
                opp_model = PseudoSiamesePolicy().to(device)
                opp_model.load_state_dict(member_sd)
                opp_model.eval()
                opponent_fns[seat] = build_bot("cnn", rng=rng, model=opp_model, device=device)
        else:
            opp_map = sample_opponent_seats(rng, league_opponents, hero_model, device)
            opponent_fns = {s: b for s, b in opp_map.items() if s != hero_seat}

        traj = collect_opponent_trajectory(hero_model, opponent_fns, device, hero_seat, rng)
        if traj and traj.token_types:
            trajectories.append(traj)

    if not trajectories:
        raise RuntimeError("No trajectories collected")

    return append_trajectories(trajectories, output_dir)


def sample_opponent_seats(
    rng: random.Random,
    opponent_types: list[str],
    model: PseudoSiamesePolicy,
    device: torch.device,
) -> dict[int, OpponentFn]:
    """Assign random external bots to non-hero seats."""
    mapping: dict[int, OpponentFn] = {}
    for seat in range(NUM_PLAYERS):
        name = rng.choice(opponent_types)
        if name == "cnn":
            bot = build_bot("cnn", rng=rng, model=model, device=device)
        else:
            bot = build_bot(name, rng=rng)
        mapping[seat] = bot
    return mapping


def train_self_play_mixed(
    model: PseudoSiamesePolicy | None = None,
    config: SelfPlayConfig | None = None,
    league_opponents: list[str] | None = None,
) -> PseudoSiamesePolicy:
    config = config or SelfPlayConfig()
    league_opponents = league_opponents or ["cnn", "nit", "lag", "donk"]
    config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(config.seed)
    device = torch.device(config.device)

    if model is None:
        model = PseudoSiamesePolicy().to(device)
    else:
        model = model.to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=config.lr)
    ppo_cfg = PPOConfig()

    for update in range(config.total_updates):
        all_traj: list[dict] = []
        opp_map = sample_opponent_seats(rng, league_opponents, model, device)
        for _ in range(config.hands_per_update):
            hero_seat = rng.randrange(NUM_PLAYERS)
            hand_opponents = {s: b for s, b in opp_map.items() if s != hero_seat}
            all_traj.extend(collect_trajectory_mixed(model, hand_opponents, device, hero_seat, rng))

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

        if update % 10 == 0:
            bb100 = sum(rewards) / max(len(rewards), 1) * 100
            print(f"Mixed update {update}: bb/hand={bb100:.3f}")

    ckpt = config.checkpoint_dir / "policy_selfplay_mixed.pt"
    torch.save({"model": model.state_dict(), "league_opponents": league_opponents}, ckpt)
    return model
