"""Distill neural-CFR blueprint into WEVA-encoded PseudoSiamesePolicy."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import torch
import torch.nn.functional as F

from pokerbot.engine.game import GameState, play_hand
from pokerbot.encoding.eval_scope import encode_state_for_policy
from pokerbot.encoding.weva_encoder import load_weva_bucket_count, WEVA_DEFAULT_K
from pokerbot.eval.bots import RandomBot
from pokerbot.eval.jax_blueprint_bot import JAXBlueprintBot
from pokerbot.models.policy import PseudoSiamesePolicy
from pokerbot.training.self_play import _legal_mask


@dataclass
class BlueprintDistillConfig:
    cfr_checkpoint: Path = Path("checkpoints/history_value_net.pkl")
    output_path: Path = Path("checkpoints/policy_cfr_blueprint.pt")
    hands: int = 200
    epochs: int = 3
    batch_size: int = 64
    lr: float = 5e-5
    hidden: int = 128
    device: str = "cpu"
    seed: int = 42


def _collect_distill_rows(
    teacher: JAXBlueprintBot,
    hands: int,
    rng: random.Random,
) -> list[dict[str, np.ndarray]]:
    rows: list[dict[str, np.ndarray]] = []

    def teacher_policy(state: GameState):
        seat = state.action_seat
        probs = np.asarray(teacher.action_probs(state, seat), dtype=np.float32)
        mask = _legal_mask(state, seat).astype(np.float32)
        probs = probs * mask
        total = probs.sum()
        if total > 0:
            probs /= total
        else:
            probs = mask / max(mask.sum(), 1.0)

        def pick(_s: GameState):
            idx = int(rng.choices(range(len(probs)), weights=probs.tolist(), k=1)[0])
            from pokerbot.training.self_play import _index_to_action

            return _index_to_action(_s, _s.action_seat, idx)

        enc = encode_state_for_policy(state, seat, require_scope=False)
        rows.append(
            {
                "weva_id": np.int64(enc["weva_id"]),
                "cards": enc["cards"].astype(np.float32),
                "actions": enc["actions"].astype(np.float32),
                "position": enc["policy_position"].astype(np.float32),
                "legal_mask": _legal_mask(state, seat),
                "teacher_probs": probs,
            }
        )
        return pick(state)

    opponents = {i: RandomBot(rng) for i in range(6)}
    for h in range(hands):
        for seat in range(6):
            policies = dict(opponents)
            policies[seat] = teacher_policy
            _, _ = play_hand(policies, rng=rng)

    return rows


def distill_blueprint_to_policy(config: BlueprintDistillConfig | None = None) -> PseudoSiamesePolicy:
    config = config or BlueprintDistillConfig()
    device = torch.device(config.device)
    rng = random.Random(config.seed)

    teacher = JAXBlueprintBot(config.cfr_checkpoint)
    rows = _collect_distill_rows(teacher, config.hands, rng)
    if not rows:
        raise RuntimeError("No distillation rows collected")

    weva_k = WEVA_DEFAULT_K
    try:
        weva_k = load_weva_bucket_count(Path("data/weva"))
    except FileNotFoundError:
        pass

    model = PseudoSiamesePolicy(hidden=config.hidden, num_weva_buckets=weva_k).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=config.lr)

    for epoch in range(config.epochs):
        rng.shuffle(rows)
        total_loss = 0.0
        batches = 0
        for start in range(0, len(rows), config.batch_size):
            batch = rows[start : start + config.batch_size]
            weva_ids = torch.tensor([r["weva_id"] for r in batch], dtype=torch.long, device=device)
            cards = torch.tensor(np.stack([r["cards"] for r in batch]), device=device)
            actions = torch.tensor(np.stack([r["actions"] for r in batch]), device=device)
            position = torch.tensor(np.stack([r["position"] for r in batch]), device=device)
            legal_mask = torch.tensor(np.stack([r["legal_mask"] for r in batch]), device=device)
            target = torch.tensor(np.stack([r["teacher_probs"] for r in batch]), device=device)

            out = model(weva_ids, actions, position, legal_mask, cards=cards)
            log_probs = F.log_softmax(out.logits.masked_fill(~legal_mask, -1e9), dim=-1)
            loss = -(target * log_probs).sum(dim=-1).mean()

            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
            batches += 1

        print(f"Distill epoch {epoch + 1}: kl_loss={total_loss / max(batches, 1):.4f}")

    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "source": "neural_cfr_blueprint",
            "cfr_checkpoint": str(config.cfr_checkpoint),
            "gto_verified": False,
            "mode": "cfr_blueprint",
        },
        config.output_path,
    )
    print(f"Saved CFR blueprint policy to {config.output_path}")
    return model
