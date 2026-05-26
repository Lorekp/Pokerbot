"""Distill CNN policy from cached LLM + GTO labels."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from pokerbot.encoding.weva_encoder import WEVAMapper, load_weva_bucket_count, weva_id_from_tensors
from pokerbot.models.policy import PseudoSiamesePolicy
from pokerbot.training.distill_cache import load_label_cache


@dataclass
class DistillConfig:
    cache_path: Path = Path("data/cache/llm_labels.parquet")
    checkpoint_in: Path | None = Path("checkpoints/policy_sft.pt")
    checkpoint_dir: Path = Path("checkpoints")
    weva_index_path: Path = Path("data/weva")
    alpha: float = 0.5
    disagreement_only: bool = False
    batch_size: int = 64
    lr: float = 5e-5
    epochs: int = 2
    device: str = "cuda"
    amp: bool = True
    hidden: int = 128
    num_weva_buckets: int | None = None


class DistillDataset(Dataset):
    def __init__(
        self,
        records: list[dict],
        disagreement_only: bool = False,
        mapper: WEVAMapper | None = None,
    ) -> None:
        if disagreement_only:
            records = [r for r in records if not r.get("agreement", True)]
        self.records = records
        self.mapper = mapper

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        r = self.records[idx]
        cards = np.asarray(r["cards"], dtype=np.float32)
        position = np.asarray(r["position"], dtype=np.float32)
        weva_id = weva_id_from_tensors(cards, position, mapper=self.mapper)
        return {
            "weva_ids": torch.tensor(weva_id, dtype=torch.long),
            "actions": torch.tensor(r["actions"], dtype=torch.float32),
            "position": torch.tensor(r["position"], dtype=torch.float32),
            "legal_mask": torch.tensor(r["legal_mask"], dtype=torch.bool),
            "gto_label": torch.tensor(r["gto_label"], dtype=torch.long),
            "llm_label": torch.tensor(r["llm_label"], dtype=torch.long),
        }


def collate_distill(batch: list[dict]) -> dict[str, torch.Tensor]:
    return {k: torch.stack([b[k] for b in batch]) for k in batch[0]}


def train_distill(config: DistillConfig | None = None) -> PseudoSiamesePolicy:
    config = config or DistillConfig()
    config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(config.device)

    mapper = WEVAMapper.from_directory(config.weva_index_path, device=str(device))
    num_buckets = config.num_weva_buckets or load_weva_bucket_count(config.weva_index_path)

    records = load_label_cache(config.cache_path)
    if not records:
        raise ValueError(f"No records in cache {config.cache_path}")

    n = len(records)
    if n == 1:
        train_records, val_records = records, []
    elif n < 20:
        split = max(1, n - 1)
        train_records, val_records = records[:split], records[split:]
    else:
        split = int(n * 0.95)
        train_records, val_records = records[:split], records[split:]

    train_ds = DistillDataset(train_records, config.disagreement_only, mapper=mapper)
    val_ds = DistillDataset(val_records, mapper=mapper)

    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, collate_fn=collate_distill)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False, collate_fn=collate_distill)

    model = PseudoSiamesePolicy(hidden=config.hidden, num_weva_buckets=num_buckets).to(device)
    if config.checkpoint_in and config.checkpoint_in.exists():
        ckpt = torch.load(config.checkpoint_in, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        print(f"Warm-started from {config.checkpoint_in}")

    opt = torch.optim.AdamW(model.parameters(), lr=config.lr)
    scaler = torch.amp.GradScaler("cuda", enabled=config.amp and device.type == "cuda")
    alpha = config.alpha

    for epoch in range(config.epochs):
        model.train()
        total = 0.0
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.amp.autocast("cuda", enabled=config.amp and device.type == "cuda"):
                out = model(batch["weva_ids"], batch["actions"], batch["position"], batch["legal_mask"])
                loss_gto = F.cross_entropy(out.logits, batch["gto_label"])
                loss_llm = F.cross_entropy(out.logits, batch["llm_label"])
                loss = alpha * loss_gto + (1 - alpha) * loss_llm
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            opt.zero_grad()
            total += loss.item()

        model.eval()
        correct = 0
        total_n = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                out = model(batch["weva_ids"], batch["actions"], batch["position"], batch["legal_mask"])
                pred = out.logits.argmax(dim=-1)
                correct += (pred == batch["gto_label"]).sum().item()
                total_n += batch["gto_label"].size(0)
        print(f"Distill epoch {epoch + 1}: loss={total / max(len(train_loader), 1):.4f} val_gto_acc={correct / max(total_n, 1):.2%}")

    out_path = config.checkpoint_dir / "policy_distilled.pt"
    torch.save({"model": model.state_dict(), "config": config.__dict__}, out_path)
    print(f"Saved {out_path}")
    return model
