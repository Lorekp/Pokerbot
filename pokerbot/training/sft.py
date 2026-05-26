"""Phase 1 supervised fine-tuning on ToolPoker solver-annotated Parquet."""



from __future__ import annotations



import json

from dataclasses import dataclass

from pathlib import Path

from typing import Literal



import torch

from torch.utils.data import DataLoader, random_split



from pokerbot.data.bc_parquet import BCParquetDataset, LABEL_SOURCE_BC, bc_sft_loss, collate_bc_batch

from pokerbot.data.sft_dataloader import (

    LABEL_SOURCE_SOLVER,

    SolverAnnotatedParquetDataset,

    collate_sft_batch,

    weva_sft_loss,

)

from pokerbot.encoding.weva_encoder import (
    WEVA_DEFAULT_K,
    assert_production_weva_index,
    load_weva_bucket_count,
)

from pokerbot.models.policy import PseudoSiamesePolicy



SFTMode = Literal["solver", "bc"]



BC_WARNING = (

    "=" * 72 + "\n"

    "WARNING: Training in BC mode on PokerBench CSV labels.\n"

    "This checkpoint is NOT GTO-verified and must not be used as a solver baseline.\n"

    "For real Phase 1 SFT, export ToolPoker JSONL and use --mode solver.\n"

    + "=" * 72

)





def sft_checkpoint_basename(mode: SFTMode) -> str:
    return "policy_sft_bc.pt" if mode == "bc" else "policy_sft.pt"


def sft_checkpoint_metadata(config: SFTConfig, *, step: int | None = None) -> dict:
    meta = {
        "mode": config.mode,
        "gto_verified": config.mode == "solver",
        "label_source": LABEL_SOURCE_SOLVER if config.mode == "solver" else LABEL_SOURCE_BC,
    }
    if step is not None:
        meta["step"] = step
    return meta





@dataclass

class SFTConfig:

    parquet_glob: str = "data/solver/*.parquet"

    mode: SFTMode = "solver"

    weva_index_path: Path = Path("data/weva")

    batch_size: int = 64

    lr: float = 1e-4

    epochs: int = 3

    hidden: int = 128

    num_weva_buckets: int = WEVA_DEFAULT_K

    solver_beta: float = 0.3

    device: str = "cpu"

    checkpoint_dir: Path = Path("checkpoints")

    amp: bool = False

    grad_accum: int = 1

    num_workers: int = 0

    save_every_steps: int = 500

    max_steps: int | None = None

    val_fraction: float = 0.05





def _build_loaders(config: SFTConfig) -> tuple[DataLoader, DataLoader | None]:

    if config.mode == "solver":

        full_ds = SolverAnnotatedParquetDataset(config.parquet_glob)

        collate_fn = collate_sft_batch

    else:

        print(BC_WARNING)

        full_ds = BCParquetDataset(config.parquet_glob)

        collate_fn = collate_bc_batch



    n = len(full_ds)

    if n == 1:

        train_ds, val_ds = full_ds, None

    else:

        val_n = max(1, int(n * config.val_fraction))

        train_n = n - val_n

        train_ds, val_ds = random_split(

            full_ds,

            [train_n, val_n],

            generator=torch.Generator().manual_seed(42),

        )



    train_loader = DataLoader(

        train_ds,

        batch_size=config.batch_size,

        shuffle=True,

        collate_fn=collate_fn,

        num_workers=config.num_workers,

    )

    val_loader = (

        DataLoader(

            val_ds,

            batch_size=config.batch_size,

            shuffle=False,

            collate_fn=collate_fn,

            num_workers=config.num_workers,

        )

        if val_ds is not None

        else None

    )

    return train_loader, val_loader





def train_sft(config: SFTConfig | None = None) -> PseudoSiamesePolicy:

    config = config or SFTConfig()

    config.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(config.device)



    if config.mode == "solver":
        assert_production_weva_index(config.weva_index_path, context="solver SFT (--mode solver)")

    if not (config.weva_index_path / "centroids.npy").exists():

        raise FileNotFoundError(

            f"WEVA index not found at {config.weva_index_path}. "

            "Run: python scripts/build_weva_index.py --output data/weva"

        )

    num_buckets = load_weva_bucket_count(config.weva_index_path)

    if config.num_weva_buckets != num_buckets:

        print(f"Using WEVA K={num_buckets} from {config.weva_index_path / 'meta.json'}")

        config.num_weva_buckets = num_buckets



    train_loader, val_loader = _build_loaders(config)



    model = PseudoSiamesePolicy(

        hidden=config.hidden,

        num_weva_buckets=config.num_weva_buckets,

    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=config.lr)

    scaler = torch.amp.GradScaler("cuda", enabled=config.amp and device.type == "cuda")



    global_step = 0

    for epoch in range(config.epochs):

        model.train()

        total_loss = 0.0

        batch_count = 0

        opt.zero_grad()



        for batch in train_loader:

            batch = {k: v.to(device) for k, v in batch.items()}

            with torch.amp.autocast("cuda", enabled=config.amp and device.type == "cuda"):

                out = model(

                    batch["weva_ids"],

                    batch["actions"],

                    batch["position"],

                    batch["legal_mask"],

                )

                if config.mode == "solver":

                    loss, _stats = weva_sft_loss(

                        out,

                        batch["solver_label"],

                        batch["solver_policy"],

                        batch["legal_mask"],

                        beta=config.solver_beta,

                    )

                else:

                    loss, _stats = bc_sft_loss(out, batch["bc_label"], batch["legal_mask"])

                loss = loss / config.grad_accum



            scaler.scale(loss).backward()

            total_loss += loss.item() * config.grad_accum

            batch_count += 1



            if batch_count % config.grad_accum == 0:

                scaler.step(opt)

                scaler.update()

                opt.zero_grad()

                global_step += 1



                if config.save_every_steps and global_step % config.save_every_steps == 0:
                    step_name = sft_checkpoint_basename(config.mode).replace(".pt", f"_step{global_step}.pt")
                    torch.save(
                        {
                            "model": model.state_dict(),
                            **sft_checkpoint_metadata(config, step=global_step),
                        },
                        config.checkpoint_dir / step_name,
                    )



                if config.max_steps and global_step >= config.max_steps:

                    break



        if batch_count % config.grad_accum != 0:

            scaler.step(opt)

            scaler.update()

            opt.zero_grad()



        val_acc = 0.0

        if val_loader is not None:

            model.eval()

            correct = 0

            total = 0

            with torch.no_grad():

                for batch in val_loader:

                    batch = {k: v.to(device) for k, v in batch.items()}

                    out = model(

                        batch["weva_ids"],

                        batch["actions"],

                        batch["position"],

                        batch["legal_mask"],

                    )

                    pred = out.logits.argmax(dim=-1)

                    if config.mode == "solver":

                        target = batch["solver_label"]

                    else:

                        target = batch["bc_label"]

                    correct += (pred == target).sum().item()

                    total += target.size(0)

            val_acc = correct / max(total, 1)



        metric_name = "val_solver_acc" if config.mode == "solver" else "val_bc_acc"

        print(

            f"Epoch {epoch + 1}: loss={total_loss / max(batch_count, 1):.4f} "

            f"{metric_name}={val_acc:.2%} mode={config.mode}"

        )



        if config.max_steps and global_step >= config.max_steps:

            break



    ckpt = config.checkpoint_dir / sft_checkpoint_basename(config.mode)

    torch.save(

        {
            "model": model.state_dict(),
            "config": config.__dict__,
            **sft_checkpoint_metadata(config),
        },

        ckpt,

    )

    print(
        f"Saved checkpoint to {ckpt} "
        f"(mode={config.mode}, label_source={sft_checkpoint_metadata(config)['label_source']})"
    )

    return model


