#!/usr/bin/env python3
"""Diagnose why solver SFT failed to fit targets."""

from __future__ import annotations

from pathlib import Path

import json
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from pokerbot.data.sft_dataloader import SolverAnnotatedParquetDataset, weva_sft_loss
from pokerbot.engine.actions import ACTION_VOCAB, NUM_ACTIONS
from pokerbot.encoding.toolpoker_encoder import _normalize_dist_key, action_distribution_to_policy
from pokerbot.encoding.weva_encoder import load_weva_bucket_count
from pokerbot.models.policy import PseudoSiamesePolicy


def step1_action_alignment(parquet_path: str) -> dict:
    print("=" * 72)
    print("STEP 1: ACTION-SPACE ALIGNMENT")
    print("=" * 72)
    print("Policy head ACTION_VOCAB (11 actions):")
    for i, a in enumerate(ACTION_VOCAB):
        print(f"  [{i}] {a}")

    df = pd.read_parquet(parquet_path)
    print(f"\nParquet rows: {len(df)}")

    raw_actions: Counter[str] = Counter()
    mapped_mass_lost: list[float] = []
    row_reports: list[dict] = []

    for i, row in df.iterrows():
        raw = json.loads(row["raw_solver_json"])
        dist = raw.get("action_distribution", {})
        for k in dist:
            raw_actions[k] += 1

        legal = np.array(row["legal_mask"], dtype=bool)
        policy = np.array(row["solver_policy"], dtype=np.float32)

        lost_keys = []
        unmapped_mass = 0.0
        for k, v in dist.items():
            nk = _normalize_dist_key(str(k))
            if nk is None or nk not in ACTION_VOCAB:
                lost_keys.append((k, v))
                unmapped_mass += float(v)

        legal_idx = np.where(legal)[0]
        legal_actions = [ACTION_VOCAB[j] for j in legal_idx]
        top_idx = int(np.argmax(policy))

        row_reports.append(
            {
                "row": i,
                "spot_class": row.get("spot_class"),
                "legal_actions": legal_actions,
                "raw_keys": sorted(dist.keys()),
                "unmapped_mass": unmapped_mass,
                "lost_keys": lost_keys,
                "solver_label": int(row["solver_label"]),
                "top_action": ACTION_VOCAB[top_idx],
                "top_prob": float(policy[top_idx]),
                "policy_entropy": float(
                    -(policy[policy > 0] * np.log(policy[policy > 0] + 1e-12)).sum()
                ),
                "weva_bucket": int(row["weva_bucket_id"]),
            }
        )
        mapped_mass_lost.append(unmapped_mass)

    print("\nDistinct raw solver action keys across rows:")
    for k, c in raw_actions.most_common():
        nk = _normalize_dist_key(k)
        in_vocab = nk in ACTION_VOCAB if nk else False
        print(f"  {k!r}: {c} rows, normalized={nk!r}, in_vocab={in_vocab}")

    print("\nPer-row unmapped mass from raw dist:")
    print(f"  rows with any unmapped mass: {sum(1 for m in mapped_mass_lost if m > 1e-6)}/{len(df)}")
    print(f"  max unmapped mass: {max(mapped_mass_lost):.6f}")
    print(f"  mean unmapped mass: {np.mean(mapped_mass_lost):.6f}")

    bucket_targets: dict[int, list] = defaultdict(list)
    for r in row_reports:
        bucket_targets[r["weva_bucket"]].append((r["row"], r["top_action"], r["top_prob"], r["legal_actions"]))

    collisions = {b: v for b, v in bucket_targets.items() if len(v) > 1}
    print(f"\nWEVA bucket usage: {len(bucket_targets)} unique buckets for {len(df)} rows")
    print(f"  buckets with >1 row: {len(collisions)}")
    if collisions:
        for b, items in list(collisions.items())[:8]:
            actions = set(x[1] for x in items)
            legal_variants = set(tuple(x[3]) for x in items)
            print(
                f"  bucket {b}: {len(items)} rows, top_actions={actions}, "
                f"legal_variants={len(legal_variants)}"
            )

    legal_sets = Counter(tuple(sorted(r["legal_actions"])) for r in row_reports)
    print("\nDistinct legal action sets:")
    for ls, c in legal_sets.most_common():
        print(f"  {list(ls)}: {c} rows")

    entropies = [r["policy_entropy"] for r in row_reports]
    print(
        f"\nTarget policy entropy: mean={np.mean(entropies):.3f} "
        f"min={np.min(entropies):.3f} max={np.max(entropies):.3f}"
    )

    return {
        "row_reports": row_reports,
        "unmapped_any": sum(1 for m in mapped_mass_lost if m > 1e-6),
        "bucket_collisions": len(collisions),
        "n_rows": len(df),
    }


def step2b_overfit_test(parquet_path: str, device: str = "cpu", steps: int = 500, lr: float = 1e-3) -> None:
    print("\n" + "=" * 72)
    print(f"STEP 2b: SINGLE-ROW OVERFIT ({steps} steps, lr={lr})")
    print("=" * 72)

    ds = SolverAnnotatedParquetDataset(parquet_path)
    sample = ds[0]
    num_buckets = load_weva_bucket_count(Path("data/weva"))
    model = PseudoSiamesePolicy(num_weva_buckets=num_buckets).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    def log_step(step: int) -> None:
        model.eval()
        with torch.no_grad():
            b = {k: v.unsqueeze(0).to(device) for k, v in sample.items()}
            out = model(b["weva_ids"], b["actions"], b["position"], b["legal_mask"])
            legal = sample["legal_mask"].numpy().astype(bool)
            probs = out.probs[0].cpu().numpy()
            legal_probs = probs[legal]
            legal_probs = legal_probs / legal_probs.sum()
            loss, stats = weva_sft_loss(
                out, b["solver_label"], b["solver_policy"], b["legal_mask"], beta=0.3
            )
            check_idx = list(np.where(legal)[0]).index(1)
            print(
                f"  step {step:4d}: CE={stats['ce_hard']:.4f} "
                f"check_prob={legal_probs[check_idx]:.4f} top={ACTION_VOCAB[int(probs.argmax())]}"
            )

    log_step(0)
    model.train()
    for s in range(1, steps + 1):
        b = {k: v.unsqueeze(0).to(device) for k, v in sample.items()}
        out = model(b["weva_ids"], b["actions"], b["position"], b["legal_mask"])
        loss, _ = weva_sft_loss(out, b["solver_label"], b["solver_policy"], b["legal_mask"], beta=0.3)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if s in (10, 50, 100, 500) or s == steps:
            log_step(s)


def step1b_target_distortion(parquet_path: str) -> int:
    print("\n" + "=" * 72)
    print("STEP 1b: TARGET DISTORTION (bet_0.50 mass lost in encoding)")
    print("=" * 72)
    df = pd.read_parquet(parquet_path)
    distorted = 0
    for _, row in df.iterrows():
        raw = json.loads(row["raw_solver_json"])["action_distribution"]
        bet_raw = float(raw.get("bet_0.50", 0))
        policy = np.array(row["solver_policy"])
        bet_stored = float(policy[4])
        if bet_raw > 0.05 and bet_stored < 0.01:
            distorted += 1
    print(f"Rows where raw bet_0.50 > 5% but stored bet_0.50 ~ 0: {distorted}/{len(df)}")
    return distorted


def step2_gradient_sanity(parquet_path: str, device: str = "cpu") -> dict:
    print("\n" + "=" * 72)
    print("STEP 2: LOSS AND GRADIENT SANITY (single row)")
    print("=" * 72)

    ds = SolverAnnotatedParquetDataset(parquet_path)
    sample = ds[0]
    num_buckets = load_weva_bucket_count(Path("data/weva"))

    model = PseudoSiamesePolicy(num_weva_buckets=num_buckets).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)

    def batch_from_sample(s: dict) -> dict[str, torch.Tensor]:
        return {k: v.unsqueeze(0).to(device) for k, v in s.items()}

    def log_state(step: str) -> dict:
        model.eval()
        with torch.no_grad():
            b = batch_from_sample(sample)
            out = model(b["weva_ids"], b["actions"], b["position"], b["legal_mask"])
            target = sample["solver_policy"].numpy()
            legal = sample["legal_mask"].numpy().astype(bool)
            probs = out.probs[0].cpu().numpy()
            legal_probs = probs[legal]
            legal_probs = legal_probs / legal_probs.sum()
            loss, stats = weva_sft_loss(
                out, b["solver_label"], b["solver_policy"], b["legal_mask"], beta=0.3
            )
        print(f"\n--- {step} ---")
        print(f"Target (legal only): {dict(zip([ACTION_VOCAB[i] for i in np.where(legal)[0]], target[legal]))}")
        print(f"Policy (legal only): {dict(zip([ACTION_VOCAB[i] for i in np.where(legal)[0]], legal_probs))}")
        print(f"CE={stats['ce_hard']:.4f} KL={stats['kl_soft']:.4f} total={float(loss.item()):.4f}")
        return {"ce": stats["ce_hard"], "kl": stats["kl_soft"], "total": float(loss.item())}

    before = log_state("Before training")

    model.train()
    b = batch_from_sample(sample)
    out = model(b["weva_ids"], b["actions"], b["position"], b["legal_mask"])
    loss, _ = weva_sft_loss(out, b["solver_label"], b["solver_policy"], b["legal_mask"], beta=0.3)
    opt.zero_grad()
    loss.backward()
    grad_norm = 0.0
    head_grad_norm = 0.0
    for name, p in model.named_parameters():
        if p.grad is not None:
            g = p.grad.data.norm(2).item()
            grad_norm += g * g
            if "policy_head" in name:
                head_grad_norm += g * g
    grad_norm = grad_norm ** 0.5
    head_grad_norm = head_grad_norm ** 0.5
    print(f"\nGradient norm (all params): {grad_norm:.6f}")
    print(f"Gradient norm (policy_head only): {head_grad_norm:.6f}")

    for _step_i in range(10):
        model.train()
        b = batch_from_sample(sample)
        out = model(b["weva_ids"], b["actions"], b["position"], b["legal_mask"])
        loss, _ = weva_sft_loss(out, b["solver_label"], b["solver_policy"], b["legal_mask"], beta=0.3)
        opt.zero_grad()
        loss.backward()
        opt.step()

    after = log_state("After 10 steps on single row")
    print(f"\nLR={opt.param_groups[0]['lr']}, optimizer=AdamW, frozen_layers=none")
    return {"before": before, "after": after, "grad_norm": grad_norm, "head_grad_norm": head_grad_norm}


def step3_training_budget(parquet_path: str) -> None:
    print("\n" + "=" * 72)
    print("STEP 3: TRAINING BUDGET")
    print("=" * 72)
    ds = SolverAnnotatedParquetDataset(parquet_path)
    n = len(ds)
    train_n = n - max(1, int(n * 0.05))
    batch_size = 64
    epochs = 3
    steps_per_epoch = max(1, train_n // batch_size)
    total_steps = steps_per_epoch * epochs
    print(f"Rows: {n}, train split (~5% val): ~{train_n}")
    print(f"Config used: batch_size={batch_size}, epochs={epochs}, lr=1e-4")
    print(f"Effective gradient steps: ~{total_steps} (train_n/batch * epochs)")
    print(f"Note: with batch_size={batch_size} > train_n={train_n}, each epoch is 1 step on full train set")


def step3_full_train_kl(parquet_path: str, device: str = "cpu") -> float:
    """Compute mean KL on full training set with current checkpoint."""
    print("\n" + "=" * 72)
    print("TRAINING-SET KL (loaded checkpoint)")
    print("=" * 72)
    ckpt_path = "checkpoints/policy_sft.pt"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    num_buckets = load_weva_bucket_count(Path("data/weva"))
    model = PseudoSiamesePolicy(num_weva_buckets=num_buckets).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    ds = SolverAnnotatedParquetDataset(parquet_path)
    kls = []
    with torch.no_grad():
        for i in range(len(ds)):
            s = ds[i]
            b = {k: v.unsqueeze(0).to(device) for k, v in s.items()}
            out = model(b["weva_ids"], b["actions"], b["position"], b["legal_mask"])
            target = (b["solver_policy"] * b["legal_mask"].float())
            target = target / target.sum(dim=-1, keepdim=True).clamp(min=1e-8)
            kl = F.kl_div(
                F.log_softmax(out.logits.float(), dim=-1),
                target,
                reduction="batchmean",
            ).item()
            kls.append(kl)
    mean_kl = float(np.mean(kls))
    print(f"Mean KL on {len(ds)} rows: {mean_kl:.4f}")
    print(f"Median KL: {np.median(kls):.4f}, max KL: {max(kls):.4f}")
    return mean_kl


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--parquet", default="data/solver/train_prod60.parquet")
    p.add_argument("--device", default="cpu")
    p.add_argument("--skip-grad", action="store_true")
    args = p.parse_args()

    s1 = step1_action_alignment(args.parquet)
    distorted = step1b_target_distortion(args.parquet)
    if s1["unmapped_any"] > 0 or distorted > 0:
        print("\n>>> STOP: action-space / legal_mask bug drops bet_0.50 mass from targets.")
    elif s1["bucket_collisions"] > 0:
        print("\n>>> WEVA bucket collisions found — may cause conflicting supervision.")
    if not args.skip_grad:
        step2_gradient_sanity(args.parquet, device=args.device)
        step2b_overfit_test(args.parquet, device=args.device)
    step3_training_budget(args.parquet)
    step3_full_train_kl(args.parquet, device=args.device)
