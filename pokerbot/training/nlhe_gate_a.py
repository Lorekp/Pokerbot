"""Gate A: KL of SD-CFR policy vs solver labels on held-out Parquet (JAX spot replay)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from pokerbot.data.sft_dataloader import SolverAnnotatedParquetDataset
from pokerbot.models.checkpoint_loader import load_pseudo_siamese_policy
from pokerbot.training.nlhe_spot_replay import verify_parquet_encode_replay
from pokerbot.training.sd_cfr_nlhe import SDCFRNLHEConfig, SDCFRNLHETrainer, load_nlhe_checkpoint

SFT_BASELINE_COMBINED_KL = 0.2274


def _row_kl(policy: np.ndarray, target: np.ndarray, mask: np.ndarray) -> float:
    p = policy * mask
    p = p / max(p.sum(), 1e-12)
    t = target * mask
    t = t / max(t.sum(), 1e-12)
    p = np.clip(p, 1e-12, 1.0)
    t = np.clip(t, 1e-12, 1.0)
    return float(np.sum(t * (np.log(t) - np.log(p))))


def _range_matched(rec: dict[str, Any]) -> bool:
    raw = rec.get("raw_solver_json")
    if not raw:
        return False
    payload = json.loads(raw) if isinstance(raw, str) else raw
    req = payload.get("solver_request") or {}
    return bool(req.get("oop_range")) and bool(req.get("ip_range"))


def eval_trainer_on_parquet(
    trainer: SDCFRNLHETrainer,
    parquet_path: str | Path,
    *,
    tag: str,
    use_average: bool = True,
    verify_replay: bool = True,
) -> dict[str, Any]:
    ds = SolverAnnotatedParquetDataset(str(parquet_path))
    rows_out: list[dict[str, Any]] = []
    replay_failures = 0

    for i in range(len(ds)):
        rec = ds.records[i]
        mask = np.asarray(rec["legal_mask"], dtype=bool)
        target = np.asarray(rec["solver_policy"], dtype=np.float64)
        try:
            if verify_replay:
                chk = verify_parquet_encode_replay(rec)
                if not all(chk[k] for k in ("weva_ok", "cards_ok", "actions_ok")):
                    replay_failures += 1
            if use_average:
                policy = trainer.eval_strategy_policy_from_parquet(rec)
            else:
                from pokerbot.training.nlhe_spot_replay import encode_bundle_from_parquet_record

                bundle, m = encode_bundle_from_parquet_record(rec)
                policy = trainer.network_strategy_policy_from_bundle(bundle, m)
            kl = _row_kl(policy, target, mask)
        except Exception as exc:
            replay_failures += 1
            kl = float("inf")
            rows_out.append({"row": i, "error": str(exc)[:200]})
            continue
        rows_out.append(
            {
                "row": i,
                "kl_sd_cfr_vs_label": kl,
                "range_matched": _range_matched(rec),
                "spot_class": rec.get("spot_class", ""),
            }
        )

    kls = [r["kl_sd_cfr_vs_label"] for r in rows_out if "kl_sd_cfr_vs_label" in r]
    matched = [r for r in rows_out if r.get("range_matched")]
    return {
        "tag": tag,
        "n_rows": len(rows_out),
        "replay_failures": replay_failures,
        "mean_kl": float(np.mean(kls)) if kls else float("inf"),
        "mean_kl_range_matched": float(np.mean([r["kl_sd_cfr_vs_label"] for r in matched]))
        if matched
        else None,
        "rows": rows_out,
    }


def eval_sft_reference(
    parquet_path: str | Path,
    *,
    sft_path: str,
    device: str = "cpu",
    tag: str,
) -> dict[str, Any]:
    """PyTorch SFT vs labels (Phase 1 reference KL)."""
    ds = SolverAnnotatedParquetDataset(str(parquet_path))
    sft = load_pseudo_siamese_policy(sft_path, device=device)
    rows_out = []
    for i in range(len(ds)):
        rec = ds.records[i]
        mask = np.asarray(rec["legal_mask"], dtype=bool)
        target = np.asarray(rec["solver_policy"], dtype=np.float64)
        sample = ds[i]
        b = {k: v.unsqueeze(0).to(device) for k, v in sample.items()}
        with torch.no_grad():
            out = sft(
                b["weva_ids"],
                b["actions"],
                b["position"],
                b["legal_mask"],
                cards=b["cards"],
            )
            sft_p = torch.softmax(out.logits, dim=-1).cpu().numpy()[0]
        rows_out.append(
            {
                "row": i,
                "kl_sft_vs_label": _row_kl(sft_p, target, mask),
                "range_matched": _range_matched(rec),
            }
        )
    kls = [r["kl_sft_vs_label"] for r in rows_out]
    matched = [r for r in rows_out if r.get("range_matched")]
    return {
        "tag": tag,
        "n_rows": len(rows_out),
        "mean_kl": float(np.mean(kls)),
        "mean_kl_range_matched": float(np.mean([r["kl_sft_vs_label"] for r in matched])) if matched else None,
        "rows": rows_out,
    }


def run_gate_a_report(
    *,
    trainer: SDCFRNLHETrainer | None = None,
    checkpoint: str | Path | None = None,
    sft_path: str = "checkpoints/policy_sft_prod6000.pt",
    heldout_vanilla: str = "data/solver/heldout_prod600_vanilla.parquet",
    heldout_bunching: str = "data/solver/heldout_prod600_bunching.parquet",
    device: str = "cpu",
) -> dict[str, Any]:
    if trainer is None:
        trainer = SDCFRNLHETrainer(SDCFRNLHEConfig(iterations=0, traversals_per_iter=0, sft_checkpoint=None))
        if checkpoint and Path(checkpoint).exists():
            load_nlhe_checkpoint(trainer, Path(checkpoint))

    vanilla = eval_trainer_on_parquet(trainer, heldout_vanilla, tag="vanilla")
    bunching = eval_trainer_on_parquet(trainer, heldout_bunching, tag="bunching")
    combined_kls = [
        r["kl_sd_cfr_vs_label"]
        for r in vanilla["rows"] + bunching["rows"]
        if "kl_sd_cfr_vs_label" in r
    ]

    sft_vanilla = sft_bunching = None
    if Path(sft_path).exists():
        sft_vanilla = eval_sft_reference(heldout_vanilla, sft_path=sft_path, device=device, tag="vanilla_sft")
        sft_bunching = eval_sft_reference(heldout_bunching, sft_path=sft_path, device=device, tag="bunching_sft")

    report = {
        "checkpoint": str(checkpoint) if checkpoint else None,
        "sft_baseline_combined_kl_reference": SFT_BASELINE_COMBINED_KL,
        "combined_mean_kl_sd_cfr": float(np.mean(combined_kls)) if combined_kls else float("inf"),
        "vanilla": vanilla,
        "bunching": bunching,
        "sft_reference_vanilla_mean": sft_vanilla["mean_kl"] if sft_vanilla else None,
        "sft_reference_bunching_mean": sft_bunching["mean_kl"] if sft_bunching else None,
        "sft_reference_combined_mean": (
            float(np.mean([sft_vanilla["mean_kl"], sft_bunching["mean_kl"]]))
            if sft_vanilla and sft_bunching
            else None
        ),
        "gate_a_read": (
            "Diagnostic: SD-CFR policy KL vs solver labels on JAX-replayed spots. "
            "At SFT init (iter 0), combined KL should match SFT reference (~0.227). "
            "Large unexplained drift on range-matched rows during training signals a bug."
        ),
    }
    return report
