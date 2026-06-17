"""Initialize SD-CFR NLHE Flax policy from Phase 1 ``PseudoSiamesePolicy`` checkpoint."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np
import torch

from pokerbot.models.checkpoint_loader import load_pseudo_siamese_policy, num_weva_buckets_from_checkpoint
from pokerbot.training.nlhe_policy_net import NLHEWEVAPolicyNet


def _torch_conv_to_flax(weight: np.ndarray) -> np.ndarray:
    # PyTorch (O, I, H, W) -> Flax (H, W, I, O)
    return np.transpose(weight, (2, 3, 1, 0))


def _assign_flax_conv(params: dict, path: tuple[str, ...], torch_weight: np.ndarray, torch_bias: np.ndarray | None) -> None:
    node = params
    for key in path[:-1]:
        node = node[key]
    leaf = path[-1]
    node[leaf]["kernel"] = _torch_conv_to_flax(torch_weight)
    if torch_bias is not None:
        node[leaf]["bias"] = torch_bias


def _assign_flax_dense(params: dict, path: tuple[str, ...], weight: np.ndarray, bias: np.ndarray) -> None:
    node = params
    for key in path[:-1]:
        node = node[key]
    leaf = path[-1]
    node[leaf]["kernel"] = weight.T
    node[leaf]["bias"] = bias


def _assign_bn(
    params: dict,
    batch_stats: dict,
    path: tuple[str, ...],
    state: dict[str, np.ndarray],
    prefix: str,
) -> None:
    pnode = params
    bnode = batch_stats
    for key in path[:-1]:
        pnode = pnode[key]
        bnode = bnode[key]
    leaf = path[-1]
    pnode[leaf]["scale"] = state[f"{prefix}.weight"]
    pnode[leaf]["bias"] = state[f"{prefix}.bias"]
    bnode[leaf]["mean"] = state[f"{prefix}.running_mean"]
    bnode[leaf]["var"] = state[f"{prefix}.running_var"]


def _torch_load_cross_platform(path: Path) -> Any:
    """``torch.load`` a checkpoint that may have been pickled on a different OS.

    Checkpoints saved on Windows embed ``pathlib.WindowsPath`` metadata in their config; unpickling
    those on Linux raises ``NotImplementedError: cannot instantiate 'WindowsPath'`` (and vice-versa).
    The path fields here are inert provenance metadata (weva_index_path/checkpoint_dir/...), unused by
    the loader, so we transiently alias the foreign Path class to this platform's concrete Path so the
    unpickler can build it. Scoped + restored so nothing else sees the alias.
    """
    import pathlib

    foreign = "WindowsPath" if os.name != "nt" else "PosixPath"
    native = pathlib.PosixPath if os.name != "nt" else pathlib.WindowsPath
    orig = getattr(pathlib, foreign)
    setattr(pathlib, foreign, native)
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    finally:
        setattr(pathlib, foreign, orig)


def build_sft_aligned_params(
    sft_path: str | Path,
    *,
    seed: int = 0,
) -> tuple[NLHEWEVAPolicyNet, dict[str, Any]]:
    """Create Flax params with weights copied from a WEVA SFT checkpoint."""
    ckpt = _torch_load_cross_platform(Path(sft_path))
    num_buckets = num_weva_buckets_from_checkpoint(ckpt)
    net = NLHEWEVAPolicyNet(num_weva_buckets=num_buckets)
    import jax

    key = jax.random.PRNGKey(seed)
    dummy = {
        "weva_ids": jnp.zeros((1,), dtype=jnp.int32),
        "cards": jnp.zeros((1, 9, 4, 13), dtype=jnp.float32),
        "actions": jnp.zeros((1, 6, 4, 12), dtype=jnp.float32),
        "position": jnp.zeros((1, net.position_dim), dtype=jnp.float32),
        "legal_mask": jnp.ones((1, net.num_actions), dtype=bool),
    }
    variables = net.init(key, **dummy)
    params = variables["params"]
    batch_stats = variables["batch_stats"]

    sd = ckpt["model"]
    sd_np = {k: v.detach().cpu().numpy() for k, v in sd.items()}

    # WEVA embedding
    params["Embed_0"]["embedding"] = sd_np["weva_embed.weight"]

    # Card encoder: ConvBlock 0, 1 + Linear
    _assign_flax_conv(
        params,
        ("_ConvBlock_0", "Conv_0"),
        sd_np["card_encoder.0.net.0.weight"],
        sd_np["card_encoder.0.net.0.bias"],
    )
    _assign_bn(params, batch_stats, ("_ConvBlock_0", "BatchNorm_0"), sd_np, "card_encoder.0.net.1")
    _assign_flax_conv(
        params,
        ("_ConvBlock_1", "Conv_0"),
        sd_np["card_encoder.1.net.0.weight"],
        sd_np["card_encoder.1.net.0.bias"],
    )
    _assign_bn(params, batch_stats, ("_ConvBlock_1", "BatchNorm_0"), sd_np, "card_encoder.1.net.1")
    _assign_flax_dense(
        params,
        ("Dense_0",),
        sd_np["card_encoder.4.weight"],
        sd_np["card_encoder.4.bias"],
    )

    # Action encoder (same layout)
    _assign_flax_conv(
        params,
        ("_ConvBlock_2", "Conv_0"),
        sd_np["action_encoder.0.net.0.weight"],
        sd_np["action_encoder.0.net.0.bias"],
    )
    _assign_bn(params, batch_stats, ("_ConvBlock_2", "BatchNorm_0"), sd_np, "action_encoder.0.net.1")
    _assign_flax_conv(
        params,
        ("_ConvBlock_3", "Conv_0"),
        sd_np["action_encoder.1.net.0.weight"],
        sd_np["action_encoder.1.net.0.bias"],
    )
    _assign_bn(params, batch_stats, ("_ConvBlock_3", "BatchNorm_0"), sd_np, "action_encoder.1.net.1")
    _assign_flax_dense(
        params,
        ("Dense_1",),
        sd_np["action_encoder.4.weight"],
        sd_np["action_encoder.4.bias"],
    )

    # Position + policy head
    _assign_flax_dense(
        params,
        ("Dense_2",),
        sd_np["pos_embed.weight"],
        sd_np["pos_embed.bias"],
    )
    _assign_flax_dense(
        params,
        ("Dense_3",),
        sd_np["policy_head.0.weight"],
        sd_np["policy_head.0.bias"],
    )
    _assign_flax_dense(
        params,
        ("Dense_4",),
        sd_np["policy_head.2.weight"],
        sd_np["policy_head.2.bias"],
    )

    return net, {"params": params, "batch_stats": batch_stats}


def init_trainer_from_sft(trainer: Any, sft_path: str | Path) -> None:
    """Replace trainer policy variables with SFT-transferred weights (trainer must use same net)."""
    _net, variables = build_sft_aligned_params(sft_path)
    trainer._net = _net
    trainer._batch_stats = variables["batch_stats"]
    trainer.state = trainer.state.replace(params=variables["params"])


def verify_sft_forward_parity(
    sft_path: str | Path,
    *,
    records: list[dict[str, Any]] | None = None,
    device: str = "cpu",
    max_rows: int = 20,
    kl_tol: float = 1e-3,
) -> dict[str, Any]:
    """Flax init vs PyTorch SFT forward KL on replayed Parquet rows."""
    from pokerbot.training.nlhe_spot_replay import encode_bundle_from_parquet_record, load_heldout_records
    import jax

    if records is None:
        records = load_heldout_records()
    records = records[:max_rows]

    net, variables = build_sft_aligned_params(sft_path)
    params = variables["params"]
    batch_stats = variables["batch_stats"]
    torch_model = load_pseudo_siamese_policy(sft_path, device=device)
    kls: list[float] = []

    for rec in records:
        bundle, _mask = encode_bundle_from_parquet_record(rec)
        mask = np.asarray(rec["legal_mask"], dtype=bool)
        target = np.asarray(rec["solver_policy"], dtype=np.float64)

        inp = {
            "weva_ids": jnp.asarray([bundle.weva_id], dtype=jnp.int32),
            "cards": jnp.asarray(bundle.cards, dtype=jnp.float32).reshape(1, 9, 4, 13),
            "actions": jnp.asarray(bundle.actions, dtype=jnp.float32).reshape(1, 6, 4, 12),
            "position": jnp.asarray(bundle.policy_position, dtype=jnp.float32)[None, :],
            "legal_mask": jnp.asarray(mask, dtype=bool)[None, :],
        }
        logits = net.apply({"params": params, "batch_stats": batch_stats}, **inp)
        flax_p = np.asarray(jax.nn.softmax(logits[0]))

        with torch.no_grad():
            b = {
                "weva_ids": torch.tensor([bundle.weva_id], device=device),
                "cards": torch.tensor(bundle.cards, device=device).unsqueeze(0),
                "actions": torch.tensor(bundle.actions, device=device).unsqueeze(0),
                "position": torch.tensor(bundle.policy_position, device=device).unsqueeze(0),
                "legal_mask": torch.tensor(mask, device=device).unsqueeze(0),
            }
            out = torch_model(**b)
            torch_p = torch.softmax(out.logits, dim=-1).cpu().numpy()[0]

        kl_torch_sft = float(np.sum(target * (np.log(target + 1e-12) - np.log(torch_p + 1e-12))))
        kl_flax_torch = float(np.sum(torch_p * (np.log(torch_p + 1e-12) - np.log(flax_p + 1e-12))))
        kls.append(kl_flax_torch)

    return {
        "n_rows": len(kls),
        "mean_kl_flax_vs_torch": float(np.mean(kls)) if kls else float("inf"),
        "max_kl_flax_vs_torch": float(np.max(kls)) if kls else float("inf"),
        "passed": bool(kls) and max(kls) <= kl_tol,
        "kl_tol": kl_tol,
    }
