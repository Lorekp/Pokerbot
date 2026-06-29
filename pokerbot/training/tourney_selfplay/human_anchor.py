"""human_anchor — the BC trainer for the clone + the FROZEN-anchor loader.

Two responsibilities (design component #3):

1. **BC trainer** (:func:`train_clone`): masked next-action cross-entropy on real human actions, a
   thin fork of ``training/opponent_transformer_train.py`` (AdamW lr=3e-4, bf16-where-available,
   grad-clip 1.0, a Parquet DataLoader over :class:`SkillTrajectoryDataset`). It threads the
   ``skill_bins`` / ``table_skill`` tensors into the clone forward and uses :func:`clone_loss`
   (policy CE + 0.3 belief CE + 0.1 aux BCE — NO MI term). Outputs a ``HumanCloneNet`` checkpoint.

2. **Frozen-anchor loader** (:func:`load_frozen_anchor` + :func:`assert_anchor_frozen`): loads ``τ``
   as a ``no_grad`` / ``eval()`` module and ASSERTS it can never enter any optimizer param group —
   the load-bearing anti-GTO-drift invariant (design §6.1). Every league agent's policy is
   ``softmax((Q+λ·log τ)/(κ+λ))``; if ``τ`` were ever updated the KL target would move and the
   ``D_KL(π‖τ) ≤ O(1/λ)`` guarantee would dissolve. :func:`assert_anchor_frozen(model, optimizer)`
   raises if any anchor parameter id appears in the optimizer's param groups, so a wiring mistake
   fails LOUD at trainer-construction time, not silently during RL.

The top-1 human-action MATCH-RATE metric (:func:`top1_match_rate`) is the realism gate (Maia metric):
it must beat the population-prior argmax AND rise with skill bin (design A4 gate). Computed on the
held-out-by-player split so it is leakage-free.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from pokerbot.training.tourney_selfplay.clone_loss import CloneLossBreakdown, clone_loss
from pokerbot.training.tourney_selfplay.human_clone import HumanCloneConfig, HumanCloneNet


@dataclass
class CloneTrainConfig:
    lr: float = 3e-4
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    batch_size: int = 64
    epochs: int = 10
    lambda_belief: float = 0.3
    lambda_aux: float = 0.1
    device: str = "cpu"
    log_every: int = 50
    max_steps: int | None = None  # cap steps for a smoke run


@dataclass
class CloneTrainState:
    """Running metrics emitted per log step (the log-nontrivial-jobs rule: %/throughput/ETA visible)."""

    step: int = 0
    total_loss: float = 0.0
    policy_loss: float = 0.0
    belief_loss: float = 0.0
    aux_loss: float = 0.0
    history: list[dict] = field(default_factory=list)


def _autocast_dtype(device: str):
    if device.startswith("cuda") and torch.cuda.is_available():
        return torch.bfloat16
    return None  # CPU: full precision (bf16 autocast on CPU is slow/unsupported on many builds)


def train_clone(
    dataset,
    config: CloneTrainConfig | None = None,
    *,
    model: HumanCloneNet | None = None,
) -> tuple[HumanCloneNet, CloneTrainState]:
    """Train (or continue) a :class:`HumanCloneNet` by masked next-action CE on a skill dataset.

    ``dataset`` is a :class:`SkillTrajectoryDataset` (or anything with a ``collate`` producing the
    base batch + ``skill_bins`` / ``table_skill``). Returns the trained model + the final train state.
    """
    cfg = config or CloneTrainConfig()
    device = torch.device(cfg.device)
    model = model or HumanCloneNet(HumanCloneConfig())
    model.to(device)
    model.train()

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=dataset.collate,
        drop_last=False,
    )
    autocast_dtype = _autocast_dtype(cfg.device)
    state = CloneTrainState()

    for epoch in range(cfg.epochs):
        for batch in loader:
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            optimizer.zero_grad(set_to_none=True)

            def _forward() -> CloneLossBreakdown:
                out = model(
                    batch["token_types"],
                    batch["token_features"],
                    batch["attn_mask"],
                    batch["hero_seat"],
                    skill_bins=batch.get("skill_bins"),
                    table_skill=batch.get("table_skill"),
                )
                return clone_loss(
                    out,
                    {
                        "policy_targets": batch["policy_targets"],
                        "belief_targets": batch["belief_targets"],
                        "action_known_mask": batch["action_known_mask"],
                        "cards_known_mask": batch["cards_known_mask"],
                    },
                    lambda_belief=cfg.lambda_belief,
                    lambda_aux=cfg.lambda_aux,
                )

            if autocast_dtype is not None:
                with torch.autocast(device_type="cuda", dtype=autocast_dtype):
                    bd = _forward()
            else:
                bd = _forward()

            bd.total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()

            state.step += 1
            state.total_loss = float(bd.total.detach())
            state.policy_loss = float(bd.policy.detach())
            state.belief_loss = float(bd.belief.detach())
            state.aux_loss = float(bd.aux.detach())
            if state.step % cfg.log_every == 0 or state.step == 1:
                rec = {
                    "step": state.step,
                    "epoch": epoch,
                    "loss": state.total_loss,
                    "policy": state.policy_loss,
                    "belief": state.belief_loss,
                    "aux": state.aux_loss,
                }
                state.history.append(rec)
            if cfg.max_steps is not None and state.step >= cfg.max_steps:
                return model, state
    return model, state


@torch.no_grad()
def top1_match_rate(
    model: HumanCloneNet,
    dataset,
    *,
    device: str = "cpu",
    max_batches: int | None = None,
    batch_size: int = 64,
) -> dict[str, float]:
    """Top-1 human-action match-rate — the realism gate (Maia metric), per skill bin.

    Returns ``{"overall": rate, "by_skill_bin": {bin: rate}, "n": count}``. The design GATE (A4) is:
    overall beats the population-prior argmax AND ``by_skill_bin`` rises monotonically with bin. This
    function computes the rate; the league/trainer applies the gate.
    """
    model.eval()
    model.to(device)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=dataset.collate)
    n_correct = 0
    n_total = 0
    per_bin_correct: dict[int, int] = {}
    per_bin_total: dict[int, int] = {}

    for bi, batch in enumerate(loader):
        if max_batches is not None and bi >= max_batches:
            break
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        out = model(
            batch["token_types"],
            batch["token_features"],
            batch["attn_mask"],
            batch["hero_seat"],
            skill_bins=batch.get("skill_bins"),
            table_skill=batch.get("table_skill"),
        )
        logits = out["policy_logits"]  # (B,5,A)
        targets = batch["policy_targets"]  # (B,5)
        mask = batch["action_known_mask"].bool() & (targets >= 0)
        preds = logits.argmax(dim=-1)
        correct = (preds == targets) & mask
        n_correct += int(correct.sum())
        n_total += int(mask.sum())

        # per-seat skill bin for each known action (the acting opponent's bin).
        if "skill_bins" in batch:
            opp_seat = batch["opp_seat_map"]  # (B,5)
            sb = batch["skill_bins"].gather(1, opp_seat)  # (B,5) — bin of each opponent slot
            for b_bin, c, m in zip(sb[mask].tolist(), correct[mask].tolist(), mask[mask].tolist()):
                per_bin_total[b_bin] = per_bin_total.get(b_bin, 0) + 1
                per_bin_correct[b_bin] = per_bin_correct.get(b_bin, 0) + (1 if c else 0)

    by_bin = {
        b: (per_bin_correct.get(b, 0) / per_bin_total[b])
        for b in sorted(per_bin_total)
        if per_bin_total[b] > 0
    }
    return {
        "overall": (n_correct / n_total) if n_total else 0.0,
        "by_skill_bin": by_bin,
        "n": n_total,
    }


# -- the frozen-anchor invariant -------------------------------------------------------------------


def load_frozen_anchor(
    checkpoint_path: str | Path | None = None,
    *,
    config: HumanCloneConfig | None = None,
    model: HumanCloneNet | None = None,
    device: str = "cpu",
) -> HumanCloneNet:
    """Load ``τ`` as a frozen anchor: ``eval()`` + ``requires_grad_(False)`` on every parameter.

    Either loads ``checkpoint_path`` into a fresh :class:`HumanCloneNet` or freezes an in-memory
    ``model``. The returned module is the immutable anchor every league agent regularizes toward; it
    MUST never be passed to an optimizer (asserted by :func:`assert_anchor_frozen`).
    """
    if model is None:
        model = HumanCloneNet(config or HumanCloneConfig())
        if checkpoint_path is not None:
            ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
            sd = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
            model.load_state_dict(sd)
    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    # Tag the module so a downstream assert can recognize it as a frozen anchor.
    model._is_frozen_anchor = True  # type: ignore[attr-defined]
    return model


def assert_anchor_frozen(anchor: HumanCloneNet, optimizer: torch.optim.Optimizer | None = None) -> None:
    """The load-bearing assert: ``τ`` is frozen and ABSENT from any optimizer param group.

    Raises ``AssertionError`` if (a) any anchor parameter has ``requires_grad``, or (b) any anchor
    parameter id appears in ``optimizer``'s param groups. Call this at trainer construction (and in a
    unit test) so the anti-drift invariant cannot be silently violated by a wiring mistake.
    """
    bad = [n for n, p in anchor.named_parameters() if p.requires_grad]
    assert not bad, (
        f"frozen anchor τ has {len(bad)} trainable params (e.g. {bad[:3]}); "
        "the KL-leash guarantee requires τ to be no_grad/eval"
    )
    assert getattr(anchor, "_is_frozen_anchor", False), (
        "anchor not loaded via load_frozen_anchor — refusing to certify it frozen"
    )
    if optimizer is not None:
        anchor_ids = {id(p) for p in anchor.parameters()}
        for group in optimizer.param_groups:
            for p in group["params"]:
                assert id(p) not in anchor_ids, (
                    "frozen anchor τ parameter found in an optimizer param group — "
                    "this would update the KL target and dissolve the D_KL(π‖τ)≤O(1/λ) leash"
                )


def population_prior_argmax_baseline(contexts: Iterable[str]) -> float:
    """The match-rate FLOOR the clone must beat: the population-prior argmax accuracy.

    Reuses :func:`pokerbot.opponent.population_prior.context_prior` — the argmax outcome-bin per
    context is the "always predict the most common human bin" baseline. This is a coarse K=4-bin
    baseline (the clone predicts the full action index), so it is a conservative floor; the design
    gate is that the clone's top-1 exceeds it. Returns the mean argmax-prior probability mass over the
    given contexts (an upper bound on the bin-level baseline accuracy).
    """
    from pokerbot.opponent.population_prior import context_prior

    masses = []
    for ck in contexts:
        pri = context_prior(ck)
        total = sum(pri)
        if total > 0:
            masses.append(max(pri) / total)
    return sum(masses) / len(masses) if masses else 0.0
