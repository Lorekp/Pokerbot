#!/usr/bin/env python3
"""SFT generalization diagnostic: held-out KL + gate harness distribution audit."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from pokerbot.data.sft_dataloader import SolverAnnotatedParquetDataset, compute_mean_train_kl
from pokerbot.encoding.eval_scope import (
    GATE_ISOLATED_HARNESS_SCOPE_NOTE,
    TRAINING_SPOT_CLASS_FRACTION,
    encode_state_for_policy,
)
from pokerbot.encoding.spot_scope import SPOT_CLASS_ORDER, encode_spot_scope
from pokerbot.engine.actions import Action, ActionType
from pokerbot.engine.game import GameState, play_hand
from pokerbot.eval.exploitability import classify_phase1_gate_scope, _exploiter_candidates, _isolated_policies
from pokerbot.models.checkpoint_loader import load_pseudo_siamese_policy
from pokerbot.training.self_play import _legal_mask


def compute_parquet_kl(
    model: torch.nn.Module,
    parquet_glob: str | Path,
    device: torch.device | str,
) -> dict[str, float | int]:
    """Mean symmetric KL between policy and solver targets over all Parquet rows."""
    device = torch.device(device)
    ds = SolverAnnotatedParquetDataset(parquet_glob)
    model.eval()
    kls: list[float] = []
    with torch.no_grad():
        for i in range(len(ds)):
            s = ds[i]
            b = {k: v.unsqueeze(0).to(device) for k, v in s.items()}
            out = model(
                b["weva_ids"],
                b["actions"],
                b["position"],
                b["legal_mask"],
                cards=b["cards"],
            )
            target = b["solver_policy"] * b["legal_mask"].float()
            target = target / target.sum(dim=-1, keepdim=True).clamp(min=1e-8)
            kl = F.kl_div(
                F.log_softmax(out.logits.float(), dim=-1),
                target,
                reduction="batchmean",
            ).item()
            kls.append(kl)
    return {
        "n_rows": len(ds),
        "mean_kl": float(np.mean(kls)) if kls else float("inf"),
        "median_kl": float(np.median(kls)) if kls else float("inf"),
        "max_kl": float(np.max(kls)) if kls else float("inf"),
    }


def _interpret_heldout(train_kl: float, heldout_kl: float) -> str:
    if heldout_kl < 0.3:
        return "generalizes"
    if heldout_kl > 0.8:
        return "memorizes"
    return "partial"


def audit_gate_harness_distribution(
    *,
    hands: int = 400,
    seeds: list[int],
    max_attempts_factor: int = 50,
) -> dict[str, object]:
    """
    Replay Phase 1 gate collection loop and record structural scope vs policy encoding.
    """
    import random

    structural_spot_class: Counter[str] = Counter()
    policy_spot_scope: Counter[str] = Counter()
    policy_scope_is_zero = 0
    range_pair_note: Counter[str] = Counter()
    boards_seen: Counter[str] = Counter()
    streets: Counter[str] = Counter()
    total_in_scope = 0
    per_seed: dict[str, dict[str, int]] = {}

    for seed in seeds:
        rng = random.Random(seed)
        collected = 0
        attempts = 0
        max_attempts = max(hands * max_attempts_factor, hands)
        seed_structural: Counter[str] = Counter()

        class _MinimalTarget:
            def __call__(self, state: GameState) -> Action:
                legal = state.legal_actions()
                return next(
                    (a for a in legal if a.action_type == ActionType.CHECK),
                    legal[0],
                )

        class _Probe:
            def __init__(self, inner, seat: int) -> None:
                self.inner = inner
                self.seat = seat
                self.qualifies = False
                self.structural_class = ""
                self.state_at_decision: GameState | None = None

            def __call__(self, state: GameState) -> Action:
                if state.action_seat == self.seat:
                    ok, reason = classify_phase1_gate_scope(state, self.seat)
                    if ok and self.state_at_decision is None:
                        self.qualifies = True
                        self.structural_class = reason
                        self.state_at_decision = state
                return self.inner(state)

        while collected < hands and attempts < max_attempts:
            attempts += 1
            br_seat = attempts % 6
            target_seat = (br_seat + 1) % 6
            hand_rng = random.Random(rng.randint(0, 2**31 - 1))

            qualified = False
            probe_state: GameState | None = None
            probe_class = ""
            for exploiter in _exploiter_candidates(hand_rng):
                probe = _Probe(_MinimalTarget(), target_seat)
                policies = _isolated_policies(exploiter, probe, br_seat, target_seat)
                expl_rng = random.Random(hand_rng.randint(0, 2**31 - 1))
                play_hand(policies, rng=expl_rng)
                if probe.qualifies and probe.state_at_decision is not None:
                    qualified = True
                    probe_state = probe.state_at_decision
                    probe_class = probe.structural_class
                    break

            if not qualified or probe_state is None:
                continue

            st = probe_state
            enc = encode_state_for_policy(
                st, target_seat, spot_class="hu_postflop_2p", require_scope=True
            )
            scope = np.asarray(enc["spot_scope"], dtype=np.float32)
            if scope.sum() == 0:
                policy_scope_is_zero += 1
                policy_spot_scope["all_zero (unknown scope)"] += 1
            else:
                idx = int(np.argmax(scope))
                policy_spot_scope[SPOT_CLASS_ORDER[idx]] += 1

            structural_spot_class[probe_class] += 1
            seed_structural[probe_class] += 1
            streets[st.street.name] += 1
            if st.board:
                boards_seen["".join(c.to_str() for c in st.board[:3])] += 1
            range_pair_note["not_set_at_eval (full live ranges in sim)"] += 1
            collected += 1
            total_in_scope += 1

        per_seed[str(seed)] = dict(seed_structural)

    train_scope_ratio = {"hu_postflop_2p": 40, "pseudo_hu_bunching": 20}
    train_total = sum(train_scope_ratio.values())
    train_scope_frac = {k: v / train_total for k, v in train_scope_ratio.items()}

    eval_structural_total = sum(structural_spot_class.values()) or 1
    eval_structural_frac = {
        k: structural_spot_class[k] / eval_structural_total for k in structural_spot_class
    }

    scope_mismatch = eval_structural_frac.get("pseudo_hu_bunching", 0.0) == 0.0

    return {
        "hands_per_seed": hands,
        "seeds": seeds,
        "total_in_scope_decisions": total_in_scope,
        "structural_spot_class_counts": dict(structural_spot_class),
        "structural_spot_class_fraction": eval_structural_frac,
        "training_spot_class_fraction": train_scope_frac,
        "per_seed_structural_spot_class": per_seed,
        "policy_spot_scope_at_eval": dict(policy_spot_scope),
        "policy_scope_zero_count": policy_scope_is_zero,
        "policy_scope_encoding_note": (
            "Gate audit uses encode_state_for_policy with eval_spot_class=hu_postflop_2p "
            "(isolated harness vanilla-only scope)."
        ),
        "gate_isolated_harness_note": GATE_ISOLATED_HARNESS_SCOPE_NOTE,
        "training_spot_class_fraction": TRAINING_SPOT_CLASS_FRACTION,
        "range_pair_at_eval": dict(range_pair_note),
        "range_pair_training": {
            "hu_postflop_2p": ("QQ+,AK", "TT-22,AQs+"),
            "pseudo_hu_bunching": ("TT+,AQs+", "99-22,ATs+"),
        },
        "street_distribution": dict(streets),
        "unique_flop_boards_sampled": len(boards_seen),
        "top_sampled_boards": dict(boards_seen.most_common(10)),
        "structural_vs_training_aligned": not scope_mismatch,
        "scope_mismatch_reasons": [
            r
            for r, ok in [
                (
                    "harness never structurally classifies pseudo_hu_bunching (isolated 4-folder design)",
                    eval_structural_frac.get("pseudo_hu_bunching", 0.0) > 0,
                ),
                (
                    "simulated hands use full live ranges, not training range subsets",
                    False,
                ),
            ]
            if not ok
        ],
    }


def recommend_next_move(train_kl: float, heldout_kl: float, harness: dict[str, object]) -> str:
    interp = _interpret_heldout(train_kl, heldout_kl)
    scope_issues = harness.get("scope_mismatch_reasons") or []

    if interp == "generalizes":
        base = (
            "Expand solver labels to ~600 within the same range scopes and bet tree; "
            "re-run gate after SFT fit precondition passes on the larger set."
        )
    elif interp == "memorizes":
        base = (
            "Do not expand labels yet. Investigate why conv inductive bias fails to transfer "
            "(architecture/regularization or minimum dataset scale below generalization threshold)."
        )
    else:
        base = (
            "Partial generalization: expand to ~600 and re-run this held-out diagnostic; "
            "expect linear not exponential improvement."
        )

    if scope_issues:
        base += (
            " Gate harness still vanilla-only (~33% bunching training unevaluated); "
            "range subsets not enforced in sim."
        )
    return base


def main() -> None:
    p = argparse.ArgumentParser(description="SFT generalization + gate harness audit")
    p.add_argument("--checkpoint", default="checkpoints/policy_sft.pt")
    p.add_argument("--train-parquet", default="data/solver/train_prod60.parquet")
    p.add_argument("--heldout-vanilla-parquet", default="data/solver/heldout_gen.parquet")
    p.add_argument("--heldout-bunching-parquet", default="data/solver/heldout_bunching.parquet")
    p.add_argument("--heldout-parquet", default=None, help="Deprecated alias for vanilla held-out")
    p.add_argument("--device", default="cuda")
    p.add_argument("--hands", type=int, default=400)
    p.add_argument("--seeds", default="42,43,44,45,46")
    p.add_argument("--output", default="logs/sft_generalization_diagnostic.json")
    p.add_argument("--skip-heldout", action="store_true", help="Skip held-out KL if parquet missing")
    args = p.parse_args()

    device = args.device
    model = load_pseudo_siamese_policy(args.checkpoint, device=device)

    train_stats = compute_parquet_kl(model, args.train_parquet, device)
    train_kl_ref = float(train_stats["mean_kl"])

    vanilla_path = Path(args.heldout_parquet or args.heldout_vanilla_parquet)
    bunching_path = Path(args.heldout_bunching_parquet)

    vanilla_stats: dict[str, object] | None = None
    bunching_stats: dict[str, object] | None = None
    if not args.skip_heldout and vanilla_path.is_file():
        vanilla_stats = compute_parquet_kl(model, vanilla_path, device)
    elif not args.skip_heldout:
        vanilla_stats = {"error": f"missing {vanilla_path}"}

    if not args.skip_heldout and bunching_path.is_file():
        bunching_stats = compute_parquet_kl(model, bunching_path, device)
    elif not args.skip_heldout:
        bunching_stats = {"error": f"missing {bunching_path}"}

    combined_stats: dict[str, object] | None = None
    if (
        vanilla_stats
        and bunching_stats
        and "mean_kl" in vanilla_stats
        and "mean_kl" in bunching_stats
    ):
        n_v = int(vanilla_stats["n_rows"])
        n_b = int(bunching_stats["n_rows"])
        combined_mean = (
            float(vanilla_stats["mean_kl"]) * n_v + float(bunching_stats["mean_kl"]) * n_b
        ) / max(n_v + n_b, 1)
        combined_stats = {
            "n_rows": n_v + n_b,
            "mean_kl": combined_mean,
            "vanilla_rows": n_v,
            "bunching_rows": n_b,
        }

    heldout_stats = vanilla_stats

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    harness = audit_gate_harness_distribution(hands=args.hands, seeds=seeds)

    heldout_kl = (
        float(vanilla_stats["mean_kl"])
        if vanilla_stats and "mean_kl" in vanilla_stats
        else float("nan")
    )
    bunching_kl = (
        float(bunching_stats["mean_kl"])
        if bunching_stats and "mean_kl" in bunching_stats
        else float("nan")
    )
    combined_kl = (
        float(combined_stats["mean_kl"]) if combined_stats else float("nan")
    )
    interpretation = (
        _interpret_heldout(train_kl_ref, combined_kl if not np.isnan(combined_kl) else heldout_kl)
        if not (np.isnan(heldout_kl) and np.isnan(bunching_kl))
        else "heldout_missing"
    )

    report = {
        "train_kl": train_stats,
        "heldout_vanilla_kl": vanilla_stats,
        "heldout_bunching_kl": bunching_stats,
        "heldout_combined_kl": combined_stats,
        "heldout_kl": heldout_stats,
        "heldout_interpretation": interpretation,
        "bunching_vs_vanilla_note": (
            "Compare heldout_bunching_kl.mean_kl to heldout_vanilla_kl.mean_kl after "
            "heldout_bunching.parquet is generated."
        ),
        "interpretation_thresholds": {
            "generalizes": "heldout_kl < 0.3",
            "partial": "0.3 <= heldout_kl <= 0.8",
            "memorizes": "heldout_kl > 0.8",
        },
        "harness_distribution_audit": harness,
        "recommended_next_move": recommend_next_move(
            train_kl_ref,
            combined_kl if not np.isnan(combined_kl) else heldout_kl,
            harness,
        )
        if not (np.isnan(heldout_kl) and np.isnan(bunching_kl))
        else "Generate held-out labels (vanilla + bunching) first.",
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
