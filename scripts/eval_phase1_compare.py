#!/usr/bin/env python3
"""Phase 1 gate: solver-SFT WEVA policy vs rank/equity baseline on in-scope postflop HU spots."""

from __future__ import annotations

import argparse
import json
from glob import glob
from pathlib import Path

import numpy as np

from pokerbot.data.sft_dataloader import compute_mean_train_kl
from pokerbot.encoding.eval_scope import require_gate_policy_scope_encoding
from pokerbot.encoding.weva_encoder import is_production_weva_index
from pokerbot.eval.abstraction_bots import RankEquityAbstractBot
from pokerbot.eval.bots import GTOCopyBot
from pokerbot.eval.exploitability import (
    RELATIVE_EXPLOITABILITY_METRIC_NOTE,
    estimate_exploitability_scoped,
    require_exploitability_harness_sanity,
)
from pokerbot.eval.gate_reference import sha256_file
from pokerbot.models.checkpoint_loader import load_pseudo_siamese_policy


def _ci95(samples: np.ndarray) -> tuple[float, float]:
    if len(samples) < 2:
        v = float(samples[0]) if len(samples) else 0.0
        return v, v
    lo, hi = np.percentile(samples, [2.5, 97.5])
    return float(lo), float(hi)


def _resolve_solver_parquet(parquet_glob: str) -> Path | None:
    paths = sorted(glob(parquet_glob))
    for p in paths:
        if Path(p).is_file():
            return Path(p)
    return None


def main() -> None:
    p = argparse.ArgumentParser(description="Phase 1 solver-SFT vs rank/equity baseline gate")
    p.add_argument("--checkpoint", default="checkpoints/policy_sft.pt", help="Solver-SFT checkpoint")
    p.add_argument("--parquet-glob", default="data/solver/train_prod60.parquet")
    p.add_argument("--weva-index", default="data/weva")
    p.add_argument("--device", default="cuda")
    p.add_argument("--hands", type=int, default=400, help="In-scope hands per seed")
    p.add_argument("--seeds", default="42,43,44,45,46", help="Comma-separated held-out eval seeds")
    p.add_argument("--bootstrap", type=int, default=2000, help="Bootstrap draws for margin CI")
    p.add_argument("--output", default="eval_phase1_compare.json")
    args = p.parse_args()

    sanity = require_exploitability_harness_sanity(hands=400, seed=42)
    scope_encoding_precondition = require_gate_policy_scope_encoding(device=args.device)

    parquet_path = _resolve_solver_parquet(args.parquet_glob)
    ckpt_path = Path(args.checkpoint)
    weva_path = Path(args.weva_index)
    weva_production = is_production_weva_index(weva_path)
    meta = (
        json.loads((weva_path / "meta.json").read_text(encoding="utf-8"))
        if (weva_path / "meta.json").exists()
        else {}
    )

    gate_blockers: list[str] = []
    if parquet_path is None:
        gate_blockers.append(f"No solver Parquet at {args.parquet_glob!r}")
    if not ckpt_path.is_file():
        gate_blockers.append(f"Solver-SFT checkpoint missing: {ckpt_path}")
    if not weva_production:
        gate_blockers.append("WEVA index is not production scope")

    sft_fit_precondition: dict[str, object] = {"passed": False, "mean_train_kl": None, "threshold": 0.2}
    if ckpt_path.is_file() and parquet_path is not None:
        model_probe = load_pseudo_siamese_policy(ckpt_path, device=args.device)
        mean_kl = compute_mean_train_kl(model_probe, parquet_path, args.device)
        sft_fit_precondition = {
            "passed": mean_kl < 0.2,
            "mean_train_kl": mean_kl,
            "threshold": 0.2,
            "note": (
                "SFT policy must fit solver targets on training Parquet before gate is interpretable. "
                "High KL means the checkpoint has not learned its supervision signal."
            ),
        }
        if mean_kl >= 0.2:
            gate_blockers.append(
                f"SFT fit precondition failed: mean_train_kl={mean_kl:.4f} >= 0.2 "
                "(policy has not learned solver targets)"
            )

    results: dict[str, object] = {
        "gate_status": "blocked" if gate_blockers else "evaluated",
        "gate_passed": False,
        "gate_blockers": gate_blockers,
        "harness_sanity_precondition": sanity,
        "policy_scope_encoding_precondition": scope_encoding_precondition,
        "sft_fit_precondition": sft_fit_precondition,
        "prior_gate_note": (
            "Previous gate run (2026-05-27) is NOT interpretable as solver-SFT vs baseline: "
            "SFT did not fit its targets (mean_train_kl≈1.65). The 18.19 vs 0.0 result reflected "
            "an untrained policy losing to fixed exploiters — expected and uninformative about "
            "solver-SFT viability. Re-run only after sft_fit_precondition passes."
        ),
        "signed_exploiter_ev_metric_note": (
            "Gate uses exploiter_ev_bb_per_hand (signed). Lower = less exploitable. "
            "Legacy max(0, br_ev) clamp removed — threshold semantics may differ from pre-2026-05-29 runs."
        ),
        "relative_exploitability_metric_note": (
            "DEPRECATED label: values are signed exploiter_ev_bb_per_hand, not clamped relative_exploitability."
        ),
        "eval_scope": {
            "included": "2-player-valid postflop HU (hu_postflop_2p + pseudo_hu_bunching structurally)",
            "excluded": "preflop, multiway (>2 live), hands without in-scope target decision",
            "range_scope_note": (
                "Labels supervise narrow ranges (QQ+,AK / TT-22,AQs+ vanilla; "
                "TT+,AQs+ / 99-22,ATs+ bunching) and 50%+all-in bet tree — "
                "enforced at Parquet/SFT load, not re-verified in exploitability sim."
            ),
            "does_not_certify": [
                "multiway 6-max policy (self-play track, separate)",
                "preflop AoF push/fold ranges (hu_preflop_dcfr, unbuilt)",
                "certified Nash exploitability (fixed heuristic exploiters only)",
            ],
        },
        "solver_parquet": str(parquet_path) if parquet_path else None,
        "solver_sft_checkpoint": str(ckpt_path),
        "weva_index": str(weva_path),
        "weva_build_fingerprint": meta.get("build_fingerprint"),
        "weva_production": weva_production,
        "weva_policy_source": {
            "type": "solver_sft_checkpoint",
            "training_signal": "postflop_solver_parquet",
            "solver_parquet_used": parquet_path is not None,
            "checkpoint": str(ckpt_path),
            "description": (
                "PseudoSiamesePolicy trained via solver SFT on converged postflop-solver labels "
                "(label_source=postflop_solver). Gate compares this checkpoint vs rank/equity baseline."
            ),
        },
    }

    if gate_blockers:
        out = Path(args.output)
        out.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(json.dumps(results, indent=2))
        print(f"Wrote {out}")
        raise SystemExit(1)

    model = load_pseudo_siamese_policy(ckpt_path, device=args.device)
    sft_bot = GTOCopyBot(
        model,
        device=args.device,
        eval_spot_class="hu_postflop_2p",
        require_scope=True,
    )
    rank_bot = RankEquityAbstractBot()
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]

    weva_by_seed: dict[int, float] = {}
    rank_by_seed: dict[int, float] = {}
    weva_br_by_seed: dict[int, float] = {}
    rank_br_by_seed: dict[int, float] = {}
    scope_by_seed: dict[int, dict[str, object]] = {}

    for seed in seeds:
        weva_res, weva_scope = estimate_exploitability_scoped(sft_bot, hands=args.hands, seed=seed)
        rank_res, rank_scope = estimate_exploitability_scoped(rank_bot, hands=args.hands, seed=seed)
        weva_by_seed[seed] = weva_res.exploiter_ev_bb_per_hand
        rank_by_seed[seed] = rank_res.exploiter_ev_bb_per_hand
        weva_br_by_seed[seed] = weva_res.br_ev_bb_per_hand
        rank_br_by_seed[seed] = rank_res.br_ev_bb_per_hand
        scope_by_seed[seed] = {
            "solver_sft": weva_scope,
            "rank_baseline": rank_scope,
        }

    weva_arr = np.array([weva_by_seed[s] for s in seeds])
    rank_arr = np.array([rank_by_seed[s] for s in seeds])
    diff_arr = rank_arr - weva_arr

    rng = np.random.default_rng(0)
    boot = [
        float(
            (
                rank_arr[rng.integers(0, len(seeds), size=len(seeds))]
                - weva_arr[rng.integers(0, len(seeds), size=len(seeds))]
            ).mean()
        )
        for _ in range(args.bootstrap)
    ]
    boot_arr = np.array(boot)
    margin_mean = float(diff_arr.mean())
    margin_ci = _ci95(boot_arr)

    weva_wins_all = all(weva_by_seed[s] < rank_by_seed[s] for s in seeds)
    margin_outside_noise = margin_ci[0] > 0.0
    gate_passed = weva_wins_all and margin_outside_noise

    if gate_passed:
        gate_verdict = "pass"
    elif margin_ci[0] <= 0.0 <= margin_ci[1]:
        gate_verdict = "marginal_in_noise_not_passed"
    else:
        gate_verdict = "fail"

    results.update(
        {
            "gate_status": gate_verdict,
            "gate_passed": gate_passed,
            "gate_blockers": [],
            "hands_per_seed_in_scope": args.hands,
            "seeds": seeds,
            "in_scope_eval_scope_report_by_seed": scope_by_seed,
            "weva_sha256_centroids": sha256_file(weva_path / "centroids.npy")
            if (weva_path / "centroids.npy").exists()
            else None,
            "solver_sft_policy": {
                "exploiter_ev_bb_per_hand_by_seed": weva_by_seed,
                "br_ev_bb_per_hand_by_seed": weva_br_by_seed,
                "mean_exploiter_ev_bb_per_hand": float(weva_arr.mean()),
                "ci95_exploiter_ev_bb_per_hand": list(_ci95(weva_arr)),
                "relative_exploitability_fixed_exploiters_bb_by_seed": weva_by_seed,
                "mean_relative_exploitability_fixed_exploiters_bb": float(weva_arr.mean()),
                "ci95_relative_exploitability_fixed_exploiters_bb": list(_ci95(weva_arr)),
            },
            "rank_equity_baseline": {
                "exploiter_ev_bb_per_hand_by_seed": rank_by_seed,
                "br_ev_bb_per_hand_by_seed": rank_br_by_seed,
                "mean_exploiter_ev_bb_per_hand": float(rank_arr.mean()),
                "ci95_exploiter_ev_bb_per_hand": list(_ci95(rank_arr)),
                "relative_exploitability_fixed_exploiters_bb_by_seed": rank_by_seed,
                "mean_relative_exploitability_fixed_exploiters_bb": float(rank_arr.mean()),
                "ci95_relative_exploitability_fixed_exploiters_bb": list(_ci95(rank_arr)),
            },
            "margin_rank_minus_solver_sft_exploiter_ev_bb": {
                "mean": margin_mean,
                "by_seed": {str(s): rank_by_seed[s] - weva_by_seed[s] for s in seeds},
                "bootstrap_ci95": list(margin_ci),
            },
            "gate_criteria": {
                "blocking": (
                    "harness sanity passed; sft_fit_precondition mean_train_kl < 0.2; "
                    "solver-SFT mean exploiter_ev_bb_per_hand < rank baseline (lower = less exploitable); "
                    "bootstrap margin CI lower bound > 0"
                ),
                "sft_fit_precondition": sft_fit_precondition,
                "solver_sft_wins_all_seeds": weva_wins_all,
                "margin_outside_noise": margin_outside_noise,
                "br_iterations": 0,
                "br_converged": False,
            },
            "gate_certifies_if_passed": (
                "Solver-SFT on documented 2-player-valid postflop scope beats rank/equity abstraction "
                "under fixed heuristic exploiters — NOT multiway policy, NOT preflop AoF, NOT Nash."
            ),
        }
    )

    out = Path(args.output)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))
    print(f"Wrote {out}")
    if not gate_passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
