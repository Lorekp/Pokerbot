#!/usr/bin/env python3
"""Stage 1: bias vs variance diagnosis for trio MC tensor builder."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np

from aof_cfr.buckets import LABELS_169
from aof_cfr.cpu_pool import pin_compute_threads
from aof_cfr.terminal_ev import _exact_nway_shares_numba
from aof_cfr.trio_eval_mc import eval7_backend_name, mc_nway_shares
from aof_cfr.trio_payoff_tensor import (
    DEFAULT_MC_BOARDS,
    _all_combo_triples,
    _avg_shares_for_triple_legacy_per_combo,
    _combo_seed,
    exact_shares_for_triple_oracle,
)
from aof_equity import warmup
from aof_equity_grids import _card_str_to_id

# Step 1A: current code calls mc_nway_shares(..., mc_boards) once per valid combo,
# then averages — NOT a single per-triple budget.
SAMPLING_REGIME = "per_combo_mc_boards_then_uniform_combo_average"


@dataclass
class LadderPoint:
    boards: int
    max_error: float
    share_error: float


@dataclass
class FitResult:
    slope: float
    intercept: float
    r2: float
    branch: str
    reason: str


def bucket_indices(label_a: str, label_b: str, label_c: str) -> tuple[int, int, int]:
    idx = {lab: i for i, lab in enumerate(LABELS_169)}
    return idx[label_a], idx[label_b], idx[label_c]


def triple_max_error(i: int, j: int, k: int, *, mc_boards: int) -> tuple[float, float, np.ndarray, np.ndarray]:
    s0, s1, s2, bb, bs, sb, _ = _avg_shares_for_triple_legacy_per_combo(
        i, j, k, mc_boards=mc_boards
    )
    mc = np.array([s0, s1, s2, bb, bs, sb], dtype=np.float64)
    o0, o1, o2, obb, obs, osb, _ = exact_shares_for_triple_oracle(i, j, k)
    oracle = np.array([o0, o1, o2, obb, obs, osb], dtype=np.float64)
    err = float(np.max(np.abs(mc - oracle)))
    share_err = float(np.max(np.abs(mc[:3] - oracle[:3])))
    return err, share_err, mc, oracle


def fit_inv_sqrt(points: list[LadderPoint]) -> FitResult:
    if len(points) < 3:
        return FitResult(0, 0, 0, "ambiguous", "too few ladder points")
    n = np.array([p.boards for p in points], dtype=np.float64)
    y = np.array([p.max_error for p in points], dtype=np.float64)
    x = 1.0 / np.sqrt(n)
    A = np.vstack([x, np.ones(len(x))]).T
    coef, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
    slope, intercept = float(coef[0]), float(coef[1])
    pred = slope * x + intercept
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-15 else 0.0
    tail = points[-3:]
    flat = max(p.max_error for p in tail) - min(p.max_error for p in tail) < 0.0015
    decreasing = points[-1].max_error < points[0].max_error * 0.85
    if flat and y[-1] > 0.003:
        branch = "bias"
        reason = f"error flat above exact at high boards (tail spread {flat})"
    elif decreasing and intercept < 0.002 and r2 > 0.85:
        branch = "variance"
        reason = f"tracks 1/sqrt(n) toward exact (R2={r2:.3f}, intercept={intercept:.5f})"
    elif decreasing and r2 > 0.7:
        branch = "variance"
        reason = f"decreasing with 1/sqrt(n) fit R2={r2:.3f}"
    else:
        branch = "ambiguous"
        reason = f"R2={r2:.3f} intercept={intercept:.5f} — extend ladder or dig"
    return FitResult(slope, intercept, r2, branch, reason)


def combo_count(i: int, j: int, k: int) -> int:
    return len(_all_combo_triples(i, j, k))


def diagnose_4max_spot(hole_ids: list[int], *, boards: int, seed: int = 0) -> dict:
    flat = np.array(hole_ids, dtype=np.int32)
    n = len(hole_ids) // 2
    exact = _exact_nway_shares_numba(flat, n)
    mc = mc_nway_shares(flat, n, boards, seed)
    err = float(np.max(np.abs(mc - exact)))
    return {
        "n_players": n,
        "boards": boards,
        "exact": exact.tolist(),
        "mc": mc.tolist(),
        "max_error": err,
    }


def run_ladder(
    i: int, j: int, k: int, boards_list: list[int]
) -> tuple[list[LadderPoint], FitResult]:
    points: list[LadderPoint] = []
    for b in boards_list:
        err, share_err, _, _ = triple_max_error(i, j, k, mc_boards=b)
        points.append(LadderPoint(boards=b, max_error=err, share_error=share_err))
        print(
            f"  {LABELS_169[i]}/{LABELS_169[j]}/{LABELS_169[k]} "
            f"boards={b} max_err={err:.5f} share_err={share_err:.5f}",
            flush=True,
        )
    return points, fit_inv_sqrt(points)


def main() -> None:
    p = argparse.ArgumentParser(description="Stage 1 MC bias/variance diagnosis")
    p.add_argument("--out", default="data/aof/cfr_base/stage1_diagnosis.json")
    p.add_argument(
        "--boards",
        default="128,256,512,768,1024,2048,4096,8192",
        help="Comma-separated MC board ladder",
    )
    p.add_argument("--four-max-boards", type=int, default=4096)
    args = p.parse_args()

    warmup()
    pin_compute_threads(1)

    boards_list = [int(x) for x in args.boards.split(",") if x.strip()]
    triples = [
        ("22", "22", "33"),
        ("22", "42o", "97s"),
        ("AKs", "AKs", "AKs"),
    ]

    report: dict = {
        "sampling_regime": SAMPLING_REGIME,
        "regime_note": (
            "mc_nway_shares(flat,3,mc_boards,seed) is invoked once per valid combo; "
            "triple value is uniform average over combos. "
            "Total board draws for C combos ≈ C * mc_boards."
        ),
        "eval_backend": eval7_backend_name(),
        "default_mc_boards": DEFAULT_MC_BOARDS,
        "triples": [],
        "four_max_spots": [],
    }

    branches: list[str] = []
    print(f"Sampling regime: {SAMPLING_REGIME}", flush=True)
    print(f"Backend: {eval7_backend_name()}", flush=True)

    for a, b, c in triples:
        i, j, k = bucket_indices(a, b, c)
        cnt = combo_count(i, j, k)
        expected_sigma_per_combo = np.sqrt(0.25 / 768) if 768 else 0.0
        expected_sigma_mean = expected_sigma_per_combo / np.sqrt(max(cnt, 1))
        print(
            f"\nTriple {a}/{b}/{c} combos={cnt} "
            f"expected_sigma_if_iid~{expected_sigma_mean:.4f} "
            f"at 768/combo",
            flush=True,
        )
        points, fit = run_ladder(i, j, k, boards_list)
        branches.append(fit.branch)
        report["triples"].append(
            {
                "label": f"{a}/{b}/{c}",
                "indices": [i, j, k],
                "combo_count": cnt,
                "expected_sigma_at_768_per_combo": float(expected_sigma_per_combo),
                "expected_sigma_mean_at_768": float(expected_sigma_mean),
                "ladder": [asdict(pt) for pt in points],
                "fit": asdict(fit),
            }
        )

    # 4-max hand-checkable collision spots (shared mc_nway_shares kernel)
    four_spots = [
        ["2c", "2d", "2h", "2s", "3c", "3d", "3h", "3s"],
        ["As", "Ah", "Ks", "Kh", "Qs", "Qh", "Js", "Jh"],
    ]
    print("\n4-max MC vs exact (shared kernel):", flush=True)
    for cards in four_spots:
        ids = [_card_str_to_id(c) for c in cards]
        spot = diagnose_4max_spot(ids, boards=args.four_max_boards)
        spot["cards"] = cards
        report["four_max_spots"].append(spot)
        print(f"  {cards} err={spot['max_error']:.5f} @ {args.four_max_boards} boards", flush=True)

    if "bias" in branches:
        overall = "bias"
    elif all(b == "variance" for b in branches):
        overall = "variance"
    elif "ambiguous" in branches:
        overall = "ambiguous"
    else:
        overall = "mixed"
    report["recommended_branch"] = overall
    report["branch_counts"] = {b: branches.count(b) for b in set(branches)}

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nRecommended branch: {overall}", flush=True)
    print(f"Wrote {out}", flush=True)
    if overall == "ambiguous":
        sys.exit(2)
    if overall == "bias":
        sys.exit(3)


if __name__ == "__main__":
    main()
