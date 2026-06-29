#!/usr/bin/env python3
"""Cheap EMD abstraction sweep before spending on a full blueprint run.

This does NOT train poker. It evaluates the bucketer itself:

  * fit EMD centroids for K/runout candidates on sampled flop+turn spots;
  * score them on held-out spots using a higher-fidelity histogram;
  * report occupancy, train/test gap, effective bucket count, and elbow gains;
  * optionally write the best centroid JSON for the next pilot CFR run.

Use this to reject "bigger K sounds better" before launching multi-day CFR.
The K=500 incident was exactly a capacity/overfit warning; this script keeps
the first decision cheap and reproducible.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

BATTERY_NATIVE_CANDIDATES = [
    ROOT / "runs" / "bp_crate_emd" / "battery_native" / "target" / "release",
    ROOT / "runs" / "bp_crate_frozen" / "battery_native" / "target" / "release",
    ROOT,
]


def parse_int_grid(text: str) -> list[int]:
    vals = [int(x.strip()) for x in text.split(",") if x.strip()]
    if not vals:
        raise argparse.ArgumentTypeError("grid must contain at least one integer")
    return vals


def load_native():
    for cand in BATTERY_NATIVE_CANDIDATES:
        if cand.exists():
            s = str(cand)
            if s not in sys.path:
                sys.path.insert(0, s)
    try:
        import battery_native as bn  # type: ignore
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(
            "FATAL: could not import battery_native. Build it first, for example:\n"
            "  cd runs\\bp_crate_emd\\battery_native\n"
            "  cargo build --release --features python\n"
            f"import error: {type(exc).__name__}: {exc}"
        ) from exc
    needed = ("emd_strength_histograms_batch", "emd_fit", "emd_mean_inertia")
    missing = [name for name in needed if not hasattr(bn, name)]
    if missing:
        raise SystemExit(f"FATAL: battery_native is missing EMD functions: {missing}")
    return bn


def sample_spots(n: int, seed: int) -> list[tuple[list[int], list[int]]]:
    """Sample legal (hole[2], board[3 or 4]) spots, half flop and half turn."""
    rng = np.random.default_rng(seed)
    out: list[tuple[list[int], list[int]]] = []
    for i in range(n):
        n_board = 3 if i < (n + 1) // 2 else 4
        cards = rng.choice(52, size=2 + n_board, replace=False)
        out.append(([int(cards[0]), int(cards[1])], [int(c) for c in cards[2:]]))
    rng.shuffle(out)
    return out


def histograms_native(bn, spots, *, n_bins: int, n_runouts: int, seed: int) -> np.ndarray:
    t0 = time.time()
    h = bn.emd_strength_histograms_batch(spots, int(n_bins), int(n_runouts), int(seed))
    arr = np.asarray(h, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != n_bins:
        raise RuntimeError(f"native histograms returned bad shape {arr.shape}")
    print(
        f"  histograms n={len(spots)} bins={n_bins} runouts={n_runouts} "
        f"seed={seed} -> {time.time() - t0:.1f}s",
        flush=True,
    )
    return arr


def assign_metrics(x: np.ndarray, centroids: np.ndarray, *, chunk: int = 4096) -> tuple[float, np.ndarray]:
    """Return mean nearest-centroid EMD and assignment counts."""
    if x.size == 0:
        return 0.0, np.zeros(len(centroids), dtype=np.int64)
    cc = np.cumsum(np.asarray(centroids, dtype=np.float64), axis=1)
    counts = np.zeros(cc.shape[0], dtype=np.int64)
    total = 0.0
    seen = 0
    for lo in range(0, x.shape[0], chunk):
        part = x[lo:lo + chunk]
        xc = np.cumsum(part, axis=1)
        d = np.abs(xc[:, None, :] - cc[None, :, :]).sum(axis=2)
        a = d.argmin(axis=1)
        total += float(d[np.arange(a.size), a].sum())
        counts += np.bincount(a, minlength=cc.shape[0])
        seen += int(a.size)
    return total / max(seen, 1), counts


def occupancy_summary(counts: np.ndarray) -> dict[str, float]:
    total = int(counts.sum())
    k = int(counts.size)
    if total <= 0 or k <= 0:
        return {
            "empty": float(k),
            "min": 0.0,
            "p01": 0.0,
            "p05": 0.0,
            "median": 0.0,
            "max": 0.0,
            "effective_buckets": 0.0,
            "effective_ratio": 0.0,
            "entropy": 0.0,
        }
    p = counts[counts > 0].astype(np.float64) / float(total)
    entropy = float(-(p * np.log(p)).sum())
    effective = float(math.exp(entropy))
    return {
        "empty": float((counts == 0).sum()),
        "min": float(counts.min()),
        "p01": float(np.percentile(counts, 1)),
        "p05": float(np.percentile(counts, 5)),
        "median": float(np.median(counts)),
        "max": float(counts.max()),
        "effective_buckets": effective,
        "effective_ratio": effective / float(k),
        "entropy": entropy,
    }


def row_score(row: dict[str, Any]) -> float:
    """Rank candidates: held-out inertia first, then sparse/overfit penalties."""
    inertia = float(row["test_inertia_mean"])
    gap = max(0.0, float(row["gap_ratio_mean"]) - 0.10)
    eff_pen = max(0.0, 0.70 - float(row["train_effective_ratio_mean"]))
    empty_pen = float(row["train_empty_mean"]) / max(float(row["k"]), 1.0)
    sparse_pen = max(0.0, 3.0 - float(row["train_p01_mean"]))
    return inertia * (1.0 + 1.5 * gap + 0.35 * eff_pen + 0.25 * empty_pen + 0.02 * sparse_pen)


def passes_occupancy_gates(row: dict[str, Any], *, train_samples: int) -> tuple[bool, list[str]]:
    """Pre-CFR occupancy + sample-budget filter (evidence-backed K cap discipline)."""
    reasons: list[str] = []
    k = int(row["k"])
    if train_samples < 50 * k:
        reasons.append("train_samples_below_50xK")
    if float(row.get("test_empty_mean", 0.0)) > 2.0:
        reasons.append("test_empty_buckets_high")
    if float(row.get("test_p01_mean", 0.0)) < 10.0:
        reasons.append("test_p01_occupancy_low")
    if float(row.get("train_effective_ratio_mean", 0.0)) < 0.70:
        reasons.append("train_effective_ratio_low")
    mg = row.get("marginal_gain_vs_prev_k")
    if mg is not None and float(mg) < 0.08:
        reasons.append("marginal_gain_below_elbow")
    return (len(reasons) == 0, reasons)


def _comb2(x: np.ndarray | int) -> np.ndarray | float:
    arr = np.asarray(x, dtype=np.float64)
    out = arr * (arr - 1.0) / 2.0
    if out.ndim == 0:
        return float(out)
    return out


def adjusted_rand_index(labels_a: np.ndarray, labels_b: np.ndarray) -> float:
    """Label-invariant clustering agreement.

    Raw cluster ids from k-means are arbitrary. ARI compares pair co-assignment
    structure instead of label equality and returns 1.0 for identical partitions,
    including exact label permutations.
    """
    a = np.asarray(labels_a)
    b = np.asarray(labels_b)
    if a.shape != b.shape:
        raise ValueError("assignment arrays must have the same shape")
    n = int(a.size)
    if n < 2:
        return 1.0
    _, ai = np.unique(a, return_inverse=True)
    _, bi = np.unique(b, return_inverse=True)
    contingency = np.zeros((int(ai.max()) + 1, int(bi.max()) + 1), dtype=np.int64)
    np.add.at(contingency, (ai, bi), 1)

    sum_comb = float(_comb2(contingency).sum())
    row_comb = float(_comb2(contingency.sum(axis=1)).sum())
    col_comb = float(_comb2(contingency.sum(axis=0)).sum())
    total_comb = float(_comb2(n))
    expected = row_comb * col_comb / total_comb if total_comb > 0 else 0.0
    max_index = 0.5 * (row_comb + col_comb)
    denom = max_index - expected
    if abs(denom) < 1e-12:
        return 1.0 if abs(sum_comb - max_index) < 1e-12 else 0.0
    return float((sum_comb - expected) / denom)


def assignment_stability(
    test_x: np.ndarray,
    centroids_list: list[np.ndarray],
) -> dict[str, Any]:
    """Label-invariant assignment stability across fit seeds."""
    if len(centroids_list) < 2:
        return {
            "n_seeds": float(len(centroids_list)),
            "metric": "adjusted_rand_index",
            "pairwise_agreement_mean": 1.0,
            "pairwise_agreement_min": 1.0,
        }
    assignments = []
    for centroids in centroids_list:
        cc = np.cumsum(np.asarray(centroids, dtype=np.float64), axis=1)
        xc = np.cumsum(test_x, axis=1)
        d = np.abs(xc[:, None, :] - cc[None, :, :]).sum(axis=2)
        assignments.append(d.argmin(axis=1))
    agree = []
    for i in range(len(assignments)):
        for j in range(i + 1, len(assignments)):
            agree.append(adjusted_rand_index(assignments[i], assignments[j]))
    return {
        "n_seeds": float(len(centroids_list)),
        "metric": "adjusted_rand_index",
        "pairwise_agreement_mean": float(np.mean(agree)) if agree else 1.0,
        "pairwise_agreement_min": float(np.min(agree)) if agree else 1.0,
    }


def functional_ev_gate(evidence: dict[str, Any]) -> tuple[bool, list[str]]:
    """Apply the functional micro-EV thresholds from tests/test_emd_quality.py."""
    reasons: list[str] = []
    tasks = evidence.get("tasks") or {}
    if not tasks:
        reasons.append("functional_tasks_missing")
    closed_gap = False
    for name, row in tasks.items():
        gain = row.get("gain_bb")
        try:
            gain_f = float(gain)
        except (TypeError, ValueError):
            reasons.append("functional_{0}_gain_missing".format(name))
            continue
        if gain_f < -0.01:
            reasons.append("functional_{0}_gain_regressed".format(name))
        fc = row.get("frac_gap_closed_by_emd")
        if fc is not None:
            try:
                closed_gap = closed_gap or float(fc) > 0.05
            except (TypeError, ValueError):
                pass
    if tasks and not closed_gap:
        reasons.append("functional_no_task_closes_gap")

    river = evidence.get("river_fallback_sanity") or {}
    try:
        peak = float(river.get("river_peak_mass_min"))
        var = float(river.get("river_hist_variance_mean"))
    except (TypeError, ValueError):
        reasons.append("river_fallback_sanity_missing")
    else:
        if peak < 0.90:
            reasons.append("river_peak_mass_low")
        if var >= 0.08:
            reasons.append("river_hist_variance_high")
    return (len(reasons) == 0, reasons)


def functional_ev_evidence(
    *,
    n_buckets: int,
    n_bins: int,
    n_runouts: int,
    n_spots: int,
    fit_sample: int,
    seed: int,
) -> dict[str, Any]:
    from scripts.abstraction import emd_vs_madehand_quality as emdq

    rng = np.random.default_rng(seed)
    tasks = emdq.functional_micro_ev_tasks(
        rng,
        n_spots=n_spots,
        n_buckets=n_buckets,
        n_bins=n_bins,
        n_runouts=n_runouts,
        fit_sample=fit_sample,
    )
    evidence = {
        "schema": "emd_functional_micro_ev_v1",
        "config": {
            "n_buckets": int(n_buckets),
            "n_bins": int(n_bins),
            "n_runouts": int(n_runouts),
            "n_spots": int(n_spots),
            "fit_sample": int(fit_sample),
            "seed": int(seed),
        },
        "tasks": tasks,
        "river_fallback_sanity": emdq.river_fallback_sanity(
            n_bins=n_bins,
            n_runouts=n_runouts,
            n_buckets=n_buckets,
        ),
    }
    ok, reasons = functional_ev_gate(evidence)
    evidence["pass"] = ok
    evidence["fail_reasons"] = reasons
    return evidence


def assignment_stability_policy(
    *,
    stability_mean: float,
    min_assignment_stability: float | None,
) -> dict[str, Any]:
    """Return the explicit policy for assignment stability.

    The plan requires assignment stability evidence, but does not define a
    numeric ARI cutoff. Default behavior is therefore report-only; callers that
    want stability as a hard kill criterion must pass --min-assignment-stability.
    """
    if min_assignment_stability is None:
        return {
            "mode": "report_only",
            "hard_gate": False,
            "min_assignment_stability": None,
            "pass": None,
            "fail_reasons": [],
            "threshold_source": "plan_requires_stability_but_no_numeric_cutoff",
        }
    passes = float(stability_mean) >= float(min_assignment_stability)
    return {
        "mode": "hard_gate",
        "hard_gate": True,
        "min_assignment_stability": float(min_assignment_stability),
        "pass": passes,
        "fail_reasons": [] if passes else ["assignment_stability_below_min"],
        "threshold_source": "cli_min_assignment_stability",
    }


def aggregate(rows: list[dict[str, Any]]) -> dict[tuple[int, int], dict[str, Any]]:
    grouped: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for r in rows:
        grouped.setdefault((int(r["n_runouts"]), int(r["k"])), []).append(r)

    out: dict[tuple[int, int], dict[str, Any]] = {}
    numeric_keys = [
        "train_inertia",
        "test_inertia",
        "gap_ratio",
        "train_empty",
        "train_min",
        "train_p01",
        "train_p05",
        "train_median",
        "train_effective_ratio",
        "test_empty",
        "test_min",
        "test_p01",
        "test_p05",
        "test_median",
        "test_effective_ratio",
    ]
    for key, vals in grouped.items():
        row: dict[str, Any] = {"n_runouts": key[0], "k": key[1], "n_fit_seeds": len(vals)}
        for nk in numeric_keys:
            arr = np.asarray([float(v[nk]) for v in vals], dtype=np.float64)
            row[nk + "_mean"] = float(arr.mean())
            row[nk + "_std"] = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
        row["score"] = row_score(row)
        out[key] = row
    return out


def add_elbow_metrics(agg: dict[tuple[int, int], dict[str, Any]]) -> None:
    by_runouts: dict[int, list[dict[str, Any]]] = {}
    for row in agg.values():
        by_runouts.setdefault(int(row["n_runouts"]), []).append(row)
    for rows in by_runouts.values():
        rows.sort(key=lambda r: int(r["k"]))
        prev = None
        for row in rows:
            if prev is None:
                row["marginal_gain_vs_prev_k"] = None
            else:
                a = float(prev["test_inertia_mean"])
                b = float(row["test_inertia_mean"])
                row["marginal_gain_vs_prev_k"] = (a - b) / a if a > 0 else 0.0
            prev = row


def write_report(path: Path, payload: dict[str, Any]) -> None:
    rows = payload["aggregate"]
    best = payload["recommendation"]
    lines = [
        "# EMD Abstraction Sweep",
        "",
        f"best: K={best['k']} n_runouts={best['n_runouts']} score={best['score']:.6f}",
        "",
        "| K | runouts | test inertia | train inertia | gap | eff buckets | empty | p01 | stability | functional EV | marginal gain | score |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|---:|---:|",
    ]
    for r in sorted(rows, key=lambda x: (int(x["n_runouts"]), int(x["k"]))):
        mg = r.get("marginal_gain_vs_prev_k")
        mg_txt = "n/a" if mg is None else f"{100.0 * float(mg):.2f}%"
        eff = float(r["train_effective_ratio_mean"]) * float(r["k"])
        stab_policy = r.get("assignment_stability_policy") or {}
        stab_txt = f"{r.get('assignment_stability_mean', 0.0):.3f}"
        if stab_policy.get("mode") == "report_only":
            stab_txt += " (report)"
        lines.append(
            f"| {r['k']} | {r['n_runouts']} | {r['test_inertia_mean']:.6f} | "
            f"{r['train_inertia_mean']:.6f} | {100.0 * r['gap_ratio_mean']:.1f}% | "
            f"{eff:.1f} | {r['train_empty_mean']:.1f} | {r['train_p01_mean']:.1f} | "
            f"{stab_txt} | "
            f"{'pass' if r.get('functional_ev_pass') else 'fail'} | "
            f"{mg_txt} | {r['score']:.6f} |"
        )
    lines.extend([
        "",
        "Interpretation:",
        "",
        "- Prefer lower held-out test inertia only while occupancy remains healthy.",
        "- A low marginal gain means the next larger K is buying little held-out quality.",
        "- Empty buckets, tiny p01 occupancy, or a large train/test gap are overfit warnings.",
        "- Stability is label-invariant adjusted Rand index across fit seeds.",
        "- Stability is report-only unless --min-assignment-stability is provided.",
        "- Functional EV must pass the bounded micro-EV tasks from tests/test_emd_quality.py.",
        "- This is a pre-CFR filter. The surviving candidates still need fixed-budget pilot training.",
    ])
    survivors = payload.get("survivors") or []
    if survivors:
        lines.extend(["", "## Survivors (occupancy gates passed)", ""])
        for s in survivors:
            lines.append(
                f"- K={s['k']} runouts={s['n_runouts']} score={s['score']:.6f} "
                f"stability={s.get('assignment_stability_mean', 'n/a')}"
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_centroids(path: Path, *, centroids: np.ndarray, k: int, n_bins: int, n_runouts: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "kind": "emd",
        "num_buckets": int(k),
        "n_bins": int(n_bins),
        "n_runouts": int(n_runouts),
        "centroids": np.asarray(centroids, dtype=np.float64).tolist(),
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--k-grid", type=parse_int_grid, default=parse_int_grid("80,120,160,200"))
    ap.add_argument("--runouts-grid", type=parse_int_grid, default=parse_int_grid("100,200"))
    ap.add_argument("--train-samples", type=int, default=60000)
    ap.add_argument("--test-samples", type=int, default=12000)
    ap.add_argument("--n-bins", type=int, default=20)
    ap.add_argument("--oracle-runouts", type=int, default=800,
                    help="held-out histogram fidelity; use >= max(runouts-grid)")
    ap.add_argument("--kmeans-iters", type=int, default=15)
    ap.add_argument("--fit-seeds", type=parse_int_grid, default=parse_int_grid("0,1,2"))
    ap.add_argument("--functional-spots", type=int, default=400)
    ap.add_argument("--functional-fit-sample", type=int, default=400)
    ap.add_argument(
        "--min-assignment-stability",
        type=float,
        default=None,
        help=(
            "Optional hard ARI cutoff for assignment stability. By default stability is "
            "reported but not used as a kill criterion because the plan defines no numeric cutoff."
        ),
    )
    ap.add_argument("--seed", type=int, default=20260626)
    ap.add_argument("--max-k", type=int, default=200)
    ap.add_argument("--out", default="artifacts/emd_abstraction_sweep.json")
    ap.add_argument("--report", default="artifacts/emd_abstraction_sweep.md")
    ap.add_argument("--write-best", default="",
                    help="optional centroid JSON path for the best candidate")
    args = ap.parse_args()

    if max(args.k_grid) > args.max_k:
        raise SystemExit(
            f"FATAL: requested K={max(args.k_grid)} exceeds --max-k={args.max_k}. "
            "K=500 is banned; sweep K<=200 unless you intentionally edit the cap."
        )
    if args.oracle_runouts < max(args.runouts_grid):
        raise SystemExit("--oracle-runouts should be >= max(--runouts-grid)")
    if args.train_samples < max(args.k_grid):
        raise SystemExit("--train-samples must be >= max K")
    if args.min_assignment_stability is not None and not (-1.0 <= args.min_assignment_stability <= 1.0):
        raise SystemExit("--min-assignment-stability must be in ARI range [-1, 1]")

    bn = load_native()
    t0 = time.time()
    train_spots = sample_spots(args.train_samples, args.seed)
    test_spots = sample_spots(args.test_samples, args.seed ^ 0xBAD5EED)
    print(
        f"spots: train={len(train_spots)} test={len(test_spots)} "
        f"K={args.k_grid} runouts={args.runouts_grid}",
        flush=True,
    )

    test_x = histograms_native(
        bn, test_spots, n_bins=args.n_bins, n_runouts=args.oracle_runouts,
        seed=args.seed ^ 0x0A11CE,
    )

    per_seed_rows: list[dict[str, Any]] = []
    centroids_by_key: dict[tuple[int, int, int], np.ndarray] = {}

    for n_runouts in args.runouts_grid:
        train_x = histograms_native(
            bn, train_spots, n_bins=args.n_bins, n_runouts=n_runouts,
            seed=args.seed ^ (n_runouts * 1009),
        )
        train_list = train_x.tolist()
        for k in args.k_grid:
            for fit_seed in args.fit_seeds:
                print(f"fit K={k} runouts={n_runouts} seed={fit_seed}", flush=True)
                ft0 = time.time()
                centroids = np.asarray(
                    bn.emd_fit(train_list, int(k), int(args.kmeans_iters), int(fit_seed)),
                    dtype=np.float64,
                )
                train_inertia, train_counts = assign_metrics(train_x, centroids)
                test_inertia, test_counts = assign_metrics(test_x, centroids)
                tr_occ = occupancy_summary(train_counts)
                te_occ = occupancy_summary(test_counts)
                gap = (test_inertia - train_inertia) / train_inertia if train_inertia > 0 else 0.0
                row = {
                    "k": int(k),
                    "n_runouts": int(n_runouts),
                    "fit_seed": int(fit_seed),
                    "fit_s": time.time() - ft0,
                    "train_inertia": float(train_inertia),
                    "test_inertia": float(test_inertia),
                    "gap_ratio": float(gap),
                    "train_empty": tr_occ["empty"],
                    "train_min": tr_occ["min"],
                    "train_p01": tr_occ["p01"],
                    "train_p05": tr_occ["p05"],
                    "train_median": tr_occ["median"],
                    "train_effective_ratio": tr_occ["effective_ratio"],
                    "test_empty": te_occ["empty"],
                    "test_min": te_occ["min"],
                    "test_p01": te_occ["p01"],
                    "test_p05": te_occ["p05"],
                    "test_median": te_occ["median"],
                    "test_effective_ratio": te_occ["effective_ratio"],
                }
                per_seed_rows.append(row)
                centroids_by_key[(int(n_runouts), int(k), int(fit_seed))] = centroids
                print(
                    f"  test={test_inertia:.6f} train={train_inertia:.6f} "
                    f"gap={100.0 * gap:.1f}% eff={tr_occ['effective_ratio'] * k:.1f}/{k} "
                    f"empty={tr_occ['empty']:.0f} p01={tr_occ['p01']:.1f}",
                    flush=True,
                )

    agg_map = aggregate(per_seed_rows)
    add_elbow_metrics(agg_map)

    # Assignment stability across fit seeds (same K/runouts).
    stability_by_key: dict[tuple[int, int], dict[str, float]] = {}
    functional_by_key: dict[tuple[int, int], dict[str, Any]] = {}
    for n_runouts in args.runouts_grid:
        for k in args.k_grid:
            cents = [
                centroids_by_key[(int(n_runouts), int(k), int(fs))]
                for fs in args.fit_seeds
                if (int(n_runouts), int(k), int(fs)) in centroids_by_key
            ]
            stability_by_key[(int(n_runouts), int(k))] = assignment_stability(test_x, cents)
            print(f"functional EV K={k} runouts={n_runouts}", flush=True)
            functional_by_key[(int(n_runouts), int(k))] = functional_ev_evidence(
                n_buckets=int(k),
                n_bins=int(args.n_bins),
                n_runouts=int(n_runouts),
                n_spots=int(args.functional_spots),
                fit_sample=int(args.functional_fit_sample),
                seed=int(args.seed ^ (int(n_runouts) * 9176) ^ (int(k) * 131)),
            )

    survivors: list[dict[str, Any]] = []
    for key, row in agg_map.items():
        n_runouts, k = key
        stab = stability_by_key.get(key, {})
        functional = functional_by_key.get(key, {})
        row["assignment_stability_mean"] = stab.get("pairwise_agreement_mean", 1.0)
        row["assignment_stability_min"] = stab.get("pairwise_agreement_min", row["assignment_stability_mean"])
        row["assignment_stability_metric"] = stab.get("metric", "adjusted_rand_index")
        stability_policy = assignment_stability_policy(
            stability_mean=float(row.get("assignment_stability_mean", 0.0)),
            min_assignment_stability=args.min_assignment_stability,
        )
        row["assignment_stability_policy"] = stability_policy
        row["assignment_stability_pass"] = stability_policy["pass"]
        ok, gate_reasons = passes_occupancy_gates(row, train_samples=args.train_samples)
        row["occupancy_pass"] = ok
        row["occupancy_fail_reasons"] = gate_reasons
        row["functional_ev"] = functional
        row["functional_ev_pass"] = bool(functional.get("pass"))
        row["functional_ev_fail_reasons"] = functional.get("fail_reasons") or []
        survivor_reasons = []
        if not ok:
            survivor_reasons.extend(gate_reasons)
        if stability_policy["hard_gate"] and not stability_policy["pass"]:
            survivor_reasons.extend(stability_policy["fail_reasons"])
        if not row["functional_ev_pass"]:
            survivor_reasons.extend(row["functional_ev_fail_reasons"] or ["functional_ev_failed"])
        row["survivor_gate_pass"] = len(survivor_reasons) == 0
        row["survivor_gate_fail_reasons"] = survivor_reasons
        if row["survivor_gate_pass"]:
            survivors.append(dict(row))

    aggregate_rows = sorted(agg_map.values(), key=lambda r: float(r["score"]))
    survivors.sort(key=lambda r: float(r["score"]))
    best = survivors[0] if survivors else aggregate_rows[0]

    best_seed_rows = [
        r for r in per_seed_rows
        if int(r["n_runouts"]) == int(best["n_runouts"]) and int(r["k"]) == int(best["k"])
    ]
    best_seed_row = min(best_seed_rows, key=lambda r: float(r["test_inertia"]))
    best["best_fit_seed"] = int(best_seed_row["fit_seed"])

    payload = {
        "created_s": time.time(),
        "wall_s": time.time() - t0,
        "config": {
            "k_grid": args.k_grid,
            "runouts_grid": args.runouts_grid,
            "train_samples": args.train_samples,
            "test_samples": args.test_samples,
            "n_bins": args.n_bins,
            "oracle_runouts": args.oracle_runouts,
            "kmeans_iters": args.kmeans_iters,
            "fit_seeds": args.fit_seeds,
            "functional_spots": args.functional_spots,
            "functional_fit_sample": args.functional_fit_sample,
            "min_assignment_stability": args.min_assignment_stability,
            "seed": args.seed,
            "max_k": args.max_k,
        },
        "assignment_stability_policy": {
            "mode": "report_only" if args.min_assignment_stability is None else "hard_gate",
            "hard_gate": args.min_assignment_stability is not None,
            "min_assignment_stability": args.min_assignment_stability,
            "threshold_source": (
                "plan_requires_stability_but_no_numeric_cutoff"
                if args.min_assignment_stability is None
                else "cli_min_assignment_stability"
            ),
            "metric": "adjusted_rand_index",
            "artifact_note": (
                "Stability is always reported. It is a hard survivor gate only when "
                "--min-assignment-stability is supplied."
            ),
        },
        "functional_ev_thresholds": {
            "min_gain_bb": -0.01,
            "min_one_task_frac_gap_closed": 0.05,
            "river_peak_mass_min": 0.90,
            "river_hist_variance_max": 0.08,
        },
        "recommendation": best,
        "survivors": survivors,
        "aggregate": aggregate_rows,
        "per_fit": per_seed_rows,
    }

    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_report(ROOT / args.report, payload)

    print(
        f"\nBEST K={best['k']} runouts={best['n_runouts']} "
        f"seed={best['best_fit_seed']} score={best['score']:.6f} "
        f"test_inertia={best['test_inertia_mean']:.6f}",
        flush=True,
    )
    print(f"wrote {out_path}", flush=True)
    print(f"wrote {ROOT / args.report}", flush=True)

    if args.write_best:
        c = centroids_by_key[(int(best["n_runouts"]), int(best["k"]), int(best["best_fit_seed"]))]
        save_centroids(
            ROOT / args.write_best,
            centroids=c,
            k=int(best["k"]),
            n_bins=int(args.n_bins),
            n_runouts=int(best["n_runouts"]),
        )
        print(f"wrote best centroids -> {ROOT / args.write_best}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
