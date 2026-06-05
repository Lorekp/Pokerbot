"""Validate one bucket row of the trio tensor (live MC vs exact oracle).

FREQUENCY WEIGHTING IS VALIDATION-ONLY — never used in tensor builder or FP/BR reads.
"""

from __future__ import annotations

import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass

import numpy as np

from aof_cfr.buckets import LABELS_169
from aof_cfr.cpu_pool import pin_compute_threads
from aof_cfr.trio_payoff_tensor import (
    DEFAULT_MC_BOARDS,
    N_BUCKETS,
    _avg_shares_for_triple,
    _use_native_trio,
    exact_shares_for_triple_oracle,
    exact_shares_oracle_for_triple,
    trio_deal_weights,
)

MAX_EQUITY_ERROR = 0.005
MAX_WEIGHTED_EQUITY_ERROR = 0.005
DEFAULT_GATE_MARGIN = 0.0005
UCB_Z_95 = 1.645


@dataclass(frozen=True)
class TripleCheck:
    row: int
    col_j: int
    col_k: int
    label: str
    max_error: float
    mc: np.ndarray
    oracle: np.ndarray
    combo_count: int
    deal_weight: float = 0.0


def _triple_label(i: int, j: int, k: int) -> str:
    return f"{LABELS_169[i]} / {LABELS_169[j]} / {LABELS_169[k]}"


def _pack_mc(row: int, j: int, k: int, mc_boards: int) -> tuple[np.ndarray, int]:
    s0, s1, s2, bb, bs, sb, cnt = _avg_shares_for_triple(row, j, k, mc_boards=mc_boards)
    mc = np.array([s0, s1, s2, bb, bs, sb], dtype=np.float64)
    return mc, cnt


def _pack_oracle(row: int, j: int, k: int) -> np.ndarray:
    s0, s1, s2, bb, bs, sb, _ = exact_shares_for_triple_oracle(row, j, k)
    return np.array([s0, s1, s2, bb, bs, sb], dtype=np.float64)


def _pack_oracle_shares(row: int, j: int, k: int) -> np.ndarray:
    s0, s1, s2, _ = exact_shares_oracle_for_triple(row, j, k)
    return np.array([s0, s1, s2], dtype=np.float64)


def weighted_mean_error(checks: list[TripleCheck]) -> float:
    if not checks:
        return 0.0
    wsum = sum(c.deal_weight for c in checks)
    if wsum <= 0.0:
        return float(max(c.max_error for c in checks))
    return float(sum(c.deal_weight * c.max_error for c in checks) / wsum)


def unweighted_max_error(checks: list[TripleCheck]) -> float:
    return float(max((c.max_error for c in checks), default=0.0))


def validation_gate_passes(checks: list[TripleCheck]) -> tuple[bool, float, float]:
    """Pass/fail uses frequency-weighted error only."""
    werr = weighted_mean_error(checks)
    umax = unweighted_max_error(checks)
    return werr <= MAX_WEIGHTED_EQUITY_ERROR, werr, umax


@dataclass(frozen=True)
class PooledGateResult:
    n_triples: int
    n_triples_deduped: int
    weighted_mean: float
    effective_n: float
    stderr: float
    upper_bound_95: float
    unweighted_max: float
    worst_spot_label: str
    pass_threshold: bool
    pass_margin: bool

    def to_dict(self) -> dict:
        return {
            "n_triples": self.n_triples,
            "n_triples_deduped": self.n_triples_deduped,
            "weighted_mean": self.weighted_mean,
            "effective_n": self.effective_n,
            "stderr": self.stderr,
            "upper_bound_95": self.upper_bound_95,
            "unweighted_max": self.unweighted_max,
            "worst_spot_label": self.worst_spot_label,
            "pass_threshold": self.pass_threshold,
            "pass_margin": self.pass_margin,
        }


def dedupe_triple_checks(checks: list[TripleCheck]) -> list[TripleCheck]:
    """Keep one check per (row, col_j, col_k); nested sample sizes share RNG draws."""
    by_key: dict[tuple[int, int, int], TripleCheck] = {}
    for c in checks:
        by_key[(c.row, c.col_j, c.col_k)] = c
    return list(by_key.values())


def serialize_triple_check(c: TripleCheck) -> dict:
    return {
        "row": c.row,
        "col_j": c.col_j,
        "col_k": c.col_k,
        "label": c.label,
        "max_error": c.max_error,
        "deal_weight": c.deal_weight,
        "combo_count": c.combo_count,
    }


def compact_checks(checks: list[TripleCheck]) -> list[dict]:
    return [serialize_triple_check(c) for c in checks]


def triple_check_from_dict(d: dict) -> TripleCheck:
    row = int(d["row"])
    j = int(d["col_j"])
    k = int(d["col_k"])
    return TripleCheck(
        row=row,
        col_j=j,
        col_k=k,
        label=str(d.get("label", _triple_label(row, j, k))),
        max_error=float(d["max_error"]),
        mc=np.array([], dtype=np.float64),
        oracle=np.array([], dtype=np.float64),
        combo_count=int(d.get("combo_count", 0)),
        deal_weight=float(d["deal_weight"]),
    )


def pooled_gate_result(
    checks: list[TripleCheck],
    *,
    threshold: float = MAX_WEIGHTED_EQUITY_ERROR,
    margin: float = DEFAULT_GATE_MARGIN,
    dedupe: bool = True,
) -> PooledGateResult:
    """Deal-weighted pooled error with one-sided 95% upper confidence bound."""
    pool = dedupe_triple_checks(checks) if dedupe else list(checks)
    if not pool:
        return PooledGateResult(
            n_triples=len(checks),
            n_triples_deduped=0,
            weighted_mean=0.0,
            effective_n=0.0,
            stderr=0.0,
            upper_bound_95=0.0,
            unweighted_max=0.0,
            worst_spot_label="",
            pass_threshold=True,
            pass_margin=True,
        )

    wsum = sum(c.deal_weight for c in pool)
    if wsum <= 0.0:
        mu = float(max(c.max_error for c in pool))
        umax = mu
        stderr = 0.0
        n_eff = float(len(pool))
    else:
        mu = float(sum(c.deal_weight * c.max_error for c in pool) / wsum)
        umax = unweighted_max_error(pool)
        w_sq = sum(c.deal_weight * c.deal_weight for c in pool)
        n_eff = (wsum * wsum / w_sq) if w_sq > 0.0 else float(len(pool))
        var_num = sum(c.deal_weight * (c.max_error - mu) ** 2 for c in pool)
        stderr = float(np.sqrt(var_num / wsum / n_eff)) if n_eff > 0.0 else 0.0

    ucb = mu + UCB_Z_95 * stderr
    worst = max(pool, key=lambda c: c.max_error)
    margin_cutoff = threshold - margin
    return PooledGateResult(
        n_triples=len(checks),
        n_triples_deduped=len(pool),
        weighted_mean=mu,
        effective_n=float(n_eff),
        stderr=stderr,
        upper_bound_95=ucb,
        unweighted_max=umax,
        worst_spot_label=worst.label,
        pass_threshold=ucb <= threshold,
        pass_margin=mu <= margin_cutoff,
    )


def check_triple_live(
    row: int, j: int, k: int, *, mc_boards: int = DEFAULT_MC_BOARDS
) -> TripleCheck:
    mc, cnt = _pack_mc(row, j, k, mc_boards)
    oracle = _pack_oracle(row, j, k)
    err = float(np.max(np.abs(mc - oracle)))
    deal_w = trio_deal_weights()
    return TripleCheck(
        row=row,
        col_j=j,
        col_k=k,
        label=_triple_label(row, j, k),
        max_error=err,
        mc=mc,
        oracle=oracle,
        combo_count=cnt,
        deal_weight=float(deal_w[row, j, k]),
    )


def sample_triple_indices(
    *,
    samples_per_row: int,
    seed: int,
    n: int = N_BUCKETS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized RNG sample: one (row,j,k) per row x samples_per_row."""
    rng = np.random.default_rng(seed)
    rows = np.repeat(np.arange(n, dtype=np.int32), samples_per_row)
    js = rng.integers(0, n, size=rows.size, dtype=np.int32)
    ks = rng.integers(0, n, size=rows.size, dtype=np.int32)
    return rows, js, ks


def batch_check_triple_tensor_shares(
    shares: np.ndarray,
    pair_eq: np.ndarray,
    combo_counts: np.ndarray,
    rows: np.ndarray,
    js: np.ndarray,
    ks: np.ndarray,
    *,
    omp_threads: int = 0,
    deal_weights: np.ndarray | None = None,
) -> list[TripleCheck]:
    """Vectorized tensor vs exact-oracle gate samples (native OpenMP batch when available)."""
    rows = np.asarray(rows, dtype=np.int32)
    js = np.asarray(js, dtype=np.int32)
    ks = np.asarray(ks, dtype=np.int32)
    got = np.asarray(shares[rows, js, ks, :], dtype=np.float64)
    if _use_native_trio():
        from tools.ompeval_cli import native as omp_native

        oracle, oracle_cnt = omp_native.batch_exact_shares_for_triples(rows, js, ks, threads=omp_threads)
        oracle = np.asarray(oracle, dtype=np.float64)
        oracle_cnt = np.asarray(oracle_cnt, dtype=np.int32)
    else:
        oracle = np.empty((rows.size, 3), dtype=np.float64)
        oracle_cnt = np.empty(rows.size, dtype=np.int32)
        for t in range(rows.size):
            s0, s1, s2, cnt = exact_shares_oracle_for_triple(int(rows[t]), int(js[t]), int(ks[t]))
            oracle[t] = (s0, s1, s2)
            oracle_cnt[t] = cnt

    errs = np.max(np.abs(got - oracle), axis=1)
    if deal_weights is None:
        deal_weights = trio_deal_weights()
    dw = np.asarray(deal_weights[rows, js, ks], dtype=np.float64)
    cnts = np.asarray(combo_counts[rows, js, ks], dtype=np.int32)
    use_cnt = np.where(cnts > 0, cnts, oracle_cnt)

    checks: list[TripleCheck] = []
    empty = np.array([], dtype=np.float64)
    for t in range(rows.size):
        r = int(rows[t])
        j = int(js[t])
        k = int(ks[t])
        checks.append(
            TripleCheck(
                row=r,
                col_j=j,
                col_k=k,
                label=_triple_label(r, j, k),
                max_error=float(errs[t]),
                mc=empty,
                oracle=empty,
                combo_count=int(use_cnt[t]),
                deal_weight=float(dw[t]),
            )
        )
    return checks


def check_triple_tensor(
    i: int,
    j: int,
    k: int,
    shares: np.ndarray,
    pair_eq: np.ndarray,
    combo_counts: np.ndarray,
    *,
    share_only: bool = True,
) -> TripleCheck:
    got_sh = np.array(
        [shares[i, j, k, 0], shares[i, j, k, 1], shares[i, j, k, 2]],
        dtype=np.float64,
    )
    if share_only:
        oracle_sh = _pack_oracle_shares(i, j, k)
        err = float(np.max(np.abs(got_sh - oracle_sh)))
        got = np.concatenate([got_sh, pair_eq[i, j, k]])
        oracle = np.concatenate([oracle_sh, pair_eq[i, j, k]])
    else:
        got = np.array(
            [
                shares[i, j, k, 0],
                shares[i, j, k, 1],
                shares[i, j, k, 2],
                pair_eq[i, j, k, 0],
                pair_eq[i, j, k, 1],
                pair_eq[i, j, k, 2],
            ],
            dtype=np.float64,
        )
        oracle = _pack_oracle(i, j, k)
        err = float(np.max(np.abs(got - oracle)))
    deal_w = trio_deal_weights()
    return TripleCheck(
        row=i,
        col_j=j,
        col_k=k,
        label=_triple_label(i, j, k),
        max_error=err,
        mc=got,
        oracle=oracle,
        combo_count=int(combo_counts[i, j, k]),
        deal_weight=float(deal_w[i, j, k]),
    )


def check_triple_saved(
    row: int,
    j: int,
    k: int,
    shares: np.ndarray,
    pair_eq: np.ndarray,
    combo_counts: np.ndarray,
    *,
    share_only: bool = True,
) -> TripleCheck:
    got_sh = np.array([shares[j, k, 0], shares[j, k, 1], shares[j, k, 2]], dtype=np.float64)
    if share_only:
        oracle_sh = _pack_oracle_shares(row, j, k)
        err = float(np.max(np.abs(got_sh - oracle_sh)))
        got = np.concatenate([got_sh, pair_eq[j, k]])
        oracle = np.concatenate([oracle_sh, pair_eq[j, k]])
    else:
        got = np.array(
            [
                shares[j, k, 0],
                shares[j, k, 1],
                shares[j, k, 2],
                pair_eq[j, k, 0],
                pair_eq[j, k, 1],
                pair_eq[j, k, 2],
            ],
            dtype=np.float64,
        )
        oracle = _pack_oracle(row, j, k)
        err = float(np.max(np.abs(got - oracle)))
    deal_w = trio_deal_weights()
    return TripleCheck(
        row=row,
        col_j=j,
        col_k=k,
        label=_triple_label(row, j, k),
        max_error=err,
        mc=got,
        oracle=oracle,
        combo_count=int(combo_counts[j, k]),
        deal_weight=float(deal_w[row, j, k]),
    )


def _validate_worker_init() -> None:
    pin_compute_threads(1)


def _check_saved_task(args: tuple) -> TripleCheck:
    row, j, k, share_only = args[:4]
    shares, pair_eq, combo_counts = args[4]
    return check_triple_saved(row, j, k, shares, pair_eq, combo_counts, share_only=share_only)


def validate_row_samples(
    row: int,
    *,
    samples: int = 15,
    seed: int = 0,
    mc_boards: int = DEFAULT_MC_BOARDS,
    shares: np.ndarray | None = None,
    pair_eq: np.ndarray | None = None,
    combo_counts: np.ndarray | None = None,
    fixed_triples: list[tuple[int, int]] | None = None,
    share_only: bool = True,
    parallel: int = 0,
) -> tuple[list[TripleCheck], bool, float, float]:
    """Spot-check row ``row``; gate on weighted error only."""
    n = len(LABELS_169)
    rng = random.Random(seed)
    if fixed_triples is None:
        fixed_triples = [(rng.randint(0, n - 1), rng.randint(0, n - 1)) for _ in range(samples)]
    checks: list[TripleCheck] = []

    if shares is not None and pair_eq is not None and combo_counts is not None:
        workers = parallel if parallel > 0 else 0
        if workers > 1 and len(fixed_triples) > 1:
            tasks = [(row, j, k, share_only, (shares, pair_eq, combo_counts)) for j, k in fixed_triples]
            with ProcessPoolExecutor(max_workers=workers, initializer=_validate_worker_init) as pool:
                futs = [pool.submit(_check_saved_task, t) for t in tasks]
                for fut in as_completed(futs):
                    checks.append(fut.result())
        else:
            for j, k in fixed_triples:
                checks.append(
                    check_triple_saved(
                        row, j, k, shares, pair_eq, combo_counts, share_only=share_only
                    )
                )
    else:
        for j, k in fixed_triples:
            checks.append(check_triple_live(row, j, k, mc_boards=mc_boards))

    ok, werr, umax = validation_gate_passes(checks)
    return checks, ok, werr, umax


def row_slice_sanity(
    shares: np.ndarray, pair_eq: np.ndarray, combo_counts: np.ndarray
) -> dict[str, float]:
    valid = combo_counts > 0
    share_sums = shares[valid].sum(axis=-1)
    return {
        "valid_triples": float(valid.sum()),
        "invalid_triples": float((~valid).sum()),
        "max_share_sum_deviation": float(np.max(np.abs(share_sums - 1.0))) if valid.any() else 0.0,
        "min_combo_count": float(combo_counts[valid].min()) if valid.any() else 0.0,
        "max_combo_count": float(combo_counts.max()),
        "mean_combo_count": float(combo_counts[valid].mean()) if valid.any() else 0.0,
    }
