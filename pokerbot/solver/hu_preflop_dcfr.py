"""HU preflop push/fold tabular CFR+ at fixed stack depth (AoF)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np

from aof_equity import equity_hu_exact, equity_vs_range

FOLD = 0
AGGRESS = 1  # push (SB) or call (BB)

RANKS = "23456789TJQKA"


def _build_169_buckets() -> tuple[list[str], list[list[tuple[str, str]]]]:
    """Return 169 bucket labels and representative combos per bucket."""
    labels: list[str] = []
    combos: list[list[tuple[str, str]]] = []
    for r in RANKS:
        labels.append(f"{r}{r}")
        combos.append(
            [(f"{r}c", f"{r}d"), (f"{r}h", f"{r}s"), (f"{r}c", f"{r}h"), (f"{r}d", f"{r}s"), (f"{r}h", f"{r}c"), (f"{r}d", f"{r}h")]
        )
    for i, r1 in enumerate(RANKS):
        for r2 in RANKS[:i]:
            for suited in (True, False):
                lab = f"{r1}{r2}{'s' if suited else 'o'}"
                labels.append(lab)
                if suited:
                    combos.append([(f"{r1}{s}", f"{r2}{s}") for s in "cdhs"])
                else:
                    combos.append(
                        [
                            (f"{r1}c", f"{r2}d"),
                            (f"{r1}h", f"{r2}s"),
                            (f"{r1}d", f"{r2}c"),
                            (f"{r1}s", f"{r2}h"),
                        ]
                    )
    assert len(labels) == 169, len(labels)
    return labels, combos


def _combo_key(cards: tuple[str, str]) -> str:
    return "".join(cards)


def _bucket_eq(
    combos_a: list[tuple[str, str]],
    label_b: str,
    samples: int,
    seed: int,
) -> float:
    """Hero combo vs villain bucket using fast equity_vs_range path."""
    hero = _combo_key(combos_a[0])
    return equity_vs_range(hero, label_b, samples, seed, max_combos=48)


def _build_eq_matrix(
    labels: list[str],
    combos: list[list[tuple[str, str]]],
    *,
    samples: int,
    seed: int,
) -> np.ndarray:
    n = len(labels)
    mat = np.zeros((n, n), dtype=np.float64)
    sub = min(max(samples, 400), 1200)
    for i in range(n):
        mat[i, i] = 0.5
        for j in range(i + 1, n):
            e = _bucket_eq(combos[i], labels[j], sub, seed + i * 169 + j)
            mat[i, j] = e
            mat[j, i] = 1.0 - e
    return mat


@dataclass
class HuSolveResult:
    stack_bb: float
    iterations: int
    exploitability_bb: float
    sb_push_freq: np.ndarray  # (169,)
    bb_call_freq: np.ndarray  # (169,)
    bucket_labels: list[str]
    call_range_str: str
    call_pct: float
    shove_range_str: str
    manifest: dict = field(default_factory=dict)


def _regret_matching(regrets: np.ndarray) -> np.ndarray:
    pos = np.maximum(regrets, 0.0)
    s = pos.sum(axis=-1, keepdims=True)
    uniform = np.full_like(pos, 0.5)
    return np.where(s > 1e-12, pos / np.maximum(s, 1e-12), uniform)


def solve_hu_pushfold(
    *,
    stack_bb: float = 8.0,
    iterations: int = 2000,
    seed: int = 42,
    equity_samples: int = 800,
    sb_posted: float = 0.5,
    bb_posted: float = 1.0,
    bucket_stride: int = 1,
) -> HuSolveResult:
    labels_full, combos_full = _build_169_buckets()
    if bucket_stride > 1:
        idx = list(range(0, len(labels_full), bucket_stride))
        labels = [labels_full[i] for i in idx]
        combos = [combos_full[i] for i in idx]
    else:
        labels, combos = labels_full, combos_full
    n = len(labels)
    eq = _build_eq_matrix(labels, combos, samples=equity_samples, seed=seed)
    pot_called = 1.5 + 2.0 * stack_bb

    sb_regret = np.zeros((n, 2), dtype=np.float64)
    bb_regret = np.zeros((n, 2), dtype=np.float64)
    sb_strat_sum = np.zeros((n, 2), dtype=np.float64)
    bb_strat_sum = np.zeros((n, 2), dtype=np.float64)

    # Uniform deal weight over ordered hands (simplified: uniform buckets)
    deal_w = np.ones((n, n), dtype=np.float64) / (n * n)

    for _ in range(iterations):
        sb_strat = _regret_matching(sb_regret)
        bb_strat = _regret_matching(bb_regret)

        sb_util = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            for j in range(n):
                u_fold = -sb_posted
                u_push_vs_fold = 1.0
                u_push_vs_call = eq[i, j] * pot_called - stack_bb
                sb_util[i, j] = (
                    sb_strat[i, FOLD] * u_fold
                    + sb_strat[i, AGGRESS]
                    * (bb_strat[j, FOLD] * u_push_vs_fold + bb_strat[j, AGGRESS] * u_push_vs_call)
                )

        # SB counterfactual regrets
        for i in range(n):
            for j in range(n):
                w = deal_w[i, j]
                u_fold = -sb_posted
                u_push = bb_strat[j, FOLD] * 1.0 + bb_strat[j, AGGRESS] * (
                    eq[i, j] * pot_called - stack_bb
                )
                node_u = sb_strat[i, FOLD] * u_fold + sb_strat[i, AGGRESS] * u_push
                sb_regret[i, FOLD] += w * (u_fold - node_u)
                sb_regret[i, AGGRESS] += w * (u_push - node_u)

        # BB counterfactual regrets (after SB push)
        for i in range(n):
            for j in range(n):
                w = deal_w[i, j] * sb_strat[i, AGGRESS]
                if w < 1e-15:
                    continue
                u_bb_fold = -bb_posted
                u_bb_call = (1.0 - eq[i, j]) * pot_called - stack_bb
                node_u = bb_strat[j, FOLD] * u_bb_fold + bb_strat[j, AGGRESS] * u_bb_call
                bb_regret[j, FOLD] += w * (u_bb_fold - node_u)
                bb_regret[j, AGGRESS] += w * (u_bb_call - node_u)

        sb_regret = np.maximum(sb_regret, 0.0)
        bb_regret = np.maximum(bb_regret, 0.0)
        sb_strat_sum += sb_strat
        bb_strat_sum += bb_strat

    avg_sb = sb_strat_sum / max(iterations, 1)
    avg_bb = bb_strat_sum / max(iterations, 1)
    sb_push = avg_sb[:, AGGRESS]
    bb_call = avg_bb[:, AGGRESS]

    exploit = _exploitability(
        eq, sb_push, bb_call, stack_bb, pot_called, sb_posted, bb_posted, deal_w
    )

    call_rng, call_pct = _freqs_to_range_string(labels, bb_call, combos)
    shove_rng, _ = _freqs_to_range_string(labels, sb_push, combos, min_freq=0.5)

    manifest = {
        "stack_bb": stack_bb,
        "iterations": iterations,
        "exploitability_bb": exploit,
        "seed": seed,
        "equity_samples": equity_samples,
        "n_buckets": n,
        "bucket_stride": bucket_stride,
    }

    return HuSolveResult(
        stack_bb=stack_bb,
        iterations=iterations,
        exploitability_bb=exploit,
        sb_push_freq=sb_push,
        bb_call_freq=bb_call,
        bucket_labels=labels,
        call_range_str=call_rng,
        call_pct=call_pct,
        shove_range_str=shove_rng,
        manifest=manifest,
    )


def _exploitability(
    eq: np.ndarray,
    sb_push: np.ndarray,
    bb_call: np.ndarray,
    stack_bb: float,
    pot_called: float,
    sb_posted: float,
    bb_posted: float,
    deal_w: np.ndarray,
) -> float:
    """Best-response gap (sum of players' BR improvements) in bb."""
    n = eq.shape[0]
    inv_n = 1.0 / n

    def sb_push_ev(i: int) -> float:
        return sum(
            inv_n * ((1.0 - bb_call[j]) * 1.0 + bb_call[j] * (eq[i, j] * pot_called - stack_bb))
            for j in range(n)
        )

    def sb_value() -> float:
        return sum(
            inv_n * ((1.0 - sb_push[i]) * (-sb_posted) + sb_push[i] * sb_push_ev(i))
            for i in range(n)
        )

    def sb_br() -> float:
        return sum(inv_n * max(-sb_posted, sb_push_ev(i)) for i in range(n))

    def bb_value() -> float:
        v = 0.0
        for i in range(n):
            for j in range(n):
                w = deal_w[i, j] * sb_push[i]
                u = (1.0 - bb_call[j]) * (-bb_posted) + bb_call[j] * (
                    (1.0 - eq[i, j]) * pot_called - stack_bb
                )
                v += w * u
        return v

    def bb_br() -> float:
        v = 0.0
        for j in range(n):
            u_fold = -bb_posted
            u_call = 0.0
            for i in range(n):
                p_ij = deal_w[i, j] * sb_push[i]
                if p_ij < 1e-15:
                    continue
                u_call += p_ij * ((1.0 - eq[i, j]) * pot_called - stack_bb)
            # Weight by marginal P(BB hand j | SB pushed)
            p_j = sum(deal_w[i, j] * sb_push[i] for i in range(n))
            if p_j < 1e-15:
                continue
            v += max(u_fold * p_j, u_call)
        return v

    return float(max((sb_br() - sb_value()) + (bb_br() - bb_value()), 0.0))


def _freqs_to_range_string(
    labels: list[str],
    freqs: np.ndarray,
    combos: list[list[tuple[str, str]]],
    *,
    min_freq: float = 0.45,
) -> tuple[str, float]:
    """Build pokerkit range string from bucket frequencies (push/call threshold)."""
    total_mass = sum(len(combos[i]) for i in range(len(labels)))
    called_mass = sum(len(combos[i]) * float(freqs[i]) for i in range(len(labels)))
    pct = called_mass / max(total_mass, 1)
    order = sorted(range(len(labels)), key=lambda i: -freqs[i])
    parts = [labels[i] for i in order if freqs[i] >= min_freq]
    if not parts:
        parts = [labels[order[0]]]
    # Compact: if many pairs in a row, use XX+ notation heuristically
    rng = ",".join(parts[:18])
    return rng, float(pct)


def expand_to_169(
    coarse_labels: list[str],
    coarse_freq: np.ndarray,
) -> tuple[list[str], np.ndarray]:
    """Map coarse CFR frequencies onto full 169 buckets by label match / nearest rank."""
    full_labels, _ = _build_169_buckets()
    full_freq = np.zeros(len(full_labels), dtype=np.float64)
    coarse_map = {lab: float(coarse_freq[i]) for i, lab in enumerate(coarse_labels)}
    for i, lab in enumerate(full_labels):
        if lab in coarse_map:
            full_freq[i] = coarse_map[lab]
        else:
            # nearest coarse label by first char / suited
            best = coarse_labels[0]
            full_freq[i] = coarse_map.get(best, 0.0)
            for cl in coarse_labels:
                if cl[0] == lab[0] and cl[-1] == lab[-1]:
                    best = cl
                    break
            full_freq[i] = coarse_map.get(best, 0.0)
    return full_labels, full_freq


def save_hu_solution(path: Path, result: HuSolveResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "manifest": result.manifest,
        "call_range_str": result.call_range_str,
        "call_pct": result.call_pct,
        "shove_range_str": result.shove_range_str,
        "sb_push_freq": result.sb_push_freq.tolist(),
        "bb_call_freq": result.bb_call_freq.tolist(),
        "bucket_labels": result.bucket_labels,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_hu_solution(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))
