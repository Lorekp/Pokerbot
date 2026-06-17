"""Hogwild (shared-memory, lock-free) parallel CFR — near-linear many-core scaling.

All workers share ONE `SharedHogwildTable`: they read the live regret-matched strategy and accumulate
regret (CFR+) + strategy-sum IN PLACE, with no per-iteration barrier, merge, or reload. Convergence is
preserved despite occasional overwritten updates (the hogwild argument; validated here on Leduc).

This module hosts the game-agnostic primitive (`regret_match`) plus the Leduc validation harness
(`run_leduc_hogwild` — proves the shared table + concurrency converge to near-zero exploitability).
The NLHE external-sampling traversal (H4) reuses the same table + `regret_match`.
"""

from __future__ import annotations

import multiprocessing as mp
from hashlib import blake2b

import numpy as np

from pokerbot.engine.leduc_bridge import (
    apply_action_py,
    py_info_key,
    py_legal,
    py_state_from_deal,
)
from pokerbot.engine.leduc_jax import NUM_ACTIONS as LEDUC_A
from pokerbot.training.leduc_tabular_cfr import all_leduc_deals
from pokerbot.training.shared_hogwild_table import SharedHogwildTable


def regret_match(regret_row: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """CFR regret-matching: σ ∝ max(regret,0) over legal actions; uniform-legal if all ≤ 0."""
    pos = np.maximum(np.asarray(regret_row, dtype=np.float64), 0.0) * mask
    s = pos.sum()
    if s > 0:
        return pos / s
    m = mask.astype(np.float64)
    return m / m.sum()


# ---- Leduc validation harness (H2): EXTERNAL-SAMPLING MCCFR (the NLHE path) ----
# Mirrors the validated sd_cfr_nlhe.traverse: sample chance (the deal) + opponent actions, branch the
# traverser's actions, regret += opp_reach·(util[a]-node_util). Each traversal samples ONE deal, so a
# given infoset's σ is computed once per traversal from the live shared regret — this avoids the
# within-iteration multi-deal σ-mutation that broke the old vanilla loop (task_5cc4c890). Live σ on the
# shared table IS correct for MCCFR (σ tracks the regrets; no per-sweep freeze needed).
_LEDUC_DECK = (0, 0, 1, 1, 2, 2)


def _leduc_key_bytes(tuple_key) -> bytes:
    return blake2b(repr(tuple_key).encode("utf-8"), digest_size=16).digest()


def _sample_leduc_deal(rng: np.random.Generator):
    deck = list(_LEDUC_DECK)
    rng.shuffle(deck)
    return (deck[0], deck[1]), deck[2]


def _es_leduc(table, state, traverser, rng, cfr_plus):
    """Canonical external-sampling MCCFR (Lanctot 2009) on the shared table. At the traverser's
    infosets: branch all actions, regret += v[a]-v_I (no reach weight — sampling the opponent with σ
    gives the unbiased estimate). At opponent infosets: accumulate the average strategy (+= σ) and
    SAMPLE one action. σ = RM+ on the live shared regret (correct for MCCFR — σ tracks the regrets)."""
    if state.done:
        return float(state.rewards[traverser])
    seat = state.acting
    mask = py_legal(state).astype(np.float64)
    slot = table.slot_for(_leduc_key_bytes(py_info_key(state, seat)))
    sigma = regret_match(table.regret_row(slot), mask)

    if seat == traverser:
        util = np.zeros(LEDUC_A, dtype=np.float64)
        for a in range(LEDUC_A):
            if mask[a]:
                util[a] = _es_leduc(table, apply_action_py(state, a), traverser, rng, cfr_plus)
        node_util = float(sum(sigma[a] * util[a] for a in range(LEDUC_A) if mask[a]))
        delta = np.array(
            [(util[a] - node_util) if mask[a] else 0.0 for a in range(LEDUC_A)], dtype=np.float64
        )
        table.add_regret(slot, delta, cfr_plus=cfr_plus)
        return node_util

    # opponent: accumulate avg strategy (+= σ), then sample ONE action ∝ σ (inverse-CDF).
    table.add_strategy(slot, sigma)
    legal_idx = np.flatnonzero(mask)
    probs = sigma[legal_idx]
    probs = probs / max(probs.sum(), 1e-12)
    pick = int(np.searchsorted(np.cumsum(probs), float(rng.random()), side="right"))
    a = int(legal_idx[min(pick, legal_idx.shape[0] - 1)])
    return _es_leduc(table, apply_action_py(state, a), traverser, rng, cfr_plus)


def _leduc_worker(descriptor: dict, locks, traversals: int, seed: int, cfr_plus: bool = True) -> None:
    table = SharedHogwildTable.attach(descriptor, locks)
    rng = np.random.default_rng(seed)
    try:
        for _ in range(traversals):
            cards, pub = _sample_leduc_deal(rng)
            root = py_state_from_deal(cards, pub)
            _es_leduc(table, root, 0, rng, cfr_plus)
            _es_leduc(table, root, 1, rng, cfr_plus)
    finally:
        table.close()


def leduc_avg_from_table(table: SharedHogwildTable) -> dict:
    """Reconstruct {tuple infoset key -> normalized avg strategy} by walking the Leduc tree and
    reading each infoset's strategy-sum row from the shared table (no inverse-hash needed)."""
    avg: dict = {}

    def walk(state):
        if state.done:
            return
        seat = state.acting
        key = py_info_key(state, seat)
        if key not in avg:
            slot = table.slot_for(_leduc_key_bytes(key))
            row = table.strat[slot].astype(np.float64)
            avg[key] = row.copy()
        mask = py_legal(state)
        for a in range(LEDUC_A):
            if mask[a]:
                walk(apply_action_py(state, a))

    for cards, pub in all_leduc_deals():
        walk(py_state_from_deal(cards, pub))
    return avg


def run_leduc_hogwild(traversals: int = 40000, num_workers: int = 4, capacity: int = 4096,
                      cfr_plus: bool = True, seed: int = 0) -> dict:
    """Run hogwild external-sampling MCCFR on Leduc across ``num_workers`` processes sharing one table
    (``traversals`` per worker). Returns the reconstructed average strategy (→
    ``infoset_exploitability_mbb``). More workers = more concurrent traversals into one table."""
    ctx = mp.get_context("spawn")
    table = SharedHogwildTable.create(capacity, LEDUC_A, ctx=ctx)
    try:
        if num_workers <= 1:
            _leduc_worker(table.descriptor(), table.locks, traversals, seed, cfr_plus)
        else:
            procs = [
                ctx.Process(
                    target=_leduc_worker,
                    args=(table.descriptor(), table.locks, traversals, seed + 1000 * w, cfr_plus),
                )
                for w in range(num_workers)
            ]
            for p in procs:
                p.start()
            for p in procs:
                p.join(timeout=600)
                if p.exitcode != 0:
                    raise RuntimeError(f"hogwild leduc worker failed (exitcode {p.exitcode})")
        return leduc_avg_from_table(table)
    finally:
        table.close()
        table.unlink()
