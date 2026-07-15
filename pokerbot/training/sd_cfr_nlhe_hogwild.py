"""Hogwild (shared-memory, lock-free) external-sampling MCCFR on the production NLHE abstraction.

This is the NLHE sibling of ``hogwild_cfr._es_leduc`` (validated on Leduc, H1-H3). All workers share one
``SharedHogwildTable`` keyed by the 16-byte NLHE infoset fingerprint (``encode_nlhe_bundle_fast(...)
.fingerprint()`` — the SAME key space the SD-CFR trainer uses for ``_cum_regret``/``_strat_sum``), and
accumulate regret (CFR+) + average-strategy IN PLACE with no per-iteration barrier, merge, or reload.

The traversal is TEXTBOOK external-sampling MCCFR (Lanctot 2009), exactly as the proven Leduc path:
  * traverser node: branch all (abstracted-)legal actions; ``regret += util[a] - node_util`` (NO reach
    weight — sampling the opponent with σ already gives the unbiased estimate).
  * opponent node: accumulate the average strategy (``+= σ``), reservoir-capture ``(key, bundle, mask)``
    for later net distillation (the table stores only the 16-byte key, not the encoder inputs), and
    SAMPLE one action ∝ σ (inverse-CDF).
σ = regret-matching+ on the live shared regret (correct for MCCFR — σ tracks the regrets).

Chance is the deck: ``init_hand`` fully shuffles the deck up front, so ``sample_chance`` is deterministic
given the deal — one ``init_hand`` IS the chance sample, traversed once per live seat (mirrors Leduc's
deal → traverse seat 0 → traverse seat 1).

The fingerprint mode (bounded Pluribus-style vs exact/lossless) is controlled by
``POKERBOT_BETTING_ABSTRACTION`` read at encoder import — set it BEFORE importing this module (the cloud
launcher / the convergence run set it; the bounded mode is what saturates the infoset count and trains).

Validation (H4, local, small-scale) is on the TABULAR policy directly via ``TabularBlueprintBot`` →
exploitability trend + head-to-head vs the coarse anchor — no net needed. The net distillation
(``distill_strat_to_net`` / ``save_hogwild_nlhe_checkpoint``) is built + fidelity-tested here but only
RUN at scale on the converged cloud table (H6).
"""

from __future__ import annotations

import multiprocessing as mp
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from pokerbot.engine.jax_engine import CHIP_UNIT, NUM_ACTIONS as NLHE_NUM_ACTIONS
from pokerbot.engine.np_engine import (
    apply_action_np,
    is_chance_node_np,
    legal_action_mask_np,
    sample_chance_np,
)
from pokerbot.training.hogwild_cfr import regret_match
from pokerbot.training.shared_hogwild_table import SharedHogwildTable


def _sample_depth_stacks(deal_key):
    """A3 depth-randomized retrain: ONE per-deal effective starting depth ~ log-U(lo,hi)bb, broadcast to
    every seat, drawn from a SUB-key of the deal key so it is deterministic/CRN-reproducible and INDEPENDENT
    of the deck shuffle (which uses ``deal_key`` directly). Returns None when ``POKERBOT_TRAIN_DEPTH_BB`` is
    unset → ``init_hand`` keeps the fixed 100bb default (legacy). Set e.g. ``POKERBOT_TRAIN_DEPTH_BB=15,500``
    (+ POKERBOT_DEPTH_BUCKETS=1 so the depth bucket is keyed). Log-uniform over-weights SHORT stacks, where
    strategy is most depth-sensitive.

    SHARED (symmetric) depth, NOT independent per-seat: the abstraction key buckets on the EFFECTIVE depth
    (min over live seats of (stack+committed)/CHIP_UNIT). The min of 6 INDEPENDENT log-U draws collapses to
    the short end (binding-min ~85% short / ~0% deep), which would starve the deep buckets the retrain
    exists to fill. Broadcasting one depth makes min == the drawn depth, so the keyed variable sweeps the
    full range with clean log-U mass and the table trains genuine deep-vs-deep. The key cannot distinguish
    asymmetric stacks anyway (it only sees the min) and side pots cap each confrontation at the binding
    depth, so symmetric is a faithful, low-risk representation (it is also the legacy equal-stack case)."""
    import os

    raw = os.environ.get("POKERBOT_TRAIN_DEPTH_BB", "").strip()
    if not raw:
        return None
    try:
        lo, hi = (float(x) for x in raw.split(","))
    except (ValueError, TypeError):
        return None
    if not (hi > lo > 0):
        return None
    import math

    import jax
    import jax.numpy as jnp

    from pokerbot.engine.jax_engine import NUM_PLAYERS

    k = jax.random.fold_in(deal_key, 0x5747ACC)  # distinct RNG stream from the deck
    u = jax.random.uniform(k, (), minval=math.log(lo), maxval=math.log(hi))  # ONE depth per deal
    chips = jnp.round(jnp.exp(u) * CHIP_UNIT).astype(jnp.int32)
    return jnp.full((NUM_PLAYERS,), chips, dtype=jnp.int32)


def _sample_occupancy(deal_key):
    """C1 short-handed coverage: per-deal live-seat occupancy (bool[NUM_PLAYERS]) — draw a live-seat COUNT
    n in [lo,hi] weighted toward FULLER tables (the live mode), then a random size-n subset of the 6 seats,
    from a SUB-key of the deal key (deterministic/CRN, a DISTINCT stream from the deck AND the depth
    sampler). Returns None when POKERBOT_TRAIN_OCCUPANCY_MIX is unset -> the trainer keeps the fixed
    --occupancy (byte-identical to today). Set e.g. POKERBOT_TRAIN_OCCUPANCY_MIX=2,6 to also train short
    tables so the blueprint prior covers 5/4/3-handed + heads-up (the engine + search already play any
    count; this only adds the trained warm-start). KEEP IN SYNC with sd_cfr_nlhe_distributed._sample_occupancy."""
    import os

    raw = os.environ.get("POKERBOT_TRAIN_OCCUPANCY_MIX", "").strip()
    if not raw:
        return None
    try:
        lo, hi = (int(x) for x in raw.split(","))
    except (ValueError, TypeError):
        return None
    from pokerbot.engine.jax_engine import NUM_PLAYERS

    if not (2 <= lo <= hi <= NUM_PLAYERS):
        return None
    import jax
    import jax.numpy as jnp

    k = jax.random.fold_in(deal_key, 0x0CC0)  # distinct stream from deck + depth sampler
    kc, ks = jax.random.split(k)
    counts = jnp.arange(lo, hi + 1)
    weights = counts.astype(jnp.float32)  # w(n) proportional to n: fuller tables more likely (live mode)
    n = int(jax.random.choice(kc, counts, p=weights / weights.sum()))
    live = np.asarray(jax.random.choice(ks, NUM_PLAYERS, shape=(n,), replace=False))
    occ = np.zeros(NUM_PLAYERS, dtype=bool)
    occ[live] = True
    return occ


# Default HU solver layout (BTN/BB at seats 3/5 — same as SDCFRNLHEConfig.occupancy).
_DEFAULT_OCCUPANCY: tuple[bool, ...] = (False, False, False, True, False, True)


# --------------------------------------------------------------------------------------------------
# Reservoir of encoder inputs for distillation (the shared table keeps only the 16-byte key).
# --------------------------------------------------------------------------------------------------
class _Reservoir:
    """Reservoir sampler of ``(key16, bundle, mask)`` captured at opponent nodes. Bounded so the
    memory + the cross-worker hand-off stay small; the net is distilled from these (H6)."""

    def __init__(self, cap: int, rng: np.random.Generator) -> None:
        self.cap = int(cap)
        self.rng = rng
        self.items: list[tuple[bytes, Any, np.ndarray]] = []
        self.count = 0

    def add(self, item: tuple[bytes, Any, np.ndarray]) -> None:
        if self.cap <= 0:
            return
        self.count += 1
        if len(self.items) < self.cap:
            self.items.append(item)
            return
        j = int(self.rng.integers(0, self.count))
        if j < self.cap:
            self.items[j] = item


def dcfr_discount_snapshots(
    regret: dict[bytes, np.ndarray],
    strat: dict[bytes, np.ndarray],
    t: float,
    *,
    alpha: float = 1.5,
    beta: float = 0.0,
    gamma: float = 2.0,
) -> None:
    """Apply the Discounted-CFR discount IN PLACE to accumulated regret/strategy snapshot dicts between
    rounds (the per-round reseed path: run_nlhe_hogwild(... cfr_plus=False, init_*=) then discount with
    t=round index). Positive vs negative regret get separate factors; the strategy-sum is down-weighted.
    Pairs with cfr_plus=False (DCFR discounts signed regret instead of flooring it)."""
    from pokerbot.training.shared_hogwild_table import dcfr_weights

    pos, neg, sfac = dcfr_weights(t, alpha, beta, gamma)
    for row in regret.values():
        r = np.asarray(row)
        np.multiply(r, np.where(r > 0.0, pos, neg), out=r)
    for row in strat.values():
        np.asarray(row)[...] *= sfac


def merge_reservoirs(parts: list[list], cap: int, seed: int = 0) -> list:
    """Concatenate per-worker reservoirs and subsample to ``cap`` (uniform without replacement)."""
    pool: list = []
    for p in parts:
        pool.extend(p)
    if cap <= 0 or len(pool) <= cap:
        return pool
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(pool), size=cap, replace=False)
    return [pool[int(i)] for i in idx]


# --------------------------------------------------------------------------------------------------
# Action abstraction (IDENTICAL to sd_cfr_nlhe.build_allowed_action_mask — train/serve must match).
# --------------------------------------------------------------------------------------------------
def _apply_abstraction(
    mask: np.ndarray, street: int, allowed_pre: np.ndarray | None, allowed_post: np.ndarray | None
) -> np.ndarray:
    if allowed_pre is None and allowed_post is None:
        return mask
    allowed = allowed_pre if street == 0 else allowed_post
    if allowed is None:
        return mask
    return mask & allowed


# --------------------------------------------------------------------------------------------------
# Core traversal — textbook external-sampling MCCFR on the shared table (mirrors _es_leduc).
# --------------------------------------------------------------------------------------------------
def _es_nlhe(
    table: SharedHogwildTable,
    state,
    traverser: int,
    rng: np.random.Generator,
    *,
    allowed_pre: np.ndarray | None,
    allowed_post: np.ndarray | None,
    weva_table: np.ndarray,
    reservoir: _Reservoir | None,
    cfr_plus: bool = True,
    use_baseline: bool = False,
    baseline_lr: float = 0.5,
    depth: int = 0,
    max_depth: int = 64,
) -> float:
    # Lazy import: honors POKERBOT_BETTING_ABSTRACTION set before first call (fingerprint mode).
    from pokerbot.engine.jax_weva_fast import encode_nlhe_bundle_fast, fingerprint_fast

    # Depth cap (mirrors sd_cfr_nlhe.max_traverse_depth) — bounds pathological deep raise/all-in lines
    # so a single traversal can't blow up the tree. Returns the running terminal estimate.
    if depth >= max_depth or bool(state.done):
        return float(state.rewards[traverser]) / CHIP_UNIT
    if is_chance_node_np(state):
        state = sample_chance_np(state)
    seat = int(state.action_seat)
    if seat < 0:
        # Terminal/settled (e.g. all-in runout) — value is in rewards once done; else neutral.
        return float(state.rewards[traverser]) / CHIP_UNIT

    mask = legal_action_mask_np(state).astype(bool)
    mask = _apply_abstraction(mask, int(state.street), allowed_pre, allowed_post)
    # Convergence path (reservoir is None) needs ONLY the key → fast fingerprint (skips the unused full
    # card/action tensors). Distillation path (reservoir set) needs the full bundle for the net inputs.
    if reservoir is None:
        bundle = None
        key = fingerprint_fast(state, seat, weva_table=weva_table)
    else:
        bundle = encode_nlhe_bundle_fast(state, seat, weva_table=weva_table)
        key = bundle.fingerprint()
    slot = table.slot_for(key)
    sigma = regret_match(table.regret_row(slot), mask.astype(np.float64))

    if seat == traverser:
        util = np.zeros(NLHE_NUM_ACTIONS, dtype=np.float64)
        for a in range(NLHE_NUM_ACTIONS):
            if mask[a]:
                child = apply_action_np(state, a)
                util[a] = _es_nlhe(
                    table, child, traverser, rng,
                    allowed_pre=allowed_pre, allowed_post=allowed_post,
                    weva_table=weva_table, reservoir=reservoir, cfr_plus=cfr_plus,
                    use_baseline=use_baseline, baseline_lr=baseline_lr,
                    depth=depth + 1, max_depth=max_depth,
                )
        node_util = float(sum(sigma[a] * util[a] for a in range(NLHE_NUM_ACTIONS) if mask[a]))
        delta = np.array(
            [(util[a] - node_util) if mask[a] else 0.0 for a in range(NLHE_NUM_ACTIONS)],
            dtype=np.float64,
        )
        table.add_regret(slot, delta, cfr_plus=cfr_plus)
        return node_util

    # Opponent node: accumulate avg strategy, capture encoder inputs, sample ONE action ∝ σ.
    table.add_strategy(slot, sigma)
    if reservoir is not None:
        reservoir.add((key, bundle, mask.copy()))
    legal_idx = np.flatnonzero(mask)
    probs = sigma[legal_idx]
    probs = probs / max(probs.sum(), 1e-12)
    pick = int(np.searchsorted(np.cumsum(probs), float(rng.random()), side="right"))
    a = int(legal_idx[min(pick, legal_idx.shape[0] - 1)])
    child = apply_action_np(state, a)
    v = _es_nlhe(
        table, child, traverser, rng,
        allowed_pre=allowed_pre, allowed_post=allowed_post,
        weva_table=weva_table, reservoir=reservoir, cfr_plus=cfr_plus,
        use_baseline=use_baseline, baseline_lr=baseline_lr,
        depth=depth + 1, max_depth=max_depth,
    )
    if use_baseline:
        # VR-MCCFR baseline-corrected sampled value (Schmid 2019): est = Σ_a σ(a)·b(I,a) + (v − b(I,a*)).
        # Unbiased for ANY baseline (the Σσ·b term cancels in expectation); variance drops as b → E[v].
        b = table.base_row(slot)
        if b is not None:
            est = float(np.dot(sigma, b)) + (v - float(b[a]))
            b[a] += baseline_lr * (v - float(b[a]))  # racy running-mean update = accepted hogwild noise
            return est
    return v


# Pre-built device int32 action scalars (step_state is jitted on an int32 action — avoid per-edge
# eager convert-element-type, same trick as sd_cfr_nlhe._ACTION_CONST).
_ACTION_CONST_CACHE: dict[int, Any] = {}


def _action_const(a: int):
    c = _ACTION_CONST_CACHE.get(a)
    if c is None:
        import jax.numpy as jnp

        c = jnp.int32(a)
        _ACTION_CONST_CACHE[a] = c
    return c


# --------------------------------------------------------------------------------------------------
# Worker + driver.
# --------------------------------------------------------------------------------------------------
def _live_seats(occupancy: tuple[bool, ...]) -> list[int]:
    return [i for i, s in enumerate(occupancy) if s]


def _nlhe_worker(
    descriptor: dict,
    locks,
    traversals: int,
    seed: int,
    occupancy: tuple[bool, ...],
    client_bet_sizing: bool,
    reservoir_cap: int,
    result_q,
    cfr_plus: bool = True,
    worker_index: int = 0,
    warm_event=None,
    use_baseline: bool = False,
) -> None:
    import jax

    from pokerbot.engine.jax_engine import init_hand
    from pokerbot.engine.np_engine import to_numpy_state
    from pokerbot.engine.jax_weva_fast import default_weva_bucket_table
    from pokerbot.training.sd_cfr_nlhe import build_allowed_action_mask

    table = SharedHogwildTable.attach(descriptor, locks)
    weva_table = default_weva_bucket_table()
    allowed_pre = build_allowed_action_mask((4, 5, 6)) if client_bet_sizing else None
    allowed_post = build_allowed_action_mask((0, 1, 3, 4, 5)) if client_bet_sizing else None
    occ = np.asarray(occupancy, dtype=bool)
    seats = _live_seats(occupancy)
    rng = np.random.default_rng(seed)
    reservoir = _Reservoir(reservoir_cap, np.random.default_rng(seed ^ 0x5DEECE66D)) if reservoir_cap else None
    base_key = jax.random.PRNGKey(seed)
    try:
        for i in range(traversals):
            # WARM GATE (thundering-herd fix, mirrors sd_cfr_nlhe_pool.warm_event): worker 0 does the
            # first traversal alone — cold-compiling the engine JITs and populating
            # JAX_COMPILATION_CACHE_DIR — then releases the other workers, which load from the cache
            # instead of 60-way simultaneous XLA compiles (the round-1 stall). Cheap rounds 2+ since the
            # disk cache persists. timeout so a dead worker 0 can't hang the herd forever.
            if warm_event is not None and worker_index != 0 and i == 0:
                warm_event.wait(timeout=1800)
            k = jax.random.fold_in(base_key, i)
            # C1: per-deal occupancy (short-handed coverage) when POKERBOT_TRAIN_OCCUPANCY_MIX is set; else
            # the fixed --occupancy. The live-seat traversal set must follow the per-deal occupancy.
            occ_k = _sample_occupancy(k)
            occ_i = occ_k if occ_k is not None else occ
            trav_seats = _live_seats(occ_i) if occ_k is not None else seats
            root = to_numpy_state(init_hand(k, occupancy=occ_i, stacks=_sample_depth_stacks(k)))  # JAX init (1/deal), then numpy hot path
            for trav in trav_seats:
                _es_nlhe(
                    table, root, trav, rng,
                    allowed_pre=allowed_pre, allowed_post=allowed_post,
                    weva_table=weva_table, reservoir=reservoir, cfr_plus=cfr_plus,
                    use_baseline=use_baseline,
                )
            if warm_event is not None and worker_index == 0 and i == 0:
                warm_event.set()
    finally:
        # Guarantee the herd is released even if worker 0 errors before setting (else 59 × 900s hang).
        if warm_event is not None and worker_index == 0 and not warm_event.is_set():
            warm_event.set()
        if result_q is not None:
            result_q.put(reservoir.items if reservoir is not None else [])
        table.close()


def _seed_table(
    table: SharedHogwildTable,
    init_strat: dict[bytes, np.ndarray] | None,
    init_regret: dict[bytes, np.ndarray] | None,
) -> None:
    """Pre-populate the freshly-created shared table from prior accumulated snapshots (owner process,
    before workers attach). Workers then ADD on top, so regret continues across rounds (correct CFR)."""
    if not init_strat and not init_regret:
        return
    keys = set(init_strat or {}) | set(init_regret or {})
    for key in keys:
        slot = table.slot_for(key)
        if init_regret is not None and key in init_regret:
            table.regret[slot] = np.asarray(init_regret[key], dtype=np.float64)
        if init_strat is not None and key in init_strat:
            table.strat[slot] = np.asarray(init_strat[key], dtype=np.float64)


def run_nlhe_hogwild(
    traversals: int = 2000,
    num_workers: int = 1,
    capacity: int = 1 << 18,
    *,
    occupancy: tuple[bool, ...] = _DEFAULT_OCCUPANCY,
    client_bet_sizing: bool = False,
    reservoir_cap: int = 0,
    cfr_plus: bool = True,
    seed: int = 0,
    init_strat: dict[bytes, np.ndarray] | None = None,
    init_regret: dict[bytes, np.ndarray] | None = None,
    use_baseline: bool = False,
) -> dict:
    """Run hogwild external-sampling MCCFR on NLHE across ``num_workers`` processes sharing one table
    (``traversals`` deals per worker; each deal traversed once per live seat).

    ``init_strat``/``init_regret`` SEED the shared table before workers spawn (H5 cross-round
    continuation): pass the prior round's accumulated snapshots so regret genuinely CONTINUES — the
    workers add on top, and the returned snapshots are the new accumulated totals. This is correct CFR;
    do NOT instead run fresh-from-zero rounds and sum their average strategies (that converges far
    worse). With both None the run starts cold (H4 behavior).

    Returns ``{"strat": snapshot, "regret": regret_snapshot, "reservoir": [...], "n_infosets": int,
    "load_factor": float}``. ``reservoir`` is populated only when ``reservoir_cap > 0`` (for H6 net
    distillation). The fingerprint mode follows POKERBOT_BETTING_ABSTRACTION at encoder import time.
    """
    # Guard FIRST (fail fast before workers burn compute): never run hogwild blueprint training on the
    # made-hand-only 'fast' bucketer (ignores draw equity). FAIL-CLOSED unless the ack env is set.
    from pokerbot.engine.jax_weva_encode import require_potential_aware_buckets
    require_potential_aware_buckets(context="hogwild NLHE blueprint training")
    ctx = mp.get_context("spawn")
    table = SharedHogwildTable.create(capacity, NLHE_NUM_ACTIONS, ctx=ctx, with_baseline=use_baseline)
    _seed_table(table, init_strat, init_regret)
    result_q = ctx.Queue() if reservoir_cap else None
    warm_event = ctx.Event() if num_workers > 1 else None
    reservoirs: list[list] = []
    try:
        if num_workers <= 1:
            _nlhe_worker(
                table.descriptor(), table.locks, traversals, seed, occupancy,
                client_bet_sizing, reservoir_cap, result_q, cfr_plus,
                use_baseline=use_baseline,
            )
            if result_q is not None:
                reservoirs.append(result_q.get())
        else:
            procs = [
                ctx.Process(
                    target=_nlhe_worker,
                    args=(
                        table.descriptor(), table.locks, traversals, seed + 1000 * w, occupancy,
                        client_bet_sizing, reservoir_cap, result_q, cfr_plus, w, warm_event,
                        use_baseline,
                    ),
                )
                for w in range(num_workers)
            ]
            for p in procs:
                p.start()
            # Drain the queue BEFORE join (a full pipe would deadlock the workers at put()).
            if result_q is not None:
                for _ in procs:
                    reservoirs.append(result_q.get())
            for p in procs:
                p.join(timeout=3600)
                if p.exitcode != 0:
                    raise RuntimeError(f"hogwild NLHE worker failed (exitcode {p.exitcode})")
        return {
            "strat": table.snapshot_strategy(),
            "regret": table.snapshot_regret(),
            "reservoir": merge_reservoirs(reservoirs, reservoir_cap, seed) if reservoir_cap else [],
            "n_infosets": table.n_occupied(),
            "load_factor": table.load_factor(),
        }
    finally:
        table.close()
        table.unlink()


# --------------------------------------------------------------------------------------------------
# PERSISTENT-WORKER session — the production cloud path. Workers + the shared table live for the WHOLE
# box run (no per-round respawn / reseed / re-import / re-compile); the parent snapshots the live table
# on a timer and checkpoints. This removes the per-round fixed-overhead floor (measured ~150s/round at
# 20 deals) → workers run continuously at ~full utilization; reseed happens ONCE per box (on resume).
# --------------------------------------------------------------------------------------------------
def _persistent_worker(
    descriptor: dict,
    locks,
    seed: int,
    occupancy: tuple[bool, ...],
    client_bet_sizing: bool,
    cfr_plus: bool,
    worker_index: int,
    warm_event,
    stop_event,
    counts,
    use_baseline: bool = False,
) -> None:
    import jax

    from pokerbot.engine.jax_engine import init_hand
    from pokerbot.engine.np_engine import to_numpy_state
    from pokerbot.engine.jax_weva_fast import default_weva_bucket_table
    from pokerbot.training.sd_cfr_nlhe import build_allowed_action_mask

    table = SharedHogwildTable.attach(descriptor, locks)
    weva_table = default_weva_bucket_table()
    allowed_pre = build_allowed_action_mask((4, 5, 6)) if client_bet_sizing else None
    allowed_post = build_allowed_action_mask((0, 1, 3, 4, 5)) if client_bet_sizing else None
    occ = np.asarray(occupancy, dtype=bool)
    seats = _live_seats(occupancy)
    rng = np.random.default_rng(seed)
    base_key = jax.random.PRNGKey(seed)
    i = 0
    try:
        while not stop_event.is_set():
            # Warm gate (once per box): worker 0 cold-compiles on traversal 0, then releases the herd
            # (cache-warm). With the S3-restored JAX cache this is a fast cache hit on reclaimed boxes.
            if warm_event is not None and worker_index != 0 and i == 0:
                warm_event.wait(timeout=1800)
            k = jax.random.fold_in(base_key, i)
            # C1: per-deal occupancy (short-handed coverage) when POKERBOT_TRAIN_OCCUPANCY_MIX is set; else
            # the fixed --occupancy. The live-seat traversal set must follow the per-deal occupancy.
            occ_k = _sample_occupancy(k)
            occ_i = occ_k if occ_k is not None else occ
            trav_seats = _live_seats(occ_i) if occ_k is not None else seats
            root = to_numpy_state(init_hand(k, occupancy=occ_i, stacks=_sample_depth_stacks(k)))  # JAX init (1/deal), then numpy hot path
            for trav in trav_seats:
                _es_nlhe(
                    table, root, trav, rng,
                    allowed_pre=allowed_pre, allowed_post=allowed_post,
                    weva_table=weva_table, reservoir=None, cfr_plus=cfr_plus,
                    use_baseline=use_baseline,
                )
            if warm_event is not None and worker_index == 0 and i == 0:
                warm_event.set()
            i += 1
            counts[worker_index] = i  # single-writer cell → lock-free progress counter
    finally:
        if warm_event is not None and worker_index == 0 and not warm_event.is_set():
            warm_event.set()
        table.close()


class HogwildSession:
    """Handle to a running set of persistent hogwild workers sharing one live table. The parent reads
    ``snapshot()`` (a racy live read — accepted hogwild noise) on a timer to checkpoint, then ``stop()``."""

    def __init__(self, table, procs, counts, stop_event, warm_event=None) -> None:
        self.table = table
        self.procs = procs
        self.counts = counts
        self.stop_event = stop_event
        # MUST retain warm_event for the session's lifetime: it is passed to the spawn workers, which
        # unpickle it AFTER p.start() returns. If it were only a local in start_hogwild_session it would
        # be GC'd on return → the resource tracker unlinks its semaphore → workers fail to rebuild it
        # (SemLock._rebuild FileNotFoundError → all workers die). Same reason run_nlhe_hogwild holds it
        # in scope across its join() loop.
        self.warm_event = warm_event

    def total_traversals(self) -> int:
        return int(sum(self.counts[:]))

    def workers_alive(self) -> int:
        return sum(1 for p in self.procs if p.is_alive())

    def snapshot(self) -> dict:
        return {
            "strat": self.table.snapshot_strategy(),
            "regret": self.table.snapshot_regret(),
            "n_infosets": self.table.n_occupied(),
            "load_factor": self.table.load_factor(),
        }

    def stop(self, timeout: float = 120.0) -> None:
        self.stop_event.set()
        for p in self.procs:
            p.join(timeout=timeout)
        for p in self.procs:
            if p.is_alive():
                p.terminate()
        self.table.close()
        self.table.unlink()


def start_hogwild_session(
    capacity: int,
    num_workers: int,
    *,
    occupancy: tuple[bool, ...] = _DEFAULT_OCCUPANCY,
    client_bet_sizing: bool = False,
    cfr_plus: bool = True,
    seed: int = 0,
    init_strat: dict[bytes, np.ndarray] | None = None,
    init_regret: dict[bytes, np.ndarray] | None = None,
    use_baseline: bool = False,
) -> HogwildSession:
    """Create the shared table (seeded ONCE from a prior snapshot for resume), spawn ``num_workers``
    persistent workers that traverse continuously, and return a ``HogwildSession`` for snapshot/stop.

    ``use_baseline`` allocates the VR-MCCFR per-action baseline array and runs the corrected estimator.
    DCFR is driven by the PARENT (``cfr_plus=False`` + periodic ``table.apply_dcfr_discount(t)`` at each
    snapshot tick), so the session itself only needs the baseline flag here."""
    ctx = mp.get_context("spawn")
    table = SharedHogwildTable.create(capacity, NLHE_NUM_ACTIONS, ctx=ctx, with_baseline=use_baseline)
    _seed_table(table, init_strat, init_regret)
    counts = ctx.Array("q", num_workers, lock=False)
    stop_event = ctx.Event()
    warm_event = ctx.Event() if num_workers > 1 else None
    procs = [
        ctx.Process(
            target=_persistent_worker,
            args=(table.descriptor(), table.locks, seed + 1000 * w, occupancy,
                  client_bet_sizing, cfr_plus, w, warm_event, stop_event, counts, use_baseline),
        )
        for w in range(num_workers)
    ]
    for p in procs:
        p.start()
    return HogwildSession(table, procs, counts, stop_event, warm_event)


# --------------------------------------------------------------------------------------------------
# Tabular blueprint bot — the independent Nash validation (no net needed).
# --------------------------------------------------------------------------------------------------
@dataclass
class TabularBlueprintBot:
    """Wraps the converged ``{fingerprint -> strat_sum row}`` snapshot as a GameState-engine bot for
    ``estimate_exploitability`` / ``run_head_to_head``. Looks up the same fingerprint the trainer
    keys on, normalizes over the (abstracted-)legal mask, and plays argmax. Uniform-legal fallback
    for unseen infosets — exactly the trainer's serve-time lookup, just tabular (no net)."""

    snapshot: dict[bytes, np.ndarray]
    client_bet_sizing: bool = False

    def __post_init__(self) -> None:
        from pokerbot.engine.jax_weva_fast import default_weva_bucket_table
        from pokerbot.training.sd_cfr_nlhe import build_allowed_action_mask

        self._weva_table = default_weva_bucket_table()
        self._allowed_pre = build_allowed_action_mask((4, 5, 6)) if self.client_bet_sizing else None
        self._allowed_post = build_allowed_action_mask((0, 1, 3, 4, 5)) if self.client_bet_sizing else None

    def _probs_for_env(self, env, seat: int) -> np.ndarray:
        # Serve path (eval bot): env comes from game_state_to_env (JAX). Use the JAX legal mask here
        # (the numpy hot-path engine is for training traversal); the fingerprint matches either way.
        from pokerbot.engine.jax_engine import legal_action_mask
        from pokerbot.engine.jax_weva_fast import encode_nlhe_bundle_fast
        import jax.numpy as jnp

        mask = np.asarray(legal_action_mask(env, jnp.int32(seat)), dtype=bool)
        mask = _apply_abstraction(mask, int(np.asarray(env.street)), self._allowed_pre, self._allowed_post)
        key = encode_nlhe_bundle_fast(env, seat, weva_table=self._weva_table).fingerprint()
        row = self.snapshot.get(key)
        if row is not None:
            probs = np.asarray(row, dtype=np.float64) * mask
            tot = probs.sum()
            if tot > 1e-12:
                return probs / tot
        m = mask.astype(np.float64)
        return m / max(m.sum(), 1.0)

    def __call__(self, state):
        from pokerbot.search.state_bridge import game_state_to_env
        from pokerbot.training.self_play import _index_to_action

        seat = state.action_seat
        env = game_state_to_env(state, seat)
        probs = self._probs_for_env(env, seat)
        idx = int(np.argmax(probs))
        return _index_to_action(state, seat, idx)


# --------------------------------------------------------------------------------------------------
# Distillation machinery (built + fidelity-tested at H4; run at scale on the converged table at H6).
# --------------------------------------------------------------------------------------------------
def _normalize_target(row: np.ndarray, mask: np.ndarray) -> np.ndarray:
    t = np.maximum(np.asarray(row, dtype=np.float64), 0.0) * mask
    s = t.sum()
    if s > 1e-12:
        return t / s
    m = mask.astype(np.float64)
    return m / max(m.sum(), 1.0)


def distill_strat_to_net(
    snapshot: dict[bytes, np.ndarray],
    reservoir: list,
    config=None,
    *,
    steps: int = 200,
    batch_size: int = 512,
    seed: int = 0,
):
    """Policy-distill the tabular average strategy into the SD-CFR net via masked cross-entropy.

    Targets = normalized ``snapshot`` rows (NOT instantaneous advantages — the shared table has no
    per-visit regret), inputs = the reservoir's encoder bundles. Returns a configured
    ``SDCFRNLHETrainer`` whose ``state.params`` are the distilled net (ready for ``save_nlhe_checkpoint``).
    """
    import jax
    import jax.numpy as jnp
    import optax

    from pokerbot.training.sd_cfr_nlhe import SDCFRNLHEConfig, SDCFRNLHETrainer

    cfg = config or SDCFRNLHEConfig(sft_checkpoint=None)
    trainer = SDCFRNLHETrainer(cfg)

    # Build (bundle, target, mask) training rows from the reservoir keyed to the snapshot.
    rows = []
    for key, bundle, mask in reservoir:
        row = snapshot.get(key)
        if row is None:
            continue
        rows.append((bundle, _normalize_target(row, mask), np.asarray(mask, dtype=bool)))
    if not rows:
        return trainer

    rng = np.random.default_rng(seed)
    NEG = -1e9

    def loss_fn(params, weva, cards, actions, positions, masks, targets):
        variables = {"params": params, "batch_stats": trainer._batch_stats}
        logits = trainer.state.apply_fn(
            variables, weva_ids=weva, cards=cards, actions=actions,
            position=positions, legal_mask=masks,
        )
        masked = jnp.where(masks, logits, NEG)
        logp = jax.nn.log_softmax(masked, axis=-1)
        ce = -jnp.sum(targets * logp * masks, axis=-1)
        return jnp.mean(ce)

    grad_fn = jax.jit(jax.value_and_grad(loss_fn))
    n = len(rows)
    last = 0.0
    for _ in range(steps):
        idx = rng.integers(0, n, size=min(batch_size, n))
        batch = [rows[int(i)] for i in idx]
        weva = jnp.stack([jnp.int32(b[0].weva_id) for b in batch])
        cards = jnp.stack([jnp.asarray(b[0].cards, dtype=jnp.float32).reshape(9, 4, 13) for b in batch])
        actions = jnp.stack([jnp.asarray(b[0].actions, dtype=jnp.float32).reshape(6, 4, 12) for b in batch])
        positions = jnp.stack([jnp.asarray(b[0].policy_position, dtype=jnp.float32) for b in batch])
        masks = jnp.stack([jnp.asarray(b[2], dtype=bool) for b in batch])
        targets = jnp.stack([jnp.asarray(b[1], dtype=jnp.float32) for b in batch])
        loss, grads = grad_fn(trainer.state.params, weva, cards, actions, positions, masks, targets)
        trainer.state = trainer.state.apply_gradients(grads=grads)
        last = float(loss)
    trainer._last_distill_loss = last
    return trainer


def save_hogwild_nlhe_checkpoint(
    result: dict,
    path: str | Path,
    config=None,
    *,
    distill_steps: int = 200,
    seed: int = 0,
) -> None:
    """Distill the converged hogwild ``result`` into the SD-CFR net and write a
    ``save_nlhe_checkpoint``-compatible pickle (round-trips through ``load_nlhe_checkpoint`` and the
    downstream ``scripts/distill_cfr_blueprint.py`` JAX→PyTorch path for the exploitation engine)."""
    from collections import defaultdict

    from pokerbot.training.sd_cfr_nlhe import save_nlhe_checkpoint

    trainer = distill_strat_to_net(
        result["strat"], result.get("reservoir", []), config, steps=distill_steps, seed=seed
    )
    trainer._strat_sum = defaultdict(
        lambda: np.zeros(NLHE_NUM_ACTIONS, dtype=np.float64),
        {k: np.asarray(v, dtype=np.float64) for k, v in result["strat"].items()},
    )
    trainer._cum_regret = {k: np.asarray(v, dtype=np.float64) for k, v in result.get("regret", {}).items()}
    trainer._completed_iterations = 1
    save_nlhe_checkpoint(trainer, Path(path))
