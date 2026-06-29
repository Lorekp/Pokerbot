"""Native (Rust) drop-in for the LBR exploitability harnesses (`pokerbot/eval/lbr6.py` + `lbr.py`).

The Python LBR is a GIL-bound serial loop that pays a Python call PER belief combo PER decision, capping
it at ~1-2 hands/s -- so we can only afford ~1500 hands and the exploitability CI stays too wide to
conclude (the blueprint's +134 bb/100 came back CI [-40, +308]). `tools/battery_native/src/lbr.rs` is a
faithful port of the analytic 1-ply LBR (Lisy & Bowling 2017, multiway-generalised) with the WHOLE hot
loop -- engine, Bayesian belief, showdown equity, fingerprint serve -- native and fanned out across rayon
workers (one independent hand per task). That is the ~100x throughput that tightens the CI by ~10x.

This wrapper builds the per-hand root states in Python (the engine has no native `init_hand`), marshals
them as ONE contiguous int64 blob, hands the blueprint snapshot to the native `Lbr6Harness` ONCE, and
returns an `LBR6Result` / `LBRResult` that is drop-in compatible with the Python harness.

Correctness is statistical, not bit-identical (fast native RNG): the native estimator carries its own CI
and is gated by (a) CI-overlap equivalence vs the Python `lbr6_exploitability` on a shared snapshot, and
(b) the calibration anchors (always-fold ~ +750 mbb/g HU / ~ +128 bb/100 6-max). See
`tests/test_lbr_native_equiv.py`.
"""

from __future__ import annotations

import os

os.environ.setdefault("POKERBOT_BETTING_ABSTRACTION", "1")
os.environ.setdefault("POKERBOT_POSTFLOP_BUCKETS", "1")
os.environ.setdefault("POKERBOT_POSTFLOP_MODE", "fast")
os.environ.setdefault("POKERBOT_POSTFLOP_NUM_BUCKETS", "200")

import numpy as np  # noqa: E402

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from pokerbot.engine.jax_engine import CHIP_UNIT, NUM_PLAYERS, init_hand  # noqa: E402
from pokerbot.engine.np_engine import (  # noqa: E402
    ALL_IN,
    BET_BUCKET_START,
    CALL,
    CHECK,
    FOLD,
    NUM_ACTIONS,
    to_numpy_state,
)
from pokerbot.eval.lbr import LBRConfig, LBRResult  # noqa: E402
from pokerbot.eval.lbr6 import LBR6Config, LBR6Result  # noqa: E402
from pokerbot.search import native_solver  # noqa: E402  (reuse the .dll->.pyd loader + _to_flat)

_HU_OCC = np.array([False, False, False, True, False, True], dtype=bool)
_HU_LIVE = (3, 5)
_POLICY_MODE = {"blueprint": 0, "fold": 1, "nit": 2, "uniform": 3, "station": 4}

# Calibration policy modes need no snapshot; blueprint needs the real abstraction masks.
MODE_BLUEPRINT = 0
MODE_FOLD = 1
MODE_NIT = 2
MODE_UNIFORM = 3
MODE_STATION = 4


def _build_allowed_action_mask(bet_bucket_keep: tuple[int, ...] | None) -> np.ndarray | None:
    """Lean copy of the training action-abstraction mask for native LBR."""
    if bet_bucket_keep is None:
        return None
    allowed = np.zeros(NUM_ACTIONS, dtype=bool)
    for a in (FOLD, CHECK, CALL, ALL_IN):
        allowed[a] = True
    n_buckets = ALL_IN - BET_BUCKET_START
    for i in bet_bucket_keep:
        if not 0 <= int(i) < n_buckets:
            raise ValueError(f"bet_bucket_keep index {i} out of range [0,{n_buckets})")
        allowed[BET_BUCKET_START + int(i)] = True
    return allowed


# --------------------------------------------------------------------------- #
# Extension loading
# --------------------------------------------------------------------------- #
def _load_ext():
    """Return the battery_native extension exposing Lbr6Harness, or raise with a build hint."""
    ext = native_solver._try_load()
    if ext is None or not hasattr(ext, "Lbr6Harness"):
        raise RuntimeError(
            "native LBR unavailable: build it with\n"
            "  PYO3_PYTHON=$(python -c 'import sys;print(sys.executable)') \\\n"
            "  cargo build --release --features python --lib \\\n"
            "  --manifest-path tools/battery_native/Cargo.toml\n"
            "(then the .dll is auto-aliased to .pyd on import).")
    return ext


def available() -> bool:
    try:
        ext = native_solver._try_load()
        return ext is not None and hasattr(ext, "Lbr6Harness")
    except Exception:  # noqa: BLE001
        return False


def _allowed_masks():
    pre = _build_allowed_action_mask((4, 5, 6))
    post = _build_allowed_action_mask((0, 1, 3, 4, 5))
    return [bool(x) for x in pre], [bool(x) for x in post]


def _num_buckets() -> int:
    return int(os.environ.get("POKERBOT_POSTFLOP_NUM_BUCKETS", "200"))


def build_neural_harness(checkpoint: str):
    """Load the served AverageStrategyNet from `checkpoint` and build a native LbrNeuralHarness (the 3
    MLP layers as flat f32 arrays). Returns (harness, meta). Byte-close to the JAX serve (golden test)."""
    from pokerbot.training.neural_cfr import load_network_params

    ext = _load_ext()
    if not hasattr(ext, "LbrNeuralHarness"):
        raise RuntimeError("native LbrNeuralHarness missing -- rebuild battery_native (--features python).")
    p = load_network_params(checkpoint)
    hidden = int(p.get("hidden_dim", 256))
    dense = p["average_strategy_params"]["params"]["MLP_0"]

    def fl(x):
        return [float(v) for v in np.asarray(x, np.float32).reshape(-1)]

    d0, d1, d2 = dense["Dense_0"], dense["Dense_1"], dense["Dense_2"]
    out_dim = int(np.asarray(d2["bias"]).shape[0])
    harness = ext.LbrNeuralHarness(
        fl(d0["kernel"]), fl(d0["bias"]), fl(d1["kernel"]), fl(d1["bias"]),
        fl(d2["kernel"]), fl(d2["bias"]),
        [hidden, hidden, hidden], [hidden, hidden, out_dim], bool(p.get("info_suits")))
    return harness, {"hidden_dim": hidden, "iteration": p.get("iteration"),
                     "info_suits": bool(p.get("info_suits"))}


def lbr_hu_neural_native(checkpoint: str, cfg: LBRConfig | None = None) -> LBRResult:
    """Native HU LBR exploitability of the trained neural net. Deals at the trained HU occupancy
    (seats 3,5; button in {3,5}); set `cfg.stack_bb` / `cfg.ante_bb` to measure the SHALLOW tournament
    regime the net actually plays."""
    cfg = cfg or LBRConfig()
    harness, meta = build_neural_harness(checkpoint)
    lcfg = LBR6Config(
        hands=cfg.hands, seed=cfg.seed, n_seats=2, range_cap=cfg.range_cap,
        max_depth=cfg.max_depth, eq_runouts=max(2, cfg.equity_rollouts // 2),
        stack_bb=cfg.stack_bb, ante_bb=cfg.ante_bb)
    blob, seats, seeds = _build_roots(
        hands=cfg.hands, seed=cfg.seed, n_seats=2, occ=_HU_OCC,
        button_fn=lambda h: jnp.int32(_HU_LIVE[(h // 2) % 2]),
        seat_fn=lambda h: _HU_LIVE[h % 2],
        stack_bb=cfg.stack_bb, ante_bb=cfg.ante_bb)
    out = harness.lbr_neural(
        blob, seats, seeds, NUM_PLAYERS, bool(lcfg.track_ranges), int(lcfg.range_cap),
        int(lcfg.eq_samples), int(lcfg.eq_runouts), int(lcfg.settle_samples), int(lcfg.max_depth),
        [int(a) for a in cfg.bet_candidates], bool(lcfg.value_only_raises))
    ev = float(out.ev_bb_per_hand)
    return LBRResult(
        ev_bb_per_hand=ev, mbb_per_g=ev * 1000.0, bb_per_100=ev * 100.0,
        hands=int(out.hands), stderr_bb_per_hand=float(out.stderr_bb_per_hand),
        detail={"method": "native_analytic_1ply_lbr_hu_neural", "checkpoint": checkpoint,
                "meta": meta, "stack_bb": cfg.stack_bb, "ante_bb": cfg.ante_bb,
                "per_seat": _per_seat_dict(out), "native": True})


def _snapshot_to_dict(snapshot) -> dict:
    """dict[bytes16 -> list[float] len 11] from the trained snapshot (the harness parses this once)."""
    out = {}
    for k, v in snapshot.items():
        row = np.asarray(v, dtype=np.float64).reshape(-1)
        if row.shape[0] < 11:
            row = np.pad(row, (0, 11 - row.shape[0]))
        out[bytes(k)] = [float(x) for x in row[:11]]
    return out


# --------------------------------------------------------------------------- #
# Root building (engine has no native init_hand -- the wrapper deals the root)
# --------------------------------------------------------------------------- #
def _build_roots(*, hands, seed, n_seats, occ, button_fn, seat_fn, stack_bb, ante_bb):
    """Deal `hands` roots with the same JAX key schedule the Python harness uses (so the native and
    Python estimators see the same initial deals). Returns (blob_bytes, seats, seeds)."""
    key = jax.random.PRNGKey(seed)
    occ_j = jnp.asarray(occ)
    ante = jnp.int32(int(round(ante_bb * CHIP_UNIT)))
    stacks_j = (jnp.full(NUM_PLAYERS, int(round(stack_bb * CHIP_UNIT)), dtype=jnp.int32)
                if stack_bb is not None else None)
    flats, seats, seeds = [], [], []
    for h in range(hands):
        key, k = jax.random.split(key)
        button = button_fn(h)
        env0 = to_numpy_state(init_hand(
            k, button=button, stacks=stacks_j, occupancy=occ_j, ante=ante))
        flats.append(native_solver._to_flat(env0))
        seats.append(int(seat_fn(h)))
        seeds.append(int((seed * 1_000_003 + h) & 0x7FFFFFFFFFFFFFFF))
    blob = np.concatenate(flats).astype(np.int64).tobytes()
    return blob, seats, seeds


def _run(ext, snapshot_dict, *, blob, seats, seeds, n_seats, cfg, policy_mode, bet_candidates,
         depth_edges=None, emd_frozen_path: str | None = None):
    pre, post = _allowed_masks()
    num_buckets = _num_buckets()
    harness = ext.Lbr6Harness(snapshot_dict, pre, post, num_buckets,
                              [float(x) for x in (depth_edges or [])])
    emd_meta = None
    if emd_frozen_path:
        if not hasattr(harness, "set_emd_frozen"):
            raise AttributeError("native Lbr6Harness lacks set_emd_frozen; rebuild battery_native")
        emd_meta = harness.set_emd_frozen(str(emd_frozen_path))
        if int(emd_meta[0]) != int(num_buckets):
            raise RuntimeError(
                f"EMD frozen num_buckets={emd_meta[0]} != active num_buckets={num_buckets}"
            )
    out = harness.lbr_tabular(
        blob, seats, seeds, int(n_seats),
        bool(cfg.track_ranges), int(cfg.range_cap),
        int(cfg.eq_samples), int(cfg.eq_runouts), int(cfg.settle_samples), int(cfg.max_depth),
        [int(a) for a in bet_candidates], bool(cfg.value_only_raises), int(policy_mode))
    return out, emd_meta


def _per_seat_dict(out) -> dict:
    d = {}
    for (seat, n, bb100, std) in out.per_seat:
        d[int(seat)] = {"n": int(n), "bb_per_100": float(bb100), "std_bb_per_hand": float(std)}
    return d


# --------------------------------------------------------------------------- #
# 6-max blueprint (the Pluribus validator)
# --------------------------------------------------------------------------- #
def lbr6_exploitability_native(snapshot, cfg: LBR6Config | None = None, *,
                               policy: str = "blueprint", depth_edges=None,
                               emd_frozen_path: str | None = None) -> LBR6Result:
    """Native 6-max LBR lower bound. `policy`: 'blueprint' (real) or a calibration ref
    ('fold' ~ +128 bb/100, 'station' ~ +1400 bb/100, 'nit', 'uniform'). Mirrors
    `lbr6_exploitability`'s 6-max convention
    (occupancy = first n_seats live, random button, LBR seat = h % n_seats).

    `depth_edges`: pass the trained blueprint's depth-bucket edges (e.g. [50,90,140]) for a
    DEPTH-BUCKETED table, so the native preflop key hashes the depth byte and coverage is non-zero."""
    cfg = cfg or LBR6Config()
    ext = _load_ext()
    mode = _POLICY_MODE[policy]
    snap = _snapshot_to_dict(snapshot) if mode == MODE_BLUEPRINT else {}
    occ = [s < cfg.n_seats for s in range(NUM_PLAYERS)]
    blob, seats, seeds = _build_roots(
        hands=cfg.hands, seed=cfg.seed, n_seats=cfg.n_seats, occ=occ,
        button_fn=lambda h: None, seat_fn=lambda h: h % cfg.n_seats,
        stack_bb=cfg.stack_bb, ante_bb=cfg.ante_bb)
    out, emd_meta = _run(
        ext, snap, blob=blob, seats=seats, seeds=seeds, n_seats=cfg.n_seats, cfg=cfg,
        policy_mode=mode, bet_candidates=cfg.bet_candidates, depth_edges=depth_edges,
        emd_frozen_path=emd_frozen_path,
    )
    ev = float(out.ev_bb_per_hand)
    return LBR6Result(
        ev_bb_per_hand=ev, mbb_per_g=ev * 1000.0, bb_per_100=ev * 100.0,
        hands=int(out.hands), stderr_bb_per_hand=float(out.stderr_bb_per_hand),
        detail={
            "n_seats": cfg.n_seats, "method": "native_analytic_1ply_lbr", "policy": policy,
            "eq_samples": cfg.eq_samples, "track_ranges": cfg.track_ranges, "range_cap": cfg.range_cap,
            "per_seat": _per_seat_dict(out), "bet_candidates": list(cfg.bet_candidates),
            "stack_bb": cfg.stack_bb, "ante_bb": cfg.ante_bb, "native": True,
            "num_buckets": _num_buckets(), "emd_frozen_path": emd_frozen_path,
            "emd_frozen_meta": tuple(emd_meta) if emd_meta is not None else None,
        })


# --------------------------------------------------------------------------- #
# Heads-up (tabular) -- shared root convention with the HU net (seats 3,5)
# --------------------------------------------------------------------------- #
def lbr_hu_tabular_native(snapshot, cfg: LBRConfig | None = None, *,
                          policy: str = "blueprint", depth_edges=None,
                          emd_frozen_path: str | None = None) -> LBRResult:
    """Native HU LBR over a tabular snapshot (or a calibration policy). Uses the trained HU occupancy
    (seats 3,5, button in {3,5}) so the served info-state matches training, exactly like `lbr.py`."""
    cfg = cfg or LBRConfig()
    ext = _load_ext()
    mode = _POLICY_MODE[policy]
    snap = _snapshot_to_dict(snapshot) if mode == MODE_BLUEPRINT else {}
    # An LBR6Config-shaped view so _run can read the shared fields uniformly.
    lcfg = LBR6Config(
        hands=cfg.hands, seed=cfg.seed, n_seats=2, range_cap=cfg.range_cap,
        max_depth=cfg.max_depth, eq_runouts=max(2, cfg.equity_rollouts // 2),
        stack_bb=cfg.stack_bb, ante_bb=cfg.ante_bb)
    blob, seats, seeds = _build_roots(
        hands=cfg.hands, seed=cfg.seed, n_seats=2, occ=_HU_OCC,
        button_fn=lambda h: jnp.int32(_HU_LIVE[(h // 2) % 2]),
        seat_fn=lambda h: _HU_LIVE[h % 2],
        stack_bb=cfg.stack_bb, ante_bb=cfg.ante_bb)
    out, emd_meta = _run(
        ext, snap, blob=blob, seats=seats, seeds=seeds, n_seats=NUM_PLAYERS, cfg=lcfg,
        policy_mode=mode, bet_candidates=cfg.bet_candidates, depth_edges=depth_edges,
        emd_frozen_path=emd_frozen_path,
    )
    ev = float(out.ev_bb_per_hand)
    return LBRResult(
        ev_bb_per_hand=ev, mbb_per_g=ev * 1000.0, bb_per_100=ev * 100.0,
        hands=int(out.hands), stderr_bb_per_hand=float(out.stderr_bb_per_hand),
        detail={
            "method": "native_analytic_1ply_lbr_hu", "policy": policy,
            "per_seat": _per_seat_dict(out), "stack_bb": cfg.stack_bb, "ante_bb": cfg.ante_bb,
            "native": True, "num_buckets": _num_buckets(), "emd_frozen_path": emd_frozen_path,
            "emd_frozen_meta": tuple(emd_meta) if emd_meta is not None else None,
        })
