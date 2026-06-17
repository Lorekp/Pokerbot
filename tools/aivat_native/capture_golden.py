"""Capture GOLDEN vectors from the Python AIVAT oracle + profile the AIVAT path.

Run (git-bash):
    POKERBOT_BETTING_ABSTRACTION=1 python tools/aivat_native/capture_golden.py

Writes golden/golden.json (inputs + EXACT float64 outputs, hex-encoded for bit fidelity) for:
  * decision_correction
  * chance_correction
  * AIVATAccumulator.summary()
  * chance_corrections_for_hand (per-reveal corrections on a real played hand + blueprint)

Also prints a cProfile of the chance-term path so we can see where time goes.
"""

from __future__ import annotations

import cProfile
import json
import os
import pstats
import random
import struct
import time
from io import StringIO

import numpy as np

os.environ.setdefault("POKERBOT_BETTING_ABSTRACTION", "1")

from pokerbot.eval.aivat import (
    AIVATAccumulator,
    chance_correction,
    decision_correction,
)

GOLDEN = os.path.join(os.path.dirname(__file__), "golden", "golden.json")


def f64hex(x: float) -> str:
    """Loss-free hex of a float64 (so the golden file is bit-exact)."""
    return struct.pack("<d", float(x)).hex()


def arr_hex(a) -> list[str]:
    return [f64hex(v) for v in np.asarray(a, dtype=np.float64).reshape(-1).tolist()]


# ---------------------------------------------------------------------------
# 1) decision_correction goldens (deterministic, no RNG)
# ---------------------------------------------------------------------------
def capture_decision():
    rng = np.random.default_rng(12345)
    cases = []
    raw_inputs = [
        # (probs, baseline_values, taken_idx)
        ([0.2, 0.3, 0.5], [1.0, -2.0, 3.5], 0),
        ([0.0, 0.0, 1.0], [5.0, 5.0, 5.0], 2),
        ([0.1, 0.9], [10.0, -10.0], 1),
        ([0.0, 0.0, 0.0], [1.0, 2.0, 3.0], 1),       # sum<=0 -> 0.0
        ([0.5, 0.5], [1.0, 2.0], 5),                  # bad idx -> 0.0
        ([1.0], [42.0], 0),
    ]
    # plus several random large rows to exercise normalization + pairwise-ish sums
    for n in (3, 5, 9, 11, 200):
        p = rng.random(n)
        bv = rng.standard_normal(n) * 7.0
        raw_inputs.append((p.tolist(), bv.tolist(), int(rng.integers(0, n))))
    for probs, bv, ti in raw_inputs:
        out = decision_correction(probs, bv, ti)
        cases.append({
            "probs": arr_hex(probs),
            "baseline_values": arr_hex(bv),
            "taken_idx": ti,
            "out": f64hex(out),
        })
    return cases


# ---------------------------------------------------------------------------
# 2) chance_correction goldens (trivial subtraction)
# ---------------------------------------------------------------------------
def capture_chance_corr():
    rng = np.random.default_rng(777)
    cases = []
    pairs = [(1.5, 0.25), (-3.0, -3.0), (0.0, 7.25), (1e-17, 1e17)]
    for _ in range(8):
        pairs.append((float(rng.standard_normal()), float(rng.standard_normal())))
    for realized, expected in pairs:
        out = chance_correction(expected, realized)
        cases.append({
            "expected": f64hex(expected),
            "realized": f64hex(realized),
            "out": f64hex(out),
        })
    return cases


# ---------------------------------------------------------------------------
# 3) AIVATAccumulator.summary() goldens
# ---------------------------------------------------------------------------
def capture_accumulator():
    rng = np.random.default_rng(2024)
    cases = []

    def run(hands):
        acc = AIVATAccumulator()
        for raw, corr, ch in hands:
            acc.add_hand(raw, corr, ch)
        return acc.summary()

    scenarios = []
    # empty
    scenarios.append([])
    # single hand
    scenarios.append([(3.0, [0.5], [0.25])])
    # several hands, mixed corrections present/None
    hs = []
    for _ in range(50):
        raw = float(rng.standard_normal() * 20)
        nc = int(rng.integers(0, 4))
        corr = rng.standard_normal(nc).tolist() if nc else None
        nch = int(rng.integers(0, 3))
        ch = rng.standard_normal(nch).tolist() if nch else None
        hs.append((raw, corr, ch))
    scenarios.append(hs)
    # large -> exercises numpy std/mean pairwise summation (>128)
    hs2 = [(float(rng.standard_normal() * 13), [float(rng.standard_normal())], None)
           for _ in range(500)]
    scenarios.append(hs2)

    for sc in scenarios:
        summ = run(sc)
        cases.append({
            "hands_in": [
                {"raw": f64hex(r),
                 "corr": (arr_hex(c) if c else None),
                 "ch": (arr_hex(ch) if ch else None)}
                for (r, c, ch) in sc
            ],
            "summary": {
                "hands": summ["hands"],
                "raw_mean_bb": f64hex(summ["raw_mean_bb"]),
                "adj_mean_bb": f64hex(summ["adj_mean_bb"]),
                "raw_std": f64hex(summ["raw_std"]),
                "adj_std": f64hex(summ["adj_std"]),
                "variance_reduction": f64hex(summ["variance_reduction"]),
            },
        })
    return cases


# ---------------------------------------------------------------------------
# 4) chance_corrections_for_hand on a REAL played hand + real blueprint
# ---------------------------------------------------------------------------
def _play_random_hand(seed: int):
    """One all-calling-station hand reaching showdown (holes preserved on the final state)."""
    import random as _r

    from pokerbot.engine.game import play_hand
    from pokerbot.eval.bots import BOT_REGISTRY
    from pokerbot.engine.game import NUM_PLAYERS

    policies = {s: BOT_REGISTRY["calling_station"](_r.Random(seed * 13 + s))
                for s in range(NUM_PLAYERS)}
    fs, _ = play_hand(policies, rng=_r.Random(seed))
    return fs


def _load_adapter():
    import pickle

    from pokerbot.search.tabular_blueprint_adapter import TabularBlueprintAdapter

    path = os.path.expandvars(r"%LOCALAPPDATA%\Temp\bp6max_dcfrvr_iter200.pkl")
    if not os.path.exists(path):
        path = r"checkpoints/bp6max_dcfrvr_iter200.pkl"
    with open(path, "rb") as f:
        blob = pickle.load(f)
    strat = blob["strat_sum"] if isinstance(blob, dict) and "strat_sum" in blob else blob
    return TabularBlueprintAdapter(snapshot=strat)


def capture_chance_for_hand(profile: bool = True):
    from pokerbot.eval.chance_term import chance_corrections_for_hand
    from pokerbot.search.pluribus_rollout import ContinuationLeafConfig

    adapter = _load_adapter()
    cfg = ContinuationLeafConfig(select="gadget", rollout_samples=2, gadget_iters=64,
                                 max_playout_depth=48)

    cases = []
    hands = []
    for seed in range(40):
        gs = _play_random_hand(seed)
        if len(gs.board) >= 4 and gs.is_terminal():
            hands.append((seed, gs))
        if len(hands) >= 8:
            break

    # Profile the chance term over these hands.
    prof_out = ""
    if profile and hands:
        pr = cProfile.Profile()
        pr.enable()
        t0 = time.perf_counter()
        for seed, gs in hands:
            chance_corrections_for_hand(gs, hero_seat=0, adapter=adapter, leaf_cfg=cfg,
                                        include_flop=False, base_seed=0)
        wall = time.perf_counter() - t0
        pr.disable()
        s = StringIO()
        ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")
        ps.print_stats(28)
        prof_out = s.getvalue()
        print(f"\n=== chance-term wall time: {wall:.3f}s over {len(hands)} hands "
              f"({wall/len(hands)*1000:.1f} ms/hand) ===")
        print(prof_out)

    # Capture goldens (deterministic given base_seed; gadget RNG seeded inside).
    for seed, gs in hands:
        out = chance_corrections_for_hand(gs, hero_seat=0, adapter=adapter, leaf_cfg=cfg,
                                          include_flop=False, base_seed=0)
        cases.append({
            "seed": seed,
            "hero_seat": 0,
            "out": [f64hex(v) for v in out],
        })
    return cases, prof_out


# ---------------------------------------------------------------------------
# 5) chance-term REDUCTION goldens: c_C = realized - float(np.mean(vals))
#    This is the EXACT per-reveal arithmetic ported to native (the gadget v
#    values are precomputed inputs). Covers the numpy pairwise-summation trap.
# ---------------------------------------------------------------------------
def capture_chance_reduce():
    rng = np.random.default_rng(31337)
    cases = []

    def emit(realized, vals):
        v = np.asarray(vals, dtype=np.float64)
        expected = float(np.mean(v)) if len(v) else float(realized)
        out = float(realized) - float(expected)
        cases.append({
            "realized": f64hex(realized),
            "vals": [f64hex(x) for x in v.reshape(-1).tolist()],
            "out": f64hex(out),
        })

    # Edge: empty vals -> expected == realized -> out 0.0
    emit(1.25, [])
    emit(0.0, [0.0])
    emit(-3.5, [-3.5, -3.5, -3.5])
    # Turn/river-sized (<=47): the real enumeration size.
    for n in (37, 44, 46, 47):
        realized = float(rng.standard_normal() * 5)
        emit(realized, (rng.standard_normal(n) * 5).tolist())
    # Sizes straddling numpy's 128-element pairwise-summation threshold (THE TRAP).
    for n in (8, 64, 127, 128, 129, 200, 1000, 4096):
        realized = float(rng.standard_normal() * 11)
        emit(realized, (rng.standard_normal(n) * 11).tolist())
    # Real precomputed gadget vals from one hand's turn/river reveals (exact inputs).
    try:
        real_vals = _capture_real_reveal_vals()
        for realized, vals in real_vals:
            emit(realized, vals)
    except Exception as e:  # pragma: no cover
        print("  (real-vals capture skipped:", e, ")")
    return cases


def _capture_real_reveal_vals():
    """Return [(realized_v, [candidate_v ...]), ...] for the turn/river reveals of one real hand."""
    from pokerbot.eval.chance_term import (
        _complement,
        _env_with_board,
        _replay_reveals,
        _value,
        _card_idx,
    )
    from pokerbot.search.pluribus_rollout import ContinuationLeafConfig

    adapter = _load_adapter()
    cfg = ContinuationLeafConfig(select="gadget", rollout_samples=2, gadget_iters=64,
                                 max_playout_depth=48)
    gs = _play_random_hand(3)
    holes = [p.hole for p in gs.players]
    out = []
    reveals = _replay_reveals(gs)
    for ri, rev in enumerate(reveals):
        if rev.n_new != 1:
            continue
        bp = list(rev.state.board[: rev.prev_board_len])
        comp = _complement(holes, bp)
        realized_cards = [_card_idx(c) for c in
                          rev.state.board[rev.prev_board_len: rev.prev_board_len + 1]]
        seed = 100003 * (ri + 1)
        realized = _value(_env_with_board(rev, 0, realized_cards), 0, adapter, cfg, seed)
        vals = [_value(_env_with_board(rev, 0, [c]), 0, adapter, cfg, seed) for c in comp]
        out.append((float(realized), [float(x) for x in vals]))
    return out


def main():
    print("Capturing decision_correction goldens...")
    dec = capture_decision()
    print(f"  {len(dec)} cases")
    print("Capturing chance_correction goldens...")
    chc = capture_chance_corr()
    print(f"  {len(chc)} cases")
    print("Capturing AIVATAccumulator goldens...")
    acc = capture_accumulator()
    print(f"  {len(acc)} cases")
    print("Capturing chance-reduction goldens (the native boundary)...")
    cr = capture_chance_reduce()
    print(f"  {len(cr)} cases")
    print("Capturing chance_corrections_for_hand goldens (real blueprint, profiling)...")
    cfh, prof = capture_chance_for_hand(profile=True)
    print(f"  {len(cfh)} hand cases")

    golden = {
        "numpy_version": np.__version__,
        "decision_correction": dec,
        "chance_correction": chc,
        "accumulator": acc,
        "chance_reduce": cr,
        "chance_corrections_for_hand": cfh,
    }
    os.makedirs(os.path.dirname(GOLDEN), exist_ok=True)
    with open(GOLDEN, "w") as f:
        json.dump(golden, f, indent=1)
    print(f"\nWrote {GOLDEN}")


if __name__ == "__main__":
    main()
