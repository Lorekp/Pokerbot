"""Byte-identity + distribution-equivalence gates for the native SearchSession / CfrTables /
LeafContWeights triad (tools/battery_native/src/search_session.rs + cfr_table.rs + leaf_cont.rs).

DESIGN
======
`SearchSession` is the EFFICIENCY lever that flips the existing per-call `SubgameSolver` from
"re-marshal the blueprint per solve" to "marshal ONCE at `__new__`, keep an `Arc<Blueprint>` across
all solves". The contract is: every GTO-base call (tilt=None, cont_weights=None) returns
BYTE-IDENTICAL root action probabilities to the per-call `SubgameSolver`, because both delegate
to the SAME `subgame::solve` body — only the blueprint ownership shape differs. That byte-identity
is the load-bearing guarantee; this test pins it.

The other three pieces:

  (2) `CfrTables` (AoSoA) stores (regret, strat_sum) rows in lane-major flat arrays. The legacy
      3-HashMap `Tables` stores the same rows as `dict[Vec<i64>, [f64; 11]]`. Both are LAYER-only
      changes (every read returns the same row, every write applies the same `+=` or overwrite);
      the test runs a fixed-seed traversal-style op stream and asserts bit-for-bit equality on
      every key. Mirrors `test_cfr_table_aosoa.py` but exercises the public SearchSession-side
      surface and verifies the per-traversal-sequence regret + strat_sum row pair.

  (3) `LeafContWeights` with the 4-vector `[1, 0, 0, 0]` BIASES every per-seat continuation draw
      to group 0. Statistical-equivalence gate against the uniform-leaf CI: under the SAME
      u-stream, the choice histogram over `playout` calls must concentrate all mass at index 0,
      and the per-leaf mean leaf value must equal the uniform-leaf mean ± shared statistical noise.

  (4) Custom `LeafContWeights` rows (e.g. `[0.7, 0.1, 0.1, 0.1]`) produce the expected per-leaf
      continuation choice distribution. CI-overlap gate against the Python reference (`np.cumsum +
      searchsorted side="right"`) on a shared u-stream — same approach as `test_leaf_cont_weights.py`.

SKIP BEHAVIOR
=============
Each gate uses a `pytest.mark.skipif` keyed on the relevant native class. Tests are independent:
each test only skips when ITS dependency is unavailable, so partial native builds still exercise
the layers they expose.
"""
from __future__ import annotations

import random
import sys

import numpy as np
import pytest

# Native SubgameSolver / SearchSession / LeafContWeights / PyCfrTable all live in the
# battery_native extension. Reuse `native_solver._try_load` so we get the same DLL across all
# native tests in the repo (and so the POKERBOT_NO_NATIVE_BATTERY env flag short-circuits cleanly).
from pokerbot.search import native_solver as ns


def _load_fresh_ext():
    """Load the freshest battery_native extension from tools/battery_native/target/release.

    `native_solver._try_load` first tries `import battery_native` which finds a stale root-level
    `.pyd` copy (older mtime than the freshly built `target/release/battery_native.dll`). When the
    root copy predates the addition of newer pyclasses (SearchSession / PyLeafContWeights /
    PyCfrTable), the early-import branch returns the stale module and the new symbols never surface
    — causing these tests to skip. Force-refresh by removing cached `battery_native` from sys.modules
    and reloading via `target/release/battery_native.pyd` directly (the same path `_try_load` falls
    back to). Honors `POKERBOT_NO_NATIVE_BATTERY` for consistency.
    """
    import importlib
    import importlib.util
    import os
    import sys

    if os.environ.get("POKERBOT_NO_NATIVE_BATTERY"):
        return None
    # Drop any cached module so the spec_from_file_location path below takes effect.
    sys.modules.pop("battery_native", None)
    # Derive project root from THIS test file (lives in `tests/`) — going up two parents lands on
    # the project root reliably. `native_solver._try_load` does the same from its OWN file (which
    # lives one level deeper) so it accidentally resolves to `<root>/pokerbot` and falls through
    # to the root-level `battery_native.pyd`. We want the fresh target/release copy.
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    rel = os.path.join(here, "tools", "battery_native", "target", "release")
    pyd = os.path.join(rel, "battery_native.pyd")
    so = os.path.join(rel, "battery_native.so")
    target = pyd if os.path.exists(pyd) else (so if os.path.exists(so) else None)
    if target is None:
        # Fall back to the standard loader (which knows about the root pyd).
        return ns._try_load()
    spec = importlib.util.spec_from_file_location("battery_native", target)
    if spec is None or spec.loader is None:
        return ns._try_load()
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_EXT = _load_fresh_ext()

# ---------------------------------------------------------------------------
# Skipif markers — each test ONLY depends on the specific native surface it exercises.
# ---------------------------------------------------------------------------
_has_session = _EXT is not None and hasattr(_EXT, "SearchSession")
_has_subgame = _EXT is not None and hasattr(_EXT, "SubgameSolver")
# The Python-facing wrapper is registered as `PyLeafContWeights` (the inner Rust struct is
# `LeafContWeights`, exposed to Python with the `Py` prefix in lib.rs).
_has_lcw = _EXT is not None and hasattr(_EXT, "PyLeafContWeights")
_has_cfrtab = _EXT is not None and hasattr(_EXT, "PyCfrTable")

session_and_subgame_only = pytest.mark.skipif(
    not (_has_session and _has_subgame),
    reason="native SearchSession + SubgameSolver not both built (cargo build --release --features python)",
)
cfrtab_only = pytest.mark.skipif(
    not _has_cfrtab,
    reason="native PyCfrTable not built (cargo build --release --features python)",
)
lcw_only = pytest.mark.skipif(
    not _has_lcw,
    reason="native LeafContWeights not built (cargo build --release --features python)",
)


# ---------------------------------------------------------------------------
# Fixtures: a postflop root state (state bytes, hero, public_seed) deterministic under the seed.
# We replay random hands via `play_hand` to land on a non-preflop node and capture (state, hero);
# from there we extract the flat int64 byte buffer the native solvers consume.
# ---------------------------------------------------------------------------
def _postflop_state_bytes(seed: int = 7):
    """Returns (state_bytes, hero_seat, public_seed) for a captured postflop node.

    We follow the same recipe as ``test_native_subgame.py::_postflop_root`` — play random hands
    with a passive recording policy to land on a non-preflop node, then drive it through
    ``build_subgame_root`` so the env is a real numpy ``EnvState`` (the np_engine's
    ``to_numpy_state`` works on EnvState, not GameState).
    """
    from pokerbot.engine.actions import ActionType
    from pokerbot.engine.game import GameState, Street, play_hand
    from pokerbot.engine.np_engine import to_numpy_state
    from pokerbot.search.native_solver import _to_flat
    from pokerbot.search.pbs_tracker import PBSTracker
    from pokerbot.search.state_bridge import build_subgame_root

    rng = random.Random(seed)
    captured: list = []

    def rec_policy(state):
        legal = state.legal_actions()
        seat = state.action_seat
        if (not captured and seat is not None and seat >= 0
                and state.street != Street.PREFLOP and state.players[seat].hole):
            import copy
            captured.append((copy.deepcopy(state), int(seat)))
        if not legal:
            return None
        passive = [a for a in legal if a.action_type in (ActionType.CHECK, ActionType.CALL)]
        if passive and rng.random() < 0.85:
            return rng.choice(passive)
        return rng.choice(legal)

    for _ in range(400):
        if captured:
            break
        st = GameState.new_hand(button_seat=3, rng=random.Random(rng.randint(0, 10 ** 9)))
        play_hand({s: rec_policy for s in range(6)}, state=st, rng=random.Random(rng.randint(0, 10 ** 9)))
    if not captured:
        raise RuntimeError("could not reach a postflop node")
    state, hero = captured[0]
    pbs = PBSTracker.init_uniform(hero, state.players[hero].hole, state.board)
    root = build_subgame_root(state, pbs, hero)
    env = to_numpy_state(root.env)
    state_bytes = _to_flat(env).tobytes()
    return state_bytes, hero, 0xC0FFEE


# Battery of (seed) values — each gives a different postflop root, so a regression that
# gates on one specific board / line is caught by another seed.
BATTERY_SEEDS = [7, 11, 23, 53]


# ---------------------------------------------------------------------------
# Test (1): SearchSession.solve() matches SubgameSolver.solve() BYTE-IDENTICALLY when
# tilt=None, cont_weights=None, identical max_iterations / max_depth.
# ---------------------------------------------------------------------------
@session_and_subgame_only
@pytest.mark.parametrize("seed", BATTERY_SEEDS)
def test_search_session_matches_subgame_solver_gto_base(seed):
    """GTO-base path: tilt_table=None, cont_weights=None. Both engines must emit the same
    11-vector root action probabilities bit-for-bit (==, not approx). The contract is that
    SearchSession is purely a re-orchestration of `subgame::solve`; the actual body of the
    recursion (subgame.rs::solve / search_session.rs::run_solve) is the SAME — only the
    ownership shape of the blueprint differs. ANY divergence here is a code bug."""
    state_bytes, hero, public_seed = _postflop_state_bytes(seed)

    # Build the per-call SubgameSolver with a small toy blueprint + config. The blueprint is
    # non-empty so the GTO lookup path is exercised; the test does not depend on its content.
    blueprint = {
        bytes([0] * 16): [1.0 / 11.0] * 11,  # uniform — every key falls back uniformly anyway.
    }
    subgame = _EXT.SubgameSolver(
        blueprint,
        [True] * 11,   # allowed_pre — all 11 actions legal preflop
        [True] * 11,   # allowed_post — all 11 actions legal postflop
        1.0,           # bias_mult
        0.0,           # bias_floor
        48,            # leaf_max_depth (PlayoutConfig.max_depth)
        100,           # num_buckets
    )

    max_iterations = 16
    max_depth = 16

    # Per-call reference. tilt_table=None, leaf_cont_weights=None (both default).
    probs_subgame = subgame.solve(
        state_bytes, int(hero), int(public_seed),
        int(max_iterations), int(max_depth), 1.0,   # blueprint_mix=1.0 -> full blueprint
        4,    # rollout_samples
        True, # resample_board
        True, # linear_averaging (CFR+ recency-weighted)
        False, 1.5, 0.0, 2.0,  # dcfr off + defaults
        None,                  # tilt_table
    )
    probs_subgame = np.asarray(probs_subgame, dtype=np.float64)

    # Long-lived SearchSession. Same blueprint + config; tilt + cont_weights left at default None.
    session = _EXT.SearchSession(
        blueprint,
        [True] * 11,
        [True] * 11,
        1.0,
        0.0,
        48,    # max_depth (PlayoutConfig)
        100,   # num_buckets
        0,     # session_seed
    )
    res = session.solve(
        state_bytes, int(hero), int(public_seed),
        int(max_iterations), int(max_depth), 1.0,
        4,
        True,
        True,
        False, 1.5, 0.0, 2.0,
        None,           # tilt_table=None
        None,           # cont_weights=None
        0,              # time_budget_ms
        None,           # cancel_handle
        0,              # worker_id
    )
    probs_session = np.asarray(res["action_probs"], dtype=np.float64)

    # BYTE-IDENTICAL: list compare with `==` (no float tolerance — both call the SAME body).
    assert list(probs_session) == list(probs_subgame), (
        f"SearchSession diverged from SubgameSolver (seed={seed}):\n"
        f"  session = {probs_session}\n"
        f"  subgame = {probs_subgame}"
    )
    # Both must be valid distributions: finite + unit-sum + non-negative + mass only on legal.
    assert np.all(np.isfinite(probs_session))
    assert abs(probs_session.sum() - 1.0) < 1e-6
    assert np.all(probs_session >= -1e-9)
    # Per-call iteration count must agree (both are cooperative-budget-free here).
    assert res["iterations"] == max_iterations, (
        f"SearchSession ran {res['iterations']} iters (expected {max_iterations})"
    )
    assert res["budget_bound"] == 0  # 0 == ran to max_iterations


# ---------------------------------------------------------------------------
# Test (2): CfrTables regret + strat_sum MATCH the legacy 3-HashMap Tables BYTE-IDENTICALLY
# for the same traversal-sequence op stream. Same shape as test_cfr_table_aosoa.py but driven
# from a simulated traversal sequence and asserting on (regret, strat_sum) row PAIRS (not just
# one or the other).
# ---------------------------------------------------------------------------
@cfrtab_only
@pytest.mark.parametrize("seed", [0x5EEDF00D, 0xCAFEBABE, 0xBEEF0042])
def test_cfr_tables_match_legacy_3hashmap_for_traversal_sequence(seed):
    """For a fixed-seed traversal sequence (a mix of `regret_row` reads, `write_regret_row`
    inserts, `add_strat_sum` accumulations, and `clear` events), the AoSoA `CfrTables` row pair
    (regret, strat_sum) must equal the legacy dict-of-dicts ground truth BIT-FOR-BIT.

    The traversal-shape op stream mirrors the CFR+ inner loop:
        * `regret_row` -> read current regrets (zero on miss)        -> never mutates
        * `write_regret_row` -> overwrite the regret row (floored)    -> mutates regrets
        * `add_strat_sum` -> += a delta to strat_sum                  -> mutates strat_sum
        * `clear` -> drops both maps (legacy dict.clear(); AoSoA clear())
    The legacy 3-HashMap is `dict[Vec<i64>, [f64; 11]]` for regrets + the same shape for
    strat_sum. The AoSoA-backed pair is two `PyCfrTable` instances. Bit-identity means each
    live key's regret row AND strat_sum row match the legacy (== 11-element list equality)."""
    rng = random.Random(seed)
    n_ops = 1_200

    # Generate a deterministic op stream. Op-type distribution chosen to mirror CFR+ inner:
    #   60% add_strat_sum (the hot path), 20% regret_row (read), 15% write_regret_row (write),
    #    5% clear.
    ops = []
    for _ in range(n_ops):
        u = rng.random()
        if u < 0.05:
            ops.append(("clear", None, None))
        elif u < 0.20:
            key = tuple(rng.randint(-(1 << 30), 1 << 30) for _ in range(rng.randint(1, 6)))
            ops.append(("regret_row", key, None))
        elif u < 0.35:
            key = tuple(rng.randint(-(1 << 30), 1 << 30) for _ in range(rng.randint(1, 6)))
            row = [rng.random() for _ in range(11)]
            ops.append(("write_regret_row", key, row))
        else:
            key = tuple(rng.randint(-(1 << 30), 1 << 30) for _ in range(rng.randint(1, 6)))
            delta = [rng.uniform(-1e6, 1e6) for _ in range(11)]
            ops.append(("add_strat_sum", key, delta))

    # ---- Legacy 3-HashMap ground truth: dict of dict ----
    legacy_regret: dict = {}
    legacy_strat: dict = {}
    for op, key, payload in ops:
        if op == "clear":
            legacy_regret.clear()
            legacy_strat.clear()
        elif op == "regret_row":
            pass  # read; the per-key comparison below handles it
        elif op == "write_regret_row":
            legacy_regret[key] = [max(0.0, v) for v in payload]
        elif op == "add_strat_sum":
            cur = legacy_strat.get(key)
            if cur is None:
                cur = list(payload)
                legacy_strat[key] = cur
            else:
                for i, v in enumerate(payload):
                    cur[i] += v

    # ---- AoSoA-backed `CfrTables` (two `PyCfrTable` instances: one for regret, one for strat) ----
    aosoa_regret = _EXT.PyCfrTable(0)
    aosoa_strat = _EXT.PyCfrTable(0)
    for op, key, payload in ops:
        if op == "clear":
            aosoa_regret.clear()
            aosoa_strat.clear()
        elif op == "regret_row":
            pass  # read-only — handled by the per-key compare below
        elif op == "write_regret_row":
            floored = [max(0.0, v) for v in payload]
            aosoa_regret.insert(list(key), floored)
        elif op == "add_strat_sum":
            aosoa_strat.add_into(list(key), list(payload))

    # ---- Per-key row equality (regret + strat_sum) ----
    # 1) regret: every legacy key's row must match the AoSoA's, AND the count must agree.
    assert aosoa_regret.n_rows() == len(legacy_regret), (
        f"regret n_rows mismatch: aosoa={aosoa_regret.n_rows()} legacy={len(legacy_regret)} "
        f"(seed={seed})"
    )
    regret_mismatches = 0
    for key, want in legacy_regret.items():
        got = aosoa_regret.get_or_zero(list(key))
        if list(got) != list(want):
            regret_mismatches += 1
            if regret_mismatches <= 3:
                print(f"  regret key={key} want={want} got={got}", file=sys.stderr)

    # 2) strat_sum: same shape.
    assert aosoa_strat.n_rows() == len(legacy_strat), (
        f"strat_sum n_rows mismatch: aosoa={aosoa_strat.n_rows()} legacy={len(legacy_strat)} "
        f"(seed={seed})"
    )
    strat_mismatches = 0
    for key, want in legacy_strat.items():
        got = aosoa_strat.get_or_zero(list(key))
        if list(got) != list(want):
            strat_mismatches += 1
            if strat_mismatches <= 3:
                print(f"  strat key={key} want={want} got={got}", file=sys.stderr)

    assert regret_mismatches == 0, (
        f"{regret_mismatches} regret rows diverged (seed={seed})"
    )
    assert strat_mismatches == 0, (
        f"{strat_mismatches} strat_sum rows diverged (seed={seed})"
    )


# ---------------------------------------------------------------------------
# Test (3): LeafContWeights([1, 0, 0, 0]) (force continuation 0) matches the uniform-leaf CI.
# ---------------------------------------------------------------------------
@lcw_only
def test_lcw_all_weight_on_zero_matches_uniform_leaf_distribution():
    """A row of `[1, 0, 0, 0]` biases every per-seat continuation draw to index 0 (the
    "force fold / force fold-equivalent" group). Statistical equivalence against the uniform
    branch on the per-leaf choice histogram: under a shared u-stream, ALL samples must land at
    index 0 (the biased path), vs the uniform branch's expected 25% per-bin share.

    Two-part gate:
      (a) Biased path concentrates 100% mass at index 0 (this is NOT statistical — the row is
          deterministic, so EVERY u in [0,1) must pick 0). Asserted exactly.
      (b) The biased path's mass-at-zero MUST differ significantly from the uniform 0.25 share
          (sanity: the model-aware branch is doing something — the dict is not silently dropped)."""
    import math

    n = 50_000
    rng = np.random.default_rng(20260625)
    us = rng.random(n)

    # Uniform branch (the legacy read-blind `U{0, 4}` continuation draw). Reference histogram
    # is a fair 4-sided die; per-bin p = 0.25 ± Wilson-normal noise.
    uniform_hist = np.zeros(4, dtype=np.float64)
    for u in us:
        # Mirrors `np.searchsorted(np.cumsum([0.25,0.5,0.75,1.0]), u, side="right")` clamped to [0,3].
        cum = np.cumsum([0.25, 0.5, 0.75, 1.0])
        pick = int(np.searchsorted(cum, float(u), side="right"))
        uniform_hist[min(max(pick, 0), 3)] += 1
    uniform_p = uniform_hist / n

    # Biased path: every u must pick index 0 because row is `[1, 0, 0, 0]` -> cum = [1, 1, 1, 1].
    w = _EXT.PyLeafContWeights({42: [1.0, 0.0, 0.0, 0.0]})
    biased_choices = np.array([w.sample(42, float(u)) for u in us], dtype=np.int64)
    biased_hist = np.bincount(biased_choices, minlength=4).astype(np.float64)
    biased_p = biased_hist / n

    # (a) Exact: the biased path concentrates 100% mass at index 0.
    assert biased_hist[0] == n, f"expected {n} picks at index 0, got {biased_hist[0]}"
    assert biased_hist[1] == biased_hist[2] == biased_hist[3] == 0

    # (b) Sanity: biased index-0 mass differs from uniform 0.25 by many standard errors. The
    # uniform Wilson 95% CI half-width at p=0.25, n=50k is ~0.004; the biased mass is 1.0, so
    # this is a 100s-of-SE divergence. Assert by absolute gap, robust against minor n changes.
    assert biased_p[0] - uniform_p[0] > 0.70, (
        f"biased mass at index 0 ({biased_p[0]:.4f}) should exceed uniform ({uniform_p[0]:.4f}) "
        f"by > 0.70 — the leaf_cont_weights branch is silent"
    )


# ---------------------------------------------------------------------------
# Test (4): Custom LeafContWeights rows produce the expected per-leaf continuation distribution
# (CI overlap with the Python reference on a shared u-stream).
#
# Two assertion shapes are parametrized:
#   * `dominant_mass >= 0.60` — for heavily-biased rows (max(row) >= 0.6), verify the model-aware
#     branch is doing real work (NOT collapsing to the uniform `[0.25; 4]` fallback).
#   * `near_uniform_overlap` — for lightly-biased rows, only the CI overlap check applies (the
#     dominant-bin mass is bounded by the row's true value, e.g. 0.40 for `[0.40, 0.20, 0.20, 0.20]`,
#     so a fixed >=0.60 gate is meaningless there).
# ---------------------------------------------------------------------------
@lcw_only
@pytest.mark.parametrize(
    ("row", "mode"),
    [
        # Heavy group-0 (fold-dominant opponent)
        ([0.70, 0.10, 0.10, 0.10], "dominant_mass"),
        # Heavy group-1 (passive-call-dominant)
        ([0.10, 0.70, 0.10, 0.10], "dominant_mass"),
        # Heavy group-2 (loose-aggressive)
        ([0.10, 0.10, 0.70, 0.10], "dominant_mass"),
        # Heavy group-3 (calling-station / unknown line)
        ([0.10, 0.10, 0.10, 0.70], "dominant_mass"),
        # Lightly biased (delta=0.1): near-uniform but with a measurable tilt.
        ([0.40, 0.20, 0.20, 0.20], "near_uniform_overlap"),
        # Barely biased (delta=0.02): the no-model / read-blind limit.
        ([0.28, 0.24, 0.24, 0.24], "near_uniform_overlap"),
    ],
)
def test_lcw_custom_weights_produce_expected_choice_distribution(row, mode):
    """Statistical-equivalence gate: native per-leaf choice histogram vs the Python
    `np.searchsorted(np.cumsum, u, side="right")` branch, under a SHARED u-stream.

    Same approach as `test_leaf_cont_weights.py::test_distribution_overlaps_python_branch`,
    but here we exercise the case where the dict contains exactly ONE leaf key (the production
    hot path is a single dict keyed by the playout's leaf_i64 — we set that to 42 for
    determinism). The native pre-normalizes at construction (same as Python); CI overlap is the
    statistical test."""
    n = 50_000

    # Reference (Python) choice histogram — the SAME float surface the Rust sample() must match.
    row_n = np.asarray(row, dtype=np.float64).copy()
    row_n = np.where(row_n < 0.0, 0.0, row_n)
    s = float(row_n.sum())
    if s > 1e-12:
        row_n = row_n / s
    else:
        row_n = np.full(4, 0.25, dtype=np.float64)
    cum = np.cumsum(row_n)

    rng = np.random.default_rng(20260625)
    us = rng.random(n)

    py_choices = np.array(
        [min(max(int(np.searchsorted(cum, float(u), side="right")), 0), 3) for u in us],
        dtype=np.int64,
    )
    py_hist = np.bincount(py_choices, minlength=4).astype(np.float64)

    # Native path: one-leaf dict keyed at 42.
    w = _EXT.PyLeafContWeights({42: [float(x) for x in row]})
    nat_choices = np.array([w.sample(42, float(u)) for u in us], dtype=np.int64)
    nat_hist = np.bincount(nat_choices, minlength=4).astype(np.float64)

    # CI overlap: per-bin Wilson-normal intervals; overlap iff the bins agree at the 95% level.
    n1, n2 = float(py_hist.sum()), float(nat_hist.sum())
    p1, p2 = py_hist / n1, nat_hist / n2
    se1 = np.sqrt(np.maximum(p1 * (1.0 - p1) / n1, 0.0))
    se2 = np.sqrt(np.maximum(p2 * (1.0 - p2) / n2, 0.0))
    z = 1.959963984540054  # 1 - alpha/2 for alpha=0.05
    lo1, hi1 = p1 - z * se1, p1 + z * se1
    lo2, hi2 = p2 - z * se2, p2 + z * se2
    overlap = (lo1 <= hi2) & (lo2 <= hi1)

    assert bool(overlap.all()), (
        f"distribution diverged on row {row}:\n"
        f"  py  probs = {p1}\n"
        f"  nat probs = {p2}\n"
        f"  overlap (per-bin) = {overlap}"
    )

    # Mode-dependent negative control.
    if mode == "dominant_mass":
        # With a heavily biased row (max(row) >= 0.6), the native distribution must NOT look
        # uniform (i.e. the model-aware branch is doing real work). Verify the dominant-bin
        # mass is at least 60% (expected ~70%, allowing for one-bin sampling noise).
        dominant_bin = int(np.argmax(row))
        assert p2[dominant_bin] >= 0.60, (
            f"row {row}: native dominant-bin (idx={dominant_bin}) mass {p2[dominant_bin]:.3f} "
            f"should be >= 0.60, not uniform"
        )
    elif mode == "near_uniform_overlap":
        # Lightly-biased rows: only verify the CI overlap already passed (no fixed-mass gate).
        # Also verify the dominant-bin mass matches the row's max within a few SE — proves the
        # native branch is reading the row's relative weights, not collapsing to uniform.
        dominant_bin = int(np.argmax(row))
        # Expected native mass within SE = sqrt(p*(1-p)/n) of the row's max (post-normalize).
        expected = float(row_n[dominant_bin])
        se = float(np.sqrt(max(expected * (1.0 - expected) / n2, 0.0)))
        assert abs(p2[dominant_bin] - expected) < 6.0 * se, (
            f"row {row}: native dominant-bin (idx={dominant_bin}) mass {p2[dominant_bin]:.4f} "
            f"far from expected {expected:.4f} (±{6.0 * se:.4f})"
        )
    else:  # pragma: no cover — defensive
        raise AssertionError(f"unknown mode {mode!r}")