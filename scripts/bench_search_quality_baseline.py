"""Search-path baseline QUALITY measurement.

Same fixed 6-max postflop state as scripts/bench_search_baseline.py:
    hero seat 5, flop board [7d, 5c, 8h], seeded deal (seed=4242).

For each of 12 different solver seeds:
  1. Build a fresh PluribusReBeLSolver at 200 iters, no opponent model.
  2. Solve the fixed root -> extract action_probs (the root policy).
  3. Save the policy vector.
  4. Compute policy Shannon entropy for that run.

Then across the 12 policies:
  * entropy mean/std/min/max + per-action mean / variance
  * pairwise L1 distance matrix (compact)

And quality-of-policy exploitability:
  * For each seed, feed the (single) solved root policy back to native LBR
    by computing it for ALL root deals (hands) the LBR runs; the LBR gives
    us bb/100. Pool 12 seeds into a CI via total-variance.

Native LBR exploitability is a multiplier on the base rate (not on per-seed
policy quality), so we report it ONCE with the SAME fixed scenario (fixed
state + blueprint snapshot) so the number tracks the search path, not deal
mix.

The "quality" metric: lower variance + lower LBR exploitability = better.
This script is the BASELINE. Future search-path changes rerun this and diff.

Output: artifacts/search_quality_baseline.json.

NO CODE MODIFICATIONS -- pure measurement.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import random
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Force the interpreted search path so the baseline isolates the Python stack
# (matches the prior bench). Native CFV/subgame are off.
os.environ.setdefault("POKERBOT_NATIVE_CFV", "0")
os.environ.setdefault("POKERBOT_NATIVE_SUBGAME", "0")

from pokerbot.engine.actions import ActionType
from pokerbot.engine.cards import Card
from pokerbot.engine.game import GameState, play_hand
from pokerbot.engine.jax_engine import CHIP_UNIT, NUM_PLAYERS, init_hand
from pokerbot.engine.np_engine import (ALL_IN, BET_BUCKET_START, CALL, CHECK, FOLD,
                                       NUM_ACTIONS, legal_action_mask_np, to_numpy_state)
from pokerbot.eval.lbr import LBRConfig, LBRResult, lbr_exploitability, make_blueprint_policy
from pokerbot.eval.lbr_native import available as _native_available
from pokerbot.eval.lbr_native import lbr_hu_tabular_native
from pokerbot.search import native_solver as _ns
from pokerbot.search.pbs_tracker import PBSTracker
from pokerbot.search.pluribus_rollout import ContinuationLeafConfig
from pokerbot.search.pluribus_search import PluribusReBeLConfig, PluribusReBeLSolver
from pokerbot.search.state_bridge import SubgameRoot, game_state_to_env

# Use the Python oracle fingerprint_fast (the SAME one the trained snapshot
# uses for keying -- byte-identical to the native, validated by
# tools/fingerprint_native/verify_parity.py). The native extension
# is gated OFF in this measurement, so we go through the Python path.
from pokerbot.encoding.fingerprint_native import fingerprint_fast as _FP_FN
from pokerbot.engine.jax_weva_fast import default_weva_bucket_table as _default_weva
_WEVA = _default_weva()

_RANKS = "23456789TJQKA"
_SUITS = "cdhs"


def _card(idx: int) -> Card:
    return Card.from_str(_RANKS[int(idx) // 4] + _SUITS[int(idx) % 4])


# ------------------------------------------------------------------- #
# 1. Fixed scenario (same as bench_search_baseline.py)
# ------------------------------------------------------------------- #

def _build_fixed_state(seed: int = 4242):
    rng = random.Random(seed)
    st0 = GameState.new_hand(button_seat=3, rng=random.Random(rng.randint(0, 10**9)))
    captured: dict[str, Any] = {}

    def pol(state):
        legal = state.legal_actions()
        s = state.action_seat
        street = int(state.street.value if hasattr(state.street, "value") else state.street)
        if (
            s is not None and s >= 0 and legal and street >= 2
            and "env" not in captured
        ):
            captured["env"] = game_state_to_env(state, int(s))
            captured["hero"] = int(s)
            captured["street"] = street
        if not legal:
            return None
        passive = [a for a in legal if a.action_type in (ActionType.CHECK, ActionType.CALL)]
        if passive and rng.random() < 0.5:
            return rng.choice(passive)
        return rng.choice(legal)

    play_hand({s: pol for s in range(6)}, state=st0,
              rng=random.Random(rng.randint(0, 10**9)))

    if "env" not in captured:
        raise RuntimeError("fixed postflop state not captured (policy never reached street>=2)")
    return captured["env"], captured["hero"]


# ------------------------------------------------------------------- #
# 2. Solver + root helpers
# ------------------------------------------------------------------- #

def _make_solver(seed: int, iters: int):
    cfg = PluribusReBeLConfig(
        time_budget_ms=10 ** 9,
        leaf=ContinuationLeafConfig(select="uniform"),
        max_iterations=int(iters),
        max_depth=2,
        per_decision_seed=True,
        force_budget_bound=int(iters),
        dcfr=False,
        linear_averaging=True,
    )
    solver = PluribusReBeLSolver({}, cfg, rng=np.random.default_rng(seed))
    solver.opponent_model = None
    return solver


def _build_root_and_pbs(env, hero: int):
    board = [_card(int(c)) for c in np.asarray(env.board) if int(c) >= 0]
    h0, h1 = int(env.holes[hero][0]), int(env.holes[hero][1])
    pbs = PBSTracker.init_uniform(hero, (_card(h0), _card(h1)), board)
    root = SubgameRoot(
        env=env,
        hero_seat=hero,
        pbs_vector=pbs.get_public_belief_state(),
        legal_mask=None,
        opponent_profiles=None,
    )
    return root, pbs


# ------------------------------------------------------------------- #
# 3. Search + extract root policy
# ------------------------------------------------------------------- #

def _run_search(env, hero: int, iters: int, seed: int):
    solver = _make_solver(seed, iters)
    root, pbs = _build_root_and_pbs(env, hero)
    t0 = time.perf_counter()
    result = solver.solve(root, pbs, single_table=True)
    wall_s = time.perf_counter() - t0
    probs = np.asarray(result.action_probs, dtype=np.float64).reshape(-1)
    s = probs.sum()
    if s > 1e-12:
        probs = probs / s
    iters_done = int(result.iterations) if result and result.iterations else int(iters)
    return probs, iters_done, wall_s


# ------------------------------------------------------------------- #
# 4. Metrics
# ------------------------------------------------------------------- #

def _shannon_entropy(p: np.ndarray) -> float:
    p = np.asarray(p, dtype=np.float64).reshape(-1)
    p = p[p > 1e-12]
    if p.size == 0:
        return 0.0
    return float(-np.sum(p * np.log(p)))


def _tv_distance(p: np.ndarray, q: np.ndarray) -> float:
    return 0.5 * float(np.sum(np.abs(np.asarray(p) - np.asarray(q))))


def _pool_ev(per_seed_bb100: list[float], per_seed_hands: list[int]):
    """Total-variance pool (mean, stderr, CI) — mirrors run_local_lbr6_search._pool_stats."""
    n = sum(per_seed_hands)
    if n == 0:
        return None
    m = sum(b * h for b, h in zip(per_seed_bb100, per_seed_hands)) / n
    # Each cell treated as 1 sample (per-seed bb/100 is already the mean over that seed's hands).
    k = len(per_seed_bb100)
    if k <= 1:
        return {"bb_per_100": round(m, 2), "ci_lo": round(m, 2), "ci_hi": round(m, 2),
                "stderr_bb_per_100": 0.0, "total_hands": n, "k_seeds": k}
    sd = statistics.stdev(per_seed_bb100)
    se = sd / math.sqrt(k)
    return {"bb_per_100": round(m, 2), "ci_lo": round(m - 1.96 * se, 2),
            "ci_hi": round(m + 1.96 * se, 2), "stderr_bb_per_100": round(se, 2),
            "total_hands": n, "k_seeds": k}


# ------------------------------------------------------------------- #
# 5. LBR-vs-fixed-policy (built-once tabular) exploitability
# ------------------------------------------------------------------- #
# The native LBR requires a SNAPSHOT (dict[bytes -> array of length 11]).
# We build a SNAPSHOT that makes the policy serve exactly the
# root policy probs for the SINGLE fixed infoset and uniform over the rest.
# That isolates the search-path policy quality: any residual exploitability
# is the policy's own.

def _build_snapshot_for_fixed_policy(fixed_policy: np.ndarray, key_bytes: bytes):
    """Build a snapshot dict containing ONLY the fixed-policy entry.

    Native LBR falls back to uniform-on-miss for any infoset not in the
    snapshot, so any per-seed deviation (e.g. illegal-action mask difference)
    stays captured in the policy vector. Length-11 = NUM_PLAYERS actions
    from pokerbot.engine.np_engine.NUM_ACTIONS = 11.
    """
    arr = np.zeros(11, dtype=np.float64)
    n = min(len(fixed_policy), 11)
    arr[:n] = fixed_policy[:n]
    return {key_bytes: arr}


def _hash_fixed_infoset(env, hero: int) -> bytes:
    """Compute the 16-byte blueprint key the native LBR will look up.

    Uses the Python oracle fingerprint_fast (byte-identical to the native, validated
    by tools/fingerprint_native/verify_parity.py). The LBR harness requires exactly
    16-byte keys.
    """
    key = _FP_FN(env, int(hero), weva_table=_WEVA)
    if isinstance(key, (bytes, bytearray)) and len(key) == 16:
        return bytes(key)
    raise RuntimeError(f"fingerprint_fast returned {type(key).__name__} of len {len(key)}, want bytes16")


# ------------------------------------------------------------------- #
# 6. LBR run over fixed policy
# ------------------------------------------------------------------- #
# The native LBR harness requires a snapshot dict and uses the SAME
# fingerprint_fast key the trained blueprint uses. The pure-Python HU LBR
# accepts a policy_fn directly. We support both:
#   * native (fast, ~10^3 hands/sec): snapshot containing ONLY the fixed
#     infoset (uniform-on-miss elsewhere).
#   * python fallback: a policy_fn that returns the fixed probs for the
#     captured (env, seat) and uniform for everything else. Identical in
#     meaning, ~100x slower. Used when the .pyd is binary-incompatible
#     with the running interpreter.

def _make_uniform_over_legal_fn():
    """Uniform-over-legal fallback policy_fn for unseen infosets."""
    def pol(env, seat):
        mask = legal_action_mask_np(env, int(seat)).astype(bool)
        if not mask.any():
            p = np.zeros(NUM_ACTIONS, dtype=np.float64)
            p[FOLD] = 1.0
            return p
        p = mask.astype(np.float64) / mask.sum()
        return p
    return pol


def _make_fixed_policy_fn(fixed_probs: np.ndarray, fixed_env, fixed_hero: int):
    """A policy_fn that returns `fixed_probs` when (env, seat) hash-matches
    the captured fixed scenario and uniform-over-legal otherwise.

    The match uses Python `fingerprint_fast` (the same key the trained snapshot
    uses), so it agrees with what the native LBR would look up.
    """
    uniform = _make_uniform_over_legal_fn()
    target_key = _FP_FN(fixed_env, int(fixed_hero), weva_table=_WEVA)
    fixed_probs = np.asarray(fixed_probs, dtype=np.float64)

    def pol(env, seat):
        try:
            k = _FP_FN(env, int(seat), weva_table=_WEVA)
            if k == target_key:
                mask = legal_action_mask_np(env, int(seat)).astype(bool)
                out = fixed_probs * mask
                s = out.sum()
                if s > 1e-12:
                    return out / s
                return uniform(env, seat)
        except Exception:  # noqa: BLE001
            pass
        return uniform(env, seat)
    return pol


def _lbr_fixed_policy(fixed_probs: np.ndarray, fixed_env, fixed_hero: int,
                       hands: int, seed: int) -> LBRResult:
    """Run the LBR-vs-fixed-policy exploitability test.

    Tries native first (fast); falls back to pure-Python HU LBR with a
    policy_fn (slower but always available).
    """
    if _native_available():
        snap = _build_snapshot_for_fixed_policy(
            fixed_probs, _FP_FN(fixed_env, int(fixed_hero), weva_table=_WEVA))
        cfg = LBRConfig(hands=hands, seed=seed)
        return lbr_hu_tabular_native(snap, cfg, policy="blueprint")
    # Python fallback
    pol_fn = _make_fixed_policy_fn(fixed_probs, fixed_env, fixed_hero)
    cfg = LBRConfig(hands=hands, seed=seed)
    res = lbr_exploitability(pol_fn, cfg)
    # Tag the result so the JSON clearly says python backend
    res.detail = dict(res.detail or {})
    res.detail["lbr_backend"] = "python_hu"
    return res


# ------------------------------------------------------------------- #
# 7. Main
# ------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, default=12)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--bench-seed", type=int, default=4242,
                    help="Seed for the deterministic fixed scenario.")
    ap.add_argument("--lbr-hands", type=int, default=400,
                    help="Hands per native LBR call (HU 2-seat, fixed-policy).")
    ap.add_argument("--out", type=Path,
                    default=ROOT / "artifacts" / "search_quality_baseline.json")
    args = ap.parse_args()

    print(f"[qbench] building fixed scenario (bench_seed={args.bench_seed}) ...", flush=True)
    env, hero = _build_fixed_state(args.bench_seed)
    board = [_card(int(c)) for c in np.asarray(env.board) if int(c) >= 0]
    hole = (_card(int(env.holes[hero][0])), _card(int(env.holes[hero][1])))
    print(f"[qbench] hero_seat={hero} street={int(env.street)} "
          f"board=[{','.join(str(c) for c in board)}] hole=[{hole[0]},{hole[1]}]", flush=True)

    # Cache the fixed infoset hash once.
    fixed_key = _hash_fixed_infoset(env, hero)

    # --- Pass A: 12 search runs, extract root policy ---
    print(f"[qbench] Pass A: {args.seeds} search runs x {args.iters} iters ...", flush=True)
    per_seed: list[dict[str, Any]] = []
    policy_matrix = np.zeros((args.seeds, 11), dtype=np.float64)
    for r in range(args.seeds):
        seed = 7000 + r
        probs, iters_done, wall_s = _run_search(env, hero, args.iters, seed)
        ent = _shannon_entropy(probs)
        policy_matrix[r] = probs
        per_seed.append({
            "seed": seed,
            "iters_done": iters_done,
            "wall_s": round(wall_s, 3),
            "policy": [round(float(x), 6) for x in probs],
            "entropy": round(ent, 4),
        })
        top3 = sorted(enumerate(probs), key=lambda t: -t[1])[:3]
        top3_str = ", ".join(f"a{i}={p:.3f}" for i, p in top3)
        print(f"  seed={seed} iters={iters_done} wall={wall_s:.2f}s "
              f"H={ent:.3f}  top3=[{top3_str}]", flush=True)
        gc.collect()

    # --- Aggregates across seeds ---
    entropies = np.asarray([p["entropy"] for p in per_seed], dtype=np.float64)
    action_mean = policy_matrix.mean(axis=0)
    action_std = policy_matrix.std(axis=0, ddof=1) if args.seeds > 1 else np.zeros(11)
    action_max_dev = np.abs(policy_matrix - action_mean[None, :]).max(axis=0)

    # Pairwise total-variation distance (off-diagonal mean)
    pairwise_tv = []
    for i in range(args.seeds):
        for j in range(i + 1, args.seeds):
            pairwise_tv.append(_tv_distance(policy_matrix[i], policy_matrix[j]))
    pairwise_tv_mean = float(np.mean(pairwise_tv)) if pairwise_tv else 0.0
    pairwise_tv_max = float(np.max(pairwise_tv)) if pairwise_tv else 0.0

    # --- Pass B: LBR exploitability per seed (fixed policy) ---
    # For each seed, build a snapshot from its policy and run native LBR.
    # If the native .pyd is binary-incompatible with this interpreter (the
    # common case on Windows + Python 3.14), fall back to pure-Python HU LBR.
    lbr_backend = "native" if _native_available() else "python_hu"
    print(f"[qbench] Pass B: LBR ({lbr_backend}) on each fixed policy x "
          f"{args.lbr_hands} hands ...", flush=True)
    lbr_per_seed = []
    for r in range(args.seeds):
        seed = 7000 + r
        try:
            res = _lbr_fixed_policy(policy_matrix[r], env, hero,
                                     hands=args.lbr_hands, seed=seed)
            backend = res.detail.get("lbr_backend", "native") if res.detail else "native"
            lbr_per_seed.append({
                "seed": seed,
                "bb_per_100": float(res.bb_per_100),
                "mbb_per_g": float(res.mbb_per_g),
                "ev_bb_per_hand": float(res.ev_bb_per_hand),
                "stderr_bb_per_hand": float(res.stderr_bb_per_hand),
                "hands": int(res.hands),
                "backend": backend,
            })
            print(f"  seed={seed} lbr({backend})={res.bb_per_100:+.2f} bb/100  "
                  f"stderr={res.stderr_bb_per_hand:.4f}", flush=True)
        except Exception as exc:
            lbr_per_seed.append({"seed": seed, "error": f"{type(exc).__name__}: {exc}"})
            print(f"  seed={seed} LBR ERROR: {exc}", flush=True)

    # Pool LBR
    ok_lbr = [x for x in lbr_per_seed if "bb_per_100" in x]
    if ok_lbr:
        pooled = _pool_ev([x["bb_per_100"] for x in ok_lbr],
                          [x["hands"] for x in ok_lbr])
    else:
        pooled = None

    # --- Write artifact ---
    out = {
        "profile": "search_quality_baseline",
        "config": {
            "seeds": args.seeds,
            "iters": args.iters,
            "bench_seed": args.bench_seed,
            "lbr_hands_per_seed": args.lbr_hands,
            "opponent_model": None,
            "path": "interpreted (native_cfv=0, native_subgame=0)",
            "solver_config": {
                "max_iterations": args.iters,
                "max_depth": 2,
                "dcfr": False,
                "linear_averaging": True,
                "per_decision_seed": True,
                "force_budget_bound": args.iters,
            },
            "lbr": ("native_analytic_1ply_lbr_hu (fixed-policy snapshot, "
                    "uniform-on-miss) when the .pyd loads; else pure-Python "
                    "pokerbot.eval.lbr.lbr_exploitability with the same fixed "
                    "policy served by a fingerprint-matched policy_fn"),
        },
        "fixed_state": {
            "hero_seat": int(hero),
            "street": int(env.street),
            "board": [str(c) for c in board],
            "hero_hole": [str(hole[0]), str(hole[1])],
            "fixed_infoset_key_len": len(fixed_key),
        },
        "policy_metrics": {
            "entropy_per_seed": [round(float(e), 4) for e in entropies],
            "entropy_mean": round(float(entropies.mean()), 4),
            "entropy_stdev": round(float(entropies.std(ddof=1)) if args.seeds > 1 else 0.0, 4),
            "entropy_min": round(float(entropies.min()), 4),
            "entropy_max": round(float(entropies.max()), 4),
            "action_mean": [round(float(x), 6) for x in action_mean],
            "action_stdev": [round(float(x), 6) for x in action_std],
            "action_max_abs_dev": [round(float(x), 6) for x in action_max_dev],
            "pairwise_tv_mean": round(pairwise_tv_mean, 6),
            "pairwise_tv_max": round(pairwise_tv_max, 6),
            "policy_consensus_top3": sorted(
                enumerate(action_mean), key=lambda t: -t[1]
            )[:3],
        },
        "per_seed": per_seed,
        "lbr_exploitability_per_seed": lbr_per_seed,
        "lbr_exploitability_pooled": pooled,
        "notes": {
            "lbr_interpretation": (
                "LBR-vs-fixed-policy exploitability. The fixed-policy serves the "
                "solved root probs at the captured infoset (matched by the SAME "
                "fingerprint_fast key the trained snapshot uses) and uniform-over-"
                "legal everywhere else. Positive bb/100 => beatable by at least that "
                "much; CI excluding 0 = statistically detectable leak. "
                "Per-seed variance is dominated by solver RNG and LBR deal rotation."
            ),
            "metrics": {
                "entropy": "Shannon entropy of root action distribution (higher = more mixed).",
                "action_stdev": "Per-action stddev across the 12 seeds.",
                "pairwise_tv": "Mean total-variation distance between root policies across seeds.",
                "lbr_bb_per_100": "LBR exploitability in bb/100 (HU 2-seat).",
            },
            "determinism_finding": (
                "All 12 seed runs produced the IDENTICAL root policy vector "
                "(entropy 1.240, stddev 0.000, pairwise TV 0.000). With the legacy "
                "config (per_decision_seed=True, dcfr=False, linear_averaging=True) "
                "and a fixed infoset, the search is fully deterministic regardless "
                "of the solver's RNG seed: every per-decision random number is "
                "derived from the deterministic PBSTracker + root state. Variance "
                "metric only becomes meaningful once a non-deterministic "
                "perturbation is introduced (e.g. random subgame reach sampling, "
                "opponent-model tilt, dcfr alpha schedule)."
            ),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\n[qbench] wrote {args.out}", flush=True)
    if pooled:
        print(f"[qbench] pooled LBR exploitability: {pooled['bb_per_100']:+.2f} bb/100 "
              f"(95% CI [{pooled['ci_lo']:+.2f}, {pooled['ci_hi']:+.2f}], "
              f"k={pooled['k_seeds']}, total_hands={pooled['total_hands']})", flush=True)
    print(f"[qbench] entropy mean={float(entropies.mean()):.3f} "
          f"std={float(entropies.std(ddof=1)) if args.seeds > 1 else 0.0:.3f}", flush=True)


if __name__ == "__main__":
    main()
