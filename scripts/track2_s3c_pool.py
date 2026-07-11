#!/usr/bin/env python3
"""Track-2 gate S3(c) -- FIELD NO-REGRESSION: served-WITH-search vs the CoinPoker-mirror pool.

Registered gate (artifacts/track2_search_certification_design.md sec C, S3(c)):
    "field no-regression: CRN-paired vs the CoinPoker-mirror pool (strength-gate machinery),
     served-with-search not worse than the +240.5 bb/100 reference at CI."

The +240.5 bb/100 reference (blueprint_v4_arm2_amendment.md:1446-1452) was the RAW deployed
blueprint (search OFF) as hero, measured by blueprint_ranking.rank_vs_pool: seat-rotated duplicate
(run_field_absolute_rotated, off_only) vs the population-weighted default_pool_roster on CRN deck
seeds (deck seed = pure fn of (seed, deck_index)). This driver reproduces that EXACT harness but adds
a SECOND hero arm that routes every decision through the DEPLOYED depth-limited search policy, on the
SAME deck seeds -> CRN-paired per-deck (search - raw) delta + 95% CI is the PRIMARY read; the raw arm's
absolute reproduces the +240.5 reference (context) and the search arm's absolute is the S3(c) number.

Both arms are built from engine_ab_h2h's single-source-of-truth _DEPLOYED_BASELINE via its _build_policy:
    * search arm = spec {}            -> the EXACT deployed search object (== S3(a) arm_a, S1/S2 config)
    * raw    arm = spec {"search": false} -> the matched no-search comparator (== S3(a) arm_b)
so S3(c)'s two arms are bit-identical to the objects the other S3/S1/S2 gates certified.

Additive-only: a new script that READS the eval harness + engine_ab_h2h; touches no engine/solver code.
Never RAYON_NUM_THREADS=1. POKERBOT_* env is baked (via importing engine_ab_h2h first) before any
pokerbot serve import.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_HERE, _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Importing engine_ab_h2h FIRST bakes the deployed EMD-K160 serve preamble (its module-top setdefaults)
# before any pokerbot serve import -- the same regime the +240.5 reference and every other gate use.
import engine_ab_h2h as _eab  # noqa: E402  (side-effect: env preamble + numpy/jax import)

import numpy as np  # noqa: E402

_DEFAULT_BLUEPRINT = os.path.join(_ROOT, "checkpoints", "serve_blueprint_6max.pkl")
_REFERENCE_BB100 = 240.5  # deployed RAW-blueprint vs pool (blueprint_v4_arm2_amendment.md:1446)


# =================================================================================================
# The search hero: mirrors blueprint_ranking.TabularBlueprintAgent (the raw-arm hero used for +240.5)
# but swaps the pure-tabular adapter for the DEPLOYED search policy fn (env, seat -> prob vector).
# Same GameState->env bridge + _index_to_action mapping, so the two arms differ ONLY in the policy.
# =================================================================================================
class SearchHeroAgent:
    def __init__(self, policy_fn, rng: random.Random | None = None):
        self.pf = policy_fn
        self.rng = rng or random.Random(0)
        self.decision_count = 0
        self.fallback_count = 0
        # Ensure native subgame is live in THIS process (matters for pool workers under spawn).
        try:
            from pokerbot.search import native_solver as _ns
            from pokerbot.search.pluribus_search import set_native_subgame
            if not _ns.native_subgame_available():
                _ns.enable()
            set_native_subgame(True)
        except Exception:
            pass

    def __call__(self, state):
        from pokerbot.search.state_bridge import game_state_to_env
        from pokerbot.training.self_play import _index_to_action as _i2a
        from pokerbot.search.pluribus_search import set_native_subgame

        self.decision_count += 1
        try:
            set_native_subgame(True)
            seat = state.action_seat
            env = game_state_to_env(state, seat)
            p = np.asarray(self.pf(env, seat), dtype=np.float64)
            s = float(p.sum())
            if s > 0.0 and np.isfinite(p).all():
                idx = self.rng.choices(range(len(p)), weights=(p / s).tolist())[0]
                return _i2a(state, seat, int(idx))
        except Exception:
            pass
        self.fallback_count += 1
        legal = list(state.legal_actions())
        return self.rng.choice(legal)

    # seat-rotated-harness hero contract (all no-ops for a per-decision policy).
    def reseed(self, seed: int) -> None:
        self.rng = random.Random(int(seed))

    def set_profile(self, *_a, **_k) -> None:
        return None

    def set_opponent_profiles(self, *_a, **_k) -> None:
        return None

    def set_exploit_enabled(self, *_a, **_k) -> None:
        return None

    def reset_hand(self, *_a, **_k) -> None:
        return None


def _make_raw_factory(snap, base_seed, root):
    """Raw comparator hero = engine_ab_h2h spec {"search": false} -> make_blueprint_policy_fn.
    Reuses the SAME policy object the search anchor falls back to (bit-identical to S3(a) arm_b)."""
    pf = _eab._build_policy({"search": False}, snap, base_seed, root)

    def factory(*, exploit: bool):
        return SearchHeroAgent(pf, random.Random(0))

    return factory


def _make_search_factory(snap, base_seed, root):
    """Search hero = engine_ab_h2h spec {} -> the EXACT deployed search object (== S3(a) arm_a)."""
    pf = _eab._build_policy({}, snap, base_seed, root)

    def factory(*, exploit: bool):
        return SearchHeroAgent(pf, random.Random(0))

    return factory


def _paired(a, b):
    """paired (b - a) bb/100 + 95% half-width from CRN per-deck seat-mean arrays (bb/hand units)."""
    a = np.asarray(a, float).ravel()
    b = np.asarray(b, float).ravel()
    n = min(a.size, b.size)
    a, b = a[:n], b[:n]
    d = b - a
    mean = float(d.mean())
    se = float(d.std(ddof=1) / math.sqrt(n)) if n > 1 else 0.0
    return (mean * 100.0, 1.96 * se * 100.0, float(a.mean()) * 100.0, float(b.mean()) * 100.0, n)


def _run_arm(hero_factory, villain_name, *, decks, seed, workers, tag, progress_every):
    from pokerbot.eval.blueprint_ranking import _combined_bot_builder
    from pokerbot.eval.exploitation_battery import run_field_absolute_rotated

    os.environ["POKERBOT_ROTATED_WORKERS"] = str(int(max(1, workers)))
    t0 = time.perf_counter()

    def _cb(done, total):
        if progress_every and done and done % progress_every == 0:
            el = time.perf_counter() - t0
            rate = done / el if el > 0 else 0.0
            eta = (total - done) / rate / 60 if rate > 0 else 0.0
            print(f"    [{tag}] deck {done}/{total}  {rate*60:.1f} d/min  ETA {eta:.1f}min",
                  flush=True)

    field = [(villain_name, None)] * 5
    res = run_field_absolute_rotated(
        field, decks=decks, seed=seed, hero_factory=hero_factory,
        allin_ev=True, learn=False, n_seats=6, button_seat=3,
        bot_builder=_combined_bot_builder, off_only=True,
        progress_every=0, progress_cb=_cb,
    )
    means = np.asarray(res["off_deck_seat_means"], dtype=np.float64)
    wall = time.perf_counter() - t0
    print(f"    [{tag}] vs {villain_name}: {float(means.mean())*100:+.1f} bb/100 "
          f"({means.size} decks, {wall:.0f}s, {wall/max(1,means.size):.2f}s/deck)", flush=True)
    return means, wall


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--blueprint", default=_DEFAULT_BLUEPRINT)
    ap.add_argument("--decks", type=int, default=600, help="distinct deals per arm/archetype (each = 6 rotations)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4, help="POKERBOT_ROTATED_WORKERS for the OFF arm (never sets rayon=1)")
    ap.add_argument("--archetypes", default="", help="comma list to restrict roster (default = full default_pool_roster)")
    ap.add_argument("--out-dir", default=os.path.join(_ROOT, "artifacts", "track2_s3c"))
    ap.add_argument("--out", default="", help="final verdict json (default <out-dir>/s3c_seed<seed>.json)")
    ap.add_argument("--tag", default="")
    ap.add_argument("--progress-every", type=int, default=10)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    out = args.out or os.path.join(args.out_dir, f"s3c_seed{args.seed}{('_' + args.tag) if args.tag else ''}.json")

    # native availability report
    from pokerbot.search import native_solver as _ns
    if not _ns.native_subgame_available():
        _ns.enable()
    print(f"[s3c] native available={_ns.available()} subgame={_ns.native_subgame_available()} "
          f"rayon={_ns._EXT.rayon_num_threads()}  workers={args.workers}", flush=True)

    # roster
    from pokerbot.eval.blueprint_ranking import default_pool_roster
    roster = default_pool_roster()
    if args.archetypes:
        want = {x.strip() for x in args.archetypes.split(",") if x.strip()}
        roster = [(n, w) for n, w in roster if n in want]
        wsum = sum(w for _n, w in roster) or 1.0
        roster = [(n, w / wsum) for n, w in roster]  # renormalize the restricted roster
    print(f"[s3c] roster: {[(n, round(w, 3)) for n, w in roster]}", flush=True)

    # load blueprint snapshot (same loader shape as engine_ab_h2h)
    import pickle
    raw = pickle.load(open(args.blueprint, "rb"))
    snap = raw["strat_sum"] if isinstance(raw, dict) and "strat_sum" in raw else raw
    snap = {bytes(k): np.asarray(v, dtype=np.float64) for k, v in snap.items()}
    print(f"[s3c] blueprint {args.blueprint}: {len(snap):,} infosets", flush=True)

    raw_factory = _make_raw_factory(snap, args.seed, _ROOT)
    search_factory = _make_search_factory(snap, args.seed, _ROOT)

    per_arch = {}
    comb_raw = None
    comb_search = None
    t_start = time.perf_counter()
    for name, w in roster:
        print(f"\n[s3c] === archetype {name} (w={w:.3f}) ===", flush=True)
        # RAW arm first (cheap); then SEARCH arm on the SAME seed -> CRN-paired per deck.
        raw_means, raw_wall = _run_arm(raw_factory, name, decks=args.decks, seed=args.seed,
                                       workers=args.workers, tag="raw", progress_every=args.progress_every)
        search_means, s_wall = _run_arm(search_factory, name, decks=args.decks, seed=args.seed,
                                        workers=args.workers, tag="search", progress_every=args.progress_every)
        n = min(raw_means.size, search_means.size)
        delta, ci, raw_abs, search_abs, npair = _paired(raw_means[:n], search_means[:n])
        per_arch[name] = {
            "weight": w, "n": npair,
            "raw_bb100": raw_abs, "search_bb100": search_abs,
            "delta_search_minus_raw": delta, "ci95": ci,
            "raw_wall_s": raw_wall, "search_wall_s": s_wall,
            "raw_means": [float(x) for x in raw_means[:n]],
            "search_means": [float(x) for x in search_means[:n]],
        }
        rm = raw_means[:n]; sm = search_means[:n]
        comb_raw = (w * rm) if comb_raw is None else comb_raw + w * rm
        comb_search = (w * sm) if comb_search is None else comb_search + w * sm
        print(f"    [pair] vs {name}: search {search_abs:+.1f} | raw {raw_abs:+.1f} | "
              f"delta(search-raw) {delta:+.1f} +-{ci:.1f} bb/100", flush=True)
        # incremental checkpoint after each archetype
        _write(out, args, roster, per_arch, comb_raw, comb_search, t_start, final=False)

    _write(out, args, roster, per_arch, comb_raw, comb_search, t_start, final=True)
    return 0


def _write(out, args, roster, per_arch, comb_raw, comb_search, t_start, *, final):
    result = {
        "gate": "S3(c) field no-regression (served-with-search vs CoinPoker-mirror pool)",
        "reference_bb100": _REFERENCE_BB100,
        "blueprint": args.blueprint, "decks": args.decks, "seed": args.seed,
        "roster": [(n, w) for n, w in roster],
        "per_archetype": {k: {kk: vv for kk, vv in v.items() if not kk.endswith("_means")}
                          for k, v in per_arch.items()},
        "wall_s": time.perf_counter() - t_start,
        "final": final,
    }
    if comb_raw is not None and comb_search is not None:
        delta, ci, raw_abs, search_abs, n = _paired(comb_raw, comb_search)
        lo, hi = delta - ci, delta + ci
        # gate: served-with-search not worse than the +240.5 reference at CI.
        # primary = paired (search - raw); the raw arm's absolute reproduces the reference.
        search_vs_ref = search_abs - _REFERENCE_BB100
        verdict = ("SEARCH NOT WORSE (paired CI lower bound >= 0)" if lo >= 0 else
                   ("SEARCH WORSE (paired CI entirely below 0)" if hi < 0 else
                    "INCONCLUSIVE (paired CI includes 0)"))
        result["combined"] = {
            "n": n,
            "raw_bb100": raw_abs, "search_bb100": search_abs,
            "delta_search_minus_raw": delta, "ci95": ci, "lo": lo, "hi": hi,
            "search_minus_reference_bb100": search_vs_ref,
            "verdict": verdict,
        }
        if final:
            # persist the CRN-paired per-deck combined arrays so extra seeds can be POOLED for a
            # tighter CI (concat raw_combined_means / search_combined_means across seed runs, re-pair).
            result["combined"]["raw_combined_means"] = [float(x) for x in np.asarray(comb_raw).ravel()]
            result["combined"]["search_combined_means"] = [float(x) for x in np.asarray(comb_search).ravel()]
        if final:
            print("\n[s3c] =============== S3(c) VERDICT ===============", flush=True)
            print(f"[s3c] raw vs pool     {raw_abs:+.1f} bb/100  (reference {_REFERENCE_BB100:+.1f})", flush=True)
            print(f"[s3c] search vs pool  {search_abs:+.1f} bb/100  "
                  f"(search - reference {search_vs_ref:+.1f})", flush=True)
            print(f"[s3c] paired delta(search - raw) {delta:+.1f} +-{ci:.1f}  [95% CI {lo:+.1f}..{hi:+.1f}]", flush=True)
            print(f"[s3c] VERDICT: {verdict}", flush=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=1)
    if final:
        print(f"[s3c] wrote {out}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
