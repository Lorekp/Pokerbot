"""Turnkey CONTROL-VARIATE realistic-tier cash measurement — the variance-reduced "is the search hero
superhuman vs a realistic field?" number, in ~7x fewer expensive search-decks than a naive rotated run.

Why this script exists
----------------------
The decisive open question for the cash engine is the search hero's ABSOLUTE bb/100 vs the REALISTIC
calibrated tiers (recreational / weak / solid-reg), not the soft IRC caricature. A naive seat-rotated run
needs hundreds of ~1.5s/decision search-decks to tighten the CI; the control-variate decomposition
(:mod:`pokerbot.eval.cv_absolute`) pairs the expensive search hero with a CHEAP blueprint baseline under
common random numbers and pins the baseline mean with many cheap decks, cutting the expensive-deck count by
``Var(search)/Var(search-blueprint)`` (~7x at the realistic CRN correlation ~0.94).

Design choices
--------------
* FIXED iteration counts (``--search-iterations`` / ``--baseline-iterations``), not a wall-clock budget:
  per ``_build_pluribus_factory``'s docstring, a fixed count makes each solve a pure function of (public
  state, opponent model, config) -> the RESULT is independent of machine load. So this measurement is
  ROBUST to CPU contention (contention changes only wall-time), which matters on a shared box.
* The baseline is the SAME pluribus engine + blueprint at ``--baseline-iterations 1`` (~the blueprint root
  strategy, negligible search cost) -> maximal CRN correlation with the search hero -> maximal VR.
* Per-deck + per-field progress logging (heartbeat, ETA) is built in (the standing "log non-trivial jobs"
  rule) so "stalled vs slow" is visible instantly.

This reuses ``scripts/eval_exploitation_battery.py`` factory builders unchanged. Run when CPU is available:

    python scripts/eval_cv_realistic.py --paired-decks 40 --cheap-decks 120 \
        --search-iterations 80 --blueprint-snapshot <bp.pkl> --out runs/cv_realistic/summary.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

# NOTE on the preflop encoder: do NOT force POKERBOT_PREFLOP_LOSSLESS here. The serve env is
# SNAPSHOT-DEPENDENT: bp6max_dcfrvr_iter200 has NO `abstraction.preflop_lossless` flag (its own non-lossless
# encoding -> correct served at the DEFAULT, which is why the search hero wins vs recreational), whereas
# bp6max_lossless2 sets preflop_lossless=True. Forcing =1 on a dcfrvr run mis-fingerprints the working search
# hero. The blueprint-direct baseline's 100% adapter MISS was this same encoding mismatch (dcfrvr table fed to
# a lossless-expecting TabularBlueprintAdapter), NOT a missing env -> use --baseline iters (the search path)
# which keys the dcfrvr table correctly. If you ever run with a lossless2 snapshot, set the env to match IT.

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _load_battery():
    spec = importlib.util.spec_from_file_location("ebatt", ROOT / "scripts" / "eval_exploitation_battery.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


class _BlueprintHero:
    """Blueprint-DIRECT baseline hero (the high-correlation control variate for the CV measurement): plays
    the SAME blueprint the search hero is built on, via TabularBlueprintAdapter, with NO search -> cheap AND
    maximally correlated (the search hero = this blueprint + depth-limited refinement). Implements the
    run_field_absolute_rotated hero interface: callable(state)->Action + hero_seat + reseed + on_new_hand +
    a no-op set_profile (the baseline never exploits). The adapter is INJECTED so the shim is unit-testable
    without loading a 66MB blueprint."""

    def __init__(self, adapter, seed: int) -> None:
        self.adapter = adapter
        self.rng = __import__("random").Random(seed)
        self.hero_seat = 0
        self.profile = None

    def set_profile(self, profile) -> None:   # the baseline arm never exploits -> no-op (interface parity)
        self.profile = profile

    def reseed(self, s: int) -> None:
        self.rng = __import__("random").Random(s)

    def on_new_hand(self) -> None:
        pass

    def __call__(self, state):
        import numpy as _np

        from pokerbot.search.state_bridge import game_state_to_env
        from pokerbot.training.self_play import _index_to_action as _i2a

        seat = state.action_seat
        try:
            env = game_state_to_env(state, seat)
            p = _np.asarray(self.adapter.blueprint(env, seat), dtype=float)
            s = float(p.sum())
            if s > 0 and _np.isfinite(p).all():
                idx = self.rng.choices(range(len(p)), weights=(p / s).tolist())[0]
                return _i2a(state, seat, int(idx))
        except Exception:  # noqa: BLE001 — any bridge/lookup failure -> safe legal fallback (fold idx 0)
            pass
        from pokerbot.training.self_play import _index_to_action as _i2a
        return _i2a(state, seat, 0)


def _blueprint_baseline_factory(snapshot_path: str, seed: int):
    """Factory(*, exploit) -> _BlueprintHero on the SAME snapshot as the search hero (NOT BlueprintBot's
    lossless2 default — that is a different blueprint + a strict stamp check). Loaded once."""
    import pickle

    from pokerbot.search.tabular_blueprint_adapter import TabularBlueprintAdapter

    with open(snapshot_path, "rb") as f:
        raw = pickle.load(f)
    snap = raw["strat_sum"] if isinstance(raw, dict) and "strat_sum" in raw else raw
    adapter = TabularBlueprintAdapter(snap, client_bet_sizing=True)
    print(f"blueprint-direct baseline: {len(snap)} infosets (shared snapshot, no search)", flush=True)

    def factory(*, exploit: bool):
        return _BlueprintHero(adapter, seed)

    return factory


def _factory_args(snapshot: str, *, iterations: int, seed: int, leaf_select: str, search_preflop: bool = False):
    """A minimal args namespace carrying exactly the attributes _build_pluribus_factory reads."""
    return SimpleNamespace(
        blueprint_snapshot=snapshot,
        leaf_select=leaf_select,
        time_budget_ms=10 ** 9,        # unbound: the FIXED iteration count governs cost (load-independent)
        root_mixer=False,
        root_mixer_unopened=False,
        no_aivat_leafq=True,           # rotated never consumes the AIVAT leaf-Q -> skip (faster, bit-identical)
        native=None,                   # auto: Rust core iff it imports + self-checks
        frozen_br=False,
        iterations=iterations,
        seed=seed,
        search_preflop=search_preflop,  # deployed-config parity (poker_bot.py runs search_preflop=ON)
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--field", default="real_recreational_table,real_weak_table,real_solid_table",
                    help="comma list of HOMOGENEOUS calibrated tiers (recreational first = headline lands first)")
    ap.add_argument("--blueprint-snapshot", required=True, help="pluribus DCFR+VR blueprint .pkl")
    ap.add_argument("--paired-decks", type=int, default=40, help="EXPENSIVE search decks (CRN-paired)")
    ap.add_argument("--cheap-decks", type=int, default=120, help="cheap blueprint-only decks (disjoint)")
    ap.add_argument("--search-iterations", type=int, default=80, help="FIXED CFR iters for the search hero")
    ap.add_argument("--baseline", default="iters", choices=("blueprint", "iters"),
                    help="cheap CV baseline: 'iters' = pluribus at --baseline-iterations (the WORKING search "
                         "path; DEFAULT). 'blueprint' = blueprint-direct via TabularBlueprintAdapter — "
                         "CURRENTLY BROKEN: 100%% fingerprint MISS at eval states -> uniform/random play "
                         "(see measured-value-extraction-edge memory); do NOT use until the adapter-miss is fixed.")
    ap.add_argument("--baseline-iterations", type=int, default=8,
                    help="FIXED CFR iters for the cheap baseline when --baseline iters (8 >> 1's near-uniform)")
    ap.add_argument("--leaf-select", default="uniform", choices=("uniform", "gadget"))
    ap.add_argument("--search-preflop", action="store_true",
                    help="DEPLOYED-CONFIG PARITY: solve the preflop too (poker_bot.py runs this ON). With the "
                         "lossless2 snapshot this is the de-exploited deployed base — use it to confirm the "
                         "engine is robust vs solid_reg (LBR §4f predicts ~0, not the dcfrvr -324).")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--arm", default="off", choices=("off", "on"), help="off=pure base (headline), on=+exploit")
    ap.add_argument("--out", default="runs/cv_realistic/summary.json")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2),
                    help="OFF-arm process pool size (POKERBOT_ROTATED_WORKERS). Default cpu-2. "
                         "Each worker loads a full blueprint snapshot (~watch RAM); lower if memory-bound.")
    ap.add_argument("--resume", action="store_true",
                    help="resume from per-deck OFF checkpoints under --checkpoint-dir (only the missing decks "
                         "are recomputed; the aggregate equals an uninterrupted run). Safe on a fresh dir.")
    ap.add_argument("--checkpoint-dir", default=None,
                    help="dir for the per-pass OFF checkpoint streams (default: <out-parent>/ckpt). Each "
                         "completed OFF deck is appended+fsync'd so a hard kill resumes, not restarts.")
    ap.add_argument("--dry-run", action="store_true", help="build factories + print the plan, run no decks")
    args = ap.parse_args()

    # Fan the OFF arm across the CPU. Set process-wide BEFORE the factories are built so all three CV
    # passes (search@seed, base@seed, base@seed+777) read it inside run_field_absolute_rotated.
    os.environ["POKERBOT_ROTATED_WORKERS"] = str(max(1, args.workers))

    batt = _load_battery()
    from pokerbot.eval.cv_absolute import decks_for_target_ci, run_cv_absolute

    fields = [f.strip() for f in args.field.split(",") if f.strip()]
    print(f"[cv] fields={fields}  paired={args.paired_decks} cheap={args.cheap_decks}  "
          f"search_iters={args.search_iterations} baseline_iters={args.baseline_iterations}  arm={args.arm}",
          flush=True)

    search_factory = batt._build_pluribus_factory(
        _factory_args(args.blueprint_snapshot, iterations=args.search_iterations,
                      seed=args.seed, leaf_select=args.leaf_select,
                      search_preflop=args.search_preflop))
    if args.baseline == "blueprint":
        baseline_factory = _blueprint_baseline_factory(args.blueprint_snapshot, args.seed)
    else:
        baseline_factory = batt._build_pluribus_factory(
            _factory_args(args.blueprint_snapshot, iterations=args.baseline_iterations,
                          seed=args.seed, leaf_select=args.leaf_select))
    print(f"[cv] both factories built (search + {args.baseline} baseline)", flush=True)

    if args.dry_run:
        print("[cv] --dry-run: factories built OK; skipping deck runs.", flush=True)
        return

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    ckpt_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else out.parent / "ckpt"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    print(f"[cv] workers={args.workers} (POKERBOT_ROTATED_WORKERS)  checkpoints->{ckpt_dir}  "
          f"resume={args.resume}", flush=True)
    results = {}
    for fi, fname in enumerate(fields):
        t0 = time.time()
        print(f"\n[cv] === field {fi + 1}/{len(fields)}: {fname} ===", flush=True)
        # Calibrated tiers (real_*_table) are built by build_calibrated_field_bot, NOT the default build_bot.
        builder = batt.build_calibrated_field_bot if batt.field_is_calibrated(fname) else None
        res, meta = run_cv_absolute(
            batt.field_spec(fname),
            search_factory=search_factory, baseline_factory=baseline_factory,
            paired_decks=args.paired_decks, cheap_decks=args.cheap_decks,
            seed=args.seed, arm=args.arm,
            allin_ev=True, n_seats=6, bot_builder=builder,
            progress_every=max(1, args.paired_decks // 10),
            # arm=="off" => off_only skips the discarded ON arm (~2x on the expensive pass). MUST be
            # False for arm=="on" (cv_absolute reads on_deck_seat_means there).
            off_only=(args.arm == "off"),
            checkpoint_dir=str(ckpt_dir), resume=args.resume, field_tag=fname,
        )
        dt = time.time() - t0
        cv_lo, cv_hi = res.ci95
        nv_lo, nv_hi = res.naive_ci95
        print(f"[cv] {fname}: CV mean={res.mean * 100:+.1f} bb/100  CI=[{cv_lo * 100:+.1f}, {cv_hi * 100:+.1f}]  "
              f"(naive [{nv_lo * 100:+.1f}, {nv_hi * 100:+.1f}])", flush=True)
        print(f"[cv]   var_reduction={res.var_reduction:.1f}x  corr={res.corr:.3f}  beta={res.beta:.2f}  "
              f"baseline_paired={meta['baseline_paired_mean_bb'] * 100:+.1f} cheap={meta['baseline_cheap_mean_bb'] * 100:+.1f} bb/100  "
              f"[{dt:.0f}s]", flush=True)
        verdict = "POSITIVE (CI excludes 0)" if cv_lo > 0 else (
            "NEGATIVE (CI excludes 0)" if cv_hi < 0 else "INCONCLUSIVE (CI spans 0)")
        print(f"[cv]   verdict: {verdict}", flush=True)
        results[fname] = {
            "cv_mean_bb_per_100": res.mean * 100, "cv_ci95_bb_per_100": [cv_lo * 100, cv_hi * 100],
            "cv_se_bb_per_100": res.se * 100, "naive_ci95_bb_per_100": [nv_lo * 100, nv_hi * 100],
            "var_reduction": res.var_reduction, "corr": res.corr, "beta": res.beta,
            "decks_paired": res.decks_paired, "decks_cheap": res.decks_cheap,
            "verdict": verdict, "elapsed_s": dt, **meta,
        }
        out.write_text(json.dumps({"fields": results, "config": vars(args)}, indent=2), encoding="utf-8")
        print(f"[cv] wrote {out}", flush=True)


if __name__ == "__main__":
    main()
