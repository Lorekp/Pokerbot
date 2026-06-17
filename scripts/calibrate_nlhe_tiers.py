"""Calibrate the NLHE skill-tier strength ladder against a near-Nash reference.

    !!! KNOWN-ILL-POSED vs a stronger-than-base reference (do not trust the output as-is). !!!
    Empirically (run_nlhetiers1/2): the SFT base is ~600 bb/100 BELOW the BlueprintBot field, so
    'tighten-up' leaks (overfold/passivity) REDUCE losses vs that field => leak cost is non-monotone
    and even flips POSITIVE at higher scale (intermediate cost went -84 -> +76 as scale rose). A sound
    strength ladder needs an at-or-below-base reference: measure EXPLOITABILITY, or play the leaked
    hero vs a field of FLAWLESS-BASE copies (self-play), not vs the near-Nash blueprint. Until that is
    built, ship the hand-set scale=1.0 defaults (skill_tiers.DEFAULT_PROFILES are the calibration).
    The CRN-paired machinery below is correct (superhuman cost == 0 exactly); only the REFERENCE field
    is wrong. Left in the repo as the scaffold for the corrected (self-play) version.


The NLHE analog of ``scripts/calibrate_practice_tiers.py`` (AoF). For each leaky tier
(beginner..semi_pro) it measures the tier's LEAK COST in bb/100 — the CRN-paired difference between
the tier-leaked hero and the flawless (superhuman) hero on identical decks vs the SAME ``BlueprintBot``
field (near-Nash). The pairing cancels the base-vs-field loss (the SFT base is hundreds of bb/100
below the blueprint on its own — that would swamp the leak signal) so the cost reflects only the
leaks. With ``--fit`` it bisects each tier's structural ``scale`` (the ``fit_weak_mixture.py`` pattern)
so the costs form a monotone ladder (beginner most negative -> ~0 at semi_pro), and writes
``data/player_profiles/nlhe_skill_tiers.json``.

professional/superhuman keep ``scale=1.0`` (≈0 structural leak by construction). advanced+ exploitation
is validated separately vs WEAK archetypes (the hybrid-overlay battery), not here — vs near-Nash the
overlay is do-no-harm by design, so this script measures the static leak ladder only.

Parallel across (tier, seed) cells via a process pool (one model load per worker), so the 64-vCPU
spot box is used. Optionally self-uploads the result JSON + a DONE marker to S3 so a babysitter can
poll. Run on AWS spot via the existing infra (the user launches it):

    POKERBOT_BETTING_ABSTRACTION=1 \
    POKERBOT_BLUEPRINT_SNAPSHOT=/path/bp6max_v2_iter200.pkl \
    python scripts/calibrate_nlhe_tiers.py --fit --hands 3000 --seeds 0-17 --allin-ev --workers 32
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import random
from dataclasses import replace
from pathlib import Path

os.environ.setdefault("POKERBOT_BETTING_ABSTRACTION", "1")

import numpy as np

OUT_PATH = Path("data/player_profiles/nlhe_skill_tiers.json")
LEAKY_TIERS = ("beginner", "intermediate", "advanced", "expert", "semi_pro")  # professional/superhuman fixed
HERO_SEAT = 0
BUTTON_SEAT = 3

# Per-worker globals (set once per process by the pool initializer — model load is the heavy bit).
_W: dict = {}


def _init_worker(checkpoint: str, device: str) -> None:
    os.environ.setdefault("POKERBOT_BETTING_ABSTRACTION", "1")
    from pokerbot.models.checkpoint_loader import load_pseudo_siamese_policy

    _W["model"] = load_pseudo_siamese_policy(checkpoint, device)
    _W["device"] = device


class TierHero:
    """The GTO-base policy degraded by a tier's leak profile (exploitation OFF). Callable as an engine
    policy ``hero(state) -> Action``; samples the leaked legal distribution (superhuman = argmax)."""

    def __init__(self, tier: str, scale: float | None, model, device: str = "cpu") -> None:
        from pokerbot.agent import HybridAgent
        from pokerbot.opponent import skill_tiers

        self._st = skill_tiers
        self.agent = HybridAgent(model, device=device, use_transformer=False)
        self.agent.set_exploit_enabled(False)
        base = skill_tiers.tier_profile(tier)
        self.profile = base if scale is None else replace(base, scale=float(scale))
        self.argmax_only = (tier == "superhuman")
        self.rng = random.Random(0)

    def on_new_hand(self) -> None:
        self.agent.reset_hand(HERO_SEAT)

    def __call__(self, state):
        from pokerbot.agent import _hero_hand_strength
        from pokerbot.training.self_play import _index_to_action, _legal_mask

        st = self._st
        seat = state.action_seat
        self.agent.enable_decision_capture(None)     # argmax play, capture probs
        self.agent(state)
        raw = self.agent.pop_decisions_raw()
        probs = (np.asarray(raw[-1][0], dtype=np.float64).reshape(-1) if raw
                 else np.ones(st.NUM_ACTIONS) / st.NUM_ACTIONS)
        legal = _legal_mask(state, seat)
        facing_bet = not bool(legal[st.CHECK])
        leaked = st.apply_leaks(probs, legal, profile=self.profile, facing_bet=facing_bet,
                                hand_strength=_hero_hand_strength(state, seat),
                                street=state.street.name.lower())
        idx = int(leaked.argmax()) if self.argmax_only else int(
            self.rng.choices(range(len(leaked)), weights=leaked.tolist())[0])
        return _index_to_action(state, seat, idx)


def _play_one(hero, villains, hand_seed, settle):
    from pokerbot.engine.game import GameState, play_hand

    hero.rng = random.Random(hand_seed)
    hero.on_new_hand()
    for v in villains:
        v.rng = random.Random(8000 + hand_seed)
    state = GameState.new_hand(button_seat=BUTTON_SEAT, rng=random.Random(hand_seed))
    policies = {s: (hero if s == HERO_SEAT else villains[s]) for s in range(6)}
    stl = (lambda state_, nb, _s=hand_seed: settle(state_, nb, 7777 + _s)) if settle else None
    _final, d = play_hand(policies, state=state, allin_settle=stl)
    return float(d.get(HERO_SEAT, 0.0))


def _measure_cell(args) -> float:
    """One (tier, scale, seed) cell: the tier's LEAK COST in bb/100 — the CRN-paired difference
    (tier-leaked hero - flawless/superhuman hero) on identical decks vs the SAME BlueprintBot field.

    Measuring absolute bb/100 vs a near-Nash field is hopeless: the SFT base loses ~hundreds of bb/100
    to the blueprint on its own, swamping the leak signal. The paired difference cancels that base loss
    (and most variance), isolating exactly what the tier's leaks cost. The flawless arm is 0 by
    construction, so the cost is <= 0 and orders cleanly by leak strength.
    """
    tier, scale, seed, hands, allin_ev = args
    from pokerbot.eval.bots import build_bot

    settle = None
    if allin_ev:
        from pokerbot.eval.exploitation_battery import _allin_ev_settle
        settle = _allin_ev_settle
    model, device = _W["model"], _W["device"]
    leaked = TierHero(tier, scale, model, device=device)            # arm A
    flawless = TierHero("superhuman", None, model, device=device)   # arm B (exact base, argmax)
    # One field per arm, reseeded identically each hand (_play_one) ⇒ common random numbers: same deck
    # + same field behavior for both arms, so the base-vs-field loss cancels in (leaked - flawless).
    va = [build_bot("blueprint", random.Random(7000 + seed)) for _ in range(6)]
    vb = [build_bot("blueprint", random.Random(7000 + seed)) for _ in range(6)]
    diffs: list[float] = []
    for h in range(hands):
        hand_seed = seed * 1_000_000 + h
        pnl_a = _play_one(leaked, va, hand_seed, settle)
        pnl_b = _play_one(flawless, vb, hand_seed, settle)
        diffs.append(pnl_a - pnl_b)
    return float(np.mean(diffs)) * 100.0 if diffs else 0.0


def measure_tier(pool, tier: str, scale: float | None, *, hands: int, seeds, allin_ev: bool) -> float:
    """Mean hero bb/100 across ``seeds`` (each seed a parallel cell)."""
    tasks = [(tier, scale, s, hands, allin_ev) for s in seeds]
    vals = list(pool.map(_measure_cell, tasks))
    return float(np.mean(vals)) if vals else 0.0


def bisect_scale(pool, tier: str, target_bb100: float, *, hands: int, seeds, allin_ev: bool,
                 lo: float = 0.0, hi: float = 2.0, iters: int = 6) -> tuple[float, float]:
    """Find the structural ``scale`` whose measured bb/100 ≈ ``target_bb100`` (more leak ⇒ more
    negative). Returns (scale, measured_bb100). Monotone: bb/100 decreases as scale increases."""
    base_val = measure_tier(pool, tier, 1.0, hands=hands, seeds=seeds, allin_ev=allin_ev)
    best = (1.0, base_val)
    print(f"[fit] {tier:<13} scale=1.0 -> {base_val:+.3f} (target {target_bb100:+.3f})", flush=True)
    for it in range(iters):
        mid = 0.5 * (lo + hi)
        val = measure_tier(pool, tier, mid, hands=hands, seeds=seeds, allin_ev=allin_ev)
        if abs(val - target_bb100) < abs(best[1] - target_bb100):
            best = (mid, val)
        print(f"[fit] {tier:<13} iter {it} scale={mid:.4f} -> {val:+.3f}", flush=True)
        if val > target_bb100:        # hero too strong (not leaky enough) -> increase scale
            lo = mid
        else:
            hi = mid
    return best


def _parse_seeds(spec: str) -> list[int]:
    if "-" in spec:
        a, b = spec.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in spec.split(",") if x != ""]


def _maybe_upload(payload_path: Path, args) -> None:
    if not args.s3_bucket:
        return
    import boto3

    s3 = boto3.client("s3", region_name=args.region)
    base = args.s3_prefix.rstrip("/")
    res_key = f"{base}/results/nlhe_skill_tiers.json"
    done_key = f"{base}/calibrate_DONE.marker"
    s3.upload_file(str(payload_path), args.s3_bucket, res_key)
    s3.put_object(Bucket=args.s3_bucket, Key=done_key, Body=b"done")
    print(f"uploaded s3://{args.s3_bucket}/{res_key} + DONE marker", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", default="checkpoints/policy_sft_prod6000.pt")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--hands", type=int, default=3000)
    ap.add_argument("--seeds", default="0-17", help="e.g. '0-17' or '0,1,2'")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--allin-ev", action="store_true", help="all-in EV settlement (lower variance)")
    ap.add_argument("--fit", action="store_true",
                    help="bisect scale to a monotone target ladder; else just measure at scale=1.0")
    ap.add_argument("--targets", default=None,
                    help="comma bb/100 targets for the leaky tiers (default: linear ramp from the "
                         "measured beginner deficit up to ~0 at semi_pro)")
    ap.add_argument("--out", default=str(OUT_PATH))
    ap.add_argument("--s3-bucket", default=None, help="upload result JSON + DONE marker here when set")
    ap.add_argument("--s3-prefix", default="", help="S3 key prefix (results/ + calibrate_DONE.marker under it)")
    ap.add_argument("--region", default="eu-central-1")
    args = ap.parse_args()

    if not Path(args.checkpoint).exists():
        raise SystemExit(f"checkpoint not found: {args.checkpoint}")
    seeds = _parse_seeds(args.seeds)
    print(f"calibrate: tiers={list(LEAKY_TIERS)} seeds={seeds[0]}-{seeds[-1]} hands={args.hands} "
          f"workers={args.workers} allin_ev={args.allin_ev} fit={args.fit}", flush=True)

    pool = cf.ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker,
                                  initargs=(args.checkpoint, args.device))
    try:
        kw = dict(hands=args.hands, seeds=seeds, allin_ev=args.allin_ev)
        # 1. natural ladder at scale=1.0 (skip the redundant pass when fitting — bisect measures it).
        measured = {}
        if not args.fit:
            for t in LEAKY_TIERS:
                measured[t] = measure_tier(pool, t, 1.0, **kw)
                print(f"[measure] {t:<13} bb/100 @ scale=1.0 = {measured[t]:+.3f}", flush=True)

        tiers_out: dict[str, dict] = {}
        if args.fit:
            # An anchor measure for the target ramp (beginner is the leakiest floor).
            floor = measure_tier(pool, "beginner", 1.0, **kw)
            if args.targets:
                targets = {t: float(v) for t, v in zip(LEAKY_TIERS, args.targets.split(","))}
            else:
                ramp = np.linspace(min(floor, -1.0), 0.0, len(LEAKY_TIERS) + 1)[:-1]
                targets = {t: float(ramp[i]) for i, t in enumerate(LEAKY_TIERS)}
            for t in LEAKY_TIERS:
                scale, val = bisect_scale(pool, t, targets[t], **kw)
                tiers_out[t] = {"scale": round(scale, 4), "measured_bb100": round(val, 3),
                                "target_bb100": round(targets[t], 3)}
                print(f"[fit] {t:<13} DONE scale={scale:.4f} bb/100={val:+.3f} "
                      f"(target {targets[t]:+.3f})", flush=True)
        else:
            tiers_out = {t: {"scale": 1.0, "measured_bb100": round(measured[t], 3)} for t in LEAKY_TIERS}
    finally:
        pool.shutdown(wait=True)

    payload = {
        "tiers": tiers_out,
        "meta": {
            "fitted": bool(args.fit),
            "reference": "BlueprintBot (bp6max_v2_iter200 near-Nash)",
            "hands": args.hands, "seeds": [seeds[0], seeds[-1]], "allin_ev": bool(args.allin_ev),
            "note": "professional/superhuman keep scale=1.0 (~0 structural leak); advanced+ "
                    "exploitation is validated separately vs weak archetypes.",
        },
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {out}", flush=True)
    _maybe_upload(out, args)


if __name__ == "__main__":
    main()
