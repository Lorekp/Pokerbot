"""Opponent-Population Best-Response League (PBT) — the self-play value-extraction trainer (v1).

Design = workflow w7wdil18k, the INVERSION: the EVOLVING POPULATION is the OPPONENT FIELD (a weighted mix
of IRC-calibrated leak tiers), NOT the hero. A "version" is the hero's EXPLOIT-OVERLAY CONFIG — the
value-sizing escalation gains the ``value_sizing`` overlay reads from the env (POKERBOT_VS_K_BIG/MID/CHECK).
Each generation:
  1. RATE every version by its CRN-paired exploit-over-base UPLIFT (``run_field_absolute``, HybridAgent
     hero) vs a field SAMPLED from the leak population, WITH the online per-seat profiling that IS the
     design's "play X games before rating" (the hero builds each villain's profile from observed play,
     bootstrapped from the IRC prior).
  2. PBT-SELECT the top configs + PERTURB them (explore/exploit).
  3. PSRO-EVOLVE the field: up-weight the leak tier the current best version handles WORST, so the
     population co-evolves to keep the hero robust across the realistic leak space.

Anti-Nash-drift BY CONSTRUCTION: the config tunes the EXPLOIT OVERLAY (RNR-capped, fires only vs leaky
stations), so a "drifted" config simply scores a LOWER uplift — it cannot stop extracting. Fitness vs the
FIXED IRC-calibrated leak tiers keeps it realistic (see memory pbt-selfplay-program / measured-value-
extraction-edge). v1 evolves the overlay on a fixed base; evolving the BASE policy (full self-play) is v2.

Each version eval runs in a SUBPROCESS (this script with --worker) so its env-config applies at the
module-level import of value_sizing. Logged per the standing rule (per-gen best/mean, throughput).

  driver: python scripts/train_pbt_league.py --generations 6 --pop 6 --hands 300 --field-hands 300
  worker: python scripts/train_pbt_league.py --worker --field recreational,recreational,weak_reg,weak_reg,solid_reg --hands 300 --seed 0
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# The evolvable exploit-overlay knobs (env var -> (default, perturb-sigma, clip-lo, clip-hi)). These ARE
# the value-sizing escalation gains the deployed overlay uses; gen-0 seeds at the deployed constants.
KNOBS = {
    "POKERBOT_VS_K_BIG":   (1.20, 0.30, 0.0, 3.0),
    "POKERBOT_VS_K_MID":   (0.35, 0.15, 0.0, 2.0),
    "POKERBOT_VS_K_CHECK": (0.50, 0.20, 0.0, 2.0),
    # The exploit CEILING (bayes_exploit.DEFAULT_MAX_ALPHA). The value-sizing gains SATURATE at this cap
    # (smoke: K_BIG 1.2==2.6), so the cap is the real value-extraction lever — bounded by the
    # counter_exploiter safety cell below (the RNR-cull gate: raise extraction only while non-counter-
    # exploitable). The deployed 0.7 seeds gen 0; the comment at bayes_exploit:44 invites raising toward
    # 0.85 "only if the BR-gap battery confirms headroom" — this league IS that experiment.
    "POKERBOT_MAX_ALPHA":  (0.70, 0.10, 0.40, 0.95),
}
# The value-extraction TARGET (the leaky station the value-sizing overlay prints against) and the SAFETY
# regs the overlay must NOT hurt (the RNR-safety side). A version's fitness MAXIMIZES the station uplift
# SUBJECT TO not damaging the regs — the mission (max value vs the soft field, bounded downside vs strong
# players). The PSRO field co-evolution up-weights whichever safety reg the current best config handles
# WORST, keeping the overlay robust across the realistic strong-opponent space.
TARGET = "calling_station"
# counter_exploiter FIRST: it is the adaptive adversary the exploit ceiling protects against, so it is the
# binding safety cell for the POKERBOT_MAX_ALPHA knob (raising the cap is only allowed while it stays
# non-counter-exploitable). lag/nit/tag are the static-reg downside guards.
SAFETY_REGS = ["counter_exploiter", "lag", "nit", "tag"]
SAFETY_PENALTY = 1.0          # bb/100 of fitness lost per bb/100 the overlay HURTS a reg (downside guard)


# ----------------------------------------------------------------------------------------------------
# WORKER: rate ONE version (its env-config is already set) vs a field -> uplift fitness
# ----------------------------------------------------------------------------------------------------

def _worker(args) -> int:
    # The self-contained ExploitHero path (run_archetype default) — no checkpoint, fast, and the value-
    # sizing overlay FIRES because seed_hud seeds the opponent's server HUD (wtsd/wsd) the overlay gates on.
    # The version's gains are read by value_sizing at import in THIS subprocess (env set by the driver).
    from pokerbot.eval.exploitation_battery import run_archetype
    out = {}
    for arch in [a.strip() for a in args.field.split(",") if a.strip()]:
        r = run_archetype(arch, hands=args.hands, seed=args.seed, seed_hud=True, progress_every=10**9)
        out[arch] = 100.0 * float(r["uplift_bb_per_hand"])
    print("PBT_FITNESS " + json.dumps(out), flush=True)
    return 0


# ----------------------------------------------------------------------------------------------------
# DRIVER: the PBT + PSRO loop
# ----------------------------------------------------------------------------------------------------

def _eval_version(cfg: dict, field: list[str], hands: int, seed: int, args) -> dict:
    """Spawn the worker with cfg in the env so value_sizing reads the version's gains at import."""
    env = dict(os.environ)
    for k, v in cfg.items():
        env[k] = repr(float(v))
    cmd = [sys.executable, os.path.abspath(__file__), "--worker", "--field", ",".join(field),
           "--hands", str(hands), "--seed", str(seed)]
    p = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=args.eval_timeout)
    for line in p.stdout.splitlines():
        if line.startswith("PBT_FITNESS "):
            return json.loads(line[len("PBT_FITNESS "):])
    raise RuntimeError(f"worker produced no fitness (exit {p.returncode}):\n{p.stdout[-800:]}\n{p.stderr[-800:]}")


def _sample_reg(weights: dict, rng) -> str:
    """Sample ONE safety reg to test this generation, weighted (PSRO up-weights the worst-handled reg)."""
    regs = list(weights)
    w = [max(0.0, weights[t]) for t in regs]
    tot = sum(w) or 1.0
    return _weighted_pick(regs, [x / tot for x in w], rng)


def _fitness(scores: dict) -> float:
    """Composite fitness = station value uplift MINUS a penalty for any reg the overlay HURTS (downside).
    Rewards max thin-value extraction vs the soft target while bounding damage vs strong players."""
    station = scores.get(TARGET, -1e9)
    harm = sum(max(0.0, -scores.get(r, 0.0)) for r in scores if r != TARGET)
    return station - SAFETY_PENALTY * harm


def _weighted_pick(items, weights, rng):
    r = rng.random()
    c = 0.0
    for it, wt in zip(items, weights):
        c += wt
        if r <= c:
            return it
    return items[-1]


def _perturb(cfg: dict, rng) -> dict:
    out = dict(cfg)
    for k, (_d, sigma, lo, hi) in KNOBS.items():
        out[k] = float(min(hi, max(lo, cfg[k] + rng.gauss(0.0, sigma))))
    return out


def _driver(args) -> int:
    import random

    rng = random.Random(args.seed)
    logdir = args.logdir
    os.makedirs(logdir, exist_ok=True)
    log = open(os.path.join(logdir, "pbt.log"), "a", encoding="utf-8")

    def emit(msg):
        line = f"[{int(time.time())}] {msg}"
        print(line, flush=True)
        log.write(line + "\n")
        log.flush()

    # Gen-0 population: the deployed config + perturbations.
    base = {k: d for k, (d, *_r) in KNOBS.items()}
    pop = [dict(base)] + [_perturb(base, rng) for _ in range(args.pop - 1)]
    reg_w = {r: 1.0 for r in SAFETY_REGS}
    emit(f"PBT league: pop={args.pop} gens={args.generations} hands={args.hands} target={TARGET} "
         f"safety={SAFETY_REGS} knobs={list(KNOBS)} | gen0 seed=deployed-config")

    best_cfg, best_fit = dict(base), -1e9
    for gen in range(args.generations):
        reg = _sample_reg(reg_w, rng)                      # the safety reg sampled this generation
        field = [TARGET, reg]
        emit(f"gen {gen}: target={TARGET} safety-reg={reg}  reg_w={ {r: round(reg_w[r],2) for r in SAFETY_REGS} }")
        fits = []
        for i, cfg in enumerate(pop):
            t0 = time.time()
            try:
                sc = _eval_version(cfg, field, args.hands, seed=1000 + gen, args=args)  # {station:u, reg:u}
            except Exception as exc:  # noqa: BLE001 — a dead version scores -inf, never kills the run
                emit(f"  v{i} eval FAILED ({exc})")
                sc = {TARGET: -1e9}
            fit = _fitness(sc)
            fits.append(fit)
            emit(f"  v{i} fit={fit:+.1f}  station={sc.get(TARGET,0.0):+.1f} {reg}={sc.get(reg,0.0):+.1f}  "
                 f"cfg={ {k.split('_')[-1]: round(cfg[k],2) for k in KNOBS} }  ({time.time()-t0:.0f}s)")
        order = sorted(range(len(pop)), key=lambda i: fits[i], reverse=True)
        top = order[0]
        if fits[top] > best_fit:
            best_fit, best_cfg = fits[top], dict(pop[top])
        emit(f"gen {gen} BEST v{top} fit={fits[top]:+.1f} "
             f"cfg={ {k.split('_')[-1]: round(pop[top][k],2) for k in KNOBS} } | run-best={best_fit:+.1f}")

        # PBT exploit/explore: bottom half copies a top-half config + perturbs.
        elite = order[: max(1, len(pop) // 2)]
        for i in order[len(pop) // 2:]:
            pop[i] = _perturb(pop[rng.choice(elite)], rng)

        # PSRO field co-evolution: probe the run-best config vs ALL safety regs; up-weight the WORST so
        # next gen's sampled safety reg drifts toward where the overlay is most dangerous (robustness).
        if args.psro and gen < args.generations - 1:
            try:
                probe = _eval_version(best_cfg, [TARGET] + SAFETY_REGS, args.field_hands,
                                      seed=2000 + gen, args=args)
                worst = min(SAFETY_REGS, key=lambda r: probe.get(r, 0.0))
                reg_w[worst] *= 1.5
                emit(f"  psro probe: { {r: round(probe.get(r,0.0),1) for r in SAFETY_REGS} } "
                     f"-> up-weight worst '{worst}' -> w={reg_w[worst]:.2f}")
            except Exception as exc:  # noqa: BLE001
                emit(f"  psro probe failed ({exc}); reg weights unchanged")

    result = {"target": TARGET, "best_cfg": best_cfg, "best_fitness_bb_per_100": best_fit,
              "reg_weights": reg_w, "deployed_seed": base}
    with open(os.path.join(logdir, "pbt_result.json"), "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)
    emit(f"DONE: best fitness {best_fit:+.1f} bb/100 cfg={best_cfg} -> {logdir}/pbt_result.json")
    log.close()
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--worker", action="store_true", help="internal: rate one version (env-config preset)")
    ap.add_argument("--field", default="recreational,recreational,weak_reg,weak_reg,solid_reg")
    ap.add_argument("--hands", type=int, default=300)
    ap.add_argument("--field-hands", type=int, default=200, help="hands per PSRO per-tier probe")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pop", type=int, default=6)
    ap.add_argument("--generations", type=int, default=6)
    ap.add_argument("--psro", action="store_true", default=True)
    ap.add_argument("--no-psro", dest="psro", action="store_false")
    ap.add_argument("--eval-timeout", type=int, default=3600)
    ap.add_argument("--logdir", default="runs/pbt_league")
    args = ap.parse_args()
    return _worker(args) if args.worker else _driver(args)


if __name__ == "__main__":
    sys.exit(main())
