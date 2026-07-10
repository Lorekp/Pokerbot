#!/usr/bin/env python3
"""Track-2 S2 self-consistency runner (MEASUREMENT ONLY).

Registered spec: artifacts/track2_search_certification_design.md  §C S2 (+ §B.4).
This is the v5 plan's G5, scoped honestly: re-solve >=200 reached rounds-2-4 subgames sampled
from H2H play; where the blueprint row has REAL coverage (non-uniform serve), the re-solve's
root policy must agree with the anchor within tol (KL <= 0.15 median); where the blueprint serves
its uniform-legal fallback, agreement is NOT required -- we record the divergence share as the
measured "search value surface" instead (the current blueprint is known under-converged postflop;
demanding fixed-point agreement with a weak anchor would fail a GOOD search).

Reuses S1 (scripts/track2_s1_ksweep.py) plumbing WHOLESALE:
  * the DEPLOYED EMD-K160 serve env preamble (native subgame ON);
  * CapturingHero self-play state capture (generalized: EVERY postflop decision node, not just
    multiway raised flops);
  * build_root / _PBS root construction (native_exploit_equiv);
  * build_solver + solve_fixed at K=2000 (the deployed-config re-solve).

Coverage HIT vs uniform-fallback MISS is read via the ONLY reliable signal: the lbr._BP_COVERAGE
per-street counters (POKERBOT_BP_COVERAGE=1, set in the preamble BEFORE any pokerbot import).
Per single blueprint lookup we bp_coverage_reset() -> serve -> bp_coverage_report(): exactly one of
hit/miss increments, telling us whether the served row was a trained row (HIT) or the uniform-legal
fallback (MISS).

Metrics:
  * per HIT state : KL(blueprint_row || search_policy) over the legal mask (1e-9 smoothing, both
    sides, then renormalized), plus L1 and argmax match.
  * per MISS state: search-policy entropy vs the uniform-legal policy (the "value surface" share:
    how much structure search injects where the blueprint is blind).
Gate (registered): median KL over HIT states <= 0.15.
NOTE the action-EV-within-25mbb alternative arm is intentionally NOT run here (KL is the primary
tol per the spec); it is recorded as not-run in the output meta.

Output (incremental, crash-safe): artifacts/track2_s2_selfconsistency.json
Progress log: artifacts/track2_s2_selfconsistency.progress.log

Env knobs:
  TRACK2_S2_SMOKE=1   tiny corpus (~12 states) + few hands, for a wiring smoke.
  TRACK2_S2_WANT=N    total distinct states to certify on (default 200).
  TRACK2_S2_HANDS=N   cap on self-play hands during capture (default 20000; smoke 200).
"""
from __future__ import annotations

import json
import os
import pickle
import random
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

# --- DEPLOYED EMD-K160 serve env — MUST be set BEFORE any pokerbot import ---
# (mirrors scripts/track2_s1_ksweep.py:52-66 == engine_ab_h2h.py:33-41; native subgame ON;
#  RAYON_NUM_THREADS deliberately NOT pinned). PLUS POKERBOT_BP_COVERAGE=1 so lbr._BP_COVERAGE
# records per-lookup hit/miss — the ONLY reliable uniform-fallback signal (spec §C S2 / B.4).
_FROZEN = str(REPO / "data" / "abstraction" / "emd_frozen_flop_turn_160_r400.emdfrzn")
for _k, _v in {
    "POKERBOT_BETTING_ABSTRACTION": "1",
    "POKERBOT_POSTFLOP_BUCKETS": "1",
    "POKERBOT_PREFLOP_LOSSLESS": "1",
    "POKERBOT_POSTFLOP_MODE": "emd",
    "POKERBOT_POSTFLOP_NUM_BUCKETS": "160",
    "POKERBOT_POSTFLOP_EMD_FROZEN": _FROZEN,
    "POKERBOT_EMD_FROZEN_MIRROR_FALLBACK": "1",
    "POKERBOT_NATIVE_SUBGAME": "1",
    "POKERBOT_BP_COVERAGE": "1",
    "JAX_PLATFORMS": "cpu",
}.items():
    os.environ.setdefault(_k, _v)

import numpy as np  # noqa: E402

# --------------------------------------------------------------------------- config
SMOKE = os.environ.get("TRACK2_S2_SMOKE", "0") not in ("0", "", "false", "no")
WANT = int(os.environ.get("TRACK2_S2_WANT", "12" if SMOKE else "200"))
MAX_HANDS = int(os.environ.get("TRACK2_S2_HANDS", "200" if SMOKE else "20000"))

K_RESOLVE = 2000            # deployed-config re-solve budget (S1 plumbing)
KL_TOL = 0.15              # registered gate: median KL over HIT states <= 0.15
EPS = 1e-9                 # KL / entropy smoothing (added both sides, then renormalized)
# Stratification targets (spec: ~40% flop / 35% turn / 25% river). Applied as a SAMPLE over the
# distinct captured states; shortfalls are logged, never fail (turn/river are rarer).
STRAT = {1: 0.40, 2: 0.35, 3: 0.25}
SEED = 4242
CAP_SEED = 77              # self-play capture seed (deterministic)

OUT = REPO / "artifacts" / "track2_s2_selfconsistency.json"
PROG = REPO / "artifacts" / "track2_s2_selfconsistency.progress.log"
BP_PATH = REPO / "checkpoints" / "serve_blueprint_6max.pkl"

_STREET_NAME = {1: "flop", 2: "turn", 3: "river"}
_T_START = time.time()


def log(msg: str):
    line = f"[{time.strftime('%F %T')}] [track2-s2] {msg}"
    try:
        with open(PROG, "a") as f:
            f.write(line + "\n")
    except Exception:
        pass
    print(line, flush=True)


# --------------------------------------------------------------------------- snapshot
def load_snapshot():
    raw = pickle.load(open(BP_PATH, "rb"))
    return raw["strat_sum"] if isinstance(raw, dict) and "strat_sum" in raw else raw


def as_bytes_keyed(snap):
    """make_blueprint_policy does snapshot.get(bytes_key); ensure bytes keys without a full copy
    when they are already bytes (the serve pkl can be large)."""
    try:
        k0 = next(iter(snap))
    except StopIteration:
        return snap
    if isinstance(k0, (bytes, bytearray)):
        return snap
    return {bytes(k): np.asarray(v) for k, v in snap.items()}


# --------------------------------------------------------------------------- capture
def capture_states(snap, want, max_hands, seed):
    """Seeded deterministic self-play -> distinct reached rounds-2-4 (postflop) decision states.

    Generalizes S1's build_multiway_flop_corpus: same CapturingHero (blueprint SearchAgent hero + 5
    calling stations, play_hand loop) but capture_streets={1,2,3} and NO multiway/raised restriction
    -- EVERY postflop hero decision node is a candidate (spec §C S2). Dedup by public+private
    signature. Returns per-street lists {1:[...],2:[...],3:[...]}.
    """
    from bench_search_capture import CapturingHero
    from pokerbot.engine import game
    from pokerbot.engine.game import GameState
    from pokerbot.eval.bots import build_bot

    rng = random.Random(seed)
    hero_seat = 0
    hero = CapturingHero(
        use_pluribus_search=True, blueprint_snapshot=snap, search_preflop=True,
        hero_seat=hero_seat, pluribus_leaf_select="uniform",
    )
    hero.solver.config.max_iterations = 4      # cheap capture; real solves happen in the re-solve pass
    hero.solver.config.time_budget_ms = 10 ** 9
    hero.solver.config.leaf.rollout_samples = 1
    hero.capture_streets = {1, 2, 3}           # flop/turn/river (postflop decision nodes)
    others = {s: build_bot("calling_station", rng) for s in range(1, 6)}

    # per-street target counts (oversample a little so the stratified sample has slack)
    tgt = {s: int(round(STRAT[s] * want)) for s in (1, 2, 3)}
    tgt[3] = max(tgt[3], want - tgt[1] - tgt[2])

    by_street = {1: [], 2: [], 3: []}
    seen = set()
    h = 0
    for h in range(max_hands):
        hero.reset_hand(hero_seat)
        policies = {hero_seat: hero}
        policies.update(others)
        state = GameState.new_hand(rng=rng)
        try:
            game.play_hand(policies, state=state, rng=rng)
        except Exception as e:  # noqa: BLE001
            log(f"  [cap] hand {h} aborted: {e!r}")
        for c in hero.captured:
            env = c["env"]
            st = int(c["street"])
            if st not in (1, 2, 3):
                continue
            sig = (st,
                   tuple(int(x) for x in np.asarray(env.board)),
                   tuple(int(x) for x in np.asarray(env.committed_total)),
                   tuple(int(x) for x in np.asarray(env.holes[hero_seat])),
                   tuple(int(x) for x in np.asarray(env.seat_status)))
            if sig in seen:
                continue
            seen.add(sig)
            by_street[st].append(c)
        hero.captured = []
        # stop once every street bucket has met its (oversampled) target
        if all(len(by_street[s]) >= tgt[s] for s in (1, 2, 3)):
            break
        if h % 200 == 0 and h:
            got = {s: len(by_street[s]) for s in (1, 2, 3)}
            el = time.time() - _T_START
            log(f"  [cap] hand {h}: distinct {got} (targets {tgt}) elapsed {el/60:.1f}m")
    got = {s: len(by_street[s]) for s in (1, 2, 3)}
    log(f"  [cap] captured distinct-by-street {got} over <= {h+1} hands (targets {tgt})")
    return by_street, tgt


def stratified_sample(by_street, tgt, want, seed):
    """Deterministic stratified draw to ~40/35/25; shortfalls logged, backfilled from other streets."""
    rng = random.Random(seed)
    picked = {1: [], 2: [], 3: []}
    for s in (1, 2, 3):
        pool = list(by_street[s])
        rng.shuffle(pool)
        picked[s] = pool[:tgt[s]]
    # backfill toward `want` from any street with leftovers (keeps the total near want even if one
    # street undershoots) — pure size top-up, mix shift is logged.
    total = sum(len(picked[s]) for s in (1, 2, 3))
    if total < want:
        leftovers = []
        for s in (1, 2, 3):
            leftovers += by_street[s][tgt[s]:]
        rng.shuffle(leftovers)
        need = want - total
        # re-bin backfill by its own street
        for c in leftovers[:need]:
            picked[int(c["street"])].append(c)
    corpus = []
    for s in (1, 2, 3):
        for i, c in enumerate(picked[s]):
            corpus.append({"id": f"{_STREET_NAME[s]}_{i:03d}", "cls": _STREET_NAME[s],
                           "street": s, "rec": c})
    achieved = {s: sum(1 for x in corpus if x["street"] == s) for s in (1, 2, 3)}
    return corpus, achieved


# --------------------------------------------------------------------------- solver (S1 plumbing)
def build_solver(snap):
    import engine_ab_h2h as ab
    pf = ab._build_policy({"iterations": K_RESOLVE}, snap, SEED, str(REPO))
    sv = [c.cell_contents for c in pf.__closure__
          if hasattr(getattr(c, "cell_contents", None), "config")][0]
    sv.config.native = True
    return sv


def solve_fixed(sv, root, pbs, K):
    """Run exactly K native iterations (wall budget removed so the fit never triggers). S1:231-241."""
    sv.config.max_iterations = int(K)
    sv.config.time_budget_ms = 10 ** 12
    sv.config.single_table_iter_floor = None
    sv.config.single_table_budget_ms = None
    t0 = time.perf_counter()
    res = sv.solve(root, pbs)
    ms = (time.perf_counter() - t0) * 1000.0
    probs = np.asarray(res.action_probs, dtype=np.float64)
    return probs, int(res.iterations), ms, str(res.metadata.get("served", "?"))


# --------------------------------------------------------------------------- metrics
def _restrict(vec, mask):
    """Project an action-prob vector onto the legal mask (length-aligned, clipped >=0)."""
    v = np.asarray(vec, dtype=np.float64)
    n = len(mask)
    full = np.zeros(n, dtype=np.float64)
    m = min(len(v), n)
    full[:m] = np.clip(v[:m], 0.0, None)
    return full[mask]


def kl_l1_argmax(bp_row, sp, mask):
    p = _restrict(bp_row, mask) + EPS
    q = _restrict(sp, mask) + EPS
    p /= p.sum()
    q /= q.sum()
    kl = float(np.sum(p * np.log(p / q)))
    l1 = float(np.sum(np.abs(p - q)))
    amatch = int(np.argmax(p) == np.argmax(q))
    return kl, l1, amatch


def value_surface(sp, mask):
    """MISS state: how much structure search injects vs the uniform-legal blueprint fallback."""
    q = _restrict(sp, mask) + EPS
    q /= q.sum()
    n = int(q.size)
    u = np.full(n, 1.0 / n)
    H = float(-np.sum(q * np.log(q)))
    H_uni = float(np.log(n)) if n > 1 else 0.0
    ratio = (H / H_uni) if H_uni > 1e-12 else 1.0
    l1_vs_uniform = float(np.sum(np.abs(q - u)))
    return {"entropy": round(H, 5), "entropy_uniform": round(H_uni, 5),
            "entropy_ratio": round(ratio, 5), "l1_vs_uniform": round(l1_vs_uniform, 5),
            "n_legal": n}


# --------------------------------------------------------------------------- output
def summarize(results):
    hit = [r for r in results if r.get("coverage") == "HIT"]
    miss = [r for r in results if r.get("coverage") == "MISS"]
    kls = [r["kl"] for r in hit if r.get("kl") is not None]
    med_kl = float(np.median(kls)) if kls else None
    l1s = [r["l1"] for r in hit if r.get("l1") is not None]
    amatch = [r["argmax_match"] for r in hit if r.get("argmax_match") is not None]
    surf_l1 = [r["surface"]["l1_vs_uniform"] for r in miss if r.get("surface")]
    per_street_cov = {}
    for s in (1, 2, 3):
        srs = [r for r in results if r["street"] == s and "coverage" in r]
        nh = sum(1 for r in srs if r["coverage"] == "HIT")
        per_street_cov[_STREET_NAME[s]] = {
            "n": len(srs), "hits": nh, "misses": len(srs) - nh,
            "hit_rate": round(nh / len(srs), 4) if srs else None}
    gate = None
    if med_kl is not None:
        gate = {"verdict": "PASS" if med_kl <= KL_TOL else "FAIL",
                "median_KL_over_HIT_states": round(med_kl, 5), "threshold": KL_TOL,
                "n_hit_states": len(hit)}
    else:
        gate = {"verdict": "NO_HIT_STATES",
                "note": "0 HIT states — blueprint served uniform-fallback on every sampled node; "
                        "gate is vacuous (expected-ish given known postflop under-coverage). "
                        "The measured object is the value-surface share below.",
                "n_hit_states": 0}
    return {
        "gate_S2": gate,
        "coverage": {
            "n_states": len(results), "n_hit": len(hit), "n_miss": len(miss),
            "postflop_hit_rate": round(len(hit) / len(results), 4) if results else None,
            "per_street": per_street_cov,
        },
        "hit_metrics": {
            "median_KL": round(med_kl, 5) if med_kl is not None else None,
            "mean_L1": round(float(np.mean(l1s)), 5) if l1s else None,
            "argmax_agreement": round(float(np.mean(amatch)), 4) if amatch else None,
        },
        "search_value_surface": {
            "note": "MISS states: search vs uniform-legal blueprint fallback. divergence_share = "
                    "fraction of MISS states where search L1-departs uniform by > 0.2.",
            "n_miss": len(miss),
            "median_l1_vs_uniform": round(float(np.median(surf_l1)), 5) if surf_l1 else None,
            "mean_entropy_ratio": round(
                float(np.mean([r["surface"]["entropy_ratio"] for r in miss if r.get("surface")])), 5)
                if miss else None,
            "divergence_share": round(float(np.mean([x > 0.2 for x in surf_l1])), 4) if surf_l1 else None,
        },
    }


def write_out(meta, results):
    slim = [{k: v for k, v in r.items() if k not in ("root", "pbs", "rec")} for r in results]
    payload = {"meta": meta, **summarize(results), "results": slim}
    tmp = str(OUT) + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=1)
    os.replace(tmp, OUT)


# --------------------------------------------------------------------------- main
def main() -> int:
    open(PROG, "w").close()
    from bench_search_capture import CapturingHero  # noqa: F401 (import-time abstraction bake check)
    from native_exploit_equiv import build_root
    from pokerbot.engine.np_engine import legal_action_mask_np
    from pokerbot.eval.lbr import (bp_coverage_report, bp_coverage_reset,
                                   make_blueprint_policy)
    from pokerbot.search import native_solver as _ns
    from pokerbot.search.pluribus_search import set_native_subgame

    if not _ns.native_subgame_available():
        _ns.enable()
    assert _ns.native_subgame_available(), "native subgame unavailable (need the built .pyd + EMD blob)"
    set_native_subgame(True)

    snap = load_snapshot()
    snap_bp = as_bytes_keyed(snap)
    bp_policy = make_blueprint_policy(snap_bp)
    log(f"snapshot infosets: {len(snap):,}  SMOKE={SMOKE} WANT={WANT} MAX_HANDS={MAX_HANDS} "
        f"K_RESOLVE={K_RESOLVE} KL_TOL={KL_TOL}")

    # ---- capture ----
    by_street, tgt = capture_states(snap, want=WANT, max_hands=MAX_HANDS, seed=CAP_SEED)
    corpus, achieved = stratified_sample(by_street, tgt, WANT, SEED)
    mix = {_STREET_NAME[s]: (achieved[s] / max(len(corpus), 1)) for s in (1, 2, 3)}
    log(f"corpus: {len(corpus)} states  achieved-by-street={achieved}  "
        f"mix={ {k: round(v,3) for k,v in mix.items()} }")
    if len(corpus) < WANT:
        log(f"WARNING: only {len(corpus)} states (< WANT={WANT}); proceeding (turn/river undersupply)")

    sv = build_solver(snap)

    # Warm the native engine once (blueprint marshal + EMD blob) so state 1 doesn't pay it (S1:433).
    try:
        from pokerbot.engine.jax_weva_encode import _POSTFLOP_NUM_BUCKETS
        t0 = time.perf_counter()
        _ns.warm_subgame_engine(sv.adapter, num_buckets=int(_POSTFLOP_NUM_BUCKETS),
                                max_playout_depth=int(sv.config.leaf.max_playout_depth))
        log(f"native engine warmed in {(time.perf_counter()-t0)*1000:.0f}ms")
    except Exception as e:  # noqa: BLE001
        log(f"warm-up skipped: {e!r}")

    meta = {
        "spec": "artifacts/track2_search_certification_design.md §C S2",
        "git_head": os.popen("git rev-parse HEAD").read().strip(),
        "config": "engine_ab_h2h._DEPLOYED_BASELINE (deployed, native subgame ON)",
        "K_resolve": K_RESOLVE, "kl_tol": KL_TOL, "eps": EPS, "seed": SEED, "cap_seed": CAP_SEED,
        "strat_targets": {_STREET_NAME[s]: STRAT[s] for s in (1, 2, 3)},
        "achieved_mix": {k: round(v, 4) for k, v in mix.items()},
        "coverage_signal": "lbr._BP_COVERAGE per-lookup (POKERBOT_BP_COVERAGE=1)",
        "action_ev_arm": "NOT-RUN (KL is the primary tol per spec; action-EV-within-25mbb skipped)",
        "smoke": SMOKE, "n_states": len(corpus), "started": time.strftime("%F %T"),
    }

    results = [{"id": it["id"], "cls": it["cls"], "street": it["street"],
                "rec": it["rec"]} for it in corpus]

    n = len(corpus)
    last_log = time.time()
    for si, it in enumerate(corpus):
        rec = it["rec"]
        env = rec["env"]
        seat = int(rec["hero_seat"])
        mask = legal_action_mask_np(env, seat).astype(bool)
        n_legal = int(mask.sum())

        # (a) blueprint root row + HIT/MISS via the per-lookup coverage counter (the reliable signal).
        bp_coverage_reset()
        bp_row = np.asarray(bp_policy(env, seat), dtype=np.float64)
        rep = bp_coverage_report()
        hits = sum(v["hits"] for v in rep.values())
        misses = sum(v["misses"] for v in rep.values())
        coverage = "HIT" if hits > 0 and misses == 0 else ("MISS" if misses > 0 else "UNKNOWN")

        # (b) deployed-config search re-solve at K=2000 (S1 plumbing).
        root, pbs = build_root(rec)
        sp, iters, ms, served = solve_fixed(sv, root, pbs, K_RESOLVE)

        r = results[si]
        r.update({"coverage": coverage, "n_legal": n_legal, "iters": iters,
                  "wall_ms": round(ms, 1), "served": served,
                  "bp_row": [round(float(x), 5) for x in bp_row],
                  "search_probs": [round(float(x), 5) for x in sp]})
        if coverage == "HIT":
            kl, l1, amatch = kl_l1_argmax(bp_row, sp, mask)
            r.update({"kl": round(kl, 5), "l1": round(l1, 5), "argmax_match": amatch,
                      "surface": None})
        else:
            r.update({"kl": None, "l1": None, "argmax_match": None,
                      "surface": value_surface(sp, mask)})

        now = time.time()
        if SMOKE or (now - last_log) >= 55.0 or si == n - 1:
            write_out(meta, results[:si + 1])
            last_log = now
            pct = 100.0 * (si + 1) / n
            el = now - _T_START
            eta = el / (si + 1) * (n - si - 1)
            kltxt = f"KL={r['kl']}" if coverage == "HIT" else f"surf_l1={r['surface']['l1_vs_uniform']}"
            log(f"[{pct:5.1f}%] {it['id']:>12s} cls={it['cls']:<6s} cov={coverage:<7s} "
                f"nlegal={n_legal} {kltxt} K{K_RESOLVE} wall={ms:.0f}ms "
                f"| elapsed {el/60:.1f}m ETA {eta/60:.1f}m")

    meta["finished"] = time.strftime("%F %T")
    write_out(meta, results)
    summ = summarize(results)
    log("=== S2 SUMMARY ===")
    log(f"  coverage: {json.dumps(summ['coverage'])[:300]}")
    log(f"  gate_S2:  {json.dumps(summ['gate_S2'])[:300]}")
    log(f"  value_surface: {json.dumps(summ['search_value_surface'])[:300]}")
    log(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
