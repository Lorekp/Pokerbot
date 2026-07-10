#!/usr/bin/env python3
"""QUALITY A/B: SETTLE_SHOWDOWN_LEAF gate OFF vs ON, coupled with leaf_cache_worlds (M sweep).

Successor of the audit's scripts/audit_leaf_settle/ab_settle_leaf.py (branch
worktree-agent-a111c7295cd6b4807), rebuilt on the production gate API. Corpus (read-only from the
MAIN tree): Leg-B river states w/ exact Q_BR (EV-loss machinery) + search_roots.pkl live probes.

Per state, K=2000 fixed iterations on the DEPLOYED config (native subgame ON, two-sided,
per-decision CRN — same public seed across all arms), solved once per ARM in the same process:

    off_m4 : gate OFF, leaf_cache_worlds=4   (DEPLOYED baseline)
    on_m1  : gate ON,  leaf_cache_worlds=1   (naive settle — the audit's world-freeze tails)
    on_m4  : gate ON,  leaf_cache_worlds=4   (proposed deploy coupling)
    on_m16 : gate ON,  leaf_cache_worlds=16  (tail-sensitivity probe)

Reports per arm: root-policy L1 vs baseline, argmax flips, playout terminal-class stats, river
EV-loss vs exact Q_BR (median / p90 / max — the tails are the promotion question).

Output: scripts/settle_leaf_fix/ab_settle_quality.json (worktree).
"""
from __future__ import annotations

import json
import os
import pickle
import sys
import time
from pathlib import Path

WORKTREE = Path(__file__).resolve().parents[2]
MAIN = Path(".")
sys.path.insert(0, str(WORKTREE))
sys.path.insert(0, str(WORKTREE / "scripts"))

_FROZEN = str(MAIN / "data" / "abstraction" / "emd_frozen_flop_turn_160_r400.emdfrzn")
for _k, _v in {
    "POKERBOT_BETTING_ABSTRACTION": "1",
    "POKERBOT_POSTFLOP_BUCKETS": "1",
    "POKERBOT_PREFLOP_LOSSLESS": "1",
    "POKERBOT_POSTFLOP_MODE": "emd",
    "POKERBOT_POSTFLOP_NUM_BUCKETS": "160",
    "POKERBOT_POSTFLOP_EMD_FROZEN": _FROZEN,
    "POKERBOT_EMD_FROZEN_MIRROR_FALLBACK": "1",
    "POKERBOT_NATIVE_SUBGAME": "1",
    "JAX_PLATFORMS": "cpu",
}.items():
    os.environ.setdefault(_k, _v)

import numpy as np  # noqa: E402

K = int(os.environ.get("AB_K", "2000"))
N_RIVER = int(os.environ.get("AB_N_RIVER", "20"))
N_LIVE = int(os.environ.get("AB_N_LIVE", "15"))
SEED = 4242
MBB_PER_CHIP = 10.0
OUT = WORKTREE / "scripts" / "settle_leaf_fix" / "ab_settle_quality.json"
BP_PATH = MAIN / "checkpoints" / "serve_blueprint_6max.pkl"

ARMS = [
    ("off_m4", False, 4),
    ("on_m1", True, 1),
    ("on_m4", True, 4),
    ("on_m16", True, 16),
]
BASE_ARM = "off_m4"


def log(msg):
    print(f"[{time.strftime('%T')}] {msg}", flush=True)


def load_snapshot():
    raw = pickle.load(open(BP_PATH, "rb"))
    return raw["strat_sum"] if isinstance(raw, dict) and "strat_sum" in raw else raw


def build_river_corpus(n):
    from native_exploit_equiv import SubgameRoot, _PBS
    from pokerbot.engine.np_engine import SEAT_ALLIN

    states = pickle.load(open(MAIN / "artifacts" / "audit_river_states.pkl", "rb"))
    states = [s for s in states
              if not any(int(x) == int(SEAT_ALLIN) for x in np.asarray(s["env"].seat_status))]
    by_deal = {(int(s["deal"]), int(s["hero"])): s for s in states}
    legb = json.load(open(MAIN / "artifacts" / "audit_river_exact.json"))
    results_b = legb["results"] if "results" in legb else legb

    corpus = []
    for r in results_b:
        st = by_deal.get((int(r["deal"]), int(r["hero"])))
        if st is None:
            continue
        env, hero = st["env"], int(st["hero"])
        q = {int(a): float(v) for a, v in r["q_br_chips"].items()}
        root = SubgameRoot(env=env, hero_seat=hero, pbs_vector=np.zeros(1),
                           legal_mask=np.ones(1, dtype=bool))
        corpus.append({
            "id": f"river_d{r['deal']}_h{r['hero']}_s{r['state']}",
            "cls": "river", "street": 3, "root": root, "pbs": _PBS(env),
            "q": q, "v_br": max(q.values()), "pot_bb": float(r["pot_bb"]),
        })
        if len(corpus) >= n:
            break
    return corpus


def build_live_probe_corpus(n):
    from native_exploit_equiv import build_root
    roots = pickle.load(open(MAIN / "artifacts" / "search_roots.pkl", "rb"))[:n]
    corpus = []
    for i, rec in enumerate(roots):
        root, pbs = build_root(rec)
        corpus.append({"id": f"liveprobe_{i:02d}", "cls": "live_probe",
                       "street": int(rec.get("street", -1)), "root": root, "pbs": pbs})
    return corpus


def build_solver(snap):
    import engine_ab_h2h as ab
    pf = ab._build_policy({"iterations": K}, snap, SEED, str(WORKTREE))
    sv = [c.cell_contents for c in pf.__closure__
          if hasattr(getattr(c, "cell_contents", None), "config")][0]
    sv.config.native = True
    return sv


def solve_fixed(sv, item):
    sv.config.max_iterations = int(K)
    sv.config.time_budget_ms = 10 ** 12
    sv.config.single_table_iter_floor = None
    sv.config.single_table_budget_ms = None
    t0 = time.perf_counter()
    res = sv.solve(item["root"], item["pbs"])
    ms = (time.perf_counter() - t0) * 1000.0
    return np.asarray(res.action_probs, dtype=np.float64), int(res.iterations), ms


def evloss_mbb(probs, q, v_br):
    acts = sorted(q.keys())
    p = np.asarray([max(probs[a], 0.0) if a < len(probs) else 0.0 for a in acts])
    s = p.sum()
    p = p / s if s > 1e-9 else p
    ev = float(sum(pi * q[a] for pi, a in zip(p, acts)))
    return (v_br - ev) * MBB_PER_CHIP


def q(vals, frac):
    v = sorted(vals)
    return v[min(len(v) - 1, int(frac * len(v)))]


def main():
    from pokerbot.search import native_solver as ns
    from pokerbot.search.pluribus_search import set_native_subgame
    if not ns.native_subgame_available():
        ns.enable()
    assert ns.native_subgame_available(), "native subgame unavailable"
    set_native_subgame(True)
    ext = ns._EXT
    caps = set(ext.native_caps())
    assert "settle_showdown_leaf" in caps, f"stale worktree pyd: {sorted(caps)}"
    ext.leaf_settle_stats_set(True)

    snap = load_snapshot()
    log(f"snapshot infosets: {len(snap):,}")
    corpus = build_river_corpus(N_RIVER) + build_live_probe_corpus(N_LIVE)
    log(f"corpus: {len(corpus)} states; arms: {[a[0] for a in ARMS]}; K={K}")
    sv = build_solver(snap)

    try:
        from pokerbot.engine.jax_weva_encode import _POSTFLOP_NUM_BUCKETS
        ns.warm_subgame_engine(sv.adapter, num_buckets=int(_POSTFLOP_NUM_BUCKETS),
                               max_playout_depth=int(sv.config.leaf.max_playout_depth))
        log("native engine warmed")
    except Exception as e:  # noqa: BLE001
        log(f"warm-up skipped: {e!r}")

    results = []
    for si, it in enumerate(corpus):
        row = {"id": it["id"], "cls": it["cls"], "street": it["street"]}
        arms = {}
        for name, gate, m in ARMS:
            ns.set_settle_showdown_leaf(bool(gate))
            sv.config.leaf_cache_worlds = int(m)
            ext.leaf_settle_stats_reset()
            probs, iters, ms = solve_fixed(sv, it)
            nd, nu, no, bb_d, bb_u = ext.leaf_settle_stats()
            tot = max(1, nd + nu + no)
            arms[name] = {
                "probs": [round(float(x), 6) for x in probs],
                "argmax": int(np.argmax(probs)), "iters": iters, "wall_ms": round(ms, 1),
                "n_done": int(nd), "n_unsettled_showdown": int(nu), "n_other": int(no),
                "frac_unsettled": round(nu / tot, 4),
                "mass_share_unsettled": round(bb_u / max(1e-9, bb_d + bb_u), 4),
            }
        ns.set_settle_showdown_leaf(False)
        base = np.asarray(arms[BASE_ARM]["probs"])
        for name, _, _ in ARMS:
            pa = np.asarray(arms[name]["probs"])
            nn = min(base.size, pa.size)
            arms[name]["l1_vs_base"] = round(float(np.abs(base[:nn] - pa[:nn]).sum()), 5)
            arms[name]["argmax_flip_vs_base"] = bool(
                arms[name]["argmax"] != arms[BASE_ARM]["argmax"])
            if it["cls"] == "river":
                arms[name]["evloss_mbb"] = round(
                    evloss_mbb(pa, it["q"], it["v_br"]), 3)
        row["arms"] = arms
        results.append(row)
        msg = " ".join(
            f"{n}:L1={arms[n]['l1_vs_base']:.3f}"
            + (f",ev={arms[n].get('evloss_mbb')}" if it["cls"] == "river" else "")
            for n, _, _ in ARMS)
        log(f"[{si+1}/{len(corpus)}] {it['id']:>24s} {msg}")
        with open(OUT, "w") as f:
            json.dump({"K": K, "n": len(corpus), "arms": [a[0] for a in ARMS],
                       "results": results}, f, indent=1)

    # summary
    summary = {}
    riv = [r for r in results if r["cls"] == "river"]
    for name, _, _ in ARMS:
        l1s = [r["arms"][name]["l1_vs_base"] for r in results]
        flips = [r["arms"][name]["argmax_flip_vs_base"] for r in results]
        s = {
            "median_L1_vs_base": float(np.median(l1s)),
            "argmax_flips_vs_base": int(np.sum(flips)),
        }
        if riv:
            evs = [r["arms"][name]["evloss_mbb"] for r in riv]
            s.update({
                "river_evloss_median_mbb": float(np.median(evs)),
                "river_evloss_mean_mbb": float(np.mean(evs)),
                "river_evloss_p90_mbb": float(q(evs, 0.9)),
                "river_evloss_max_mbb": float(max(evs)),
            })
        summary[name] = s
    with open(OUT, "w") as f:
        json.dump({"K": K, "n": len(corpus), "arms": [a[0] for a in ARMS],
                   "summary": summary, "results": results}, f, indent=1)
    log("SUMMARY " + json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
