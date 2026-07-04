#!/usr/bin/env python3
"""LEG B/C bridge: EXACT river K-headroom.

Leg B computed, per HU river decision state, the exact BR Q-values Q_BR(a | hero combo) against a
converged reference-equilibrium opponent. This follow-up re-serves the SAME states through the
deployed path at K=50000 (same public seed; the K=2000 serve is in the Leg B artifact) and evaluates
BOTH mixtures under the SAME exact Q — giving the K-headroom as an exact EV number on rivers:

    headroom = EVloss(K=2000) - EVloss(K=50000)   (mbb per decision, >0 means budget helps)

Usage: python scripts/audit_river_k_headroom.py [max_states]
Output: artifacts/audit_river_k_headroom.json
"""
from __future__ import annotations

import json
import os
import pickle
import sys
import time

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
_SCRIPTS = os.path.join(REPO, "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

os.environ.setdefault("JAX_PLATFORMS", "cpu")
_FROZEN = os.path.join(REPO, "data", "abstraction", "emd_frozen_flop_turn_160_r400.emdfrzn")
for _k, _v in {
    "POKERBOT_BETTING_ABSTRACTION": "1",
    "POKERBOT_POSTFLOP_BUCKETS": "1",
    "POKERBOT_POSTFLOP_NUM_BUCKETS": "160",
    "POKERBOT_POSTFLOP_MODE": "emd",
    "POKERBOT_POSTFLOP_EMD_FROZEN": _FROZEN,
    "POKERBOT_EMD_FROZEN_MIRROR_FALLBACK": "1",
    "POKERBOT_PREFLOP_LOSSLESS": "1",
}.items():
    os.environ.setdefault(_k, _v)

import numpy as np  # noqa: E402

MBB_PER_CHIP = 10.0
PROG = os.path.join(REPO, "artifacts", "decision_quality_progress.log")


def log(msg: str):
    with open(PROG, "a") as f:
        f.write(f"[{time.strftime('%F %T')}] [legBC] {msg}\n")
    print(msg, flush=True)


def main() -> int:
    from pokerbot.search import native_solver as _ns
    from pokerbot.search.pluribus_search import set_native_subgame
    import engine_ab_h2h as ab

    max_states = int(sys.argv[1]) if len(sys.argv) > 1 else 40

    if not _ns.native_subgame_available():
        _ns.enable()
    assert _ns.native_subgame_available()
    set_native_subgame(True)

    raw = pickle.load(open(os.path.join(REPO, "checkpoints", "serve_blueprint_6max.pkl"), "rb"))
    snap = raw["strat_sum"] if isinstance(raw, dict) and "strat_sum" in raw else raw
    pf_ref = ab._build_policy({"iterations": 50000}, snap, 4242, REPO)

    legb = json.load(open(os.path.join(REPO, "artifacts", "audit_river_exact.json")))
    results_b = legb["results"] if "results" in legb else legb
    states = pickle.load(open(os.path.join(REPO, "artifacts", "audit_river_states.pkl"), "rb"))
    from pokerbot.engine.side_pots_jax import SEAT_ALLIN
    states = [s for s in states
              if not any(int(x) == int(SEAT_ALLIN) for x in np.asarray(s["env"].seat_status))]
    by_deal = {(s["deal"], s["hero"]): s for s in states}

    out = []
    for r in results_b[:max_states]:
        st = by_deal.get((r["deal"], r["hero"]))
        if st is None:
            continue
        env, hero = st["env"], st["hero"]
        t0 = time.time()
        pi_ref = np.asarray(pf_ref(env, hero), dtype=np.float64)
        q = {int(a): float(v) for a, v in r["q_br_chips"].items()}
        acts = sorted(q.keys())
        v_br = max(q.values())
        # K=50000 mixture EV under the exact Q (renormalized over the reference tree's root menu)
        p_ref = np.asarray([max(pi_ref[a], 0.0) for a in acts])
        p_ref = p_ref / p_ref.sum() if p_ref.sum() > 1e-9 else p_ref
        ev_ref = float(sum(p * q[a] for p, a in zip(p_ref, acts)))
        loss_ref_mbb = (v_br - ev_ref) * MBB_PER_CHIP
        loss_dep_mbb = float(r["ev_loss_mbb"])
        out.append({
            "state": r["state"], "deal": r["deal"], "pot_bb": r["pot_bb"],
            "ev_loss_mbb_k2000": round(loss_dep_mbb, 3),
            "ev_loss_mbb_k50000": round(loss_ref_mbb, 3),
            "headroom_mbb": round(loss_dep_mbb - loss_ref_mbb, 3),
            "served_k50000": {str(a): round(float(p), 4) for a, p in zip(acts, p_ref)},
            "wall_s": round(time.time() - t0, 1),
        })
        log(f"state {r['state']}: k2000 loss {loss_dep_mbb:.1f} -> k50000 loss "
            f"{loss_ref_mbb:.1f} mbb (headroom {loss_dep_mbb - loss_ref_mbb:+.1f}) "
            f"[{out[-1]['wall_s']}s]")
        with open(os.path.join(REPO, "artifacts", "audit_river_k_headroom.json"), "w") as f:
            json.dump({"partial": True, "results": out}, f, indent=1)

    a2 = np.asarray([o["ev_loss_mbb_k2000"] for o in out])
    a5 = np.asarray([o["ev_loss_mbb_k50000"] for o in out])
    summary = {
        "n_states": len(out),
        "mean_loss_mbb_k2000": round(float(a2.mean()), 3),
        "mean_loss_mbb_k50000": round(float(a5.mean()), 3),
        "mean_headroom_mbb": round(float((a2 - a5).mean()), 3),
        "median_headroom_mbb": round(float(np.median(a2 - a5)), 3),
    }
    with open(os.path.join(REPO, "artifacts", "audit_river_k_headroom.json"), "w") as f:
        json.dump({"summary": summary, "results": out}, f, indent=1)
    log(f"DONE: mean river loss K2000={summary['mean_loss_mbb_k2000']}mbb vs "
        f"K50000={summary['mean_loss_mbb_k50000']}mbb; mean headroom "
        f"{summary['mean_headroom_mbb']}mbb")
    return 0


if __name__ == "__main__":
    sys.exit(main())
