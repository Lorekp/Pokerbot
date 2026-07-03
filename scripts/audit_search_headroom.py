#!/usr/bin/env python3
"""LEG C (decision-quality audit): search convergence headroom at the deployed K=2000.

On game-flow flop/turn hero-decision states, serve the SAME node through the deployed native
two-sided search at K=2000 (deployed), K=8000 and K=50000 (reference), all with the SAME
per-decision public seed (the higher-K solves replay the lower-K stream prefix). Metrics:

  * action-flip rate: argmax(pi_2000) != argmax(pi_50000)
  * L1(pi_2000, pi_50000) and L1(pi_8000, pi_50000)  (K-doubling stability trend)
  * EV headroom estimate: (pi_50000 - pi_2000) . Q_ref  and BR-gap max_a Q_ref[a] - pi_K . Q_ref,
    where Q_ref is the root per-action CFV vector of a K=50000 ONE-SIDED solve vs blueprint-mixed
    opponents (native solve_exploit want_root_q with the frozen exploit table EMPTY and
    frozen_read_response=False — i.e. pure GTO base dynamics). CAVEAT: Q_ref is measured under
    one-sided dynamics (two_sided is firewalled off the exploit FFI), so the EV numbers are an
    ESTIMATE of the root-mixture EV difference against blueprint-playing opponents, not an exact
    equilibrium loss. The flip/L1 numbers use the exact deployed two-sided config.

Usage: python scripts/audit_search_headroom.py [n_states] -> artifacts/audit_search_headroom.json
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

NUM_ACTIONS = 11
PROG = os.path.join(REPO, "artifacts", "decision_quality_progress.log")
OUT = os.path.join(REPO, "artifacts", "audit_search_headroom.json")

K_DEP, K_MID, K_REF = 2000, 8000, 50000


def log(msg: str):
    with open(PROG, "a") as f:
        f.write(f"[{time.strftime('%F %T')}] [legC] {msg}\n")
    print(msg, flush=True)


class _EmptyFrozen:
    keys_actor: list = []
    keys_ctx: list = []
    keys_mask: list = []
    vals: list = []


def main() -> int:
    import jax
    from pokerbot.engine.jax_engine import init_hand
    from pokerbot.engine.np_engine import (
        apply_action_np, current_player_np, is_chance_node_np, is_terminal_np,
        legal_action_mask_np, sample_chance_np, to_numpy_state,
    )
    from pokerbot.eval.solver_exploitability import make_blueprint_policy_fn
    from pokerbot.search import native_solver as _ns
    from pokerbot.search.pluribus_search import set_native_subgame
    from pokerbot.engine.jax_weva_encode import _POSTFLOP_NUM_BUCKETS
    import engine_ab_h2h as ab

    _args = [a for a in sys.argv[1:] if not a.startswith("--")]
    resume = "--resume" in sys.argv[1:]
    n_states = int(_args[0]) if _args else 30

    if not _ns.native_subgame_available():
        _ns.enable()
    assert _ns.native_subgame_available(), "native SubgameSolver required"
    set_native_subgame(True)

    raw = pickle.load(open(os.path.join(REPO, "checkpoints", "serve_blueprint_6max.pkl"), "rb"))
    snap = raw["strat_sum"] if isinstance(raw, dict) and "strat_sum" in raw else raw

    # Three serve stacks through the REAL deployed path (engine_ab_h2h._build_policy):
    pf_dep = ab._build_policy({}, snap, 4242, REPO)                       # deployed K=2000
    pf_mid = ab._build_policy({"iterations": K_MID}, snap, 4242, REPO)
    pf_ref = ab._build_policy({"iterations": K_REF}, snap, 4242, REPO)
    sv = [c.cell_contents for c in pf_dep.__closure__
          if hasattr(getattr(c, "cell_contents", None), "config")][0]
    cfg = sv.config

    # ---- collect game-flow flop/turn hero decision states ----
    bp = make_blueprint_policy_fn(snap)
    rng = np.random.default_rng(31337)
    occ = jax.numpy.array([True] * 6)
    want_per_street = {1: (n_states + 1) // 2, 2: n_states - (n_states + 1) // 2}
    got = {1: 0, 2: 0}
    states = []
    d = 0
    while (got[1] < want_per_street[1] or got[2] < want_per_street[2]) and d < 30000:
        d += 1
        e = to_numpy_state(init_hand(jax.random.PRNGKey(770001 + d), occupancy=occ))
        guard = 0
        while not is_terminal_np(e) and guard < 100:
            guard += 1
            if is_chance_node_np(e):
                e = sample_chance_np(e)
                continue
            s = current_player_np(e)
            if s < 0:
                break
            street = int(e.street)
            m = legal_action_mask_np(e, s).astype(bool)
            if street in (1, 2) and m.sum() > 1 and got[street] < want_per_street[street] \
                    and rng.random() < 0.35:
                pot = (int(e.pot_dead) + int(np.asarray(e.committed_total).sum())) / 100.0
                states.append({"deal": d, "env": e, "hero": int(s), "street": street,
                               "pot_bb": round(pot, 2)})
                got[street] += 1
                break
            if m.sum() <= 1:
                a = int(np.argmax(m))
            else:
                p = np.asarray(bp(e, s), dtype=np.float64) * m
                ssum = p.sum()
                p = p / ssum if ssum > 1e-12 else m.astype(np.float64) / m.sum()
                a = int(rng.choice(NUM_ACTIONS, p=p))
            e = apply_action_np(e, a)
    log(f"collected {len(states)} flop/turn states (flop {got[1]}, turn {got[2]}) from {d} deals")

    # ---- resume-from-partial: the collection walk above is DETERMINISTIC (fixed default_rng
    # seed + fixed PRNGKey per deal + fixed want_per_street for a given n_states), verified by
    # bit-identical state metrics across independent runs. Reloaded results must match on deal
    # id at their index; any mismatch aborts the resume and restarts the leg cleanly.
    done: dict[int, dict] = {}
    if resume and os.path.exists(OUT):
        try:
            prev = json.load(open(OUT))
            for r in prev.get("results", []):
                done[int(r["state"])] = r
        except Exception:
            done = {}
        for dsi, r in sorted(done.items()):
            if dsi >= len(states) or int(states[dsi]["deal"]) != int(r["deal"]) \
                    or int(states[dsi]["street"]) != int(r["street"]):
                log(f"resume MISMATCH at state {dsi} (deal {r['deal']}) — "
                    f"collection not reproducible, restarting leg cleanly")
                done = {}
                break
        if done:
            log(f"resume: {len(done)} states already measured "
                f"(indices {sorted(done)}), skipping them")

    results = []
    t_all = time.time()
    for si, st in enumerate(states):
        if si in done:
            results.append(done[si])
            continue
        env, hero = st["env"], st["hero"]
        t0 = time.time()
        pi_dep = np.asarray(pf_dep(env, hero), dtype=np.float64)
        t_dep = time.time() - t0
        pi_mid = np.asarray(pf_mid(env, hero), dtype=np.float64)
        t1 = time.time()
        pi_ref = np.asarray(pf_ref(env, hero), dtype=np.float64)
        t_ref = time.time() - t1

        # Q_ref: one-sided K=50000 root CFVs vs blueprint-mixed opponents (empty exploit table).
        pub_seed = sv._public_seed(env, hero)
        t2 = time.time()
        outq = _ns.native_subgame_solve_exploit(
            env, hero, sv.adapter,
            public_seed=pub_seed, max_iterations=K_REF, max_depth=cfg.max_depth,
            blueprint_mix=cfg.blueprint_mix, rollout_samples=cfg.leaf.rollout_samples,
            resample_board=cfg.leaf.resample_board,
            max_playout_depth=cfg.leaf.max_playout_depth,
            num_buckets=int(_POSTFLOP_NUM_BUCKETS), linear_averaging=cfg.linear_averaging,
            frozen_table=_EmptyFrozen(), rnr=[float("nan")] * 6,
            frozen_read_response=False, rnr_restrict=False, want_root_q=True,
            repair_deck=cfg.repair_deck, decouple_sampler=cfg.decouple_sampler,
            range_aware=cfg.range_aware, range_uniform_floor=cfg.range_uniform_floor,
            range_reach_full=cfg.range_reach_full,
        )
        t_q = time.time() - t2
        q_ref = None
        if outq is not None:
            _, rq = outq
            q_ref = np.asarray(rq, dtype=np.float64) if rq is not None else None

        legal = np.asarray(legal_action_mask_np(env, hero), dtype=bool)
        la = [int(a) for a in np.flatnonzero(pi_ref > 1e-9)] or [int(np.argmax(pi_ref))]
        flip = int(np.argmax(pi_dep) != np.argmax(pi_ref))
        l1_dep = float(np.abs(pi_dep - pi_ref).sum())
        l1_mid = float(np.abs(pi_mid - pi_ref).sum())

        res = {
            "state": si, "deal": st["deal"], "street": st["street"], "pot_bb": st["pot_bb"],
            "flip_2000_vs_50000": flip,
            "l1_2000_vs_50000": round(l1_dep, 4),
            "l1_8000_vs_50000": round(l1_mid, 4),
            "argmax": {"k2000": int(np.argmax(pi_dep)), "k8000": int(np.argmax(pi_mid)),
                       "k50000": int(np.argmax(pi_ref))},
            "pi_2000": [round(float(x), 4) for x in pi_dep],
            "pi_50000": [round(float(x), 4) for x in pi_ref],
            "wall_s": {"k2000": round(t_dep, 1), "k50000": round(t_ref, 1),
                       "qref": round(t_q, 1)},
        }
        if q_ref is not None:
            sup = np.zeros(NUM_ACTIONS, dtype=bool)
            for pv in (pi_dep, pi_mid, pi_ref):
                sup |= (pv > 1e-9)
            qm = np.where(sup, q_ref, -np.inf)
            v_dep = float((pi_dep * q_ref).sum())
            v_mid = float((pi_mid * q_ref).sum())
            v_ref = float((pi_ref * q_ref).sum())
            v_max = float(qm.max())
            res["q_ref"] = {str(int(a)): round(float(q_ref[a]), 4) for a in np.flatnonzero(sup)}
            res["ev_headroom_ref_minus_dep"] = round(v_ref - v_dep, 5)
            res["br_gap_dep"] = round(v_max - v_dep, 5)
            res["br_gap_mid"] = round(v_max - v_mid, 5)
            res["br_gap_ref"] = round(v_max - v_ref, 5)
        results.append(res)
        log(f"state {si + 1}/{len(states)}: street={st['street']} pot={st['pot_bb']}bb "
            f"flip={flip} L1_2k={res['l1_2000_vs_50000']} L1_8k={res['l1_8000_vs_50000']} "
            f"headroom={res.get('ev_headroom_ref_minus_dep')} "
            f"({time.time() - t0:.0f}s, total {(time.time() - t_all) / 60:.1f}min)")
        with open(OUT, "w") as f:
            json.dump({"partial": True, "results": results}, f, indent=1)

    flips = np.asarray([r["flip_2000_vs_50000"] for r in results])
    l1d = np.asarray([r["l1_2000_vs_50000"] for r in results])
    l1m = np.asarray([r["l1_8000_vs_50000"] for r in results])
    hr = np.asarray([r["ev_headroom_ref_minus_dep"] for r in results
                     if "ev_headroom_ref_minus_dep" in r])
    bg_dep = np.asarray([r["br_gap_dep"] for r in results if "br_gap_dep" in r])
    bg_ref = np.asarray([r["br_gap_ref"] for r in results if "br_gap_ref" in r])
    summary = {
        "n_states": len(results),
        "flip_rate": round(float(flips.mean()), 4),
        "l1_2000_vs_50000_mean": round(float(l1d.mean()), 4),
        "l1_8000_vs_50000_mean": round(float(l1m.mean()), 4),
        "ev_headroom_mean_qunits": round(float(hr.mean()), 5) if hr.size else None,
        "ev_headroom_p90_qunits": round(float(np.percentile(hr, 90)), 5) if hr.size else None,
        "br_gap_dep_mean_qunits": round(float(bg_dep.mean()), 5) if bg_dep.size else None,
        "br_gap_ref_mean_qunits": round(float(bg_ref.mean()), 5) if bg_ref.size else None,
        "note": "q units determined post-hoc (bb vs chips) by magnitude vs pot",
    }
    with open(OUT, "w") as f:
        json.dump({"summary": summary, "results": results}, f, indent=1)
    log(f"DONE: flip_rate={summary['flip_rate']} L1_2k={summary['l1_2000_vs_50000_mean']} "
        f"L1_8k={summary['l1_8000_vs_50000_mean']} headroom_mean={summary['ev_headroom_mean_qunits']} "
        f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
