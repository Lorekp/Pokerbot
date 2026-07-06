#!/usr/bin/env python3
"""R3 MINI-PROOF (artifacts/search_gap_audit.md §3.1, pre-registered): do HAND-CONDITIONAL
subgame opponent infosets recover a large share of the Leg-B river EV-loss, or is that loss
genuinely upstream (range model / abstraction)?

Reuses the 40 solved Leg-B river reference states ON DISK (artifacts/audit_river_states.pkl +
artifacts/audit_river_exact.json): the per-state root BR values q_br_chips of the CONVERGED
reference equilibrium price ANY new served root mixture exactly, so no reference is re-solved.

Arms (per state; same public-state-derived CRN seed across arms — per_decision_seed):
  B  — the DEPLOYED serve (native two-sided K=2000): per-state EV-loss reused verbatim from
       artifacts/audit_river_exact.json (Leg B measured it; internal consistency re-checked here).
  Cb — BUDGET CONTROL: the deployed-config PYTHON scalar solve, public-only infosets, K=20000.
       Isolates (Python-scalar + K + sampler seam) from hand-conditioning. Leg C predicts ≈ B.
  C  — Cb + config.hand_conditional_infosets=True: opponent subgame infoset keys extended with
       the seat's exact private hand id; leaf/terminal cache made world-conditional (required —
       see IMPLEMENTATION NOTES below). K=20000 (removes the budget confound per the spec).
  D  — C + config.sample_hero_range=True (hero-leak fix): hero's hole REDRAWN from his reach
       range on non-hero-traverser (opponent-regret-building) iterations; serve still read off
       at his actual hand.

EV-loss per arm = v_br - sum_a served[a] * q_br[a] over the reference root actions (identical
metric/units to Leg B: chips -> mbb x10, %pot).

RANGES: all Python arms sample opponent holes from the SAME reach-range model the deployed
native serve and the Leg-B reference used (engine range_weights(..., "reach") blended with the
deployed 25% uniform floor), seeded into PBSTracker.beliefs — the Python loop's PBS sampler is
otherwise uniform, which would confound Cb-vs-B. Arm D additionally seeds beliefs[hero] with
hero's reach range.

IMPLEMENTATION NOTES (explicit, non-quiet — repeated in the verdict doc):
  1. The §3.1 text says "opponent infoset keys extended by an exact-hand id". In the scalar
     loop the river showdown value flows through _leaf_value's PUBLIC-keyed frozen 4-world-mean
     cache; hand-extended infosets fed by that hand-blind value signal are a no-op BY
     CONSTRUCTION. The knob therefore also keys the leaf cache on the world's non-folded holes
     (exact showdown per world). This is a required completion of the spec, not a liberty.
  2. Arm D "sampled from his reach range on non-averaging traversal iterations": implemented as
     a redraw of hero's hole on iterations whose TRAVERSER is not the hero (those are exactly
     the iterations that build opponent regrets in the two-sided alternating scheme; hero's own
     averaging iterations keep his actual hand). Opponent sampling still excludes hero's ACTUAL
     cards (deployed card-removal behavior), a residual inconsistency documented in the verdict.

Usage:
  python scripts/r3_miniproof.py identity          # knob-absent vs knob-off byte-identity gate
  python scripts/r3_miniproof.py smoke [si]        # one state, K=2000, all arms (timing)
  python scripts/r3_miniproof.py run [--k 20000] [--workers 4] [--resume]
  python scripts/r3_miniproof.py verdict           # recompute summary table from results JSON
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

# --- POKERBOT_* env BEFORE any pokerbot import (deployed EMD-K160 regime, Leg-B preamble) ---
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
# The Python scalar loop is the object under measurement: hard-force the native SUBGAME path off
# (native engine PRIMITIVES stay on — bit-identical dispatch) and keep the eval-hygiene blend off.
os.environ["POKERBOT_NATIVE_SUBGAME"] = "0"
os.environ["POKERBOT_SEARCH_BLUEPRINT_BLEND"] = "0"

import numpy as np  # noqa: E402

NUM_ACTIONS = 11
CHIP_PER_BB = 100.0
MBB_PER_CHIP = 10.0
K_DEFAULT = 20000
SEED = 4242  # Leg B's base seed (per_decision_seed re-derives per state; rng_salt=0 deployed)

STATES_PKL = os.path.join(REPO, "artifacts", "audit_river_states.pkl")
LEGB_JSON = os.path.join(REPO, "artifacts", "audit_river_exact.json")
OUT_JSON = os.path.join(REPO, "artifacts", "r3_miniproof_results.json")
PROG = os.path.join(REPO, "artifacts", "r3_miniproof_progress.log")

ARMS = ("Cb", "C", "D")
ARM_KNOBS = {  # (hand_conditional_infosets, sample_hero_range)
    "Cb": (False, False),
    "C": (True, False),
    "D": (True, True),
}


def log(msg: str):
    line = f"[{time.strftime('%F %T')}] [r3] {msg}"
    with open(PROG, "a") as f:
        f.write(line + "\n")
    print(line, flush=True)


# ---------------------------------------------------------------------------
# State selection — byte-replicates audit_river_exact.solve()'s deterministic pick
# ---------------------------------------------------------------------------

def _load_snap():
    raw = pickle.load(open(os.path.join(REPO, "checkpoints", "serve_blueprint_6max.pkl"), "rb"))
    return raw["strat_sum"] if isinstance(raw, dict) and "strat_sum" in raw else raw


def _select_states(max_states: int = 40):
    from pokerbot.engine.side_pots_jax import SEAT_ALLIN

    states = pickle.load(open(STATES_PKL, "rb"))
    states = [s for s in states
              if not any(int(x) == int(SEAT_ALLIN) for x in np.asarray(s["env"].seat_status))]
    states = sorted(states, key=lambda s: s["pot_bb"])
    if len(states) > max_states:
        idxs = np.linspace(0, len(states) - 1, max_states).round().astype(int)
        states = [states[i] for i in sorted(set(int(i) for i in idxs))]
    return states


def _legb_results() -> dict[int, dict]:
    d = json.load(open(LEGB_JSON))
    return {int(r["state"]): r for r in d["results"]}


def _check_selection(states, legb) -> None:
    """Every reselected state must match Leg B's recorded (deal, hero) at its index."""
    for si, st in enumerate(states):
        r = legb.get(si)
        assert r is not None, f"Leg B has no result for state {si}"
        assert int(st["deal"]) == int(r["deal"]) and int(st["hero"]) == int(r["hero"]), (
            f"state selection MISMATCH at {si}: pkl deal/hero {st['deal']}/{st['hero']} "
            f"vs Leg B {r['deal']}/{r['hero']} — selection not reproducible, ABORT")


def _legb_consistency(states, legb) -> float:
    """Recompute Arm B's loss from its stored served mixture + q_br; return max |diff| mbb."""
    worst = 0.0
    for si in range(len(states)):
        r = legb[si]
        q = {int(a): float(v) for a, v in r["q_br_chips"].items()}
        served = {int(a): float(p) for a, p in r["served"].items()}
        v_br = max(q.values())
        ev = sum(served[a] * q[a] for a in q)
        worst = max(worst, abs((v_br - ev) * MBB_PER_CHIP - float(r["ev_loss_mbb"])))
    return worst


# ---------------------------------------------------------------------------
# Per-state solve machinery (runs inside worker processes)
# ---------------------------------------------------------------------------

_G: dict = {}  # per-process lazy globals


def _init_globals():
    if _G.get("ready"):
        return
    from pokerbot.search import native_solver as _ns
    from pokerbot.search.pluribus_search import set_native_subgame, set_blueprint_blend
    from pokerbot.search.tabular_blueprint_adapter import TabularBlueprintAdapter
    from pokerbot.engine.jax_weva_encode import _POSTFLOP_NUM_BUCKETS
    from pokerbot.search.pbs_tracker import _COMBO_CARD_INDICES

    set_blueprint_blend(False)
    if not _ns.native_subgame_available():
        _ns.enable()            # loads the native ext (engine primitives + range_weights surface)
    assert _ns.native_subgame_available(), "native ext required for reach range extraction"
    set_native_subgame(False)   # the Python scalar loop is the measured object
    snap = _load_snap()
    adapter = TabularBlueprintAdapter(snap, client_bet_sizing=True)
    # Native SubgameSolver instance used ONLY for range_weights(...,"reach") — the same reach
    # model Leg B fed its reference (the deployed R1 blueprint-reach product).
    eng = _ns._build_subgame_engine(adapter, int(_POSTFLOP_NUM_BUCKETS), 64)
    assert eng is not None, "native SubgameSolver required for reach range extraction"
    idx_of = {}
    for i, (c0, c1) in enumerate(_COMBO_CARD_INDICES):
        idx_of[(c0, c1)] = i
        idx_of[(c1, c0)] = i
    _G.update(ready=True, snap=snap, eng=eng, ns=_ns, idx_of=idx_of,
              n_combos_1326=len(_COMBO_CARD_INDICES))


def _make_solver(k_iters: int, hand_cond: bool, hero_sample: bool):
    """Deployed-config solver (engine_ab_h2h._DEPLOYED_BASELINE values) on the Python loop."""
    from pokerbot.search.pluribus_search import PluribusReBeLConfig, PluribusReBeLSolver
    from pokerbot.search.pluribus_rollout import ContinuationLeafConfig

    cfg = PluribusReBeLConfig(
        max_iterations=int(k_iters),
        time_budget_ms=10 ** 9,          # iteration count is the limiter (reproducible)
        max_depth=48,
        blueprint_mix=1.0,               # deployed baseline
        two_sided=True,
        repair_deck=True,
        decouple_sampler=True,
        range_aware=True,                # native-only knob; inert on the Python loop (parity)
        range_uniform_floor=0.25,
        range_reach_full=True,           # native-only; Python arms get reach via the seeded PBS
        leaf_cache_worlds=4,
        avg_skip_frac=0.0,
        leaf_biased_continuations=False,
        leaf_bias_mult=5.0,
        serve_final_iterate=False,
        river_overbet=False,
        search_preflop=False,
        linear_averaging=True,
        per_decision_seed=True,
        rng_salt=0,
        blueprint_blend=False,
        leaf=ContinuationLeafConfig(select="uniform"),
    )
    cfg.hand_conditional_infosets = bool(hand_cond)
    cfg.sample_hero_range = bool(hero_sample)
    return PluribusReBeLSolver(_G["snap"], cfg, rng=np.random.default_rng(SEED))


def _reach_rows(env, hero: int, opp: int, floor: float = 0.25):
    """(w_hero_1326, w_opp_1326, range_mode) — Leg B's exact reach extraction + floor blend,
    mapped onto the PBSTracker 1326-combo basis (combos valid on this board; others 0)."""
    eng, ns, idx_of = _G["eng"], _G["ns"], _G["idx_of"]
    board = [int(c) for c in np.asarray(env.board) if int(c) >= 0]
    sb = ns._to_flat(env).tobytes()
    wq = eng.range_weights(sb, hero, "reach")    # opp row valid
    wq2 = eng.range_weights(sb, opp, "reach")    # hero row valid (roles swapped)

    def _seat_mat(flat, seat):
        if flat is None:
            return None
        return np.asarray(flat, dtype=np.float64).reshape(6, 52, 52)[seat]

    dead = set(board)
    combos = [(c0, c1) for c0 in range(52) for c1 in range(c0 + 1, 52)
              if c0 not in dead and c1 not in dead]
    C = len(combos)

    def _row(mat):
        row = np.zeros(_G["n_combos_1326"], dtype=np.float64)
        if mat is None:
            v = np.ones(C) / C
        else:
            v = np.array([mat[c0, c1] for (c0, c1) in combos], dtype=np.float64)
            s = v.sum()
            v = v / s if s > 1e-12 else np.ones(C) / C
        v = (1.0 - floor) * v + floor * (np.ones(C) / C)  # Leg B's deployed floor blend
        v /= v.sum()
        for k, (c0, c1) in enumerate(combos):
            row[idx_of[(c0, c1)]] = v[k]
        return row

    mode = "reach" if (_seat_mat(wq, opp) is not None) else "uniform_fallback"
    return _row(_seat_mat(wq2, hero)), _row(_seat_mat(wq, opp)), mode


def _make_pbs(env, hero: int, w_hero_row, w_opp_row, opp: int, seed_hero_row: bool):
    from pokerbot.engine.cards import Card
    from pokerbot.search.pbs_tracker import PBSTracker

    _RANKS = "23456789TJQKA"
    _SUITS = "cdhs"

    def _card(idx: int) -> Card:
        return Card.from_str(_RANKS[int(idx) // 4] + _SUITS[int(idx) % 4])

    board = [_card(int(c)) for c in np.asarray(env.board) if int(c) >= 0]
    hero_hole = (_card(int(env.holes[hero][0])), _card(int(env.holes[hero][1])))
    pbs = PBSTracker.init_uniform(hero, hero_hole, board)
    pbs.beliefs[opp] = w_opp_row.copy()
    if seed_hero_row:
        pbs.beliefs[hero] = w_hero_row.copy()  # consumed only by sample_hero_range (arm D)
    return pbs


def _serve_arm(st, arm: str, k_iters: int):
    """Run one arm's Python-loop solve at this state; return (probs, iterations, wall_s)."""
    from pokerbot.search.state_bridge import SubgameRoot
    from pokerbot.search.native_solver import legal_action_mask_np

    env, hero, opp = st["env"], int(st["hero"]), int(st["opp"])
    hand_cond, hero_sample = ARM_KNOBS[arm]
    solver = _make_solver(k_iters, hand_cond, hero_sample)
    w_hero_row, w_opp_row, mode = _G[f"reach_{st['deal']}_{hero}"]
    pbs = _make_pbs(env, hero, w_hero_row, w_opp_row, opp, seed_hero_row=hero_sample)
    legal = legal_action_mask_np(env, hero).astype(bool)
    root = SubgameRoot(env=env, hero_seat=hero,
                       pbs_vector=pbs.get_public_belief_state(), legal_mask=legal)
    t0 = time.time()
    res = solver.solve(root, pbs)
    return (np.asarray(res.action_probs, dtype=np.float64), int(res.iterations),
            time.time() - t0, mode)


def _ev_loss(probs, legb_res, pot_root_chips: float):
    q = {int(a): float(v) for a, v in legb_res["q_br_chips"].items()}
    acts = sorted(q)
    served = {a: float(probs[a]) for a in acts}
    ssum = sum(served.values())
    if ssum > 1e-9:
        served = {a: p / ssum for a, p in served.items()}
    v_br = max(q.values())
    ev = sum(served[a] * q[a] for a in acts)
    loss = v_br - ev
    return {
        "ev_loss_chips": round(loss, 4),
        "ev_loss_mbb": round(loss * MBB_PER_CHIP, 3),
        "ev_loss_pct_pot": round(100.0 * loss / pot_root_chips, 3),
        "served": {str(a): round(p, 4) for a, p in served.items()},
    }


def solve_state(args):
    """Worker entry: (si, arms_needed, k_iters) -> per-arm result dict."""
    si, arms_needed, k_iters = args
    _init_globals()
    states = _select_states(40)
    legb = _legb_results()
    st = states[si]
    env, hero, opp = st["env"], int(st["hero"]), int(st["opp"])
    rkey = f"reach_{st['deal']}_{hero}"
    if rkey not in _G:
        _G[rkey] = _reach_rows(env, hero, opp)
    pot_root = float(int(env.pot_dead) + int(np.asarray(env.committed_total).sum()))
    out = {"state": si, "deal": int(st["deal"]), "hero": hero, "pot_bb": st["pot_bb"],
           "facing_bet": bool(st["facing_bet"]), "range_mode": _G[rkey][2],
           "k_iters": int(k_iters), "seed": SEED, "arms": {}}
    for arm in arms_needed:
        probs, iters, wall, _mode = _serve_arm(st, arm, k_iters)
        r = _ev_loss(probs, legb[si], pot_root)
        r.update(iterations=iters, wall_s=round(wall, 1))
        out["arms"][arm] = r
    return out


# ---------------------------------------------------------------------------
# Identity gate: knob-absent vs knob-off must be byte-identical
# ---------------------------------------------------------------------------

def identity() -> int:
    from pokerbot.search.pluribus_search import PluribusReBeLConfig, PluribusReBeLSolver
    from pokerbot.search.pluribus_rollout import ContinuationLeafConfig
    from pokerbot.search.state_bridge import SubgameRoot
    from pokerbot.search.native_solver import legal_action_mask_np

    _init_globals()
    states = _select_states(40)
    st = states[0]
    env, hero, opp = st["env"], int(st["hero"]), int(st["opp"])
    _G[f"reach_{st['deal']}_{hero}"] = _reach_rows(env, hero, opp)
    w_hero_row, w_opp_row, _ = _G[f"reach_{st['deal']}_{hero}"]
    legal = legal_action_mask_np(env, hero).astype(bool)

    def _run(tag: str, set_knobs: bool):
        cfg = PluribusReBeLConfig(
            max_iterations=400, time_budget_ms=10 ** 9, blueprint_mix=1.0, two_sided=True,
            repair_deck=True, decouple_sampler=True, range_uniform_floor=0.25,
            leaf_cache_worlds=4, linear_averaging=True, per_decision_seed=True,
            blueprint_blend=False, leaf=ContinuationLeafConfig(select="uniform"),
        )
        if set_knobs:  # explicit OFF (vs absent = dataclass defaults)
            cfg.hand_conditional_infosets = False
            cfg.sample_hero_range = False
        sv = PluribusReBeLSolver(_G["snap"], cfg, rng=np.random.default_rng(SEED))
        pbs = _make_pbs(env, hero, w_hero_row, w_opp_row, opp, seed_hero_row=False)
        root = SubgameRoot(env=env, hero_seat=hero,
                           pbs_vector=pbs.get_public_belief_state(), legal_mask=legal)
        res = sv.solve(root, pbs)
        keys = sorted(k for k in sv._regrets.keys() if isinstance(k, str))
        return np.asarray(res.action_probs, dtype=np.float64), keys

    p_abs, k_abs = _run("absent", False)
    p_off, k_off = _run("off", True)
    same_probs = bool(np.array_equal(p_abs, p_off))
    same_keys = k_abs == k_off
    no_hand_suffix = not any("|h" in k for k in k_off)
    log(f"identity gate: probs_equal={same_probs} keys_equal={same_keys} "
        f"({len(k_off)} infosets) no_hand_suffix={no_hand_suffix}")
    # And knob ON must actually change keys (dead-knob trap):
    cfgv = _make_solver(400, True, False)
    pbs = _make_pbs(env, hero, w_hero_row, w_opp_row, opp, seed_hero_row=False)
    root = SubgameRoot(env=env, hero_seat=hero,
                       pbs_vector=pbs.get_public_belief_state(), legal_mask=legal)
    cfgv.solve(root, pbs)
    on_keys = [k for k in cfgv._regrets.keys() if isinstance(k, str)]
    n_hand = sum(1 for k in on_keys if "|h" in k)
    hero_hand = any(k.startswith(f"{hero}|") and "|h" in k for k in on_keys)
    log(f"identity gate: knob-ON hand-keyed infosets={n_hand}/{len(on_keys)} "
        f"(hero hand-keyed: {hero_hand} — must be False)")
    ok = same_probs and same_keys and no_hand_suffix and n_hand > 0 and not hero_hand
    log(f"identity gate: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


# ---------------------------------------------------------------------------
# Run / verdict
# ---------------------------------------------------------------------------

def _summarize(results: list[dict], legb: dict[int, dict]) -> dict:
    per_arm = {}
    done = [r for r in results if all(a in r["arms"] for a in ARMS)]
    sis = [r["state"] for r in done]
    b_losses = np.array([float(legb[si]["ev_loss_mbb"]) for si in sis])
    per_arm["B"] = {"n": len(sis), "mean_mbb": float(b_losses.mean()) if len(sis) else None,
                    "median_mbb": float(np.median(b_losses)) if len(sis) else None}
    for arm in ARMS:
        v = np.array([float(r["arms"][arm]["ev_loss_mbb"]) for r in done])
        per_arm[arm] = {"n": len(v), "mean_mbb": float(v.mean()) if len(v) else None,
                        "median_mbb": float(np.median(v)) if len(v) else None}
    out = {"n_states_complete": len(done), "per_arm": per_arm}
    if len(done):
        medB = per_arm["B"]["median_mbb"]
        for arm in ARMS:
            out[f"median_drop_vs_B_{arm}_pct"] = round(
                100.0 * (medB - per_arm[arm]["median_mbb"]) / medB, 2) if medB else None
        best = max(out["median_drop_vs_B_C_pct"], out["median_drop_vs_B_D_pct"])
        out["verdict"] = ("PASS (>=30% median drop)" if best >= 30.0
                          else ("KILL (<15% median drop)" if best < 15.0
                                else "GRAY (15-30%)"))
    return out


def run(k_iters: int, workers: int, resume: bool) -> int:
    import concurrent.futures as cf

    _init_globals()
    states = _select_states(40)
    legb = _legb_results()
    _check_selection(states, legb)
    cons = _legb_consistency(states, legb)
    log(f"run: 40 states selected (deal/hero match Leg B); Arm-B q_br consistency "
        f"max|diff|={cons:.3f} mbb (rounding-level expected)")
    # Tolerance: audit_river_exact.json stores served probs at 4 decimals and q_br at 0.01 chips;
    # a 4-decimal prob x a ~1600-chip q spread accumulates to ~5-8 mbb on the widest state —
    # measured 5.695 mbb max. Negligible vs the 609 mbb median under test; gate at 10 mbb.
    assert cons < 10.0, "Leg B stored losses do not reproduce from q_br — metric mismatch, ABORT"

    results: dict[int, dict] = {}
    if resume and os.path.exists(OUT_JSON):
        try:
            prev = json.load(open(OUT_JSON))
            for r in prev.get("results", []):
                si = int(r["state"])
                if int(states[si]["deal"]) == int(r["deal"]) and int(r.get("k_iters", -1)) == k_iters:
                    results[si] = r
        except Exception:
            results = {}
        n_arms = sum(len(r["arms"]) for r in results.values())
        log(f"resume: {len(results)} states with {n_arms} arm results reloaded")

    todo = []
    for si in range(len(states)):
        have = set(results.get(si, {}).get("arms", {}).keys())
        for a in ARMS:
            if a not in have:
                todo.append((si, (a,), k_iters))
    # Big-pot states first (states are pot-ascending; big pots have SMALL river trees => cheap):
    # banks most of the sample early and pushes the deep small-pot raise-war trees to the tail.
    todo.sort(key=lambda t: -t[0])
    log(f"run: {len(todo)} (state, arm) tasks, K={k_iters}, workers={workers}")

    def _flush():
        rs = [results[k] for k in sorted(results)]
        payload = {"config": {"k_iters": k_iters, "seed": SEED, "arms": list(ARMS),
                              "deployed_arm_B_source": "artifacts/audit_river_exact.json",
                              "workers": workers},
                   "summary": _summarize(rs, legb), "results": rs}
        with open(OUT_JSON, "w") as f:
            json.dump(payload, f, indent=1)

    t0 = time.time()
    n_done = 0
    if todo:
        with cf.ProcessPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(solve_state, t): t for t in todo}
            for fut in cf.as_completed(futs):
                si = futs[fut][0]
                r = fut.result()
                if si in results:
                    results[si]["arms"].update(r["arms"])
                else:
                    results[si] = r
                n_done += 1
                el = time.time() - t0
                eta = el / n_done * (len(todo) - n_done) / 60.0
                arms_s = " ".join(
                    f"{a}={r['arms'][a]['ev_loss_mbb']}mbb({r['arms'][a]['wall_s']}s)"
                    for a in r["arms"])
                log(f"state {si} ({n_done}/{len(todo)} tasks): pot {r['pot_bb']}bb "
                    f"B={legb[si]['ev_loss_mbb']}mbb {arms_s} | ETA {eta:.0f}min")
                _flush()
    _flush()
    s = _summarize([results[k] for k in sorted(results)], legb)
    log(f"run DONE: {json.dumps(s)}")
    return 0


def verdict() -> int:
    legb = _legb_results()
    d = json.load(open(OUT_JSON))
    s = _summarize(d["results"], legb)
    print(json.dumps(s, indent=1))
    return 0


def smoke(si: int) -> int:
    _init_globals()
    states = _select_states(40)
    legb = _legb_results()
    _check_selection(states, legb)
    r = solve_state((si, ARMS, 2000))
    log(f"smoke state {si}: B={legb[si]['ev_loss_mbb']}mbb "
        + " ".join(f"{a}={r['arms'][a]['ev_loss_mbb']}mbb({r['arms'][a]['wall_s']}s)"
                   for a in ARMS))
    return 0


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    cmd = args[0] if args else "run"
    if cmd == "identity":
        sys.exit(identity())
    elif cmd == "smoke":
        sys.exit(smoke(int(args[1]) if len(args) > 1 else 0))
    elif cmd == "verdict":
        sys.exit(verdict())
    else:
        k = K_DEFAULT
        w = 4
        for a in sys.argv[1:]:
            if a.startswith("--k="):
                k = int(a.split("=", 1)[1])
            if a.startswith("--workers="):
                w = int(a.split("=", 1)[1])
        sys.exit(run(k, w, resume="--resume" in sys.argv[1:]))
