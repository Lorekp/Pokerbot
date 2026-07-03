#!/usr/bin/env python3
"""LEG B (decision-quality audit): river EXACT-solve EV loss of the deployed serve.

River subgames are terminal-exact under the deployed 11-action abstraction: no chance nodes below,
no depth-limit leaves — every line ends in a fold or a showdown whose payoff is a pure function of
the public betting line + the two hole combos. So for heads-up river decision nodes reached by REAL
GAME-FLOW deals we can compute a CONVERGED reference equilibrium of the (range vs range) river
subgame with vectorized CFR+ and measure, exactly within the abstraction + range model:

    per-decision EV loss = BR_value(hero combo | reference opp strategy)
                           - sum_a served_probs[a] * Q_BR(a | hero combo)

where served_probs is the DEPLOYED serve (blueprint preflop + native two-sided K=2000 search — the
exact engine_ab_h2h `{}` baseline built by its _build_policy), and Q_BR(a) is the value of taking a
at the root and playing best response afterwards against the reference equilibrium opponent.

RANGES (the audit's central caveat): hero + opponent ranges are the engine's OWN reach-range model
(SubgameSolver.range_weights mode "reach" — the R1 blueprint-reach product with fold-now fallback),
blended with the deployed 25% uniform floor. The reference is therefore "optimal play GIVEN our own
range model + action abstraction", not oracle-GTO. Deviation numbers measure the search/serve, not
the range model.

Usage:
  python scripts/audit_river_exact.py collect [n_states]   -> artifacts/audit_river_states.pkl
  python scripts/audit_river_exact.py solve  [max_states]  -> artifacts/audit_river_exact.json
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

FOLD, CHECK, CALL = 0, 1, 2
BET_BUCKET_START, ALL_IN, NUM_ACTIONS = 3, 10, 11
CHIP_PER_BB = 100.0
MBB_PER_CHIP = 10.0  # 1 chip = 0.01 bb = 10 mbb

STATES_PKL = os.path.join(REPO, "artifacts", "audit_river_states.pkl")
OUT_JSON = os.path.join(REPO, "artifacts", "audit_river_exact.json")
PROG = os.path.join(REPO, "artifacts", "decision_quality_progress.log")


def log(msg: str):
    with open(PROG, "a") as f:
        f.write(f"[{time.strftime('%F %T')}] [legB] {msg}\n")
    print(msg, flush=True)


# ---------------------------------------------------------------------------
# Shared setup
# ---------------------------------------------------------------------------

def _load_snap():
    raw = pickle.load(open(os.path.join(REPO, "checkpoints", "serve_blueprint_6max.pkl"), "rb"))
    return raw["strat_sum"] if isinstance(raw, dict) and "strat_sum" in raw else raw


def _allowed_post():
    from pokerbot.search.tabular_blueprint_adapter import _build_allowed_action_mask
    return _build_allowed_action_mask((0, 1, 3, 4, 5))  # the deployed postflop keep-list


# ---------------------------------------------------------------------------
# COLLECT: game-flow walk (blueprint-sampled actions) to HU river decision nodes
# ---------------------------------------------------------------------------

def collect(n_states: int) -> int:
    import jax
    from pokerbot.engine.jax_engine import init_hand
    from pokerbot.engine.np_engine import (
        apply_action_np, current_player_np, is_chance_node_np, is_terminal_np,
        legal_action_mask_np, sample_chance_np, to_numpy_state,
    )
    from pokerbot.eval.solver_exploitability import make_blueprint_policy_fn
    from pokerbot.engine.side_pots_jax import SEAT_LIVE

    snap = _load_snap()
    bp = make_blueprint_policy_fn(snap)
    rng = np.random.default_rng(20260703)
    occ = jax.numpy.array([True] * 6)

    states = []
    t0 = time.time()
    d = 0
    while len(states) < n_states and d < 40000:
        d += 1
        e = to_numpy_state(init_hand(jax.random.PRNGKey(880001 + d), occupancy=occ))
        guard = 0
        picked = False
        while not is_terminal_np(e) and guard < 120 and not picked:
            guard += 1
            if is_chance_node_np(e):
                e = sample_chance_np(e)
                continue
            s = current_player_np(e)
            if s < 0:
                break
            nboard = int((np.asarray(e.board) >= 0).sum())
            live = [i for i in range(6) if int(e.seat_status[i]) == int(SEAT_LIVE)]
            if nboard == 5 and len(live) == 2 and int(s) in live:
                # HU river decision node reached by game flow
                pot = (int(e.pot_dead) + int(np.asarray(e.committed_total).sum())) / CHIP_PER_BB
                to_call = (int(e.current_bet) - int(np.asarray(e.committed_street)[int(s)]))
                states.append({
                    "deal": d, "env": e, "hero": int(s),
                    "opp": [x for x in live if x != int(s)][0],
                    "pot_bb": round(pot, 2), "facing_bet": bool(to_call > 0),
                })
                picked = True
                break
            # blueprint-sampled game-flow action
            m = legal_action_mask_np(e, s).astype(bool)
            if m.sum() <= 1:
                a = int(np.argmax(m))
            else:
                p = np.asarray(bp(e, s), dtype=np.float64) * m
                ssum = p.sum()
                p = p / ssum if ssum > 1e-12 else m.astype(np.float64) / m.sum()
                a = int(rng.choice(NUM_ACTIONS, p=p))
            e = apply_action_np(e, a)
        if d % 500 == 0:
            log(f"collect: deal {d}, states {len(states)}/{n_states} "
                f"({time.time() - t0:.0f}s)")

    with open(STATES_PKL, "wb") as f:
        pickle.dump(states, f)
    pots = [s["pot_bb"] for s in states]
    log(f"collect DONE: {len(states)} HU river states from {d} deals; "
        f"pot bb min/med/max = {min(pots):.1f}/{sorted(pots)[len(pots)//2]:.1f}/{max(pots):.1f}; "
        f"facing_bet = {sum(s['facing_bet'] for s in states)}/{len(states)}; wrote {STATES_PKL}")
    return 0


# ---------------------------------------------------------------------------
# Exact river subgame solver (vectorized CFR+ over hole combos)
# ---------------------------------------------------------------------------

class RiverTree:
    """Public river betting tree under the deployed abstraction (engine transitions)."""

    def __init__(self, root_env, hero: int, opp: int, allowed_post: np.ndarray,
                 max_nodes=400_000, raise_cap: int = 4):
        from pokerbot.engine.np_engine import (
            apply_action_np, current_player_np, is_terminal_np, legal_action_mask_np,
            is_chance_node_np, sample_chance_np,
        )
        self._chance = is_chance_node_np
        self._sample = sample_chance_np
        self.hero, self.opp = hero, opp
        self.nodes = []      # dict: type, actor, actions, children, or terminal payoff spec
        self.max_nodes = max_nodes
        self._apply = apply_action_np
        self._cur = current_player_np
        self._term = is_terminal_np
        self._mask = legal_action_mask_np
        self._allowed = allowed_post
        ct = np.asarray(root_env.committed_total)
        self.ct_root = {hero: int(ct[hero]), opp: int(ct[opp])}
        self.overflow = False
        # RAISE-WAR CAP: after `raise_cap` aggressive actions in the subgame, the menu collapses to
        # {fold, call, all-in}. Without this, small-pot rivers explode (>400k public nodes: the
        # 1/3-pot bucket stays min-raise-legal for many levels). Deployed play puts ~zero mass on
        # deep raise-war lines; the cap's EV effect is checked by a sensitivity run (cap k vs k-1).
        self.raise_cap = int(raise_cap)
        self.root = self._build(root_env, 0)

    def _terminal_spec(self, env):
        """(kind, payoff params). kind: 'fold' (someone folded) or 'showdown'.

        Deltas are measured FROM THE ROOT decision point (chips added after the root are the only
        decision-relevant quantities; the root pot is common to every terminal and goes to the
        winner). hero_delta/opp_delta: the terminal stack change from the root onward.
        """
        ct = np.asarray(env.committed_total)
        pot_T = int(env.pot_dead) + int(ct.sum())
        h_add = int(ct[self.hero]) - self.ct_root[self.hero]
        o_add = int(ct[self.opp]) - self.ct_root[self.opp]
        from pokerbot.engine.side_pots_jax import SEAT_FOLDED
        h_folded = int(env.seat_status[self.hero]) == int(SEAT_FOLDED)
        o_folded = int(env.seat_status[self.opp]) == int(SEAT_FOLDED)
        if h_folded:
            return {"t": "fold", "hero_delta": float(-h_add), "opp_delta": float(pot_T - o_add)}
        if o_folded:
            return {"t": "fold", "hero_delta": float(pot_T - h_add), "opp_delta": float(-o_add)}
        # showdown: win -> receive pot_T (delta = pot_T - add); tie -> pot_T/2 - add; lose -> -add.
        return {"t": "showdown", "pot_T": float(pot_T), "h_add": float(h_add),
                "o_add": float(o_add)}

    def _build(self, env, n_raises: int):
        if len(self.nodes) > self.max_nodes:
            self.overflow = True
            return -1
        # Drain chance/settle nodes (betting-closed on the river -> pending settle; deterministic:
        # no cards remain to deal, sample_chance only advances to the terminal/showdown state).
        guard = 0
        while (not self._term(env)) and self._chance(env) and guard < 8:
            env = self._sample(env)
            guard += 1
        idx = len(self.nodes)
        self.nodes.append(None)  # placeholder
        if self._term(env) or int(env.street) != 3:
            # street change impossible on river except to terminal; treat both as terminal
            self.nodes[idx] = self._terminal_spec(env)
            return idx
        actor = int(self._cur(env))
        m = self._mask(env, actor).astype(bool) & self._allowed.astype(bool)
        if not m.any():
            m = self._mask(env, actor).astype(bool)
        if n_raises >= self.raise_cap:
            for a in range(BET_BUCKET_START, ALL_IN):
                m[a] = False  # raise war capped: fold/check/call/all-in only
        actions = [int(a) for a in np.flatnonzero(m)]
        children = []
        for a in actions:
            if self.overflow:
                break
            aggressive = a >= BET_BUCKET_START  # bucket bets + all-in
            children.append(self._build(self._apply(env, a), n_raises + (1 if aggressive else 0)))
        self.nodes[idx] = {"t": "dec", "actor": actor, "actions": actions, "children": children}
        return idx


class RiverCFR:
    """Vectorized CFR+ (alternating, linear averaging) over valid combos for both players."""

    def __init__(self, tree: RiverTree, combos: np.ndarray, ranks: np.ndarray,
                 w_hero: np.ndarray, w_opp: np.ndarray):
        self.tree = tree
        self.combos = combos          # (C,2) int card ids
        self.C = combos.shape[0]
        self.ranks = ranks            # (C,) int hand rank (higher wins)
        self.w = {tree.hero: w_hero, tree.opp: w_opp}   # root range weights (sum 1)
        # regrets/strategy sums per decision node: (n_actions, C) for the actor
        self.regret = {}
        self.ssum = {}
        for i, nd in enumerate(tree.nodes):
            if nd["t"] == "dec":
                n = len(nd["actions"])
                self.regret[i] = np.zeros((n, self.C))
                self.ssum[i] = np.zeros((n, self.C))
        # ---- showdown machinery: O(C*52) sorted-rank prefix sums (state-level, terminal-shared) --
        # pair index lookup: map (c0,c1) -> combo idx
        self.pair_idx = {}
        for i in range(self.C):
            self.pair_idx[(int(combos[i, 0]), int(combos[i, 1]))] = i
        # group combos by strictly increasing rank
        uniq, self.grp_of = np.unique(ranks, return_inverse=True)   # grp_of: (C,) group id
        self.G = uniq.shape[0]
        self.c0 = combos[:, 0]
        self.c1 = combos[:, 1]

    def _sw(self, wr: np.ndarray):
        """Per-combo (W_lt, W_eq, W_gt): opp-reach mass with lower/equal/higher rank, EXCLUDING
        combos sharing a card with combo i. Exact card removal via per-card group prefix sums."""
        G, C = self.G, self.C
        g = self.grp_of
        # group totals + per-card per-group sums (bincount: ~5-10x faster than np.add.at)
        grp_w = np.bincount(g, weights=wr, minlength=G)
        grp_pc = (np.bincount(g * 52 + self.c0, weights=wr, minlength=G * 52)
                  + np.bincount(g * 52 + self.c1, weights=wr, minlength=G * 52)).reshape(G, 52)
        cum_w = np.concatenate(([0.0], np.cumsum(grp_w)))            # exclusive prefix per group
        cum_pc = np.vstack([np.zeros((1, 52)), np.cumsum(grp_pc, axis=0)])
        tot_w = cum_w[-1]
        tot_pc = cum_pc[-1]
        a, b_ = self.c0, self.c1
        # lower-rank mass excluding overlaps: no other combo contains BOTH a and b
        w_lt = cum_w[g] - cum_pc[g, a] - cum_pc[g, b_]
        # equal-rank mass excluding overlaps and i itself
        w_eq = grp_w[g] - grp_pc[g, a] - grp_pc[g, b_] + wr
        # total compatible mass
        w_all = tot_w - tot_pc[a] - tot_pc[b_] + wr
        w_gt = w_all - w_lt - w_eq
        return w_lt, w_eq, w_gt, w_all

    # ---- terminal counterfactual values (per traverser combo) ----
    def _term_cfv(self, nd, trav: int, opp_reach: np.ndarray) -> np.ndarray:
        t = self.tree
        hero_is_trav = (trav == t.hero)
        if nd["t"] == "fold":
            v = nd["hero_delta"] if hero_is_trav else nd["opp_delta"]
            # compatible opp mass per trav combo (card removal on the reach vector)
            pc = (np.bincount(self.c0, weights=opp_reach, minlength=52)
                  + np.bincount(self.c1, weights=opp_reach, minlength=52))
            w = opp_reach.sum() - pc[self.c0] - pc[self.c1] + opp_reach
            return v * w
        # showdown
        pot_T, h_add, o_add = nd["pot_T"], nd["h_add"], nd["o_add"]
        if hero_is_trav:
            win, lose, tie = pot_T - h_add, -h_add, pot_T / 2.0 - h_add
        else:
            win, lose, tie = pot_T - o_add, -o_add, pot_T / 2.0 - o_add
        w_lt, w_eq, w_gt, _ = self._sw(opp_reach)
        return win * w_lt + lose * w_gt + tie * w_eq

    def _strategy(self, i) -> np.ndarray:
        r = np.maximum(self.regret[i], 0.0)
        tot = r.sum(axis=0)
        n = r.shape[0]
        st = np.where(tot > 1e-12, r / np.maximum(tot, 1e-12), 1.0 / n)
        return st

    def _walk(self, i, trav, my_reach, opp_reach, t_iter):
        nd = self.tree.nodes[i]
        if nd["t"] != "dec":
            return self._term_cfv(nd, trav, opp_reach)
        actor = nd["actor"]
        st = self._strategy(i)
        if actor == trav:
            n = len(nd["actions"])
            cf = np.zeros((n, self.C))
            for k, ch in enumerate(nd["children"]):
                cf[k] = self._walk(ch, trav, my_reach * st[k], opp_reach, t_iter)
            ev = (st * cf).sum(axis=0)
            # regret matching+ update
            self.regret[i] = np.maximum(self.regret[i] + (cf - ev[None, :]), 0.0)
            # linear-averaged strategy sum weighted by traverser reach
            self.ssum[i] += t_iter * my_reach[None, :] * st
            return ev
        else:
            v = np.zeros(self.C)
            for k, ch in enumerate(nd["children"]):
                v += self._walk(ch, trav, my_reach, opp_reach * st[k], t_iter)
            return v

    def run(self, iters: int, check_every=200, eps_chips=0.2, max_seconds=600.0):
        t = self.tree
        t0 = time.time()
        hist = []
        for it in range(1, iters + 1):
            for trav in (t.hero, t.opp):
                other = t.opp if trav == t.hero else t.hero
                self._walk(t.root, trav, np.ones(self.C), self.w[other].copy(), float(it))
            if it % check_every == 0 or it == iters or (time.time() - t0) > max_seconds:
                e_h, e_o, gv = self.exploitability()
                hist.append((it, e_h + e_o))
                if (e_h + e_o) < eps_chips or (time.time() - t0) > max_seconds:
                    return it, e_h, e_o, gv, hist
        e_h, e_o, gv = self.exploitability()
        return iters, e_h, e_o, gv, hist

    def avg_strategy(self, i) -> np.ndarray:
        s = self.ssum[i]
        tot = s.sum(axis=0)
        n = s.shape[0]
        return np.where(tot > 1e-12, s / np.maximum(tot, 1e-12), 1.0 / n)

    def _br_walk(self, i, br_player, opp_reach):
        """BR values for br_player vs the AVERAGE strategy; opp_reach carries avg-strat products."""
        nd = self.tree.nodes[i]
        if nd["t"] != "dec":
            return self._term_cfv(nd, br_player, opp_reach)
        if nd["actor"] == br_player:
            vals = [self._br_walk(ch, br_player, opp_reach) for ch in nd["children"]]
            return np.max(np.stack(vals), axis=0)
        st = self.avg_strategy(i)
        v = np.zeros(self.C)
        for k, ch in enumerate(nd["children"]):
            v += self._br_walk(ch, br_player, opp_reach * st[k])
        return v

    def exploitability(self):
        """(exploit_hero_side, exploit_opp_side, game_value_hero) in AVERAGE chips per hand,
        weighting each side's BR values by its own root range."""
        t = self.tree
        # game value of avg profile for hero:
        gv = self._ev_avg(t.root, np.ones(self.C), self.w[t.opp].copy(), t.hero)
        gv_h = float(self.w[t.hero] @ gv)
        br_h = self._br_walk(t.root, t.hero, self.w[t.opp].copy())
        br_o = self._br_walk(t.root, t.opp, self.w[t.hero].copy())
        v_br_h = float(self.w[t.hero] @ br_h)
        v_br_o = float(self.w[t.opp] @ br_o)
        gv_o = self._ev_avg_opp(gv_h)
        return v_br_h - gv_h, v_br_o - gv_o, gv_h

    def _ev_avg(self, i, my_reach, opp_reach, player):
        nd = self.tree.nodes[i]
        if nd["t"] != "dec":
            return self._term_cfv(nd, player, opp_reach)
        st = self.avg_strategy(i)
        if nd["actor"] == player:
            v = np.zeros(self.C)
            for k, ch in enumerate(nd["children"]):
                v += st[k] * self._ev_avg(ch, my_reach, opp_reach, player)
            return v
        v = np.zeros(self.C)
        for k, ch in enumerate(nd["children"]):
            v += self._ev_avg(ch, my_reach, opp_reach * st[k], player)
        return v

    def _ev_avg_opp(self, gv_h):
        # zero-sum in DELTAS? hero_delta + opp_delta = pot_root at every terminal (the root pot is
        # distributed). So EV_opp = pot_root_mass - EV_hero under matched reach weighting — but the
        # per-combo weights differ. Compute directly instead:
        t = self.tree
        gvo = self._ev_avg(t.root, np.ones(self.C), self.w[t.hero].copy(), t.opp)
        return float(self.w[t.opp] @ gvo)


# ---------------------------------------------------------------------------
# SOLVE: per collected state, reference equilibrium + deployed-serve EV loss
# ---------------------------------------------------------------------------

def solve(max_states: int, resume: bool = False) -> int:
    from pokerbot.engine.np_engine import evaluate_seven
    from pokerbot.search import native_solver as _ns
    from pokerbot.search.pluribus_search import set_native_subgame
    from pokerbot.engine.jax_weva_encode import _POSTFLOP_NUM_BUCKETS
    import engine_ab_h2h as ab

    if not _ns.native_subgame_available():
        _ns.enable()
    assert _ns.native_subgame_available(), "native SubgameSolver required for the deployed serve"
    set_native_subgame(True)

    snap = _load_snap()
    # THE DEPLOYED SERVE — engine_ab_h2h `{}` baseline via its own _build_policy (reuse, not copy).
    pf = ab._build_policy({}, snap, 4242, REPO)
    sv = [c.cell_contents for c in pf.__closure__
          if hasattr(getattr(c, "cell_contents", None), "config")][0]
    eng = _ns._build_subgame_engine(sv.adapter, int(_POSTFLOP_NUM_BUCKETS),
                                    int(sv.config.leaf.max_playout_depth))
    assert eng is not None

    allowed = _allowed_post()
    states = pickle.load(open(STATES_PKL, "rb"))
    # strict HU: exclude states with an earlier-street all-in seat still in the hand (multiway
    # showdown breaks the 2-player zero-sum subgame model)
    from pokerbot.engine.side_pots_jax import SEAT_ALLIN
    n0 = len(states)
    states = [s for s in states
              if not any(int(x) == int(SEAT_ALLIN) for x in np.asarray(s["env"].seat_status))]
    log(f"solve: {len(states)}/{n0} collected states strictly-HU, auditing up to {max_states}")

    # stratified pick: sort by pot, take evenly spaced -> covers pot spectrum
    states = sorted(states, key=lambda s: s["pot_bb"])
    if len(states) > max_states:
        idxs = np.linspace(0, len(states) - 1, max_states).round().astype(int)
        states = [states[i] for i in sorted(set(int(i) for i in idxs))]

    # ---- resume-from-partial: state selection above is DETERMINISTIC (fixed pkl on disk,
    # stable sort by pot_bb, linspace pick), so completed indices can be reloaded verbatim.
    # Safety: every reloaded result's deal id must match the reselected state at that index;
    # any mismatch aborts the resume and restarts the leg cleanly.
    done: dict[int, dict] = {}
    if resume and os.path.exists(OUT_JSON):
        try:
            prev = json.load(open(OUT_JSON))
            for r in prev.get("results", []):
                done[int(r["state"])] = r
        except Exception:
            done = {}
        for dsi, r in sorted(done.items()):
            if dsi >= len(states) or int(states[dsi]["deal"]) != int(r["deal"]) \
                    or int(states[dsi]["hero"]) != int(r["hero"]):
                log(f"resume MISMATCH at state {dsi} (deal {r['deal']}) — "
                    f"selection not reproducible, restarting leg cleanly")
                done = {}
                break
        if done:
            log(f"resume: {len(done)} states already solved "
                f"(indices {sorted(done)}), skipping them")

    results = []
    floor = float(sv.config.range_uniform_floor)  # deployed 0.25
    t_all = time.time()

    for si, st in enumerate(states):
        if si in done:
            results.append(done[si])
            continue
        env, hero, opp = st["env"], st["hero"], st["opp"]
        t0 = time.time()
        board = [int(c) for c in np.asarray(env.board) if int(c) >= 0]
        hero_hole = (int(env.holes[hero][0]), int(env.holes[hero][1]))

        # ---- ranges from the engine's own reach model ----
        sb = _ns._to_flat(env).tobytes()
        wq = eng.range_weights(sb, hero, "reach")          # opp row valid
        wq2 = eng.range_weights(sb, opp, "reach")          # hero row valid (roles swapped)

        def _seat_mat(flat, seat):
            if flat is None:
                return None
            a = np.asarray(flat, dtype=np.float64).reshape(6, 52, 52)
            return a[seat]

        w_opp_raw = _seat_mat(wq, opp)
        w_hero_raw = _seat_mat(wq2, hero)

        # combos valid on this board
        dead = set(board)
        combos = []
        for c0 in range(52):
            if c0 in dead:
                continue
            for c1 in range(c0 + 1, 52):
                if c1 in dead:
                    continue
                combos.append((c0, c1))
        combos = np.asarray(combos, dtype=np.int64)
        C = combos.shape[0]

        def _range_vec(mat):
            if mat is None:
                return np.ones(C) / C
            v = mat[combos[:, 0], combos[:, 1]]
            s = v.sum()
            base = v / s if s > 1e-12 else np.ones(C) / C
            return (1.0 - floor) * base + floor * (np.ones(C) / C)

        w_opp = _range_vec(w_opp_raw)
        w_hero = _range_vec(w_hero_raw)
        w_opp /= w_opp.sum()
        w_hero /= w_hero.sum()

        # ---- ranks ----
        ranks = np.empty(C, dtype=np.int64)
        b = np.asarray(board, dtype=np.int32)
        for i in range(C):
            seven = np.concatenate([combos[i].astype(np.int32), b])
            ranks[i] = int(evaluate_seven(seven))

        # ---- tree (adaptive raise cap: largest cap whose public tree stays solvable) ----
        tree, cap_used = None, None
        for cap in (4, 3, 2):
            t_try = RiverTree(env, hero, opp, allowed, max_nodes=100_000, raise_cap=cap)
            if not t_try.overflow and len(t_try.nodes) <= 6000:
                tree, cap_used = t_try, cap
                break
        if tree is None:
            log(f"state {si}: tree > 6000 nodes even at raise_cap=2 — skipped")
            continue
        n_dec = sum(1 for n in tree.nodes if n["t"] == "dec")

        cfr = RiverCFR(tree, combos, ranks, w_hero, w_opp)
        iters, e_h, e_o, gv, hist = cfr.run(12000, check_every=200, eps_chips=0.1,
                                            max_seconds=240.0)
        pot_root = float(int(env.pot_dead) + int(np.asarray(env.committed_total).sum()))

        # ---- deployed serve at this node ----
        probs_dep = np.asarray(pf(env, hero), dtype=np.float64)

        # ---- root Q via BR vs reference (average) opponent strategy ----
        root = tree.nodes[tree.root]
        assert root["t"] == "dec" and root["actor"] == hero
        hidx = cfr.pair_idx.get((min(hero_hole), max(hero_hole)))
        assert hidx is not None
        q_br = {}
        for k, ch in enumerate(root["children"]):
            v = cfr._br_walk(ch, hero, cfr.w[opp].copy())
            q_br[root["actions"][k]] = float(v[hidx])
        _, _, _, w_all = cfr._sw(cfr.w[opp])
        z = float(w_all[hidx])  # compatible opp mass for hero's combo
        q_norm = {a: v / max(z, 1e-12) for a, v in q_br.items()}
        v_br = max(q_norm.values())
        served = {a: float(probs_dep[a]) for a in root["actions"]}
        ssum = sum(served.values())
        served = {a: p / ssum for a, p in served.items()} if ssum > 1e-9 else served
        ev_dep = sum(served[a] * q_norm[a] for a in root["actions"])
        ev_loss_chips = v_br - ev_dep
        # reference root strategy for hero's combo (for L1 comparison)
        st_ref = cfr.avg_strategy(tree.root)[:, hidx]
        ref = {root["actions"][k]: float(st_ref[k]) for k in range(len(root["actions"]))}
        l1 = sum(abs(ref[a] - served[a]) for a in root["actions"])

        res = {
            "state": si, "deal": st["deal"], "hero": hero, "pot_bb": st["pot_bb"],
            "facing_bet": st["facing_bet"], "n_combos": int(C), "n_decision_nodes": n_dec,
            "n_tree_nodes": len(tree.nodes), "raise_cap": cap_used,
            "cfr_iters": int(iters),
            "exploitability_chips": round(e_h + e_o, 4),
            "exploitability_mbb": round((e_h + e_o) * MBB_PER_CHIP, 3),
            "ev_loss_chips": round(ev_loss_chips, 4),
            "ev_loss_mbb": round(ev_loss_chips * MBB_PER_CHIP, 3),
            "ev_loss_pct_pot": round(100.0 * ev_loss_chips / pot_root, 3),
            "l1_vs_ref": round(l1, 4),
            "served": {str(a): round(p, 4) for a, p in served.items()},
            "ref": {str(a): round(p, 4) for a, p in ref.items()},
            "q_br_chips": {str(a): round(v, 2) for a, v in q_norm.items()},
            "range_mode": "reach" if w_opp_raw is not None else "uniform_fallback",
            "wall_s": round(time.time() - t0, 1),
        }
        results.append(res)
        log(f"state {si + 1}/{len(states)}: pot {st['pot_bb']}bb dec_nodes={n_dec} "
            f"iters={iters} eps={res['exploitability_mbb']}mbb "
            f"EVloss={res['ev_loss_mbb']}mbb L1={res['l1_vs_ref']} ({res['wall_s']}s, "
            f"total {(time.time() - t_all) / 60:.1f}min)")

        with open(OUT_JSON, "w") as f:
            json.dump({"partial": True, "results": results}, f, indent=1)

    losses = np.asarray([r["ev_loss_mbb"] for r in results])
    summary = {
        "n_states": len(results),
        "ev_loss_mbb_mean": round(float(losses.mean()), 3),
        "ev_loss_mbb_median": round(float(np.median(losses)), 3),
        "ev_loss_mbb_p90": round(float(np.percentile(losses, 90)), 3),
        "ev_loss_mbb_max": round(float(losses.max()), 3),
        "worst_states": sorted(results, key=lambda r: -r["ev_loss_mbb"])[:5],
    }
    with open(OUT_JSON, "w") as f:
        json.dump({"summary": summary, "results": results}, f, indent=1)
    log(f"solve DONE: mean={summary['ev_loss_mbb_mean']}mbb p90={summary['ev_loss_mbb_p90']}mbb "
        f"over {len(results)} states; wrote {OUT_JSON}")
    return 0


if __name__ == "__main__":
    _args = [a for a in sys.argv[1:] if not a.startswith("--")]
    _resume = "--resume" in sys.argv[1:]
    cmd = _args[0] if _args else "collect"
    n = int(_args[1]) if len(_args) > 1 else (70 if cmd == "collect" else 50)
    sys.exit(collect(n) if cmd == "collect" else solve(n, resume=_resume))
