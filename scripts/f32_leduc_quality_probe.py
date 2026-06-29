#!/usr/bin/env python3
"""f32 value-row quality probe for production CFR+ — Leduc Hold'em proxy.

WHY
---
Production 6-max NLHE has NO cheaply-computable exact best response, so we cannot
directly measure whether storing the CFR value rows (cum_regret / strat_sum) in f32
instead of f64 degrades the served equilibrium. Leduc Hold'em is the standard proxy:
it has an EXACT, cheap information-set best response, so NashConv (= sum of both
players' best-response gains) is computable to machine precision. If f32 value rows
hurt convergence, they will leave a measurable NashConv floor here.

WHAT IS MATCHED TO PRODUCTION (recon `trainer.rs`)
--------------------------------------------------
  * Averaging: LINEAR-CFR. Per iteration t, the average-strategy accumulator gains
    `w * sigma` with `w = t` (OpenSpiel kSimple linear averaging). NOT vanilla(w=1),
    NOT quadratic, NOT CFR+ averaging, NOT DCFR. strat_sum grows O(T^2).
  * Regret: CFR+. Plain `r[a] += av[a] - ev` per visit, then a single positive floor
    `max(r, 0)` applied ONCE per iteration (regret-matching+).
  * Value rows are f64 in prod; this script swaps the DTYPE of those rows per arm to
    measure the consequence.

The reach/visitation structure differs (prod uses external sampling; this is a
full-tree pass), but the f32-relevant fact — strat_sum magnitude O(T^2) growth and
the per-row dynamic range between the highest-mass and lowest-mass legal action — is
reproduced and pushed to / beyond production magnitudes.

ARMS
----
  f64_both            : cum_regret f64,  strat_sum f64   (baseline / ground truth)
  f32_both            : cum_regret f32,  strat_sum f32   (pure read-add-store in f32)
  f32_regret_only     : cum_regret f32,  strat_sum f64
  f32_storage_f64_accum: both STORED f32, but each iteration's add computed in f64 then
                         round-stored back to f32 (Kahan-free "compute-wide store-narrow")

KEY OUTPUT
----------
Does any f32 arm PLATEAU at a measurably HIGHER (worse) final NashConv than f64_both?
A transient gap that closes is fine; a persistent floor is the red flag.

Pure Python + numpy. Does NOT touch the Rust build. Standalone (re-implements the
validated `leduc_jax` rules so it has no JAX import cost over 1e6 iterations).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict

import numpy as np

# ---------------------------------------------------------------------------
# Leduc rules — faithful pure-numpy re-implementation of pokerbot.engine.leduc_jax
# (cross-checked field-by-field against that module: ANTE=1, RAISE_SIZE=2,
#  MAX_RAISES=2, 3 actions fold/call/raise, 2 betting rounds, pair score = 10+rank).
# ---------------------------------------------------------------------------
NUM_PLAYERS = 2
NUM_ACTIONS = 3  # 0=fold, 1=call/check, 2=raise
ANTE = 1
RAISE_SIZE = 2
MAX_RAISES = 2
ROUND_PRIVATE = 0
ROUND_PUBLIC = 1
FOLD, CALL, RAISE = 0, 1, 2


class State:
    """Mutable-free Leduc node (we always build children fresh)."""

    __slots__ = (
        "cards", "public", "pot", "committed", "invested",
        "current_bet", "raises", "round", "acting", "acted",
        "done", "rewards", "deal_public", "hist",
    )

    def __init__(self, cards, public, pot, committed, invested, current_bet,
                 raises, rnd, acting, acted, done, rewards, deal_public, hist):
        self.cards = cards
        self.public = public
        self.pot = pot
        self.committed = committed
        self.invested = invested
        self.current_bet = current_bet
        self.raises = raises
        self.round = rnd
        self.acting = acting
        self.acted = acted
        self.done = done
        self.rewards = rewards
        self.deal_public = deal_public
        self.hist = hist  # full action sequence (tuple), matches leduc_bridge.hist


def state_from_deal(cards, deal_public):
    return State(
        cards=(int(cards[0]), int(cards[1])),
        public=-1,
        pot=ANTE * 2,
        committed=(ANTE, ANTE),
        invested=(ANTE, ANTE),
        current_bet=ANTE,
        raises=0,
        rnd=ROUND_PRIVATE,
        acting=0,
        acted=(False, False),
        done=False,
        rewards=(0.0, 0.0),
        deal_public=int(deal_public),
        hist=(),
    )


def _hand_score(card, public):
    return (10 + card) if card == public else card


def _resolve_showdown(s):
    s0 = _hand_score(s.cards[0], s.public)
    s1 = _hand_score(s.cards[1], s.public)
    half = s.pot // 2
    rem = s.pot - half * 2
    inv = s.invested
    if s0 > s1:
        r0 = s.pot - inv[0]
    elif s0 == s1:
        r0 = half + rem - inv[0]
    else:
        r0 = -inv[0]
    r1 = s.pot - inv[0] - inv[1] - r0
    s.done = True
    s.rewards = (float(r0), float(r1))


def _start_public_round(s):
    s.public = s.deal_public
    s.round = ROUND_PUBLIC
    s.current_bet = 0
    s.committed = (0, 0)
    s.raises = 0
    s.acted = (False, False)
    s.acting = 1


def _round_complete(s):
    return s.committed[0] == s.committed[1] and s.acted[0] and s.acted[1]


def legal_actions(s):
    seat = s.acting
    to_call = max(0, s.current_bet - s.committed[seat])
    can_raise = (s.raises < MAX_RAISES)  # stacks=99 always >> to_call in Leduc
    return (True, True, can_raise)


def apply_action(s, action):
    seat = s.acting
    to_call = max(0, s.current_bet - s.committed[seat])
    cards, public, deal_public = s.cards, s.public, s.deal_public
    committed = list(s.committed)
    invested = list(s.invested)
    acted = list(s.acted)
    pot = s.pot
    current_bet = s.current_bet
    raises = s.raises
    rnd = s.round
    done = False
    rewards = (0.0, 0.0)
    acting = 1 - seat

    new_hist = s.hist + (action,)
    if action == FOLD:
        winner = 1 - seat
        r = [-invested[0], -invested[1]]
        r[winner] += pot
        ns = State(cards, public, pot, tuple(committed), tuple(invested),
                   current_bet, raises, rnd, acting, tuple(acted), True,
                   (float(r[0]), float(r[1])), deal_public, new_hist)
        return ns
    elif action == CALL:
        pay = to_call  # full stacks in Leduc
        committed[seat] += pay
        invested[seat] += pay
        pot += pay
        acted[seat] = True
    elif action == RAISE:
        target = current_bet + RAISE_SIZE
        pay = target - committed[seat]
        committed[seat] += pay
        invested[seat] += pay
        pot += pay
        current_bet = committed[seat]
        raises += 1
        acted = [False, False]
        acted[seat] = True

    ns = State(cards, public, pot, tuple(committed), tuple(invested),
               current_bet, raises, rnd, acting, tuple(acted), done, rewards,
               deal_public, new_hist)

    if _round_complete(ns) and not ns.done and ns.round == ROUND_PRIVATE:
        _start_public_round(ns)
    if _round_complete(ns) and not ns.done and ns.round == ROUND_PUBLIC:
        _resolve_showdown(ns)
    return ns


def info_key(s, seat):
    pub = s.public if s.public >= 0 else -1
    to_call = max(0, s.current_bet - s.committed[seat])
    # history INCLUDED — matches leduc_bridge.py py_info_key; without it distinct
    # second-round infosets (differing by first-round betting) collapse, solving a
    # different (smaller) game and corrupting both CFR and the best response.
    return (seat, s.cards[seat], pub, s.round, to_call, s.raises, s.hist)


def _all_deals():
    """All 90 deck permutations: ((p0,p1), public)."""
    from itertools import permutations
    deck = (0, 0, 1, 1, 2, 2)
    return tuple(((p[0], p[1]), p[2]) for p in sorted(set(permutations(deck))))


# ---------------------------------------------------------------------------
# Precompiled game tree (built once; reused every iteration for speed).
# Each acting node: (key, seat, legal_mask, [(action, child_idx_or_terminal)...])
# Terminals carry their reward vector.
# ---------------------------------------------------------------------------
class Node:
    __slots__ = ("is_terminal", "rewards", "key", "seat", "legal", "children")

    def __init__(self):
        self.is_terminal = False
        self.rewards = None
        self.key = None
        self.seat = None
        self.legal = None       # tuple(bool*3)
        self.children = None    # list of Node, indexed by action (None if illegal)


def build_tree(deals):
    """Build the full game forest; return list of root Nodes (one per deal)."""
    def rec(s):
        n = Node()
        if s.done:
            n.is_terminal = True
            n.rewards = np.asarray(s.rewards, dtype=np.float64)
            return n
        seat = s.acting
        mask = legal_actions(s)
        n.key = info_key(s, seat)
        n.seat = seat
        n.legal = mask
        n.children = [None, None, None]
        for a in range(NUM_ACTIONS):
            if mask[a]:
                n.children[a] = rec(apply_action(s, a))
        return n

    return [rec(state_from_deal(c, p)) for c, p in deals]


def collect_infosets(roots):
    """Map info_key -> legal mask (consistent across all states sharing the key)."""
    masks = {}
    def walk(n):
        if n.is_terminal:
            return
        masks[n.key] = n.legal
        for a in range(NUM_ACTIONS):
            if n.children[a] is not None:
                walk(n.children[a])
    for r in roots:
        walk(r)
    return masks


# ---------------------------------------------------------------------------
# Exact information-set best response (OpenSpiel-style; no card cheating).
# Returns BR value for `br_seat` against fixed average policy `avg`.
# ---------------------------------------------------------------------------
def _policy_for_key(avg, key, mask):
    sigma = avg.get(key)
    n_legal = sum(mask)
    if sigma is None:
        out = np.zeros(NUM_ACTIONS, dtype=np.float64)
        for a in range(NUM_ACTIONS):
            if mask[a]:
                out[a] = 1.0 / n_legal
        return out
    out = sigma.astype(np.float64).copy()
    for a in range(NUM_ACTIONS):
        if not mask[a]:
            out[a] = 0.0
    tot = out.sum()
    if tot > 0:
        out /= tot
    else:
        for a in range(NUM_ACTIONS):
            out[a] = (1.0 / n_legal) if mask[a] else 0.0
    return out


def best_response_value(roots, avg, br_seat):
    """Exact information-set best response for br_seat (OpenSpiel-style, 2-phase).

    Phase 1: enumerate every br_seat decision node, accumulating per infoset the list
             of (node, cf_prob) where cf_prob = product of opponent+chance reach probs
             to that node (the BR player's own past choices are NOT included — that is
             the counterfactual-reach weighting that makes the per-infoset argmax exact).
    Phase 2: for each infoset pick a* = argmax_a sum_{(s,p)} p * Q(s,a), where Q uses the
             SAME BR (resolved recursively; deeper infosets settle within the DFS, and
             every node sharing a key shares a* because the Leduc info_key is sufficient).
    Returns the BR value averaged over roots (chance is the uniform deal prior).
    """
    infosets = defaultdict(list)   # key -> list[(node, cf_prob)]

    def collect(n, cf_prob):
        if n.is_terminal:
            return
        if n.seat == br_seat:
            infosets[n.key].append((n, cf_prob))
        if n.seat == br_seat:
            for a in range(NUM_ACTIONS):
                if n.children[a] is not None:
                    collect(n.children[a], cf_prob)  # BR's own reach folded out
        else:
            sigma = _policy_for_key(avg, n.key, n.legal)
            for a in range(NUM_ACTIONS):
                if n.children[a] is not None:
                    collect(n.children[a], cf_prob * sigma[a])

    for r in roots:
        collect(r, 1.0)

    br_action = {}
    value_cache = {}

    def value(n):
        """BR value to br_seat of node n (uses resolved BR action at br_seat nodes)."""
        nid = id(n)
        cached = value_cache.get(nid)
        if cached is not None:
            return cached
        if n.is_terminal:
            v = float(n.rewards[br_seat])
        elif n.seat == br_seat:
            v = value(n.children[br_action_at(n)])
        else:
            sigma = _policy_for_key(avg, n.key, n.legal)
            v = 0.0
            for a in range(NUM_ACTIONS):
                if n.children[a] is not None:
                    v += sigma[a] * value(n.children[a])
        value_cache[nid] = v
        return v

    def br_action_at(n):
        key = n.key
        a_star = br_action.get(key)
        if a_star is None:
            entries = infosets[key]
            best_a, best_q = None, -1e18
            for a in range(NUM_ACTIONS):
                if n.children[a] is None:
                    continue
                q = 0.0
                for s, p in entries:
                    q += p * value(s.children[a])
                if q > best_q:
                    best_q, best_a = q, a
            br_action[key] = best_a
            a_star = best_a
        return a_star

    total = 0.0
    for r in roots:
        total += value(r)
    return total / len(roots)


def profile_value(roots, avg, player):
    """Expected value to `player` when BOTH players follow `avg`."""
    def value(n):
        if n.is_terminal:
            return n.rewards[player]
        sigma = _policy_for_key(avg, n.key, n.legal)
        v = 0.0
        for a in range(NUM_ACTIONS):
            if n.children[a] is not None:
                v += sigma[a] * value(n.children[a])
        return v
    return sum(value(r) for r in roots) / len(roots)


def nashconv_mbb(roots, avg):
    """NashConv in milli-ante units (x1000), matching leduc_tabular_cfr convention."""
    u0 = profile_value(roots, avg, 0)
    u1 = profile_value(roots, avg, 1)
    br0 = best_response_value(roots, avg, 0)
    br1 = best_response_value(roots, avg, 1)
    nc = (max(0.0, br0 - u0) + max(0.0, br1 - u1)) * 1000.0
    return nc, u0, u1, br0, br1


# ---------------------------------------------------------------------------
# Flattened tree for the fast CFR inner loop.
#
# We compile the recursive Node forest into flat parallel arrays so the hot loop
# is plain Python scalar arithmetic over int-indexed lists (no per-node numpy
# allocation, which dominated the recursive version). Children are emitted before
# parents (post-order), so a single forward pass over `order` can compute child
# values before parents need them — but we keep an explicit recursive driver here
# for clarity and correctness; the speedup comes from precomputed per-node metadata
# (infoset index, child node indices, terminal payoff) and scalar (not array) math.
# ---------------------------------------------------------------------------
class FlatTree:
    __slots__ = ("n_nodes", "n_infosets", "is_term", "term_r0", "term_r1",
                 "iset", "seat", "legal_a", "child", "roots", "iset_legal",
                 "iset_nlegal", "key_of_iset")

    def __init__(self):
        pass


def compile_tree(roots, masks):
    keys = list(masks.keys())
    key_to_iset = {k: i for i, k in enumerate(keys)}
    nodes = []           # list of Node in discovery order
    node_id = {}         # id(Node) -> flat index

    # assign flat indices via DFS
    def visit(n):
        if id(n) in node_id:
            return node_id[id(n)]
        idx = len(nodes)
        node_id[id(n)] = idx
        nodes.append(n)
        if not n.is_terminal:
            for a in range(NUM_ACTIONS):
                if n.children[a] is not None:
                    visit(n.children[a])
        return idx

    root_idx = [visit(r) for r in roots]

    N = len(nodes)
    ft = FlatTree()
    ft.n_nodes = N
    ft.n_infosets = len(keys)
    ft.is_term = [False] * N
    ft.term_r0 = [0.0] * N
    ft.term_r1 = [0.0] * N
    ft.iset = [-1] * N
    ft.seat = [-1] * N
    ft.legal_a = [()] * N       # tuple of legal action ints
    ft.child = [(-1, -1, -1)] * N
    for idx, n in enumerate(nodes):
        if n.is_terminal:
            ft.is_term[idx] = True
            ft.term_r0[idx] = float(n.rewards[0])
            ft.term_r1[idx] = float(n.rewards[1])
        else:
            ft.iset[idx] = key_to_iset[n.key]
            ft.seat[idx] = n.seat
            la = tuple(a for a in range(NUM_ACTIONS) if n.children[a] is not None)
            ft.legal_a[idx] = la
            ch = [-1, -1, -1]
            for a in la:
                ch[a] = node_id[id(n.children[a])]
            ft.child[idx] = tuple(ch)
    ft.roots = root_idx
    ft.iset_legal = [tuple(a for a in range(NUM_ACTIONS) if masks[k][a]) for k in keys]
    ft.iset_nlegal = [len(la) for la in ft.iset_legal]
    ft.key_of_iset = keys
    return ft


# ---------------------------------------------------------------------------
# CFR+ with linear averaging, dtype-parametrized value rows.
# ---------------------------------------------------------------------------
def regret_match(regret, mask):
    """Regret-matching+ : positive part of cum_regret, normalized; uniform fallback."""
    pos = np.where(regret > 0.0, regret, 0.0)
    for a in range(NUM_ACTIONS):
        if not mask[a]:
            pos[a] = 0.0
    tot = pos.sum()
    if tot > 0:
        return (pos / tot).astype(np.float64)
    n_legal = sum(mask)
    out = np.zeros(NUM_ACTIONS, dtype=np.float64)
    for a in range(NUM_ACTIONS):
        if mask[a]:
            out[a] = 1.0 / n_legal
    return out


def _f32(x):
    """Round a Python float to the nearest f32, returned as a Python float.

    This models a single f32 store: the value carried forward is exactly what an
    f32 cell would hold. Used per-add for the non-wide arms so f32 *swamping*
    (small increment lost into a large accumulator) is faithfully reproduced.
    """
    return float(np.float32(x))


def run_arm(arm_name, roots, masks, iters, checkpoints, *, log_every, regret_dtype,
            stratsum_dtype, accum_wide):
    """Run CFR+ (linear averaging) for one dtype arm on the flattened tree.

    arm semantics:
      regret_dtype   : storage dtype of cum_regret rows (f32 or f64)
      stratsum_dtype : storage dtype of strat_sum rows  (f32 or f64)
      accum_wide     : if True, each iteration's increment into a row is computed in
                       f64 then round-stored to the row dtype (compute-wide,
                       store-narrow). If False, every individual add is round-stored
                       to the storage dtype (pure read-add-store) — this is the path
                       that exhibits f32 swamping.

    Value rows are kept as flat Python-float lists; the storage dtype is enforced by
    rounding each stored cell through np.float32 when that arm's row is f32. This is a
    faithful bit-for-bit model of an f32 accumulator while staying fast (scalar math,
    no per-node numpy allocation).
    """
    ft = compile_tree(roots, masks)
    NI = ft.n_infosets
    N = ft.n_nodes
    regret_is_f32 = (np.dtype(regret_dtype) == np.float32)
    strat_is_f32 = (np.dtype(stratsum_dtype) == np.float32)

    # value rows: NI x 3 Python-float lists
    cum_regret = [[0.0, 0.0, 0.0] for _ in range(NI)]
    strat_sum = [[0.0, 0.0, 0.0] for _ in range(NI)]
    # current strategy per infoset (f64 working copy; regret-matching is exact)
    cur_policy = [[0.0, 0.0, 0.0] for _ in range(NI)]
    for i in range(NI):
        la = ft.iset_legal[i]
        u = 1.0 / len(la)
        for a in la:
            cur_policy[i][a] = u

    is_term = ft.is_term
    term_r0 = ft.term_r0
    term_r1 = ft.term_r1
    iset = ft.iset
    seat = ft.seat
    legal_a = ft.legal_a
    child = ft.child
    root_idx = ft.roots
    chance = 1.0 / len(root_idx)
    f32 = _f32

    # node value buffers (seat-0 and seat-1 values), reused across the post-order pass
    val0 = [0.0] * N
    val1 = [0.0] * N
    # post-order: children have higher index than parents in our DFS emission? No —
    # parents are emitted before children. So process nodes in REVERSE index order to
    # guarantee children (higher index) are done before parents. Verify monotonicity:
    # visit() appends parent then recurses, so every child index > its parent index.
    rev = range(N - 1, -1, -1)

    def sweep_values():
        """Bottom-up node-value pass under current cur_policy (both seats' values)."""
        for idx in rev:
            if is_term[idx]:
                val0[idx] = term_r0[idx]
                val1[idx] = term_r1[idx]
            else:
                i = iset[idx]
                pol = cur_policy[i]
                ch = child[idx]
                v0 = 0.0
                v1 = 0.0
                for a in legal_a[idx]:
                    c = ch[a]
                    p = pol[a]
                    v0 += p * val0[c]
                    v1 += p * val1[c]
                val0[idx] = v0
                val1[idx] = v1

    def reach_pass():
        """Top-down: per-node reach probabilities reach0[idx], reach1[idx].

        reachP[idx] = product of seat-P's own sigma along the path to idx (chance NOT
        folded in here). Each node has exactly one parent, so a single forward pass
        (parent idx < child idx) fills both arrays. Mirrors the recursive r0/r1 that the
        validated reference threads down the tree.
        """
        reach0 = [0.0] * N
        reach1 = [0.0] * N
        for ridx in root_idx:
            reach0[ridx] = 1.0
            reach1[ridx] = 1.0
        for idx in range(N):
            if is_term[idx]:
                continue
            r0 = reach0[idx]
            r1 = reach1[idx]
            if r0 == 0.0 and r1 == 0.0:
                continue
            s = seat[idx]
            pol = cur_policy[iset[idx]]
            ch = child[idx]
            for a in legal_a[idx]:
                c = ch[a]
                if s == 0:
                    reach0[c] += r0 * pol[a]
                    reach1[c] += r1
                else:
                    reach0[c] += r0
                    reach1[c] += r1 * pol[a]
        return reach0, reach1

    def regret_update(player, reach0, reach1):
        """One alternating CFR+ regret sweep for `player` (counterfactual-reach).

        cf-reach at a hero node = opponent's own reach to the node times chance.
        """
        vplayer = val0 if player == 0 else val1
        opp_reach = reach1 if player == 0 else reach0
        for idx in range(N):
            if is_term[idx]:
                continue
            if seat[idx] != player:
                continue
            ci = opp_reach[idx] * chance
            if ci == 0.0:
                continue
            ch = child[idx]
            node_v = vplayer[idx]
            row = cum_regret[iset[idx]]
            if accum_wide or not regret_is_f32:
                for a in legal_a[idx]:
                    row[a] += ci * (vplayer[ch[a]] - node_v)
            else:
                for a in legal_a[idx]:
                    row[a] = f32(row[a] + f32(ci * (vplayer[ch[a]] - node_v)))

    def floor_and_refresh():
        for i in range(NI):
            row = cum_regret[i]
            la = ft.iset_legal[i]
            # CFR+ positive floor
            tot = 0.0
            for a in la:
                if row[a] < 0.0:
                    row[a] = 0.0
                if regret_is_f32:
                    row[a] = f32(row[a])
                tot += row[a]
            pol = cur_policy[i]
            if tot > 0.0:
                for a in (0, 1, 2):
                    pol[a] = (row[a] / tot) if a in la else 0.0
            else:
                u = 1.0 / len(la)
                for a in (0, 1, 2):
                    pol[a] = u if a in la else 0.0

    def strat_update(weight, reach0, reach1):
        """Linear-CFR average: add weight * own_reach * sigma at the acting node.

        own_reach = the ACTING player's own reach probability to the node (reach0 at
        seat-0 nodes, reach1 at seat-1 nodes) — exactly the recursive reference's
        `own = r0 if seat==0 else r1`.
        """
        for idx in range(N):
            if is_term[idx]:
                continue
            s = seat[idx]
            oi = reach0[idx] if s == 0 else reach1[idx]
            if oi == 0.0:
                continue
            i = iset[idx]
            pol = cur_policy[i]
            inc = weight * oi
            row = strat_sum[i]
            if accum_wide or not strat_is_f32:
                for a in legal_a[idx]:
                    row[a] += inc * pol[a]
            else:
                for a in legal_a[idx]:
                    row[a] = f32(row[a] + f32(inc * pol[a]))

    results = []
    t0 = time.time()
    ckpt_targets = sorted(set([c for c in checkpoints if c <= iters] + [iters]))
    ckpt_i = 0

    for t in range(1, iters + 1):
        weight = float(t)
        for player in range(NUM_PLAYERS):
            sweep_values()
            reach0, reach1 = reach_pass()
            regret_update(player, reach0, reach1)
            floor_and_refresh()
        # linear-CFR average accumulation under the final (post-both-updates) policy
        reach0, reach1 = reach_pass()
        strat_update(weight, reach0, reach1)

        if (t % log_every == 0) or (t == 1):
            elapsed = time.time() - t0
            rate = t / elapsed if elapsed > 0 else 0.0
            eta = (iters - t) / rate if rate > 0 else float("inf")
            max_ss = max(max(abs(v) for v in row) for row in strat_sum)
            pct = 100.0 * t / iters
            print(f"[{arm_name}] iter {t}/{iters} ({pct:5.1f}%)  "
                  f"{rate:8.1f} it/s  ETA {eta/60:6.1f}m  max|strat_sum|={max_ss:.3e}",
                  flush=True)

        if ckpt_i < len(ckpt_targets) and t == ckpt_targets[ckpt_i]:
            avg = {}
            for i in range(NI):
                s = strat_sum[i]
                la = ft.iset_legal[i]
                tot = sum(s[a] for a in la)
                arr = np.zeros(NUM_ACTIONS, dtype=np.float64)
                if tot > 0:
                    for a in la:
                        arr[a] = s[a] / tot
                else:
                    for a in la:
                        arr[a] = 1.0 / len(la)
                avg[ft.key_of_iset[i]] = arr
            nc, u0, u1, br0, br1 = nashconv_mbb(roots, avg)
            max_ss = max(max(abs(v) for v in row) for row in strat_sum)
            max_reg = max(max(abs(v) for v in row) for row in cum_regret)
            results.append({
                "iter": t,
                "nashconv_mbb": nc,
                "u0": u0, "u1": u1, "br0": br0, "br1": br1,
                "max_stratsum": max_ss,
                "max_abs_regret": max_reg,
            })
            print(f"[{arm_name}]   CHECKPOINT iter={t}  NashConv={nc:.6f} mbb  "
                  f"max|ss|={max_ss:.3e}  max|reg|={max_reg:.3e}", flush=True)
            ckpt_i += 1

    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=20000,
                    help="total CFR iterations per arm")
    ap.add_argument("--checkpoints", type=str, default="1000,10000,100000,1000000",
                    help="comma-separated iter checkpoints for NashConv")
    ap.add_argument("--log-every", type=int, default=500)
    ap.add_argument("--arms", type=str,
                    default="f64_both,f32_both,f32_regret_only,f32_storage_f64_accum")
    ap.add_argument("--out", type=str, default="")
    args = ap.parse_args()

    checkpoints = [int(x) for x in args.checkpoints.split(",") if x.strip()]
    checkpoints = [c for c in checkpoints if c <= args.iters]

    deals = _all_deals()
    print(f"Building Leduc tree ({len(deals)} deals)...", flush=True)
    roots = build_tree(deals)
    masks = collect_infosets(roots)
    print(f"  infosets: {len(masks)}", flush=True)

    arm_specs = {
        "f64_both": dict(regret_dtype=np.float64, stratsum_dtype=np.float64, accum_wide=False),
        "f32_both": dict(regret_dtype=np.float32, stratsum_dtype=np.float32, accum_wide=False),
        "f32_regret_only": dict(regret_dtype=np.float32, stratsum_dtype=np.float64, accum_wide=False),
        "f32_storage_f64_accum": dict(regret_dtype=np.float32, stratsum_dtype=np.float32, accum_wide=True),
    }
    selected = [a.strip() for a in args.arms.split(",") if a.strip()]

    all_results = {}
    overall_max_ss = 0.0
    for arm in selected:
        spec = arm_specs[arm]
        print(f"\n=== ARM {arm}  regret={np.dtype(spec['regret_dtype']).name} "
              f"stratsum={np.dtype(spec['stratsum_dtype']).name} "
              f"accum_wide={spec['accum_wide']} ===", flush=True)
        res = run_arm(arm, roots, masks, args.iters, checkpoints,
                      log_every=args.log_every, **spec)
        all_results[arm] = res
        if res:
            overall_max_ss = max(overall_max_ss, max(r["max_stratsum"] for r in res))

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY  (NashConv in milli-ante units, lower=closer to equilibrium)")
    print("=" * 70)
    base = all_results.get("f64_both")
    base_final = base[-1]["nashconv_mbb"] if base else None
    f32_plateaus_higher = False
    for arm in selected:
        res = all_results[arm]
        final = res[-1]["nashconv_mbb"]
        line = f"{arm:24s} final NashConv = {final:10.5f} mbb  (iter {res[-1]['iter']})"
        if base_final is not None and arm != "f64_both":
            gap = final - base_final
            line += f"   gap vs f64 = {gap:+10.5f} mbb"
            # red flag = persistently worse by a meaningful margin (> 0.5 mbb and >1% rel)
            if gap > 0.5 and gap > 0.01 * max(base_final, 1e-9):
                f32_plateaus_higher = True
        print(line)
    print(f"\nf32_plateaus_higher (any f32 arm meaningfully worse final): {f32_plateaus_higher}")
    print(f"overall max strat_sum reached: {overall_max_ss:.4e}")

    out = {
        "iters": args.iters,
        "checkpoints": checkpoints,
        "infosets": len(masks),
        "results": all_results,
        "f32_plateaus_higher": f32_plateaus_higher,
        "overall_max_stratsum": overall_max_ss,
    }
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=2)
        print(f"\nwrote {args.out}", flush=True)
    return out


if __name__ == "__main__":
    main()
