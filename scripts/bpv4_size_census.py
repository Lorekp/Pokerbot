#!/usr/bin/env python
"""bpv4 size census -- exact + sampled sizing of action-HISTORY infoset keying.

Supports the blueprint-v4 (Pluribus-grade) design doc
(artifacts/blueprint_pluribus_grade_design.md). Read-only w.r.t. the engine: drives the
installed `battery_native` pyd (core.rs rules, chip-exact) from the outside.

KEY FACT this exploits: with equal starting stacks, ante 0 and the pot-fraction action
menu, the ABSTRACTED BETTING TREE IS CARD-INDEPENDENT (legality/sizing never read
board/holes; chance only advances the street). So the number of distinct abstracted
action histories per street is EXACTLY countable by memoized DP over betting-relevant
state (each distinct betting state expanded once; per-history multiplicities recovered
by summing child counts). Card buckets multiply per street afterwards.

Modes:
  enumerate  -- exact per-street decision-history counts + street-entry counts + distinct
                (pos,betting6[,hist3]) key groups (validation anchor vs bp3's 714,324
                preflop rows) for a given action menu.
  walk       -- sampled trainer-visitation census: hero branches ALL allowed actions
                (external sampling, all 6 hero seats per deal), opponents sample
                uniformly over allowed-legal; counts DISTINCT full-history infoset rows
                per street vs deals processed (Heaps-law growth curve for the
                visited-vs-enumerable ratio).

Usage:
  python scripts/bpv4_size_census.py enumerate --menu current --out artifacts/_bpv4_census_current.json
  python scripts/bpv4_size_census.py enumerate --menu full7 --max-states 20000000
  python scripts/bpv4_size_census.py walk --menu current --deals 20000 --out artifacts/_bpv4_walk_current.json
"""
import argparse
import hashlib
import json
import os
import struct
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import battery_native as bn

# ---- flat i64 layout (lib.rs::flat, verified against installed pyd: FLAT_LEN=311) ----
O_HOLES = 0
O_BOARD = 12
O_SEAT_STATUS = 17
O_STACKS = 23
O_COMMITTED_STREET = 29
O_COMMITTED_TOTAL = 35
O_STREET = 41
O_POT_DEAD = 42
O_CURRENT_BET = 43
O_MIN_RAISE = 44
O_BUTTON = 45
O_ACTION_SEAT = 46
O_LAST_AGGRESSOR = 47
O_ACTED = 48
O_LAST_RAISE_FULL = 54
O_DECK = 55
O_DECK_PTR = 107
O_RNG_KEY = 108
O_HIST_SEAT = 110
O_HIST_ACTION = 174
O_HIST_AMOUNT = 238
O_HIST_LEN = 302
O_DONE = 303
O_REWARDS = 304
O_PENDING_DEAL = 310
FLAT_LEN = 311

FOLD, CHECK, CALL, ALL_IN = 0, 1, 2, 10
BET0 = 3  # bucket index base
N_BUCKETS = 7
BET_FRACS = [0.33333334, 0.5, 0.66, 0.75, 1.0, 1.5, 2.0]
POS_BY_OFFSET = [3, 4, 5, 0, 1, 2]  # BTN,SB,BB,UTG,HJ,CO by (seat-button)%6

def f32(x):
    return struct.unpack("<f", struct.pack("<f", x))[0]

BET_FRACS_F32 = [f32(x) for x in BET_FRACS]

# Action menus. keep = bet-bucket keep-list indices (0..6 over BET_FRACS); F/X/C/AI always in.
MENUS = {
    # deployed trainer keep-lists (TrainerCfg defaults: pre [4,5,6], post [0,1,3,4,5])
    "current": {"pre": [4, 5, 6], "post": [0, 1, 3, 4, 5]},
    # widened preflop (adds engine-legal 2.0bb/2.12bb opens via buckets 2,3), same postflop
    "widepre": {"pre": [2, 3, 4, 5, 6], "post": [0, 1, 3, 4, 5]},
    # full 7-bucket vocabulary everywhere (the menu ceiling of the frozen engine)
    "full7": {"pre": list(range(7)), "post": list(range(7))},
    # Pluribus-shaped: rich early, thin late is NOT expressible per-node without a tree
    # spec; this approximates "rich pre + moderate post"
    "richpre": {"pre": list(range(7)), "post": [0, 1, 3, 4, 5]},
    # arm-3 (SPLIT preflop): FIRST raise (no raise yet this hand, current_bet <= BB) uses the
    # widened 5-size menu; RE-raises (3-bet+, current_bet > BB) keep the legacy 3-size menu.
    # "pre_rr" is optional per menu; when absent, "pre" applies at all preflop nodes (legacy).
    "arm3": {"pre": [2, 3, 4, 5, 6], "pre_rr": [4, 5, 6], "post": [0, 1, 3, 4, 5]},
    # arm-5 design-of-record menu grade (design blueprint_pluribus_grade_design_v2 §menu_grade):
    # preflop MENU reverted to deployed [4,5,6]+jam (the arm-4-measured ~90%-of-reached-rows RAM
    # dominator that never fixed the trap); FLOP first-raise thinned to {0.5p,1p,AI} = post [1,4]
    # (POKERBOT_POST_KEEP=1,4). Runs WITH --mrr so turn/river first-raise = mrr_tr_first_keep [1,4]+jam
    # and every postflop RE-raise = mrr_reraise_keep [4]+jam. This is the design's Pluribus-grade
    # postflop menu; the A/B pairs fullhist-key vs T1-key UNDER THIS SAME MENU (single-variable = key).
    "t1menu": {"pre": [4, 5, 6], "post": [1, 4]},
}

def menu_masks(menu):
    pre = [False] * 11
    post = [False] * 11
    for i in (FOLD, CHECK, CALL, ALL_IN):
        pre[i] = post[i] = True
    for b in menu["pre"]:
        pre[BET0 + b] = True
    for b in menu["post"]:
        post[BET0 + b] = True
    pre_rr = None
    if "pre_rr" in menu:
        pre_rr = [False] * 11
        for i in (FOLD, CHECK, CALL, ALL_IN):
            pre_rr[i] = True
        for b in menu["pre_rr"]:
            pre_rr[BET0 + b] = True
    return pre, pre_rr, post

def arr(state):
    return np.frombuffer(state, np.int64)

def action_amount(a, action):
    """Chip-exact mirror of core.rs action_amount (f32-trunc sizing), for dedupe only."""
    seat = int(a[O_ACTION_SEAT])
    committed = int(a[O_COMMITTED_STREET + seat])
    to_call = max(int(a[O_CURRENT_BET]) - committed, 0)
    stack = int(a[O_STACKS + seat])
    pot = int(a[O_POT_DEAD]) + int(a[O_COMMITTED_STREET:O_COMMITTED_STREET + 6].sum())
    if action == CALL:
        return to_call
    if action == ALL_IN:
        return stack
    if BET0 <= action < ALL_IN:
        frac = BET_FRACS_F32[action - BET0]
        prod = int(f32(f32(pot) * frac))  # trunc toward zero
        target = max(int(a[O_CURRENT_BET]), prod)
        return to_call + max(target - committed, 0)
    return 0

def allowed_actions(state, a, pre_mask, post_mask, dedupe, mrr=False, flop_rr_keep=None,
                    pre_rr_mask=None, bb=0):
    """legal AND menu, with equal-final-commit aggressive actions deduped (lowest idx kept).

    Preflop menu split (arm-3): when pre_rr_mask is not None, preflop nodes at which a raise
    already exists (current_bet > bb) use pre_rr_mask instead of pre_mask; first-raise nodes
    (current_bet <= bb, i.e. unopened/limped pot) keep the widened pre_mask. bb = big-blind
    amount (root current_bet). Betting-state-only inputs, so memo-consistent.

    mrr=True applies Pluribus's deep-node rule (design M-RR, §4.2.2 / P-supp p.12-13): POSTFLOP,
    two thinning rules apply (mirrors trainer.rs abstracted_legal_mrr, lines 994-1007):
      (a) RE-raise (facing a live bet, street>=1): thinned to {1x pot, all-in} (keep [4]).
          On the FLOP (street==1) the raise-size keep-list is overridable via flop_rr_keep
          (arm-3 uses [1, 4] = {1/2 pot, 1x pot, all-in}); turn/river re-raise stays [4].
      (b) Turn/river FIRST raise (street>=2, not facing a live bet): thinned to
          {1/2 pot, 1x pot, all-in} (BET0+1 and BET0+4).
    The FIRST bet on flop (street==1, not facing bet) keeps the full postflop menu in all cases.
    This is action-abstraction thinning (perfect recall of the *thinned* actions preserved), NOT
    history/key truncation -- the design's "structural relief valve" that collapses the ~10^30
    history space toward Pluribus's finite 664.8M-class scale."""
    legal = bn.legal_action_mask(state, -1)
    street = int(a[O_STREET])
    if street == 0:
        menu = pre_mask
        if pre_rr_mask is not None and int(a[O_CURRENT_BET]) > bb:
            menu = pre_rr_mask  # a raise already exists this hand -> RE-raise menu
    else:
        menu = post_mask
    acts = [i for i in range(11) if legal[i] and menu[i]]
    if mrr and street >= 1:
        seat = int(a[O_ACTION_SEAT])
        facing_bet = int(a[O_CURRENT_BET]) > int(a[O_COMMITTED_STREET + seat])
        if facing_bet:
            # (a) RE-raise: raise sizes thinned to keep-list [4] ({1x pot, all-in});
            # on the flop, flop_rr_keep (when given) overrides the keep-list.
            rr_buckets = flop_rr_keep if (street == 1 and flop_rr_keep is not None) else [4]
            keep = {FOLD, CHECK, CALL, ALL_IN} | {BET0 + b for b in rr_buckets}
            acts = [i for i in acts if i in keep]
        elif street >= 2:
            # (b) Turn/river first raise: thin to {1/2 pot, 1x pot, all-in}.
            keep = {FOLD, CHECK, CALL, ALL_IN, BET0 + 1, BET0 + 4}  # BET0+1 = 0.5x, BET0+4 = 1.0x
            acts = [i for i in acts if i in keep]
    if not any(acts):
        acts = [i for i in range(11) if legal[i]][:1]  # engine guarantees CHECK fallback
    if not dedupe:
        return acts
    out, seen_amt = [], set()
    for i in acts:
        if i in (FOLD, CHECK, CALL):
            out.append(i)
            continue
        amt = action_amount(a, i)
        if amt in seen_amt:
            continue
        seen_amt.add(amt)
        out.append(i)
    return out

def betting6(a, seat):
    """Mirror of fingerprint.rs betting_abstraction_key (for the hist3 anchor census)."""
    committed_sum = int(a[O_COMMITTED_STREET:O_COMMITTED_STREET + 6].sum())
    pot = int(a[O_POT_DEAD]) + committed_sum
    to_call = max(int(a[O_CURRENT_BET]) - int(a[O_COMMITTED_STREET + seat]), 0)
    pot_bb = pot / 100.0
    eff = None
    for i in range(6):
        st = int(a[O_SEAT_STATUS + i])
        if st not in (0, 1):
            v = int(a[O_STACKS + i])
            eff = v if eff is None else min(eff, v)
    eff_bb = (eff or 0) / 100.0

    def bucket(x, edges):
        b = 0
        for e in edges:
            if x >= e:
                b += 1
            else:
                break
        return b

    pot_b = bucket(pot_bb, [3.0, 6.0, 12.0, 25.0, 50.0, 100.0, 200.0])
    tocall_b = bucket(to_call / max(pot, 1), [0.01, 0.34, 0.67, 1.01, 2.01])
    spr_b = bucket(eff_bb / max(pot_bb, 1e-6), [1.0, 2.5, 5.0, 10.0, 20.0])
    is_btn = 1 if seat == int(a[O_BUTTON]) else 0
    la = int(a[O_LAST_AGGRESSOR])
    facing = 1 if (la >= 0 and la != seat) else 0
    return (int(a[O_STREET]), pot_b, tocall_b, spr_b, is_btn, facing)

def hist3(a):
    """Mirror of fingerprint.rs preflop_hist_bytes."""
    n_calls = n_raises = 0
    opener_seat = -1
    for i in range(int(a[O_HIST_LEN])):
        act = int(a[O_HIST_ACTION + i])
        if act == CALL:
            if n_raises == 0 and n_calls < 3:
                n_calls += 1
        elif BET0 <= act <= ALL_IN:
            if n_raises == 0:
                opener_seat = int(a[O_HIST_SEAT + i])
            if n_raises < 3:
                n_raises += 1
    if opener_seat >= 0:
        off = (opener_seat - int(a[O_BUTTON])) % 6
        op = POS_BY_OFFSET[off]
    else:
        op = -1
    return (n_calls, n_raises, op + 1)

# ---------------------------------------------------------------------------
# T1 BOUNDED-RECALL POSTFLOP KEY (arm-5 design of record, blueprint_pluribus_grade_design_v2 §4.1).
#
# Twin of the designed fingerprint.rs::push_full_hist_bytes edit (573-581): PREFLOP stays fullhist
# (exact recall, uncapped), each POSTFLOP street keeps EXACT action-id bytes up to AND INCLUDING the
# byte that lands the Nth RAISE (aggressive: id>=BET0 or id==ALL_IN) on that street; FOLD/CHECK/CALL
# are ALWAYS emitted verbatim and do NOT count toward the cap. Once the current street's raise count
# reaches N, the tail is replaced by ONE summary token and the street advances. Prior (closed) streets
# are already bounded (their cap fired at street close) and emitted exact-and-full.
#
# Summary token (design summary byte, here a tuple for the census; same information content):
#   ( war_depth = min(subsequent_raises_beyond_Nth, 3),
#     spr_bucket at the cap point (0..5 via SPR_EDGES),
#     pot_bucket at the cap point (0..7 via POT_EDGES; design packs to 0..3, kept finer here so the
#                 census is a CONSERVATIVE upper bound on distinct rows),
#     opener_flag = 1 if last aggressor at cap point is the preflop opener )
# All four features are read from the SAME state fields betting6/betting_abstraction_key already touch,
# so no extra traversal. The census tags each history entry with the street of the node that produced
# it (threaded down the recursion) -> exact street membership, no boundary-reconstruction heuristic
# (the Rust twin reconstructs the identical boundaries by transition-tracking the same forward scan).
# ---------------------------------------------------------------------------

def t1_cap_point_buckets(a, seat, opener_seat):
    """SPR/pot bucket + initiative flag at the CAP-POINT state `a` (the state right after the Nth
    raise closed on a street). Reuses betting6's SPR/pot bucketing (== betting_abstraction_key)."""
    _, pot_b, _tocall_b, spr_b, _is_btn, _facing = betting6(a, seat)
    la = int(a[O_LAST_AGGRESSOR])
    opener_flag = 1 if (la >= 0 and la == opener_seat) else 0
    return (spr_b, min(pot_b, 7), opener_flag)


def t1_postflop_key(entries, cap):
    """Build the bounded-recall POSTFLOP key from street-tagged history entries.

    entries: list of dicts, one per postflop history entry (street>=1), in order, each
      {street, action, is_raise, cap_buckets} where cap_buckets = (spr_b, pot_b, opener_flag) at the
      CAP-POINT is set ONLY on the entry that lands the Nth raise on its street (None otherwise).
      PREFLOP entries are excluded (the caller keeps them exact in the preflop portion of the row key).
    cap: N (per-street raise cap). Returns a hashable tuple.

    Per street: emit exact action ids (folds/checks/calls always; only raises counted) until the Nth
    raise, then emit ONE summary token
        ("T1", war_depth=min(raises_beyond_Nth,3), spr_b, pot_b, opener_flag)
    and stop emitting per-action ids for that street. war_depth distinguishes 0/1/2/3+ further raises
    (design summary-byte bits[0-1]) -> at most 4 distinct tails per (cap_buckets), which is exactly
    why the postflop set is FINITE and saturates (vs the uncapped chain that grew unboundedly)."""
    out = []
    cur_street = None
    raises_this_street = 0
    capped = False
    cap_buckets = None
    further = 0

    def flush_summary():
        # emit the summary token for the just-closed capped street (if any)
        if cap_buckets is not None:
            spr_b, pot_b, opener_flag = cap_buckets
            out.append(("T1", min(further, 3), spr_b, pot_b, opener_flag))

    for e in entries:
        st = e["street"]
        if st != cur_street:
            flush_summary()
            cur_street = st
            raises_this_street = 0
            capped = False
            cap_buckets = None
            further = 0
            out.append(("st", st))
        if e["is_raise"]:
            raises_this_street += 1
            if raises_this_street < cap:
                out.append(("a", e["action"]))
            elif raises_this_street == cap:
                out.append(("a", e["action"]))  # emit the Nth raise exactly
                cap_buckets = e["cap_buckets"]   # record cap-point summary buckets
                capped = True
            else:
                further += 1                      # war depth beyond the Nth (aliased to a count)
        elif not capped:
            out.append(("a", e["action"]))        # fold/check/call before cap: exact
        # fold/check/call AFTER cap are dropped (subsumed by the summary; they carry no new size info)
    flush_summary()
    return tuple(out)


def memo_key(state_bytes):
    """Betting-relevant canonical state (excludes cards/deck/history/rng), hashed to a
    96-bit digest to keep the memo memory-lean (collision prob ~(n^2)/2^97, negligible at
    <=10^8 states). Byte slices of the flat i64 buffer:
      [17..55) = seat_status, stacks, committed_street, committed_total, street, pot_dead,
                 current_bet, min_raise, button, action_seat, last_aggressor, acted,
                 last_raise_full;  [303..304) = done;  [310..311) = pending_deal."""
    return hashlib.blake2b(
        state_bytes[17 * 8:55 * 8] + state_bytes[303 * 8:304 * 8]
        + state_bytes[310 * 8:311 * 8],
        digest_size=12).digest()

ZERO = (0, 0, 0, 0, 0, 0, 0, 0)  # d0..d3, e1..e3, terminal-paths

def add(u, v):
    return tuple(x + y for x, y in zip(u, v))

def enumerate_menu(menu_name, dedupe, max_states, log_every, progress, preflop_only=False,
                   mrr=False, flop_rr_keep=None):
    pre_mask, pre_rr_mask, post_mask = menu_masks(MENUS[menu_name])
    deck = list(range(52))
    root = bn.init_hand(0, [10000] * 6, [1] * 6, 0, deck, [1, 2])
    bb = int(arr(root)[O_CURRENT_BET])  # big blind = root current_bet
    memo = {}
    key_groups = set()      # (pos, betting6) at preflop decision nodes  (deployed keying)
    key_groups_h3 = set()   # (pos, betting6, hist3)                     (bp3 keying anchor)
    t0 = time.time()
    aborted = [False]
    sys.setrecursionlimit(500)

    # iterative DFS with explicit stack: (state, phase, children, accum)
    def solve(root_state):
        stack = [[root_state, None, None, ZERO]]
        results = {}  # id-less: we use memo keyed by memo_key
        out = None
        while stack:
            frame = stack[-1]
            state, kids, ki, acc = frame
            a = arr(state)
            k = memo_key(state)
            if kids is None:
                if k in memo:
                    stack.pop()
                    out = memo[k]
                    if stack:
                        stack[-1][3] = add(stack[-1][3], out)
                    continue
                # sample_chance advances a closed river to STREET_TERMINAL WITHOUT setting
                # done (trainer handles this "post-chance terminal" the same way).
                if int(a[O_DONE]) or int(a[O_STREET]) >= 4 or (
                        not int(a[O_PENDING_DEAL]) and int(a[O_ACTION_SEAT]) < 0):
                    memo[k] = (0,) * 7 + (1,)
                    stack.pop()
                    out = memo[k]
                    if stack:
                        stack[-1][3] = add(stack[-1][3], out)
                    continue
                if preflop_only and int(a[O_STREET]) > 0:
                    # count as a flop-entry path and stop (street>0 unreachable here except
                    # via the pending_deal branch below, which already counts entries)
                    memo[k] = ZERO
                    stack.pop()
                    out = memo[k]
                    if stack:
                        stack[-1][3] = add(stack[-1][3], out)
                    continue
                if int(a[O_PENDING_DEAL]):
                    child = bn.sample_chance(state)
                    ca = arr(child)
                    st = int(ca[O_STREET])
                    entry = [0] * 8
                    if 1 <= st <= 3:
                        entry[3 + st] = 1
                    frame[1] = [child]
                    frame[2] = 0
                    frame[3] = tuple(entry)
                    continue
                # decision node
                if len(memo) >= max_states:
                    aborted[0] = True
                    memo[k] = ZERO
                    stack.pop()
                    out = memo[k]
                    if stack:
                        stack[-1][3] = add(stack[-1][3], out)
                    continue
                seat = int(a[O_ACTION_SEAT])
                street = int(a[O_STREET])
                d = [0] * 8
                d[min(street, 3)] = 1
                if street == 0:
                    off = (seat - int(a[O_BUTTON])) % 6
                    pos = POS_BY_OFFSET[off]
                    b6 = betting6(a, seat)
                    key_groups.add((pos, b6))
                    key_groups_h3.add((pos, b6, hist3(a)))
                acts = allowed_actions(state, a, pre_mask, post_mask, dedupe, mrr,
                                       flop_rr_keep, pre_rr_mask, bb)
                frame[1] = [bn.apply_action(state, act) for act in acts]
                frame[2] = 0
                frame[3] = tuple(d)
                if len(memo) % log_every == 0:
                    msg = f"states={len(memo)} elapsed={time.time()-t0:.0f}s"
                    progress(msg)
                continue
            if ki < len(kids):
                frame[2] = ki + 1
                stack.append([kids[ki], None, None, ZERO])
                continue
            memo[k] = acc
            stack.pop()
            out = acc
            if stack:
                stack[-1][3] = add(stack[-1][3], out)
        return out

    res = solve(root)
    return {
        "menu": menu_name,
        "mrr": mrr,
        "flop_rr_keep": flop_rr_keep,
        "dedupe_equal_amount": dedupe,
        "aborted_at_max_states": aborted[0],
        "distinct_betting_states": len(memo),
        "decision_histories": {
            "preflop": res[0], "flop": res[1], "turn": res[2], "river": res[3]},
        "street_entry_histories": {"flop": res[4], "turn": res[5], "river": res[6]},
        "terminal_histories": res[7],
        "preflop_key_groups_state6": len(key_groups),
        "preflop_key_groups_state6_hist3": len(key_groups_h3),
        "elapsed_s": round(time.time() - t0, 1),
    }

def walk_census(menu_name, dedupe, n_deals, seed, progress, snap_every=1000, mrr=False,
                flop_rr_keep=None, postflop_key="fullhist", t1_raise_cap=3):
    """Trainer-visitation census: for each deal, run 6 hero traversals (hero branches ALL
    allowed actions; opponents sample ONE uniform-over-allowed). Count distinct FULL-HISTORY
    rows per street: key = (street, actor-derived pos [pre] , card-class/bucket, action hist).
    Card part: preflop = 169-class of hero; postflop = made-hand bucket 200 via
    bn.fingerprint_fast digest (cheap, deployed-shape); the history part is exact.

    postflop_key: "fullhist" = exact flat action history (baseline; the CURRENT shipped fullhist
      key). "t1" = arm-5 BOUNDED-RECALL postflop key: PREFLOP stays exact, each POSTFLOP street keeps
      exact action bytes up to+including the Nth (t1_raise_cap) raise then one summary token. The
      SATURATION arm: measures whether the T1 reached-row curve PLATEAUS vs fullhist's linear growth.

    Both key modes are computed on EVERY node in the SAME traversal, so preflop rows are IDENTICAL
    between arms by construction (T1 only rewrites the postflop portion) -> the sanity check is free."""
    use_t1 = (postflop_key == "t1")
    use_presum = (postflop_key == "presum")
    pre_mask, pre_rr_mask, post_mask = menu_masks(MENUS[menu_name])
    bb = int(arr(bn.init_hand(0, [10000] * 6, [1] * 6, 0, list(range(52)), [1, 2]))[
        O_CURRENT_BET])  # big blind = root current_bet
    rng = np.random.default_rng(seed)
    rows = [set(), set(), set(), set()]        # postflop-key rows per street (fullhist OR t1)
    rows_summary = [set(), set(), set(), set()]  # deployed summary keys (digest) per street
    rows_betting = [set(), set(), set(), set()]  # CARD-INDEPENDENT history-only keys per street: the
    # faithful saturation signal (real card dim is a bounded 200-bucket; digest[:5] over-counts it, so
    # the row curve's true boundedness is (betting-key count) x 200 -- if betting keys saturate, rows do)
    curve = []
    t0 = time.time()
    nodes = [0]

    def opener_of(a):
        """Preflop opener seat (first aggressor in the flat history), -1 if none. Used for the
        T1 summary's initiative flag; matches hist3's opener detection."""
        for i in range(int(a[O_HIST_LEN])):
            act = int(a[O_HIST_ACTION + i])
            if BET0 <= act <= ALL_IN:
                return int(a[O_HIST_SEAT + i])
        return -1

    def pre_summary_of(a, n_pre):
        """PRESUM (arm-6) preflop pot-type summary carried into POSTFLOP keys: replaces the EXACT
        preflop-action prefix with (n_calls<=3, n_raises<=3, opener_pos, n_active_to_flop) -- a
        bp3-hist3-style low-cardinality descriptor computed over ONLY the first n_pre (preflop)
        history entries. Isolates the preflop-prefix multiplier that menu/recall both failed to move."""
        nr = nc = 0
        opener = -1
        for i in range(n_pre):
            act = int(a[O_HIST_ACTION + i])
            if act == CALL:
                if nr == 0 and nc < 3:
                    nc += 1
            elif BET0 <= act <= ALL_IN:
                if nr == 0:
                    opener = int(a[O_HIST_SEAT + i])
                if nr < 3:
                    nr += 1
        op = POS_BY_OFFSET[(opener - int(a[O_BUTTON])) % 6] if opener >= 0 else -1
        n_active = sum(1 for i in range(6) if int(a[O_SEAT_STATUS + i]) in (0, 1))
        return (nc, nr, op, n_active)

    def traverse(state, hero, post_entries):
        # post_entries: list of street-tagged POSTFLOP history entries seen on the path so far, in
        # order. Each = {street, action, is_raise, cap_buckets}. cap_buckets (spr_b,pot_b,opener_flag)
        # is set only on the entry that lands the Nth raise on its street (buckets read from the state
        # AFTER that raise = the "cap point"). Preflop entries are excluded (kept exact in the key).
        a = arr(state)
        if int(a[O_DONE]) or int(a[O_STREET]) >= 4:
            return
        if int(a[O_PENDING_DEAL]):
            traverse(bn.sample_chance(state), hero, post_entries)
            return
        if int(a[O_ACTION_SEAT]) < 0:
            return
        seat = int(a[O_ACTION_SEAT])
        street = min(int(a[O_STREET]), 3)
        nodes[0] += 1
        if nodes[0] > 250_000:
            raise RuntimeError("node cap")
        digest = bn.fingerprint_fast(state, seat, 200)
        rows_summary[street].add(digest)
        if use_t1:
            # PREFLOP portion exact (flat preflop actions), POSTFLOP portion bounded-recall.
            n_pre = int(a[O_HIST_LEN]) - len(post_entries)  # #preflop entries = total - postflop
            pre_hist = tuple(int(a[O_HIST_ACTION + i]) for i in range(n_pre))
            post_key = t1_postflop_key(post_entries, t1_raise_cap)
            rows[street].add((digest[:5], pre_hist, post_key))
            rows_betting[street].add((pre_hist, post_key))
        elif use_presum:
            # arm-6 PRESUM: preflop DECISIONS keep exact fullhist (rows identical to baseline, so the
            # preflop sanity check still holds); POSTFLOP infosets replace the EXACT preflop-action
            # prefix with a low-cardinality pot-type SUMMARY (n_calls/n_raises/opener_pos/n_active),
            # keeping the postflop action sequence EXACT. Isolates the measured dominant multiplier
            # (exact-preflop-prefix x card) that menu (arm-4) and recall (arm-5) both failed to touch.
            hl = int(a[O_HIST_LEN])
            if street == 0:
                hist = tuple(int(a[O_HIST_ACTION + i]) for i in range(hl))
                rows[street].add((digest[:5], hist))
                rows_betting[street].add(hist)
            else:
                n_pre = hl - len(post_entries)  # #preflop entries = total - postflop
                pre_sum = pre_summary_of(a, n_pre)
                post_hist = tuple(int(a[O_HIST_ACTION + i]) for i in range(n_pre, hl))
                rows[street].add((digest[:5], pre_sum, post_hist))
                rows_betting[street].add((pre_sum, post_hist))
        else:
            hist = tuple(int(a[O_HIST_ACTION + i]) for i in range(int(a[O_HIST_LEN])))
            rows[street].add((digest[:5], hist))  # digest[:5] carries card-bucket byte-space;
            # (bucket,hist) distinctness == (digest_cardpart,hist): betting6 is hist-redundant
            rows_betting[street].add(hist)
        acts = allowed_actions(state, a, pre_mask, post_mask, dedupe, mrr,
                               flop_rr_keep, pre_rr_mask, bb)

        def child_post_entries(act, child):
            """Append this action's POSTFLOP entry (street tag known = current street) with its
            cap_buckets if it lands the Nth raise. Preflop actions (street 0) add nothing.
            `child` = bn.apply_action(state, act) (reused from the traverse call, no double-apply)."""
            if (not use_t1 and not use_presum) or street == 0:
                return post_entries
            is_raise = (BET0 <= act <= ALL_IN)  # aggressive: any bet-bucket or all-in
            cap_buckets = None
            if use_t1:
                # count raises already on THIS street within post_entries
                r_this = 0
                for e in reversed(post_entries):
                    if e["street"] != street:
                        break
                    if e["is_raise"]:
                        r_this += 1
                if is_raise and (r_this + 1) == t1_raise_cap:
                    # this raise lands the Nth on this street -> summary buckets from the CAP-POINT
                    # state (state AFTER applying this raise = the child), keyed seat-independently.
                    cap_buckets = t1_cap_point_buckets(arr(child), seat, opener_of(a))
            return post_entries + [{"street": street, "action": act,
                                    "is_raise": is_raise, "cap_buckets": cap_buckets}]

        def go(act):
            child = bn.apply_action(state, act)
            traverse(child, hero, child_post_entries(act, child))

        if seat == hero:
            for act in acts:
                go(act)
        else:
            go(acts[int(rng.integers(len(acts)))])

    deck0 = np.arange(52)
    done_deals = 0
    for d in range(n_deals):
        deck = rng.permutation(deck0).astype(int).tolist()
        root = bn.init_hand(d % 6, [10000] * 6, [1] * 6, 0, deck, [1, 2])
        try:
            nodes[0] = 0
            for hero in range(6):
                traverse(root, hero, [])
        except (RuntimeError, RecursionError):
            pass
        done_deals += 1
        if done_deals % snap_every == 0:
            snap = {
                "deals": done_deals,
                "rows": [len(s) for s in rows],
                "rows_betting": [len(s) for s in rows_betting],
                "rows_summary": [len(s) for s in rows_summary],
                "elapsed_s": round(time.time() - t0, 1),
            }
            curve.append(snap)
            progress(json.dumps(snap))
    # marginal reached-rows per snap_every deals (the SATURATION signal): does the TOTAL postflop
    # reached-row curve's slope collapse toward zero (plateau) as deals accumulate?
    marginal = []
    prev_total = 0
    prev_bet = 0
    for snap in curve:
        total_post = snap["rows"][1] + snap["rows"][2] + snap["rows"][3]
        bet_post = snap["rows_betting"][1] + snap["rows_betting"][2] + snap["rows_betting"][3]
        marginal.append({"deals": snap["deals"],
                         "postflop_total": total_post,
                         "new_postflop_this_block": total_post - prev_total,
                         "betting_postflop_total": bet_post,
                         "new_betting_this_block": bet_post - prev_bet})
        prev_total = total_post
        prev_bet = bet_post
    return {
        "menu": menu_name, "mrr": mrr, "flop_rr_keep": flop_rr_keep, "dedupe_equal_amount": dedupe,
        "postflop_key": postflop_key, "t1_raise_cap": (t1_raise_cap if use_t1 else None),
        "snap_every": snap_every,
        "deals": done_deals, "seed": seed,
        "rows_final": {"preflop": len(rows[0]), "flop": len(rows[1]),
                       "turn": len(rows[2]), "river": len(rows[3]),
                       "postflop_total": len(rows[1]) + len(rows[2]) + len(rows[3])},
        "rows_betting_final": {"preflop": len(rows_betting[0]), "flop": len(rows_betting[1]),
                               "turn": len(rows_betting[2]), "river": len(rows_betting[3]),
                               "postflop_total": len(rows_betting[1]) + len(rows_betting[2]) + len(rows_betting[3])},
        "rows_summary_final": {"preflop": len(rows_summary[0]), "flop": len(rows_summary[1]),
                                "turn": len(rows_summary[2]), "river": len(rows_summary[3])},
        "marginal_postflop_per_block": marginal,
        "curve": curve,
        "elapsed_s": round(time.time() - t0, 1),
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["enumerate", "walk"])
    ap.add_argument("--menu", default="current", choices=sorted(MENUS))
    ap.add_argument("--no-dedupe", action="store_true")
    ap.add_argument("--max-states", type=int, default=12_000_000)
    ap.add_argument("--deals", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--log-every", type=int, default=200_000)
    ap.add_argument("--out", default="")
    ap.add_argument("--preflop-only", action="store_true")
    ap.add_argument("--mrr", action="store_true",
                    help="apply Pluribus deep-node rule: postflop re-raises thinned to {1x pot, all-in}"
                         " AND turn/river first-raises thinned to {1/2 pot, 1x pot, all-in}")
    ap.add_argument("--flop-rr-keep", default="",
                    help="(only meaningful with --mrr) comma list of bet-bucket indices kept for"
                         " FLOP re-raise nodes, e.g. '1,4' = {1/2 pot, 1x pot, all-in}."
                         " Absent = legacy [4]. Turn/river re-raise keep stays [4].")
    ap.add_argument("--postflop-key", default="fullhist", choices=["fullhist", "t1", "presum"],
                    help="postflop infoset-key mode for the WALK census. fullhist = exact flat action"
                         " history (baseline, the CURRENT shipped fullhist key). t1 = arm-5 BOUNDED-"
                         "RECALL key: preflop exact, each postflop street exact up to+including the"
                         " Nth (--t1-raise-cap) raise then one summary token. Measures saturation.")
    ap.add_argument("--t1-raise-cap", type=int, default=3,
                    help="(t1 only) per-postflop-street raise cap N. Design of record = 3.")
    ap.add_argument("--snap-every", type=int, default=1000,
                    help="deals per curve/marginal snapshot (saturation-trend granularity).")
    args = ap.parse_args()
    flop_rr_keep = None
    if args.flop_rr_keep:
        flop_rr_keep = [int(x) for x in args.flop_rr_keep.split(",")]

    def progress(msg):
        tag = "+mrr" if args.mrr else ""
        if args.mrr and flop_rr_keep is not None:
            tag += "+frr" + ".".join(str(b) for b in flop_rr_keep)
        if args.mode == "walk" and args.postflop_key == "t1":
            tag += f"+t1cap{args.t1_raise_cap}"
        print(f"[census {args.mode} {args.menu}{tag}] {msg}", flush=True)

    if args.mode == "enumerate":
        res = enumerate_menu(args.menu, not args.no_dedupe, args.max_states,
                             args.log_every, progress, preflop_only=args.preflop_only,
                             mrr=args.mrr, flop_rr_keep=flop_rr_keep)
        res["preflop_only"] = args.preflop_only
    else:
        res = walk_census(args.menu, not args.no_dedupe, args.deals, args.seed, progress,
                          snap_every=args.snap_every, mrr=args.mrr, flop_rr_keep=flop_rr_keep,
                          postflop_key=args.postflop_key, t1_raise_cap=args.t1_raise_cap)
    js = json.dumps(res, indent=2)
    print(js)
    if args.out:
        with open(args.out, "w") as f:
            f.write(js)

if __name__ == "__main__":
    main()
