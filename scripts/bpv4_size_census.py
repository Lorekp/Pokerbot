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
    return pre, post

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

def allowed_actions(state, a, pre_mask, post_mask, dedupe, mrr=False):
    """legal AND menu, with equal-final-commit aggressive actions deduped (lowest idx kept).

    mrr=True applies Pluribus's deep-node rule (design M-RR, §4.2.2 / P-supp p.12-13): POSTFLOP,
    a RE-raise (raising while FACING a live bet) is thinned to {1x pot, all-in}; the FIRST bet of
    a street keeps the full postflop menu. This is action-abstraction thinning (perfect recall of
    the *thinned* actions preserved), NOT history/key truncation — the design's "structural relief
    valve, ahead of key truncation" that collapses the ~10^30 history space toward Pluribus's
    finite 664.8M-class scale."""
    legal = bn.legal_action_mask(state, -1)
    street = int(a[O_STREET])
    menu = pre_mask if street == 0 else post_mask
    acts = [i for i in range(11) if legal[i] and menu[i]]
    if mrr and street >= 1:
        seat = int(a[O_ACTION_SEAT])
        facing_bet = int(a[O_CURRENT_BET]) > int(a[O_COMMITTED_STREET + seat])
        if facing_bet:  # RE-raise decision -> thin raise sizes to {1x pot, all-in}
            keep = {FOLD, CHECK, CALL, ALL_IN, BET0 + 4}  # BET0+4 = 1.0x pot bucket
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
                   mrr=False):
    pre_mask, post_mask = menu_masks(MENUS[menu_name])
    deck = list(range(52))
    root = bn.init_hand(0, [10000] * 6, [1] * 6, 0, deck, [1, 2])
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
                acts = allowed_actions(state, a, pre_mask, post_mask, dedupe, mrr)
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

def walk_census(menu_name, dedupe, n_deals, seed, progress, snap_every=1000, mrr=False):
    """Trainer-visitation census: for each deal, run 6 hero traversals (hero branches ALL
    allowed actions; opponents sample ONE uniform-over-allowed). Count distinct FULL-HISTORY
    rows per street: key = (street, actor-derived pos [pre] , card-class/bucket, action hist).
    Card part: preflop = 169-class of hero; postflop = made-hand bucket 200 via
    bn.fingerprint_fast digest (cheap, deployed-shape); the history part is exact."""
    pre_mask, post_mask = menu_masks(MENUS[menu_name])
    rng = np.random.default_rng(seed)
    rows = [set(), set(), set(), set()]        # full-history keys per street
    rows_summary = [set(), set(), set(), set()]  # deployed summary keys (digest) per street
    curve = []
    t0 = time.time()
    nodes = [0]

    def traverse(state, hero):
        a = arr(state)
        if int(a[O_DONE]) or int(a[O_STREET]) >= 4:
            return
        if int(a[O_PENDING_DEAL]):
            traverse(bn.sample_chance(state), hero)
            return
        if int(a[O_ACTION_SEAT]) < 0:
            return
        seat = int(a[O_ACTION_SEAT])
        street = min(int(a[O_STREET]), 3)
        nodes[0] += 1
        if nodes[0] > 250_000:
            raise RuntimeError("node cap")
        hist = tuple(int(a[O_HIST_ACTION + i]) for i in range(int(a[O_HIST_LEN])))
        digest = bn.fingerprint_fast(state, seat, 200)
        rows_summary[street].add(digest)
        rows[street].add((digest[:5], hist))  # digest[:5] carries card-bucket byte-space;
        # (bucket,hist) distinctness == (digest_cardpart,hist): betting6 is hist-redundant
        acts = allowed_actions(state, a, pre_mask, post_mask, dedupe, mrr)
        if seat == hero:
            for act in acts:
                traverse(bn.apply_action(state, act), hero)
        else:
            act = acts[int(rng.integers(len(acts)))]
            traverse(bn.apply_action(state, act), hero)

    deck0 = np.arange(52)
    done_deals = 0
    for d in range(n_deals):
        deck = rng.permutation(deck0).astype(int).tolist()
        root = bn.init_hand(d % 6, [10000] * 6, [1] * 6, 0, deck, [1, 2])
        try:
            nodes[0] = 0
            for hero in range(6):
                traverse(root, hero)
        except (RuntimeError, RecursionError):
            pass
        done_deals += 1
        if done_deals % snap_every == 0:
            snap = {
                "deals": done_deals,
                "rows_fullhist": [len(s) for s in rows],
                "rows_summary": [len(s) for s in rows_summary],
                "elapsed_s": round(time.time() - t0, 1),
            }
            curve.append(snap)
            progress(json.dumps(snap))
    return {
        "menu": menu_name, "mrr": mrr, "dedupe_equal_amount": dedupe, "deals": done_deals,
        "seed": seed,
        "rows_fullhist_final": {"preflop": len(rows[0]), "flop": len(rows[1]),
                                 "turn": len(rows[2]), "river": len(rows[3])},
        "rows_summary_final": {"preflop": len(rows_summary[0]), "flop": len(rows_summary[1]),
                                "turn": len(rows_summary[2]), "river": len(rows_summary[3])},
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
                    help="apply Pluribus deep-node rule: postflop re-raises thinned to {1x pot, all-in}")
    args = ap.parse_args()

    def progress(msg):
        tag = "+mrr" if args.mrr else ""
        print(f"[census {args.mode} {args.menu}{tag}] {msg}", flush=True)

    if args.mode == "enumerate":
        res = enumerate_menu(args.menu, not args.no_dedupe, args.max_states,
                             args.log_every, progress, preflop_only=args.preflop_only,
                             mrr=args.mrr)
        res["preflop_only"] = args.preflop_only
    else:
        res = walk_census(args.menu, not args.no_dedupe, args.deals, args.seed, progress,
                          mrr=args.mrr)
    js = json.dumps(res, indent=2)
    print(js)
    if args.out:
        with open(args.out, "w") as f:
            f.write(js)

if __name__ == "__main__":
    main()
