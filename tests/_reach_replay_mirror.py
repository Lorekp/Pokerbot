"""Faithful Python mirror of subgame.rs replay_public_states (the native reach conditioner's
per-step history replay), built on the certified np_engine twins. Used as an in-repo reach-
ENGAGEMENT gate for the arena reconstruction WITHOUT needing the battery_native pyd: it
reproduces the exact per-step verification (rebuild_hand_start, actor match, apply_action_recorded
under both amount conventions, the phantom aggressor-option CHECK, derived+zero antes) and the
final public_eq acceptance. Verified to reproduce scripts/probe_arena_reach_engagement.py
byte-for-byte (same engaged/uniform verdict per spot). Test helper (underscore ⇒ not collected).
"""
from __future__ import annotations

import os


import numpy as np  # noqa: E402

from pokerbot.engine.jax_engine import (  # noqa: E402
    ALL_IN, BB_CHIPS, BET_BUCKET_START, BET_SIZE_BUCKETS, CALL, CHECK, FOLD, MAX_HISTORY,
    NUM_ACTIONS, SB_CHIPS, STREET_PREFLOP, STREET_TERMINAL,
)
from pokerbot.engine.np_engine import (  # noqa: E402
    SEAT_ALLIN, SEAT_FOLDED, SEAT_LIVE, _close_street, _first_live_after, _street_complete,
    apply_action_np, is_chance_node_np, legal_action_mask_np, sample_chance_np,
)

SEAT_EMPTY = np.int8(0)
_BF32 = [np.float32(x) for x in BET_SIZE_BUCKETS]


def _nth_live_after(seat_status, button, n):
    seat = int(button)
    N = len(seat_status)
    for _ in range(n):
        for _ in range(N):
            seat = (seat + 1) % N
            if seat_status[seat] != SEAT_EMPTY:
                break
    return seat


def _pot_for_sizing(s):
    return int(s.pot_dead) + int(np.asarray(s.committed_street).sum())


def _engine_action_amount(s, seat, action):
    committed = int(np.asarray(s.committed_street)[seat])
    to_call = max(0, int(s.current_bet) - committed)
    stack = int(np.asarray(s.stacks)[seat])
    pot = _pot_for_sizing(s)
    if action == CALL:
        return to_call
    if action == ALL_IN:
        return stack
    if BET_BUCKET_START <= action < ALL_IN:
        frac = _BF32[action - BET_BUCKET_START]
        prod = int(np.int32(np.float32(pot) * frac))
        target = max(int(s.current_bet), prod)
        return to_call + max(0, target - committed)
    return 0


def rebuild_hand_start(root, ante):
    N = len(root.seat_status)
    status0 = np.array([SEAT_LIVE if root.seat_status[i] != SEAT_EMPTY else SEAT_EMPTY
                        for i in range(N)], dtype=np.int8)
    stacks0 = np.array([int(root.stacks[i]) + int(root.committed_total[i]) for i in range(N)],
                       dtype=np.int32)
    button = int(root.button)
    sb = _nth_live_after(status0, button, 1)
    bb = _nth_live_after(status0, button, 2)
    if sb == bb:
        return None
    committed = np.zeros(N, dtype=np.int32)
    sb_paid = min(SB_CHIPS, max(0, int(stacks0[sb])))
    bb_paid = min(BB_CHIPS, max(0, int(stacks0[bb])))
    stacks0[sb] -= sb_paid
    stacks0[bb] -= bb_paid
    committed[sb] = sb_paid
    committed[bb] = bb_paid
    committed_total0 = np.zeros(N, dtype=np.int32)
    pot_dead0 = 0
    for i in range(N):
        a = int(ante[i])
        if status0[i] == SEAT_EMPTY:
            if a != 0:
                return None
            continue
        if a < 0 or a > int(stacks0[i]):
            return None
        stacks0[i] -= a
        pot_dead0 += a
        committed_total0[i] = committed[i] + a
    for i in range(N):
        if status0[i] != SEAT_EMPTY and stacks0[i] == 0:
            status0[i] = SEAT_ALLIN
    current_bet0 = int(committed.max())
    n_live = int((status0 == SEAT_LIVE).sum())
    action_seat0 = _nth_live_after(status0, button, 1) if n_live <= 2 else _nth_live_after(status0, button, 3)
    deck = np.zeros(52, dtype=np.int32)
    used = [False] * 52
    p = 0
    for c in np.asarray(root.board):
        c = int(c)
        if c >= 0:
            if used[c]:
                return None
            deck[p] = c
            used[c] = True
            p += 1
    for c in range(52):
        if not used[c]:
            deck[p] = c
            p += 1
    return root._replace(
        holes=np.full((N, 2), -1, dtype=np.int32), board=np.full(5, -1, dtype=np.int32),
        seat_status=status0, stacks=stacks0, committed_street=committed,
        committed_total=committed_total0, street=np.int32(STREET_PREFLOP), pot_dead=np.int32(pot_dead0),
        current_bet=np.int32(current_bet0), min_raise=np.int32(BB_CHIPS), action_seat=np.int32(action_seat0),
        last_aggressor=np.int32(-1), acted=np.zeros(N, dtype=bool), last_raise_full=np.bool_(True),
        deck=deck, deck_ptr=np.int32(0), hist_seat=np.full(MAX_HISTORY, -1, dtype=np.int32),
        hist_action=np.full(MAX_HISTORY, -1, dtype=np.int32), hist_amount=np.zeros(MAX_HISTORY, dtype=np.int32),
        hist_len=np.int32(0), done=np.bool_(False), rewards=np.zeros(N, dtype=np.int32),
        pending_deal=np.bool_(False))


def apply_action_recorded(s, action, recorded, total_convention):
    N = len(s.seat_status)
    seat = int(s.action_seat)
    if seat < 0 or seat >= N or action >= NUM_ACTIONS:
        return None
    full = np.asarray(legal_action_mask_np(s, seat)).astype(bool)
    engine_amt = _engine_action_amount(s, seat, action)
    if action < BET_BUCKET_START or action == ALL_IN or recorded == engine_amt:
        if not full[action]:
            return None
        return apply_action_np(s, action)
    if s.seat_status[seat] != SEAT_LIVE:
        return None
    committed = int(np.asarray(s.committed_street)[seat])
    to_call = max(0, int(s.current_bet) - committed)
    reopen_ok = (not bool(np.asarray(s.acted)[seat])) or bool(s.last_raise_full) or to_call == 0
    if not reopen_ok:
        return None
    if recorded < 0:
        return None
    inc = (recorded - committed) if total_convention else recorded
    if inc <= 0:
        return None
    stacks = np.asarray(s.stacks).copy()
    amount = min(inc, int(stacks[seat]))
    if amount <= 0:
        return None
    cs = np.asarray(s.committed_street).copy()
    ct = np.asarray(s.committed_total).copy()
    status = np.asarray(s.seat_status).copy()
    stacks[seat] -= amount
    cs[seat] += amount
    ct[seat] += amount
    if stacks[seat] <= 0:
        status[seat] = SEAT_ALLIN
    prev_bet = int(s.current_bet)
    new_street_bet = int(cs[seat])
    if new_street_bet <= prev_bet:
        return None
    full_raise = new_street_bet >= prev_bet + int(s.min_raise)
    if not full_raise and stacks[seat] > 0:
        return None
    acted = np.asarray(s.acted).copy()
    acted[seat] = True
    if full_raise:
        acted = np.zeros(N, dtype=bool)
        acted[seat] = True
    raise_size = new_street_bet - prev_bet
    hist_seat = np.asarray(s.hist_seat).copy()
    hist_action = np.asarray(s.hist_action).copy()
    hist_amount = np.asarray(s.hist_amount).copy()
    idx = int(s.hist_len)
    if idx < MAX_HISTORY:
        hist_seat[idx] = seat
        hist_action[idx] = action
        hist_amount[idx] = amount
    s2 = s._replace(
        stacks=stacks, committed_street=cs, committed_total=ct, seat_status=status,
        current_bet=np.int32(new_street_bet), min_raise=np.int32(max(BB_CHIPS, raise_size)),
        last_aggressor=np.int32(seat), last_raise_full=np.bool_(full_raise), acted=acted,
        hist_seat=hist_seat, hist_action=hist_action, hist_amount=hist_amount,
        hist_len=np.int32(min(idx + 1, MAX_HISTORY)))
    n_live = int(((status == SEAT_LIVE) | (status == SEAT_ALLIN)).sum())
    terminal = n_live <= 1
    if terminal:
        s2 = s2._replace(street=np.int32(STREET_TERMINAL), pending_deal=np.bool_(False))
    elif _street_complete(s2):
        s2 = _close_street(s2)
    terminal = terminal or int(s2.street) >= STREET_TERMINAL or n_live <= 1
    if terminal or bool(s2.pending_deal):
        s2 = s2._replace(action_seat=np.int32(-1), done=np.bool_(terminal))
    else:
        s2 = s2._replace(action_seat=_first_live_after(s2.seat_status, s2.action_seat),
                         done=np.bool_(False))
    return s2


def phantom_check_applicable(s):
    N = len(s.seat_status)
    if bool(s.done) or is_chance_node_np(s):
        return False
    actor = int(s.action_seat)
    if actor < 0 or actor >= N or actor != int(s.last_aggressor):
        return False
    if s.seat_status[actor] != SEAT_LIVE:
        return False
    if max(0, int(s.current_bet) - int(np.asarray(s.committed_street)[actor])) != 0:
        return False
    for i in range(N):
        if s.seat_status[i] == SEAT_LIVE and (int(np.asarray(s.committed_street)[i]) < int(s.current_bet)
                                              or not bool(np.asarray(s.acted)[i])):
            return False
    return True


def apply_phantom_check(s):
    full = np.asarray(legal_action_mask_np(s, int(s.action_seat))).astype(bool)
    if not full[CHECK]:
        return None
    before = int(s.hist_len)
    idx = before
    if idx >= MAX_HISTORY:
        return None
    out = apply_action_np(s, CHECK)
    if int(out.hist_len) != idx + 1:
        return None
    hs = np.asarray(out.hist_seat).copy(); hs[idx] = -1
    ha = np.asarray(out.hist_action).copy(); ha[idx] = -1
    hm = np.asarray(out.hist_amount).copy(); hm[idx] = 0
    return out._replace(hist_seat=hs, hist_action=ha, hist_amount=hm, hist_len=np.int32(before))


_PUB = ["street", "pot_dead", "current_bet", "min_raise", "button", "action_seat", "last_aggressor",
        "last_raise_full", "done", "pending_deal", "hist_len", "seat_status", "stacks",
        "committed_street", "committed_total", "acted"]


def public_eq(a, b):
    bad = []
    for f in _PUB:
        if not np.array_equal(np.asarray(getattr(a, f)), np.asarray(getattr(b, f))):
            bad.append(f)
    for f in ("board",):
        if not np.array_equal(np.asarray(getattr(a, f)), np.asarray(getattr(b, f))):
            bad.append(f)
    return bad


def replay_recorded(root, ante, total_convention, phantom, trace=False):
    hist_len = int(root.hist_len)
    s = rebuild_hand_start(root, ante)
    if s is None:
        return None, "rebuild_hand_start=None"
    for t in range(hist_len):
        g = 0
        while True:
            while is_chance_node_np(s):
                s = sample_chance_np(s)
                g += 1
                if g > 12:
                    return None, f"chance-guard t={t}"
            if not (phantom and phantom_check_applicable(s)):
                break
            s = apply_phantom_check(s)
            if s is None:
                return None, f"phantom-check-failed t={t}"
            g += 1
            if g > 12:
                return None, f"phantom-guard t={t}"
        if bool(s.done):
            return None, f"done-before-root t={t}"
        seat = int(root.hist_seat[t]); action = int(root.hist_action[t])
        if seat < 0 or seat >= len(s.seat_status) or not (0 <= action < NUM_ACTIONS):
            return None, f"bad-histentry t={t}"
        if int(s.action_seat) != seat:
            return None, f"actor-mismatch t={t} expect={seat} got={int(s.action_seat)} act={action}"
        s = apply_action_recorded(s, action, int(root.hist_amount[t]), total_convention)
        if s is None:
            return None, f"apply_recorded-rejected t={t} act={action} amt={int(root.hist_amount[t])}"
        if int(s.hist_len) != t + 1:
            return None, f"histlen-mismatch t={t}"
    g = 0
    while True:
        while is_chance_node_np(s):
            s = sample_chance_np(s)
            g += 1
            if g > 12:
                return None, "tail-chance-guard"
        if not (phantom and int(s.street) < int(root.street) and phantom_check_applicable(s)):
            break
        s = apply_phantom_check(s)
        if s is None:
            return None, "tail-phantom-failed"
        g += 1
        if g > 12:
            return None, "tail-phantom-guard"
    bad = public_eq(s, root)
    if bad:
        return None, f"public_eq-mismatch:{bad}"
    return s, "OK"


def replay_public_states(root, trace=False):
    N = len(root.seat_status)
    # derived ante identity
    status0 = np.array([SEAT_LIVE if root.seat_status[i] != SEAT_EMPTY else SEAT_EMPTY for i in range(N)])
    stacks0 = np.array([int(root.stacks[i]) + int(root.committed_total[i]) for i in range(N)])
    sb = _nth_live_after(status0, int(root.button), 1)
    bb = _nth_live_after(status0, int(root.button), 2)
    blind = np.zeros(N, dtype=np.int64)
    blind[sb] = min(SB_CHIPS, max(0, int(stacks0[sb])))
    blind[bb] = min(BB_CHIPS, max(0, int(stacks0[bb])))
    hist_commit = np.zeros(N, dtype=np.int64)
    for t in range(int(root.hist_len)):
        hist_commit[int(root.hist_seat[t])] += max(0, int(root.hist_amount[t]))
    ante = np.zeros(N, dtype=np.int32)
    derived_ok = True
    any_nonzero = False
    for i in range(N):
        a = int(root.committed_total[i]) - int(blind[i]) - int(hist_commit[i])
        if a < 0:
            derived_ok = False
            break
        if a > 0:
            any_nonzero = True
        ante[i] = a
    reasons = []
    for total_convention in (True, False):
        for phantom in (False, True):
            if derived_ok:
                r, why = replay_recorded(root, ante, total_convention, phantom)
                reasons.append(f"tc={total_convention} ph={phantom} ante=derived: {why}")
                if r is not None:
                    return r, reasons
            if (not derived_ok) or any_nonzero:
                r, why = replay_recorded(root, np.zeros(N, dtype=np.int32), total_convention, phantom)
                reasons.append(f"tc={total_convention} ph={phantom} ante=zero: {why}")
                if r is not None:
                    return r, reasons
    return None, reasons
