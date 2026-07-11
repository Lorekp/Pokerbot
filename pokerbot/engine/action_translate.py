"""SERVE-side OFF-MENU ACTION TRANSLATION for blueprint KEY construction — Python twin of
``tools/battery_native/src/translate.rs`` (POKERBOT_ACTION_TRANSLATE=1, default OFF =>
byte-identical everywhere).

WHY (2026-07-11 serve-coverage root-cause): a fullhist/presum-keyed blueprint bakes the exact
abstracted action-id sequence into every lookup key. Any OPPONENT action outside the TRAINED menu
(the trainer's ``abstracted_legal_menus`` thinning: POKERBOT_PRE_KEEP / POKERBOT_PRE_RR_KEEP /
M-RR keep-lists) lands the history — and therefore EVERY subsequent key of the hand — outside the
trained key space, degrading serve to uniform-over-menu for the rest of the hand. Pluribus/Libratus
answered this with ACTION TRANSLATION (pseudo-harmonic mapping, Ganzfried & Sandholm 2013): map
each observed off-menu action to the nearest in-menu size BEFORE key construction.

WHAT: ``translate_state_for_key(state)`` replays the recorded hand history from a reconstructed
root through the RAW (non-closure-wrapped) ``np_engine._apply_action_np_impl`` — closure-injected
CHECKs are already RECORDED in ``hist_action``, the d33ba74 presum-boundary convention —
substituting every recorded action that is OFF the trained menu at its replayed decision point with
the pseudo-harmonic-nearest trained-menu action, and returns the fully replayed TRANSLATED state
(betting scalars, presum-boundary inputs, and the 6-byte S-segment all come from the translated
world, so the key lands in trained key space). Board/hole cards are reproduced exactly by replaying
with the REAL deck.

DETERMINISM / PARITY: sizes are compared as raise-to totals in exact chips; the pseudo-harmonic
boundary f_{A,B}(x) >= 1/2 (f = ((B-x)(P+A)) / ((B-A)(P+x)), P = sizing pot) is evaluated in exact
integer arithmetic — the Rust and Python twins can never diverge on float rounding. Every decision
(candidate set, clamping, fallbacks, fail-open divergence checks) mirrors translate.rs line for
line; parity is gated by scripts/probe_fullhist_key_parity.py --translate.

FAIL-OPEN: any replay divergence returns the ORIGINAL state untouched — exactly the
pre-translation behavior. In-menu histories return the ORIGINAL state object (byte-identity by
construction when there is nothing to translate).
"""
from __future__ import annotations

import os
from typing import Any

import numpy as np

from pokerbot.engine.jax_engine import (
    ALL_IN,
    BET_BUCKET_START,
    CALL,
    CHECK,
    CHIP_UNIT,
    MAX_HISTORY,
    NUM_ACTIONS,
)

_ENABLED = os.environ.get("POKERBOT_ACTION_TRANSLATE", "") == "1"

# Trainer keep-list twins (same env vars + compiled defaults as trainer.rs ctor sites /
# translate.rs::menu_cfg). Parsed once at import — set the env BEFORE importing pokerbot.
def _env_keep(name: str, default: tuple[int, ...] | None) -> tuple[int, ...] | None:
    raw = os.environ.get(name, "")
    ids = tuple(int(tok) for tok in (t.strip() for t in raw.split(",")) if tok)
    return ids if ids else default


def _allowed_mask(keep: tuple[int, ...]) -> np.ndarray:
    m = np.zeros(NUM_ACTIONS, dtype=bool)
    m[0] = m[CHECK] = m[CALL] = m[ALL_IN] = True  # F/X/C/AI in every keep-list
    for i in keep:
        a = BET_BUCKET_START + i
        if a < NUM_ACTIONS:
            m[a] = True
    return m


_MRR = os.environ.get("POKERBOT_MRR", "")
_MRR_ON = _MRR == "1" or _MRR.lower() == "true"
_PRE_MASK = _allowed_mask(_env_keep("POKERBOT_PRE_KEEP", (4, 5, 6)))
_POST_MASK = _allowed_mask(_env_keep("POKERBOT_POST_KEEP", (0, 1, 3, 4, 5)))
_PRE_RR = _env_keep("POKERBOT_PRE_RR_KEEP", None)
_PRE_RR_MASK = _allowed_mask(_PRE_RR) if _PRE_RR is not None else None
_FLOP_RR = _env_keep("POKERBOT_FLOP_RR_KEEP", None)
_FLOP_RR_MASK = _allowed_mask(_FLOP_RR) if _FLOP_RR is not None else None
_MRR_RERAISE_MASK = _allowed_mask((4,))
_MRR_TR_FIRST_MASK = _allowed_mask((1, 4))


def enabled() -> bool:
    return _ENABLED


def trained_legal(state, seat: int) -> np.ndarray:
    """TRAINED-MENU legal mask at a (replayed) state — twin of translate.rs::trained_legal /
    trainer.rs::abstracted_legal_menus node classification."""
    from pokerbot.engine.np_engine import legal_action_mask_np

    full = legal_action_mask_np(state, seat).astype(bool)
    street = int(state.street)
    if street == 0:
        if _PRE_RR_MASK is not None and int(state.current_bet) > int(CHIP_UNIT):
            allowed = _PRE_RR_MASK
        else:
            allowed = _PRE_MASK
    elif _MRR_ON:
        facing_bet = int(state.current_bet) > int(state.committed_street[seat])
        if facing_bet:
            if street == 1 and _FLOP_RR_MASK is not None:
                allowed = _FLOP_RR_MASK
            else:
                allowed = _MRR_RERAISE_MASK
        elif street >= 2:
            allowed = _MRR_TR_FIRST_MASK
        else:
            allowed = _POST_MASK
    else:
        allowed = _POST_MASK
    return full & allowed


def _raise_to_total(state, seat: int, action: int) -> int:
    """Raise-to total (seat's committed_street AFTER the action) — the exact-chip size scalar."""
    from pokerbot.engine.np_engine import _action_amount

    return int(state.committed_street[seat]) + int(_action_amount(state, seat, action))


def _pseudo_harmonic_pick_lower(a_tot: int, b_tot: int, x_tot: int, pot: int) -> bool:
    """Map to the LOWER bracket A iff f_{A,B}(x) >= 1/2 — exact integer form (translate.rs twin)."""
    return 2 * (b_tot - x_tot) * (pot + a_tot) >= (b_tot - a_tot) * (pot + x_tot)


def _translate_bet(state, seat: int, action: int, menu: np.ndarray) -> int | None:
    from pokerbot.engine.np_engine import _action_amount, _pot_for_sizing

    pot = max(1, int(_pot_for_sizing(state)))
    x_tot = _raise_to_total(state, seat, action)
    cand: list[tuple[int, int]] = []
    stack = int(state.stacks[seat])
    for b in range(BET_BUCKET_START, ALL_IN):
        if menu[b] and int(_action_amount(state, seat, b)) < stack:
            cand.append((b, _raise_to_total(state, seat, b)))
    if menu[ALL_IN]:
        cand.append((ALL_IN, _raise_to_total(state, seat, ALL_IN)))
    if not cand:
        return None
    if x_tot <= cand[0][1]:
        return cand[0][0]
    if x_tot >= cand[-1][1]:
        return cand[-1][0]
    hi = next(i for i, (_, t) in enumerate(cand) if t >= x_tot)
    b_act, b_tot = cand[hi]
    a_act, a_tot = cand[hi - 1]
    if a_tot == b_tot:
        return a_act
    return a_act if _pseudo_harmonic_pick_lower(a_tot, b_tot, x_tot, pot) else b_act


def _root_with_deck(button: int, start_stacks, occupancy, rng_key, deck: np.ndarray):
    """``jax_weva_fast._reconstruct_root_np`` with the REAL deck (so replayed chance deals
    reproduce the exact hole/board cards) — otherwise field-for-field the same twin of
    ``core::init_hand_n``."""
    from pokerbot.engine.jax_engine import MAX_BOARD, STREET_PREFLOP
    from pokerbot.engine.np_engine import EnvState as _NPEnvState

    n_seats = len(start_stacks)
    SEAT_EMPTY_, SEAT_LIVE_, SEAT_ALLIN_ = 0, 2, 3
    SB_CHIPS = CHIP_UNIT // 2
    BB_CHIPS = CHIP_UNIT

    seat_status = np.array(
        [SEAT_LIVE_ if occupancy[i] else SEAT_EMPTY_ for i in range(n_seats)], dtype=np.int8)
    stacks = np.array([int(start_stacks[i]) for i in range(n_seats)], dtype=np.int32)

    def _first_live_after(from_seat: int) -> int:
        for off in range(1, n_seats + 1):
            seat = (from_seat + off) % n_seats
            if seat_status[seat] == SEAT_LIVE_:
                return seat
        return -1

    def _nth_live_after(from_seat: int, k: int) -> int:
        seat = from_seat
        for _ in range(k):
            seat = _first_live_after(seat)
        return seat

    sb = _nth_live_after(button, 1)
    bb = _nth_live_after(button, 2)
    committed = np.zeros(n_seats, dtype=np.int32)
    sb_live = 1 if seat_status[sb] != SEAT_EMPTY_ else 0
    bb_live = 1 if seat_status[bb] != SEAT_EMPTY_ else 0
    sb_paid = min(SB_CHIPS * sb_live, max(int(stacks[sb]), 0))
    bb_paid = min(BB_CHIPS * bb_live, max(int(stacks[bb]), 0))
    stacks[sb] -= sb_paid
    stacks[bb] -= bb_paid
    committed[sb] = sb_paid
    committed[bb] = bb_paid
    committed_total = committed.copy()
    for i in range(n_seats):
        if seat_status[i] != SEAT_EMPTY_ and int(stacks[i]) == 0:
            seat_status[i] = SEAT_ALLIN_
    current_bet = int(committed.max()) if n_seats else 0
    min_raise = BB_CHIPS

    deck = np.asarray(deck, dtype=np.int32).copy()
    holes = np.full((n_seats, 2), -1, dtype=np.int32)
    p = 0
    for i in range(n_seats):
        if seat_status[i] != SEAT_EMPTY_:
            holes[i, 0] = deck[p]
            holes[i, 1] = deck[p + 1]
            p += 2
    n_live = int(np.sum((seat_status == SEAT_LIVE_) | (seat_status == SEAT_ALLIN_)))
    btn_first = _nth_live_after(button, 1)
    utg = _nth_live_after(button, 3)
    action_seat = btn_first if n_live <= 2 else utg

    return _NPEnvState(
        holes=holes,
        board=np.full(MAX_BOARD, -1, dtype=np.int32),
        seat_status=seat_status,
        stacks=stacks,
        committed_street=committed,
        committed_total=committed_total,
        street=np.asarray(STREET_PREFLOP, dtype=np.int32),
        pot_dead=np.asarray(0, dtype=np.int32),
        current_bet=np.asarray(current_bet, dtype=np.int32),
        min_raise=np.asarray(min_raise, dtype=np.int32),
        button=np.asarray(button, dtype=np.int32),
        action_seat=np.asarray(action_seat, dtype=np.int32),
        last_aggressor=np.asarray(-1, dtype=np.int32),
        acted=np.zeros(n_seats, dtype=np.bool_),
        last_raise_full=np.asarray(True, dtype=np.bool_),
        deck=deck,
        deck_ptr=np.asarray(p, dtype=np.int32),
        rng_key=np.asarray(rng_key, dtype=np.uint32),
        hist_seat=np.full(MAX_HISTORY, -1, dtype=np.int32),
        hist_action=np.full(MAX_HISTORY, -1, dtype=np.int32),
        hist_amount=np.zeros(MAX_HISTORY, dtype=np.int32),
        hist_len=np.asarray(0, dtype=np.int32),
        done=np.asarray(False, dtype=np.bool_),
        rewards=np.zeros(n_seats, dtype=np.int32),
        pending_deal=np.asarray(False, dtype=np.bool_),
    )


def translate_state_for_key(state: Any):
    """Replay-translate the state's recorded history into the trained action abstraction; return
    the translated state for key construction, or the ORIGINAL state on identity/fail-open."""
    t = _translate_impl(state)
    return state if t is None else t


def _translate_impl(state: Any):
    from pokerbot.engine import np_engine as _np

    hist_len = max(0, min(int(np.asarray(state.hist_len)), MAX_HISTORY))
    if hist_len == 0:
        return None
    stacks = np.asarray(state.stacks)
    committed_total = np.asarray(state.committed_total)
    seat_status = np.asarray(state.seat_status)
    n_seats = int(stacks.shape[0])
    start_stacks = [int(stacks[i]) + int(committed_total[i]) for i in range(n_seats)]
    occupancy = [int(seat_status[i]) != 0 for i in range(n_seats)]
    r = _root_with_deck(int(np.asarray(state.button)), start_stacks, occupancy,
                        np.asarray(state.rng_key), np.asarray(state.deck))
    hist_seat = np.asarray(state.hist_seat)
    hist_action = np.asarray(state.hist_action)
    changed = False
    injected = 0
    close_on = _np._close_called_street_enabled()
    # Recorded seats are a CONSISTENCY SIGNAL, not a replay driver: the production GameState engine
    # (pool evals / live bridge, engine/game.py play_hand) advances past ``_first_postflop_actor``
    # unconditionally, so its postflop action order starts one live seat LATE (BB-first) vs the
    # trainer core's real-NLHE SB-first order — recorded seats cannot be matched against the trainer
    # engine step-for-step. The replay therefore follows the TRAINER engine's own actor rotation and
    # replays the recorded ACTION BYTES (the only history component the key hashes);
    # ``seats_consistent`` (np-engine self-play streams, which come from this very engine) selects
    # the exact recorded-closure-CHECK consumption below. Twin of translate.rs.
    seats_consistent = True
    for i in range(hist_len):
        while not bool(r.done) and _np.is_chance_node_np(r):
            r = _np.sample_chance_np(r)
        if bool(r.done) or int(r.action_seat) < 0:
            return None  # replay diverged — fail open
        seats_consistent = seats_consistent and int(r.action_seat) == int(hist_seat[i])
        recorded = int(hist_action[i])
        if not (0 <= recorded < NUM_ACTIONS):
            return None
        a = recorded
        seat = int(r.action_seat)
        menu = trained_legal(r, seat)
        if not menu[a]:
            if BET_BUCKET_START <= a <= ALL_IN:
                t = _translate_bet(r, seat, a, menu)
                if t is None:
                    if menu[CALL]:
                        a = CALL
                    elif menu[CHECK]:
                        a = CHECK
                    else:
                        return None
                else:
                    a = t
                changed = changed or a != recorded
            else:
                # F/X/C/AI are in every keep-list => an off-menu non-bet means genuine divergence.
                return None
        changed = changed or not seats_consistent
        # RAW (non-closure) step — recorded closure CHECKs are replayed as recorded tokens.
        r = _np._apply_action_np_impl(r, a)
        # CLOSURE-CHECK SYNTHESIS (train/serve parity, 2026-07-11 root-cause; twin of
        # translate.rs): under POKERBOT_CLOSE_CALLED_STREET=1 the trainer RECORDS the
        # closure-injected closing CHECK in hist_action, baking it into every trained postflop key.
        # Histories bridged from real tables / the pool harness (state_bridge.game_state_to_env
        # copies GameState.action_history verbatim) never carry that token — real NLHE closes the
        # street on the call — so every called bet/raise line missed on every later street. When
        # the replayed child is the aggressor-reopen node (closure predicate): np-engine self-play
        # streams (seats consistent so far) carry the recorded closing CHECK as their next token —
        # consume it naturally; any other stream gets the CHECK synthesized exactly like the
        # trainer did.
        if close_on:
            ag = int(r.action_seat)
            reopen = (not bool(r.done) and not bool(r.pending_deal) and int(r.street) < 4
                      and ag >= 0 and ag == int(r.last_aggressor)
                      and int(r.current_bet) == int(r.committed_street[ag]))
            if reopen:
                consume_recorded = (seats_consistent and i + 1 < hist_len
                                    and int(hist_seat[i + 1]) == ag
                                    and int(hist_action[i + 1]) == CHECK)
                if not consume_recorded:
                    r = _np._apply_action_np_impl(r, CHECK)
                    injected += 1
                    changed = True
    if not changed:
        return None  # in-menu history: the input state IS the translated state (byte-identity)
    while not bool(r.done) and _np.is_chance_node_np(r):
        r = _np.sample_chance_np(r)
    # Same street/decision structure AND the same acting seat as the real state — the key's
    # S-segment (to_call, facing_agg) and hole/weva bytes are hero-relative, so a replay whose
    # actor rotation diverged from the recorded game (e.g. a legacy BB-first-postflop GameState
    # history, engine/game.py pre-POSTFLOP_SB_FIRST) would key a DIFFERENT seat's decision — fail
    # open instead. Twin of translate.rs.
    if bool(r.done) or int(r.street) != int(np.asarray(state.street)) \
            or int(r.hist_len) != hist_len + injected \
            or int(r.action_seat) != int(np.asarray(state.action_seat)):
        return None
    # CARD FIELDS come from the ORIGINAL state: a live/pool-bridged state's ``deck`` is a canonical
    # placeholder (state_bridge fills [0..52)), so the replay's dealt cards are bookkeeping only.
    # The betting replay never reads card VALUES; the key's card segments (postflop bucket, weva)
    # must read the REAL observed cards.
    return r._replace(
        holes=np.asarray(state.holes, dtype=np.int32).copy(),
        board=np.asarray(state.board, dtype=np.int32).copy(),
        deck=np.asarray(state.deck, dtype=np.int32).copy(),
        deck_ptr=np.asarray(int(state.deck_ptr), dtype=np.int32),
    )
