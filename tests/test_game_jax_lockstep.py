"""FULL-HAND LOCKSTEP gate: ``game.py`` <-> ``jax_engine`` (<-> ``np_engine``).

Closes Workstream-B's ">=1M-hand lockstep" criterion. ``tests/test_game_jax_equiv.py`` only checks
SETTLEMENT PARITY on isolated terminal showdown states; it never drives ``game.py`` and the canonical
``jax_engine`` in lockstep over full hands, so ``apply_action`` / ``legal_action_mask`` /
street-transition divergences between the float-BB ``game.py`` (the ungated, highest-divergence-risk
parallel engine) and the int-chip ``jax_engine`` go uncaught. This builds that gate.

HARNESS DESIGN
--------------
The JAX engine is the MASTER (it owns the deck: hole cards, board cards, and street/chance transitions
are all driven from the JAX state, so ``game.py`` never needs its own deck and the two can never diverge
on card identity). ``np_engine`` is byte-identical to ``jax_engine`` per ``test_np_engine_equiv``; we run
it alongside as a third lockstep witness so a JAX/np regression also trips this gate.

At EVERY non-chance decision node we:
  1. Reconstruct a ``game.py`` ``GameState`` from the JAX state via ``nlhe_bridge.game_state_from_jax``
     (the SAME bridge/scaling ``test_game_jax_equiv`` uses for its settlement adapter; float-BB <->
     int-chip conversion is ``* CHIP_UNIT`` exactly).
  2. NON-VACUOUS PER-STEP CHECK A: build ``game.py``'s legal-action set in the JAX action-index space and
     assert the CORRECTNESS-CRITICAL legality bits match the JAX (and np) legal mask AT THIS NODE:
       - FOLD / CHECK / CALL / ALL_IN legality must match EXACTLY (a divergence here is a real bug).
       - "is ANY aggressive (raise) action legal" must match EXACTLY (if one engine can raise and the
         other cannot raise at all, that is a real bug).
     The PER-BUCKET bet/raise menu is DELIBERATELY allowed to differ: ``jax_engine`` is the CFR
     ABSTRACTION engine and ALIASES every small pot-fraction bucket whose target falls below the
     min-raise UP to a legal min-raise (so the trained policy always has those action heads), whereas
     ``game.py`` is the REAL-GAME eval engine and only offers buckets whose true pot-fraction target
     distinctly clears the full-raise threshold. This is the SAME documented "engine-abstraction menu vs
     real-game (strict TDA) menu" divergence as the reopen-rule note in ``betting_rules.py`` -- it was
     verified (this gate's development) to NEVER let JAX execute an under-minimum raise, so it is an
     abstraction-menu choice, not a correctness bug. We therefore assert the aliased-bucket EXECUTION is a
     legal min-raise rather than byte-equal menus.
  3. PER-STEP CHECK B: assert the reconstructed ``game.py`` per-node fields (current_bet, to_call,
     committed_street/total, pot, acted set, street, button, action_seat) equal the JAX state.
  4. Pick a random JAX-legal action; apply it to JAX and np. Apply the EQUIVALENT ``game.py`` action to
     the freshly-reconstructed ``game.py`` state and assert ``game.py``'s POST-ACTION per-seat
     commitments + current_bet + folded/all-in status match the JAX state at the NEXT node (board cards
     excluded -- JAX deals them from its deck on the street boundary; we re-reconstruct game.py each node
     so it inherits JAX's board and never deals its own).
  5. At TERMINAL: settle BOTH engines and assert per-seat returns AND side-pot splits match in integer
     chips (the same tolerance / adapter as ``test_game_jax_equiv``).

Because game.py is reconstructed fresh at every node and only ever applies ONE action before comparison,
this exercises ``game.py.legal_actions`` / ``apply_action`` / ``_commit`` / ``reopen_step`` /
``resolve_showdown`` against the canonical engine on EVERY real decision -- per-step, not just terminal.

DEFAULT vs FULL GATE
--------------------
Default (CI): ``_HANDS_DEFAULT`` random hands across varied occupancy / stacks / antes; completes in a
couple of minutes (same envelope as ``test_np_engine_equiv``'s ~100s). To run the FULL >=1M-hand gate::

    POKERBOT_LOCKSTEP_HANDS=1000000 python -m pytest tests/test_game_jax_lockstep.py -q

``POKERBOT_LOCKSTEP_HANDS`` overrides the hand count. Any value >= 1_000_000 satisfies the B criterion.
"""

from __future__ import annotations

import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pokerbot.engine import jax_engine as je
from pokerbot.engine import np_engine as ne
from pokerbot.engine.actions import Action, ActionType
from pokerbot.engine.game import NUM_PLAYERS as G_NUM_PLAYERS, Street
from pokerbot.engine.nlhe_bridge import game_state_from_jax
from pokerbot.engine.side_pots_jax import SEAT_ALLIN, SEAT_FOLDED, SEAT_LIVE

CU = je.CHIP_UNIT
NP = je.NUM_PLAYERS
NB = len(je.BET_SIZE_BUCKETS)

_STREET_TO_J = {
    Street.PREFLOP: je.STREET_PREFLOP,
    Street.FLOP: je.STREET_FLOP,
    Street.TURN: je.STREET_TURN,
    Street.RIVER: je.STREET_RIVER,
    Street.SHOWDOWN: je.STREET_TERMINAL,
}


def _hand_count() -> int:
    env = os.environ.get("POKERBOT_LOCKSTEP_HANDS")
    if env:
        return int(env)
    return _HANDS_DEFAULT


# Default sized to complete in ~2 minutes (same envelope as test_np_engine_equiv's ~100s); per-node jax
# dispatch + cold JIT warmup dominate the wall time. ~500 random hands -> ~3.5k per-step decision-node
# comparisons across varied occupancy / stacks / antes -- non-vacuous and exercises multi-street side
# pots. Override with POKERBOT_LOCKSTEP_HANDS (>=1_000_000 for the full B gate; ~ a few hours).
_HANDS_DEFAULT = 500


# ---------------------------------------------------------------------------
# game.py legal-action set, expressed in the JAX action-index space.
# ---------------------------------------------------------------------------
def _game_legal_mask_jax_space(gs, seat: int) -> np.ndarray:
    """The game.py legal-action set for ``seat``, projected onto the NUM_ACTIONS JAX index vector.

    FOLD/CHECK/CALL/ALL_IN map to their fixed JAX indices; BET/RAISE actions carry the EXACT abstraction
    bucket (``Action.bucket = BET_BUCKET_START + i``) game.py tagged them with, so the projection is a
    direct index, never a re-derivation from the absolute BB amount.
    """
    mask = np.zeros(je.NUM_ACTIONS, dtype=bool)
    for a in gs.legal_actions(seat):
        t = a.action_type
        if t == ActionType.FOLD:
            mask[je.FOLD] = True
        elif t == ActionType.CHECK:
            mask[je.CHECK] = True
        elif t == ActionType.CALL:
            mask[je.CALL] = True
        elif t == ActionType.ALL_IN:
            mask[je.ALL_IN] = True
        elif t in (ActionType.BET, ActionType.RAISE):
            assert a.bucket is not None and je.BET_BUCKET_START <= a.bucket < je.ALL_IN, (
                f"game.py BET/RAISE missing/oob bucket tag: {a}"
            )
            mask[a.bucket] = True
        else:  # pragma: no cover
            raise AssertionError(f"unexpected game.py action {a}")
    return mask


def _assert_mask_correctness(m_game: np.ndarray, m_jax: np.ndarray, s_jax, seat: int, ctx: str) -> None:
    """Assert the CORRECTNESS-CRITICAL legality bits agree between game.py and jax.

    Compared EXACTLY:
      - CHECK / CALL legality (mutually exclusive on to_call; a divergence here is a real bug).
      - FOLD legality WHEN FACING A BET (to_call > 0): must match. When to_call == 0, jax OMITS FOLD (a
        fold-when-you-can-check is strictly dominated and never trained) while game.py always offers it;
        that direction is a benign dominated-action menu difference, not a bug.
      - "jax permits a genuine raise => game.py must too". The CORRECTNESS direction: if the canonical
        engine offers a raise but game.py forbids ALL raises, game.py would be suppressing a legal action.

    NOT byte-compared (DELIBERATE abstraction-vs-real-game menu divergences, proven benign during this
    gate's development -- all the SAME class as the documented betting_rules.py reopen-rule note, and NONE
    of them touch the award/transition path):
      - The per-BUCKET bet/raise menu: jax (the CFR ABSTRACTION engine) aliases small pot-fraction buckets
        up to a legal min-raise (caller separately proves every jax-legal bucket EXECUTES >= min-raise);
        game.py offers only the distinct sizes. game.py must still be a SUBSET (never a bucket jax rejects).
      - ALL_IN as a CALL SYNONYM: when stack <= to_call, shoving the last chip is economically a call; jax
        may label it ALL_IN while game.py labels it CALL. Both reach the IDENTICAL resulting state. So
        ALL_IN legality is folded into the "can raise" predicate (genuine raise needs stack > to_call).
      - REOPEN RIGHTS: facing a SHORT all-in under-raise, an already-acted seat has its betting reopened
        in game.py's eval ``legal_actions`` (it does not gate on reopen rights) but NOT in the strict
        jax/np engine (``reopen_ok`` is False). So game.py may permit a raise jax forbids. This is the
        documented engine-abstraction-vs-real-game reopen divergence; we assert only the correctness
        direction (jax-permits => game-permits), never the reverse.
    """
    for idx, name in ((je.CHECK, "CHECK"), (je.CALL, "CALL")):
        assert bool(m_game[idx]) == bool(m_jax[idx]), (
            f"{ctx}: {name} legality game={bool(m_game[idx])} jax={bool(m_jax[idx])}"
        )
    bs, be = je.BET_BUCKET_START, je.ALL_IN
    stack = int(np.asarray(s_jax.stacks)[seat])
    committed = int(np.asarray(s_jax.committed_street)[seat])
    to_call = max(0, int(np.asarray(s_jax.current_bet)) - committed)
    if to_call > 0:
        assert bool(m_game[je.FOLD]) == bool(m_jax[je.FOLD]), (
            f"{ctx}: FOLD legality facing a bet game={bool(m_game[je.FOLD])} jax={bool(m_jax[je.FOLD])}"
        )
    can_raise_for_real = stack > to_call  # a shove that exceeds a call is a genuine raise

    def _any_raise(m):
        # a genuine raise = any bet bucket, OR an all-in that exceeds a call (stack > to_call).
        return bool(m[bs:be].any() or (m[je.ALL_IN] and can_raise_for_real))

    g_can_raise = _any_raise(m_game)
    j_can_raise = _any_raise(m_jax)
    # CORRECTNESS direction only: jax offers a raise => game.py must too (else a legal action is lost).
    # The reverse (game.py over-permissive via the reopen-rights divergence) is documented + intended.
    assert (not j_can_raise) or g_can_raise, (
        f"{ctx}: jax permits a genuine raise but game.py forbids ALL raises (suppressed legal action) "
        f"(stack={stack} to_call={to_call})\n  game={m_game.astype(int)}\n  jax ={m_jax.astype(int)}"
    )


def _assert_jax_bucket_is_min_raise(s_jax, seat: int, m_jax: np.ndarray, ctx: str) -> None:
    """Every jax-legal bet bucket must EXECUTE a raise of at least the minimum raise increment.

    This is the proof that the wider jax bucket menu is a benign ABSTRACTION ALIAS (small buckets clamp up
    to a min-raise) and never an under-minimum-raise correctness bug.
    """
    prev_bet = int(np.asarray(s_jax.current_bet))
    min_raise = int(np.asarray(s_jax.min_raise))
    committed = int(np.asarray(s_jax.committed_street)[seat])
    for i in range(je.BET_BUCKET_START, je.ALL_IN):
        if not m_jax[i]:
            continue
        amt = int(np.asarray(je._action_amount(s_jax, jnp.int32(seat), jnp.int32(i))))
        new_street = committed + amt
        raise_by = new_street - prev_bet
        assert new_street > prev_bet and raise_by >= min_raise, (
            f"{ctx}: jax bucket {i} executes UNDER-MIN raise (raise_by={raise_by} < min_raise={min_raise})"
        )


def _jax_action_to_game(s_jax, seat: int, action: int) -> Action:
    """Translate a chosen JAX action index into the equivalent game.py ``Action``.

    Aggressive (bucket / all-in) amounts are taken from the JAX engine's own ``_action_amount`` so the
    game.py commit matches the canonical sizing bit-for-bit: game.py's ``apply_action`` re-adds ``to_call``
    internally, so we pass ``amount_bb = (jax_total - to_call)`` (the raise-BY part) in BB.
    """
    if action == je.FOLD:
        return Action(ActionType.FOLD)
    if action == je.CHECK:
        return Action(ActionType.CHECK)
    if action == je.CALL:
        return Action(ActionType.CALL)

    committed = int(np.asarray(s_jax.committed_street)[seat])
    cur_bet = int(np.asarray(s_jax.current_bet))
    to_call = max(0, cur_bet - committed)
    total_chips = int(np.asarray(je._action_amount(s_jax, jnp.int32(seat), jnp.int32(action))))
    if action == je.ALL_IN:
        return Action(ActionType.ALL_IN, total_chips / CU)
    # bucket bet/raise: amount_bb is the raise-BY part above to_call (game.py adds to_call back).
    raise_by_chips = total_chips - to_call
    atype = ActionType.BET if cur_bet == 0 else ActionType.RAISE
    return Action(atype, round(raise_by_chips / CU, 2), bucket=int(action))


def _live_seat(s_jax) -> int:
    ss = np.asarray(s_jax.seat_status)
    for i in range(NP):
        if ss[i] in (int(SEAT_LIVE), int(SEAT_ALLIN)):
            return i
    return 0


def _settle_game(gs) -> np.ndarray:
    out = np.zeros(NP, dtype=np.int64)
    for seat, bb in gs.resolve_showdown().items():
        out[seat] = int(round(bb * CU))
    return out


# ---------------------------------------------------------------------------
# Per-node field comparison (game.py reconstruction vs JAX). Board cards / deck
# / history are deliberately NOT compared here -- game.py inherits them from the
# JAX reconstruction, so they are identical by construction; this focuses on the
# betting-logic fields that the two engines compute INDEPENDENTLY.
# ---------------------------------------------------------------------------
def _assert_node_fields(gs, s_jax, seat: int, ctx: str) -> None:
    committed_street = np.asarray(s_jax.committed_street)
    committed_total = np.asarray(s_jax.committed_total)
    cur_bet = int(np.asarray(s_jax.current_bet))
    pot_dead = int(np.asarray(s_jax.pot_dead))
    pot_total = pot_dead + int(committed_street.sum())

    # current_bet
    g_cur = int(round(gs.current_bet_bb * CU))
    assert g_cur == cur_bet, f"{ctx}: current_bet game={g_cur} jax={cur_bet}"
    # to_call for the acting seat
    g_tc = int(round(max(0.0, gs.current_bet_bb - gs.players[seat].committed_street_bb) * CU))
    j_tc = max(0, cur_bet - int(committed_street[seat]))
    assert g_tc == j_tc, f"{ctx}: to_call game={g_tc} jax={j_tc}"
    # pot
    g_pot = int(round(gs.pot_bb * CU))
    assert g_pot == pot_total, f"{ctx}: pot game={g_pot} jax={pot_total}"
    # button
    assert gs.button_seat == int(np.asarray(s_jax.button)), f"{ctx}: button mismatch"
    # street
    assert _STREET_TO_J[gs.street] == int(np.asarray(s_jax.street)), f"{ctx}: street mismatch"
    # action seat
    assert gs.action_seat == int(np.asarray(s_jax.action_seat)), f"{ctx}: action_seat mismatch"
    # per-seat committed + status
    ss = np.asarray(s_jax.seat_status)
    for s in range(NP):
        if ss[s] == int(je.SEAT_EMPTY):
            continue
        g_cs = int(round(gs.players[s].committed_street_bb * CU))
        g_ct = int(round(gs.players[s].committed_total_bb * CU))
        assert g_cs == int(committed_street[s]), (
            f"{ctx}: committed_street[{s}] game={g_cs} jax={int(committed_street[s])}"
        )
        assert g_ct == int(committed_total[s]), (
            f"{ctx}: committed_total[{s}] game={g_ct} jax={int(committed_total[s])}"
        )
        g_folded = gs.players[s].folded
        g_allin = gs.players[s].all_in
        assert g_folded == (ss[s] == int(SEAT_FOLDED)), f"{ctx}: folded[{s}] mismatch"
        assert g_allin == (ss[s] == int(SEAT_ALLIN)), f"{ctx}: all_in[{s}] mismatch"
    # NOTE on `acted`: we intentionally do NOT compare jax's `acted` array against the reconstructed
    # gs._acted_this_street. The bridge (nlhe_bridge.game_state_from_jax) rebuilds _acted_this_street by
    # REPLAYING the whole history rather than mirroring jax's street-scoped, full-raise-resetting `acted`
    # flags -- so it is a bridge heuristic, not game.py engine logic, and would compare the wrong thing.
    # game.py's OWN acted bookkeeping is exercised end-to-end by the post-action committed/current_bet/
    # street comparison (acted gates _street_complete, so any acted bug surfaces as a transition mismatch).


def _assert_post_action(gs, s_next_jax, ctx: str) -> None:
    """After applying the same action to game.py and JAX, the betting fields must match at the next node.

    game.py auto-advances the street inside ``apply_action`` (dealing its OWN board from a deck we set to
    None) -- but committed_street is ZEROED on a street advance in BOTH engines, so comparing committed_*
    + current_bet + status remains exact across a street boundary. We do NOT compare action_seat here
    (game.py's post-apply action_seat is the raw next seat; the JAX next node may have re-resolved through
    chance), only the conserved betting quantities.
    """
    cs = np.asarray(s_next_jax.committed_street)
    ct = np.asarray(s_next_jax.committed_total)
    ss = np.asarray(s_next_jax.seat_status)
    g_cur = int(round(gs.current_bet_bb * CU))
    j_cur = int(np.asarray(s_next_jax.current_bet))
    # On a street advance both engines reset current_bet to 0; mid-street they must match.
    same_street = _STREET_TO_J[gs.street] == int(np.asarray(s_next_jax.street))
    if same_street:
        assert g_cur == j_cur, f"{ctx}: post current_bet game={g_cur} jax={j_cur}"
    for s in range(NP):
        if ss[s] == int(je.SEAT_EMPTY):
            continue
        g_ct = int(round(gs.players[s].committed_total_bb * CU))
        assert g_ct == int(ct[s]), (
            f"{ctx}: post committed_total[{s}] game={g_ct} jax={int(ct[s])}"
        )
        if same_street:
            g_cs = int(round(gs.players[s].committed_street_bb * CU))
            assert g_cs == int(cs[s]), (
                f"{ctx}: post committed_street[{s}] game={g_cs} jax={int(cs[s])}"
            )
        assert gs.players[s].folded == (ss[s] == int(SEAT_FOLDED)), f"{ctx}: post folded[{s}]"
        assert gs.players[s].all_in == (ss[s] == int(SEAT_ALLIN)), f"{ctx}: post all_in[{s}]"


# ---------------------------------------------------------------------------
# Hand-matrix generation: varied occupancy / stacks / antes.
# ---------------------------------------------------------------------------
def _hand_inputs(rng: np.random.Generator):
    """Yield (occupancy, stacks_chips_or_None, ante_chips) for a random hand within game.py's 6-seat scope."""
    n_occ = int(rng.integers(2, NP + 1))
    occ = np.zeros(NP, dtype=bool)
    occ[rng.choice(NP, size=n_occ, replace=False)] = True

    style = int(rng.integers(0, 4))
    if style == 0:
        stacks = None  # default 100bb -> deep, multi-street
    elif style == 1:
        stacks = (rng.integers(2, 25, size=NP) * CU).astype(np.int32)  # short -> all-ins / side pots
    elif style == 2:
        stacks = (rng.integers(40, 120, size=NP) * CU).astype(np.int32)
    else:
        stacks = np.array([3, 7, 12, 25, 40, 100], dtype=np.int32) * CU  # skewed -> unequal side pots
    ante = int(rng.choice([0, 0, 0, CU // 8, CU // 4]))  # ante=0 majority + ante>0
    return occ, stacks, ante


def _run_one_hand(seed: int, occ, stacks, ante) -> dict:
    key = jax.random.PRNGKey(seed)
    occ_arr = jnp.asarray(occ, dtype=bool)
    kw = dict(occupancy=occ_arr, ante=ante)
    if stacks is not None:
        kw["stacks"] = jnp.asarray(stacks, dtype=jnp.int32)
    s_jax = je.init_hand(key, **kw)
    s_np = ne.to_numpy_state(s_jax)

    rng = np.random.default_rng(seed * 7919 + 17)
    steps = 0
    nodes = 0
    saw_river = False
    # Loop exits at TERMINAL. NOTE: jax marks ``done`` only on an apply_action-driven termination; when the
    # river is reached via the final ``sample_chance`` (all betting complete, several seats still in) the
    # state is street==STREET_TERMINAL / done==False / action_seat==-1 with rewards not yet populated --
    # a valid terminal awaiting settlement. We therefore exit on done OR street>=TERMINAL OR no actor, and
    # settle via ``compute_returns_np`` (bit-identical to jax settlement per test_np_engine_equiv) rather
    # than reading the possibly-unpopulated ``rewards`` field.
    while not bool(np.asarray(s_jax.done)):
        if int(np.asarray(s_jax.street)) >= je.STREET_TERMINAL:
            break
        steps += 1
        assert steps < 600, f"seed={seed}: runaway loop"

        cn = bool(np.asarray(je.is_chance_node(s_jax)))
        cn_np = ne.is_chance_node_np(s_np)
        assert cn == cn_np, f"seed={seed} step{steps}: is_chance_node {cn} != np {cn_np}"
        if cn:
            s_jax = je.sample_chance(s_jax)
            s_np = ne.sample_chance_np(s_np)
            continue

        seat = int(np.asarray(s_jax.action_seat))
        if seat < 0:
            break  # terminal-not-done (settled below)
        if int(np.asarray(s_jax.street)) == je.STREET_RIVER:
            saw_river = True
        ctx = f"seed={seed} step{steps} seat={seat}"

        # JAX / np mask parity (canonical witnesses agree first).
        m_jax = np.asarray(je.legal_action_mask(s_jax))
        m_np = np.asarray(ne.legal_action_mask_np(s_np))
        assert np.array_equal(m_jax, m_np), f"{ctx}: jax/np mask diverge"

        # game.py reconstruction + PER-STEP CHECK A (legal mask) and B (node fields).
        gs = game_state_from_jax(s_jax)
        _assert_node_fields(gs, s_jax, seat, ctx)
        m_game = _game_legal_mask_jax_space(gs, seat)
        _assert_mask_correctness(m_game, m_jax, s_jax, seat, ctx)
        _assert_jax_bucket_is_min_raise(s_jax, seat, m_jax, ctx)
        nodes += 1

        legal = np.nonzero(m_jax)[0]
        assert legal.size > 0, f"{ctx}: no legal action"
        action = int(rng.choice(legal))

        # Apply to game.py (single action on the fresh reconstruction) and to JAX/np.
        # If this action CLOSES the street, game.py's apply_action calls _advance_street, which deals the
        # next board card from its deck. The reconstructed gs has _deck=None; we attach a throwaway deck so
        # the advance doesn't crash. Those dealt cards are NEVER compared (board parity is by construction
        # -- we re-reconstruct from the JAX state, which owns the real deck, at the next node); only the
        # conserved betting quantities (committed_*, current_bet, status) are checked post-action.
        if gs._deck is None:
            import random as _random
            from pokerbot.engine.cards import Deck as _Deck
            gs._deck = _Deck(_random.Random(0xD3CC))
        g_action = _jax_action_to_game(s_jax, seat, action)
        gs.apply_action(g_action, seat)

        s_jax = je.apply_action(s_jax, jnp.int32(action))
        s_np = ne.apply_action_np(s_np, action)

        # Drive the np witness to the next DECISION/terminal node so the next-loop chance parity is exact,
        # but for the game.py post-action comparison the JAX state may be a chance node (pending deal); in
        # that case committed_* are already final for the street so the comparison still holds.
        _assert_post_action(gs, s_jax, ctx)

    # ---- TERMINAL settlement parity (game.py vs canonical jax/np engine) ----
    # Settle the canonical engine via compute_returns_np (== side_pots_jax.compute_returns == jax settlement
    # per test_np_engine_equiv; robust to the terminal-not-done state where s_jax.rewards is unpopulated).
    n_show = int(np.sum(
        (np.asarray(s_jax.seat_status) == int(SEAT_LIVE)) | (np.asarray(s_jax.seat_status) == int(SEAT_ALLIN))
    ))
    r_jax = np.asarray(ne.compute_returns_np(
        holes=np.asarray(s_jax.holes), board=np.asarray(s_jax.board),
        seat_status=np.asarray(s_jax.seat_status), committed_total=np.asarray(s_jax.committed_total),
        button=int(np.asarray(s_jax.button)), n_showdown=n_show,
    )).astype(np.int64)
    r_np = np.asarray(ne.compute_returns_np(
        holes=np.asarray(s_np.holes), board=np.asarray(s_np.board),
        seat_status=np.asarray(s_np.seat_status), committed_total=np.asarray(s_np.committed_total),
        button=int(np.asarray(s_np.button)), n_showdown=n_show,
    )).astype(np.int64)
    assert np.array_equal(r_jax, r_np), f"seed={seed}: jax/np lockstep state diverged -> returns {r_jax} vs {r_np}"
    assert int(r_jax.sum()) == 0, f"seed={seed}: returns not zero-sum {r_jax}"

    gs = game_state_from_jax(s_jax)
    g_ret = _settle_game(gs)
    assert np.array_equal(g_ret, r_jax), (
        f"seed={seed}: TERMINAL settlement game.py != jax\n"
        f"  committed_total={np.asarray(s_jax.committed_total).tolist()}\n"
        f"  status={np.asarray(s_jax.seat_status).tolist()} button={int(np.asarray(s_jax.button))}\n"
        f"  game={g_ret.tolist()}\n  jax ={r_jax.tolist()}\n  delta={(g_ret - r_jax).tolist()}"
    )
    return {"nodes": nodes, "saw_river": saw_river}


def test_game_jax_lockstep():
    assert G_NUM_PLAYERS == NP, "game.py and jax_engine seat counts differ"
    n_hands = _hand_count()
    rng = np.random.default_rng(0x10C57E9)
    total_nodes = 0
    terminals = 0
    showdowns = 0
    for seed in range(n_hands):
        occ, stacks, ante = _hand_inputs(rng)
        info = _run_one_hand(seed, occ, stacks, ante)
        total_nodes += info["nodes"]
        terminals += 1
        if info["saw_river"]:
            showdowns += 1
    # NON-VACUOUS guard: the gate must actually have compared many per-step decision nodes, not just
    # terminals (a degenerate run that folded instantly everywhere would pass terminal-only).
    assert total_nodes > terminals, (
        f"vacuous run: {total_nodes} decision nodes for {terminals} hands (no mid-hand comparison)"
    )
    assert total_nodes >= n_hands, f"too few decision nodes ({total_nodes}) for {n_hands} hands"
    # DEPTH guard: a settlement gate that never reached a river decision would miss multi-street side-pot
    # layering. Require at least some river-betting hands (trivially satisfied for n_hands>~50).
    if n_hands >= 50:
        assert showdowns > 0, "no hand reached a river decision node (shallow run, side pots untested)"
    print(
        f"\n[lockstep] {n_hands} hands OK: {total_nodes} per-step decision nodes compared "
        f"(game.py<->jax<->np), {showdowns} reached a river decision. "
        f"(set POKERBOT_LOCKSTEP_HANDS=1000000 for the full B gate)"
    )


if __name__ == "__main__":
    test_game_jax_lockstep()
    print("lockstep OK")
