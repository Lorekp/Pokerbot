"""Adapter for DeepBot — a pretrained 6-max LSTM bot from third_party/deepbot-poker.

Loading mirrors code/agent_play.py:47-54 exactly.  The ctypes OMPEval dependency
(comp_hand_equity in u_bot.py) is killed before the module loads by monkeypatching
u_bot.comp_hand_equity with a numpy Monte-Carlo equity function.

Public API
----------
make_deepbot_policy(pkl_path=<default>) -> PolicyFn: (env, seat) -> np.ndarray[11]

The returned function is deterministic given (env, seat):
  - A separate DeepBot instance per seat holds the LSTM hidden state.
  - Per-decision: the LSTM hidden state is saved before calling net_predict and
    restored afterwards (the state advance from one decision to the next is done
    by re-running the same call without the save; see details below).
  - New-hand detection (street==0 with fewer hist entries than last tick) triggers
    bot.model.reset() so cross-hand memory does not break CRN pairing.
  - Hand equity sampling is seeded via common.stable_seed for CRN determinism.
"""
from __future__ import annotations

import copy
import logging
import os
import sys
from typing import Callable, Optional

import numpy as np

from pokerbot.eval.external_bots.common import (
    build_view,
    card_id_to_ppe,
    card_id_to_str,
    finalize_action,
    safe_policy,
    stable_seed,
)
from pokerbot.engine.np_engine import (
    SEAT_ALLIN,
    SEAT_FOLDED,
    SEAT_LIVE,
    evaluate_seven,
    legal_action_mask_np,
    NUM_ACTIONS,
)

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default pkl path (relative to repo root)
# ---------------------------------------------------------------------------

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_HERE)))
_DEFAULT_PKL = os.path.join(
    _REPO_ROOT,
    "third_party", "deepbot-poker", "data",
    "trained_agents_git", "6max_full", "gen_300", "bots", "1", "bot_1_flat.pkl",
)

# ---------------------------------------------------------------------------
# sys.path bootstrap for third_party trees (idempotent)
# ---------------------------------------------------------------------------

def _bootstrap_paths() -> None:
    ppe_path = os.path.join(_REPO_ROOT, "third_party", "pypokerengine")
    deepbot_bots = os.path.join(_REPO_ROOT, "third_party", "deepbot-poker", "code", "bots")
    deepbot_main = os.path.join(_REPO_ROOT, "third_party", "deepbot-poker", "code", "main_functions")
    for p in (ppe_path, deepbot_bots, deepbot_main):
        if p not in sys.path:
            sys.path.insert(0, p)


_bootstrap_paths()

# ---------------------------------------------------------------------------
# Monte-Carlo equity — replaces ctypes comp_hand_equity from u_bot.py
#
# Signature must match u_bot.comp_hand_equity exactly:
#   comp_hand_equity(hole_card, community_card, n_act_players,
#                    nb_board_cards=5, std_err_tol=..., verbose=False)
#   -> float  (hero equity, 0..1)
#
# hole_card      : list of PPE suit-first strings, e.g. ['SA', 'DK']
# community_card : list of PPE suit-first strings for dealt board cards
# n_act_players  : total active players (hero + opponents)
# nb_board_cards : 3 = showdown at flop only, 5 = full runout
# ---------------------------------------------------------------------------

# 50 rollouts: ~60ms per equity call; good enough precision for this LSTM signal.
_N_ROLLOUTS = 50

_PPE_RANK = "23456789TJQKA"  # index = rank_idx (0-based from 2)
_PPE_SUIT_TO_IDX = {"C": 0, "D": 1, "H": 2, "S": 3}


def _ppe_to_card_id(s: str) -> int:
    """Convert PPE suit-first string 'SA' -> card id 0..51."""
    suit_idx = _PPE_SUIT_TO_IDX[s[0]]
    rank_idx = _PPE_RANK.index(s[1])
    return rank_idx * 4 + suit_idx


def _mc_hand_equity(
    hole_card,
    community_card,
    n_act_players: int,
    nb_board_cards: int = 5,
    std_err_tol=None,
    verbose: bool = False,
    rng: Optional[np.random.Generator] = None,
) -> float:
    """Monte-Carlo hero equity vs n_act_players-1 random opponents.

    Uses evaluate_seven from pokerbot.engine.np_engine.
    Cards arrive as PPE suit-first strings (e.g. 'SA', 'DK').
    """
    if rng is None:
        rng = np.random.default_rng(42)

    hero_ids = [_ppe_to_card_id(c) for c in hole_card]
    board_ids = [_ppe_to_card_id(c) for c in community_card]
    n_opps = max(1, n_act_players - 1)

    known = set(hero_ids + board_ids)
    deck = np.array([c for c in range(52) if c not in known], dtype=np.int32)

    n_board_needed = max(0, nb_board_cards - len(board_ids))
    cards_needed = 2 * n_opps + n_board_needed

    hero_arr = np.array(hero_ids, dtype=np.int32)
    board_base = np.array(board_ids, dtype=np.int32)

    wins = 0
    valid_rollouts = 0

    for _ in range(_N_ROLLOUTS):
        if len(deck) < cards_needed:
            break
        rng.shuffle(deck)
        draw = deck[:cards_needed]

        if n_board_needed > 0:
            board_full = np.concatenate([board_base, draw[:n_board_needed]])
        else:
            board_full = board_base[:nb_board_cards]
        opp_start = n_board_needed

        # Evaluate hero's 7-card hand (pad board to 5 if shorter)
        if len(board_full) < 5:
            padded = np.full(5, 0, dtype=np.int32)
            padded[: len(board_full)] = board_full
        else:
            padded = board_full[:5]

        hero_seven = np.concatenate([hero_arr, padded])
        hero_score = int(evaluate_seven(hero_seven))

        hero_wins = True
        for opp_i in range(n_opps):
            opp_hole = draw[opp_start + opp_i * 2: opp_start + opp_i * 2 + 2]
            opp_seven = np.concatenate([opp_hole, padded])
            if int(evaluate_seven(opp_seven)) >= hero_score:
                hero_wins = False
                break

        valid_rollouts += 1
        if hero_wins:
            wins += 1

    if valid_rollouts == 0:
        return 1.0 / max(1, n_act_players)
    return wins / valid_rollouts


# ---------------------------------------------------------------------------
# Monkeypatch u_bot.comp_hand_equity BEFORE importing bot_DeepBot
# ---------------------------------------------------------------------------

# Mutable cell so _policy closures can update the rng seed per call
_equity_rng: list[Optional[np.random.Generator]] = [None]


def _patched_comp_hand_equity(
    hole_card, community_card, n_act_players, nb_board_cards=5,
    std_err_tol=None, verbose=False
) -> float:
    rng = _equity_rng[0] if _equity_rng[0] is not None else np.random.default_rng(0)
    return _mc_hand_equity(
        hole_card, community_card, n_act_players, nb_board_cards, rng=rng,
    )


import u_bot as _u_bot  # noqa: E402  (after sys.path bootstrap)
_u_bot.comp_hand_equity = _patched_comp_hand_equity

import bot_DeepBot as _bot_deepbot_mod  # noqa: E402
_bot_deepbot_mod.comp_hand_equity = _patched_comp_hand_equity

from bot_DeepBot import DeepBot  # noqa: E402
import pickle  # noqa: E402
import torch  # noqa: E402

from u_formatting import get_full_dict  # noqa: E402

# ---------------------------------------------------------------------------
# Street name mapping
# ---------------------------------------------------------------------------

_STREET_NAMES = ["preflop", "flop", "turn", "river"]

# ---------------------------------------------------------------------------
# Seat state mapping (DeepBot reads string labels)
# ---------------------------------------------------------------------------

_SEAT_STATE = {
    0: "participating",  # EMPTY — shouldn't appear mid-hand
    1: "folded",
    2: "participating",
    3: "allin",
}


def _seat_state_str(status_int: int) -> str:
    return _SEAT_STATE.get(int(status_int), "participating")


# ---------------------------------------------------------------------------
# Action history reconstruction
#
# DeepBot.prep_input reads action_histories keyed by street name.
# Each entry: {'uuid', 'action': 'RAISE'/'CALL'/'FOLD'/'SMALLBLIND'/'BIGBLIND',
#              'amount': cumulative street-total AFTER the action}
#
# EnvState history: hist_action codes 0=FOLD 1=CHECK 2=CALL 3-9=bet-buckets 10=ALL_IN
# hist_amount[i] = chips committed BY that action (a per-action delta).
#
# Strategy:
#   1. Prepend synthetic SB/BB posts to preflop.
#   2. Replay history deltas, computing per-seat cumulative-per-street totals.
#   3. Street boundary: when the cumulative running total for ALL active seats
#      equals their (committed_total - committed_street), the prior streets are
#      exhausted and we advance to the current street.  Remaining actions belong
#      to the current street.
#   4. Only fields needed by DeepBot are populated:
#        - comp_last_amount reads action_histories[current_street] for hero uuid
#        - comp_last_amount_opp reads action_histories[current_street] for each opp
#        - opp action type (RAISE/CALL/FOLD) on current street
#        - new_round_handle checks preflop for hero's non-SB/BB action
# ---------------------------------------------------------------------------

def _action_str(a: int) -> str:
    """Convert EnvState action code to PPE action string."""
    if a == 0:
        return "FOLD"
    if a == 1:
        return "CHECK"   # DeepBot ignores CHECK, acts like a 0-cost CALL
    if a == 2:
        return "CALL"
    if a == 10:
        return "RAISE"
    return "RAISE"  # bet/raise buckets 3-9


def _build_action_histories(env, seat: int) -> dict:
    """Reconstruct PPE-style action_histories from EnvState.

    Returns dict keyed by street name with PPE-format action entries.
    """
    hist_len = int(env.hist_len)
    hist_seat_arr = np.asarray(env.hist_seat[:hist_len], dtype=np.int32)
    hist_action_arr = np.asarray(env.hist_action[:hist_len], dtype=np.int32)
    hist_amount_arr = np.asarray(env.hist_amount[:hist_len], dtype=np.int32)

    btn = int(env.button)
    sb_seat = (btn + 1) % 6
    bb_seat = (btn + 2) % 6

    # Per-seat prior-street totals (what was committed before the current street)
    committed_total = np.asarray(env.committed_total, dtype=np.int64)
    committed_street = np.asarray(env.committed_street, dtype=np.int64)
    prior_total = committed_total - committed_street  # per-seat prior sum
    # prior_total includes SB/BB for preflop

    # Replay: accumulate running per-seat committed across all prior streets
    # to find the split index between prior streets and current street.
    per_seat_running = np.zeros(6, dtype=np.int64)
    per_seat_running[sb_seat] = 50   # synthetic SB post
    per_seat_running[bb_seat] = 100  # synthetic BB post

    # Active seats: any that are LIVE or ALLIN (status 2 or 3) or had committed
    seat_status = np.asarray(env.seat_status, dtype=np.int8)
    ever_active = (committed_total > 0) | (committed_street > 0)

    # Check if prior_total is already satisfied by SB/BB (i.e., current_street == 0)
    current_street_idx = int(env.street)

    # Find the split: walk history and find where prior_total is covered
    split_idx = hist_len  # default: all history is prior-street
    if current_street_idx == 0:
        # All history belongs to preflop (current street)
        split_idx = 0
    else:
        # Prior-street actions are those whose cumulative sum equals prior_total
        # We walk forward and find when all ever-active seats are covered
        for i in range(hist_len):
            s = int(hist_seat_arr[i])
            delta = int(hist_amount_arr[i])
            per_seat_running[s] += delta
            # Check if all ever-active seats have been covered by prior_total
            all_covered = all(
                per_seat_running[j] >= prior_total[j]
                for j in range(6)
                if ever_active[j]
            )
            if all_covered:
                split_idx = i + 1  # first i+1 entries are prior-street
                break

    # Prior-street actions (for new_round_handle street check etc.)
    # We only need these to satisfy the 'preflop' key for new_round_handle.
    # Build a minimal preflop history for the synthetic SB/BB.
    preflop_entries = [
        {"uuid": f"uuid-{sb_seat + 1}", "action": "SMALLBLIND", "amount": 50},
        {"uuid": f"uuid-{bb_seat + 1}", "action": "BIGBLIND",   "amount": 100},
    ]

    # Add any prior-street preflop actions (actions 0..split_idx-1 if street > 0)
    # These are only needed by new_round_handle which checks hero's non-SB/BB preflop actions.
    # For simplicity, we record them all in 'preflop' for street>0 scenarios.
    # The per-seat cumulative for prior streets restarts from zero each street,
    # but for the preflop key we just add them sequentially — DeepBot only checks
    # for the absence/presence of hero actions in preflop (new_round_handle), not amounts.
    per_seat_prior_committed = np.zeros(6, dtype=np.int64)
    per_seat_prior_committed[sb_seat] = 50
    per_seat_prior_committed[bb_seat] = 100
    for i in range(split_idx):
        s = int(hist_seat_arr[i])
        a = int(hist_action_arr[i])
        delta = int(hist_amount_arr[i])
        per_seat_prior_committed[s] += delta
        if a == 1:  # CHECK
            amt = 0
        elif a == 0:  # FOLD
            amt = 0
        else:
            amt = int(per_seat_prior_committed[s])
        preflop_entries.append({
            "uuid": f"uuid-{s + 1}",
            "action": _action_str(a),
            "amount": amt,
        })

    # Current-street actions (split_idx..hist_len-1)
    current_street_entries = []
    per_seat_cur = np.zeros(6, dtype=np.int64)
    for i in range(split_idx, hist_len):
        s = int(hist_seat_arr[i])
        a = int(hist_action_arr[i])
        delta = int(hist_amount_arr[i])
        per_seat_cur[s] += delta
        if a == 1:  # CHECK
            amt = 0
        elif a == 0:  # FOLD
            amt = 0
        else:
            amt = int(per_seat_cur[s])
        current_street_entries.append({
            "uuid": f"uuid-{s + 1}",
            "action": _action_str(a),
            "amount": amt,
        })

    # Assign to named streets
    street_name = _STREET_NAMES[min(current_street_idx, 3)]
    histories: dict[str, list] = {"preflop": [], "flop": [], "turn": [], "river": []}
    histories["preflop"] = preflop_entries
    histories[street_name] = current_street_entries  # may overwrite preflop if street==0

    return histories


def _build_round_state(env, seat: int) -> dict:
    """Build a PPE-style round_state dict from EnvState for DeepBot."""
    stacks = np.asarray(env.stacks, dtype=np.int64)
    committed_street = np.asarray(env.committed_street, dtype=np.int64)
    seat_status = np.asarray(env.seat_status, dtype=np.int8)
    street_idx = int(env.street)
    btn = int(env.button)
    sb_seat = (btn + 1) % 6

    seats_list = [
        {
            "uuid": f"uuid-{i + 1}",
            "name": f"p-{i + 1}",
            "stack": int(stacks[i]),
            "state": _seat_state_str(int(seat_status[i])),
        }
        for i in range(6)
    ]

    board = np.asarray(env.board, dtype=np.int32)
    community_card = [card_id_to_ppe(int(c)) for c in board if int(c) >= 0]

    pot_dead = int(env.pot_dead)
    pot_main = pot_dead + int(committed_street.sum())

    action_histories = _build_action_histories(env, seat)

    return {
        "street": _STREET_NAMES[min(street_idx, 3)],
        "next_player": seat,
        "small_blind_pos": sb_seat,
        "small_blind_amount": 50,
        "community_card": community_card,
        "seats": seats_list,
        "pot": {"main": {"amount": pot_main}, "side": []},
        "action_histories": action_histories,
    }


def _build_valid_actions(env, seat: int) -> list:
    """Build PPE valid_actions list for decision_algo."""
    cs = int(env.committed_street[seat])
    cur_bet = int(env.current_bet)
    to_call = max(0, cur_bet - cs)
    stack = int(env.stacks[seat])
    min_raise = int(env.min_raise)

    call_amount = cur_bet if to_call > 0 else 0
    min_raise_total = cur_bet + min_raise
    shove_to = cs + stack

    if stack <= 0 or min_raise_total > shove_to:
        min_rt, max_rt = -1, -1
    else:
        min_rt = min_raise_total
        max_rt = shove_to

    return [
        {"action": "fold", "amount": 0},
        {"action": "call", "amount": call_amount},
        {"action": "raise", "amount": {"min": min_rt, "max": max_rt}},
    ]


# ---------------------------------------------------------------------------
# LSTM hidden-state save/restore helpers
#
# DeepBot's forward() mutates u_gen and u_opp (LSTM hidden/cell states) in-place.
# For determinism, we SAVE the state before the call and RESTORE after so that
# two identical (env, seat) queries produce the same output without corrupting
# the running game-state memory.
#
# For the RUNNING game (advancing hand state across decisions), we want the LSTM
# state to advance after each decision.  We achieve this by:
#   1. Save state before call → call → record output → RESTORE state
#   2. Call once more (same inputs) to advance the state irreversibly
#
# This is equivalent to: compute output with saved state, then re-run to advance.
# Two calls per decision is acceptable at ~60-80ms each.
# ---------------------------------------------------------------------------

def _save_lstm_state(bot: "DeepBot") -> tuple:
    """Deep-copy the LSTM hidden/cell state dicts from a DeepBot."""
    u_gen_copy = {k: v.clone() for k, v in bot.model.u_gen.items()}
    u_opp_copy = {k: v.clone() for k, v in bot.model.u_opp.items()}
    return u_gen_copy, u_opp_copy


def _restore_lstm_state(bot: "DeepBot", state: tuple) -> None:
    """Restore a previously saved LSTM state into bot.model."""
    u_gen_copy, u_opp_copy = state
    for k, v in u_gen_copy.items():
        bot.model.u_gen[k] = v.clone()
    for k, v in u_opp_copy.items():
        bot.model.u_opp[k] = v.clone()


# ---------------------------------------------------------------------------
# New-hand detector
# ---------------------------------------------------------------------------

class _HandTracker:
    """Detects a new hand starting by watching (street, hist_len)."""

    def __init__(self) -> None:
        self._last_hist_len: int = -1

    def is_new_hand(self, env) -> bool:
        h = int(env.hist_len)
        street = int(env.street)
        if self._last_hist_len < 0:
            self._last_hist_len = h
            return True
        is_new = street == 0 and h <= self._last_hist_len
        self._last_hist_len = h
        return is_new


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def make_deepbot_policy(pkl_path: Optional[str] = None) -> Callable:
    """Load DeepBot from pkl and return a PolicyFn.

    Parameters
    ----------
    pkl_path : path to the flat tensor pkl file
               (default: 6max_full/gen_300/bots/1/bot_1_flat.pkl)

    Returns
    -------
    PolicyFn : (env, seat) -> np.ndarray[11], safe-wrapped, deterministic.
    """
    if pkl_path is None:
        pkl_path = _DEFAULT_PKL

    # Load flat tensor and build reference dict
    ref_full_dict = DeepBot(network="6max_full").full_dict
    with open(pkl_path, "rb") as f:
        flat = pickle.load(f)
    n_params = flat.numel() if hasattr(flat, "numel") else len(flat)
    log.info("deepbot_adapter: loaded %s  params=%d", pkl_path, n_params)

    # One DeepBot per seat (carries per-seat LSTM state across decisions)
    bots: list[Optional[DeepBot]] = [None] * 6
    trackers: list[_HandTracker] = [_HandTracker() for _ in range(6)]

    # Per-seat cache of the LAST decision's (hist_len_key, pre_lstm_state, result)
    # so repeated calls with the same env return identical results (determinism).
    _last_call: list[Optional[tuple]] = [None] * 6   # (hist_len_key, pre_state, result_vec)
    # LSTM state that was current AFTER the last irreversible decision advance
    _post_decision_state: list[Optional[tuple]] = [None] * 6

    def _make_bot() -> DeepBot:
        full_dict = get_full_dict(all_params=flat, ref_full_dict=ref_full_dict)
        bot = DeepBot(full_dict=full_dict, network="6max_full")
        bot.i_stack = 10000
        return bot

    def _env_key(env, seat: int) -> int:
        """A cheap per-decision fingerprint for detecting replay vs new decision."""
        h = int(env.hist_len)
        st = int(env.street)
        # We also mix in a hash of the hero holes + board to make it robust
        seed = stable_seed(env, seat)
        return hash((h, st, seed))

    def _policy(env, seat: int) -> np.ndarray:
        seat = int(seat)

        # Lazy-init per-seat bot
        if bots[seat] is None:
            bots[seat] = _make_bot()
            bots[seat].uuid = f"uuid-{seat + 1}"

        bot = bots[seat]
        tracker = trackers[seat]

        # New-hand reset: clears LSTM hidden state
        if tracker.is_new_hand(env):
            bot.model.reset()
            _last_call[seat] = None
            _post_decision_state[seat] = None

        key = _env_key(env, seat)

        # Determinism: if this is a REPEAT call with the same env, return cached result
        if _last_call[seat] is not None:
            cached_key, cached_pre, cached_result = _last_call[seat]
            if cached_key == key:
                return cached_result.copy()

        # New decision: restore post-decision LSTM state from prior decision
        # (or leave as-is if this is the first decision of the hand, post reset())
        if _post_decision_state[seat] is not None:
            _restore_lstm_state(bot, _post_decision_state[seat])

        # Save pre-decision state for determinism cache
        pre_state = _save_lstm_state(bot)

        # Seed equity RNG via stable_seed for CRN determinism
        seed = stable_seed(env, seat)
        _equity_rng[0] = np.random.default_rng(seed & 0xFFFFFFFF)

        # Build inputs
        hole_card = list(build_view(env, seat).hero_hole_ppe)
        round_state = _build_round_state(env, seat)
        valid_actions = _build_valid_actions(env, seat)
        bot.uuid = f"uuid-{seat + 1}"

        with torch.no_grad():
            action_str, amount = bot.declare_action(valid_actions, hole_card, round_state)

        # Save post-decision LSTM state (the "advanced" state for next decision)
        _post_decision_state[seat] = _save_lstm_state(bot)

        # Translate declare_action output to 11-vector
        if action_str == "fold":
            result = finalize_action(env, seat, "fold")
        elif action_str == "call":
            to_call = max(0, int(env.current_bet) - int(env.committed_street[seat]))
            if to_call == 0:
                result = finalize_action(env, seat, "check")
            else:
                result = finalize_action(env, seat, "call")
        elif action_str == "raise":
            amt = int(amount) if amount is not None else -1
            shove_to = int(env.committed_street[seat]) + int(env.stacks[seat])
            if amt < 0 or amt >= shove_to:
                result = finalize_action(env, seat, "all_in")
            else:
                result = finalize_action(env, seat, "raise", raise_to=amt)
        else:
            # Unknown — fall back
            to_call = max(0, int(env.current_bet) - int(env.committed_street[seat]))
            if to_call > 0:
                result = finalize_action(env, seat, "call")
            else:
                result = finalize_action(env, seat, "check")

        # Cache (key, pre_state, result) for determinism on repeat calls
        _last_call[seat] = (key, pre_state, result.copy())

        return result

    _policy.__name__ = "deepbot_policy"
    return safe_policy(_policy)


# ---------------------------------------------------------------------------
# __main__ smoke block
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import os
    import time
    from collections import Counter

    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    os.environ.setdefault("POKERBOT_BETTING_ABSTRACTION", "1")
    os.environ.setdefault("POKERBOT_POSTFLOP_BUCKETS", "1")
    os.environ.setdefault("POKERBOT_PREFLOP_LOSSLESS", "1")
    os.environ.setdefault("POKERBOT_POSTFLOP_MODE", "fast")
    os.environ.setdefault("POKERBOT_POSTFLOP_NUM_BUCKETS", "200")

    import jax
    import jax.numpy as jnp

    from pokerbot.engine.jax_engine import init_hand
    from pokerbot.engine.np_engine import (
        apply_action_np,
        current_player_np,
        is_chance_node_np,
        is_terminal_np,
        sample_chance_np,
        to_numpy_state,
    )
    import pokerbot.eval.external_bots.common as _common

    # (1) Load and verify param count
    print("=== DeepBot smoke ===")
    with open(_DEFAULT_PKL, "rb") as f:
        _flat = pickle.load(f)
    n_params = _flat.numel() if hasattr(_flat, "numel") else len(_flat)
    print(f"Loaded {_DEFAULT_PKL}")
    print(f"  param count = {n_params}  (expected 24971)")
    assert abs(n_params - 24971) < 100, f"Unexpected param count: {n_params}"

    # (2) Play ~50 full random hands with deepbot at one seat, assert 11-vector contract
    policy_fn = make_deepbot_policy()
    occ = jnp.array([True] * 6)
    SEED_BASE = 7777
    N_HANDS = 50
    HERO_SEAT = 2

    freq: Counter = Counter()
    n_decisions = 0
    n_det_fail = 0
    n_illegal = 0
    n_wrong_sum = 0
    latencies = []
    rng = np.random.default_rng(SEED_BASE)
    _common._failure_count = 0

    print(f"\nRunning {N_HANDS} hands (hero_seat={HERO_SEAT}, others random-legal)...")
    for hand_idx in range(N_HANDS):
        env = to_numpy_state(init_hand(
            jax.random.PRNGKey(SEED_BASE + hand_idx * 1009), occupancy=occ
        ))
        guard = 0
        while not is_terminal_np(env) and guard < 400:
            guard += 1
            if is_chance_node_np(env):
                env = sample_chance_np(env)
                continue
            seat = current_player_np(env)
            if seat < 0:
                break
            mask = legal_action_mask_np(env, seat)
            if int(mask.sum()) <= 1:
                env = apply_action_np(env, int(np.argmax(mask)))
                continue
            if seat == HERO_SEAT:
                t0 = time.perf_counter()
                probs = policy_fn(env, seat)
                lat = time.perf_counter() - t0
                latencies.append(lat)
                n_decisions += 1

                # Determinism check (second call with same env)
                probs2 = policy_fn(env, seat)
                if not np.allclose(np.asarray(probs), np.asarray(probs2), atol=1e-12):
                    n_det_fail += 1
                    print(f"  [FAIL] non-deterministic at hand {hand_idx}")

                # Sum check
                s = float(np.asarray(probs).sum())
                if abs(s - 1.0) > 1e-9:
                    n_wrong_sum += 1

                # Illegal mass check
                ill = float(np.asarray(probs)[~mask].sum())
                if ill > 1e-9:
                    n_illegal += 1
                    print(f"  [FAIL] illegal mass={ill:.4f}  mask={mask.astype(int).tolist()}")

                idx = int(np.argmax(probs))
                if idx == 0:   freq["fold"] += 1
                elif idx == 1: freq["check"] += 1
                elif idx == 2: freq["call"] += 1
                elif idx == 10: freq["allin"] += 1
                else:          freq["raise_bet"] += 1

                pc = np.clip(np.asarray(probs, dtype=np.float64), 0, None)
                pc = pc / pc.sum()
                a = int(rng.choice(len(pc), p=pc))
            else:
                lf = mask.astype(np.float64)
                a = int(rng.choice(len(lf), p=lf / lf.sum()))
            env = apply_action_np(env, a)

    avg_lat = float(np.mean(latencies)) * 1000 if latencies else 0.0
    total = max(n_decisions, 1)
    print(f"\nSmoke results ({N_HANDS} hands, hero_seat={HERO_SEAT}):")
    print(f"  decisions={n_decisions}  failures={_common._failure_count}")
    print(f"  avg_latency={avg_lat:.0f}ms")
    print(f"  det_fail={n_det_fail}  illegal={n_illegal}  wrong_sum={n_wrong_sum}")
    print(f"  action_freq: " + "  ".join(
        f"{k}={v}/{total}={100*v/total:.1f}%"
        for k, v in sorted(freq.items(), key=lambda x: -x[1])
    ))

    assert _common._failure_count == 0, f"safe_policy failures: {_common._failure_count}"
    assert n_wrong_sum == 0, "probs-sum violations"
    assert n_illegal == 0, "illegal-mass violations"
    assert n_det_fail == 0, "determinism failures"
    print("\n=== SMOKE PASS (phase 1: 11-vector contract) ===")

    # (3) Run external_ladder_h2h.py --opponent deepbot --decks 2 --iters 200
    import subprocess
    import sys as _sys

    ladder_out = os.path.join(_REPO_ROOT, "artifacts", "deepbot_smoke_ladder.json")
    ladder_cmd = [
        _sys.executable, "-u",
        os.path.join(_REPO_ROOT, "scripts", "external_ladder_h2h.py"),
        "--opponent", "deepbot",
        "--decks", "2",
        "--iters", "200",
        "--settle-samples", "50",
        "--seed", "9999",
        "--out", ladder_out,
    ]
    print("\n=== Running ladder H2H smoke (2 decks, K=200) ===")
    print("  CMD: " + " ".join(ladder_cmd))
    result = subprocess.run(ladder_cmd, capture_output=False)
    print(f"\n  ladder exit code: {result.returncode}")
    if result.returncode != 0:
        print("  [FAIL] ladder H2H smoke failed")
    else:
        print(f"  [PASS] ladder H2H smoke done  output: {ladder_out}")
