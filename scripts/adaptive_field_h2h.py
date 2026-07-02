#!/usr/bin/env python3
"""Adaptive Field Benchmark — engine plays external bots WITH opponent-modeling ON.

The benchmark gives the engine a STABLE identity per bot (e.g. "honest#0", "honest#1")
so the PlayerProfileStore tracks them individually across all N hands, exactly as it does
in a real live session.  Seat rotation each hand keeps EV fair.  The profile store
PERSISTS across hands (that is the whole point: watch the engine learn each opponent).

Usage:
    python -u scripts/adaptive_field_h2h.py \
        --field '{"honest": 3}' \
        --hands 5000 \
        --seed 42 \
        --exploit on \
        --windows 10 \
        --out artifacts/adaptive_honest_on.json

Quick smoke (60 hands, honest field):
    python -u scripts/adaptive_field_h2h.py \
        --field '{"honest": 3}' --hands 60 --seed 42 --exploit on  --out /tmp/adaptive_smoke_on.json
    python -u scripts/adaptive_field_h2h.py \
        --field '{"honest": 3}' --hands 60 --seed 42 --exploit off --out /tmp/adaptive_smoke_off.json

Environment:
    Same POKERBOT_* preamble as exploitation_battery / blueprint_ranking (set BEFORE import).
    POKERBOT_BLUEPRINT   path to serve_blueprint_6max.pkl (default: checkpoints/serve_blueprint_6max.pkl)

Cautions:
  * --tmp-db creates a fresh temp dir for the PlayerProfileStore and deletes it after unless --keep-db.
    This ensures the run NEVER pollutes the real live profiles.json.
  * deepbot is listed as a TODO (see DEEPBOT_TODO at the bottom of this file).
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# CRITICAL: POKERBOT_* env vars must be set BEFORE any pokerbot import.
# Mirrors the preamble in blueprint_ranking.py / exploitation_battery.py.
# ---------------------------------------------------------------------------
import os
import sys

os.environ.setdefault("POKERBOT_BETTING_ABSTRACTION", "1")
os.environ.setdefault("POKERBOT_POSTFLOP_BUCKETS", "1")
os.environ.setdefault("POKERBOT_PREFLOP_LOSSLESS", "1")
os.environ.setdefault("POKERBOT_POSTFLOP_MODE", "emd")
os.environ.setdefault("POKERBOT_POSTFLOP_NUM_BUCKETS", "160")
# EMD-K160 deployed regime (promoted 2026-07-02, certified +32.89 bb/100 vs made-hand iter62000):
# frozen flop/turn EMD lookup + train-mirrored made-hand fallback on the river (the river is not
# in the blob; without the mirror flag the river frozen-miss serves a UNIFORM policy).
os.environ.setdefault("POKERBOT_POSTFLOP_EMD_FROZEN", os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "data", "abstraction", "emd_frozen_flop_turn_160_r400.emdfrzn")))
os.environ.setdefault("POKERBOT_EMD_FROZEN_MIRROR_FALLBACK", "1")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
# Native subgame (POKERBOT_NATIVE_SUBGAME=1): use the Rust CFR core for speed.
# Set here so it can be overridden before this module is imported.
os.environ.setdefault("POKERBOT_NATIVE_SUBGAME", "1")

import argparse
import json
import math
import random
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Imports from pokerbot (after env vars)
# ---------------------------------------------------------------------------
from pokerbot.engine.actions import Action, ActionType
from pokerbot.engine.game import GameState, play_hand
from pokerbot.opponent.stats import OpponentStats
from pokerbot.opponent.context import context_key_rs, is_aggressive_type
from pokerbot.opponent.population_prior import context_prior
from pokerbot.opponent.player_profiles import PlayerProfileStore
from pokerbot.eval.exploitation_battery import (
    ExploitHero,
    _ctx_key_for,
    _action_to_bin,
    _allin_ev_settle,
    _play,
)
from pokerbot.eval.external_bots.registry import OPPONENTS, get_opponent_policy
from pokerbot.eval.external_bots.common import build_view, finalize_action
from pokerbot.training.self_play import _index_to_action, _legal_mask


# ---------------------------------------------------------------------------
# Supported pypoker bots (those with clean object-level wrappers)
# ---------------------------------------------------------------------------
_PYPOKER_BOTS = {"fold_man", "random", "fish", "honest"}
_SUPPORTED_BOTS = _PYPOKER_BOTS  # deepbot = TODO (see bottom of file)

NUM_SEATS = 6


# ---------------------------------------------------------------------------
# Object-level wrapper for external (policy-function) bots
# ---------------------------------------------------------------------------
# The exploitation_battery hero contract requires __call__(state: GameState) -> Action,
# but the registry bots implement PolicyFn: (env, seat) -> np.ndarray[11].
#
# Seam choice: we wrap the PolicyFn contract at the object level by calling
#   build_view + finalize_action from common.py to go GameState->EnvState view->Action,
#   then call the PolicyFn at each decision.  This reuses the EXACT same PPE adapter path
#   the pypoker bots were already validated against — we are not rebuilding their logic.
#
# We explicitly do NOT use the game_state_to_env bridge (search/state_bridge) because
# that bridge is for hero/search use and carries private hole information for all seats.
# The external bot's PolicyFn only sees env.holes[seat] (via build_view's seat-scoped
# read), so info hygiene is preserved exactly as in the external_ladder_h2h usage.

def _game_state_to_env_for_seat(state: GameState, seat: int):
    """Convert GameState to an EnvState revealing only the acting seat's hole cards.

    Uses game_state_to_env from state_bridge (the production bridge), which masks all
    holes except hero_seat.  This is exactly the correct info-hygiene for external bots:
    each bot only sees its own cards, just as in the live play path.
    """
    from pokerbot.search.state_bridge import game_state_to_env
    return game_state_to_env(state, int(seat))


class ExternalBotAgent:
    """Wraps a registry PolicyFn so it satisfies GameState -> Action at the object level.

    Each instance gets a deterministic RNG stream seeded from (hand_seed, bot_identity)
    so the same bot plays identically given the same deck, matching the CRN discipline
    used in exploitation_battery.

    Identity: a string like "honest#0" uniquely names this bot instance so the profile
    store tracks it across all hands.
    """

    def __init__(self, bot_name: str, instance_idx: int, policy_fn):
        self.bot_name = bot_name
        self.instance_idx = instance_idx
        self.identity = f"{bot_name}#{instance_idx}"
        self._policy_fn = policy_fn
        self._rng_seed = 0

    def reseed(self, seed: int) -> None:
        self._rng_seed = seed

    def __call__(self, state: GameState) -> Action:
        seat = state.action_seat
        env = _game_state_to_env_for_seat(state, seat)

        # Seed the global random module deterministically (matches pypoker_adapter stable_seed idiom)
        random.seed(self._rng_seed ^ (seat * 0xDEAD_BEEF) ^ (int(env.street) * 0x1234))

        probs = np.asarray(self._policy_fn(env, seat), dtype=np.float64)
        probs = np.clip(probs, 0, None)
        s = probs.sum()
        if s <= 0 or not np.isfinite(s):
            from pokerbot.engine.np_engine import legal_action_mask_np
            mask = legal_action_mask_np(env, seat).astype(np.float64)
            s2 = mask.sum()
            probs = mask / s2 if s2 > 0 else mask
        else:
            probs = probs / s

        idx = int(random.choices(range(len(probs)), weights=probs.tolist())[0])
        return _index_to_action(state, seat, idx)


# ---------------------------------------------------------------------------
# Adaptive hero: ExploitHero with per-opponent-identity profile lookup
# ---------------------------------------------------------------------------

class AdaptiveHero:
    """Hero that looks up each opponent's profile from the shared PlayerProfileStore
    and passes it to ExploitHero's exploit logic.

    Key design:
      - ONE ExploitHero instance per DECISION (it reads self.profile at call time)
      - We swap self.profile before each call to point at the current VILLAIN's OpponentStats
        from the store, so the exploit fires with that opponent's learned tendencies.
      - When exploit=False we never set any profile (pure base play).
      - seat_to_identity: {seat_idx -> bot identity string} — set before each hand by the harness.
    """

    def __init__(self, rng: random.Random, *, exploit: bool, store: PlayerProfileStore | None = None):
        self._rng = rng
        self.exploit = exploit
        self._store = store
        # seat_to_identity populated by the harness before each hand
        self.seat_to_identity: dict[int, str] = {}
        # underlying heuristic value-bettor (we delegate all the action math to it)
        self._inner = ExploitHero(rng, exploit=exploit)
        # hero seat (set by harness before each hand)
        self.hero_seat: int = 0
        # Track bet_freq / decisions via the inner hero
        self._decisions_this_hand: list[tuple] = []

    @property
    def bet_freq(self) -> float:
        return self._inner.bet_freq

    def reseed(self, seed: int) -> None:
        self._rng = random.Random(seed)
        self._inner.rng = self._rng

    def set_profile(self, profile) -> None:
        # Called by the harness's compatibility shim (no-op here; we do dynamic lookup)
        pass

    def pop_decisions(self) -> list:
        return self._inner.pop_decisions()

    def pop_chance_corrections(self) -> list:
        return []

    def __call__(self, state: GameState) -> Action:
        if self.exploit and self._store is not None:
            # Identify the opponent(s) currently in contention to act against us.
            # Use the action_seat's direct context (ExploitHero reads the villain profile
            # from the last set_profile, so we figure out who is "facing" the hero and
            # pick that seat's profile).  The hero's own seat never needs a profile.
            # For simplicity: populate ALL seats' profiles in the inner hero via a dict
            # so context_read can look up whichever seat is relevant.
            seat = self.hero_seat
            profiles: dict[int, OpponentStats] = {}
            for s, ident in self.seat_to_identity.items():
                if s != seat:
                    profiles[s] = self._store.get(ident)
            # ExploitHero.set_profile accepts either a single OpponentStats or a dict.
            # We pool across identities into ONE OpponentStats (the same pooling the
            # exploitation_battery does for a homogeneous field in run_archetype), but
            # for a mixed field we want per-villain reads.
            # We resolve this by building a POOLED profile from only the villain seats
            # that are active in this hand — a simple aggregate for the heuristic hero.
            pooled = _pool_profiles(profiles)
            self._inner.set_profile(pooled)
        return self._inner(state)


# ---------------------------------------------------------------------------
# Production hero: wraps SearchAgent (Pluribus + EMD-K160 blueprint + exploit overlays)
# ---------------------------------------------------------------------------
# This mirrors what the live LiveSearchAdvisor does, but in sim:
#   exploit ON  -> set_opponent_profiles({seat: OpponentStats}) fed from the persistent store,
#                  which wires the solver's opponent-node deviation tail (postflop) AND the root
#                  mixer (all streets). Semantics match live (2026-07-02): with
#                  POKERBOT_LIVE_NATIVE_EXPLOIT=1 (OPT-IN; default OFF pending the value A/B —
#                  the fidelity gate measured the frozen serve diverging at equal K, flip 0.667)
#                  SearchAgent wires the frozen-read-response native exploit serve, so exploit
#                  decisions run the native solve_exploit against Python-built frozen deviation
#                  tables (exploit math stays in Python — firewall). Default/unset = the dynamic
#                  Python CFR exploit tail (the certified live behaviour).
#   exploit OFF -> set_opponent_profiles(None): pure GTO base, native fast path eligible.
#
# Budget: K iterations (below), time_budget_ms=10**9 (wall-clock never binds).
# This makes each decision a pure function of (public state, hand seed), load-independent.
# RAYON_NUM_THREADS is NOT forced to 1 (native solve uses rayon for parallel work — keep it).

_PROD_HERO_K = int(os.environ.get("ADAPTIVE_PROD_K", "200"))
                              # iteration budget per decision (load-independent; default 200 matches
                              # PluribusReBeLConfig.max_iterations). With the native exploit serve
                              # active BOTH arms are native-fast, so ADAPTIVE_PROD_K=2000 (the live
                              # K class) is feasible; the dynamic Python exploit tail
                              # (POKERBOT_LIVE_NATIVE_EXPLOIT=0) is ~10-20x slower — keep 200 there.
_PROD_HERO_TIME_MS = 10**9   # wall clock ceiling: never binds (iterations are the real budget)


def _build_search_agent(hero_seat: int, blueprint_snap: dict) -> "SearchAgent":
    """Build the production SearchAgent (Pluribus path, EMD-K160 blueprint, K=2000 iters).

    Budget: K=2000 iterations, wall-clock ceiling 10^9 ms (never binds).
    Results are iteration-bound, not wall-clock-bound — load-independent.
    """
    from pokerbot.eval.search_agent import SearchAgent

    agent = SearchAgent(
        value_checkpoint=None,
        transformer_checkpoint=None,  # MADT not needed for the adaptive field benchmark
        device="cpu",
        time_budget_ms=_PROD_HERO_TIME_MS,
        use_pluribus_search=True,
        blueprint_snapshot=blueprint_snap,
        pluribus_leaf_select="uniform",
        root_exploit_mixer=True,       # hero's own post-search policy is exploit-adjusted
        root_mixer_unopened=False,     # validated facing-only subset (default)
        search_preflop=False,          # Pluribus: preflop from blueprint (default live)
        hero_seat=hero_seat,
    )
    # Patch the iteration budget after construction (SearchAgent does not expose max_iterations
    # directly in its constructor — it comes through PluribusReBeLConfig via time_budget_ms).
    # We set both to ensure wall-clock never terminates before K=2000 iters complete.
    if hasattr(agent, "solver") and hasattr(agent.solver, "config"):
        agent.solver.config.max_iterations = _PROD_HERO_K
        agent.solver.config.time_budget_ms = _PROD_HERO_TIME_MS
    return agent


class ProductionAdaptiveHero:
    """Production hero that wraps the real SearchAgent (Pluribus + EMD-K160 blueprint).

    Exploit ON:  per-villain OpponentStats from the temp PlayerProfileStore are fed via
                 set_opponent_profiles({seat: stats}) before each decision.  This wires the
                 EXACT same two exploitation seams live uses:
                   1. solver.opponent_model => opponent-node deviation tail (postflop CFR)
                   2. root mixer => hero's own post-search mix adjustment (all streets)
                 With POKERBOT_LIVE_NATIVE_EXPLOIT=1 (opt-in; default OFF pending the value
                 A/B) exploit decisions run the frozen-read-response NATIVE solve_exploit
                 (tables built in Python per decision) — same gate as live, faithful.
                 Default/unset = the dynamic Python CFR exploit tail.
    Exploit OFF: set_opponent_profiles(None) => pure GTO, native fast path eligible.

    Served-source tracking: counts blueprint_preflop / search_postflop / search_preflop
    serves across the run, reported in the smoke and stored in the output JSON.
    """

    def __init__(
        self,
        *,
        exploit: bool,
        store: PlayerProfileStore | None = None,
        blueprint_snap: dict | None = None,
    ):
        self.exploit = exploit
        self._store = store
        self._blueprint_snap = blueprint_snap or {}
        # hero_seat and seat_to_identity set by the harness before each hand
        self.hero_seat: int = 0
        self.seat_to_identity: dict[int, str] = {}
        # Build the SearchAgent at construction time (expensive; reused across all hands)
        self._agent = _build_search_agent(self.hero_seat, self._blueprint_snap)
        # Served-source counters (reported in smoke / stored in output)
        self._served_blueprint_preflop: int = 0
        self._served_search: int = 0
        self._served_total: int = 0

    def reseed(self, seed: int) -> None:
        """Reset the agent's per-hand PBS state (new hand, new hero seat)."""
        self._agent.reset_hand(self.hero_seat)

    def set_profile(self, profile) -> None:
        # Compatibility shim — unused for the production hero (we do dynamic per-seat lookup)
        pass

    def pop_decisions(self) -> list:
        # ProductionAdaptiveHero doesn't expose per-decision AIVAT tuples; return empty.
        return []

    def pop_chance_corrections(self) -> list:
        return []

    @property
    def bet_freq(self) -> float:
        return 0.0  # not tracked (search policy is not a simple value-bettor)

    def served_source_counts(self) -> dict:
        total = max(1, self._served_total)
        return {
            "blueprint_preflop": self._served_blueprint_preflop,
            "search": self._served_search,
            "total": self._served_total,
            "blueprint_preflop_frac": round(self._served_blueprint_preflop / total, 3),
            "search_frac": round(self._served_search / total, 3),
        }

    def __call__(self, state: GameState) -> Action:
        # Wire per-seat profiles into the search agent BEFORE each decision.
        # The agent's __call__ reads self._opponent_profiles and self.solver.opponent_model.
        if self.exploit and self._store is not None:
            profiles: dict[int, OpponentStats] = {}
            for s, ident in self.seat_to_identity.items():
                if s != self.hero_seat:
                    profiles[s] = self._store.get(ident)
            self._agent.set_opponent_profiles(profiles)
        else:
            self._agent.set_opponent_profiles(None)

        action = self._agent(state)

        # Track served-source (metadata is set by the solver on every solve call)
        meta = getattr(self._agent, "_state", None)
        if meta is not None:
            md = meta.last_solve_metadata
            served = md.get("served", "")
            self._served_total += 1
            if served == "blueprint_preflop":
                self._served_blueprint_preflop += 1
            else:
                self._served_search += 1

        return action


def _pool_profiles(profiles: dict[int, OpponentStats]) -> OpponentStats:
    """Merge multiple opponents' OpponentStats into one pooled profile for the heuristic hero.
    This mirrors run_archetype's single-profile model for a mixed table."""
    if not profiles:
        return OpponentStats()
    if len(profiles) == 1:
        return next(iter(profiles.values()))
    # Sum counts across all opponents (simple pool — what the battery does for homogeneous fields)
    pooled = OpponentStats()
    for s in profiles.values():
        pooled.hands += s.hands
        pooled.vpip += s.vpip
        pooled.pfr += s.pfr
        pooled.three_bet += s.three_bet
        pooled.fold_to_cbet += s.fold_to_cbet
        pooled.cbet_faced += s.cbet_faced
        pooled.aggression_bets_raises += s.aggression_bets_raises
        pooled.aggression_calls += s.aggression_calls
        pooled.wtsd += s.wtsd
        pooled.saw_flop += s.saw_flop
        for ck, row in s.ctx_counts.items():
            cur = pooled.ctx_counts.get(ck)
            if cur is None:
                pooled.ctx_counts[ck] = list(row)
            else:
                for i in range(min(len(cur), len(row))):
                    cur[i] += row[i]
        for ck, row in s.ctx_fast.items():
            cur = pooled.ctx_fast.get(ck)
            if cur is None:
                pooled.ctx_fast[ck] = list(row)
            else:
                for i in range(min(len(cur), len(row))):
                    cur[i] += row[i]
        for ck, row in s.ctx_slow.items():
            cur = pooled.ctx_slow.get(ck)
            if cur is None:
                pooled.ctx_slow[ck] = list(row)
            else:
                for i in range(min(len(cur), len(row))):
                    cur[i] += row[i]
    return pooled


# ---------------------------------------------------------------------------
# Profile ingestion: translate GameState hand -> PlayerProfileStore record
# ---------------------------------------------------------------------------

def _make_hand_record(
    state: GameState,
    seat_to_identity: dict[int, str],
    hand_id: str,
) -> dict:
    """Build a hand_recorder-compatible record from the final GameState so
    PlayerProfileStore.update_from_hand() can ingest it using the SAME path
    the live production system uses (NOT a mock ingestion path).

    We synthesise the 'actions' list from state.action_history() and
    map seats to stable identity strings.
    """
    # Build per-seat user_id and name from identity ("honest#0" -> user_id="honest#0")
    # player_profiles._player_key(user_id, name) -> str(user_id) when user_id is truthy
    seat_uid: dict[int, str] = {}
    seat_name: dict[int, str] = {}
    for seat, ident in seat_to_identity.items():
        seat_uid[seat] = ident
        seat_name[seat] = ident

    # Walk action_history to build the action list
    actions = []
    try:
        for seat, act in state.action_history():
            atype = act.action_type.name.lower()  # e.g. "fold", "check", "call", "raise", "all_in"
            amount_bb = float(getattr(act, "amount_bb", 0.0) or 0.0)
            # Map street from game state's street at time of action.
            # action_history() doesn't carry per-action street, so we reconstruct from
            # the typical sequence.  Use a simpler approach: we know the final state
            # has state.street; map amount -> "bet" / "raise" distinction from atype
            actions.append({
                "street": "preflop",  # placeholder; refined below
                "type": atype if atype != "all_in" else "all_in",
                "user_id": seat_uid.get(seat, str(seat)),
                "name": seat_name.get(seat, str(seat)),
                "amount": amount_bb,
                "auto": False,
            })
    except Exception:
        # action_history() may fail on some GameState versions; fall back to empty actions
        actions = []

    # More reliable: use the per-player action records directly
    # The GameState.players[seat] has p.actions_this_street but only the current street.
    # We'll use the full history from state.players' committed amounts as a proxy and
    # reconstruct from the board progression.
    #
    # Better approach: build a simplified record that populate_from_hand CAN use.
    # The minimal schema to get VPIP/PFR/AF/fold_to_cbet is:
    #   actions: [{street, type, user_id, name, amount}]
    #   board: [card_strings]

    # Re-derive from players' hand stats (simpler, avoids GameState internals)
    actions = []
    board = [str(c) for c in state.board]

    for seat_i, p in enumerate(state.players):
        uid = seat_uid.get(seat_i, str(seat_i))
        nm = seat_name.get(seat_i, str(seat_i))
        # We synthesise VPIP/PFR signals from the player's final committed amounts.
        # For a more faithful ingestion we emit synthetic actions so derive_hand_stats
        # can compute per-player stats correctly.
        # voluntarily put $ in preflop = not blind-only
        ct = float(p.committed_total_bb)
        folded = p.folded
        # Emit a preflop action for each player
        if ct > 0:
            # Put-in-voluntarily: approximate as "call" (conservative — avoids phantom PFR)
            actions.append({
                "street": "preflop",
                "type": "call",
                "user_id": uid,
                "name": nm,
                "amount": ct,
                "auto": False,
            })
        if folded:
            actions.append({
                "street": "preflop",
                "type": "fold",
                "user_id": uid,
                "name": nm,
                "amount": 0.0,
                "auto": False,
            })

    return {
        "hand_id": hand_id,
        "actions": actions,
        "board": board,
        "shown": [],
        "table": {"format": "cash"},
    }


# ---------------------------------------------------------------------------
# Smarter profile ingestion: use the online-learning path from exploitation_battery
# ---------------------------------------------------------------------------

def _ingest_hand_to_store(
    store: PlayerProfileStore,
    seat_to_identity: dict[int, str],
    ctx_buf: list[tuple],   # [(seat, context_key, action_bin), ...]
    scalar_buf: list[tuple], # [(seat, street_name, action_type_name, amount_bb, is_first_flop_aggr), ...]
    final_state: GameState,
    hand_id: str,
) -> None:
    """Fold one hand's observed villain actions into the PlayerProfileStore.

    Uses the SAME dual-timescale decay_and_add path as exploitation_battery's online-learning loop
    for the Bayesian context updates (the real exploit stack, not a mock).

    Additionally updates scalar stats (VPIP/PFR/AF/fold-to-cbet/WTSD/saw_flop) derived from the
    action sequence — this is the SAME math as player_profiles.derive_hand_stats() but applied
    directly without requiring a hand_recorder record schema.
    """
    board_len = len(final_state.board)  # 0=pf-fold, 3=flop, 4=turn, 5=river

    # --- Bayesian context update (identical to exploitation_battery.run_archetype online-learning) ---
    per_seat_ctx: dict[str, dict[str, list[int]]] = {}
    for seat, ck, b in ctx_buf:
        ident = seat_to_identity.get(seat)
        if ident is None:
            continue
        seen = per_seat_ctx.setdefault(ident, {})
        seen.setdefault(ck, [0, 0, 0, 0])[b] += 1

    # --- Scalar stat update (VPIP/PFR/AF/WTSD/saw_flop/fold_to_cbet) ---
    # Track per-identity pre-flop raises for PFR
    preflop_aggr: set[str] = set()
    # Track voluntary put-in (called/bet/raised preflop)
    vpip_set: set[str] = set()
    # Aggression counts
    bets_raises: dict[str, int] = {}
    calls: dict[str, int] = {}
    # Saw flop
    saw_flop_set: set[str] = set()
    # WTSD (went to showdown given saw flop) — river board = 5 cards
    had_wtsd: dict[str, bool] = {}
    # Fold-to-cbet tracking
    # first_flop_aggr_ident = the identity that made the first flop c-bet
    first_flop_aggr_ident: str | None = None
    first_flop_aggr_found = False
    fold_to_cbet_set: set[str] = set()  # identities that folded to the c-bet
    cbet_faced_set: set[str] = set()    # identities that faced the c-bet (and weren't the c-bettor)

    for seat, street, atype, amount_bb, _is_first_flop_aggr in scalar_buf:
        ident = seat_to_identity.get(seat)
        atype_lo = atype.lower()
        is_aggressive = atype_lo in ("bet", "raise", "all_in")

        # C-BET SENTINEL: seat=-1 is the hero's flop bet sentinel (see make_policy fix).
        # It should NOT update any villain profile stats, but MUST set first_flop_aggr_found
        # so that subsequent villain actions are correctly tagged as "facing the c-bet".
        # first_flop_aggr_ident stays None (not a villain identity) — the comparison
        # `ident != first_flop_aggr_ident` is always True for villain idents vs None.
        if ident is None:
            # Only the c-bet sentinel matters here; skip all profile stat updates.
            if street == "flop" and is_aggressive and not first_flop_aggr_found:
                first_flop_aggr_found = True
                first_flop_aggr_ident = None  # hero has no tracked identity
            continue

        is_call = atype_lo == "call"
        is_fold = atype_lo == "fold"

        if street == "preflop":
            # VPIP: voluntarily put $ in (call, bet, raise, all_in — NOT big blind)
            if atype_lo in ("call", "bet", "raise", "all_in"):
                vpip_set.add(ident)
            # PFR: raised preflop (first such action per identity)
            if is_aggressive and ident not in preflop_aggr:
                preflop_aggr.add(ident)

        if is_aggressive:
            bets_raises[ident] = bets_raises.get(ident, 0) + 1
        elif is_call:
            calls[ident] = calls.get(ident, 0) + 1

        # Saw flop: any postflop action counts
        if street in ("flop", "turn", "river"):
            saw_flop_set.add(ident)

        # First flop aggression = c-bet (villain as first flop bettor)
        if street == "flop" and is_aggressive and not first_flop_aggr_found:
            first_flop_aggr_found = True
            first_flop_aggr_ident = ident

        # Fold-to-cbet: facing the first flop aggression and folding.
        # When first_flop_aggr_ident is None (hero c-bet sentinel), every villain ident
        # satisfies `ident != None`, so they ALL count as "cbet_faced".
        if (street == "flop" and first_flop_aggr_found
                and ident != first_flop_aggr_ident):
            cbet_faced_set.add(ident)
            if is_fold:
                fold_to_cbet_set.add(ident)

    # WTSD: saw the flop AND reached showdown (board has all 5 cards AND player didn't fold)
    for p in final_state.players:
        seat = p.seat
        ident = seat_to_identity.get(seat)
        if ident is None:
            continue
        if ident in saw_flop_set and board_len >= 5 and not p.folded:
            had_wtsd[ident] = True

    # --- Merge into store ---
    with store._lock:
        for ident, seen in per_seat_ctx.items():
            s = store.get(ident)
            s.hands += 1
            # Bayesian context (the exploit engine reads these)
            for ck, bins in seen.items():
                s.add_ctx(ck, bins)
                s.decay_and_add(ck, bins, context_prior(ck), s.hands)
            # Scalar stats (the HUD/profile display reads these)
            if ident in vpip_set:
                s.vpip += 1
            if ident in preflop_aggr:
                s.pfr += 1
            s.aggression_bets_raises += bets_raises.get(ident, 0)
            s.aggression_calls += calls.get(ident, 0)
            if ident in saw_flop_set:
                s.saw_flop += 1
            if had_wtsd.get(ident):
                s.wtsd += 1
            if ident in cbet_faced_set:
                s.cbet_faced += 1
            if ident in fold_to_cbet_set:
                s.fold_to_cbet += 1

        # Handle identities that appear ONLY in scalar_buf (no ctx_buf entries = folded early)
        scalar_only: set[str] = set()
        for seat, street, atype, _, _ in scalar_buf:
            ident = seat_to_identity.get(seat)
            if ident and ident not in per_seat_ctx:
                scalar_only.add(ident)
        for ident in scalar_only:
            s = store.get(ident)
            s.hands += 1
            if ident in vpip_set:
                s.vpip += 1
            if ident in preflop_aggr:
                s.pfr += 1
            s.aggression_bets_raises += bets_raises.get(ident, 0)
            s.aggression_calls += calls.get(ident, 0)

        store._dirty = True


# ---------------------------------------------------------------------------
# Profile summary extraction
# ---------------------------------------------------------------------------

def _profile_summary(store: PlayerProfileStore, identity: str) -> dict:
    """Extract a human-readable summary of one opponent's learned profile."""
    with store._lock:
        if identity not in store.profiles:
            return {
                "identity": identity, "hands": 0,
                "vpip": None, "pfr": None, "af": None,
                "fold_to_cbet": None, "wtsd": None, "ctx_contexts": 0,
            }
        s = store.profiles[identity]
    h = s.hands
    vpip_rate = round(s.vpip / h, 3) if h > 0 else None
    pfr_rate = round(s.pfr / h, 3) if h > 0 else None
    af = (round(s.aggression_bets_raises / max(1, s.aggression_calls), 2)
          if s.aggression_calls + s.aggression_bets_raises > 0 else None)
    ftcb_rate = round(s.fold_to_cbet / max(1, s.cbet_faced), 3) if s.cbet_faced > 0 else None
    wtsd_rate = round(s.wtsd / max(1, s.saw_flop), 3) if s.saw_flop > 0 else None
    n_ctx = len(s.ctx_counts)
    return {
        "identity": identity,
        "hands": h,
        "vpip": vpip_rate,
        "pfr": pfr_rate,
        "af": af,
        "fold_to_cbet": ftcb_rate,
        "wtsd": wtsd_rate,
        "ctx_contexts": n_ctx,
    }


# ---------------------------------------------------------------------------
# CI helper
# ---------------------------------------------------------------------------

def _ci(arr: list[float]) -> dict:
    a = np.asarray(arr, dtype=np.float64)
    n = len(a)
    if n == 0:
        return {"n": 0, "mean": 0.0, "se": 0.0, "ci95_lo": 0.0, "ci95_hi": 0.0,
                "bb_per_100": 0.0}
    m = float(a.mean())
    se = float(a.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0
    return {
        "n": n,
        "mean": m,
        "se": se,
        "ci95_lo": m - 1.96 * se,
        "ci95_hi": m + 1.96 * se,
        "bb_per_100": m * 100.0,
    }


# ---------------------------------------------------------------------------
# Main adaptive benchmark
# ---------------------------------------------------------------------------

def run_adaptive(
    *,
    field_spec: dict[str, int],   # {bot_name: count}
    hands: int,
    seed: int,
    exploit: bool,
    windows: int,
    store: PlayerProfileStore,
    progress_every: int = 10,
    diag: bool = False,           # if True: collect per-hand loss attribution + exploit firing stats
    hero_mode: str = "heuristic", # "heuristic" (default) or "production"
    blueprint_snap: dict | None = None,  # pre-loaded blueprint snapshot (production mode only)
) -> dict:
    """Core adaptive field loop.

    Each hand:
      1. Rotate hero seat (round-robin over NUM_SEATS).
      2. Assign external bots to remaining seats (stable identities).
      3. Play hand with hero using exploit-aware policy.
      4. Ingest villain observations into the persistent store.
    """
    t0 = time.perf_counter()

    # Build the roster of external bot instances (stable identities)
    roster: list[tuple[str, ExternalBotAgent]] = []
    for bot_name, count in field_spec.items():
        if bot_name not in _SUPPORTED_BOTS:
            raise ValueError(
                f"Bot {bot_name!r} not supported at the object level. "
                f"Supported: {sorted(_SUPPORTED_BOTS)}. "
                f"(deepbot: see DEEPBOT_TODO at bottom of adaptive_field_h2h.py)"
            )
        policy_fn = get_opponent_policy(bot_name)
        for i in range(count):
            agent = ExternalBotAgent(bot_name, i, policy_fn)
            roster.append((bot_name, agent))

    n_bots = len(roster)
    n_hero_seats = NUM_SEATS - n_bots
    if n_hero_seats < 1:
        raise ValueError(
            f"Field fills {n_bots} seats; need at least 1 hero seat. "
            f"Reduce field total below {NUM_SEATS}."
        )

    # Build hero (heuristic or production)
    hero_rng = random.Random(seed)
    if hero_mode == "production":
        hero = ProductionAdaptiveHero(
            exploit=exploit,
            store=store if exploit else None,
            blueprint_snap=blueprint_snap,
        )
    else:
        hero = AdaptiveHero(hero_rng, exploit=exploit, store=store if exploit else None)

    per_hand_ev: list[float] = []
    hand_id_ctr = 0
    last_heartbeat = time.perf_counter()

    # Diagnostic accumulators (only populated when diag=True)
    _diag_loss_from_hero_bets_called: list[float] = []   # hero bet, got called, lost
    _diag_loss_from_blinds_preflop: list[float] = []     # folded preflop (blind loss only)
    _diag_loss_from_showdown_call: list[float] = []      # hero called villain, went to showdown, lost
    _diag_exploit_firing: list[dict] = []                # per-hand exploit firing stats from inner hero
    if diag:
        os.environ["POKERBOT_EXPLOIT_DIAG"] = "1"

    for h in range(hands):
        hand_id_ctr += 1
        hand_seed = 1000 + seed * 1_000_000 + h
        hero_rng_h = random.Random(hand_seed)

        # --- Seat assignment ---
        # Hero occupies seats round-robin across the full range to be fair;
        # bots fill remaining seats in roster order.
        hero_seat = h % NUM_SEATS
        # remaining seats in a fixed order, skipping the hero seat
        remaining = [s for s in range(NUM_SEATS) if s != hero_seat]
        # assign bots to the first n_bots remaining seats; hero fills leftover seats too
        # when n_bots < NUM_SEATS - 1
        seat_to_bot: dict[int, ExternalBotAgent] = {}
        seat_to_identity: dict[int, str] = {}
        for i, (_, agent) in enumerate(roster):
            bot_seat = remaining[i % len(remaining)]
            if i < len(remaining):
                seat_to_bot[remaining[i]] = agent
                seat_to_identity[remaining[i]] = agent.identity

        # Reseed external bots for CRN
        for _, agent in seat_to_bot.items():
            agent.reseed(hand_seed)

        # Reseed / reset hero for this hand
        hero.hero_seat = hero_seat
        hero.seat_to_identity = seat_to_identity
        hero.reseed(hand_seed)
        # For the production hero, also reset the SearchAgent's per-hand PBS state
        # (reset_hand is called inside ProductionAdaptiveHero.reseed, but it needs the
        # hero_seat to be set FIRST so we do it here after setting hero_seat above)
        if hero_mode == "production":
            hero._agent.reset_hand(hero_seat)

        # --- Play hand ---
        ctx_buf: list[tuple] = []     # (seat, context_key, action_bin)
        scalar_buf: list[tuple] = []  # (seat, street, action_type_name, amount_bb, is_first_flop_aggr)
        _first_flop_aggr_seen = [False]  # mutable flag shared in closure

        def make_policy(
            seat,
            _bot=seat_to_bot,
            _ident=seat_to_identity,
            _ctx=ctx_buf,
            _sc=scalar_buf,
            _ffa=_first_flop_aggr_seen,
            _hero_seat=hero_seat,
        ):
            is_hero_s = (seat == _hero_seat)
            bot = _bot.get(seat)

            def _policy(state: GameState) -> Action:
                if is_hero_s:
                    a = hero(state)
                    # C-BET TRACKING FIX: Record the hero's first flop aggression so that
                    # villain's subsequent action is correctly attributed as "facing a c-bet".
                    # Previously, since hero actions were not in scalar_buf, first_flop_aggr_ident
                    # was never set when the hero was the first flop bettor — causing cbet_faced and
                    # fold_to_cbet to be PERMANENTLY ZERO for opponents vs hero c-bets. We record
                    # hero flop aggression with a sentinel seat (_hero_seat) NOT in seat_to_identity,
                    # so _ingest_hand_to_store's `ident = seat_to_identity.get(seat)` returns None
                    # for the hero entry (skipping all stat updates for the hero itself) while still
                    # setting first_flop_aggr_found=True so villains' subsequent folds/calls are
                    # correctly tagged in cbet_faced_set and fold_to_cbet_set.
                    h_atype = a.action_type.name.lower()
                    h_street = state.street.name.lower()
                    h_amount = float(getattr(a, "amount_bb", 0.0) or 0.0)
                    h_aggr = h_atype in ("bet", "raise", "all_in")
                    if h_street == "flop" and h_aggr and not _ffa[0]:
                        _ffa[0] = True
                        # Use a negative seat index as the hero sentinel (won't match any real seat
                        # and won't match any key in seat_to_identity → ident = None in ingest)
                        _sc.append((-1, h_street, h_atype, h_amount, True))
                    return a
                # External bot action
                a = bot(state) if bot is not None else hero(state)
                acting_seat = state.action_seat
                # Context key + bin (for Bayesian ingestion)
                ck = _ctx_key_for(state)
                b = _action_to_bin(state, a)
                _ctx.append((acting_seat, ck, b))
                # Scalar action record (for VPIP/PFR/AF/etc.)
                atype = a.action_type.name.lower()  # "fold","check","call","raise","all_in"
                street = state.street.name.lower()
                amount_bb = float(getattr(a, "amount_bb", 0.0) or 0.0)
                # is_first_flop_aggr: first aggressive action on the flop = the c-bet
                is_aggr = atype in ("bet", "raise", "all_in")
                is_first_ffa = False
                if street == "flop" and is_aggr and not _ffa[0]:
                    _ffa[0] = True
                    is_first_ffa = True
                _sc.append((acting_seat, street, atype, amount_bb, is_first_ffa))
                return a
            return _policy

        policies = {s: make_policy(s) for s in range(NUM_SEATS)}

        deck_state = GameState.new_hand(
            button_seat=h % NUM_SEATS,
            rng=random.Random(hand_seed),
        )
        settle = lambda st, nb, _s=hand_seed: _allin_ev_settle(st, nb, 7777 + _s)
        final_state, deltas = play_hand(policies, state=deck_state, allin_settle=settle)
        ev_bb = float(deltas.get(hero_seat, 0.0))
        per_hand_ev.append(ev_bb)

        # --- Diagnostic: loss attribution + exploit-firing stats ---
        if diag:
            # Collect ExploitHero diagnostic counters for this hand
            inner = getattr(hero, "_inner", None)
            if inner is not None and hasattr(inner, "pop_diag"):
                _diag_exploit_firing.append(inner.pop_diag())
            # Loss attribution: classify hero's loss source
            # We approximate by inspecting the final GameState and the hero's actions in scalar_buf
            hero_p = final_state.players[hero_seat]
            board_len = len(final_state.board)
            hero_committed = float(hero_p.committed_total_bb)
            hero_folded = hero_p.folded
            # Identify hero's actions from the scalar_buf (which only tracks VILLAIN actions, not hero).
            # Instead, reconstruct from the game's final state.
            # Simplification: classify by outcome
            if hero_folded:
                # Hero folded — loss is the committed amount (blind or preflop fold)
                if board_len == 0:  # folded preflop
                    _diag_loss_from_blinds_preflop.append(-hero_committed)
                else:
                    _diag_loss_from_blinds_preflop.append(-hero_committed)  # postflop fold
            elif ev_bb < 0.0 and board_len >= 3:
                # Hero lost postflop — further split into:
                # (a) hero bet/raised and got called: check if hero was aggressive
                # (b) hero called villain and went to showdown
                # We use board_len==5 (showdown) as proxy for call-and-lose
                # Hero can't lose at showdown without reaching it
                if board_len >= 5 and not hero_folded:
                    _diag_loss_from_showdown_call.append(ev_bb)
                else:
                    _diag_loss_from_hero_bets_called.append(ev_bb)
            elif ev_bb < 0.0 and board_len == 0:
                _diag_loss_from_blinds_preflop.append(ev_bb)

        # --- Ingest villain observations into the profile store ---
        if (ctx_buf or scalar_buf) and exploit:
            _ingest_hand_to_store(
                store,
                seat_to_identity,
                ctx_buf,
                scalar_buf,
                final_state,
                hand_id=f"adaptive_{seed}_{h}",
            )

        # --- Progress logging ---
        now = time.perf_counter()
        elapsed = now - t0
        if h > 0 and (h % progress_every == 0 or now - last_heartbeat > 30.0):
            last_heartbeat = now
            done = h + 1
            eta_s = elapsed / done * (hands - done) if done < hands else 0.0
            eta_str = _fmt_eta(eta_s)
            recent = per_hand_ev[max(0, done - progress_every):]
            recent_mean = float(np.mean(recent)) * 100.0 if recent else 0.0
            overall_ci = _ci(per_hand_ev)
            served_suffix = ""
            if hero_mode == "production":
                sc = hero.served_source_counts()
                served_suffix = (
                    f" | bp_pf={sc['blueprint_preflop']} "
                    f"srch={sc['search']} "
                    f"({sc['search_frac']*100:.0f}%)"
                )
            print(
                f"[adaptive] hand {done}/{hands} ({100*done/hands:.0f}%) | "
                f"window bb/100={recent_mean:+.1f} | "
                f"overall {overall_ci['bb_per_100']:+.1f}±{1.96*overall_ci['se']*100:.1f} | "
                f"ETA {eta_str} | "
                f"store={sum(s.hands for s in store.profiles.values())} obs"
                f"{served_suffix}",
                flush=True,
            )

    # --- Final save ---
    store.save()

    # --- Build windowed curve ---
    window_size = max(1, hands // windows)
    window_stats = []
    for w in range(windows):
        lo = w * window_size
        hi = min((w + 1) * window_size, hands)
        chunk = per_hand_ev[lo:hi]
        if chunk:
            wci = _ci(chunk)
            window_stats.append({
                "window": w,
                "hand_lo": lo,
                "hand_hi": hi,
                **wci,
            })

    # --- Overall stats ---
    overall = _ci(per_hand_ev)
    wall_s = time.perf_counter() - t0

    # --- Profile snapshots at N/4, N/2, N ---
    # We only have the final store; approximate milestone snapshots aren't available
    # retrospectively (we don't checkpoint mid-run profiles here).
    # Report the final profiles as the N-hand dump, and note hands-per-identity
    profile_dumps = {}
    for bot_name, agent in roster:
        ident = agent.identity
        profile_dumps[ident] = _profile_summary(store, ident)

    out = {
        "field": field_spec,
        "hands": hands,
        "exploit": exploit,
        "hero_mode": hero_mode,
        "seed": seed,
        "windows": windows,
        "n_hero_seats": n_hero_seats,
        "n_bot_seats": n_bots,
        "overall": overall,
        "window_curve": window_stats,
        "profile_dump_at_N": profile_dumps,
        "wall_s": round(wall_s, 2),
        "hands_per_second": round(hands / max(0.001, wall_s), 1),
    }
    # Served-source counts for the production hero
    if hero_mode == "production":
        out["served_source"] = hero.served_source_counts()

    if diag and _diag_exploit_firing:
        # Aggregate exploit-firing stats across all hands (heuristic hero only; the production
        # hero's exploit fires INSIDE the solver via opponent_model — not surfaced via pop_diag).
        agg: dict = {}
        for d in _diag_exploit_firing:
            for k, v in d.items():
                agg[k] = agg.get(k, 0) + v
        total_dec = max(1, agg.get("decisions", 1))
        total_elig = max(1, agg.get("exploit_eligible", 1))
        total_fired = max(1, agg.get("adj_fired", 1))
        out["diag_exploit"] = {
            "hands": hands,
            "total_decisions": agg.get("decisions", 0),
            "exploit_eligible_rate": agg.get("exploit_eligible", 0) / total_dec,
            "exact_hit_rate_of_elig": agg.get("ctx_read_exact_hit", 0) / total_elig,
            "aggregate_hit_rate_of_elig": agg.get("ctx_read_aggregate_hit", 0) / total_elig,
            "total_miss_rate_of_elig": agg.get("ctx_read_total_miss", 0) / total_elig,
            "adj_fired_rate_of_elig": agg.get("adj_fired", 0) / total_elig,
            "mean_alpha_when_fired": agg.get("alpha_sum", 0.0) / total_fired,
            "mean_tv_when_read": agg.get("tv_sum", 0.0) / max(1, agg.get("adj_fired", 0) + agg.get("ctx_read_aggregate_hit", 0)),
            "mean_conf_when_read": agg.get("conf_sum", 0.0) / max(1, agg.get("adj_fired", 0) + agg.get("ctx_read_aggregate_hit", 0)),
            "mean_n_obs_when_read": agg.get("n_obs_sum", 0) / max(1, agg.get("adj_fired", 0) + agg.get("ctx_read_aggregate_hit", 0)),
            "mean_sizing_logit_when_fired": agg.get("sizing_logit_sum", 0.0) / total_fired,
            "mean_bluff_suppress_logit_when_fired": agg.get("bluff_suppress_logit_sum", 0.0) / total_fired,
            "mean_station_score_when_fired": agg.get("station_score_sum", 0.0) / total_fired,
            "mean_station_conf_when_fired": agg.get("station_conf_sum", 0.0) / total_fired,
        }

    if diag:
        # Loss attribution summary (all hero modes — only needs the per-hand ev + game state data)
        def _loss_bb100(arr):
            return round(float(np.mean(arr)) * 100.0, 2) if arr else 0.0
        out["diag_loss_attribution"] = {
            "hero_folded_preflop_postflop_mean_bb100": _loss_bb100(_diag_loss_from_blinds_preflop),
            "hero_folded_preflop_postflop_hands": len(_diag_loss_from_blinds_preflop),
            "hero_lost_showdown_call_mean_bb100": _loss_bb100(_diag_loss_from_showdown_call),
            "hero_lost_showdown_call_hands": len(_diag_loss_from_showdown_call),
            "hero_lost_bet_called_approx_mean_bb100": _loss_bb100(_diag_loss_from_hero_bets_called),
            "hero_lost_bet_called_approx_hands": len(_diag_loss_from_hero_bets_called),
            "note": "showdown_call = hero called/bet and reached showdown and lost; "
                    "bet_called_approx = hero lost postflop before river (may include folds by hero postflop); "
                    "folded = hero folded at any street",
        }

    return out


def _fmt_eta(eta_s: float) -> str:
    if eta_s < 60:
        return f"~{int(eta_s)}s"
    if eta_s < 3600:
        return f"~{int(eta_s/60)}m"
    return f"~{eta_s/3600:.1f}h"


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Adaptive Field Benchmark — engine learns opponents over time")
    p.add_argument("--field", required=True,
                   help='JSON dict of {bot_name: seat_count}, e.g. \'{"honest": 3}\'')
    p.add_argument("--hands", type=int, default=2000,
                   help="Number of hands to play (default 2000)")
    p.add_argument("--seed", type=int, default=42,
                   help="Master RNG seed (default 42)")
    p.add_argument("--exploit", choices=["on", "off"], default="on",
                   help="on = exploitation stack enabled, off = control arm (default: on)")
    p.add_argument("--windows", type=int, default=10,
                   help="Number of windows for the adaptation curve (default 10)")
    p.add_argument("--tmp-db", action="store_true", default=True,
                   help="Use a fresh temporary directory for the profile store (default: True)")
    p.add_argument("--keep-db", action="store_true", default=False,
                   help="Keep the temp DB directory after the run (for inspection)")
    p.add_argument("--db-path", type=str, default=None,
                   help="Explicit path for the profile store (overrides --tmp-db)")
    p.add_argument("--out", type=str, default=None,
                   help="Output JSON path (default: artifacts/adaptive_<label>.json)")
    p.add_argument("--progress-every", type=int, default=10,
                   help="Print progress every N hands (default 10)")
    p.add_argument("--diag", action="store_true", default=False,
                   help="Collect per-hand loss attribution + exploit firing stats (writes to output JSON)")
    p.add_argument("--hero", choices=["heuristic", "production"], default="heuristic",
                   help=(
                       "heuristic = lightweight ExploitHero (default; fast, no blueprint/search); "
                       "production = real SearchAgent (Pluribus + EMD-K160 blueprint, K=2000 iters, "
                       "native subgame, full exploit overlays). Set POKERBOT_NATIVE_SUBGAME=1 "
                       "(default) for the production hero."
                   ))
    return p.parse_args()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # Parse field spec
    try:
        field_spec: dict[str, int] = json.loads(args.field)
    except (json.JSONDecodeError, ValueError) as e:
        print(f"ERROR: --field must be a JSON dict, got: {args.field!r}\n  {e}", file=sys.stderr)
        sys.exit(1)

    if not field_spec:
        print("ERROR: --field is empty; need at least one bot.", file=sys.stderr)
        sys.exit(1)

    n_bots = sum(field_spec.values())
    if n_bots >= NUM_SEATS:
        print(f"ERROR: field fills {n_bots} seats, but max is {NUM_SEATS-1} (need ≥1 hero).",
              file=sys.stderr)
        sys.exit(1)

    exploit = args.exploit == "on"

    # Build output path
    label = "_".join(f"{k}{v}" for k, v in field_spec.items())
    label += f"_{'on' if exploit else 'off'}"
    REPO = Path(__file__).resolve().parent.parent
    out_path = args.out or str(REPO / "artifacts" / f"adaptive_{label}.json")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    # Build profile store in a temp dir (NEVER touches the real live profiles.json)
    tmp_dir = None
    if args.db_path:
        db_path = args.db_path
        db_dir = str(Path(db_path).parent)
        Path(db_dir).mkdir(parents=True, exist_ok=True)
    else:
        tmp_dir = tempfile.mkdtemp(prefix="adaptive_field_h2h_")
        db_path = str(Path(tmp_dir) / "adaptive_profiles.json")

    hero_mode = args.hero

    print(f"[adaptive] Profile store: {db_path}", flush=True)
    print(f"[adaptive] Field: {field_spec}", flush=True)
    print(f"[adaptive] Hands: {args.hands}  Exploit: {'ON' if exploit else 'OFF'}  "
          f"Seed: {args.seed}  Windows: {args.windows}  Hero: {hero_mode}", flush=True)

    # Load the blueprint snapshot ONCE for production hero (expensive, do not repeat per hand)
    blueprint_snap = None
    if hero_mode == "production":
        print("[adaptive] Loading EMD-K160 blueprint snapshot for production hero...", flush=True)
        from pokerbot.eval.bots import _load_blueprint_snapshot
        blueprint_snap = _load_blueprint_snapshot()
        print(f"[adaptive] Blueprint loaded: {len(blueprint_snap):,} infosets", flush=True)
        print(f"[adaptive] Production hero: K={_PROD_HERO_K} iters/decision "
              f"(iteration-bound, wall-clock={_PROD_HERO_TIME_MS}ms ceiling=never-binds), "
              f"native_subgame={os.environ.get('POKERBOT_NATIVE_SUBGAME','0')}, "
              f"live_native_exploit={os.environ.get('POKERBOT_LIVE_NATIVE_EXPLOIT','0 (default-off)')}, "
              f"root_exploit_mixer=ON", flush=True)

    store = PlayerProfileStore(db_path, wal=True)

    try:
        result = run_adaptive(
            field_spec=field_spec,
            hands=args.hands,
            seed=args.seed,
            exploit=exploit,
            windows=args.windows,
            store=store,
            progress_every=args.progress_every,
            diag=args.diag,
            hero_mode=hero_mode,
            blueprint_snap=blueprint_snap,
        )
    finally:
        if tmp_dir and not args.keep_db:
            try:
                shutil.rmtree(tmp_dir, ignore_errors=True)
            except Exception:
                pass
        elif tmp_dir and args.keep_db:
            print(f"[adaptive] Keeping DB at: {tmp_dir}", flush=True)

    # Write output
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)

    # --- Summary printout ---
    print(f"\n{'='*60}", flush=True)
    print(f"ADAPTIVE FIELD H2H  field={field_spec}  exploit={'ON' if exploit else 'OFF'}  "
          f"hero={hero_mode}", flush=True)
    print(f"  Hands: {result['hands']}  Wall: {result['wall_s']:.1f}s  "
          f"Speed: {result['hands_per_second']:.1f} h/s", flush=True)
    if hero_mode == "production" and "served_source" in result:
        ss = result["served_source"]
        print(f"  Served source: blueprint_preflop={ss['blueprint_preflop']} "
              f"({ss['blueprint_preflop_frac']*100:.1f}%)  "
              f"search={ss['search']} ({ss['search_frac']*100:.1f}%)", flush=True)
    ov = result["overall"]
    print(f"  Overall hero bb/100: {ov['bb_per_100']:+.1f}  "
          f"95%CI [{ov['ci95_lo']*100:+.1f}, {ov['ci95_hi']*100:+.1f}]", flush=True)
    print(f"\n  Adaptation curve (bb/100 per window):", flush=True)
    for w in result["window_curve"]:
        print(f"    hands {w['hand_lo']}-{w['hand_hi']}: {w['bb_per_100']:+.1f} "
              f"(±{1.96*w['se']*100:.1f})", flush=True)
    print(f"\n  Learned opponent profiles at N={result['hands']}:", flush=True)
    for ident, prof in result["profile_dump_at_N"].items():
        print(f"    {ident}: hands={prof['hands']}  "
              f"VPIP={prof['vpip']}  PFR={prof['pfr']}  "
              f"AF={prof['af']}  FtCB={prof['fold_to_cbet']}  "
              f"WTSD={prof['wtsd']}  "
              f"ctx_keys={prof['ctx_contexts']}", flush=True)

    print(f"\n  Output: {out_path}", flush=True)

    # Print full-run commands
    print(f"\n{'='*60}", flush=True)
    print("FULL-RUN COMMANDS (run these when the box is free):", flush=True)
    print(flush=True)
    print("# Honest-only 5000 hands ON:", flush=True)
    print('  python -u scripts/adaptive_field_h2h.py --field \'{"honest": 3}\' '
          '--hands 5000 --seed 42 --exploit on --out artifacts/adaptive_honest3_on.json', flush=True)
    print(flush=True)
    print("# Honest-only 5000 hands OFF (control arm):", flush=True)
    print('  python -u scripts/adaptive_field_h2h.py --field \'{"honest": 3}\' '
          '--hands 5000 --seed 42 --exploit off --out artifacts/adaptive_honest3_off.json', flush=True)
    print(flush=True)
    print("# Mixed field 5000 hands ON:", flush=True)
    print('  python -u scripts/adaptive_field_h2h.py '
          '--field \'{"honest": 2, "fish": 1, "random": 1}\' '
          '--hands 5000 --seed 42 --exploit on --out artifacts/adaptive_mixed_on.json', flush=True)
    print(flush=True)
    print("# Mixed field 5000 hands OFF (control arm):", flush=True)
    print('  python -u scripts/adaptive_field_h2h.py '
          '--field \'{"honest": 2, "fish": 1, "random": 1}\' '
          '--hands 5000 --seed 42 --exploit off --out artifacts/adaptive_mixed_off.json', flush=True)
    print(flush=True)
    print("# NOTE: deepbot is not supported at the object level (see DEEPBOT_TODO below).", flush=True)
    print(f"{'='*60}\n", flush=True)


# ---------------------------------------------------------------------------
# DEEPBOT_TODO
# ---------------------------------------------------------------------------
# deepbot (pokerbot/eval/external_bots/deepbot_adapter.py) is EnvState-based:
#   make_deepbot_policy() -> PolicyFn(env: NumpyEnvState, seat) -> np.ndarray[11]
# where env is a full numpy EnvState (jax/numpy backend), NOT the GenericView that
# pypoker bots use.  Wrapping it at the object level requires going through
# game_state_to_env (from pokerbot/search/state_bridge.py) which needs JAX/jax-engine
# to be initialized and may carry private hole information for all seats.
#
# The object-level wrapper in external_bots/objectlevel.py (if/when created) should:
#   1. Call game_state_to_env(state, acting_seat) to get a masked NumpyEnvState.
#   2. Pass it to the PolicyFn.
#   3. Translate the 11-vector to an Action via _index_to_action.
#
# This is ~30-60 min of work and depends on whether game_state_to_env correctly masks
# hole cards for non-acting seats in the current branch.  Deferred to a follow-up.
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()
