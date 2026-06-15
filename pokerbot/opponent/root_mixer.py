"""Root-policy exploitation mixer — the validated read->mixer_logits->safe_mix step, base-agnostic.

This is the SAME mechanism the heuristic battery hero proved extracts CI>0 value from leaky
archetypes, applied to ANY base policy's action probabilities at the hero's own decision (the
"root"). It was developed and battle-tested as ``SearchAgent._apply_root_mixer`` over nine battery
rounds (every guard below carries its measured failure); extracted here verbatim so the eval mirror
(``SearchAgent``) and the live bridge (``LiveSearchAdvisor``) share one implementation.

Contract: no read => probs returned unchanged (safe fallback, keeps the base). The unopened branch
(tilting our own steals/value-bets from the villain's response read) is gated behind
``unopened=True`` — it lost in every configuration against the broken caricature bots, but those
violated the human range structure its priors assume; the fair-bot battery decides its fate.
"""

from __future__ import annotations

import numpy as np

# Read-gated bluff-catch thresholds (search base): fire the facing call-boost ONLY vs a villain
# whose aggression is SHOWDOWN-confirmed bluff-heavy. conf >= MIN_CONF (~3 reveals via the
# hierarchy) AND p_bluff > MIN_RATIO (above the ~0.29 population prior mean). Frequency alone never
# triggers it (the lag -1.1 lesson). The conditional lag's value-heavy bets (p_bluff ~0.14) never
# clear MIN_RATIO, so the hero defers to the search policy vs lags.
BLUFF_CATCH_MIN_CONF = 0.3
BLUFF_CATCH_MIN_RATIO = 0.35
# Unopened-branch defer gate: if the villain RESPONDS to our bets more aggressively than baseline
# by more than this (post[AGG]-pop[AGG] in the facing_bet read), skip the unopened tilt and defer
# to the search policy — trap-mode vs an aggressive responder surrenders initiative (lag -1.1).
# Passive responders (station calls / nit folds => negative excess) are well below it and keep the
# value-bet/pressure tilts that win.
AGGR_DEFER_EXCESS = 0.08


def _select_profile(profiles: dict | None, last_aggressor):
    """Pick the profile of the villain the hero is responding to: the last aggressor (the bettor on a
    facing decision). Falls back to the first available profile when there is no aggressor or that seat
    is unprofiled. BIT-IDENTICAL to the legacy "first non-None" read whenever every seat maps to the
    same OpponentStats object (the pooled-table case); only a genuinely per-seat dict changes which
    villain is read."""
    if not profiles:
        return None
    if last_aggressor is not None:
        p = profiles.get(last_aggressor)
        if p is not None:
            return p
    return next((p for p in profiles.values() if p is not None), None)


# Value-fish gate (Phase 2): the UNOPENED value-bet tilt fires ONLY when EVERY live opponent who could
# punish a thin value-bet is a CONFIRMED sticky caller (a station). A lag (raises/punishes), a reg/nit
# (won't pay thin value), or an UNidentified villain each flip the gate OFF -> defer to the base policy.
# This is what makes the +1.4 bb/100 value-extraction do-no-harm in ANY table constellation: it self-
# disables the moment one dangerous/unknown opponent is in the pot (measured ungated leak: -0.55 bb/100
# vs a 5-lag table). The loose-passive IRC population reference is self-correcting: a reg reads TIGHTER
# than the fishy average (folds more -> excluded); a station reads looser (-> included).
FISH_MIN_HANDS = 25            # below this a villain is "unknown" -> not fish -> defer (cold-start safe)
FISH_MAX_FOLD_EXCESS = 0.05    # folds to bets at most pop+this (sticky; pays value). Tighter -> excluded.
FISH_MAX_RAISE_EXCESS = 0.03   # rarely raises facing a bet (does not punish thin value)
FISH_MAX_INIT_AGG = 0.05       # not a lag: own-initiative aggression at/below the Nash reference


def _is_value_fish(prof) -> bool:
    """True iff ``prof`` is a CONFIDENT sticky-caller (station) that pays thin value and does not punish:
    enough hands AND does not fold-to-bets above population AND rarely raises AND low own-initiative
    aggression. Unknown / aggressive (lag) / tight (nit, reg) all return False so the caller defers."""
    from pokerbot.opponent.bayes_exploit import aggregate_read

    if prof is None or getattr(prof, "hands", 0) < FISH_MIN_HANDS:
        return False
    fr = aggregate_read(prof, facing="facing_bet")
    if fr is None:
        return False
    fold_excess = float(fr.post[0] - fr.pop[0])
    raise_excess = float((fr.post[2] + fr.post[3]) - (fr.pop[2] + fr.pop[3]))
    if fold_excess > FISH_MAX_FOLD_EXCESS or raise_excess > FISH_MAX_RAISE_EXCESS:
        return False  # folds too much (nit/reg) or punishes (lag) -> not a thin-value target
    ur = aggregate_read(prof, facing="unopened")
    init_agg = float((ur.post[2] + ur.post[3]) - (ur.pop[2] + ur.pop[3])) if ur is not None else 0.0
    return init_agg <= FISH_MAX_INIT_AGG


def _all_live_villains_fish(state, hero: int, profiles: dict | None) -> bool:
    """Unopened value-bet gate: True iff there is >=1 live opponent AND every live (not-folded) opponent
    is a confirmed value-fish. Any lag / reg / nit / unidentified villain -> False -> defer. Robust in
    any constellation (2 fish + 1 aggressor -> False; all-unknown at session start -> False)."""
    if not profiles:
        return False
    live = [s for s, p in enumerate(state.players) if s != hero and not p.folded]
    if not live:
        return False
    return all(_is_value_fish(profiles.get(s)) for s in live)


def apply_root_exploit_mixer(state, seat: int, probs: np.ndarray, profiles: dict | None,
                             *, unopened: bool = False) -> np.ndarray:
    """Tilt the hero's own policy ``probs`` at ``state``/``seat`` by the villain read.

    Reads the villain's RESPONSE context (how they react to a bet when we can bet; how often
    they over-bet when we face one), tilts our policy by ``read.mixer_logits(hand_strength)``
    under the bounded-(alpha) safe mix, and renormalizes over legal actions. No read => probs
    returned unchanged (safe fallback, keeps the GTO base)."""
    from pokerbot.agent import _hero_hand_strength
    from pokerbot.opponent.bayes_exploit import (DEFAULT_MAX_ALPHA, aggregate_read, bluff_read,
                                                 context_read)
    from pokerbot.opponent.context import context_key, is_aggressive_type
    from pokerbot.opponent.mixer import safe_mix_policies
    from pokerbot.training.self_play import _legal_mask

    if profiles is None or probs.sum() <= 0:
        return probs
    # PER-VILLAIN TARGETING: read the profile of the villain the hero is actually responding to —
    # the last aggressor (the bettor on a facing decision; the most-defined villain otherwise). On a
    # MIXED table the old "first non-None" read fired the facing branch off a BLEND of every seat
    # (a lag + stations + a reg pooled into one muddy read), which mis-released/mis-caught and cost
    # ~0.8 bb/100 on the heterogeneous fields (the measured +exploit regression on the blueprint
    # base). BIT-IDENTICAL for the legacy single-pooled case: when every seat maps to the SAME
    # OpponentStats object, profiles[aggressor] IS that object == first-non-None. Only a genuinely
    # per-seat ``profiles`` dict (the new run_field_absolute per-seat mode / live PlayerProfileStore)
    # changes behavior — and then it correctly targets the opponent in front of us.
    prof = _select_profile(profiles, getattr(state, "last_aggressor", None))
    if prof is None:
        return probs
    legal = np.asarray(_legal_mask(state, seat), dtype=bool)
    hs = _hero_hand_strength(state, seat)
    hs = 0.5 if hs is None else float(np.clip(hs, 0.0, 1.0))
    board = [str(c) for c in (getattr(state, "board", None) or [])]
    street = {"PREFLOP": 0, "FLOP": 1, "TURN": 2, "RIVER": 3}.get(state.street.name, 0)
    facing = not bool(legal[1])  # CHECK illegal => an outstanding bet
    if street == 0:
        # Preflop, CHECK is illegal for everyone except the BB option, so the check-legal test
        # would classify EVERY preflop decision as "facing a bet". Blinds are not a value bet to
        # release against — an unraised preflop pot is a STEAL spot (measured: misrouting it
        # through the release logic folded medium hands for one blind vs an 85%-fold nit table,
        # -0.60 bb/hand). Facing preflop = a genuine RAISE occurred in the history.
        facing = any(is_aggressive_type(getattr(a.action_type, "name", ""))
                     for _s, a in state.action_history())
    target = "unopened" if facing else "facing_bet"
    # FACING-ONLY default: the unopened branch (tilt our own steals/value-bets/traps from the
    # villain's response read) lost in EVERY configuration across nine battery rounds vs the
    # caricature bots — bet-cut mass leaked to folds (station -3.8), the clamp converted it to
    # limp-spew (station -4.6 on two seeds), steal boosts mispriced multiway fold-equity
    # (tag -1.1), trap mode taxed every open. The validated subset is exploiting THEIR aggression
    # mistakes only (facing branch: pot-odds bluff-catch vs over-bettors, release vs
    # value-bettors) — lag +0.85 CI>0 across three seam revisions. ``unopened=True`` re-enables
    # the branch (its guards — value floor, steal shaping, fold-leak clamp — all stay active);
    # those losses were measured against bots with no human range structure, so the fair-bot
    # battery is the arbiter.
    if not facing and not unopened:
        return probs
    # VALUE-FISH GATE: the unopened thin-value tilt fires only when EVERY live opponent is a confirmed
    # station (see _all_live_villains_fish). This is the do-no-harm guarantee that lets the +1.4 bb/100
    # value-extraction run live: a single lag/reg/nit/unknown in the pot -> defer to the search policy.
    if not facing and not _all_live_villains_fish(state, seat, profiles):
        return probs
    villain_ck = context_key(street, board, 0 if facing else 1)
    read = (context_read(prof, villain_ck) or aggregate_read(prof, facing=target))
    if read is None:
        return probs
    if facing:
        # DIRECTIONAL facing-a-bet logic. mixer_logits is calibrated for RESPONSE (facing_bet)
        # reads; fed the villain's UNOPENED-aggression read it boosts CALL for ANY deviation
        # direction (measured: call 0.50->0.56 with AIR vs a passive nuts-only bettor — the nit
        # leg's -0.35 bb/hand leak; the same call-boost is what correctly prints vs the maniac).
        # Use the read's aggression EXCESS vs population directly:
        #   over-bettor  (excess>0): their bets are bluff-heavy -> bluff-catch (call) more,
        #                            hand-strength-independent (correct vs a jam-anything maniac);
        #   under-bettor (excess<0): their rare bets are value-heavy -> release weak hands
        #                            (fold air, don't raise-bluff); value hands keep the base line.
        excess = float((read.post[2] + read.post[3]) - (read.pop[2] + read.pop[3]))
        # SHOWDOWN-CONDITIONED bluff ratio (G2): frequency reads cannot split value from bluff —
        # an over-bettor whose bets are value-heavy (the conditional lag) punished the pure
        # frequency call-boost (hs_lag -0.32). When revealed-hand ground truth exists, the Beta
        # posterior P(bluff | aggressive action) re-weights both directions below; with no reveal
        # data (None — legacy profiles, Phase-0 arms) every line is byte-identical to before.
        bl = bluff_read(prof, villain_ck)
        adj = np.zeros(len(probs))
        if excess > 0:
            # READ-GATED BLUFF-CATCH (2026-06-13 cloud gate): on the SEARCH base the blanket
            # frequency call-boost LOST -1.1 bb/hand vs the conditional lag — frequency cannot
            # tell a value-bettor from a bluffer, so "they bet a lot => call more" pays off
            # value-heavy aggression, and the search policy ALREADY responds to the bet via its
            # solver-internal opponent bias (the boost double-counts). So bluff-catch ONLY when
            # SHOWDOWN ground truth confirms the villain is bluff-heavy; otherwise add nothing
            # here and DEFER to the search policy. (The heuristic-base hero, which has no such
            # solver response, keeps its own inline bluff-catch and is unaffected — it wins 4/4.)
            confirmed_bluffer = (bl is not None and bl[1] >= BLUFF_CATCH_MIN_CONF
                                 and bl[0] > BLUFF_CATCH_MIN_RATIO)
            if confirmed_bluffer:
                # POT-ODDS-AWARE bluff-catching vs a CONFIRMED bluffer: a call still needs hs to
                # beat the required equity of THIS bet size (don't pay off a big bet with air).
                player = state.players[seat]
                tc = max(0.0, float(state.current_bet_bb) - float(player.committed_street_bb))
                pot = float(getattr(state, "pot_bb", 0.0) or 0.0)
                req = tc / (pot + tc) if (pot + tc) > 1e-9 else 0.5
                p_b, c_b = bl
                disc = c_b * 1.2 * (p_b - 0.20)  # ground-truthed equity discount (bluff-heavy => +)
                if hs + disc > req:
                    adj[2] = min(2.0 * excess, 1.0)
                    adj[0] = -min(excess, 0.5)
                    if hs < 0.6:
                        # A hand that beats bluffs only: CALL-DOWN dominates raising — a raise
                        # folds out exactly the bluffs that fund the call.
                        adj[3:] = np.minimum(adj[3:], 0.0) - 0.3 * bl[1]
            # else: no confirmed bluff evidence => defer to the search policy (adj stays zero).
        else:
            w = min(1.0, max(0.0, (0.6 - hs) / 0.6))  # only weak/medium hands release
            rel = -excess
            if bl is not None and bl[0] > 0.30:
                # Their RARE bets are proven bluff-prone at showdown ⇒ the "rare = value" release
                # assumption is wrong for THIS villain; shrink the release toward the base line.
                rel *= max(0.0, 1.0 - bl[1] * min(1.0, (bl[0] - 0.30) / 0.30))
            adj[0] = min(2.0 * rel * w, 1.0)
            adj[2] = -min(rel * w, 0.5)
            adj[3:] = -min(rel * w, 0.5)
    else:
        # READ-GATED UNOPENED branch (2026-06-13, two cloud confirms). The unopened tilt value-bets
        # a villain who CALLS our bets (high PASSIVE in the facing_bet read => "station, pays off
        # thin value"). That is RIGHT for a true station but WRONG for a LAG: the lag ALSO calls
        # our bets light (so it reads passive-responder, resp_agg ~ -0.06, NOT caught by a
        # response-aggression gate) yet it punishes thin value (raises back, calls down with hands
        # that beat us, barrels when checked to) — the measured -0.83 bb/hand leak. The signal that
        # separates them SHOULD be the villain's OWN-INITIATIVE aggression (do they BET a lot when
        # first in / checked to). HONEST RESULT (2026-06-13 diagnostic): this gate does NOT fire vs
        # the lag either — init_agg is NEGATIVE for EVERY archetype because the betting abstraction's
        # postflop menus are mostly (ALL_IN, CHECK, FOLD), so bots check and recorded unopened
        # aggression sits below the Nash reference for all of them (the _pick_bet/legal-menu quirk,
        # task #12). NEITHER read separates a lag from a station: the lag's danger is its value/bluff
        # RANGE structure, invisible to frequency reads (the project's core lesson). So the
        # search+mixer CANNOT be made do-no-harm by gating and is OFF on the live search base
        # (live_advisor_bridge default False). The gate is retained (harmless; would fire only if a
        # future abstraction exposes real betting frequencies); the validated exploitation vehicle is
        # the heuristic/HybridAgent mixer (clean 4/4 incl lag +0.83).
        init_read = (context_read(prof, context_key(street, board, 0))
                     or aggregate_read(prof, facing="unopened"))
        init_agg = (float((init_read.post[2] + init_read.post[3])
                          - (init_read.pop[2] + init_read.pop[3])) if init_read is not None else 0.0)
        if init_agg > AGGR_DEFER_EXCESS:
            return probs
        adj = read.mixer_logits(hs)
        # Value opens are never suppressed: vs an aggressive/no-fold responder read the mixer
        # tilts toward trap-mode across ALL hand strengths (measured vs tag: aggro-delta -0.19
        # at hs>=0.6 on ~every preflop open — a systematic tax surrendering initiative multiway).
        # Trapping stays available for medium/weak hands; with a value hand the bet mass keeps
        # at least the base line.
        if hs >= 0.6:
            adj[3:] = np.maximum(adj[3:], 0.0)
    # Extremity-scaled alpha, mirroring the solver tail's own gating (bayes_exploit.search_tilt:
    # max_alpha * conf * min(1, tv/_TV_FULL)). conf-only scaling over-deviates against a BALANCED
    # opponent (high confidence, ~zero deviation to exploit): measured tag-control failure
    # (-1.13 bb/hand, bet_shift -0.185) on the search base, which the heuristic hero's cruder
    # base had masked. tv~0 => alpha~0 => the control stays clean by construction.
    alpha = DEFAULT_MAX_ALPHA * read.conf * min(1.0, read.tv / 0.25)
    if alpha <= 1e-3:
        return probs
    # Steal shaping: a fold-equity bluff-boost wins the pot whatever the size but loses ~size
    # when called, so without a value hand concentrate any positive bet-boost on the SMALLEST
    # legal sizing (measured: the unshaped boost is hand-strength-independent across ALL sizes,
    # and the nit leg lost -0.36 bb/hand barreling big into the nit's continue range).
    # MULTIWAY fold-equity: the read prices ONE villain's fold rate, but a steal must get
    # through EVERY live opponent — compound fold-through = p_fold^k. The single-villain boost
    # already prices one fold, so scale the bluff-boost by p_fold^(k-1): heads-up unchanged;
    # vs five 65%-folders (0.65^4 ~ 0.18) the steal dies (measured -1.10 bb/hand stealing junk
    # into a 5-tag field, then playing bloated pots vs a never-fold-postflop bot); vs five
    # 85%-folders (0.85^4 ~ 0.52) stealing stays right at half the naive boost. Value bets
    # (hs >= 0.6) are not fold-equity bets and are untouched.
    if hs < 0.6:
        live = sum(1 for p in state.players if not p.folded) - 1
        p_fold = float(np.clip(read.post[0], 0.0, 1.0))
        mult = p_fold ** max(0, live - 1)
        bet_idx = [a for a in range(3, len(adj)) if legal[a]]
        if bet_idx:
            smallest = bet_idx[0]
            for a in bet_idx:
                if adj[a] > 0:
                    adj[a] = adj[a] * mult if a == smallest else 0.0
    out = safe_mix_policies(probs, np.log(probs + 1e-8) + adj, alpha, max_alpha=DEFAULT_MAX_ALPHA)
    out = out * legal
    s = out.sum()
    if s <= 0:
        return probs
    out = out / s
    if not facing and out[0] > probs[0] + 1e-12:
        # FOLD-LEAK CLAMP: in unopened spots the mixer's bet-cut ("they call too much — bluff
        # less") renormalizes the removed mass into ALL actions, including FOLD — measured vs a
        # calling station: -3.76 bb/hand from folding playable hands preflop for one blind, and
        # the same leak was tag's every-hand trap-mode tax. A read about the villain CONTINUING
        # never implies WE fold more; route the cut mass to the passive continues instead. (The
        # facing branch sets adj[0] explicitly and is exempt — releasing vs a value bet is the
        # one place a read legitimately adds folds.)
        extra = out[0] - probs[0]
        out[0] = probs[0]
        passive = [i for i in (1, 2) if legal[i]]
        if passive:
            for i in passive:
                out[i] += extra / len(passive)
        else:
            out[0] += extra  # nothing to continue with (shouldn't happen) — keep mass on fold
        out = out / out.sum()
    return out
