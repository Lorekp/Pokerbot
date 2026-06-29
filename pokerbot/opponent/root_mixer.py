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

    if prof is None:
        return False
    # HUD-driven path: a server-HUD station (whole-history stats: wtsd/wsd/foldToCbet/...) qualifies even
    # with few LOCALLY observed hands. Keying only on the derived count fields (FISH_MIN_HANDS) left the
    # captured-HUD lever (the headline value-extraction signal, population-hud-analysis) inert on the live
    # path. Require a CONFIDENT, clearly-sticky, weak-showdown read and not a (HUD-)aggressor.
    if getattr(prof, "hud", None):
        from pokerbot.opponent.value_sizing import _pct, station_payoff_score
        hud = prof.hud
        vpip, pfr = _pct(hud, "vpip"), _pct(hud, "pfr")
        ftc = _pct(hud, "foldToCbet", "foldtocbet")
        passive = vpip is not None and pfr is not None and (vpip - pfr) >= 0.25  # big limp/flat gap
        sticky = ftc is not None and ftc <= 0.40                                 # doesn't fold to c-bets
        # FIX B3: a play-money HUD often lacks the sticky/weak postflop stats (foldToCbet/wsd) the
        # original station gate required, so a clearly loose-passive limp/flat field never admitted ->
        # the fish gate stayed shut and the whole table read "tough". Admit a LOOSE (VPIP>=0.50)
        # loose-passive read as fish too. `passive` REQUIRES the wide VPIP-PFR gap, so a loose-AGGRESSIVE
        # maniac (high VPIP, small gap) is NOT admitted here and is still vetoed by _table_too_aggressive.
        loose = vpip is not None and vpip >= 0.50
        score, conf = station_payoff_score(prof)
        # AUDIT wiup89smb finding #5: a `loose and passive` read (whole-history VPIP>=0.50 with the wide
        # VPIP-PFR limp/flat gap) is a high-confidence loose-passive STATION from the server stats ALONE,
        # so admit it WITHOUT the station_payoff_score conf gate — that gate fails CLOSED on a partial
        # play-money HUD lacking the postflop wsd/foldToCbet stats, leaving a visibly-soft station inert.
        # The conf>=0.8 gate still guards the score/sticky paths (those genuinely need showdown stats).
        # `passive` REQUIRES the wide gap, so a loose-AGGRESSIVE maniac is still NOT admitted here.
        if loose and passive:
            return True
        if conf >= 0.8 and (score >= 0.5 or (passive and sticky)):
            return True
    if getattr(prof, "hands", 0) < FISH_MIN_HANDS:
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


# Table-aggression kill-switch (Phase 2): the cloud gauntlet showed the WHOLE exploit (solver opponent-
# tail + facing branch, not just the unopened tilt) is +EV on every realistic field — incl. a 5-TAG "pro"
# table (+0.99) — and loses ONLY vs a pathological 5-LAG table (-0.48). So disable the entire exploit when
# >= AGG_KILL_COUNT live villains are confirmed LOOSE-aggressive (lags). Crucially this keys on LOOSE
# aggression (raises AND folds little) so it fires on lags but NOT on tags — a tag raises but folds tight,
# so the +0.99 vs pros is kept while the -0.48 vs lags is zeroed.
AGG_KILL_COUNT = 3            # >= this many confirmed lags at the table -> exploit OFF (protect > extract)
LAG_MIN_AGG_EXCESS = 0.10    # own-initiative OR facing-raise aggression this far above the Nash reference
LAG_MAX_FOLD_EXCESS = 0.0    # LOOSE: folds to bets at most the population average (a tag folds MORE)
LAG_MIN_WSD = 0.45           # a loose-aggressor winning BELOW this at showdown (server HUD) is a maniac FISH
                             # (pays off), not a skilled LAG (>= break-even) -> excused from the lag count


def _is_dangerous_aggressor(prof) -> bool:
    """True iff ``prof`` is a confident LOOSE-aggressive villain (a lag): elevated own-initiative or
    facing-raise aggression AND does not fold tight. A TAG (raises but folds more than population) and a
    station/nit/unknown all return False — so the kill-switch fires on lags, not on profitable tags."""
    from pokerbot.opponent.bayes_exploit import aggregate_read

    if prof is None or getattr(prof, "hands", 0) < FISH_MIN_HANDS:
        return False
    ur = aggregate_read(prof, facing="unopened")
    init_agg = float((ur.post[2] + ur.post[3]) - (ur.pop[2] + ur.pop[3])) if ur is not None else 0.0
    fr = aggregate_read(prof, facing="facing_bet")
    if fr is None:
        return False
    fold_excess = float(fr.post[0] - fr.pop[0])
    raise_excess = float((fr.post[2] + fr.post[3]) - (fr.pop[2] + fr.pop[3]))
    aggressive = init_agg > LAG_MIN_AGG_EXCESS or raise_excess > LAG_MIN_AGG_EXCESS
    loose = fold_excess <= LAG_MAX_FOLD_EXCESS
    if aggressive and loose:
        # AUDIT wiup89smb finding #1 (HIGH): the frequency-only test above can't tell a skilled LAG from a
        # loose-aggressive MANIAC FISH (raises a lot, folds little, but LOSES at showdown) — so a maniac-fish
        # table tripped the kill-switch and the WHOLE exploit was disabled against the softest, most
        # exploitable field (the toughness fix reads them SOFT, then this nulled the exploit). The clean
        # discriminator is WON-AT-SHOWDOWN: a maniac fish shows up weak (low WSD = pays off), a skilled LAG
        # wins at showdown (WSD at/above break-even). Veto a CONFIDENT weak-showdown server-HUD read so the
        # maniac is exploited, not killed. NB do NOT use station_payoff_score here — it x0.3-damps an
        # archetype "lag", which a real maniac IS, so it cannot separate the two; WSD is read DIRECTLY. An
        # unknown / HUD-less aggressor has wsd=None -> NOT vetoed (fails closed), preserving lag protection.
        from pokerbot.opponent.value_sizing import _pct
        wsd = _pct(getattr(prof, "hud", None) or {}, "wsd")
        if wsd is not None and wsd < LAG_MIN_WSD:
            return False   # confident weak-showdown loose-aggressor = maniac FISH, exploit it (do not kill)
    return aggressive and loose


def _table_too_aggressive(profiles: dict | None, k: int = AGG_KILL_COUNT) -> bool:
    """True iff at least ``k`` seated villains are confirmed lags -> disable the whole exploit for this
    table (protect-over-extract). Counts ALL seated profiles (table composition), not just currently-live,
    so a lag-dominated table stays guarded even in late-street heads-up spots."""
    if not profiles:
        return False
    return sum(1 for p in profiles.values() if _is_dangerous_aggressor(p)) >= k


def apply_root_exploit_mixer(state, seat: int, probs: np.ndarray, profiles: dict | None,
                             *, unopened: bool = False, solver_best_responds: bool = False,
                             stability_tracker=None, gift_register=None, villain_key=None,
                             n_players: int | None = None) -> np.ndarray:
    """Tilt the hero's own policy ``probs`` at ``state``/``seat`` by the villain read.

    Reads the villain's RESPONSE context (how they react to a bet when we can bet; how often
    they over-bet when we face one), tilts our policy by ``read.mixer_logits(hand_strength)``
    under the bounded-(alpha) safe mix, and renormalizes over legal actions. No read => probs
    returned unchanged (safe fallback, keeps the GTO base).

    ``solver_best_responds``: set when the BASE policy already best-responds to the profiled opponent
    (the pluribus SEARCH vehicle, whose ``solver.opponent_model`` makes hero CFR value-bet the fish
    internally). The UNOPENED value tilt would then stack a SECOND value push on the same read and
    over-exploit (the measured negative lean on a homogeneous station table) — so under this flag the
    value branch DEFERS to the solver, mirroring the facing branch's existing defer-to-search-policy.
    The heuristic / HybridAgent vehicle has NO solver response (solver_best_responds=False) and keeps
    the full tilt (its battery-validated +EV path is unchanged).

    F1 wiring (F1 audit note, 2026-06-26): when ``stability_tracker`` AND ``villain_key`` are
    supplied, the FACING branch's alpha ceiling is composed with ``effective_alpha(confidence,
    n_players, stability)`` (the same shape as ``apply_live_value_sizing``). A villain the tracker
    has caught ADAPTING clamps the ceiling toward CAP_FLOOR — closes the structural uncapping the
    audit flagged. Without a tracker wired, stability defaults to 1.0 (commit fully, bounded only
    by the N-ceiling). ``gift_register`` BEFEWP relaxes the BASE ceiling UP TO — but never past —
    the RNR ceiling. ``n_players`` overrides the ``n_live_players(state, hero=seat)`` computation
    when the caller knows better (testing, custom simulators)."""
    from pokerbot.agent import _hero_hand_strength
    from pokerbot.opponent.bayes_exploit import (DEFAULT_MAX_ALPHA, aggregate_read, bluff_read,
                                                 context_read)
    from pokerbot.opponent.context import context_key, is_aggressive_type
    from pokerbot.opponent.mixer import safe_mix_policies
    from pokerbot.opponent.rnr_cap import (CAP_FLOOR, default_exploit_cap, effective_alpha,
                                           n_live_players)
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
    # DOUBLE-COUNT GUARD: if the base already best-responds to the read (search vehicle's
    # solver.opponent_model value-bets the fish internally), the unopened VALUE tilt double-counts ->
    # over-exploits. Defer the value branch to the solver (the facing branch already defers its
    # bluff-catch). Off for the heuristic/Hybrid vehicle (no solver response) -> full tilt retained.
    if not facing and solver_best_responds:
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
    # F1 wire-up: compose with the N-dependent RNR ceiling + (optional) stability_tracker clamp.
    # Mirrors apply_live_value_sizing (root_mixer.py:602-632): the FACING branch now respects the
    # same N- and stability-dependent cap the value-sizing overlay already enforces, so a 6-max
    # counter-adapter can no longer over-exploit. Stability defaults to 1.0 (commit fully when
    # no tracker is wired) — backward-compatible with the existing battery tests.
    try:
        if n_players is None:
            n_players = int(n_live_players(state, hero=seat))
        rnr_ceiling = float(default_exploit_cap(max(2, n_players)))
        stability = 1.0
        if stability_tracker is not None and villain_key is not None:
            try:
                stability = float(stability_tracker.opponent_stability(villain_key))
            except Exception:  # noqa: BLE001 — a broken tracker MUST NOT break the serve
                stability = 1.0
        # Compose: the FACING-branch alpha may not exceed min(ceiling_after_stab, alpha).
        # When stability=1.0 the ceiling relaxes to rnr_ceiling (no tracker = commit fully,
        # bounded only by N). When stability=0 the cap collapses to CAP_FLOOR for any N.
        capped_alpha = min(effective_alpha(confidence=float(read.conf),
                                           n_players=max(2, n_players),
                                           stability=stability),
                            rnr_ceiling)
        # BEFEWP: gift_register may raise the BASE ceiling UP TO the rnr_ceiling (clamped above).
        if gift_register is not None and villain_key is not None:
            try:
                capped_alpha = min(max(capped_alpha,
                                       float(gift_register.capped_max_alpha(villain_key, capped_alpha))),
                                   rnr_ceiling)
            except Exception:  # noqa: BLE001
                pass
        alpha = min(alpha, capped_alpha)
    except Exception:  # noqa: BLE001 — never let the RNR wire break the legacy path
        pass
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


def _overlay_hand_strength(state, seat: int) -> float | None:
    """Board-aware hero strength for the overlays. Delegates to the SHARED
    ``value_sizing.overlay_hand_strength`` (one source of truth used live + in the sim battery), so the
    sim measures exactly the live value/air separation rather than the preflop-rank proxy."""
    from pokerbot.opponent.value_sizing import overlay_hand_strength
    return overlay_hand_strength(state, seat)


def _live_rake_context(state):
    """Resolve the postflop ``RakeContext`` (pot/dealt/stake/is_preflop) for the rake-aware SIZE
    discount, from the live ``GameState`` + env stake. Returns None on any failure (→ no discount).

    * stake_key: ``POKERBOT_CASH_STAKE`` if it names a known stake, else the documented rake-VALLEY
      default ``POKERBOT_CASH_STAKE_DEFAULT`` (``0.10/0.25``) — the SAME resolution precedence the live
      coach / table_select use. CoinPoker play-money is raked exactly like cash, so a stake always resolves.
    * pot_bb: the whole-hand pot BEFORE hero's value bet (state.pot_bb).
    * n_dealt: ACTUAL players DEALT this hand (cap tier) — a folded player was still dealt, so count any
      seat that holds cards / has committed chips / is not folded. NEVER max-seats (would over-tier the cap).
    * is_preflop: no board yet ⇒ rake hasn't applied (no-flop-no-drop) ⇒ the discount must not fire."""
    import os

    try:
        from pokerbot.economics.rake import STAKES
        from pokerbot.opponent.value_sizing import RakeContext
    except Exception:  # noqa: BLE001
        return None
    sk = os.environ.get("POKERBOT_CASH_STAKE", "").strip()
    if sk not in STAKES:
        sk = os.environ.get("POKERBOT_CASH_STAKE_DEFAULT", "0.10/0.25").strip()
    if sk not in STAKES:
        return None
    try:
        pot_bb = float(getattr(state, "pot_bb", 0.0) or 0.0)
        players = list(getattr(state, "players", None) or [])
        n_dealt = sum(
            1 for p in players
            if (getattr(p, "hole", None) is not None)
            or (float(getattr(p, "committed_total_bb", 0.0) or 0.0) > 0.0)
            or (not getattr(p, "folded", False))
        )
        n_dealt = max(2, int(n_dealt))
        board = [c for c in (getattr(state, "board", None) or []) if c is not None]
        is_preflop = len(board) < 3
    except Exception:  # noqa: BLE001
        return None
    return RakeContext(pot_bb=pot_bb, n_dealt=n_dealt, stake_key=sk, is_preflop=is_preflop)


def _exp_size(p: np.ndarray) -> float:
    """Expected bet SIZE in pot fractions over the bet/raise mass of policy ``p``.

    Sum over the bet/raise buckets of ``prob * frac`` (BET_SIZE_BUCKETS), NORMALIZED over the
    bet/raise probability mass only (so a policy that mostly checks still reports the size it would
    pick WHEN it bets). All-in is mapped to the largest menu fraction (an over-bet shove). Returns
    0.0 when there is no bet/raise mass (a pure check/call/fold policy has no size to report).

    This is the live mirror of the sim acceptance's bet_shift metric (the change in this quantity
    from the pre-overlay search base to the post-overlay mixed policy)."""
    from pokerbot.engine.actions import BET_SIZE_BUCKETS, NUM_ACTIONS

    n = len(p)
    lo = 3                                  # first bet/raise bucket (0=fold,1=check,2=call)
    if n <= lo:
        return 0.0
    fracs = list(BET_SIZE_BUCKETS) + [BET_SIZE_BUCKETS[-1]]   # all-in ≈ the largest size
    mass = 0.0
    acc = 0.0
    for i in range(lo, n):
        bi = i - lo
        if bi >= len(fracs):
            break
        w = float(p[i])
        if w <= 0.0:
            continue
        mass += w
        acc += w * float(fracs[bi])
    return float(acc / mass) if mass > 1e-12 else 0.0


def apply_live_value_sizing(state, seat: int, probs: np.ndarray, profiles: dict | None,
                            *, max_alpha: float | None = None,
                            stability_tracker=None, gift_register=None,
                            villain_key=None, out: dict | None = None) -> np.ndarray:
    """LIVE-only HUD value-SIZING / bluff-suppression overlay on the SEARCH base.

    The validated value_sizing overlay (value_sizing.py, +11.9 bb/100 vs stations) lived ONLY on the
    HybridAgent/battery path; the live LiveSearchAdvisor exploited via the root mixer above, so the
    headline HUD lever never fired live. This wires it on: vs a confirmed STATION table that is NOT
    loose-aggressive (the SAME firewalls the root mixer uses), tilt the hero's own value bets toward
    bigger sizes + cut bluffs, bounded by the alpha-cap. Distinct from the proven-dead bluff-CATCH read
    mixer (this is sizing, not a frequency call-boost). Returns ``probs`` unchanged when it doesn't fire.

    RNR CEILING (Workstream F): the alpha ceiling is no longer the flat 0.7 — it is the N-DEPENDENT
    Restricted-Nash-Response cap ``rnr_cap.default_exploit_cap(n_players)`` (1.0 heads-up / 3-max where
    a BR is minimax-safe; the lower multiway ceilings where it is not). The played weight is folded
    through read STABILITY: ``alpha = effective_alpha(confidence, n_players, stability)``, so a villain
    the (optional) ``stability_tracker`` has caught ADAPTING is clamped toward CAP_FLOOR. With no
    tracker wired, stability defaults to 1.0 (commit fully, bounded only by the N-ceiling). The
    GiftRegister (BEFEWP) may relax the BASE ceiling UP TO — but never past — the RNR ceiling.
    Passing ``max_alpha`` explicitly overrides the RNR ceiling (back-compat / tests).

    Env kill-switch ``POKERBOT_LIVE_VALUE_SIZING=0``. The component overlays keep their own switches
    (POKERBOT_VALUE_SIZING / POKERBOT_BLUFF_SUPPRESS / POKERBOT_RIVER_THIN_VALUE).

    ``out`` (optional, keyword-only, default None): a capture dict for the live-smoke harness. When a
    dict is passed it is populated with the overlay diagnostics REGARDLESS of fire/no-fire — the SAME
    bet_shift metric the sim acceptance used (the change in hero's expected bet SIZE from the pre-overlay
    base to the post-overlay mix), plus the station score, the RNR ceiling, the rake discount, and the
    reason it did not fire. Purely observational and exception-safe: a capture failure NEVER alters the
    returned policy (default None == BYTE-IDENTICAL to the pre-capture behaviour)."""
    import os

    # --- capture scaffold (observational; never affects the returned array) ------------------------
    # Pre-filled so EVERY early return leaves a complete, reason-stamped row. Populated in-place.
    if out is not None:
        try:
            _street = getattr(getattr(state, "street", None), "name", "") or ""
            out.clear()
            out.update({
                "fired": False, "reason": "", "exp_size_pre": _exp_size(probs),
                "exp_size_post": _exp_size(probs), "bet_shift": 0.0,
                "station_score": 0.0, "station_conf": 0.0, "n_players": 0,
                "rnr_ceiling": 0.0, "alpha": 0.0, "rake_keep": 1.0,
                "street": str(_street), "hand_strength": float("nan"),
            })
        except Exception:  # noqa: BLE001 — capture must never break the serve
            out = None

    def _bail(reason: str):
        if out is not None:
            try:
                out["reason"] = reason
            except Exception:  # noqa: BLE001
                pass
        return probs

    if os.environ.get("POKERBOT_LIVE_VALUE_SIZING", "1").lower() in ("0", "false", "no", "off"):
        return _bail("kill_switch_off")
    if profiles is None or probs.sum() <= 0:
        return _bail("no_profiles_or_empty_probs")
    # FIREWALLS (reuse the root-mixer gates): only vs an all-station table, never vs a lag-heavy one.
    if _table_too_aggressive(profiles):
        return _bail("table_too_aggressive")
    if not _all_live_villains_fish(state, seat, profiles):
        return _bail("not_all_villains_fish")
    prof = _select_profile(profiles, getattr(state, "last_aggressor", None))
    if prof is None:
        return _bail("no_profile_selected")

    from pokerbot.opponent.mixer import _stats_confidence, safe_mix_policies
    from pokerbot.opponent.rnr_cap import default_exploit_cap, effective_alpha, n_live_players
    from pokerbot.opponent.value_sizing import (bluff_suppression_logits, river_thin_value_logits,
                                                station_payoff_score, value_sizing_logits)
    from pokerbot.training.self_play import _legal_mask

    # BOARD-AWARE strength (not the preflop-rank proxy): so the value-SIZE escalation never fires on air.
    hs = _overlay_hand_strength(state, seat)
    board = [str(c) for c in (getattr(state, "board", None) or [])]
    # RAKE-AWARE SIZE discount (Workstream F): postflop, BELOW the cap, growing the pot forfeits 5% to
    # rake — so dampen the value-SIZE escalation by the marginal rake of the mid→big step (vanishes above
    # the cap, where rake is flat). Resolved from the live pot/dealt + env stake; None ⇒ no discount.
    rake_ctx = _live_rake_context(state)
    if out is not None:
        try:
            from pokerbot.opponent.value_sizing import rake_keep_factor, station_payoff_score
            _ssc, _scf = station_payoff_score(prof)
            out["station_score"] = float(_ssc)
            out["station_conf"] = float(_scf)
            out["hand_strength"] = float(hs) if hs is not None else float("nan")
            out["rake_keep"] = float(rake_keep_factor(prof, rake_ctx))
        except Exception:  # noqa: BLE001
            pass
    overlay = (value_sizing_logits(prof, hs, rake_ctx=rake_ctx)
               + bluff_suppression_logits(prof, hs)
               + river_thin_value_logits(prof, hs, board))
    if not np.any(overlay):
        return _bail("zero_overlay")

    legal = np.asarray(_legal_mask(state, seat), dtype=bool)
    base_logits = np.log(np.clip(probs, 1e-9, 1.0))
    exploit_logits = base_logits + overlay
    exploit_logits[~legal] = -1e9            # never let the overlay leak prob onto an illegal action

    # --- RNR exploitation cap (Workstream F) -------------------------------------------------------
    # Replaces the flat max_alpha=0.7. The HARD ceiling is the stability-adjusted N-dependent RNR cap
    # cap_eff(stability, n) (== default_exploit_cap(n) when stable; relaxes to CAP_FLOOR on a caught
    # adapter). Nothing below may exceed it.
    from pokerbot.opponent.rnr_cap import cap_eff as _cap_eff
    n_players = n_live_players(state, hero=seat)
    # The villain the overlay reads is the one whose stability gates the cap (per-read stability,
    # mirroring the AoF _per_read_stability single-opponent case for this targeted overlay).
    vkey = villain_key
    if vkey is None:
        vkey = getattr(state, "last_aggressor", None)
    stability = 1.0
    if stability_tracker is not None and vkey is not None:
        try:
            stability = float(stability_tracker.opponent_stability(vkey))
        except Exception:  # noqa: BLE001 — a malformed tracker must never break the serve path
            stability = 1.0
    rnr_hard = _cap_eff(stability, n_players)        # the hard RNR bound for this read
    if max_alpha is not None:
        # Explicit override (back-compat / tests): the caller's value is the hard ceiling.
        ceiling = float(max_alpha)
    elif gift_register is not None and vkey is not None:
        # GiftRegister (BEFEWP) composition: a banked gift relaxes the LEGACY base cap (0.7) toward the
        # RNR bound but is then HARD-CAPPED by it (the RNR cap is the bound; the gift can never push
        # past it). Where the RNR ceiling RAISED the legacy cap (HU/3-max, rnr_hard >= 0.7) the gift
        # lifts 0.7 toward rnr_hard; where it LOWERED it (6-max, rnr_hard < 0.7) rnr_hard already binds
        # and the gift is inert. With a fully-banked gift the result still never exceeds rnr_hard.
        _LEGACY_BASE_CAP = 0.7
        base_cap = min(rnr_hard, _LEGACY_BASE_CAP)
        try:
            base_cap = float(gift_register.capped_max_alpha(vkey, base_cap))
        except Exception:  # noqa: BLE001
            pass
        ceiling = min(rnr_hard, base_cap)
    else:
        # No gift, no override: the RNR cap IS the ceiling. This is the intended RNR change — HU/3-max
        # commit up to 1.0 (minimax-safe), 6-max is bounded at the (placeholder) 0.5 ceiling.
        ceiling = rnr_hard

    # EXTREMITY-gate the blend weight (mirrors the root mixer's tv-scaling): the live overlay previously
    # blended at a fixed _stats_confidence*max_alpha, which over-deviates against a marginal/neutral read.
    # Scale alpha by the station score so a borderline station collapses the tilt toward 0.
    score, _sconf = station_payoff_score(prof)
    confidence = _stats_confidence(prof) * float(min(1.0, score))
    # The single RNR clamp: alpha = min(confidence, ceiling). `effective_alpha` enforces the same
    # min(confidence, cap_eff) shape; we additionally min with `ceiling` so an override/gift-relaxed
    # bound is honoured. safe_mix_policies re-clips at max_alpha=ceiling as a defensive backstop.
    alpha = min(effective_alpha(confidence, n_players, stability), ceiling)
    mixed = safe_mix_policies(probs, exploit_logits, alpha, max_alpha=ceiling)
    mixed = mixed * legal
    s = float(mixed.sum())
    final = mixed / s if s > 1e-12 else probs
    if out is not None:
        try:
            post = _exp_size(final)
            out["n_players"] = int(n_players)
            out["rnr_ceiling"] = float(ceiling)
            out["alpha"] = float(alpha)
            out["exp_size_post"] = float(post)
            out["bet_shift"] = float(post - out.get("exp_size_pre", post))
            # "fired" = the overlay actually moved the policy (alpha > 0 and a measurable size shift).
            fired = bool(alpha > 1e-6 and abs(out["bet_shift"]) > 1e-9)
            out["fired"] = fired
            out["reason"] = "fired" if fired else "alpha_zero_or_no_shift"
        except Exception:  # noqa: BLE001
            pass
    return final
