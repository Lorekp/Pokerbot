"""HUD-driven value-SIZING escalation — the value-extraction lever the population analysis isolated.

The exploit mixer already raises our bet/raise *frequency* against a station (``mixer_logits`` /
``exploit_adjustment``), but it did NOT use the server HUD stats (``OpponentStats.hud``: wtsd, wsd,
foldToCbet, fta, ...) and applied no per-opponent **sizing** escalation. Real-population data
(IRC 7.4M hands; ``artifacts/population_hud_analysis.md``) is unambiguous: the field is loose-passive
(VPIP 48 / PFR 10 / AF 0.67 medians) and — the key tell — WTSD 37% with WSD only 46% ⇒ they go to
showdown with **losing** hands. Against such a player the highest-EV adjustment is to bet **bigger**
for value (they call > MDF with worse), NOT to bluff more.

This module turns a per-opponent "calling-station payoff" read into a logit overlay that shifts our
bet-size distribution toward the larger buckets **for value hands only**, with two safety guards:
  1. it fires only when hero holds a value hand (scaled by hand strength) — never escalates air;
  2. it is DAMPENED by the opponent's showdown-range strength (high WSD = they show up with real
     hands ⇒ over-betting value-cuts) and confidence-gated by sample size.
The overlay is additive to the existing exploit logits and is bounded, so the mixer's α-cap and the
LAG/RNR firewall still apply on top. Env kill-switch: ``POKERBOT_VALUE_SIZING=0``.

Sizing theory (MDF): a value bet is profitable whenever the opponent continues more than
bet/(pot+bet) of the time — at 1x pot they must fold < 50%, at 2x pot < 33%. A station continuing
~60% makes every size up to ~1.3x pot +EV *even accounting for the hands that beat us when called*,
so escalating size strictly increases extraction. (GTO Wizard MDF/over-bet, Upswing calling-station.)
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

from pokerbot.engine.actions import BET_SIZE_BUCKETS, NUM_ACTIONS

# Action layout: 0=fold, 1=check, 2=call, 3..NUM_ACTIONS-2 = bet/raise size buckets ascending
# (1/3, 1/2, 2/3, 3/4, 1, 1.5, 2 pot), NUM_ACTIONS-1 = all-in.
_BET_LO = 3
_ALL_IN = NUM_ACTIONS - 1
_BIG_LO = max(_BET_LO, NUM_ACTIONS - 4)   # first "big" (>= ~1x pot) bucket; 7 for NUM_ACTIONS=11

# Reference thresholds (fractions). _FOLD_REF ~ a loose MDF-ish baseline: folding LESS than this to a
# bet/cbet = sticky. _WSD_REF/_WSD_SPAN map won-at-showdown into a [0,1] "weak showdown range" signal.
_FOLD_REF = 0.45
_WSD_REF = 0.50
_WSD_SPAN = 0.30
_MIN_N = 25.0                              # derived-count sample for full confidence (no HUD)

# Overlay gains (logit units). Comparable to the existing exploit terms (~1.0-1.5); α-capped downstream.
_K_BIG = 1.2                               # boost to the over-bet buckets (>=1x pot, excl. all-in)
_K_MID = 0.35                              # smaller boost to mid bets (keeps total betting up)
_K_CHECK = 0.5                             # discourage checking back a value hand vs a payoff station

# Bluff-suppression gains: the symmetric (air-hand) overlay — slash our bluff betting vs a non-folder.
_K_BLUFF = 0.9                             # cut to bet/raise buckets when bluffing a sticky station
_K_BLUFF_CHECK = 0.5                       # lean toward checking/giving up air vs a non-folder

# River thin-value expansion: bet marginally-ahead hands (2nd pair / weak TP) the base would CHECK BACK,
# vs a confirmed weak-showdown station on the RIVER (no card to come ⇒ a check-back has no protective
# value, and the station calls with worse). Fires only in the marginal band BELOW the normal value
# cutoff; value hands (h>=0.5) are already handled by value_sizing_logits.
_THIN_LO = 0.40                            # floor: below this we're not ahead of a station's call range
_THIN_HI = 0.50                            # ceiling: at/above this value_sizing already fires
_WSD_THIN_MAX = 0.50                       # require a weak showdown range (won-at-sd < this) to bet thin
_K_THIN = 0.6                              # boost to the small/mid value buckets for a thin value bet
_K_THIN_CHECK = 0.45                       # discourage checking back a thin-value hand vs a station


# --- Rake-aware SIZE-escalation discount -----------------------------------------------------------
# The SIZE escalation (mid->big bet buckets) grows the final pot, and CoinPoker rakes min(5%*pot, cap).
# BELOW the cap, every BB of extra pot forfeits 5% to rake -> escalating thin value into a small pot can
# cost more rake than the marginal extraction is worth. ABOVE the cap the rake is FLAT, so growing the
# pot is rake-FREE and the escalation must NOT be discounted (over-bets above the cap are rake-efficient).
# We therefore DAMPEN the _K_BIG/_K_MID boosts by ``rake_keep`` in (0,1]: 1.0 = keep the full escalation
# (no rake bite, or no context), shrinking toward 0 only when the marginal rake of the mid->big step
# rivals the marginal extraction. Default-safe: no context (no stake/pot) => rake_keep = 1.0 (today).

# Representative pot-fraction of the mid band (1/3..0.75) and the big band (1.0..2.0) the escalation
# shifts mass between — the medians of BET_SIZE_BUCKETS' lower/upper halves (0.66 and 1.5 for the
# standard 7-bucket menu). Computed from the single-source-of-truth menu so it tracks any re-bucketing.
_MID_BUCKETS = [b for b in BET_SIZE_BUCKETS if b < 1.0] or [BET_SIZE_BUCKETS[0]]
_BIG_BUCKETS = [b for b in BET_SIZE_BUCKETS if b >= 1.0] or [BET_SIZE_BUCKETS[-1]]
_MID_FRAC = float(np.median(_MID_BUCKETS))   # ~0.66 pot
_BIG_FRAC = float(np.median(_BIG_BUCKETS))   # ~1.5  pot

_RAKE_EPS = 1e-6


def _rake_aware_enabled() -> bool:
    return os.environ.get("POKERBOT_RAKE_AWARE_SIZING", "1") != "0"


@dataclass(frozen=True)
class RakeContext:
    """Postflop rake context for the SIZE-escalation discount. Optional everywhere — absent (or preflop)
    => no discount (rake_keep == 1.0, today's behaviour byte-for-byte)."""
    pot_bb: float                 # whole-hand pot BEFORE hero's value bet, in BB
    n_dealt: int                  # ACTUAL players dealt this hand (cap tier) — never max-seats
    stake_key: str                # a key in rake.STAKES (e.g. "0.10/0.25")
    is_preflop: bool = False      # True => rake is a postflop sink that hasn't applied yet => no discount


def _pot_after_bet(pot_before_bb: float, frac: float, call_through: float) -> float:
    """Final pot after hero bets ``frac``·pot and the station calls through with prob ``call_through``.

    Hero wagers ``frac·P``; the station matches ``call_through·frac·P`` in expectation. Expected final
    pot = ``P·(1 + frac·(1 + call_through))``. (call_through=1 => P·(1+2·frac); a pure fold => P+frac·P.)"""
    return pot_before_bb * (1.0 + frac * (1.0 + call_through))


def rake_keep_factor(stats, ctx: RakeContext | None) -> float:
    """Magnitude-scaling factor in (0,1] applied to the SIZE-escalation boost (_K_BIG/_K_MID).

    ``rake_keep = 1 - clip(marginal_rake / max(marginal_extraction, eps), 0, 1)`` where
      * ``marginal_rake`` = ``rake.marginal_rake_bb`` of moving from the mid-size resulting pot to the
        big-size resulting pot (the extra rake the escalation actually incurs). Above the cap this is 0
        (flat rake) => no discount; below the cap it is 5%·(pot growth) => the discount bites.
      * ``marginal_extraction`` = a CONSERVATIVE estimate of the extra value the bigger size extracts =
        ``call_through · (big_frac - mid_frac) · pot`` — the station's expected call-through on the size
        increment. Conservative (underestimating extraction) makes the discount a SOFT scaling, not a
        hard block: even when marginal rake equals the estimate we only floor at rake_keep>0 below.

    Default-safe: no context, preflop, missing stake, or the kill-switch off => 1.0 (no discount)."""
    if ctx is None or not _rake_aware_enabled():
        return 1.0
    if ctx.is_preflop or ctx.pot_bb <= 0.0:
        return 1.0
    try:
        from pokerbot.economics.rake import STAKES, marginal_rake_bb
    except Exception:  # noqa: BLE001 — economics layer optional; degrade to no-discount
        return 1.0
    if ctx.stake_key not in STAKES:
        return 1.0

    # Station call-through on the value bet: how often they pay it off (the same sticky read that drives
    # the escalation). Use the payoff score as a proxy for continue-frequency, floored so extraction is
    # never zero (a value hand always extracts SOMETHING) — this keeps the discount conservative.
    score, _conf = station_payoff_score(stats)
    call_through = float(np.clip(0.5 + 0.5 * score, 0.0, 1.0))   # 0.5 (neutral) .. 1.0 (pure station)

    pot_mid = _pot_after_bet(ctx.pot_bb, _MID_FRAC, call_through)
    pot_big = _pot_after_bet(ctx.pot_bb, _BIG_FRAC, call_through)
    marg_rake = marginal_rake_bb(pot_mid, pot_big, ctx.n_dealt, ctx.stake_key)

    # Conservative extraction estimate: the station calls ``call_through`` of the extra wager
    # (big_frac - mid_frac)·pot. Floor at eps so a degenerate pot can't divide-by-zero.
    marg_extract = max(call_through * (_BIG_FRAC - _MID_FRAC) * ctx.pot_bb, _RAKE_EPS)
    discount = float(np.clip(marg_rake / marg_extract, 0.0, 1.0))
    return float(np.clip(1.0 - discount, 0.0, 1.0))


def overlay_hand_strength(state, seat, *, n_iter: int = 300) -> float | None:
    """BOARD-AWARE hero strength for the value-sizing / bluff-suppression overlays — equity vs a random
    hand, so a genuinely MADE hand (top pair, two pair, a straight) reads as value and unimproved high
    cards (ace-high air) read as air. This is the SINGLE source of truth used by BOTH the live overlay
    (root_mixer.apply_live_value_sizing) and the sim battery, so the sim measures exactly the live value/
    air separation rather than the preflop-rank proxy (which scored AKo on a missed board ~0.90 = top
    pair). Preflop (no board) falls back to the preflop strength proxy; deterministic (fixed MC seed) so
    it is CRN-safe; any failure -> None so the overlay defers."""
    try:
        player = state.players[seat]
        if player.hole is None:
            return None
        board = [c for c in (getattr(state, "board", None) or []) if c is not None]
        if len(board) < 3:
            from pokerbot.encoding.weva_encoder import _preflop_strength
            return float(_preflop_strength(player.hole))
        from pokerbot.encoding.postflop_bucketer import expected_hand_strength
        hole_ids = (int(player.hole[0].index), int(player.hole[1].index))
        board_ids = tuple(int(c.index) for c in board[:5])
        return float(expected_hand_strength(hole_ids, board_ids, n_iter=n_iter))
    except Exception:  # noqa: BLE001 — strength is best-effort; the overlay defers on None
        return None


def _enabled() -> bool:
    return os.environ.get("POKERBOT_VALUE_SIZING", "1") != "0"


def _bluff_suppress_enabled() -> bool:
    return os.environ.get("POKERBOT_BLUFF_SUPPRESS", "1") != "0"


def _river_thin_value_enabled() -> bool:
    # Default OFF: this is the one overlay with genuine value-cut risk (betting into a range that may
    # beat us), so it ships behind a flag pending CRN-battery validation before any live enablement.
    return os.environ.get("POKERBOT_RIVER_THIN_VALUE", "0") != "0"


def _pct(hud: dict, *keys):
    """Read a HUD stat as a fraction in [0,1]. CoinPoker server stats arrive as integer percents
    (e.g. wtsd=31); accept already-normalized values too."""
    for k in keys:
        v = hud.get(k)
        if v is None:
            continue
        try:
            f = float(v)
        except (TypeError, ValueError):
            continue
        return f / 100.0 if f > 1.0 else max(0.0, f)
    return None


def station_payoff_score(stats) -> tuple[float, float]:
    """Resolve ``(score, confidence)`` in [0,1]² — how much this opponent is a sticky calling station
    with a WEAK showdown range (i.e. how much over-betting value pays). Prefers the server HUD
    (whole-history, dense); falls back to the derived count fields; neutral when blind.

      score = 0.6·stickiness + 0.4·showdown_weakness
        stickiness       = how far BELOW _FOLD_REF they fold to a bet/c-bet (sticky ⇒ high)
        showdown_weakness = how far BELOW _WSD_REF their won-at-showdown is (pay-off ⇒ high)

    Aggressive/tight regs (lag/tag) damp the score (escalating value-size into them risks a raise or a
    fold = value-cut); a confirmed calling-station archetype gives the score a floor."""
    hud = getattr(stats, "hud", None) or {}
    wsd = _pct(hud, "wsd")
    ftc = _pct(hud, "foldToCbet", "foldtocbet")
    fta = _pct(hud, "fta", "fold")
    have_hud = any(v is not None for v in (wsd, ftc, fta, _pct(hud, "wtsd")))

    # derived fold-to-cbet fallback (count-based) when the server stat is absent
    cf = float(getattr(stats, "cbet_faced", 0) or 0)
    if ftc is None and cf > 0:
        ftc = float(getattr(stats, "fold_to_cbet", 0) or 0) / cf

    folds = [v for v in (ftc, fta) if v is not None]
    sticky = max(0.0, min(1.0, (_FOLD_REF - min(folds)) / _FOLD_REF)) if folds else 0.0
    # Value-cut guard fails CLOSED: when the showdown-strength stat is absent we cannot confirm the
    # opponent shows up weak, so the weakness term is 0 (was a 0.5 "neutral" that let the overlay
    # over-escalate value into an unknown — possibly strong — showdown range). The calling_station
    # archetype floor below still recognizes a count-confirmed station.
    weak = max(0.0, min(1.0, (_WSD_REF - wsd) / _WSD_SPAN)) if wsd is not None else 0.0

    score = 0.6 * sticky + 0.4 * weak
    arch = stats.archetype() if hasattr(stats, "archetype") else "unknown"
    if arch == "calling_station":
        score = max(score, 0.5)             # corroborated payoff target -> floor
    elif arch in ("lag", "tag"):
        score *= 0.3                          # don't escalate value-size into a reg (raise/fold risk)

    if have_hud:
        conf = 0.9                            # server stat = whole-history, high confidence
    else:
        n = max(cf, float(getattr(stats, "saw_flop", 0) or 0),
                float(getattr(stats, "hands", 0) or 0) / 2.0)
        conf = min(1.0, n / _MIN_N)
    return float(max(0.0, min(1.0, score))), float(max(0.0, min(1.0, conf)))


def value_sizing_logits(
    stats, hand_strength: float | None, *, rake_ctx: RakeContext | None = None
) -> np.ndarray:
    """Logit overlay that escalates OUR value bet-SIZE vs a payoff station. Zero unless hero holds a
    value hand AND the opponent reads as a sticky/weak-showdown station with enough confidence. Bounded
    and additive — the mixer's confidence-α and the LAG/RNR firewall still clamp it on top.

    ``rake_ctx`` (optional, keyword-only, default None): postflop rake context (pot/dealt/stake). When
    supplied, the SIZE-escalation boost is DAMPENED by ``rake_keep_factor`` so we don't over-escalate
    into a pot where the marginal rake (5% below the cap) eats the marginal extraction. ABOVE the cap
    rake is flat ⇒ rake_keep→1 ⇒ no discount. No context / preflop / kill-switch off ⇒ rake_keep == 1.0,
    i.e. BYTE-IDENTICAL to the no-rake-context behaviour. The discount only scales the escalation
    MAGNITUDE; it is not a safety bound (the mixer α-cap and LAG/RNR firewall remain the hard clamps)."""
    adj = np.zeros(NUM_ACTIONS, dtype=np.float64)
    if not _enabled() or NUM_ACTIONS < 5:
        return adj
    h = 0.55 if hand_strength is None else float(np.clip(hand_strength, 0.0, 1.0))
    value = max(0.0, (h - 0.5) * 2.0)         # only value hands escalate (air never over-bets here)
    if value <= 0.0:
        return adj
    score, conf = station_payoff_score(stats)
    mag = score * conf * value                # in [0,1]
    if mag <= 0.0:
        return adj
    rake_keep = rake_keep_factor(stats, rake_ctx)   # 1.0 (no discount) by default; <1 below the cap
    adj[_BIG_LO:_ALL_IN] += _K_BIG * mag * rake_keep   # over-bet (>=1x pot, excl all-in); rake-discounted
    adj[_BET_LO:_BIG_LO] += _K_MID * mag * rake_keep   # mid bets; same rake discount on the size boost
    adj[1] -= _K_CHECK * mag                    # don't check back value vs a station (read, not a size)
    return adj


def bluff_suppression_logits(stats, hand_strength: float | None) -> np.ndarray:
    """Logit overlay that SUPPRESSES our bluff frequency vs a sticky calling station — the symmetric
    counterpart to ``value_sizing_logits``. The population leak #1 is "a sub-42% folder makes most
    bluffs −EV" (a bluff needs fold ≥ bet/(bet+pot); a 36% folder fails that for most sizes). So when
    hero holds AIR (strength < 0.5) and the opponent reads as a non-folding station, push mass OFF the
    bet/raise buckets and toward checking/giving up. Zero unless hero holds air AND the opponent is a
    confident sticky station. Negative-risk: betting less air vs a non-folder cannot be punished by the
    station, and the magnitude auto-damps vs lag/tag (``station_payoff_score`` ×0.3). Bounded + additive,
    so the mixer α-cap + LAG/RNR firewall still clamp it. Env kill-switch ``POKERBOT_BLUFF_SUPPRESS=0``."""
    adj = np.zeros(NUM_ACTIONS, dtype=np.float64)
    if not _bluff_suppress_enabled() or NUM_ACTIONS < 5:
        return adj
    h = 0.55 if hand_strength is None else float(np.clip(hand_strength, 0.0, 1.0))
    air = max(0.0, (0.5 - h) * 2.0)            # only air suppresses (value never suppresses)
    if air <= 0.0:
        return adj
    score, conf = station_payoff_score(stats)
    mag = score * conf * air                   # in [0,1]
    if mag <= 0.0:
        return adj
    adj[_BET_LO:_ALL_IN] -= _K_BLUFF * mag      # de-prioritize bluff bets across sizes (they call anyway)
    adj[1] += _K_BLUFF_CHECK * mag              # prefer checking/giving up air vs a non-folder
    return adj


def river_thin_value_logits(stats, hand_strength: float | None, board) -> np.ndarray:
    """River-only thin-value EXPANSION: bet marginally-ahead hands (hand_strength in [_THIN_LO,_THIN_HI),
    e.g. 2nd pair / weak top pair) that the GTO base checks back, when the opponent reads as a confirmed
    weak-showdown station. On the river a check-back forfeits value with no protective upside, and a
    station calls with worse, so a thin value bet prints. This is a bet-FREQUENCY expansion (small/mid
    sizes), complementing value_sizing_logits' SIZE escalation for already-value hands.

    Guards (this is the one overlay with real value-cut risk — betting into a range that may beat us):
      * RIVER only (len(board)==5) — no card to come, so checking has no equity-denial value;
      * marginal band only (_THIN_LO..._THIN_HI) — air never thin-value-bets, value is value_sizing's job;
      * weak-showdown gate — require server WSD < _WSD_THIN_MAX (we must be ahead when called);
      * station_payoff_score stickiness/confidence — never fires vs a reg (lag/tag damped ×0.3).
    Small/mid sizes only (big bets fold out worse / get raised). Default OFF (POKERBOT_RIVER_THIN_VALUE)."""
    adj = np.zeros(NUM_ACTIONS, dtype=np.float64)
    if not _river_thin_value_enabled() or NUM_ACTIONS < 5:
        return adj
    if len([c for c in (board or []) if c is not None]) != 5:   # river only
        return adj
    if hand_strength is None:                   # fail closed: never thin-value-bet an UNKNOWN hand
        return adj
    h = float(np.clip(hand_strength, 0.0, 1.0))
    if not (_THIN_LO <= h < _THIN_HI):          # only the marginal-ahead band
        return adj
    thin = (h - _THIN_LO) / (_THIN_HI - _THIN_LO)   # 0 at floor -> 1 at the value cutoff
    # Value-cut guard: require a genuinely weak showdown range (we beat their calls). Absent server WSD,
    # do NOT fire (no blind thin value).
    wsd = _pct(getattr(stats, "hud", None) or {}, "wsd")
    if wsd is None or wsd >= _WSD_THIN_MAX:
        return adj
    score, conf = station_payoff_score(stats)
    mag = score * conf * thin
    if mag <= 0.0:
        return adj
    adj[_BET_LO:_BIG_LO] += _K_THIN * mag        # small/mid value sizes only (don't over-bet 2nd pair)
    adj[1] -= _K_THIN_CHECK * mag                # don't check back a thin value hand vs a station
    return adj
