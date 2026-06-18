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

import numpy as np

from pokerbot.engine.actions import NUM_ACTIONS

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
    weak = max(0.0, min(1.0, (_WSD_REF - wsd) / _WSD_SPAN)) if wsd is not None else 0.5  # neutral

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


def value_sizing_logits(stats, hand_strength: float | None) -> np.ndarray:
    """Logit overlay that escalates OUR value bet-SIZE vs a payoff station. Zero unless hero holds a
    value hand AND the opponent reads as a sticky/weak-showdown station with enough confidence. Bounded
    and additive — the mixer's confidence-α and the LAG/RNR firewall still clamp it on top."""
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
    adj[_BIG_LO:_ALL_IN] += _K_BIG * mag       # push toward the over-bet (>=1x pot, excl. all-in)
    adj[_BET_LO:_BIG_LO] += _K_MID * mag       # keep mid bets live so total betting still rises
    adj[1] -= _K_CHECK * mag                    # don't check back value vs a station
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
    h = 0.45 if hand_strength is None else float(np.clip(hand_strength, 0.0, 1.0))
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
