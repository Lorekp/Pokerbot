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


def _enabled() -> bool:
    return os.environ.get("POKERBOT_VALUE_SIZING", "1") != "0"


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
