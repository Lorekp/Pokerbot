"""SIZE-ELASTIC calibrated opponents — the faithful exploitation-uplift field for the value-SIZING lever.

WHY this module exists (the #1 substantive eval fix):

  The deployed cash edge is value-EXTRACTION: vs a loose-passive payoff station the hero bets BIGGER
  for value (``pokerbot.opponent.value_sizing`` — the HUD value-SIZING overlay), because the station
  continues past MDF and pays off the over-bet. To MEASURE that overlay, the eval field's continue/fold
  decision must DEPEND ON BET SIZE: a bigger value bet should get called MORE OFTEN (in absolute EV)
  than a smaller one against a station.

  But every existing eval villain is SIZE-BLIND. ``CalibratedOpponentBot._postflop`` (calibrated_opponents
  .py:218-237) and the ``IRCVillain`` family decide continue/fold purely on hand strength vs a fixed
  ``call_vs_bet_hs`` threshold — they never read ``to_call``/``pot``. So whether the hero value-bets 1/3
  pot or 2x pot, the station calls with the SAME range, and the only EV difference is the size itself on
  the hands it would have called anyway. That makes the value-SIZING overlay's measured uplift ~0 — not
  because the overlay is worthless, but because the field cannot express the mechanism the overlay
  exploits. The sizing/frequency overlays are literally invisible to the current battery.

  ``ResponsiveStation`` fixes that. It is a thin subclass of ``CalibratedOpponentBot`` that overrides ONLY
  the postflop FACING branch to make the continue probability a function of (hand strength, bet size):

      size_frac = to_call / pot                # the bet as a fraction of the pot at decision time
      mdf(size_frac) = size_frac / (1 + size_frac)   # minimum-defense frequency (== pot_odds here)
      p_call(hs, size_frac) = clip(equity_proxy(hs) - (mdf(size_frac) - slack), 0, 1)

  As ``size_frac`` rises, ``mdf`` rises, so ``p_call`` FALLS — the bigger the bet, the more of the
  bottom of the range folds. That is the elasticity the size-blind field lacks.

  Two archetypes set ``slack`` (the realistic over/under-defense vs the GTO MDF baseline):
    * ``ResponsiveStation`` (slack > 0): over-calls PAST MDF (the loose payoff fish) — its showdown
      range is so wide it still pays off over-bets, EXCEPT the very bottom (pure air) which folds to an
      over-bet. This is the realistic "calls a half-pot bet with bottom pair, folds the busted draw to a
      2x over-bet" boundary, and it is exactly what makes a BIGGER value bet net MORE: the over-bet is
      still paid by the wide made-hand range, so escalating size strictly increases extraction.
    * ``ResponsiveReg`` (slack = 0): defends ~MDF (the do-no-harm control) — value-sizing escalation
      against it does NOT print extra, because it folds enough of its range to make each size equal-EV.
      A reg seat is how we confirm the overlay is FIELD-SPECIFIC, not a free EV gift everywhere.

  The bot's OWN bets are routed through a ported ``irc_villains._pick_sized_bet`` so it expresses realistic
  pot-fraction sizes in BOTH directions (``CalibratedOpponentBot._pick_bet`` just grabs ``bets[-1]`` — the
  max legal raise — which over-states its aggression sizing). The IRC-percentile HUD calibration (VPIP /
  PFR / AF / WTSD / WSD / fold_to_cbet) is INHERITED UNCHANGED from ``CalibratedOpponentBot``; only the
  facing continue/fold rule is replaced. Preflop, unopened betting, and value-raising are all unchanged.

This is a VILLAIN: there is no rake and no variance reduction here (those belong to the hero/measurement
side). Wires additively into ``bots.build_bot`` and ``exploitation_battery`` without touching any existing
field definition. Nothing here touches the aof_* engine or the firewall modules.
"""

from __future__ import annotations

import random

from pokerbot.engine.actions import Action, ActionType
from pokerbot.engine.game import GameState
from pokerbot.eval.calibrated_opponents import CalibratedOpponentBot

# Reuse the IRC villain's realistic pot-fraction sizing verbatim (single source of truth: do NOT
# re-implement). It returns the legal BET/RAISE closest to a target pot-fraction, jams only when the
# intended bet already commits a large stack share, else returns None (caller checks/calls).
from pokerbot.eval.irc_villains import _pick_sized_bet

# Default own-bet pot fractions (value bets and the rare bluff). Realistic loose-passive sizing: small
# value bets, slightly bigger bluffs (still modest — a station does not over-bet bluff). These mirror the
# IRCStation defaults so a ResponsiveStation's OWN aggression sizing matches the size-blind station's.
_VALUE_SIZE_FRAC = 0.5
_BLUFF_SIZE_FRAC = 0.5


def mdf(size_frac: float) -> float:
    """Minimum-defense frequency for a bet of ``size_frac`` of the pot: ``size_frac / (1 + size_frac)``.

    This is the fraction of its range a defender must continue with to make the bettor's pure-bluff
    indifferent. Equivalently it is the bettor's pot odds (``to_call / (pot + to_call)``) when
    ``size_frac == to_call / pot``, so a defender calling EXACTLY ``mdf`` is the GTO non-exploitable
    baseline; a station over-defends (calls MORE than ``mdf``), a nit under-defends (calls less)."""
    s = max(0.0, float(size_frac))
    return s / (1.0 + s)


def equity_proxy(hs: float) -> float:
    """Map the engine hand-strength proxy to a [0,1] continue-equity. ``_hand_strength`` is already an
    equity-like proxy in [0,1] (equity-vs-random for made hands), so this is the identity clamp — kept as
    a named seam so the elasticity formula reads as 'equity minus the size-implied fold pressure'."""
    return max(0.0, min(1.0, float(hs)))


def facing_size_frac(state: GameState, seat: int) -> float | None:
    """The bet the villain at ``seat`` is FACING, as a fraction of the pot at decision time.

    ``to_call = current_bet_bb - committed_street_bb`` (what this seat must add to continue); the pot at
    decision time (``state.pot_bb``) already INCLUDES the bettor's chips, so ``size_frac = to_call / pot``
    gives ``mdf(size_frac) == pot_odds`` exactly (see ``mdf``). Returns None when not actually facing a
    bet (``to_call <= 0``) or the pot is degenerate, so the caller defers to the inherited rule."""
    player = state.players[seat]
    to_call = float(state.current_bet_bb) - float(player.committed_street_bb)
    pot = float(getattr(state, "pot_bb", 0.0) or 0.0)
    if to_call <= 1e-9 or pot <= 1e-9:
        return None
    return to_call / pot


def p_call(hs: float, size_frac: float, slack: float) -> float:
    """Size-ELASTIC continue probability: ``clip(equity_proxy(hs) - (mdf(size_frac) - slack), 0, 1)``.

    Strictly DECREASING in ``size_frac`` (mdf rises with size), so a bigger bet folds out more of the
    bottom of the range — the elasticity the size-blind field lacks. ``slack > 0`` shifts the whole curve
    UP (the station over-calls past MDF: its wide made-hand range still pays the over-bet, only the very
    bottom folds); ``slack == 0`` defends ~MDF (the reg control)."""
    return max(0.0, min(1.0, equity_proxy(hs) - (mdf(size_frac) - float(slack))))


class ResponsiveStation(CalibratedOpponentBot):
    """SIZE-ELASTIC loose-passive station (the faithful value-SIZING target).

    Identical to ``CalibratedOpponentBot`` EXCEPT the postflop FACING branch, where continue/fold is a
    function of (hand strength, FACING BET SIZE) via ``p_call`` with ``slack > 0`` (over-defends past
    MDF). Inherits the IRC-percentile HUD calibration unchanged (seed from the recreational/weak_reg
    tier). Its OWN bets are routed through ``_pick_sized_bet`` so it expresses realistic pot fractions.

    The over-defense ``slack`` is the realistic boundary: the station pays off value over-bets (its wide
    showdown range continues), but the very bottom (pure air) folds to an over-bet — so escalating value
    SIZE strictly increases the hero's extraction, which is the mechanism the value-SIZING overlay
    exploits and the size-blind field cannot express."""

    #: Over-defense above the GTO MDF baseline (continue-prob offset). >0 ⇒ calls past MDF (a station).
    slack: float = 0.18
    #: OWN-bet pot fractions (value / bluff). Routed through ``_pick_sized_bet`` for realistic sizing.
    value_size: float = _VALUE_SIZE_FRAC
    bluff_size: float = _BLUFF_SIZE_FRAC

    def __init__(self, rng: random.Random | None = None, *, slack: float | None = None,
                 value_size: float | None = None, bluff_size: float | None = None, **kwargs) -> None:
        super().__init__(rng, **kwargs)
        if slack is not None:
            self.slack = float(slack)
        if value_size is not None:
            self.value_size = float(value_size)
        if bluff_size is not None:
            self.bluff_size = float(bluff_size)

    # -- the ONE overridden branch: size-elastic facing continue/fold --------------------------------
    def _postflop(self, state: GameState, seat: int, legal, hs: float) -> Action:
        types = {a.action_type for a in legal}
        facing = ActionType.CALL in types
        if facing:
            # value-raise the very top (kept identical to the inherited rule so AF/raise structure is
            # unchanged — only the call/fold split below becomes size-elastic).
            if hs >= 0.86 and self.rng.random() < 0.25:
                return self._sized_bet_or(state, seat, legal, self.value_size,
                                          fallback=ActionType.CALL)
            sf = facing_size_frac(state, seat)
            if sf is None:  # not genuinely facing a sized bet -> inherited threshold rule
                return super()._postflop(state, seat, legal, hs)
            if self.rng.random() < p_call(hs, sf, self.slack):
                return self._take(legal, ActionType.CALL)
            return self._take(legal, ActionType.FOLD)
        # unopened: value-bet / bluff / check — same frequencies as the parent, but SIZED realistically.
        if hs >= self._value_bet_hs and self.rng.random() < self._bet_freq:
            return self._sized_bet_or(state, seat, legal, self.value_size)
        if hs < self._bluff_bet_hs and self.rng.random() < self._bluff_freq:
            return self._sized_bet_or(state, seat, legal, self.bluff_size)
        return self._take(legal, ActionType.CHECK, ActionType.FOLD)

    # -- helpers -------------------------------------------------------------------------------------
    @staticmethod
    def _take(legal, *types_pref):
        for t in types_pref:
            a = next((a for a in legal if a.action_type == t), None)
            if a is not None:
                return a
        return legal[0]

    def _sized_bet_or(self, state: GameState, seat: int, legal, target_frac: float,
                      *, fallback: ActionType = ActionType.CHECK) -> Action:
        """Bet at a realistic pot fraction via the ported ``_pick_sized_bet``; if no sized bet is legal
        (pot too small for min_raise / no jam) fall back to ``fallback`` then CHECK/CALL/FOLD."""
        a = _pick_sized_bet(state, seat, target_frac)
        if a is not None:
            return a
        return self._take(legal, fallback, ActionType.CHECK, ActionType.CALL, ActionType.FOLD)


class ResponsiveReg(ResponsiveStation):
    """SIZE-ELASTIC reg control (do-no-harm): defends ~MDF (``slack == 0``). Identical machinery to
    ``ResponsiveStation`` but with no over-defense, so the value-SIZING escalation does NOT print extra
    against it (each size is ~equal-EV at the MDF boundary). Seed from the solid_reg HUD tier. This is
    the FIELD-SPECIFICITY control: it proves the overlay's uplift comes from the station's over-defense,
    not from a free EV gift that any size-elastic opponent would hand over."""

    slack: float = 0.0
    value_size: float = 0.66
    bluff_size: float = 0.75


# --- tier-seeded factories ------------------------------------------------------------------------
# Seed the HUD calibration from the IRC skill tiers (the SAME population JSON the calibrated opponents
# use). recreational/weak_reg ⇒ the loose-passive station; solid_reg ⇒ the disciplined reg control.
_STATION_TIER = "weak_reg"   # the modal soft-field player the value-sizing lever targets
_REG_TIER = "solid_reg"      # disciplined winning reg (do-no-harm control)


def _tier_hud(tier: str) -> dict:
    """The inherited IRC-percentile HUD target vector for a tier (VPIP/PFR/AF/WTSD/WSD/fold_to_cbet),
    passed verbatim to ``CalibratedOpponentBot`` so the realized frequencies match the population."""
    from pokerbot.eval.calibrated_opponents import tier_target

    return dict(tier_target(tier))


def build_responsive_station(rng: random.Random | None = None, params: dict | None = None,
                             *, tier: str = _STATION_TIER) -> ResponsiveStation:
    """Build a size-elastic station with the IRC ``tier`` HUD calibration. ``params`` overrides HUD stats
    and/or the responsiveness knobs (slack / value_size / bluff_size)."""
    target = _tier_hud(tier)
    if params:
        target.update(params)
    knob_keys = ("slack", "value_size", "bluff_size")
    knobs = {k: target.pop(k) for k in knob_keys if k in target}
    return ResponsiveStation(rng, **knobs, **target)


def build_responsive_reg(rng: random.Random | None = None, params: dict | None = None,
                         *, tier: str = _REG_TIER) -> ResponsiveReg:
    """Build a size-elastic reg control (slack=0) with the IRC ``tier`` HUD calibration."""
    target = _tier_hud(tier)
    if params:
        target.update(params)
    knob_keys = ("slack", "value_size", "bluff_size")
    knobs = {k: target.pop(k) for k in knob_keys if k in target}
    return ResponsiveReg(rng, **knobs, **target)


def build_responsive_bot(name: str, rng: random.Random | None = None, params: dict | None = None):
    """Factory for the responsive (size-elastic) field bots. Unknown names raise."""
    if name == "responsive_station":
        return build_responsive_station(rng, params)
    if name == "responsive_reg":
        return build_responsive_reg(rng, params)
    raise ValueError(f"unknown responsive bot {name!r}; "
                     "known: ['responsive_reg', 'responsive_station']")


_RESPONSIVE_NAMES = ("responsive_station", "responsive_reg")
