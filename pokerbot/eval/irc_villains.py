"""IRC-calibrated realistic-betting villains for the Phase-2 value-extraction gauntlet.

WHY these exist (separate from the hs_* eval bots): the value-EXTRACTION exploit (value-bet stations
thinner / more) is OFF on the blueprint+search base because it can't be gated do-no-harm — the read
can't tell a value-only station from a bluffy lag. The diagnosed cause is that the hs_* bots express
aggression poorly: when a pot is too small for any SIZED bet to clear ``min_raise``, ``bots._pick_bet``
silently CHECKS (task #12), flattening every archetype's own-initiative aggression to "passive" so the
read is blind.

These villains fix bet EXPRESSION (the isolated, no-blast-radius approach): each archetype bets at a
realistic POT FRACTION via ``_pick_sized_bet`` — picking the legal sizing closest to its target, jamming
only when the intended bet already commits a large share of stack, and checking back only genuinely tiny
pots (which is itself realistic). Calibrated to the IRC population (loose-passive: VPIP~48%, PFR~10%,
AF~0.65; see data/player_profiles/irc_population_prior.json), the archetypes now differ in OWN-INITIATIVE
aggression the way real players do:
  * irc_station — loose-passive whale: calls very wide, value-bets only strong hands SMALL, ~no bluffs
    (the prime thin-value target).
  * irc_lag — loose-aggressive: bets wide incl. real bluffs BIG, calls down light (the do-no-harm danger
    the gate must defer to).
  * irc_nit — tight-passive: folds most, rare bets are value.
  * irc_tag — tight-aggressive control (the engine should find little; protection reference).

Used ONLY by the value-extraction gauntlet via ``build_irc_villain``; ``pokerbot.eval.bots`` and every
existing eval/base-strength measurement are untouched.
"""

from __future__ import annotations

import random

from pokerbot.engine.actions import Action, ActionType
from pokerbot.engine.game import GameState
from pokerbot.eval.bots import _hand_strength


def _pick_sized_bet(state: GameState, seat: int, target_frac: float,
                    jam_frac: float = 0.75) -> Action | None:
    """Express a bet as a realistic POT FRACTION. Returns the legal BET/RAISE whose size is closest to
    ``target_frac`` of the pot; if no sized bet is legal (pot too small for min_raise) it jams ONLY when
    the intended bet (target_frac*pot) already commits >= ``jam_frac`` of stack — else returns None so
    the caller checks/calls (a human doesn't shove 100bb into a 2bb pot; they check it back)."""
    legal = state.legal_actions(seat)
    pot = max(float(getattr(state, "pot_bb", 0.0) or 0.0), 1e-9)
    sized = [a for a in legal if a.action_type in (ActionType.BET, ActionType.RAISE)]
    if sized:
        return min(sized, key=lambda a: abs((float(a.amount_bb) / pot) - target_frac))
    allin = next((a for a in legal if a.action_type == ActionType.ALL_IN), None)
    if allin is not None:
        stack = float(state.players[seat].stack_bb)
        if target_frac * pot >= jam_frac * max(stack, 1e-9):
            return allin
    return None


class IRCVillain:
    """Parameterized realistic-betting villain. Subclasses set the archetype thresholds; all betting
    goes through ``_pick_sized_bet`` so aggression is expressed at a human pot fraction, not capped to
    a forced check."""

    # Preflop play
    pre_fold_hs = 0.0          # fold hands weaker than this preflop when facing chips (0 = call any)
    # Postflop, facing a bet
    call_hs = 0.30             # call (continue) with at least this equity
    raise_hs = 0.80            # value-raise threshold
    raise_freq = 0.0           # P(raise | hs>=raise_hs)
    bluff_raise_freq = 0.0     # P(bluff-raise | weak)
    # Postflop, unopened (we can bet/check)
    bet_hs = 0.75              # value-bet threshold
    bet_freq = 0.5             # P(bet | hs>=bet_hs)
    bluff_bet_freq = 0.0       # P(bet | weak) — own-initiative bluffing
    # Sizing (pot fraction) for value bets and bluffs
    value_size = 0.55
    bluff_size = 0.66

    def __init__(self, rng: random.Random | None = None, **params: float) -> None:
        self.rng = rng or random.Random()
        for k, v in params.items():
            if not hasattr(self, k):
                raise ValueError(f"{type(self).__name__}: unknown param {k!r}")
            setattr(self, k, float(v))

    @staticmethod
    def _take(legal, *types_pref):
        for t in types_pref:
            a = next((a for a in legal if a.action_type == t), None)
            if a is not None:
                return a
        return legal[0]

    def _bet_or(self, state, seat, legal, target_frac):
        """Try to make a sized bet; fall back to check (realistic for an un-bettable tiny pot)."""
        a = _pick_sized_bet(state, seat, target_frac)
        return a if a is not None else self._take(legal, ActionType.CHECK, ActionType.FOLD)

    def __call__(self, state: GameState) -> Action:
        seat = state.action_seat
        legal = state.legal_actions(seat)
        if not legal:
            return Action(ActionType.CHECK)
        hs = _hand_strength(state, seat)
        types = {a.action_type for a in legal}
        preflop = state.street.name == "PREFLOP"

        if ActionType.CALL in types:  # facing a bet/raise
            if preflop and hs < self.pre_fold_hs:
                return self._take(legal, ActionType.FOLD)
            if hs >= self.raise_hs and self.rng.random() < self.raise_freq:
                a = _pick_sized_bet(state, seat, self.value_size)
                if a is not None:
                    return a
            if hs >= self.call_hs:
                return self._take(legal, ActionType.CALL)
            if self.bluff_raise_freq and self.rng.random() < self.bluff_raise_freq:
                a = _pick_sized_bet(state, seat, self.bluff_size)
                if a is not None:
                    return a
            return self._take(legal, ActionType.FOLD)

        # Unopened: bet (value or bluff) or check.
        if hs >= self.bet_hs and self.rng.random() < self.bet_freq:
            return self._bet_or(state, seat, legal, self.value_size)
        if self.bluff_bet_freq and self.rng.random() < self.bluff_bet_freq:
            return self._bet_or(state, seat, legal, self.bluff_size)
        return self._take(legal, ActionType.CHECK, ActionType.FOLD)


class IRCStation(IRCVillain):
    """Loose-passive whale: calls very wide, value-bets only strong hands SMALL, ~no bluffs."""
    pre_fold_hs = 0.0
    call_hs = 0.18
    raise_hs = 0.88
    raise_freq = 0.15
    bet_hs = 0.72
    bet_freq = 0.45
    bluff_bet_freq = 0.02
    value_size = 0.5
    bluff_size = 0.5


class IRCLag(IRCVillain):
    """Loose-aggressive: bets wide incl. real bluffs BIG, calls down light, occasional bluff-raise."""
    pre_fold_hs = 0.18
    call_hs = 0.34
    raise_hs = 0.70
    raise_freq = 0.5
    bluff_raise_freq = 0.18
    bet_hs = 0.42
    bet_freq = 0.7
    bluff_bet_freq = 0.30
    value_size = 0.75
    bluff_size = 0.9


class IRCNit(IRCVillain):
    """Tight-passive: folds most preflop, rare bets are value, releases to aggression."""
    pre_fold_hs = 0.52
    call_hs = 0.6
    raise_hs = 0.85
    raise_freq = 0.35
    bet_hs = 0.78
    bet_freq = 0.5
    bluff_bet_freq = 0.0
    value_size = 0.5
    bluff_size = 0.5


class IRCTag(IRCVillain):
    """Tight-aggressive control: value-bets strong, folds weak, defends reasonably (do-no-harm ref)."""
    pre_fold_hs = 0.45
    call_hs = 0.5
    raise_hs = 0.74
    raise_freq = 0.5
    bluff_raise_freq = 0.08
    bet_hs = 0.58
    bet_freq = 0.65
    bluff_bet_freq = 0.12
    value_size = 0.66
    bluff_size = 0.75


_IRC_REGISTRY = {
    "irc_station": IRCStation,
    "irc_lag": IRCLag,
    "irc_nit": IRCNit,
    "irc_tag": IRCTag,
}


def build_irc_villain(name: str, rng: random.Random | None = None, params: dict | None = None):
    """Factory for the IRC realistic villains. Unknown names raise (the gauntlet validates specs)."""
    cls = _IRC_REGISTRY.get(name)
    if cls is None:
        raise ValueError(f"unknown IRC villain {name!r}; known: {sorted(_IRC_REGISTRY)}")
    return cls(rng, **(params or {}))
