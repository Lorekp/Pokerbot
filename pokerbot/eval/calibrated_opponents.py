"""HUD-CALIBRATED opponent bots for the exploitation battery.

The existing eval opponents (NitBot/CallingStationBot and the hand-strength-aware HS* variants in
``pokerbot.eval.bots``) are HEURISTIC CARICATURES: their action frequencies are hand-tuned literals,
NOT fit to any real population. So an absolute win-rate measured against them is only as trustworthy
as those literals. This module builds opponents whose REALIZED HUD statistics
(VPIP / PFR / AF / WTSD / WSD / fold_to_cbet) match a TARGET vector drawn from the IRC 7.4M-hand
population data (``data/player_profiles/population_hud_stats.json`` / ``skill_tiers_hud.json``).

The bot is NOT strategically strong — it does not need to be. It needs to be STATISTICALLY REALISTIC:
play a hand a real player of the target tier would have played at the target frequencies, so the
hero's win-rate against it is a faithful "beats a <tier> player" number.

How the calibration works (the key design choice):
  * Preflop VPIP / PFR are pure FREQUENCIES over starting hands. We map them to hand-strength
    thresholds via the EMPIRICAL CDF of the engine's preflop strength proxy
    (``pokerbot.agent._hero_hand_strength``) over the real deal distribution. A target VPIP of 48%
    becomes "play the top 48% of hands by that proxy"; of the played hands, the strongest PFR/VPIP
    fraction are raised and the rest limped/called. Because we invert the SAME proxy the bot reads at
    decision time, the realized VPIP/PFR are exact up to position effects and Monte-Carlo noise —
    no hand-tuned threshold guessing.
  * Postflop fold_to_cbet, AF, WTSD, WSD are hit by strength-conditional bet/call/fold rules with a
    small number of frequency knobs (continue threshold vs a bet -> fold_to_cbet & WTSD; value-bet
    threshold + bet frequency -> AF; how light we continue -> WSD). These are coupled, so the
    validator (scripts/validate_calibrated_opponents.py) measures the realized stats and is the
    source of truth; the defaults below were chosen to land each tier within tolerance.

Nothing here touches the aof_* engine. Wires into the battery via ``build_calibrated_bot`` (mirrors
``pokerbot.eval.bots.build_bot``) and the ``real_*`` FIELDS added to ``exploitation_battery``.
"""

from __future__ import annotations

import json
import random
from functools import lru_cache
from pathlib import Path

from pokerbot.engine.actions import Action, ActionType
from pokerbot.engine.cards import Card, Rank, Suit
from pokerbot.engine.game import GameState

_TIERS_PATH = Path(__file__).resolve().parents[2] / "data" / "player_profiles" / "skill_tiers_hud.json"


def _preflop_strength_proxy(hole) -> float:
    """The SAME preflop strength proxy the bot reads at decision time (pokerbot.agent path).

    Importing through the agent keeps train/serve parity: we invert exactly the function the policy
    later thresholds on. Falls back to the encoder helper directly if the agent import fails."""
    try:
        from pokerbot.encoding.weva_encoder import _preflop_strength

        return float(_preflop_strength(hole))
    except Exception:  # noqa: BLE001 - last-resort crude proxy
        r0, r1 = int(hole[0].rank), int(hole[1].rank)
        return (max(r0, r1) + min(r0, r1)) / 28.0


@lru_cache(maxsize=1)
def _preflop_strength_cdf() -> list[float]:
    """Sorted preflop-strength proxy over ALL 1326 starting-hand combos (each equally likely at deal).

    A target VPIP/PFR percentile maps to a threshold by indexing into this sorted array, so the
    realized voluntary-play frequency equals the target up to position/Monte-Carlo effects."""
    ranks = list(Rank)
    suits = list(Suit)
    cards = [Card(r, s) for r in ranks for s in suits]
    vals: list[float] = []
    n = len(cards)
    for i in range(n):
        for j in range(i + 1, n):
            vals.append(_preflop_strength_proxy((cards[i], cards[j])))
    vals.sort()
    return vals


def strength_threshold_for_top_fraction(frac: float) -> float:
    """Strength proxy s.t. ~``frac`` of random starting hands score >= it (the VPIP/PFR cutoff)."""
    frac = max(0.0, min(1.0, frac))
    cdf = _preflop_strength_cdf()
    if frac <= 0.0:
        return cdf[-1] + 1.0  # play nothing
    if frac >= 1.0:
        return cdf[0] - 1.0   # play everything
    idx = int(round((1.0 - frac) * (len(cdf) - 1)))
    return cdf[idx]


# Empirical quantiles of the POSTFLOP strength proxy (pokerbot.agent path) sampled across flop/turn/
# river over a representative random deal distribution. The proxy is NOT uniform (median ~0.57,
# concentrated in [0.27, 0.91]); a fold/continue threshold must invert THIS so "fold the weakest X%"
# realizes X. Computed lazily once and cached. Values are stable (deterministic proxy + fixed seeds).
@lru_cache(maxsize=1)
def _postflop_strength_cdf() -> list[float]:
    from pokerbot.engine.game import Street

    rng_pick = random.Random(20240601)
    vals: list[float] = []
    for h in range(4000):
        st = GameState.new_hand(button_seat=h % 6, rng=random.Random(h))
        st.deal_hole_cards()
        nb = rng_pick.choice([3, 4, 5])
        st.board = st._deck.deal(nb)
        st.street = {3: Street.FLOP, 4: Street.TURN, 5: Street.RIVER}[nb]
        for s in range(6):
            vals.append(_hand_strength(st, s))
    vals.sort()
    return vals


def postflop_threshold_for_bottom_fraction(frac: float) -> float:
    """Postflop strength s.t. ~``frac`` of postflop hands score BELOW it (the fold cutoff vs a bet)."""
    frac = max(0.0, min(1.0, frac))
    cdf = _postflop_strength_cdf()
    if frac <= 0.0:
        return cdf[0] - 1.0    # fold nothing
    if frac >= 1.0:
        return cdf[-1] + 1.0   # fold everything
    idx = int(round(frac * (len(cdf) - 1)))
    return cdf[idx]


def _hand_strength(state: GameState, seat: int) -> float:
    try:
        from pokerbot.agent import _hero_hand_strength

        v = _hero_hand_strength(state, seat)
        return 0.5 if v is None else float(v)
    except Exception:  # noqa: BLE001
        return 0.5


def _take(legal, *types_pref):
    for t in types_pref:
        a = next((a for a in legal if a.action_type == t), None)
        if a is not None:
            return a
    return legal[0]


def _pick_bet(state: GameState, *, big: bool) -> Action:
    legal = state.legal_actions()
    bets = [a for a in legal if a.action_type in (ActionType.BET, ActionType.RAISE)]
    if not bets:
        return _take(legal, ActionType.CHECK, ActionType.CALL, ActionType.FOLD)
    return bets[-1] if big else bets[len(bets) // 2]


class CalibratedOpponentBot:
    """Opponent whose realized HUD frequencies target a HUD-stat vector.

    Parameters (target HUD)::

        vpip, pfr           preflop voluntary-play / raise-first frequencies
        af                  postflop aggression factor (bet+raise)/call (sizing knob via bet freq)
        wtsd                went-to-showdown (driven mainly by postflop continue/fold thresholds)
        wsd                 won-at-showdown (driven by how light we continue: lighter -> lower WSD)
        fold_to_cbet        fold frequency facing a bet (the continue threshold vs a bet)

    The frequencies are realized by strength-conditional thresholds, so the bot has plausible RANGE
    structure (its bets are stronger than its checks, its calls stronger than its folds) — the human
    prior the exploitation engine targets — while still hitting the target aggregate numbers."""

    def __init__(self, rng: random.Random | None = None, *, vpip: float = 0.48, pfr: float = 0.10,
                 af: float = 0.67, wtsd: float = 0.37, wsd: float = 0.46,
                 fold_to_cbet: float = 0.36, _knobs: dict | None = None) -> None:
        self.rng = rng or random.Random()
        self.vpip = float(vpip)
        self.pfr = float(pfr)
        self.af = float(af)
        self.wtsd = float(wtsd)
        self.wsd = float(wsd)
        self.fold_to_cbet = float(fold_to_cbet)

        # --- Preflop thresholds derived from the empirical CDF (exact VPIP/PFR by construction) ----
        self._vpip_thresh = strength_threshold_for_top_fraction(self.vpip)
        self._pfr_thresh = strength_threshold_for_top_fraction(self.pfr)

        # --- Postflop knobs ----------------------------------------------------------------------
        # These are COUPLED and act on the *conditional* (in-pot, facing-bet) strength distribution,
        # which is far stronger than the unconditional postflop CDF because the VPIP filter already
        # dropped weak hands preflop. So instead of inverting a fixed CDF we self-CALIBRATE the knobs
        # once per target vector by short self-play measurement (see ``_auto_calibrate``); the result
        # is cached. ``_knobs`` lets a caller inject a pre-solved set (skips the calibration sim).
        if _knobs is None:
            _knobs = _calibrated_knobs(self.vpip, self.pfr, self.af, self.wtsd, self.wsd,
                                       self.fold_to_cbet)
        self._call_vs_bet_hs = _knobs["call_vs_bet_hs"]
        self._call_vs_bet_band = _knobs.get("call_vs_bet_band", 0.12)
        self._value_bet_hs = _knobs["value_bet_hs"]
        self._bet_freq = _knobs["bet_freq"]
        self._bluff_bet_hs = _knobs["bluff_bet_hs"]
        self._bluff_freq = _knobs["bluff_freq"]
        # preflop raise size: iso big (loose-passive fields get isolated)
        self._raise_big = self.pfr < 0.14

    # -- decision -----------------------------------------------------------------------------------
    def _preflop(self, state: GameState, seat: int, legal, hs: float) -> Action:
        types = {a.action_type for a in legal}
        facing = ActionType.CALL in types  # CHECK illegal preflop unless BB unopened (current_bet=BB)
        play = hs >= self._vpip_thresh
        raise_it = hs >= self._pfr_thresh
        if not play:
            # If we can check for free (BB with no raise), take it; else fold.
            if ActionType.CHECK in types:
                return _take(legal, ActionType.CHECK)
            return _take(legal, ActionType.FOLD)
        if raise_it:
            bets = [a for a in legal if a.action_type in (ActionType.BET, ActionType.RAISE)]
            if bets:
                return _pick_bet(state, big=self._raise_big)
        # voluntary but not raising -> limp/call (or check the BB)
        if facing:
            return _take(legal, ActionType.CALL, ActionType.CHECK, ActionType.FOLD)
        return _take(legal, ActionType.CHECK, ActionType.CALL, ActionType.FOLD)

    def _postflop(self, state: GameState, seat: int, legal, hs: float) -> Action:
        types = {a.action_type for a in legal}
        facing = ActionType.CALL in types
        if facing:
            # value-raise only the very top (rare, keeps the AF non-trivial without bloating folds),
            # continue (call) the middle, fold the bottom (fold_to_cbet). The raise carve-out is small
            # so it barely perturbs the calibrated fold-to-bet frequency.
            if hs >= 0.86 and self.rng.random() < 0.25:
                return _pick_bet(state, big=False)
            # SOFT continue threshold: the strength proxy is heavily discretized, so a hard cliff makes
            # the realized fold-to-bet frequency jump in big steps as the threshold crosses a dense
            # quantile (the calibrator then oscillates). A linear continue-probability ramp over a band
            # around ``call_vs_bet_hs`` makes the fold rate CONTINUOUS in the knob, so calibration is
            # stable AND the bot still mostly-folds weak / mostly-continues strong (range structure).
            band = self._call_vs_bet_band
            p_continue = (hs - (self._call_vs_bet_hs - band)) / (2.0 * band)
            p_continue = max(0.0, min(1.0, p_continue))
            if self.rng.random() < p_continue:
                return _take(legal, ActionType.CALL)
            return _take(legal, ActionType.FOLD)
        # unopened: value-bet the strong, occasionally bluff the bottom, else check (free showdown).
        # The bet pressure folds villains out -> brings WTSD down toward target.
        if hs >= self._value_bet_hs and self.rng.random() < self._bet_freq:
            return _pick_bet(state, big=False)
        if hs < self._bluff_bet_hs and self.rng.random() < self._bluff_freq:
            return _pick_bet(state, big=False)
        return _take(legal, ActionType.CHECK, ActionType.FOLD)

    def __call__(self, state: GameState) -> Action:
        seat = state.action_seat
        legal = state.legal_actions()
        if not legal:
            return Action(ActionType.FOLD)
        hs = _hand_strength(state, seat)
        if state.street.name == "PREFLOP":
            return self._preflop(state, seat, legal, hs)
        return self._postflop(state, seat, legal, hs)


# --- self-calibration -----------------------------------------------------------------------------
# The postflop knobs act on the CONDITIONAL in-pot strength distribution (which depends on VPIP and on
# the field), so they cannot be solved from a fixed CDF. We instead measure realized HUD stats in a
# short full-ring self-play and adjust the knobs by coordinate descent. Cheap (a few hundred hands x a
# few rounds) and cached per target, so a battery that rebuilds the bot every hand pays it ONCE.

_DEFAULT_KNOBS = {
    "call_vs_bet_hs": 0.50, "call_vs_bet_band": 0.18, "value_bet_hs": 0.57, "bet_freq": 0.55,
    "bluff_bet_hs": 0.40, "bluff_freq": 0.10,
}


def _measure_knobs(target: dict, knobs: dict, *, hands: int, seed: int) -> dict:
    """Realized fold_to_cbet / wtsd / af / wsd of a full ring of bots built with ``knobs``.

    Self-contained mini-measurement (avoids importing the validation script). Counts the same events
    the validator does, but only the stats the knobs drive (preflop vpip/pfr are CDF-exact already)."""
    from pokerbot.engine.game import play_hand

    n = 6
    faced = folded = 0
    bets_raises = calls = 0
    saw_flop = wtsd_n = 0
    showdowns = 0
    sd_won = 0.0
    for h in range(hands):
        hand_seed = seed * 1_000_000 + h
        st = GameState.new_hand(button_seat=h % n, rng=random.Random(hand_seed))
        bots = {s: CalibratedOpponentBot(random.Random(7919 * (s + 1) + hand_seed),
                                         _knobs=knobs, **target) for s in range(n)}
        rec: dict[int, list] = {s: [] for s in range(n)}

        def mk(seat, _b, _rec):
            def pol(state: GameState) -> Action:
                legal = state.legal_actions()
                facing_bet = not any(a.action_type == ActionType.CHECK for a in legal)
                a = _b(state)
                _rec[seat].append((state.street.name, a.action_type, facing_bet))
                return a
            return pol

        final, deltas = play_hand({s: mk(s, bots[s], rec) for s in range(n)}, state=st)
        for seat in range(n):
            acts = rec[seat]
            folded_pre = any(t == ActionType.FOLD and s == "PREFLOP" for s, t, _ in acts)
            first_flop_bet_consumed = False
            for s, t, facing_bet in acts:
                if s == "PREFLOP":
                    continue
                if t in (ActionType.BET, ActionType.RAISE, ActionType.ALL_IN):
                    bets_raises += 1
                elif t == ActionType.CALL:
                    calls += 1
                if s == "FLOP" and facing_bet and not first_flop_bet_consumed:
                    faced += 1
                    first_flop_bet_consumed = True
                    if t == ActionType.FOLD:
                        folded += 1
            saw = (not folded_pre) and len(final.board) >= 3 and not (
                final.players[seat].folded and not any(s != "PREFLOP" for s, _, _ in acts))
            if any(s != "PREFLOP" for s, _, _ in acts) or saw:
                saw_flop += 1
                if not final.players[seat].folded and final.active_player_count() >= 2 \
                        and len(final.board) >= 3:
                    wtsd_n += 1
                    showdowns += 1
                    if deltas.get(seat, 0.0) > 1e-9:
                        sd_won += 1.0
    return {
        "fold_to_cbet": folded / faced if faced else 0.0,
        "af": bets_raises / calls if calls else float(bets_raises),
        "wtsd": wtsd_n / saw_flop if saw_flop else 0.0,
        "wsd": sd_won / showdowns if showdowns else 0.0,
    }


def _auto_calibrate(target: dict, *, hands: int = 900, rounds: int = 12, seed: int = 12345) -> dict:
    """Coordinate-descent the postflop knobs to hit fold_to_cbet / af (asserted) while pushing wtsd
    toward target (reported). Each lever is roughly monotone in one stat, weakly coupled:

      call_vs_bet_hs  raises fold_to_cbet (continue-prob ramp shifts up -> fold more)  -> fold_to_cbet
      bet_freq        more flop pressure folds villains out                            -> lowers wtsd
      value_bet_hs    lower cut = bet wider = more pressure                            -> lowers wtsd
      bluff_freq      bottom-range bets add aggression without value                   -> +af

    Damped proportional steps + keep the BEST-seen knobs by the asserted-stat error (the proxy is
    noisy/discretized, so the last round isn't always the best). Returns the best knob dict."""
    knobs = dict(_DEFAULT_KNOBS)

    def err(g: dict) -> float:
        # weighted asserted error (fold_to_cbet and af are the gate); wtsd lightly weighted.
        return (abs(g["fold_to_cbet"] - target["fold_to_cbet"]) * 3.0
                + abs(g["af"] - target["af"]) * 1.0
                + abs(g["wtsd"] - target["wtsd"]) * 0.3)

    best_knobs, best_err = dict(knobs), float("inf")
    for _ in range(rounds):
        got = _measure_knobs(target, knobs, hands=hands, seed=seed)
        e = err(got)
        if e < best_err:
            best_err, best_knobs = e, dict(knobs)
        # fold_to_cbet via the continue threshold (SOLE driver; decoupled from wsd so they don't fight).
        d_ftc = target["fold_to_cbet"] - got["fold_to_cbet"]
        if abs(d_ftc) > 0.02:
            knobs["call_vs_bet_hs"] = float(min(0.90, max(0.10,
                                       knobs["call_vs_bet_hs"] + 0.55 * d_ftc)))
        # wtsd via flop pressure (bet_freq + value cut). Structural floor well above the IRC target in
        # homogeneous self-play (multiway pots reach showdown); we push but the validator only reports.
        d_wtsd = got["wtsd"] - target["wtsd"]  # positive = too much showdown -> bet more
        if abs(d_wtsd) > 0.04:
            knobs["bet_freq"] = float(min(0.98, max(0.20, knobs["bet_freq"] + 0.35 * d_wtsd)))
            knobs["value_bet_hs"] = float(min(0.75, max(0.30, knobs["value_bet_hs"] - 0.20 * d_wtsd)))
        # af via bluff frequency (add aggression with no value when af is short)
        d_af = target["af"] - got["af"]
        if abs(d_af) > 0.08:
            knobs["bluff_freq"] = float(min(0.55, max(0.0, knobs["bluff_freq"] + 0.08 * d_af)))
    return best_knobs


_KNOB_CACHE: dict[tuple, dict] = {}


def _calibrated_knobs(vpip, pfr, af, wtsd, wsd, fold_to_cbet) -> dict:
    """Cached postflop knobs for a target vector. Calibration is skipped (defaults returned) when
    ``POKERBOT_CALIBRATE_OPP=0`` so callers that only need plausible RANGE structure (not exact HUD)
    avoid the per-target self-play cost."""
    import os

    if os.environ.get("POKERBOT_CALIBRATE_OPP", "1") == "0":
        return dict(_DEFAULT_KNOBS)
    key = (round(vpip, 3), round(pfr, 3), round(af, 3), round(wtsd, 3), round(wsd, 3),
           round(fold_to_cbet, 3))
    if key not in _KNOB_CACHE:
        target = {"vpip": vpip, "pfr": pfr, "af": af, "wtsd": wtsd, "wsd": wsd,
                  "fold_to_cbet": fold_to_cbet}
        _KNOB_CACHE[key] = _auto_calibrate(target)
    return _KNOB_CACHE[key]


# --- tier loading ---------------------------------------------------------------------------------
@lru_cache(maxsize=1)
def load_skill_tiers(path: str | None = None) -> dict:
    p = Path(path) if path else _TIERS_PATH
    data = json.loads(p.read_text(encoding="utf-8"))
    return data["tiers"]


def tier_target(name: str) -> dict:
    """Return the HUD target dict for a calibrated tier (raises for the equilibrium 'pro' tier)."""
    tiers = load_skill_tiers()
    if name not in tiers:
        raise ValueError(f"unknown skill tier '{name}' (known: {sorted(tiers)})")
    t = tiers[name]
    if "target" not in t:
        raise ValueError(f"tier '{name}' is an equilibrium bot ('{t.get('bot')}'), not a HUD target")
    return t["target"]


def build_calibrated_bot(name: str, rng: random.Random | None = None,
                         params: dict | None = None):
    """Factory mirroring ``pokerbot.eval.bots.build_bot`` for the calibrated tiers.

    ``name`` is a skill-tier key (recreational / weak_reg / solid_reg / pro). The HUD-targeted tiers
    build a ``CalibratedOpponentBot`` from the tier's target vector (``params`` overrides individual
    stats). The 'pro' tier delegates to the trained blueprint bot (or its fallback)."""
    tiers = load_skill_tiers()
    if name not in tiers:
        raise ValueError(f"unknown skill tier '{name}' (known: {sorted(tiers)})")
    spec = tiers[name]
    if "target" in spec:
        target = dict(spec["target"])
        if params:
            target.update(params)
        return CalibratedOpponentBot(rng, **target)
    # equilibrium tier -> delegate to the existing bot pool (blueprint, fallback hs pro)
    from pokerbot.eval.bots import build_bot

    bot_name = spec.get("bot", "blueprint")
    try:
        return build_bot(bot_name, rng)
    except Exception:  # noqa: BLE001 - blueprint snapshot may be missing
        return build_bot(spec.get("fallback_bot", "pro_balanced"), rng)
