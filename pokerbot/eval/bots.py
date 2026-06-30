"""Fixed opponent bot pool for evaluation."""

from __future__ import annotations

import random

from pokerbot.engine.actions import Action, ActionType, BET_SIZE_BUCKETS
from pokerbot.engine.game import GameState


def _pick_bet(state: GameState, aggressive: bool) -> Action:
    legal = state.legal_actions()
    bets = [a for a in legal if a.action_type in (ActionType.BET, ActionType.RAISE)]
    if not bets:
        return next((a for a in legal if a.action_type == ActionType.CHECK), legal[0])
    if aggressive:
        return bets[-1]
    return bets[len(bets) // 2]


class NitBot:
    """Tight-passive: folds most hands, rarely raises."""

    def __init__(self, rng: random.Random | None = None) -> None:
        self.rng = rng or random.Random()

    def __call__(self, state: GameState) -> Action:
        legal = state.legal_actions()
        if state.street.name == "PREFLOP":
            if self.rng.random() < 0.85:
                return Action(ActionType.FOLD) if Action(ActionType.FOLD) in legal else legal[0]
            if self.rng.random() < 0.7:
                return next((a for a in legal if a.action_type == ActionType.CALL), legal[0])
        if self.rng.random() < 0.9:
            # Tight-passive: CHECK when free, FOLD when facing a bet. (The old one-liner scanned the
            # LEGAL list — where FOLD is always first — so it OPEN-FOLDED 90% of nodes even in
            # unopened pots, forfeiting free showdowns no player would: a fold-bot, not a nit, which
            # made stealing worthless — it folded to a limp as readily as to a raise.)
            check = next((a for a in legal if a.action_type == ActionType.CHECK), None)
            if check is not None:
                return check
            return next((a for a in legal if a.action_type == ActionType.FOLD), legal[0])
        return _pick_bet(state, aggressive=False)


class LAGBot:
    """Loose-aggressive: bets and raises frequently."""

    def __init__(self, rng: random.Random | None = None) -> None:
        self.rng = rng or random.Random()

    def __call__(self, state: GameState) -> Action:
        legal = state.legal_actions()
        if self.rng.random() < 0.55:
            bets = [a for a in legal if a.action_type in (ActionType.BET, ActionType.RAISE, ActionType.ALL_IN)]
            if bets:
                return self.rng.choice(bets)
        return next(
            (a for a in legal if a.action_type in (ActionType.CHECK, ActionType.CALL)),
            legal[0],
        )


class CallingStationBot:
    """Loose-passive: calls wide, rarely folds or raises."""

    def __init__(self, rng: random.Random | None = None) -> None:
        self.rng = rng or random.Random()

    def __call__(self, state: GameState) -> Action:
        legal = state.legal_actions()
        if Action(ActionType.CALL) in legal and self.rng.random() < 0.75:
            return Action(ActionType.CALL)
        if Action(ActionType.CHECK) in legal:
            return Action(ActionType.CHECK)
        if Action(ActionType.FOLD) in legal and self.rng.random() < 0.15:
            return Action(ActionType.FOLD)
        return legal[0]


class TagBot:
    """Tight-aggressive: selective but raises when continuing."""

    def __init__(self, rng: random.Random | None = None) -> None:
        self.rng = rng or random.Random()

    def __call__(self, state: GameState) -> Action:
        legal = state.legal_actions()
        if state.street.name == "PREFLOP":
            if self.rng.random() < 0.65:
                return Action(ActionType.FOLD) if Action(ActionType.FOLD) in legal else legal[0]
        if self.rng.random() < 0.45:
            bets = [a for a in legal if a.action_type in (ActionType.BET, ActionType.RAISE)]
            if bets:
                return bets[-1]
        return next(
            (a for a in legal if a.action_type in (ActionType.CHECK, ActionType.CALL)),
            legal[0],
        )


class DonkBot:
    """Leads into preflop aggressor postflop (exploitable donk-betting)."""

    def __init__(self, rng: random.Random | None = None) -> None:
        self.rng = rng or random.Random()

    def __call__(self, state: GameState) -> Action:
        legal = state.legal_actions()
        if state.street.name in ("FLOP", "TURN") and state.current_bet_bb <= 1e-9:
            if self.rng.random() < 0.4:
                frac = self.rng.choice(BET_SIZE_BUCKETS[:3])
                target = state.pot_bb * frac
                for a in legal:
                    if a.action_type == ActionType.BET and a.amount_bb >= target * 0.8:
                        return a
        if self.rng.random() < 0.3:
            return next((a for a in legal if a.action_type == ActionType.FOLD), legal[0])
        return next(
            (a for a in legal if a.action_type in (ActionType.CHECK, ActionType.CALL)),
            legal[0],
        )


class RandomBot:
    def __init__(self, rng: random.Random | None = None) -> None:
        self.rng = rng or random.Random()

    def __call__(self, state: GameState) -> Action:
        legal = state.legal_actions()
        return self.rng.choice(legal)


class GTOCopyBot:
    """Wraps a policy network as a bot."""

    def __init__(
        self,
        model,
        device="cpu",
        seat: int = 0,
        *,
        eval_spot_class: str | None = None,
        require_scope: bool = True,
    ) -> None:
        self.model = model
        self.device = device
        self.seat = seat
        self.eval_spot_class = eval_spot_class
        self.require_scope = require_scope

    def __call__(self, state: GameState) -> Action:
        import torch

        from pokerbot.encoding.eval_scope import encode_state_for_policy
        from pokerbot.training.self_play import _index_to_action, _legal_mask

        seat = state.action_seat
        enc = encode_state_for_policy(
            state,
            seat,
            spot_class=self.eval_spot_class,
            require_scope=self.require_scope,
        )
        weva_ids = torch.tensor([enc["weva_id"]], dtype=torch.long, device=self.device)
        cards = torch.from_numpy(enc["cards"]).unsqueeze(0).to(self.device)
        actions = torch.from_numpy(enc["actions"]).unsqueeze(0).to(self.device)
        position = torch.from_numpy(enc["policy_position"]).unsqueeze(0).to(self.device)
        mask = torch.from_numpy(_legal_mask(state, seat)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            idx, _ = self.model.act(
                weva_ids, actions, position, mask, cards=cards, deterministic=True
            )
        return _index_to_action(state, seat, int(idx.item()))


class HumanLikeBot:
    """Nearest-neighbor over SoelMgd/Poker_Dataset Q&A (exploitative human style)."""

    _shared_questions: list[str] | None = None
    _shared_actions: list[str] | None = None

    def __init__(self, rng: random.Random | None = None, *, limit: int = 5000) -> None:
        self.rng = rng or random.Random()
        self._questions: list[str] = []
        self._actions: list[str] = []
        self._loaded = False
        self._limit = limit

    def _load(self) -> None:
        if self._loaded:
            return
        if HumanLikeBot._shared_questions is not None:
            self._questions = HumanLikeBot._shared_questions
            self._actions = HumanLikeBot._shared_actions or []
            self._loaded = True
            return
        from pokerbot.data.exploitative import load_exploitative_dataset

        samples = load_exploitative_dataset("train", self._limit)
        for s in samples:
            if s.parsed_action:
                self._questions.append(s.question.lower())
                self._actions.append(s.parsed_action)
        HumanLikeBot._shared_questions = self._questions
        HumanLikeBot._shared_actions = self._actions
        self._loaded = True

    def __call__(self, state: GameState) -> Action:
        from pokerbot.engine.actions import parse_pokerbench_output

        self._load()
        legal = state.legal_actions()
        if not self._questions:
            return legal[0]

        query = f"{state.street.name} pot {state.pot_bb:.1f}".lower()
        best_i = 0
        best_score = -1
        for i, q in enumerate(self._questions[:500]):
            tokens = set(query.split())
            score = len(tokens & set(q.split()))
            if score > best_score:
                best_score = score
                best_i = i
        try:
            action = parse_pokerbench_output(self._actions[best_i])
            if action in legal:
                return action
        except ValueError:
            pass
        return self.rng.choice(legal)


class GearChangerBot:
    """Plays one archetype, then SWITCHES to a contrasting one at hand ``switch_at`` — the
    non-stationarity / re-adaptation test. The battery calls ``on_new_hand()`` once per hand so the
    bot knows when to flip; standalone it never switches (stays on ``before``)."""

    def __init__(self, rng: random.Random | None = None, *, before: str = "calling_station",
                 after: str = "lag", switch_at: int = 100) -> None:
        self.rng = rng or random.Random()
        self._before = build_bot(before, self.rng)
        self._after = build_bot(after, self.rng)
        self.switch_at = switch_at
        self.hand_count = 0
        self.before_name, self.after_name = before, after

    def on_new_hand(self) -> None:
        self.hand_count += 1

    @property
    def switched(self) -> bool:
        return self.hand_count >= self.switch_at

    def __call__(self, state: GameState) -> Action:
        return (self._after if self.switched else self._before)(state)


class AdaptiveBRBot:
    """Tracks the HERO's recent aggression frequency and counter-exploits it (passive hero ⇒ bet/bluff
    a lot; aggressive hero ⇒ tighten to value/check-call). An adaptive best-responder — the upper
    bound on the exploitability our opponent-modeling exposes. The battery feeds it
    ``observe_hero(action)``; with no observations it plays a balanced default."""

    def __init__(self, rng: random.Random | None = None, *, window: int = 40) -> None:
        from collections import deque

        self.rng = rng or random.Random()
        self._hero_aggr: deque[int] = deque(maxlen=window)

    def observe_hero(self, action: Action) -> None:
        self._hero_aggr.append(
            1 if action.action_type in (ActionType.BET, ActionType.RAISE, ActionType.ALL_IN) else 0
        )

    def __call__(self, state: GameState) -> Action:
        legal = state.legal_actions()
        hero_aggr = (sum(self._hero_aggr) / len(self._hero_aggr)) if self._hero_aggr else 0.5
        bet_p = 0.70 if hero_aggr < 0.30 else (0.20 if hero_aggr > 0.60 else 0.45)
        if self.rng.random() < bet_p:
            bets = [a for a in legal if a.action_type in (ActionType.BET, ActionType.RAISE, ActionType.ALL_IN)]
            if bets:
                return self.rng.choice(bets)
        # otherwise: never fold a free check; vs a bet, call when the hero is passive, else fold often
        check = next((a for a in legal if a.action_type == ActionType.CHECK), None)
        if check is not None:
            return check
        call = next((a for a in legal if a.action_type == ActionType.CALL), None)
        if call is not None and (hero_aggr < 0.45 or self.rng.random() < 0.5):
            return call
        return next((a for a in legal if a.action_type == ActionType.FOLD), legal[0])


def _hand_strength(state: GameState, seat: int) -> float:
    """Shared hand-strength proxy (same one the hero uses); 0.5 when unavailable."""
    try:
        from pokerbot.agent import _hero_hand_strength

        v = _hero_hand_strength(state, seat)
        return 0.5 if v is None else float(v)
    except Exception:  # noqa: BLE001
        return 0.5


class _HSBotBase:
    """HAND-STRENGTH-CONDITIONAL archetypes — the human-realistic eval opponents.

    The original caricature bots act with UNCONDITIONAL random ranges (NitBot's rare bets are
    any-two, TagBot max-bets 45% of nodes regardless of holding), which violates the human range
    structure the exploitation engine's priors encode (IRC population data: rare bets = value,
    continuing correlates with equity). Testing a human-prior engine against range-free bots
    systematically understates (or sign-flips) its edge. These variants keep each archetype's
    FREQUENCY signature but condition every action on hand strength, so a read like "folds to
    bets 80%" coexists with "their bets are value-heavy" — like the players the engine targets."""

    def __init__(self, rng: random.Random | None = None, **params: float) -> None:
        self.rng = rng or random.Random()
        # Parameterized thresholds (G3 fitted gauntlet): defaults are the original hand-crafted
        # literals, byte-identical behavior; the IRC percentile fit overrides per weak_mixture entry.
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


class HSStationBot(_HSBotBase):
    """Loose-passive, range-aware: calls with any piece of equity, releases pure air to bets,
    value-bets only strong hands. The exploit: relentless thin value; bluffs still mostly fail."""

    call_hs = 0.22
    call_jitter = 0.10
    bet_hs = 0.78
    bet_freq = 0.5

    def __call__(self, state: GameState) -> Action:
        seat = state.action_seat
        hs = _hand_strength(state, seat)
        legal = state.legal_actions()
        types = {a.action_type for a in legal}
        if ActionType.CALL in types:                       # facing chips to call
            if hs > self.call_hs + self.call_jitter * self.rng.random():  # sticky; air folds eventually
                return self._take(legal, ActionType.CALL)
            return self._take(legal, ActionType.FOLD)
        if hs > self.bet_hs and self.rng.random() < self.bet_freq:
            return _pick_bet(state, aggressive=False)
        return self._take(legal, ActionType.CHECK, ActionType.FOLD)


class HSNitBot(_HSBotBase):
    """Tight-passive, range-aware: plays only strong hands, RARE BETS ARE VALUE (the human prior).
    The exploit: steal relentlessly, release when the nit wakes up."""

    pre_fold_hs = 0.52
    raise_hs = 0.80
    raise_freq = 0.4
    call_hs = 0.66
    bet_hs = 0.74
    bet_freq = 0.6

    def __call__(self, state: GameState) -> Action:
        seat = state.action_seat
        hs = _hand_strength(state, seat)
        legal = state.legal_actions()
        types = {a.action_type for a in legal}
        if state.street.name == "PREFLOP" and hs < self.pre_fold_hs and ActionType.CALL in types:
            return self._take(legal, ActionType.FOLD)
        if ActionType.CALL in types:                       # facing
            if hs > self.raise_hs:
                return _pick_bet(state, aggressive=False) if self.rng.random() < self.raise_freq \
                    else self._take(legal, ActionType.CALL)
            if hs > self.call_hs:
                return self._take(legal, ActionType.CALL)
            return self._take(legal, ActionType.FOLD)
        if hs > self.bet_hs and self.rng.random() < self.bet_freq:  # unopened: rare bets = value
            return _pick_bet(state, aggressive=False)
        return self._take(legal, ActionType.CHECK, ActionType.FOLD)


class HSLagBot(_HSBotBase):
    """Loose-aggressive, range-aware: bets wide-but-not-random (equity OR a real bluff rate),
    calls down light. The exploit: bluff-catch wider, value-bet bigger."""

    raise_hs = 0.72
    raise_freq = 0.5
    call_hs = 0.34
    bluff_raise_freq = 0.20
    bet_hs = 0.42
    bluff_bet_freq = 0.25

    def __call__(self, state: GameState) -> Action:
        seat = state.action_seat
        hs = _hand_strength(state, seat)
        legal = state.legal_actions()
        types = {a.action_type for a in legal}
        if ActionType.CALL in types:                       # facing
            if hs > self.raise_hs and self.rng.random() < self.raise_freq:
                return _pick_bet(state, aggressive=True)
            if hs > self.call_hs:
                return self._take(legal, ActionType.CALL)
            if self.rng.random() < self.bluff_raise_freq:  # occasional bluff-raise
                return _pick_bet(state, aggressive=True)
            return self._take(legal, ActionType.FOLD)
        if hs > self.bet_hs or self.rng.random() < self.bluff_bet_freq:  # wide value + real bluffs
            return _pick_bet(state, aggressive=True)
        return self._take(legal, ActionType.CHECK, ActionType.FOLD)


class HSTagBot(_HSBotBase):
    """Tight-aggressive, range-aware: the near-solid control. Value-bets strong, folds weak,
    defends reasonably — the engine should find LITTLE here (do-no-harm reference)."""

    pre_fold_hs = 0.45
    raise_hs = 0.74
    raise_freq = 0.5
    call_hs = 0.52
    bluff_defend_freq = 0.12
    bet_hs = 0.60
    bet_freq = 0.65
    stab_hs = 0.35
    stab_freq = 0.15

    def __call__(self, state: GameState) -> Action:
        seat = state.action_seat
        hs = _hand_strength(state, seat)
        legal = state.legal_actions()
        types = {a.action_type for a in legal}
        if state.street.name == "PREFLOP" and hs < self.pre_fold_hs and ActionType.CALL in types:
            return self._take(legal, ActionType.FOLD)
        if ActionType.CALL in types:
            if hs > self.raise_hs and self.rng.random() < self.raise_freq:
                return _pick_bet(state, aggressive=False)
            if hs > self.call_hs:
                return self._take(legal, ActionType.CALL)
            if self.rng.random() < self.bluff_defend_freq:  # small bluff-defend frequency
                return self._take(legal, ActionType.CALL)
            return self._take(legal, ActionType.FOLD)
        if hs > self.bet_hs and self.rng.random() < self.bet_freq:
            return _pick_bet(state, aggressive=False)
        if hs < self.stab_hs and self.rng.random() < self.stab_freq:  # occasional stab
            return _pick_bet(state, aggressive=False)
        return self._take(legal, ActionType.CHECK, ActionType.FOLD)


class CounterExploiterBot(_HSBotBase):
    """ADVERSARIAL STRONG PLAYER (G5): watches the hero's recent action mix and punishes the
    dominant tilt — exactly the deviations the exploit mixer introduces. Hand-strength-conditional
    (fair), so its punishment has real range structure:

      hero over-aggressive (bluff-heavy mixes) -> tighten + call-down/trap: only continue with
        equity, let the hero barrel into it, never pay raises without a hand;
      hero over-folding   -> relentless pressure: bet/raise wide, steal everything;
      hero over-calling   -> pure thin value, zero bluffs (the station punish inverted onto us);
      balanced            -> near-solid tag baseline (no exploitable tilt of its own).

    The battery feeds ``observe_hero(action)`` (the AdaptiveBRBot hook). With no observations it
    plays the tag baseline. Window ~60 hero actions, re-evaluated continuously."""

    AGGRO_HI = 0.42   # hero aggressive-action rate above this = over-aggressive
    FOLD_HI = 0.38    # hero fold rate above this = over-folding
    CALL_HI = 0.45    # hero passive-continue rate above this = over-calling

    def __init__(self, rng: random.Random | None = None, *, window: int = 60) -> None:
        from collections import deque

        super().__init__(rng)
        self._hero: "deque[str]" = deque(maxlen=window)

    def observe_hero(self, action: Action) -> None:
        t = action.action_type
        if t in (ActionType.BET, ActionType.RAISE, ActionType.ALL_IN):
            self._hero.append("a")
        elif t == ActionType.FOLD:
            self._hero.append("f")
        elif t == ActionType.CALL:
            self._hero.append("c")
        else:
            self._hero.append("k")

    def _tilt(self) -> str:
        if len(self._hero) < 15:
            return "balanced"
        n = len(self._hero)
        aggro = sum(1 for x in self._hero if x == "a") / n
        fold = sum(1 for x in self._hero if x == "f") / n
        call = sum(1 for x in self._hero if x == "c") / n
        # dominant tilt by largest exceedance
        exceed = {"aggro": aggro - self.AGGRO_HI, "fold": fold - self.FOLD_HI,
                  "call": call - self.CALL_HI}
        kind, x = max(exceed.items(), key=lambda kv: kv[1])
        return kind if x > 0 else "balanced"

    def __call__(self, state: GameState) -> Action:
        seat = state.action_seat
        hs = _hand_strength(state, seat)
        legal = state.legal_actions()
        types = {a.action_type for a in legal}
        tilt = self._tilt()
        facing = ActionType.CALL in types
        if tilt == "aggro":
            # Trap/call-down: continue on equity, never bloat, make their bluffs pay us.
            if facing:
                if hs > 0.42:
                    return self._take(legal, ActionType.CALL)
                return self._take(legal, ActionType.FOLD)
            if hs > 0.80 and self.rng.random() < 0.3:   # mostly check strong hands too (trap)
                return _pick_bet(state, aggressive=False)
            return self._take(legal, ActionType.CHECK, ActionType.FOLD)
        if tilt == "fold":
            # Relentless pressure: every fold the hero makes is pure profit.
            if facing:
                if hs > 0.55 or self.rng.random() < 0.35:
                    return _pick_bet(state, aggressive=True) if self.rng.random() < 0.5 \
                        else self._take(legal, ActionType.CALL)
                return self._take(legal, ActionType.FOLD)
            if hs > 0.25 or self.rng.random() < 0.45:
                return _pick_bet(state, aggressive=False)
            return self._take(legal, ActionType.CHECK, ActionType.FOLD)
        if tilt == "call":
            # Thin value, zero bluffs: the hero pays off, so never bluff and bet thinner.
            if facing:
                if hs > 0.60:
                    return self._take(legal, ActionType.CALL)
                return self._take(legal, ActionType.FOLD)
            if hs > 0.55:
                return _pick_bet(state, aggressive=False)
            return self._take(legal, ActionType.CHECK, ActionType.FOLD)
        # balanced baseline = the near-solid tag line
        if state.street.name == "PREFLOP" and hs < 0.45 and facing:
            return self._take(legal, ActionType.FOLD)
        if facing:
            if hs > 0.74 and self.rng.random() < 0.5:
                return _pick_bet(state, aggressive=False)
            if hs > 0.52:
                return self._take(legal, ActionType.CALL)
            return self._take(legal, ActionType.FOLD)
        if hs > 0.60 and self.rng.random() < 0.65:
            return _pick_bet(state, aggressive=False)
        return self._take(legal, ActionType.CHECK, ActionType.FOLD)


def _street_giveup(state: GameState) -> float:
    """Bluff-frequency multiplier emulating PRO give-up discipline: bluff more on early streets,
    barrel-give-up later. The caricature/HS bots bluff at a street-flat rate (over-bluffs rivers,
    which a competent reg never does); a pro tapers."""
    return {"PREFLOP": 1.0, "FLOP": 1.0, "TURN": 0.7, "RIVER": 0.45}.get(state.street.name, 1.0)


class HSProBalancedBot(_HSBotBase):
    """PROFESSIONAL balanced reg (G5 strong reference): polarized, range-aware, and HARD to exploit.

    Unlike the HS archetypes (which leak a one-sided tilt the engine feasts on), this bot is balanced
    in the GTO sense:
      * value-bets a strong range AND a CALIBRATED bluff fraction drawn ONLY from the very bottom
        (air with no showdown) -- never the medium hands, which it check-calls as bluff-catchers;
      * defends vs bets at an MDF-like threshold (folds the bottom, continues enough to deny a pure
        steal), so over-folding it does not work;
      * tapers its bluffs by street (give-up discipline) so barreling thin air does not auto-profit.

    The exploit engine should extract ~0 here without leaking (do-no-harm), and the RNR cap must keep
    it from over-adjusting -- this is the balanced counterpart to CounterExploiterBot's adversary."""

    pre_fold_hs = 0.46
    value_bet_hs = 0.62
    value_bet_freq = 0.72
    bluff_top_hs = 0.22          # only the BOTTOM (hs below this) is ever turned into a bluff
    bluff_bet_freq = 0.28
    call_defend_hs = 0.40        # MDF-ish bluff-catch floor vs a bet
    raise_value_hs = 0.80
    raise_value_freq = 0.45
    checkraise_bluff_freq = 0.08  # rare, polarized check-raise bluff from the bottom

    def __call__(self, state: GameState) -> Action:
        seat = state.action_seat
        hs = _hand_strength(state, seat)
        legal = state.legal_actions()
        types = {a.action_type for a in legal}
        gv = _street_giveup(state)
        if state.street.name == "PREFLOP" and hs < self.pre_fold_hs and ActionType.CALL in types:
            return self._take(legal, ActionType.FOLD)
        if ActionType.CALL in types:                       # facing a bet/raise
            if hs > self.raise_value_hs and self.rng.random() < self.raise_value_freq:
                return _pick_bet(state, aggressive=False)   # value raise
            if hs > self.call_defend_hs:
                return self._take(legal, ActionType.CALL)   # bluff-catch / continue (denies the steal)
            if hs < self.bluff_top_hs and self.rng.random() < self.checkraise_bluff_freq * gv:
                return _pick_bet(state, aggressive=True)     # balanced polarized bluff-raise
            return self._take(legal, ActionType.FOLD)
        # unopened: polarized value + a calibrated bottom-range bluff, check the medium
        if hs > self.value_bet_hs and self.rng.random() < self.value_bet_freq:
            return _pick_bet(state, aggressive=False)
        if hs < self.bluff_top_hs and self.rng.random() < self.bluff_bet_freq * gv:
            return _pick_bet(state, aggressive=False)
        return self._take(legal, ActionType.CHECK, ActionType.FOLD)


class HSProAggroBot(_HSBotBase):
    """PROFESSIONAL aggressive reg (G5 strong reference): disciplined LAG with a BALANCED 3bet/barrel
    range. It punishes passivity and over-folding -- but with a REAL polarized range, not random air:
      * 3bets/raises for value AND a controlled fraction of pure bottom-range air (light 3bets),
        while FOLDING the capped middle (range discipline a maniac lacks);
      * barrels with balance and tapers bluffs by street;
      * still defends correctly vs aggression.

    Versus hs_lag (loose, undisciplined) this is the version a winning reg actually plays. It is the
    honest test of the RNR counter-exploitation cap: reacting to its pressure by over-folding loses,
    and over-bluff-catching also loses, because its aggression is balanced."""

    pre_fold_hs = 0.40
    raise_value_hs = 0.66
    raise_value_freq = 0.60
    bluff_top_hs = 0.26          # light-3bet/bluff only from the bottom
    bluff_raise_freq = 0.22
    call_defend_hs = 0.38
    bet_value_hs = 0.50
    bet_value_freq = 0.78
    barrel_bluff_freq = 0.34

    def __call__(self, state: GameState) -> Action:
        seat = state.action_seat
        hs = _hand_strength(state, seat)
        legal = state.legal_actions()
        types = {a.action_type for a in legal}
        gv = _street_giveup(state)
        if state.street.name == "PREFLOP" and hs < self.pre_fold_hs and ActionType.CALL in types:
            # capped-middle discipline: fold the weak-but-not-bluff hands rather than spew
            if not (hs < self.bluff_top_hs and self.rng.random() < self.bluff_raise_freq):
                return self._take(legal, ActionType.FOLD)
        if ActionType.CALL in types:                       # facing
            if hs > self.raise_value_hs and self.rng.random() < self.raise_value_freq:
                return _pick_bet(state, aggressive=True)    # value 3bet/raise
            if hs < self.bluff_top_hs and self.rng.random() < self.bluff_raise_freq * gv:
                return _pick_bet(state, aggressive=True)    # balanced light 3bet/bluff-raise
            if hs > self.call_defend_hs:
                return self._take(legal, ActionType.CALL)
            return self._take(legal, ActionType.FOLD)
        if hs > self.bet_value_hs and self.rng.random() < self.bet_value_freq:
            return _pick_bet(state, aggressive=True)        # wide-but-real value barrel
        if hs < self.bluff_top_hs and self.rng.random() < self.barrel_bluff_freq * gv:
            return _pick_bet(state, aggressive=True)        # tapered bluff barrel
        return self._take(legal, ActionType.CHECK, ActionType.FOLD)


_BLUEPRINT_SNAPSHOT_CACHE: dict = {}


class ThreeBetHappyBot:
    """Caricature 3-bet-happy villain — re-raises ~60% of opens preflop.

    Closes the explicit battery coverage gap from the OPPONENT-EVAL audit: with NO villain in the
    league exercising the 3-bet frequency that real regs use, the value-overlay + RNR cap path was
    unable to certify its ceiling compliance. This bot is intentionally unballanced (worse than any
    real reg) so a properly-capped exploit should still extract value without over-shooting.
    """

    def __init__(self, rng: random.Random | None = None) -> None:
        self.rng = rng or random.Random()

    def __call__(self, state: GameState) -> Action:
        legal = state.legal_actions()
        if state.street.name == "PREFLOP":
            # If facing a single open (no further raise this street), 3-bet 60% of the time.
            n_raises = sum(1 for a in state.action_history()
                           if a[1].action_type in (ActionType.RAISE, ActionType.ALL_IN))
            if n_raises == 1 and self.rng.random() < 0.60:
                three_bet = next((a for a in legal if a.action_type in (ActionType.RAISE, ActionType.ALL_IN)), None)
                if three_bet is not None:
                    return three_bet
        # Postflop: standard tag-like play — half the time bet, otherwise check/call.
        if self.rng.random() < 0.45:
            bet = next((a for a in legal if a.action_type in (ActionType.BET, ActionType.RAISE, ActionType.ALL_IN)), None)
            if bet is not None:
                return bet
        return next((a for a in legal if a.action_type in (ActionType.CHECK, ActionType.CALL)), legal[0])


class FoldTo3BetBot:
    """Caricature 3-bet folder — folds ~75% of the time when facing a 3-bet preflop.

    Closes the explicit battery coverage gap from the OPPONENT-EVAL audit. A real nit reg folds
    near 100% of 3-bets to a 4-bet, but a profile that mis-reads this villain as a LAG (because
    their open frequency is high) would over-bluff at 4-bet sizing. This bot isolates the failure
    mode so the RNR / value-overlay / search path can be certified against it.
    """

    def __init__(self, rng: random.Random | None = None) -> None:
        self.rng = rng or random.Random()

    def __call__(self, state: GameState) -> Action:
        legal = state.legal_actions()
        if state.street.name == "PREFLOP":
            n_raises = sum(1 for a in state.action_history()
                           if a[1].action_type in (ActionType.RAISE, ActionType.ALL_IN))
            if n_raises >= 2 and self.rng.random() < 0.75:
                # Facing a 3-bet (or higher) — fold 75% of the time.
                fold = next((a for a in legal if a.action_type == ActionType.FOLD), None)
                if fold is not None:
                    return fold
            # Otherwise, open wide (loose-passive opener).
            if n_raises == 0 and self.rng.random() < 0.55:
                open_bet = next((a for a in legal if a.action_type in (ActionType.RAISE, ActionType.ALL_IN)), None)
                if open_bet is not None:
                    return open_bet
        # Postflop: call/check almost always — calling-station tendency.
        if self.rng.random() < 0.20:
            bet = next((a for a in legal if a.action_type in (ActionType.BET, ActionType.RAISE, ActionType.ALL_IN)), None)
            if bet is not None:
                return bet
        return next((a for a in legal if a.action_type in (ActionType.CHECK, ActionType.CALL)), legal[0])


def _load_blueprint_snapshot():
    """Lazy module-level snapshot for BlueprintBot AND the live serve (LiveSearchAdvisor calls this when
    no explicit blueprint is passed). Loaded once per process; per-hand bot construction stays cheap.

    DEFAULT = the STABLE_CURRENT ship pointer ``checkpoints/serve_blueprint_6max.pkl`` (== iter62000,
    100,042 infosets), matching ``scripts/ship_blueprint.py``'s STABLE_CURRENT. This is the B1 PROVENANCE
    FIX (2026-06-30): the default previously hardcoded the older ``%TEMP%\\bp6max_lossless2_iter200.pkl``
    (96,553 infosets), which is ~2x MORE LBR-exploitable table-only (+92.4 vs +41.6 bb/100), so the
    deployed default served a measurably worse blueprint than the certified ship unit. Override with
    ``POKERBOT_BLUEPRINT_SNAPSHOT``; the legacy temp file is only a fallback if the stable pointer is
    missing. CRITICAL: the table is best-responder-exploitable served WITHOUT preflop search, so the serve
    MUST keep ``search_preflop=ON`` AND ``POKERBOT_PREFLOP_LOSSLESS=1`` (+ the other encoder flags) set
    before importing pokerbot. The loaded blueprint is LOGGED LOUDLY (path / source / infosets / abstraction
    / sha) so the served unit is never ambiguous again."""
    import hashlib
    import os
    import pickle
    import sys
    from pathlib import Path

    _STABLE = Path(__file__).resolve().parents[2] / "checkpoints" / "serve_blueprint_6max.pkl"
    _LEGACY = r"checkpoints/bp6max_lossless2_iter200.pkl"
    env_path = os.environ.get("POKERBOT_BLUEPRINT_SNAPSHOT")
    if env_path:
        path, source = env_path, "POKERBOT_BLUEPRINT_SNAPSHOT (override)"
    elif _STABLE.exists():
        path, source = str(_STABLE), "STABLE_CURRENT ship pointer (default)"
    else:
        path, source = _LEGACY, "LEGACY TEMP fallback (STABLE pointer MISSING!)"

    if path not in _BLUEPRINT_SNAPSHOT_CACHE:
        with open(path, "rb") as f:
            raw = f.read()
        snap = pickle.loads(raw)
        stamp = None
        if isinstance(snap, dict) and "strat_sum" in snap:
            # Cross-check the serve abstraction against the stamp the trainer wrote. A STAMPED snapshot
            # (the lossless era) is checked STRICTLY — serving it under the wrong encoder flags silently
            # misses every affected node and plays a beatable strategy, so we raise rather than warn.
            # An unstamped (legacy weva) snapshot keeps the lenient env-only warn for back-compat.
            from pokerbot.eval.lbr import check_blueprint_serve_env

            stamp = snap.get("abstraction")
            check_blueprint_serve_env(expected=stamp, strict=stamp is not None)
            snap = snap["strat_sum"]
        n_infosets = len(snap) if hasattr(snap, "__len__") else -1
        sha = hashlib.sha256(raw).hexdigest()[:12]
        # LOUD provenance banner — the served blueprint must NEVER be ambiguous again (the B1 lesson:
        # the audited "best" was not the served one). Printed to stderr + the logging subsystem.
        banner = (
            "\n===================== BLUEPRINT SERVE =====================\n"
            f"[BLUEPRINT] path     : {path}\n"
            f"[BLUEPRINT] source   : {source}\n"
            f"[BLUEPRINT] infosets : {n_infosets:,}   sha256[:12]={sha}\n"
            f"[BLUEPRINT] abstract : {stamp}\n"
            "===========================================================\n"
        )
        print(banner, file=sys.stderr, flush=True)
        try:
            import logging

            logging.getLogger("pokerbot.blueprint").warning(
                "SERVING blueprint path=%s source=%s infosets=%s sha256=%s abstraction=%s",
                path, source, n_infosets, sha, stamp)
        except Exception:  # noqa: BLE001 — logging must never break the serve
            pass
        _BLUEPRINT_SNAPSHOT_CACHE[path] = snap
    return _BLUEPRINT_SNAPSHOT_CACHE[path]


class BlueprintBot:
    """NEAR-NASH STRONG PLAYER (G5): SAMPLES the trained tabular blueprint's mixed strategy at
    every decision (not argmax — a mixed near-equilibrium villain). Fingerprint misses serve the
    adapter's legal-uniform fallback; any bridge failure falls back to the tag baseline. The
    do-no-harm gate: the exploit mixer must extract ≈0 here without leaking."""

    def __init__(self, rng: random.Random | None = None) -> None:
        from pokerbot.search.tabular_blueprint_adapter import TabularBlueprintAdapter

        self.rng = rng or random.Random()
        self.adapter = TabularBlueprintAdapter(_load_blueprint_snapshot(), client_bet_sizing=True)
        self._fallback = HSTagBot(self.rng)

    def __call__(self, state: GameState) -> Action:
        try:
            import numpy as _np

            from pokerbot.search.state_bridge import game_state_to_env
            from pokerbot.training.self_play import _index_to_action as _i2a

            seat = state.action_seat
            env = game_state_to_env(state, seat)
            p = self.adapter.blueprint(env, seat)
            s = float(p.sum())
            if s > 0 and _np.isfinite(p).all():
                idx = self.rng.choices(range(len(p)), weights=(p / s).tolist())[0]
                return _i2a(state, seat, int(idx))
        except Exception:  # noqa: BLE001
            pass
        return self._fallback(state)


def build_bot(name: str, rng: random.Random | None = None, model=None, device: str = "cpu",
              params: dict | None = None):
    """Factory for league / eval bots. ``params`` overrides HS-bot thresholds (the IRC-fitted
    weak_mixture gauntlet); None ⇒ the hand-crafted defaults, byte-identical."""
    if name == "hs_station":
        return HSStationBot(rng, **(params or {}))
    if name == "hs_nit":
        return HSNitBot(rng, **(params or {}))
    if name == "hs_lag":
        return HSLagBot(rng, **(params or {}))
    if name == "hs_tag":
        return HSTagBot(rng, **(params or {}))
    if name == "nit":
        return NitBot(rng)
    if name == "lag":
        return LAGBot(rng)
    if name == "calling_station":
        return CallingStationBot(rng)
    if name == "tag":
        return TagBot(rng)
    if name == "donk":
        return DonkBot(rng)
    if name == "three_bet_happy":
        return ThreeBetHappyBot(rng)
    if name == "fold_to_3bet":
        return FoldTo3BetBot(rng)
    if name == "random":
        return RandomBot(rng)
    if name == "human_like":
        return HumanLikeBot(rng=rng)
    if name == "gear_changer":
        return GearChangerBot(rng)
    if name == "adaptive_br":
        return AdaptiveBRBot(rng)
    if name == "counter_exploiter":
        return CounterExploiterBot(rng)
    if name == "pro_balanced":
        return HSProBalancedBot(rng, **(params or {}))
    if name == "pro_aggro":
        return HSProAggroBot(rng, **(params or {}))
    if name == "blueprint":
        return BlueprintBot(rng)
    if name == "alpha_hu":
        from pokerbot.opponents.alpha_nl_holdem import AlphaNLHoldemBot

        return AlphaNLHoldemBot()
    if name == "cnn" and model is not None:
        return GTOCopyBot(model, device=device)
    raise ValueError(f"Unknown bot: {name}")


BOT_REGISTRY = {
    "nit": NitBot,
    "lag": LAGBot,
    "calling_station": CallingStationBot,
    "tag": TagBot,
    "donk": DonkBot,
    "three_bet_happy": ThreeBetHappyBot,
    "fold_to_3bet": FoldTo3BetBot,
    "random": RandomBot,
    "human_like": HumanLikeBot,
}
