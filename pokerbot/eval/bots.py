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
            return next(
                (a for a in legal if a.action_type in (ActionType.CHECK, ActionType.FOLD, ActionType.CALL)),
                legal[0],
            )
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

    def __init__(self, model, device="cpu", seat: int = 0) -> None:
        self.model = model
        self.device = device
        self.seat = seat

    def __call__(self, state: GameState) -> Action:
        import torch

        from pokerbot.encoding.tensors import encode_state
        from pokerbot.training.self_play import _index_to_action, _legal_mask

        seat = state.action_seat
        enc = encode_state(state, seat)
        weva_ids = torch.tensor([enc["weva_id"]], dtype=torch.long, device=self.device)
        actions = torch.from_numpy(enc["actions"]).unsqueeze(0).to(self.device)
        position = torch.from_numpy(enc["position"]).unsqueeze(0).to(self.device)
        mask = torch.from_numpy(_legal_mask(state, seat)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            idx, _ = self.model.act(weva_ids, actions, position, mask, deterministic=True)
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


def build_bot(name: str, rng: random.Random | None = None, model=None, device: str = "cpu"):
    """Factory for league / eval bots."""
    if name == "nit":
        return NitBot(rng)
    if name == "lag":
        return LAGBot(rng)
    if name == "donk":
        return DonkBot(rng)
    if name == "random":
        return RandomBot(rng)
    if name == "human_like":
        return HumanLikeBot(rng=rng)
    if name == "alpha_hu":
        from pokerbot.opponents.alpha_nl_holdem import AlphaNLHoldemBot

        return AlphaNLHoldemBot()
    if name == "cnn" and model is not None:
        return GTOCopyBot(model, device=device)
    raise ValueError(f"Unknown bot: {name}")


BOT_REGISTRY = {
    "nit": NitBot,
    "lag": LAGBot,
    "donk": DonkBot,
    "random": RandomBot,
    "human_like": HumanLikeBot,
}
