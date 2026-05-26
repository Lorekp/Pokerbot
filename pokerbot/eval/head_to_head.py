"""Head-to-head bb/100 evaluation."""

from __future__ import annotations

import random
from dataclasses import dataclass

from pokerbot.engine.game import NUM_PLAYERS, GameState, play_hand
from pokerbot.eval.bots import BOT_REGISTRY, GTOCopyBot


@dataclass
class HeadToHeadResult:
    hero_seat: int
    hands: int
    total_bb: float

    @property
    def bb_per_100(self) -> float:
        return (self.total_bb / max(self.hands, 1)) * 100


def run_head_to_head(
    hero_policy,
    opponent_policy,
    hands: int = 1000,
    seed: int = 42,
) -> HeadToHeadResult:
    rng = random.Random(seed)
    total = 0.0
    hero_seat = 0
    for h in range(hands):
        hero_seat = h % NUM_PLAYERS
        policies = {i: opponent_policy for i in range(NUM_PLAYERS)}
        policies[hero_seat] = hero_policy
        _, deltas = play_hand(policies, rng=rng)
        total += deltas.get(hero_seat, 0.0)
    return HeadToHeadResult(hero_seat=hero_seat, hands=hands, total_bb=total)


def evaluate_vs_bot_pool(hero_policy, hands_per_bot: int = 200, seed: int = 42) -> dict[str, float]:
    results = {}
    for name, cls in BOT_REGISTRY.items():
        opp = cls(random.Random(seed))
        r = run_head_to_head(hero_policy, opp, hands=hands_per_bot, seed=seed)
        results[name] = r.bb_per_100
    return results
