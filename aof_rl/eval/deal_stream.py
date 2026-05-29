"""Deterministic deal stream for common-random-number evaluation."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Iterator

@dataclass(frozen=True)
class DealSpec:
    hand_index: int
    table_size: int
    seats: tuple[str, ...]
    holes: dict[str, tuple[str, str]]
    payoff_base_seed: int


class DealStream:
    """Yield identical deals for paired policy comparisons."""

    def __init__(
        self,
        *,
        hands: int,
        base_seed: int,
        table_sizes: tuple[int, ...] = (2, 3, 4),
        stack_bb: float = 8.0,
    ) -> None:
        self.hands = hands
        self.base_seed = base_seed
        self.table_sizes = table_sizes
        self.stack_bb = stack_bb

    def _table_size_for(self, hand_index: int) -> int:
        if len(self.table_sizes) == 1:
            return self.table_sizes[0]
        rng = random.Random(self.base_seed + hand_index * 7919)
        return rng.choice(self.table_sizes)

    def iter_deals(self) -> Iterator[DealSpec]:
        from aof_rl.env import reset_hand_from_deal_index

        for hand_index in range(self.hands):
            table_size = self._table_size_for(hand_index)
            state = reset_hand_from_deal_index(
                hand_index,
                table_size=table_size,
                stack_bb=self.stack_bb,
                base_seed=self.base_seed,
            )
            yield DealSpec(
                hand_index=hand_index,
                table_size=table_size,
                seats=state.seats,
                holes=dict(state.holes),
                payoff_base_seed=self.base_seed + hand_index * 17,
            )

    def iter_for_seed(self, eval_seed: int) -> Iterator[DealSpec]:
        """Deals for one evaluation seed (offset base for multi-seed runs)."""
        sub = DealStream(
            hands=self.hands,
            base_seed=self.base_seed ^ (eval_seed * 0x85EBCA6B),
            table_sizes=self.table_sizes,
            stack_bb=self.stack_bb,
        )
        yield from sub.iter_deals()
