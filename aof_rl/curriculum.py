"""Training curriculum stages for AoF PPO league."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Literal

SpotConstraint = Literal[
    "any",
    "open_or_facing",
    "max_one_shover",
    "min_two_shovers",
    "full",
]


@dataclass(frozen=True)
class CurriculumStage:
    name: str
    table_sizes: tuple[int, ...]
    spot_constraint: SpotConstraint = "any"
    flat_rake_bb: float = 0.0
    range_randomization: bool = False
    updates: int = 0


@dataclass
class CurriculumConfig:
    stages: list[CurriculumStage] = field(default_factory=list)

    @staticmethod
    def default_pipeline() -> CurriculumConfig:
        return CurriculumConfig(
            stages=[
                CurriculumStage("hu_only", (2,), "open_or_facing", updates=50),
                CurriculumStage(
                    "three_handed_one_shover",
                    (3,),
                    "max_one_shover",
                    updates=50,
                ),
                CurriculumStage(
                    "multiway_shovers", (3,), "min_two_shovers", updates=50
                ),
                CurriculumStage("four_handed_full", (4,), "full", updates=100),
                CurriculumStage(
                    "rake_randomized",
                    (2, 3, 4),
                    "full",
                    flat_rake_bb=0.05,
                    range_randomization=True,
                    updates=100,
                ),
            ]
        )


class CurriculumScheduler:
    def __init__(self, config: CurriculumConfig) -> None:
        self.config = config
        if not config.stages:
            config.stages = CurriculumConfig.default_pipeline().stages
        self._total = sum(s.updates for s in config.stages) or 1

    def stage_for_update(self, update: int) -> CurriculumStage:
        u = 0
        for stage in self.config.stages:
            u += max(stage.updates, 1)
            if update < u:
                return stage
        return self.config.stages[-1]

    def sample_table_size(self, stage: CurriculumStage, rng: random.Random) -> int:
        return int(rng.choice(stage.table_sizes))

    def sample_rake_bb(self, stage: CurriculumStage, rng: random.Random) -> float:
        if stage.flat_rake_bb <= 0:
            return 0.0
        return float(rng.uniform(0.0, stage.flat_rake_bb))
