"""PokerBench dataset loading and synthetic fallback."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Literal

Split = Literal["train", "test"]
Stage = Literal["preflop", "postflop"]


@dataclass(frozen=True)
class PokerBenchSample:
    instruction: str
    output: str
    stage: Stage
    split: Split

    # Structured fields when loaded from CSV-style metadata
    hero_position: str | None = None
    hero_holding: str | None = None
    pot_size: float | None = None


SYNTHETIC_PREFLOP = [
    {
        "instruction": (
            "You are a specialist in playing 6-handed No Limit Texas Holdem. "
            "The small blind is 0.5 chips and the big blind is 1 chips. Everyone started with 100 chips. "
            "The player positions involved in this game are UTG, HJ, CO, BTN, SB, BB. "
            "In this hand, your position is CO, and your holding is [Ace of Heart and King of Spade]. "
            "Before the flop, UTG fold, HJ fold, and CO is first to act. "
            "Now it is your turn to make a move. The current pot size is 1.5 chips. "
            "Do not explain your answer. Your optimal action is:"
        ),
        "output": "raise 2.0",
    },
    {
        "instruction": (
            "You are a specialist in playing 6-handed No Limit Texas Holdem. "
            "The small blind is 0.5 chips and the big blind is 1 chips. Everyone started with 100 chips. "
            "In this hand, your position is BB, and your holding is [Seven of Heart and Two of Diamond]. "
            "Before the flop, UTG raise 2.0, everyone else fold. "
            "Now it is your turn to make a move. The current pot size is 4.0 chips. "
            "Do not explain your answer. Your optimal action is:"
        ),
        "output": "fold",
    },
]

SYNTHETIC_POSTFLOP = [
    {
        "instruction": (
            "You are a specialist in playing 6-handed No Limit Texas Holdem. "
            "In this hand, your position is IP, holding [Eight of Heart and Eight of Club]. "
            "Board: King of Spade, Seven of Heart, Two of Diamond, Jack of Club, Seven of Club. "
            "Post-flop action: OOP check / IP check / turn check / OOP bet 5 / IP raise 14. "
            "Pot size is 32 chips. Do not explain your answer. Your optimal action is:"
        ),
        "output": "check",
    },
    {
        "instruction": (
            "You are a specialist in playing 6-handed No Limit Texas Holdem. "
            "In this hand, your position is OOP, holding [Ace of Diamond and Queen of Club]. "
            "Board: Ace of Spade, Nine of Heart, Four of Diamond. "
            "Post-flop action: OOP check / IP bet 3. "
            "Pot size is 12 chips. Do not explain your answer. Your optimal action is:"
        ),
        "output": "call",
    },
]


def _synthetic_samples(stage: Stage, split: Split, limit: int | None) -> list[PokerBenchSample]:
    pool = SYNTHETIC_PREFLOP if stage == "preflop" else SYNTHETIC_POSTFLOP
    samples = [
        PokerBenchSample(
            instruction=row["instruction"],
            output=row["output"],
            stage=stage,
            split=split,
        )
        for row in pool
    ]
    if limit is not None:
        samples = samples[:limit]
    return samples


def load_from_huggingface(
    stage: Stage,
    split: Split,
    limit: int | None = None,
    cache_dir: Path | None = None,
) -> list[PokerBenchSample]:
    """Load PokerBench JSON prompts from HuggingFace (RZ412/PokerBench)."""
    from huggingface_hub import hf_hub_download

    fname = {
        ("preflop", "train"): "preflop_60k_train_set_prompt_and_label.json",
        ("preflop", "test"): "preflop_1k_test_set_prompt_and_label.json",
        ("postflop", "train"): "postflop_500k_train_set_prompt_and_label.json",
        ("postflop", "test"): "postflop_10k_test_set_prompt_and_label.json",
    }[(stage, split)]

    if cache_dir is not None:
        local = cache_dir / fname
        if local.exists():
            return load_from_json(local, stage, split, limit)

    path = Path(
        hf_hub_download(
            repo_id="RZ412/PokerBench",
            repo_type="dataset",
            filename=fname,
            cache_dir=str(cache_dir) if cache_dir else None,
        )
    )
    return load_from_json(path, stage, split, limit)


def load_from_json(path: Path, stage: Stage, split: Split, limit: int | None = None) -> list[PokerBenchSample]:
    with path.open(encoding="utf-8") as f:
        rows = json.load(f)
    if limit is not None:
        rows = rows[:limit]
    return [
        PokerBenchSample(
            instruction=str(r["instruction"]),
            output=str(r["output"]).strip().lower(),
            stage=stage,
            split=split,
        )
        for r in rows
    ]


def load_pokerbench(
    stage: Stage = "preflop",
    split: Split = "train",
    limit: int | None = None,
    data_dir: Path | None = None,
    use_synthetic_fallback: bool = True,
) -> list[PokerBenchSample]:
    """
    Load PokerBench samples. Tries HuggingFace, then local JSON, then synthetic mini-set.
    """
    if data_dir is not None:
        fname = {
            ("preflop", "train"): "preflop_60k_train_set_prompt_and_label.json",
            ("preflop", "test"): "preflop_1k_test_set_prompt_and_label.json",
            ("postflop", "train"): "postflop_500k_train_set_prompt_and_label.json",
            ("postflop", "test"): "postflop_10k_test_set_prompt_and_label.json",
        }[(stage, split)]
        path = data_dir / fname
        if path.exists():
            return load_from_json(path, stage, split, limit)

    try:
        return load_from_huggingface(stage, split, limit, data_dir)
    except Exception:
        if use_synthetic_fallback:
            return _synthetic_samples(stage, split, limit)
        raise


def iter_all_training(
    preflop_limit: int | None = None,
    postflop_limit: int | None = None,
    data_dir: Path | None = None,
) -> Iterator[PokerBenchSample]:
    for s in load_pokerbench("preflop", "train", preflop_limit, data_dir):
        yield s
    for s in load_pokerbench("postflop", "train", postflop_limit, data_dir):
        yield s


def train_val_split(
    samples: list[PokerBenchSample],
    val_frac: float = 0.05,
    seed: int = 42,
) -> tuple[list[PokerBenchSample], list[PokerBenchSample]]:
    rng = random.Random(seed)
    idx = list(range(len(samples)))
    rng.shuffle(idx)
    cut = max(1, int(len(samples) * (1 - val_frac)))
    train = [samples[i] for i in idx[:cut]]
    val = [samples[i] for i in idx[cut:]]
    return train, val
