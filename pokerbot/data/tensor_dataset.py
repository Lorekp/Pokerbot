"""Bridge PokerBench samples to tensor training batches."""

from __future__ import annotations

import re
from typing import Iterator

import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset

from pokerbot.data.pokerbench import PokerBenchSample
from pokerbot.data.pokerbench_csv import CSVSample, iter_csv_rows
from pokerbot.engine.actions import NUM_ACTIONS
from pokerbot.encoding.csv_encoder import encode_csv_row
from pokerbot.eval.pokerbench_eval import label_to_action_index
from pokerbot.encoding.tensors import SPEC


# Regex helpers for NL fallback (legacy JSON path)
_HOLDING_RE = re.compile(r"holding is \[(.+?)\]", re.I)
_POSITION_RE = re.compile(r"your position is ([A-Z]{2,3}|IP|OOP)", re.I)
_POT_RE = re.compile(r"pot size is ([\d.]+)", re.I)


def _cards_from_text(text: str) -> np.ndarray:
    tensor = np.zeros(SPEC.card_shape, dtype=np.float32)
    from pokerbot.engine.cards import Card, RANK_CHARS, SUIT_CHARS

    rank_map = {
        "ace": "A", "king": "K", "queen": "Q", "jack": "J", "ten": "T",
        "nine": "9", "eight": "8", "seven": "7", "six": "6", "five": "5",
        "four": "4", "three": "3", "two": "2",
    }
    suit_map = {"heart": "h", "hearts": "h", "diamond": "d", "diamonds": "d",
                "spade": "s", "spades": "s", "club": "c", "clubs": "c"}

    def _try_card(words: list[str]) -> Card | None:
        if len(words) < 2:
            return None
        r = rank_map.get(words[0])
        s = suit_map.get(words[-1])
        if r is None and len(words[0]) == 1:
            c = words[0].upper()
            r = c if c in RANK_CHARS else None
        if s is None and len(words[-1]) == 1:
            c = words[-1].lower()
            s = c if c in SUIT_CHARS else None
        if not r or not s:
            return None
        try:
            return Card.from_str(f"{r}{s}")
        except ValueError:
            return None

    slot = 0
    holding = _HOLDING_RE.search(text)
    if holding:
        parts = holding.group(1).split(" and ")
        for part in parts:
            words = part.strip().lower().replace(" of ", " ").split()
            card = _try_card(words)
            if card is None:
                continue
            tensor[slot, int(card.suit), card.rank - 2] = 1.0
            slot += 1

    board_match = re.search(
        r"board:?\s*(.+?)(?:\.|post-flop|pot size|your position|$)", text, re.I
    )
    if board_match:
        blob = board_match.group(1)
        for part in re.split(r",|\band\b", blob):
            words = part.strip().lower().replace(" of ", " ").split()
            card = _try_card(words)
            if card is None:
                continue
            if slot < SPEC.card_shape[0]:
                tensor[slot, int(card.suit), card.rank - 2] = 1.0
                slot += 1
    return tensor


def _actions_from_text(text: str) -> np.ndarray:
    tensor = np.zeros(SPEC.action_shape, dtype=np.float32)
    pot = 1.5
    pot_m = _POT_RE.search(text)
    if pot_m:
        pot = float(pot_m.group(1))
    tensor[:, :, 6] = pot / 200.0
    if "fold" in text.lower():
        tensor[0, 0, 1] = 0.5
    if "raise" in text.lower() or "bet" in text.lower():
        tensor[0, 0, 3] = 0.5
    return tensor


def _position_from_text(text: str) -> np.ndarray:
    pos = np.zeros(6, dtype=np.float32)
    m = _POSITION_RE.search(text)
    if m:
        name = m.group(1).upper()
        mapping = {"UTG": 0, "HJ": 1, "CO": 2, "BTN": 3, "SB": 4, "BB": 5, "IP": 2, "OOP": 0}
        idx = mapping.get(name, 0)
        pos[idx] = 1.0
    else:
        pos[0] = 1.0
    return pos


def _encoded_to_batch_item(encoded: dict) -> dict[str, torch.Tensor]:
    return {
        "cards": torch.from_numpy(encoded["cards"]),
        "actions": torch.from_numpy(encoded["actions"]),
        "position": torch.from_numpy(encoded["position"]),
        "label": torch.tensor(encoded["label"], dtype=torch.long),
        "legal_mask": torch.from_numpy(encoded["legal_mask"]),
    }


class PokerBenchTensorDataset(Dataset):
    """Legacy JSON/NL-backed dataset."""

    def __init__(self, samples: list[PokerBenchSample]) -> None:
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        s = self.samples[idx]
        cards = _cards_from_text(s.instruction)
        actions = _actions_from_text(s.instruction)
        position = _position_from_text(s.instruction)
        action_idx = label_to_action_index(s.output)
        legal = torch.ones(NUM_ACTIONS, dtype=torch.bool)
        return {
            "cards": torch.from_numpy(cards),
            "actions": torch.from_numpy(actions),
            "position": torch.from_numpy(position),
            "label": torch.tensor(action_idx, dtype=torch.long),
            "legal_mask": legal,
        }


class StructuredPokerBenchDataset(Dataset):
    """In-memory CSV-backed structured dataset."""

    def __init__(self, csv_samples: list[CSVSample]) -> None:
        self.csv_samples = csv_samples
        self._cache: list[dict[str, torch.Tensor]] | None = None

    def __len__(self) -> int:
        return len(self.csv_samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        sample = self.csv_samples[idx]
        encoded = encode_csv_row(sample.row, sample.stage)
        return _encoded_to_batch_item(encoded)


class StreamingStructuredDataset(IterableDataset):
    """Iterable CSV stream for full 560K training without RAM load."""

    def __init__(
        self,
        stage: str,
        split: str = "train",
        limit: int | None = None,
        cache_dir: str | None = None,
    ) -> None:
        self.stage = stage
        self.split = split
        self.limit = limit
        self.cache_dir = cache_dir

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        from pathlib import Path

        cache = Path(self.cache_dir) if self.cache_dir else Path("data/pokerbench")
        for sample in iter_csv_rows(
            self.stage,  # type: ignore[arg-type]
            self.split,  # type: ignore[arg-type]
            self.limit,
            cache,
        ):
            encoded = encode_csv_row(sample.row, sample.stage)
            yield _encoded_to_batch_item(encoded)


class CombinedStreamingDataset(IterableDataset):
    """Stream preflop then postflop CSV rows."""

    def __init__(
        self,
        preflop_limit: int | None = None,
        postflop_limit: int | None = None,
        cache_dir: str | None = None,
    ) -> None:
        self.preflop_limit = preflop_limit
        self.postflop_limit = postflop_limit
        self.cache_dir = cache_dir

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        yield from StreamingStructuredDataset("preflop", "train", self.preflop_limit, self.cache_dir)
        yield from StreamingStructuredDataset("postflop", "train", self.postflop_limit, self.cache_dir)


def collate_batch(batch: list[dict]) -> dict[str, torch.Tensor]:
    return {k: torch.stack([b[k] for b in batch]) for k in batch[0]}


def load_structured_samples(
    preflop_limit: int | None = None,
    postflop_limit: int | None = None,
    split: str = "train",
    cache_dir: str | None = None,
) -> list[CSVSample]:
    from pathlib import Path

    from pokerbot.data.pokerbench_csv import load_csv_samples

    cache = Path(cache_dir) if cache_dir else Path("data/pokerbench")
    pre = load_csv_samples("preflop", split, preflop_limit, cache)  # type: ignore[arg-type]
    post = load_csv_samples("postflop", split, postflop_limit, cache)  # type: ignore[arg-type]
    return pre + post
