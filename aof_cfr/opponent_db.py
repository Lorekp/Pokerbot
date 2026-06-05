"""Persistent per-opponent push/fold database for the AoF exploit layer.

The core IP of the exploitative engine: every push/fold an opponent makes is a labelled
observation at a fully-specified infoset (position, hand bucket, history, stack-depth — the
v2 infoset key). We store sufficient statistics ``[push_count, total_count]`` per
``(identity, infoset_key)`` across every session ever played, so the profiler can estimate
an opponent's strategy (and how exploitable they are) before the first hand of a new session.

Storage mirrors ``pokerbot/opponent/player_profiles.py``: a single atomic JSON keyed by a
persistent identity (``user_id`` when available, else ``name:<username>``), idempotent by
``hand_id``. Fine for thousands of players; migrate the same accessors to sqlite past ~10k.

Play-money / own-account / bot-arena use only.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_PATH = "data/aof_player_profiles/profiles.json"


def identity_key(user_id: object | None, name: str | None) -> str | None:
    """Persistent identity: user_id if present, else name:<username>, else None (anonymous)."""
    if user_id:
        return str(user_id)
    if name:
        return f"name:{name}"
    return None


@dataclass
class AoFOpponentDB:
    path: Path = field(default_factory=lambda: Path(DEFAULT_PATH))
    # identity -> {infoset_key: [push_count, total_count]}
    counts: dict[str, dict[str, list[int]]] = field(default_factory=dict)
    names: dict[str, str] = field(default_factory=dict)
    seen_hands: set[str] = field(default_factory=set)
    _dirty: bool = False

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        self.load()

    def load(self) -> None:
        if not self.path.exists():
            return
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        self.counts = {
            ident: {k: [int(v[0]), int(v[1])] for k, v in rows.items()}
            for ident, rows in (raw.get("counts") or {}).items()
        }
        self.names = dict(raw.get("names") or {})
        self.seen_hands = set(raw.get("seen_hands") or [])

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "counts": self.counts,
            "names": self.names,
            "seen_hands": sorted(self.seen_hands),
        }
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=0), encoding="utf-8")
        os.replace(tmp, self.path)  # atomic
        self._dirty = False

    def record_action(self, identity: str, infoset_key: str, pushed: bool) -> None:
        """Fold one observed push/fold decision into the counts for ``identity``."""
        rows = self.counts.setdefault(identity, {})
        cell = rows.setdefault(infoset_key, [0, 0])
        cell[0] += 1 if pushed else 0
        cell[1] += 1
        self._dirty = True

    def record_hand(self, hand_id: str | None, observations: list[tuple[str, str, bool]]) -> int:
        """Idempotently fold one hand's observations ``[(identity, infoset_key, pushed)]``.

        Re-recording the same ``hand_id`` is a no-op (restart / log reprocessing safe).
        Returns the number of observations recorded (0 if the hand was already seen).
        """
        if hand_id is not None:
            if hand_id in self.seen_hands:
                return 0
            self.seen_hands.add(hand_id)
        for identity, infoset_key, pushed in observations:
            if not identity:
                continue
            self.record_action(identity, infoset_key, pushed)
            self._dirty = True
        return len(observations)

    def cell(self, identity: str, infoset_key: str) -> tuple[int, int]:
        """(push_count, total_count) for an infoset; (0, 0) if unseen."""
        c = self.counts.get(identity, {}).get(infoset_key)
        return (int(c[0]), int(c[1])) if c else (0, 0)

    def n_obs(self, identity: str) -> int:
        return sum(t for _, t in self.counts.get(identity, {}).values())

    @property
    def dirty(self) -> bool:
        return self._dirty
