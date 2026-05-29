"""Combo-exact HU showdown equity table (suit-isomorphism canonical keys)."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Literal

import numpy as np

from aof_equity_grids import _card_str_to_id, _equity_two_combos_exact
from aof_equity_score2 import (
    DENOM2,
    equity_float32_from_score2,
)

PayoffHuMode = Literal["table", "exact_enum", "mc_legacy"]

KEY_SPEC_VERSION = 1
ARTIFACT_SCHEMA_VERSION = 2
N_CANONICAL_KEYS = 97_513
DEFAULT_TABLE_PATH = Path("data/aof/hu_showdown_equity.npz")
DEFAULT_SCHEMA_PATH = Path("data/aof/hu_showdown_equity.schema.json")
DEFAULT_KEYS_FIXTURE = Path("bench/datasets/fixtures/full_keys.uint64.npy")
EVALUATOR_BACKEND = "ompeval_native"

_table_keys: np.ndarray | None = None
_table_eq: np.ndarray | None = None
_table_path: Path | None = None

_payoff_hu_counters = {"lookup": 0, "miss": 0}


def _inc_profiler(name: str) -> None:
    try:
        from aof_rl.profiling.league_profiler import get_profiler

        prof = get_profiler()
        if prof is not None:
            prof.inc_counter(name)
    except Exception:
        pass


def payoff_hu_counters() -> dict[str, int]:
    return dict(_payoff_hu_counters)


def reset_payoff_hu_counters() -> None:
    for k in _payoff_hu_counters:
        _payoff_hu_counters[k] = 0


def _card_rank_suit(card_id: int) -> tuple[int, int]:
    return card_id // 4, card_id % 4


def _sort_pair(
    a: tuple[int, int], b: tuple[int, int]
) -> tuple[tuple[int, int], tuple[int, int]]:
    if a[0] != b[0]:
        return (a, b) if a[0] > b[0] else (b, a)
    return (a, b) if a[1] <= b[1] else (b, a)


def _canonicalize_suits(
    cards: list[tuple[int, int]],
) -> list[tuple[int, int]]:
    suit_map: dict[int, int] = {}
    next_suit = 0
    out: list[tuple[int, int]] = []
    for rank, suit in cards:
        if suit not in suit_map:
            suit_map[suit] = next_suit
            next_suit += 1
        out.append((rank, suit_map[suit]))
    return out


def canonical_showdown_pattern(
    h0: int, h1: int, v0: int, v1: int
) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int], tuple[int, int]]:
    """
    Suit-isomorphism canonical 4-card pattern preserving hero/villain slots.

    Hero = (h0,h1), villain = (v0,v1). Only suit relabeling; no hero/villain swap.
    """
    raw = [
        _card_rank_suit(h0),
        _card_rank_suit(h1),
        _card_rank_suit(v0),
        _card_rank_suit(v1),
    ]
    canon = _canonicalize_suits(raw)
    hp = _sort_pair(canon[0], canon[1])
    vp = _sort_pair(canon[2], canon[3])
    return (hp[0], hp[1], vp[0], vp[1])


def canonical_showdown_key(h0: int, h1: int, v0: int, v1: int) -> int:
    """Packed uint64 key for NPZ lookup (schema version 1)."""
    pat = canonical_showdown_pattern(h0, h1, v0, v1)
    key = 0
    for i, (rank, suit) in enumerate(pat):
        part = ((rank & 0xF) << 2) | (suit & 0x3)
        key |= part << (16 * i)
    return int(key)


def key_spec_hash() -> str:
    payload = json.dumps(
        {"version": KEY_SPEC_VERSION, "name": "canonical_showdown_key_v1"},
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _git_commit() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        return out.strip()[:12]
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _assert_keys_sorted(keys: np.ndarray, *, path: Path | str | None = None) -> None:
    if len(keys) <= 1:
        return
    if not np.all(keys[:-1] <= keys[1:]):
        label = f" at {path}" if path is not None else ""
        raise AssertionError(f"keys not sorted ascending{label}")


def _key_search_index(keys: np.ndarray, key: int) -> int:
    """searchsorted on uint64 keys; Python int must be cast to avoid lookup bugs."""
    return int(np.searchsorted(keys, np.uint64(key)))


def cards_to_ids(
    hero: tuple[str, str], vill: tuple[str, str]
) -> tuple[int, int, int, int]:
    return (
        _card_str_to_id(hero[0]),
        _card_str_to_id(hero[1]),
        _card_str_to_id(vill[0]),
        _card_str_to_id(vill[1]),
    )


def equity_hu_showdown_exact(
    hero: tuple[str, str],
    vill: tuple[str, str],
    *,
    table_path: Path | str | None = None,
    mode: PayoffHuMode = "table",
) -> float:
    """
    Hero equity in [0,1] for a specific combo matchup.

    mode=table: O(1) NPZ lookup (fail-closed if missing).
    mode=exact_enum: full board enumeration (no artifact).
    mode=mc_legacy: deprecated MC path for measurement only.
    """
    h0, h1, v0, v1 = cards_to_ids(hero, vill)
    if mode == "exact_enum":
        return float(_equity_two_combos_exact(h0, h1, v0, v1))
    if mode == "mc_legacy":
        from aof_equity import equity_hu_exact

        return equity_hu_exact(hero, vill, 1200, 0)
    return _lookup_table(h0, h1, v0, v1, table_path=table_path)


def _lookup_table(
    h0: int, h1: int, v0: int, v1: int, *, table_path: Path | str | None
) -> float:
    global _table_keys, _table_eq, _table_path
    path = Path(table_path) if table_path is not None else DEFAULT_TABLE_PATH
    if _table_keys is None or _table_path != path:
        load_showdown_table(path)
    assert _table_keys is not None and _table_eq is not None
    key = canonical_showdown_key(h0, h1, v0, v1)
    idx = _key_search_index(_table_keys, key)
    if idx < len(_table_keys) and int(_table_keys[idx]) == key:
        _payoff_hu_counters["lookup"] += 1
        _inc_profiler("payoff.hu.lookup")
        return float(_table_eq[idx])
    _payoff_hu_counters["miss"] += 1
    _inc_profiler("payoff.hu.miss")
    raise KeyError(
        f"HU showdown key {key} not in table {path} "
        f"(miss #{_payoff_hu_counters['miss']})"
    )


def load_showdown_table(path: Path | str) -> None:
    global _table_keys, _table_eq, _table_path
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"HU showdown table not found: {path}")
    data = np.load(path, allow_pickle=False)
    keys = np.asarray(data["keys"], dtype=np.uint64)
    eq = np.asarray(data["equity"], dtype=np.float32)
    _assert_keys_sorted(keys, path=path)
    if "score2" in data:
        score2 = np.asarray(data["score2"], dtype=np.uint32)
        if len(score2) != len(keys):
            raise ValueError(f"score2 length mismatch in {path}")
        expected = equity_float32_from_score2(score2)
        if not np.array_equal(eq, expected):
            raise ValueError(f"equity != float32(score2/DENOM2) in {path}")
    _table_keys = keys
    _table_eq = eq
    _table_path = path


def clear_showdown_table_cache() -> None:
    global _table_keys, _table_eq, _table_path
    _table_keys = None
    _table_eq = None
    _table_path = None


def save_showdown_table(
    path: Path,
    keys: np.ndarray,
    equity: np.ndarray | None = None,
    *,
    score2: np.ndarray | None = None,
    schema_path: Path | None = None,
    meta: dict[str, object] | None = None,
) -> None:
    if score2 is None and equity is None:
        raise ValueError("save_showdown_table requires score2 or equity")
    path.parent.mkdir(parents=True, exist_ok=True)
    order = np.argsort(keys)
    keys_s = np.asarray(keys, dtype=np.uint64)[order]
    if score2 is not None:
        score2_s = np.asarray(score2, dtype=np.uint32)[order]
        eq_s = np.asarray(equity_float32_from_score2(score2_s), dtype=np.float32)
    else:
        score2_s = None
        eq_s = np.asarray(equity, dtype=np.float32)[order]
    if score2_s is not None:
        np.savez_compressed(path, keys=keys_s, score2=score2_s, equity=eq_s)
    else:
        np.savez_compressed(path, keys=keys_s, equity=eq_s)
    schema = schema_path or path.with_suffix(".schema.json")
    payload: dict[str, object] = {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION if score2_s is not None else 1,
        "key_spec_version": KEY_SPEC_VERSION,
        "key_spec_hash": key_spec_hash(),
        "denom2": DENOM2,
        "n_keys": int(len(keys_s)),
        **(meta or {}),
    }
    if score2_s is not None:
        payload["evaluator_backend"] = EVALUATOR_BACKEND
        if "build_git_commit" not in payload:
            payload["build_git_commit"] = _git_commit()
    schema.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def save_showdown_table_npy(dir: Path, keys: np.ndarray, score2: np.ndarray) -> None:
    """Write flat uncompressed .npy staging files (internal builder use)."""
    dir.mkdir(parents=True, exist_ok=True)
    order = np.argsort(keys)
    np.save(dir / "keys.npy", np.asarray(keys, dtype=np.uint64)[order])
    np.save(dir / "score2.npy", np.asarray(score2, dtype=np.uint32)[order])


def load_canonical_keys(keys_path: Path | None = None) -> np.ndarray:
    path = keys_path or DEFAULT_KEYS_FIXTURE
    keys = np.load(path, allow_pickle=False)
    keys = np.asarray(keys, dtype=np.uint64)
    if len(keys) != N_CANONICAL_KEYS:
        raise ValueError(f"expected {N_CANONICAL_KEYS} keys, got {len(keys)} from {path}")
    _assert_keys_sorted(keys, path=path)
    return keys


def build_showdown_table_v2(
    keys: np.ndarray | None = None,
    *,
    threads: int = 0,
    keys_path: Path | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build sorted keys + score2 + derived equity via native batch builder."""
    from tools.ompeval_cli.native import build_scores, native_backend_available

    if not native_backend_available():
        raise RuntimeError("native build_scores backend not available")
    if keys is None:
        keys = load_canonical_keys(keys_path)
    else:
        keys = np.asarray(keys, dtype=np.uint64)
        _assert_keys_sorted(keys)
    score2 = np.asarray(build_scores(keys, threads), dtype=np.uint32)
    if len(score2) != len(keys):
        raise ValueError(f"build_scores returned {len(score2)} scores for {len(keys)} keys")
    equity = np.asarray(equity_float32_from_score2(score2), dtype=np.float32)
    return keys, score2, equity


def enumerate_all_disjoint_pairs() -> list[tuple[int, int, int, int]]:
    """All ordered hero/villain disjoint 2+2 card tuples (h0,h1,v0,v1)."""
    out: list[tuple[int, int, int, int]] = []
    for h0 in range(52):
        for h1 in range(h0 + 1, 52):
            for v0 in range(52):
                if v0 in (h0, h1):
                    continue
                for v1 in range(v0 + 1, 52):
                    if v1 in (h0, h1):
                        continue
                    out.append((h0, h1, v0, v1))
    return out


def build_showdown_equity_table(
    pairs: list[tuple[int, int, int, int]] | None = None,
    *,
    progress_every: int = 5000,
) -> tuple[np.ndarray, np.ndarray]:
    """Build sorted keys + equity arrays from disjoint pairs."""
    if pairs is None:
        pairs = enumerate_all_disjoint_pairs()
    key_to_eq: dict[int, float] = {}
    for i, (h0, h1, v0, v1) in enumerate(pairs):
        k = canonical_showdown_key(h0, h1, v0, v1)
        if k not in key_to_eq:
            key_to_eq[k] = float(_equity_two_combos_exact(h0, h1, v0, v1))
        if progress_every and (i + 1) % progress_every == 0:
            print(f"  ... {i + 1}/{len(pairs)} raw pairs, {len(key_to_eq)} keys", flush=True)
    keys = np.array(sorted(key_to_eq.keys()), dtype=np.uint64)
    eq = np.array([key_to_eq[int(k)] for k in keys], dtype=np.float32)
    return keys, eq
