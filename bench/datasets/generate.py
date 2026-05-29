#!/usr/bin/env python3
"""Deterministic HU showdown benchmark fixture generator."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np

from aof_equity_grids import _card_str_to_id
from aof_equity_showdown import (
    canonical_showdown_key,
    enumerate_all_disjoint_pairs,
    key_spec_hash,
)
from bench._keys import unpack_key

N_CANONICAL = 97_513
SMOKE_N = 256
STRATIFIED_N = 4096
GATE_SAMPLE_N = 10_000
RUNTIME_SANITY_N = 1000

_RANKS = "23456789TJQKA"


def _card_id_to_str(cid: int) -> str:
    rank = cid // 4
    suit = cid % 4
    suits = "cdhs"
    return f"{_RANKS[rank]}{suits[suit]}"


def _pair_cards(h0: int, h1: int, v0: int, v1: int) -> tuple[int, int, int, int]:
    return (h0, h1, v0, v1)


def _build_full_keys() -> np.ndarray:
    keys: set[int] = set()
    for h0, h1, v0, v1 in enumerate_all_disjoint_pairs():
        keys.add(canonical_showdown_key(h0, h1, v0, v1))
    if len(keys) != N_CANONICAL:
        raise SystemExit(f"expected {N_CANONICAL} keys, got {len(keys)}")
    return np.array(sorted(keys), dtype=np.uint64)


def _handpick_smoke_keys(full: np.ndarray) -> list[int]:
    picks: list[tuple[int, int, int, int]] = [
        # Gate A: AcAd vs KhKd
        _pair_cards(
            _card_str_to_id("Ac"),
            _card_str_to_id("Ad"),
            _card_str_to_id("Kh"),
            _card_str_to_id("Kd"),
        ),
        # Pocket vs pocket
        _pair_cards(
            _card_str_to_id("Qs"),
            _card_str_to_id("Qh"),
            _card_str_to_id("Js"),
            _card_str_to_id("Jh"),
        ),
        # Hero suited vs vill offsuit
        _pair_cards(
            _card_str_to_id("Ah"),
            _card_str_to_id("Kh"),
            _card_str_to_id("Qc"),
            _card_str_to_id("Jd"),
        ),
        # Shared-suit flush blocker (hero AhKh vs QhJc)
        _pair_cards(
            _card_str_to_id("Ah"),
            _card_str_to_id("Kh"),
            _card_str_to_id("Qh"),
            _card_str_to_id("Jc"),
        ),
        # Low disconnected vs high
        _pair_cards(
            _card_str_to_id("2c"),
            _card_str_to_id("7d"),
            _card_str_to_id("Ac"),
            _card_str_to_id("Kd"),
        ),
        # Connected suited vs pair
        _pair_cards(
            _card_str_to_id("9s"),
            _card_str_to_id("8s"),
            _card_str_to_id("Tc"),
            _card_str_to_id("Td"),
        ),
        # Four distinct suits, wide ranks
        _pair_cards(
            _card_str_to_id("As"),
            _card_str_to_id("2h"),
            _card_str_to_id("3d"),
            _card_str_to_id("4c"),
        ),
        # Same rank different suits (split pot risk)
        _pair_cards(
            _card_str_to_id("5c"),
            _card_str_to_id("5d"),
            _card_str_to_id("5h"),
            _card_str_to_id("6s"),
        ),
    ]
    out: list[int] = []
    seen: set[int] = set()
    for h0, h1, v0, v1 in picks:
        k = int(canonical_showdown_key(h0, h1, v0, v1))
        if k not in seen:
            seen.add(k)
            out.append(k)
    for k in full:
        ki = int(k)
        if ki not in seen:
            seen.add(ki)
            out.append(ki)
        if len(out) >= SMOKE_N:
            break
    if len(out) < SMOKE_N:
        raise SystemExit(f"smoke keys: only {len(out)}")
    return out[:SMOKE_N]


def _topology_bucket(key: int) -> str:
    pat = unpack_key(key)
    hero = pat[0:2]
    vill = pat[2:4]

    def hand_type(pair: tuple[tuple[int, int], tuple[int, int]]) -> str:
        if pair[0][0] == pair[1][0]:
            return "pair"
        if pair[0][1] == pair[1][1]:
            return "suited"
        return "offsuit"

    ht = hand_type((hero[0], hero[1]))
    vt = hand_type((vill[0], vill[1]))
    suits = {c[1] for c in pat}
    suit_n = len(suits)
    rank_gap = abs(hero[0][0] - hero[1][0])
    gap_b = min(rank_gap, 5)
    return f"h{ht}_v{vt}_s{suit_n}_g{gap_b}"


def _stratified_keys(full: np.ndarray) -> list[int]:
    buckets: dict[str, list[int]] = defaultdict(list)
    for k in full:
        buckets[_topology_bucket(int(k))].append(int(k))
    bucket_names = sorted(buckets.keys())
    out: list[int] = []
    seen: set[int] = set()
    idx = 0
    while len(out) < STRATIFIED_N and bucket_names:
        name = bucket_names[idx % len(bucket_names)]
        if buckets[name]:
            k = buckets[name].pop(0)
            if k not in seen:
                seen.add(k)
                out.append(k)
        idx += 1
        if idx > STRATIFIED_N * len(bucket_names) * 2:
            break
    for k in full:
        ki = int(k)
        if ki not in seen:
            seen.add(ki)
            out.append(ki)
        if len(out) >= STRATIFIED_N:
            break
    if len(out) < STRATIFIED_N:
        raise SystemExit(f"stratified keys: only {len(out)}")
    return out[:STRATIFIED_N]


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def generate(out_dir: Path, seed: int) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    print("Building full key universe...", flush=True)
    full = _build_full_keys()
    np.save(out_dir / "full_keys.uint64.npy", full)

    smoke = np.array(_handpick_smoke_keys(full), dtype=np.uint64)
    np.save(out_dir / "smoke_keys.uint64.npy", smoke)

    strat = np.array(_stratified_keys(full), dtype=np.uint64)
    np.save(out_dir / "stratified_keys.uint64.npy", strat)

    pairs = enumerate_all_disjoint_pairs()
    rng = random.Random(seed)
    gate_pairs = rng.sample(pairs, GATE_SAMPLE_N)
    gate_rows = []
    for h0, h1, v0, v1 in gate_pairs:
        gate_rows.append(
            {
                "h0": h0,
                "h1": h1,
                "v0": v0,
                "v1": v1,
                "key": int(canonical_showdown_key(h0, h1, v0, v1)),
            }
        )
    _write_jsonl(out_dir / "gate_sample_pairs.jsonl", gate_rows)

    rng2 = random.Random(seed + 1)
    sanity_pairs = rng2.sample(pairs, RUNTIME_SANITY_N)
    sanity_rows = []
    for h0, h1, v0, v1 in sanity_pairs:
        sanity_rows.append(
            {
                "h0": h0,
                "h1": h1,
                "v0": v0,
                "v1": v1,
                "key": int(canonical_showdown_key(h0, h1, v0, v1)),
                "hero": [_card_id_to_str(h0), _card_id_to_str(h1)],
                "vill": [_card_id_to_str(v0), _card_id_to_str(v1)],
            }
        )
    _write_jsonl(out_dir / "runtime_sanity.jsonl", sanity_rows)

    files = {
        "full_keys.uint64.npy": len(full),
        "smoke_keys.uint64.npy": len(smoke),
        "stratified_keys.uint64.npy": len(strat),
        "gate_sample_pairs.jsonl": len(gate_rows),
        "runtime_sanity.jsonl": len(sanity_rows),
    }
    sha256 = {name: _sha256_file(out_dir / name) for name in files}
    bucket_counts: dict[str, int] = defaultdict(int)
    for k in strat:
        bucket_counts[_topology_bucket(int(k))] += 1

    manifest = {
        "seed": seed,
        "key_spec_hash": key_spec_hash(),
        "n_canonical_keys": N_CANONICAL,
        "files": files,
        "sha256": sha256,
        "stratified_bucket_counts": dict(sorted(bucket_counts.items())),
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return manifest


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("bench/datasets/fixtures"),
    )
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    manifest = generate(args.out_dir, args.seed)
    print(json.dumps(manifest, indent=2))
    print(f"Wrote fixtures -> {args.out_dir}")


if __name__ == "__main__":
    main()
