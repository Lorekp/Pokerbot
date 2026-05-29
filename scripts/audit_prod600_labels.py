#!/usr/bin/env python3
"""Audit prod600 solver labels against coverage and convergence requirements."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROD60_EXCLUDE_BOARDS = {"Td9d6h", "Ah7c2s", "Ks8h3d", "QcJh4s", "8c7d2d", "8c7d2h"}
TARGET_TOTAL = 600
TARGET_VANILLA = 400
TARGET_BUNCHING = 200
TARGET_BOARDS_MIN = 50
MAX_EXPLOITABILITY_PCT = 0.5
TEXTURE_DOMINANCE_THRESHOLD = 0.30

RANK_ORDER = "23456789TJQKA"


def _board_str(board: list[str]) -> str:
    return "".join(board)


def _rank_value(rank: str) -> int:
    return RANK_ORDER.index(rank.upper())


def _extract_board(row: pd.Series) -> list[str]:
    raw = json.loads(row["raw_solver_json"])
    return list(raw["solver_request"]["board"])


def _row_key(row: pd.Series) -> str:
    raw = json.loads(row["raw_solver_json"])
    board = _board_str(raw["solver_request"]["board"])
    hero = "".join(raw.get("hero_hole") or [])
    return f"{raw.get('spot_id')}|{board}|hero={hero}"


def classify_board_texture(board: list[str]) -> dict[str, str]:
    ranks = [c[0].upper() for c in board]
    suits = [c[1].lower() for c in board]
    rank_vals = sorted(_rank_value(r) for r in ranks)

    paired = len(set(ranks)) < 3
    monotone = len(set(suits)) == 1
    rainbow = len(set(suits)) == 3
    two_tone = len(set(suits)) == 2

    gaps = [rank_vals[i + 1] - rank_vals[i] for i in range(len(rank_vals) - 1)]
    connected = max(gaps) <= 4 if gaps else False
    flush_draw = max(Counter(suits).values()) == 2

    wet = paired or monotone or connected or flush_draw
    dry = not wet

    return {
        "pairedness": "paired" if paired else "unpaired",
        "suit_pattern": "monotone" if monotone else ("rainbow" if rainbow else "two_tone"),
        "wetness": "wet" if wet else "dry",
    }


def _dominant_bucket(counts: Counter[str], total: int) -> tuple[str | None, float]:
    if total <= 0 or not counts:
        return None, 0.0
    label, count = counts.most_common(1)[0]
    return label, count / total


def audit_prod600(
    parquet_path: Path,
    manifest_path: Path,
    prod60_parquet: Path | None = None,
) -> dict[str, Any]:
    df = pd.read_parquet(parquet_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    boards: list[list[str]] = [_extract_board(df.iloc[i]) for i in range(len(df))]
    board_keys = [_board_str(b) for b in boards]

    spot_counts = df["spot_class"].value_counts().to_dict()
    vanilla_n = int(spot_counts.get("hu_postflop_2p", 0))
    bunching_n = int(spot_counts.get("pseudo_hu_bunching", 0))

    distinct_boards = sorted(set(board_keys))
    distinct_board_count = len(distinct_boards)

    prod60_boards: set[str] = set()
    if prod60_parquet and prod60_parquet.is_file():
        df60 = pd.read_parquet(prod60_parquet)
        prod60_boards = {_board_str(_extract_board(df60.iloc[i])) for i in range(len(df60))}

    overlap_prod60 = sorted(set(board_keys) & prod60_boards)
    overlap_excluded = sorted(set(board_keys) & PROD60_EXCLUDE_BOARDS)

    exploitabilities = df["solver_exploitability_pct"].astype(float)
    max_exploitability = float(exploitabilities.max()) if len(exploitabilities) else float("inf")
    stragglers = df.loc[exploitabilities > MAX_EXPLOITABILITY_PCT]

    per_spot = manifest.get("per_spot_results", [])
    elapsed = [float(r["elapsed_sec"]) for r in per_spot if "elapsed_sec" in r]
    solve_times = {
        "median_sec": float(np.median(elapsed)) if elapsed else None,
        "max_sec": float(np.max(elapsed)) if elapsed else None,
        "count": len(elapsed),
    }

    flagged = manifest.get("enumeration", {}).get("flagged_failures", [])
    conv = manifest.get("convergence", {})

    distinct_board_keys = sorted(set(board_keys))
    board_to_texture = {}
    for bk in distinct_board_keys:
        sample_board = next(b for b in boards if _board_str(b) == bk)
        board_to_texture[bk] = classify_board_texture(sample_board)

    pairedness = Counter(t["pairedness"] for t in board_to_texture.values())
    suit_pattern = Counter(t["suit_pattern"] for t in board_to_texture.values())
    wetness = Counter(t["wetness"] for t in board_to_texture.values())
    texture_n = len(distinct_board_keys)

    dom_paired, frac_paired = _dominant_bucket(pairedness, texture_n)
    dom_suit, frac_suit = _dominant_bucket(suit_pattern, texture_n)
    dom_wet, frac_wet = _dominant_bucket(wetness, texture_n)

    texture_flags: list[str] = []
    for name, frac in [("pairedness", frac_paired), ("suit_pattern", frac_suit), ("wetness", frac_wet)]:
        if frac > TEXTURE_DOMINANCE_THRESHOLD:
            texture_flags.append(f"{name} bucket >30% dominated ({frac:.1%})")

    gates: dict[str, bool] = {
        "total_600": len(df) == TARGET_TOTAL,
        "vanilla_400": vanilla_n == TARGET_VANILLA,
        "bunching_200": bunching_n == TARGET_BUNCHING,
        "boards_ge_50": distinct_board_count >= TARGET_BOARDS_MIN,
        "no_stragglers": max_exploitability <= MAX_EXPLOITABILITY_PCT,
        "no_flagged_failures": len(flagged) == 0,
        "no_prod60_board_overlap": len(overlap_prod60) == 0,
        "exclude_boards_absent": len(overlap_excluded) == 0,
    }
    passed = all(gates.values())

    report: dict[str, Any] = {
        "parquet_path": str(parquet_path),
        "manifest_path": str(manifest_path),
        "passed": passed,
        "gates": gates,
        "counts": {
            "total": len(df),
            "vanilla": vanilla_n,
            "bunching": bunching_n,
            "distinct_boards": distinct_board_count,
        },
        "spot_class_distribution": spot_counts,
        "board_texture": {
            "distinct_boards": texture_n,
            "pairedness": dict(pairedness),
            "suit_pattern": dict(suit_pattern),
            "wetness": dict(wetness),
            "flags": texture_flags,
        },
        "prod60_overlap": {
            "overlap_boards": overlap_prod60,
            "excluded_boards_present": overlap_excluded,
        },
        "exploitability": {
            "max_pct": max_exploitability,
            "median_pct": float(exploitabilities.median()) if len(exploitabilities) else None,
            "straggler_count": int(len(stragglers)),
            "straggler_keys": [_row_key(stragglers.iloc[i]) for i in range(len(stragglers))],
        },
        "solve_times": solve_times,
        "manifest_convergence": conv,
        "flagged_failures": flagged,
        "range_scope_distribution": manifest.get("range_scope_distribution", {}),
        "config_path": manifest.get("config_path"),
    }
    return report


def main() -> None:
    p = argparse.ArgumentParser(description="Audit prod600 label set")
    p.add_argument("--parquet", default="data/solver/train_prod600.parquet")
    p.add_argument("--manifest", default="data/solver/manifest_prod600.json")
    p.add_argument("--prod60-parquet", default="data/solver/train_prod60.parquet")
    p.add_argument("--output", default="logs/prod600_audit_report.json")
    args = p.parse_args()

    prod60 = Path(args.prod60_parquet) if args.prod60_parquet else None
    report = audit_prod600(Path(args.parquet), Path(args.manifest), prod60)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Audit {'PASSED' if report['passed'] else 'FAILED'}")
    print(f"  total={report['counts']['total']} vanilla={report['counts']['vanilla']} "
          f"bunching={report['counts']['bunching']} boards={report['counts']['distinct_boards']}")
    print(f"  max_exploitability={report['exploitability']['max_pct']:.4f}%")
    print(f"  solve_times median={report['solve_times']['median_sec']:.1f}s "
          f"max={report['solve_times']['max_sec']:.1f}s")
    if report["flagged_failures"]:
        print(f"  flagged_failures={len(report['flagged_failures'])}")
    if not report["passed"]:
        failed = [k for k, v in report["gates"].items() if not v]
        print(f"  failed_gates: {failed}")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
