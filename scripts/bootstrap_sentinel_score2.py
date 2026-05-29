#!/usr/bin/env python3
"""Bootstrap SENTINEL_SCORE2_AA_AKO from Python exact oracle (Gate A)."""

from __future__ import annotations

import subprocess
import sys

from aof_equity_score2 import DENOM2, equity_from_score2, score2_two_combos_exact
from aof_equity_showdown import canonical_showdown_key


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


def main() -> int:
    # Gate A: AcAd vs KhKd
    h0, h1, v0, v1 = 48, 49, 46, 45
    key = canonical_showdown_key(h0, h1, v0, v1)
    score2 = score2_two_combos_exact(h0, h1, v0, v1)
    eq = equity_from_score2(score2)
    commit = _git_commit()
    print(f"commit={commit}")
    print(f"cards=AcAd vs KhKd ({h0},{h1},{v0},{v1})")
    print(f"key={key}")
    print(f"score2={score2}")
    print(f"equity={eq:.17g}")
    print(f"DENOM2={DENOM2}")
    print()
    print(f"SENTINEL_KEY_AA_AKO = {key}")
    print(f"SENTINEL_SCORE2_AA_AKO = {score2}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
