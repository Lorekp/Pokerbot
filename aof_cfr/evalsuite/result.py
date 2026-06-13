"""Versioned eval-suite artifact: one metrics.json per run, diffable across changes.

Header pins everything the numbers depend on (git sha, CFR-base manifest hash, infoset key spec,
game model version, in-model showdown fee, seeds) so a before/after diff is meaningful.

Fee accounting (DC2-resolved 2026-06-12): the fee is a flat 0.12BB deducted from CONTESTED pots
at showdown — NOT per dealt hand. When the shipped base was solved with the fee in-model
(manifest ``flat_fee_bb`` > 0) the arena settlements already charge it, so ``net_bb100`` ==
gross. For legacy no-fee artifacts we keep the old unconditional −12 bb/100 subtraction as a
deliberately conservative floor (true incidence is contested-pots-only, far cheaper).
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

SUITE_VERSION = 1
FLAT_FEE_BB_PER_HAND = 0.12          # CoinPoker AoF NLHE flat fee (user-confirmed schedule)
FLAT_FEE_BB100 = FLAT_FEE_BB_PER_HAND * 100.0
_MANIFEST = Path("data/aof/cfr_base/cfr_base.manifest.json")


def _fee_in_model() -> float:
    try:
        from aof_cfr.pushfold_payoffs import flat_fee_default
        return float(flat_fee_default())
    except ImportError:
        return 0.0


@dataclass
class AoFEvalCell:
    metric: str                       # e.g. "uplift", "winrate", "nash_gap", "coldstart", ...
    players: int
    depth: float
    field_name: str                   # opponent field / adversary id ("-" if n/a)
    arm: str                          # "gto" | "shift_frozen" | "adaptive_online" | "-"
    hands: int
    seeds: int
    mean_bb100: float
    se_bb100: float
    ci95_bb100: float
    gate: str | None = None           # human-readable gate expression
    passed: bool | None = None
    meta: dict = field(default_factory=dict)

    @property
    def net_bb100(self) -> float:
        if _fee_in_model() > 0.0:      # fee charged inside the settlements -> gross IS net
            return self.mean_bb100
        return self.mean_bb100 - FLAT_FEE_BB100

    def to_dict(self) -> dict:
        d = asdict(self)
        if self.metric == "winrate":
            d["net_bb100"] = round(self.net_bb100, 3)
        return d


def _git_sha() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                              text=True, timeout=10).stdout.strip() or "unknown"
    except Exception:  # noqa: BLE001
        return "unknown"


def _manifest_hash() -> str:
    try:
        return hashlib.sha256(_MANIFEST.read_bytes()).hexdigest()[:16]
    except OSError:
        return "missing"


def _game_model_version() -> int:
    try:
        from aof_cfr.pushfold_payoffs import GAME_MODEL_VERSION
        return int(GAME_MODEL_VERSION)
    except ImportError:
        return 1                       # pre-Phase-4 chip model


def report_header(seeds) -> dict:
    from aof_cfr.infoset import infoset_key_spec_hash
    return {
        "suite_version": SUITE_VERSION,
        "git_sha": _git_sha(),
        "cfr_base_manifest_hash": _manifest_hash(),
        "infoset_key_spec_hash": infoset_key_spec_hash(),
        "game_model_version": _game_model_version(),
        "flat_fee_bb_per_hand": FLAT_FEE_BB_PER_HAND,
        "flat_fee_bb_in_model": _fee_in_model(),
        "seeds": list(seeds),
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def write_report(cells: list[AoFEvalCell], out_dir: str | Path, *, seeds=(), tag: str = "") -> Path:
    """Write ``<out_dir>/<stamp>_<gitsha>[_<tag>]/metrics.json`` (or ``<out_dir>/<tag>/`` when a
    tag is given — used to freeze named baselines). Returns the metrics.json path."""
    header = report_header(seeds)
    sub = tag if tag else f"{header['timestamp'].replace(':', '').replace('+0000', 'Z')}_{header['git_sha']}"
    d = Path(out_dir) / sub
    d.mkdir(parents=True, exist_ok=True)
    # A GATED cell is one with a gate expression. passed True -> pass, False -> fail, None on a
    # gated cell -> ABSTENTION (e.g. underpowered safety cell): it must NOT read as a pass. We fail
    # the suite verdict on any failure OR abstention, and surface the counts so a "certified" run
    # can't quietly have skipped a gate. (Ungated/informational cells have gate=None and are ignored.)
    gated = [c for c in cells if c.gate is not None]
    n_failed = sum(1 for c in gated if c.passed is False)
    n_skipped = sum(1 for c in gated if c.passed is None)
    payload = {"header": header,
               "gates": {"all_passed": (n_failed == 0 and n_skipped == 0),
                         "n_gated": len(gated), "n_failed": n_failed, "n_skipped": n_skipped},
               "cells": [c.to_dict() for c in cells]}
    p = d / "metrics.json"
    p.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    return p
