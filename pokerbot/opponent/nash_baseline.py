"""Near-Nash (tight-aggressive) per-context action-bin reference — the SAFETY baseline the opponent's
observed behavior is measured AGAINST when deciding how to deviate.

H7.2 / research (Ganzfried 2022): the safety-critical reference must be near-equilibrium, NOT the
population mean. The population is LOOSE-PASSIVE (IRC: VPIP~48%, AF~0.65) — so baselining "deviation"
against it (a) UNDER-exploits the typical station (they look ~normal vs an already-loose population) and
(b) is the wrong safety anchor (a GTO opponent looks like a deviator vs loose-passive). Measuring
deviation against a TAG/Nash reference fixes both: a station's high PASSIVE mass shows up as a large
positive deviation (→ correctly value-bet harder), while a GTO opponent sits ≈ on the reference
(→ ~zero deviation → no phantom exploitation of equilibrium play).

This is SEPARATE from the population prior (`population_prior.py`), which keeps its two legitimate roles:
posterior SHRINKAGE of small-n counts (anti-overfit) and the cold-start exploit DIRECTION. Only the
deviation REFERENCE (`ContextRead.pop`) moves to near-Nash. Texture-independent (like the prior); the
individual's per-texture data supplies the texture nuance.

Bins: [FOLD, PASSIVE(check/call), AGG_SMALL(<~pot bet/raise), AGG_LARGE(>=~pot + all-in)].
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from pokerbot.opponent.context import NUM_BINS, all_contexts

# TAG / near-equilibrium bin mass by (street-phase, facing). Low PASSIVE is the defining contrast with
# the loose-passive population. Values are deliberately round near-equilibrium approximations (the
# reference only needs to anchor "is this opponent deviating from solid play"); the bounded-ε safety
# proof (opponent/exploitability.added_exploitability) is what certifies they stay safe.
_PRE = {
    "unopened":     [0.62, 0.03, 0.28, 0.07],  # RFI: fold-or-raise, almost no limp
    "facing_bet":   [0.62, 0.25, 0.09, 0.04],  # vs open: fold / flat / 3bet
    "facing_raise": [0.55, 0.30, 0.10, 0.05],  # vs 3bet
    "facing_3bet":  [0.50, 0.35, 0.10, 0.05],  # vs 4bet+
}
_POST = {
    "unopened":     [0.00, 0.40, 0.40, 0.20],  # nobody bet yet: check or (c)bet; can't fold
    "facing_bet":   [0.45, 0.42, 0.09, 0.04],  # defend: fold / call / raise
    "facing_raise": [0.50, 0.38, 0.08, 0.04],
    "facing_3bet":  [0.55, 0.35, 0.07, 0.03],
}


def _base_dist(street: str, facing: str) -> list[float]:
    table = _PRE if street == "preflop" else _POST
    return table.get(facing, table["facing_bet"])


@lru_cache(maxsize=1)
def _reference_table() -> dict[str, tuple[float, ...]]:
    out: dict[str, tuple[float, ...]] = {}
    for ck in all_contexts():
        street, _texture, facing = ck.split("|")
        d = np.asarray(_base_dist(street, facing), dtype=np.float64)
        out[ck] = tuple(d / d.sum())
    return out


def nash_reference(context_key: str) -> np.ndarray:
    """Normalized near-Nash bin distribution for a context (generator fallback for unseen keys)."""
    hit = _reference_table().get(context_key)
    if hit is not None:
        return np.asarray(hit, dtype=np.float64)
    parts = context_key.split("|")
    if len(parts) == 3:
        d = np.asarray(_base_dist(parts[0], parts[2]), dtype=np.float64)
        return d / d.sum()
    return np.ones(NUM_BINS, dtype=np.float64) / NUM_BINS
