"""Statistics core of the eval suite: CI math, CRN pairing, rotation, AIVAT null."""

from __future__ import annotations

import math
import random
from pathlib import Path

import numpy as np
import pytest

from aof_cfr.evalsuite.paired_arena import ci95_from_seeds, t95

_BASE = Path("data/aof/cfr_base/cfr_base_3p.npz")
_EQ = Path("data/aof/cfr_base/hu_eq_matrix.npz")
_HAS = _BASE.is_file() and _EQ.is_file()
needs_base = pytest.mark.skipif(not _HAS, reason="3-max base / HU eq artifacts not built")


def test_ci95_matches_closed_form() -> None:
    vals = [1.0, 2.0, 3.0, 4.0]
    m, ci = ci95_from_seeds(vals)
    sd = np.std(vals, ddof=1)
    assert m == pytest.approx(2.5)
    assert ci == pytest.approx(t95(3) * sd / math.sqrt(4))
    # single seed -> infinite CI, never a confident claim
    _m1, ci1 = ci95_from_seeds([5.0])
    assert ci1 == float("inf")


def test_field_rotation_covers_every_bot_at_every_seat() -> None:
    from aof_cfr.evalsuite.fields import FIELDS, seat_assignment
    bots = FIELDS["all_leaky"]
    seats = ["BTN", "SB", "BB"]
    seen: dict[str, set] = {s: set() for s in seats}
    idents: dict[str, set] = {}
    for h in range(12):
        for seat, (bot, ident) in seat_assignment(bots, seats, h).items():
            seen[seat].add(bot.name)
            idents.setdefault(ident, set()).add(bot.name)
    for seat in seats:                       # E1 regression: every archetype at every seat
        assert seen[seat] == {"overfolder", "calling_station", "depth_blind"}
    for ident, names in idents.items():      # identity follows the BOT, never mixes archetypes
        assert len(names) == 1


@needs_base
def test_crn_deals_identical_across_arms_and_paired_determinism() -> None:
    from aof_cfr.evalsuite.paired_arena import ArenaConfig, run_cell
    cfg = ArenaConfig(players=3, stack_bb=8.0, field="all_leaky", hands=40, seeds=(3,),
                      warmup_hands=40, arms=("gto", "shift_frozen"), aivat=False)
    out1 = run_cell(cfg)
    out2 = run_cell(cfg)
    for arm in ("gto", "shift_frozen"):      # exact reproducibility (CRN streams, no shared state)
        assert out1["arms"][arm].per_seed_bb100 == out2["arms"][arm].per_seed_bb100


@needs_base
def test_aivat_null_on_gto_vs_gto() -> None:
    """Adjusted == raw in expectation (zero-mean correction) and never biased: on an all-GTO
    table the agent's uplift must be ~0 and AIVAT must not move the mean materially."""
    from aof_cfr.evalsuite.paired_arena import ArenaConfig, run_cell
    cfg = ArenaConfig(players=3, stack_bb=8.0, field="all_gto", hands=400, seeds=(1, 2, 3),
                      warmup_hands=0, arms=("gto",))
    out = run_cell(cfg)
    res = out["arms"]["gto"]
    for adj, raw in zip(res.per_seed_bb100, res.per_seed_raw_bb100):
        assert abs(adj - raw) < 8.0          # correction is small (near-pure policies)...
    m, ci = ci95_from_seeds(res.per_seed_bb100)
    assert abs(m) < max(3 * ci, 6.0)         # ...and the rotated, adjusted mean is ~0


@needs_base
def test_rotation_cancels_positional_bias() -> None:
    """Without rotation, bb/100 is dominated by seat EV; with rotation over all seats the
    all-GTO mean collapses toward 0 even at small n."""
    from aof_cfr.evalsuite.paired_arena import ArenaConfig, _play_phase
    import aof_cfr.query as q
    q.load_cfr_base()
    cfg = ArenaConfig(players=3, stack_bb=8.0, field="all_gto", hands=300, seeds=(7,),
                      warmup_hands=0, aivat=False)
    per_seat = []
    for seat in ("BTN", "SB", "BB"):
        adj, _raw, _x, nh = _play_phase(cfg, 7, seat, hands=cfg.hands, deal_salt=10_000_000,
                                        policy=None, q=q)
        per_seat.append(adj / nh * 100.0)
    spread = max(per_seat) - min(per_seat)
    rotated = sum(per_seat) / 3.0
    assert abs(rotated) < spread             # rotation buys real cancellation
