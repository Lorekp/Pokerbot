"""Gauntlet tournament: exploitation adds real uplift vs leaky fields, ~none vs GTO.

Runs are fully deterministic (paired deals + per-slot RNG, fixed seed), so these assertions
are stable. The claim is relative and robust: the live-profiling agent gains materially more by
deviating against leaky opponents than against GTO opponents (where deviations are rare sampling
misreads that cost ~0 at the indifference margin).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

_HAS = (Path("data/aof/cfr_base/cfr_base_3p.npz").is_file()
        and Path("data/aof/cfr_base/hu_eq_matrix.npz").is_file())
pytestmark = pytest.mark.skipif(not _HAS, reason="AoF base / HU eq artifacts not built")


def test_exploitation_beats_leaks_more_than_gto() -> None:
    from scripts.gauntlet_tournament import OVERFOLDER, FIELDS, run_paired

    leaky = run_paired(3, 8.0, 400, [OVERFOLDER, OVERFOLDER, OVERFOLDER], seed=7)
    gto = run_paired(3, 8.0, 400, FIELDS["all_gto"], seed=7)

    assert leaky["exploit_uplift_bb100"] > 0.0            # exploitation adds EV vs leaks
    assert leaky["n_exploit_decisions"] > gto["n_exploit_decisions"]
    assert leaky["exploit_uplift_bb100"] > gto["exploit_uplift_bb100"]


def test_gto_field_no_meaningful_edge() -> None:
    from scripts.gauntlet_tournament import FIELDS, run_paired

    gto = run_paired(3, 8.0, 400, FIELDS["all_gto"], seed=7)
    # Against GTO the router barely deviates and gains/loses no real edge (noise band).
    assert abs(gto["exploit_uplift_bb100"]) < 2.0


def test_paired_determinism() -> None:
    from scripts.gauntlet_tournament import FIELDS, run_tournament

    a = run_tournament(3, 8.0, 200, FIELDS["all_leaky"], 7, db_on=True)
    b = run_tournament(3, 8.0, 200, FIELDS["all_leaky"], 7, db_on=True)
    assert a == b
