"""Fast CI subset of the AoF eval suite — small n, fixed seeds, loose gates (minutes).

The full certification sweep is ``python scripts/run_aof_eval_suite.py --full`` (hours, nightly).
This subset pins uplift > 0 vs an overfolder at small n, the
shipping adaptive arm actually firing exploit decisions, and the report artifact shape.
"""

from __future__ import annotations

from pathlib import Path

import pytest

_BASE = Path("data/aof/cfr_base/cfr_base_3p.npz")
_EQ = Path("data/aof/cfr_base/hu_eq_matrix.npz")
_HAS = _BASE.is_file() and _EQ.is_file()
needs_base = pytest.mark.skipif(not _HAS, reason="3-max base / HU eq artifacts not built")


@needs_base
def test_uplift_positive_vs_overfolder_small_n() -> None:
    from aof_cfr.evalsuite.paired_arena import ArenaConfig, run_cell
    cfg = ArenaConfig(players=3, stack_bb=8.0, field="overfolder_only", hands=300,
                      seeds=(1, 2, 3), warmup_hands=400,
                      arms=("gto", "shift_frozen", "adaptive_online"))
    out = run_cell(cfg)
    assert out["uplift"]["shift_frozen"]["mean"] > 0.0
    assert out["uplift"]["adaptive_online"]["mean"] > 0.0
    # the SHIPPING arm must actually fire exploit decisions (E3: it had never been evaluated)
    assert out["arms"]["adaptive_online"].exploit_decisions > 0


@needs_base
def test_report_artifact_shape(tmp_path) -> None:
    import json

    from aof_cfr.evalsuite.result import AoFEvalCell, write_report
    cells = [AoFEvalCell(metric="winrate", players=3, depth=8.0, field_name="all_leaky",
                         arm="gto", hands=100, seeds=2, mean_bb100=10.0, se_bb100=1.0,
                         ci95_bb100=2.0)]
    p = write_report(cells, tmp_path, seeds=(1, 2), tag="testtag")
    data = json.loads(p.read_text(encoding="utf-8"))
    h = data["header"]
    for k in ("suite_version", "git_sha", "cfr_base_manifest_hash", "infoset_key_spec_hash",
              "game_model_version", "flat_fee_bb_per_hand", "seeds", "timestamp"):
        assert k in h
    assert p.parent.name == "testtag"


def test_net_bb100_both_fee_branches(monkeypatch) -> None:
    """net_bb100 must follow the ACTUAL fee config, not a hard-coded branch. Fee-in-model (the
    shipped 0.12BB) -> net == gross (settlements already charge it); legacy no-fee artifact ->
    net = gross - 12 bb/100 (the deliberately conservative breakeven floor). Pins both, since the
    shipped path (net==gross) previously had no assertion and a flipped branch passed silently."""
    import aof_cfr.pushfold_payoffs as pf
    from aof_cfr.evalsuite.result import FLAT_FEE_BB100, AoFEvalCell
    cell = AoFEvalCell(metric="winrate", players=3, depth=8.0, field_name="all_leaky", arm="gto",
                       hands=100, seeds=2, mean_bb100=10.0, se_bb100=1.0, ci95_bb100=2.0)
    monkeypatch.setattr(pf, "flat_fee_default", lambda: 0.12)
    assert cell.net_bb100 == pytest.approx(10.0)                       # fee in model -> net == gross
    monkeypatch.setattr(pf, "flat_fee_default", lambda: 0.0)
    assert cell.net_bb100 == pytest.approx(10.0 - FLAT_FEE_BB100)      # legacy floor


@needs_base
def test_coldstart_zero_budget_is_base() -> None:
    """With 0 observed hands and population_confidence=0 the shipping arm must play base
    (uplift ~ 0, no exploit decisions) — the cold-start safety invariant."""
    from aof_cfr.evalsuite.paired_arena import ArenaConfig, run_cell
    cfg = ArenaConfig(players=3, stack_bb=8.0, field="all_leaky", hands=200, seeds=(5,),
                      warmup_hands=0, arms=("gto", "adaptive_online"),
                      population_confidence=0.0, refresh_every=50)
    out = run_cell(cfg)
    # adaptive arm LEARNS online, so late hands may exploit; but the first refresh window is base
    assert out["arms"]["adaptive_online"].hands == 600
