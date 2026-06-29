"""Tests for the unified tournament-eval suite driver (scripts/tournament/eval_tournament_suite.py).

The load-bearing test is ``test_smoke_fold_grid_well_formed``: a ``--smoke --policy fold`` run over a
small {NORMAL,SATELLITE} x {bubble,deep} grid must emit a WELL-FORMED unified JSON whose bubble/satellite
cells RECOVER the closed-form always-fold anchor floor and whose certificates pass. That single assertion
covers the whole orchestration: grid iteration, the C0-C5 certificates, per-cell net-free scoring, the
reporting columns (verdict / chip-vs-$ divergence / uplift_zero_frac / per-seat breakdown /
fixed-iteration solves), and the on-disk JSON.

``test_c5_serve_roundtrip_byte_identity`` is the driver-level mirror of
``tests/test_seven_neural_serve.py::test_seven_remap_reproduces_native_encoding`` (the #1 serve-skew
guard): it asserts the C5 certificate reproduces the native 7-seat ``encode_info_state`` byte-for-byte
when jax is available (and is SKIPPED-not-FAILED when jax is absent — a net-free env).

The driver module is loaded DIRECTLY by file path so the test does not depend on it being importable as a
package module; the driver itself sets ``JAX_PLATFORMS=cpu`` + ``POKERBOT_INFO_SUITS=1`` on import.
"""

from __future__ import annotations

import importlib.util
import json
import pathlib

import pytest

_ROOT = pathlib.Path(__file__).resolve().parents[1]


def _load_driver():
    spec = importlib.util.spec_from_file_location(
        "eval_tournament_suite_under_test",
        _ROOT / "scripts" / "tournament" / "eval_tournament_suite.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_drv = _load_driver()


# The foundation ``eval_score`` module eagerly imports ``rollout_leaf -> sim -> aof_equity_grids``, whose
# numba dependency is absent from the lean jax test venv (jax+torch+pydantic+phevaluator only). The
# eval_score-dependent assertions (C2, the per-cell calibration, the smoke/CLI/resume drivers) are SKIPPED
# in that env and run fully on the production stack (which has numba); the pure-primitive certs
# (C0/C1/C3/C4/C5) + grid/config parsing run EVERYWHERE. The driver itself does all heavy imports lazily,
# so its module load never trips this.
try:
    import pokerbot.tournament.eval_score as _es  # noqa: F401
    _HAS_EVAL_SCORE = True
    _EVAL_SCORE_ERR = ""
except Exception as _exc:  # noqa: BLE001
    _HAS_EVAL_SCORE = False
    _EVAL_SCORE_ERR = repr(_exc)

_needs_eval_score = pytest.mark.skipif(
    not _HAS_EVAL_SCORE,
    reason=f"foundation eval_score not importable in this env ({_EVAL_SCORE_ERR})")


# --------------------------------------------------------------------------- grid / config parsing
def test_grid_constants_are_the_full_surface():
    assert _drv.FORMATS == ["NORMAL", "SNG", "SATELLITE", "PKO"]
    assert _drv.REGIME_ORDER == ["deep", "mid", "shallow", "bubble", "ft", "hu"]
    # every regime carries a depth + an icm_spot flag (bubble/ft/hu are the ICM-pressure spots).
    for k in _drv.REGIME_ORDER:
        r = _drv.REGIMES[k]
        assert "depth_bb" in r and "icm_spot" in r
    assert _drv.REGIMES["bubble"]["icm_spot"] is True
    assert _drv.REGIMES["deep"]["icm_spot"] is False


def test_resolve_list_all_and_explicit():
    assert _drv._resolve_list("all", allowed=_drv.FORMATS, name="format") == _drv.FORMATS
    assert _drv._resolve_list("NORMAL,PKO", allowed=_drv.FORMATS, name="format") == ["NORMAL", "PKO"]
    with pytest.raises(SystemExit):
        _drv._resolve_list("BOGUS", allowed=_drv.FORMATS, name="format")


# --------------------------------------------------------------------------- certificates C0-C4 (net-free)
def test_certificates_c1_c3_pass_everywhere():
    """C1 (icm conservation) + C3 (anchor closed form) depend ONLY on the numba-free icm/anchor modules,
    so they pass in ANY env — including the lean jax venv without numba. The eval_score-backed certs
    (C0 foundation import, C2 scoring, C4 the CV core) are exercised under the @_needs_eval_score guard."""
    for fn, name in [(_drv._cert_c1, "C1_icm_conservation"), (_drv._cert_c3, "C3_anchor_closed_form")]:
        cert = fn()
        assert cert["name"] == name
        assert cert["pass"] is True, f"{name} failed: {cert}"


@_needs_eval_score
def test_certificates_c0_c2_c4_pass():
    """C0 (foundation import — pulls the full tournament layer), C2 (chip->$ conservation + PKO reduces
    to ICM), C4 (the variance-reduction core identities) all route through the foundation eval_score /
    full tournament package, so they need the production stack (numba present)."""
    for fn, name in [(_drv._cert_c0, "C0_imports"), (_drv._cert_c2, "C2_score_chip_to_dollar"),
                     (_drv._cert_c4, "C4_variance_reduction_core")]:
        cert = fn()
        assert cert["name"] == name
        assert cert["pass"] is True, f"{name} failed: {cert}"


def test_c3_recovers_documented_sevenmax_anchor():
    """The 7-max ante=0.125 always-fold anchor is the documented +203.571 bb/100 floor."""
    c3 = _drv._cert_c3()
    assert abs(c3["sevenmax_bb_per_100"] - 203.57142857142858) < 1e-6
    assert c3["monotone_in_n"] is True


# --------------------------------------------------------------------------- C5 serve-skew round-trip
def test_c5_serve_roundtrip_byte_identity():
    """C5 reproduces the native 7-seat encode_info_state byte-for-byte (the #1 serve-skew guard).

    Mirrors tests/test_seven_neural_serve.py::test_seven_remap_reproduces_native_encoding at the driver
    level. SKIPPED (not failed) when jax is unavailable: the certificate is informational in a net-free
    env (``skipped: True``)."""
    pytest.importorskip("jax")
    c5 = _drv._cert_c5(seed=7)
    if c5.get("skipped"):
        pytest.skip("jax/engine import failed inside C5 — net-free env, certificate informational")
    assert c5["pass"] is True, f"C5 serve round-trip failed: {c5}"
    assert c5["identity_map"] is True
    assert c5["fields_byte_identical"] is True
    assert c5["encode_info_state_byte_identical"] is True


# --------------------------------------------------------------------------- per-cell anchor calibration
@_needs_eval_score
def test_anchor_cell_recovers_floor_and_reports_columns():
    """A single net-free bubble cell recovers the closed-form anchor floor and carries every reporting
    column: per-seat breakdown, mean+/-1.96se, verdict, chip-vs-$ divergence, uplift_zero_frac CRN-health,
    and the fixed-iteration solve count."""
    from pokerbot.tournament.anchor import always_fold_exploitability

    cell = _drv._anchor_calibration_cell("NORMAL", "bubble", n_seats=6, decks=4, seed=0, policy="fold")
    n_seats = cell["n_seats"]
    expect = always_fold_exploitability(n_seats, ante_bb=cell["ante_bb"])
    assert cell["recovers_anchor_floor"] is True
    assert abs(cell["measured_mean_bb_per_100"] - expect) < 1e-6
    assert cell["verdict"] == "POSITIVE"           # the steal value is positive
    # CRN health: every net-free deck's uplift is exactly 0 -> perfect cancellation.
    assert cell["uplift_zero_frac"] == 1.0
    # fixed-iteration solve count = decks * seats (NOT wall-clock).
    assert cell["fixed_iteration_solves"] == 4 * n_seats
    # per-seat breakdown present for every seat.
    assert set(cell["per_seat_breakdown"].keys()) == {str(s) for s in range(n_seats)}
    # chip-vs-$ divergence column present and (at a bubble) the chips are worth < linear $ -> > 0.
    assert "chip_dollar_divergence" in cell
    assert cell["icm_pressure_spot"] is True


@_needs_eval_score
def test_bubble_divergence_exceeds_deep():
    """The chip-vs-$ divergence is LARGER at the bubble (ICM pressure) than deep (chip-EV ~ $-EV).

    Same NORMAL format/seats/depth held fixed where possible; the bubble cell is an ICM-pressure spot so
    a chip of stack is worth strictly less than its linear $-share, i.e. a bigger divergence column."""
    deep = _drv._anchor_calibration_cell("NORMAL", "deep", n_seats=6, decks=2, seed=0, policy="fold")
    bub = _drv._anchor_calibration_cell("NORMAL", "bubble", n_seats=6, decks=2, seed=0, policy="fold")
    assert bub["chip_dollar_divergence"] > deep["chip_dollar_divergence"]


# --------------------------------------------------------------------------- the smoke headline test
@_needs_eval_score
def test_smoke_fold_grid_well_formed(tmp_path):
    """--smoke --policy fold on a 2x2 grid emits a well-formed unified JSON whose bubble/satellite cells
    recover the anchor floor and whose certificates pass. THE load-bearing orchestration test."""
    from pokerbot.tournament.anchor import always_fold_exploitability

    out = tmp_path / "smoke_suite.json"
    report = _drv.run_suite(
        formats=["NORMAL", "SATELLITE"], regimes=["deep", "bubble"], policy="fold",
        decks=4, hands=20, spots=10, ckpt="checkpoints/sd_cfr_7max_neural/seven_v1.pkl",
        seed=0, smoke=True, out=str(out), quiet=True,
    )

    # well-formed: the file exists and reloads, schema + the four cells are present.
    assert out.exists()
    reloaded = json.loads(out.read_text(encoding="utf-8"))
    assert reloaded["schema"] == "tournament_eval_suite/v1"
    assert reloaded["grid"]["n_cells"] == 4
    assert set(reloaded["cells"].keys()) == {
        "NORMAL:deep", "NORMAL:bubble", "SATELLITE:deep", "SATELLITE:bubble"}

    # certificates: C0-C4 (and C5 when jax present, else skipped) all pass.
    assert report["summary"]["certificates_all_pass"] is True

    # bubble/satellite cells recover the closed-form anchor floor.
    for key in ("NORMAL:bubble", "SATELLITE:bubble"):
        cal = reloaded["cells"][key]["calibration"]
        expect = always_fold_exploitability(cal["n_seats"], ante_bb=cal["ante_bb"])
        assert cal["recovers_anchor_floor"] is True
        assert abs(cal["measured_mean_bb_per_100"] - expect) < 1e-6
        assert cal["verdict"] == "POSITIVE"

    # every cell carries the reporting columns + a pillar block (deferred when siblings aren't built).
    for cell in reloaded["cells"].values():
        cal = cell["calibration"]
        for col in ("measured_mean_bb_per_100", "se_bb_per_100", "ci95", "verdict",
                    "chip_dollar_divergence", "uplift_zero_frac", "per_seat_breakdown",
                    "fixed_iteration_solves"):
            assert col in cal, f"missing reporting column {col} in {cell['label']}"
        assert "pillars" in cell
        for pk in ("E1_E2_lbr_icm", "E3_E4_mtt_lifecycle", "E6_regime_battery"):
            assert pk in cell["pillars"]
            assert cell["pillars"][pk]["status"] in ("built", "deferred", "error")

    # smoke clamps N down: decks <= 8.
    assert reloaded["config"]["decks"] <= 8


# --------------------------------------------------------------------------- checkpoint resume
@_needs_eval_score
def test_checkpoint_resume_round_trips(tmp_path):
    """A crash-survivable per-cell JSONL checkpoint replays completed cells on resume; a header mismatch
    refuses to merge a foreign stream (the exploitation_battery header-guard idiom)."""
    ckpt = tmp_path / "cells.jsonl"
    out1 = tmp_path / "r1.json"
    _drv.run_suite(formats=["NORMAL"], regimes=["deep", "bubble"], policy="fold", decks=2, hands=10,
                   spots=5, ckpt="x.pkl", seed=0, smoke=True, out=str(out1),
                   checkpoint_path=str(ckpt), resume=False, quiet=True)
    assert ckpt.exists()
    # resume: the same config replays both cells from disk (no recompute) and re-emits the same report.
    out2 = tmp_path / "r2.json"
    rep2 = _drv.run_suite(formats=["NORMAL"], regimes=["deep", "bubble"], policy="fold", decks=2, hands=10,
                          spots=5, ckpt="x.pkl", seed=0, smoke=True, out=str(out2),
                          checkpoint_path=str(ckpt), resume=True, quiet=True)
    assert rep2["summary"]["cells_total"] == 2
    for key in ("NORMAL:deep", "NORMAL:bubble"):
        assert key in rep2["cells"]

    # header guard: resuming with a DIFFERENT seed must raise (stale/foreign stream).
    meta_bad = _drv._ckpt_meta(seed=999, formats=["NORMAL"], regimes=["deep", "bubble"],
                               policy="fold", decks=2)
    with pytest.raises(ValueError):
        _drv._load_cell_checkpoint(str(ckpt), meta=meta_bad)


@_needs_eval_score
def test_main_cli_smoke(tmp_path):
    """The CLI entry point runs a tiny net-free grid and returns 0 (certs pass) writing the report."""
    out = tmp_path / "cli.json"
    rc = _drv.main([
        "eval_tournament_suite.py", "--smoke", "--policy", "fold",
        "--format", "NORMAL", "--regime", "bubble", "--out", str(out), "--quiet",
    ])
    assert rc == 0
    assert out.exists()
    rep = json.loads(out.read_text(encoding="utf-8"))
    assert rep["grid"]["n_cells"] == 1
    assert "NORMAL:bubble" in rep["cells"]
