from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

import emd_abstraction_sweep as sweep  # noqa: E402


def test_assignment_stability_is_label_invariant_for_permuted_centroids():
    test_x = np.asarray(
        [
            [0.95, 0.05],
            [0.90, 0.10],
            [0.10, 0.90],
            [0.05, 0.95],
        ],
        dtype=np.float64,
    )
    centroids = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    permuted = centroids[[1, 0]]

    stability = sweep.assignment_stability(test_x, [centroids, permuted])

    assert stability["metric"] == "adjusted_rand_index"
    assert stability["pairwise_agreement_mean"] == 1.0
    assert stability["pairwise_agreement_min"] == 1.0


def test_adjusted_rand_index_rejects_different_partitions():
    a = np.asarray([0, 0, 1, 1, 2, 2])
    b = np.asarray([0, 1, 0, 1, 0, 1])

    assert sweep.adjusted_rand_index(a, b) < 0.1


def test_functional_ev_gate_requires_real_task_passes():
    evidence = {
        "tasks": {
            "draw_continue": {"gain_bb": 0.0, "frac_gap_closed_by_emd": 0.06},
            "semi_bluff": {"gain_bb": -0.005, "frac_gap_closed_by_emd": None},
        },
        "river_fallback_sanity": {
            "river_peak_mass_min": 0.95,
            "river_hist_variance_mean": 0.01,
        },
    }

    assert sweep.functional_ev_gate(evidence) == (True, [])

    evidence["tasks"]["semi_bluff"]["gain_bb"] = -0.02
    ok, reasons = sweep.functional_ev_gate(evidence)

    assert ok is False
    assert "functional_semi_bluff_gain_regressed" in reasons


def test_assignment_stability_default_is_report_only():
    policy = sweep.assignment_stability_policy(
        stability_mean=0.62,
        min_assignment_stability=None,
    )

    assert policy["mode"] == "report_only"
    assert policy["hard_gate"] is False
    assert policy["pass"] is None
    assert policy["fail_reasons"] == []
    assert policy["threshold_source"] == "plan_requires_stability_but_no_numeric_cutoff"


def test_assignment_stability_configured_threshold_gates_deterministically():
    fail_policy = sweep.assignment_stability_policy(
        stability_mean=0.62,
        min_assignment_stability=0.70,
    )
    pass_policy = sweep.assignment_stability_policy(
        stability_mean=0.72,
        min_assignment_stability=0.70,
    )

    assert fail_policy["mode"] == "hard_gate"
    assert fail_policy["pass"] is False
    assert fail_policy["fail_reasons"] == ["assignment_stability_below_min"]
    assert pass_policy["pass"] is True
    assert pass_policy["fail_reasons"] == []
