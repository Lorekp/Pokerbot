"""Exhaustive Gate A (slow) vs training (fast) encode parity on all held-out rows."""

from pokerbot.training.nlhe_spot_replay import check_gate_a_training_encode_parity_exhaustive


def test_gate_a_training_encode_parity_exhaustive():
    report = check_gate_a_training_encode_parity_exhaustive()
    assert report["rows_checked"] > 0, "held-out parquet must have rows"
    assert report["n_replay_errors"] == 0, report.get("replay_errors")
    assert report["n_mismatches"] == 0, report.get("mismatches")
    assert report["passed"]
