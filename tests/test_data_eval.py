"""Tests for PokerBench data and evaluation."""

from pokerbot.data.pokerbench import load_pokerbench
from pokerbot.engine.actions import action_accuracy, exact_match
from pokerbot.eval.pokerbench_eval import evaluate_predictions


def test_synthetic_preflop_loads():
    samples = load_pokerbench("preflop", "train", limit=2)
    assert len(samples) >= 1
    assert samples[0].output


def test_action_accuracy_metric():
    assert action_accuracy("bet 5.0", "raise 5.0")
    assert not action_accuracy("fold", "call")
    assert exact_match("check", "check")
    assert not exact_match("bet 5.0", "bet 6.0")


def test_evaluate_predictions():
    samples = load_pokerbench("preflop", "test", limit=2)

    def oracle(instr: str) -> str:
        return samples[0].output

    m = evaluate_predictions(samples, oracle)
    assert m.total == len(samples)
    assert m.exact_match >= 0.0
