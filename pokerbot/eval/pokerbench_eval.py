"""PokerBench EM/AA evaluation metrics."""

from __future__ import annotations

from dataclasses import dataclass

from pokerbot.data.pokerbench import PokerBenchSample, load_pokerbench
from pokerbot.engine.actions import action_accuracy, exact_match, parse_pokerbench_output


@dataclass
class EvalMetrics:
    exact_match: float
    action_accuracy: float
    total: int
    stage: str

    def __str__(self) -> str:
        return (
            f"{self.stage}: EM={self.exact_match:.2%} AA={self.action_accuracy:.2%} (n={self.total})"
        )


def evaluate_predictions(
    samples: list[PokerBenchSample],
    predict_fn,
) -> EvalMetrics:
    """predict_fn(instruction) -> action string."""
    em_hits = 0
    aa_hits = 0
    for sample in samples:
        pred = predict_fn(sample.instruction).strip().lower()
        tgt = sample.output.strip().lower()
        if exact_match(pred, tgt):
            em_hits += 1
        if action_accuracy(pred, tgt):
            aa_hits += 1
    n = max(len(samples), 1)
    stage = samples[0].stage if samples else "unknown"
    return EvalMetrics(em_hits / n, aa_hits / n, len(samples), stage)


def evaluate_model_on_pokerbench(
    predict_fn,
    preflop_limit: int | None = None,
    postflop_limit: int | None = None,
) -> dict[str, EvalMetrics]:
    results = {}
    for stage, limit in [("preflop", preflop_limit), ("postflop", postflop_limit)]:
        samples = load_pokerbench(stage, "test", limit=limit)
        results[stage] = evaluate_predictions(samples, predict_fn)
    all_samples = (
        load_pokerbench("preflop", "test", limit=preflop_limit)
        + load_pokerbench("postflop", "test", limit=postflop_limit)
    )
    results["overall"] = evaluate_predictions(all_samples, predict_fn)
    return results


def label_to_action_index(label: str) -> int:
    from pokerbot.engine.actions import action_to_index

    return action_to_index(parse_pokerbench_output(label))
