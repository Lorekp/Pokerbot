"""Orchestrator for AoF RL policy evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import torch

from aof_equity import warmup
from aof_rl.eval.deal_stream import DealStream
from aof_rl.eval.eval_schema import EvalReport, PairedComparison
from aof_rl.eval.export import write_json_report, write_markdown_report
from aof_rl.eval.hu_exploitability import HuBrMode, compute_hu_exploitability
from aof_rl.eval.metrics import MetricsAccumulator
from aof_rl.eval.multiway_deviation import compute_multiway_deviation
from aof_rl.eval.policy_adapter import EvalPolicy, load_eval_policy
from aof_rl.eval.paired_eval import paired_hand_delta
from aof_rl.eval.rollouts import play_deal_symmetric
from aof_rl.teacher import NashTeacherCache

EvalProfile = Literal["fast", "default"]

_PROFILES: dict[str, dict[str, int | bool]] = {
    "fast": {"teacher_samples": 200, "payoff_samples": 400, "use_fast_teacher": True},
    "default": {
        "teacher_samples": 2000,
        "payoff_samples": 4000,
        "use_fast_teacher": False,
    },
}


@dataclass
class EvalConfig:
    checkpoint: str | None = None
    baseline_checkpoint: str | None = None
    eval_id: str = "aof_eval"
    hands: int = 2000
    seeds: list[int] = field(default_factory=lambda: [1, 2, 3])
    table_sizes: tuple[int, ...] = (2, 3, 4)
    stack_bb: float = 8.0
    profile: EvalProfile = "fast"
    teacher_samples: int | None = None
    payoff_samples: int | None = None
    teacher_version: int = 2
    obs_version: int | None = None
    device: str = "cpu"
    deterministic: bool = True
    paired: bool = False
    compute_hu_exploitability: bool = True
    hu_br_mode: HuBrMode = "bucketed_169"
    hu_equity_samples: int = 400
    hu_max_pairs: int | None = None
    compute_multiway: bool = True
    policy_mode: str = "checkpoint"
    out_json: str | None = None
    out_md: str | None = None


class AofEvalHarness:
    def __init__(self, cfg: EvalConfig) -> None:
        self.cfg = cfg

    def _resolve_samples(self) -> tuple[int, int, bool]:
        prof = _PROFILES[self.cfg.profile]
        ts = (
            self.cfg.teacher_samples
            if self.cfg.teacher_samples is not None
            else int(prof["teacher_samples"])
        )
        ps = (
            self.cfg.payoff_samples
            if self.cfg.payoff_samples is not None
            else int(prof["payoff_samples"])
        )
        fast = bool(prof["use_fast_teacher"])
        return ts, ps, fast

    def _set_deterministic(self) -> None:
        if not self.cfg.deterministic:
            return
        torch.manual_seed(0)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(0)

    def run(self) -> EvalReport:
        warmup()
        self._set_deterministic()
        device = torch.device(self.cfg.device)
        ts, ps, use_fast = self._resolve_samples()

        policy, meta = load_eval_policy(
            self.cfg.checkpoint,
            device,
            mode=self.cfg.policy_mode,
            obs_version=self.cfg.obs_version,
            teacher_version=self.cfg.teacher_version,
            deterministic=self.cfg.deterministic,
        )
        cache = NashTeacherCache(
            teacher_version=self.cfg.teacher_version,
            use_fast=use_fast,
        )

        baseline: EvalPolicy | None = None
        if self.cfg.baseline_checkpoint or self.cfg.paired:
            if self.cfg.policy_mode in ("oracle", "scripted"):
                baseline, _ = load_eval_policy(
                    None,
                    device,
                    mode=self.cfg.policy_mode,
                    deterministic=self.cfg.deterministic,
                )
            else:
                bpath = self.cfg.baseline_checkpoint or self.cfg.checkpoint
                baseline, _ = load_eval_policy(
                    bpath,
                    device,
                    mode="checkpoint",
                    obs_version=self.cfg.obs_version,
                    teacher_version=self.cfg.teacher_version,
                    deterministic=self.cfg.deterministic,
                )

        acc = MetricsAccumulator()
        base_seed = self.cfg.seeds[0] if self.cfg.seeds else 42

        for eval_seed in self.cfg.seeds:
            stream = DealStream(
                hands=self.cfg.hands,
                base_seed=base_seed,
                table_sizes=self.cfg.table_sizes,
                stack_bb=self.cfg.stack_bb,
            )
            seed_payoffs: list[float] = []
            for deal in stream.iter_for_seed(eval_seed):
                outcome, records = play_deal_symmetric(
                    deal,
                    policy,
                    cache,
                    teacher_samples=ts,
                    payoff_samples=ps,
                    eval_seed=eval_seed,
                    stack_bb=self.cfg.stack_bb,
                )
                acc.add_hand(outcome)
                acc.add_decisions(records)
                seed_payoffs.append(
                    sum(outcome.payoffs.values()) / max(len(outcome.payoffs), 1)
                )

                if baseline is not None and (
                    self.cfg.paired or self.cfg.baseline_checkpoint
                ):
                    out_b, _ = play_deal_symmetric(
                        deal,
                        baseline,
                        cache,
                        teacher_samples=ts,
                        payoff_samples=ps,
                        eval_seed=eval_seed,
                        stack_bb=self.cfg.stack_bb,
                    )
                    acc.add_paired_delta(paired_hand_delta(outcome, out_b))

            if seed_payoffs:
                acc.add_seed_mean(sum(seed_payoffs) / len(seed_payoffs))

        agg, by_seat, by_table, cal_buckets = acc.aggregate()

        hu_result = None
        if self.cfg.compute_hu_exploitability and 2 in self.cfg.table_sizes:
            hu_result = compute_hu_exploitability(
                policy,
                cache,
                mode=self.cfg.hu_br_mode,
                stack_bb=self.cfg.stack_bb,
                equity_samples=self.cfg.hu_equity_samples,
                seed=base_seed,
                teacher_samples=ts,
                max_pairs=self.cfg.hu_max_pairs,
            )

        mw_dev: dict = {}
        if self.cfg.compute_multiway:
            mw_sizes = tuple(t for t in self.cfg.table_sizes if t in (3, 4))
            if mw_sizes:
                stream_mw = DealStream(
                    hands=min(self.cfg.hands, 500),
                    base_seed=base_seed,
                    table_sizes=mw_sizes,
                    stack_bb=self.cfg.stack_bb,
                )
                mw_dev = compute_multiway_deviation(
                    policy,
                    stream_mw,
                    cache,
                    table_sizes=mw_sizes,
                    teacher_samples=ts,
                    payoff_samples=ps,
                    eval_seed=base_seed,
                )

        paired_block = None
        if acc.paired_deltas and baseline is not None:
            mean_d, ci_d, n = acc.paired_stats()
            paired_block = PairedComparison(
                policy_a=policy.name,
                policy_b=baseline.name,
                hands=n,
                mean_delta_bb_per_hand=mean_d,
                delta_bb_per_hand_ci95=ci_d,
                mean_delta_bb_per_100=mean_d * 100.0,
            )

        policy_info = {
            "checkpoint": self.cfg.checkpoint,
            "name": policy.name,
            "obs_version": getattr(policy, "obs_version", meta.get("obs_version", 3)),
            "teacher_version": self.cfg.teacher_version,
            "model_id": meta.get("experiment", policy.name),
        }

        report = EvalReport(
            eval_id=self.cfg.eval_id,
            policy=policy_info,
            config={
                "hands": self.cfg.hands,
                "seeds": self.cfg.seeds,
                "table_sizes": list(self.cfg.table_sizes),
                "stack_bb": self.cfg.stack_bb,
                "profile": self.cfg.profile,
                "teacher_samples": ts,
                "payoff_samples": ps,
                "deterministic": self.cfg.deterministic,
                "paired": self.cfg.paired or bool(self.cfg.baseline_checkpoint),
                "hu_br_mode": self.cfg.hu_br_mode,
            },
            aggregate=agg,
            by_seat=by_seat,
            by_table_size=by_table,
            calibration={"thin_edge_buckets": cal_buckets},
            hu_exploitability=hu_result,
            multiway_deviation=mw_dev,
            paired_comparison=paired_block,
        )

        if self.cfg.out_json:
            write_json_report(self.cfg.out_json, report)
        if self.cfg.out_md:
            write_markdown_report(self.cfg.out_md, report)

        return report
