"""Out-of-distribution diagnostics for inference."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from aof_rl.env import ACTION_ALL_IN, ACTION_FOLD
from aof_rl.obs_v3 import ObsV3
from aof_rl.teacher_targets import teacher_probs


@dataclass(frozen=True)
class OodConfig:
    drift_threshold: float = 0.15
    uncertainty_threshold: float = 2.0
    teacher_disagreement_threshold: float = 0.5
    ensemble_logit_diff_threshold: float = 1.0
    num_bins: int = 16

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> OodConfig:
        if not data:
            return cls()
        return cls(
            drift_threshold=float(data.get("drift_threshold", 0.15)),
            uncertainty_threshold=float(data.get("uncertainty_threshold", 2.0)),
            teacher_disagreement_threshold=float(
                data.get("teacher_disagreement_threshold", 0.5)
            ),
            ensemble_logit_diff_threshold=float(
                data.get("ensemble_logit_diff_threshold", 1.0)
            ),
            num_bins=int(data.get("num_bins", 16)),
        )


@dataclass(frozen=True)
class OodResult:
    drift_score: float
    teacher_disagreement: float
    uncertainty_flag: bool
    ensemble_disagreement: float | None
    ood_alert: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "drift_score": self.drift_score,
            "teacher_disagreement": self.teacher_disagreement,
            "uncertainty_flag": self.uncertainty_flag,
            "ensemble_disagreement": self.ensemble_disagreement,
            "ood_alert": self.ood_alert,
        }


def flatten_obs_features(obs: ObsV3) -> np.ndarray:
    """Flatten masked observation blocks into a 1-D feature vector."""
    parts: list[np.ndarray] = []
    hero = obs.hero_hand.detach().cpu().numpy().ravel()
    glob = obs.global_feats.detach().cpu().numpy().ravel()
    parts.extend([hero, glob])
    seat_t = obs.seat_tokens.detach().cpu().numpy()
    seat_m = obs.seat_mask.detach().cpu().numpy()
    for i in range(seat_t.shape[0]):
        if seat_m[i] > 0.5:
            parts.append(seat_t[i].ravel())
    hist_t = obs.history_tokens.detach().cpu().numpy()
    hist_m = obs.history_mask.detach().cpu().numpy()
    for i in range(hist_t.shape[0]):
        if hist_m[i] > 0.5:
            parts.append(hist_t[i].ravel())
    return np.concatenate(parts).astype(np.float32)


def load_reference_histogram(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _histogram_drift(
    features: np.ndarray,
    reference: dict[str, Any],
    *,
    num_bins: int,
) -> float:
    ref_bins = np.asarray(reference["bins"], dtype=np.float64)
    ref_counts = np.asarray(reference["counts"], dtype=np.float64)
    if ref_bins.ndim != 2:
        raise ValueError("reference bins must be 2-D")
    n_feat = min(features.shape[0], ref_bins.shape[0])
    if n_feat == 0:
        return 0.0
    scores: list[float] = []
    for i in range(n_feat):
        lo, hi = float(ref_bins[i, 0]), float(ref_bins[i, 1])
        if hi <= lo:
            continue
        hist, _ = np.histogram(
            np.array([features[i]], dtype=np.float64),
            bins=num_bins,
            range=(lo, hi),
        )
        ref = ref_counts[i]
        ref_sum = ref.sum()
        if ref_sum <= 0:
            continue
        ref_p = ref / ref_sum
        obs_p = hist.astype(np.float64) / max(hist.sum(), 1.0)
        scores.append(float(np.abs(obs_p - ref_p).sum()))
    if not scores:
        return 0.0
    return float(np.mean(scores))


class OodMonitor:
    def __init__(
        self,
        config: OodConfig,
        reference_histogram: dict[str, Any] | None = None,
    ) -> None:
        self.config = config
        self.reference = reference_histogram

    @classmethod
    def from_paths(
        cls,
        config: OodConfig,
        reference_path: Path | None,
    ) -> OodMonitor:
        ref = None
        if reference_path is not None and reference_path.is_file():
            ref = load_reference_histogram(reference_path)
        return cls(config, ref)

    def score(
        self,
        obs: ObsV3,
        *,
        policy_action: int,
        policy_probs: np.ndarray,
        uncertainty_bb: float,
        teacher_action: int | None = None,
        teacher=None,
        ensemble_action: int | None = None,
        ensemble_logit_diff: float | None = None,
    ) -> OodResult:
        drift = 0.0
        if self.reference is not None:
            feats = flatten_obs_features(obs)
            drift = _histogram_drift(
                feats,
                self.reference,
                num_bins=self.config.num_bins,
            )

        teacher_disagreement = 0.0
        if teacher_action is not None and teacher_action != policy_action:
            teacher_disagreement = 1.0
        elif teacher is not None:
            t_probs = teacher_probs(teacher)
            kl = 0.0
            for i in range(2):
                p = max(float(policy_probs[i]), 1e-9)
                q = max(float(t_probs[i]), 1e-9)
                kl += p * np.log(p / q)
            teacher_disagreement = float(min(1.0, abs(kl)))

        uncertainty_flag = uncertainty_bb > self.config.uncertainty_threshold

        ensemble_disagreement: float | None = None
        if ensemble_action is not None and ensemble_action != policy_action:
            ensemble_disagreement = 1.0
        elif ensemble_logit_diff is not None:
            if ensemble_logit_diff > self.config.ensemble_logit_diff_threshold:
                ensemble_disagreement = float(ensemble_logit_diff)

        ood_alert = (
            drift > self.config.drift_threshold
            or teacher_disagreement >= self.config.teacher_disagreement_threshold
            or uncertainty_flag
            or (
                ensemble_disagreement is not None
                and ensemble_disagreement >= 1.0
            )
        )
        return OodResult(
            drift_score=drift,
            teacher_disagreement=teacher_disagreement,
            uncertainty_flag=uncertainty_flag,
            ensemble_disagreement=ensemble_disagreement,
            ood_alert=ood_alert,
        )


def build_reference_histogram(
    obs_list: list[ObsV3],
    *,
    num_bins: int = 16,
) -> dict[str, Any]:
    """Build reference histogram artifact from training/validation observations."""
    if not obs_list:
        raise ValueError("obs_list must be non-empty")
    mats = np.stack([flatten_obs_features(o) for o in obs_list])
    n_feat = mats.shape[1]
    bins = np.zeros((n_feat, 2), dtype=np.float64)
    counts = np.zeros((n_feat, num_bins), dtype=np.float64)
    for i in range(n_feat):
        col = mats[:, i]
        lo, hi = float(col.min()), float(col.max())
        if hi <= lo:
            hi = lo + 1e-6
        bins[i] = (lo, hi)
        n_unique = len(np.unique(col))
        use_bins = max(1, min(num_bins, n_unique))
        counts[i], _ = np.histogram(col, bins=use_bins, range=(lo, hi))
        if counts[i].shape[0] < num_bins:
            pad = np.zeros(num_bins - counts[i].shape[0], dtype=np.float64)
            counts[i] = np.concatenate([counts[i], pad])
    return {
        "version": 1,
        "num_bins": num_bins,
        "bins": bins.tolist(),
        "counts": counts.tolist(),
    }
