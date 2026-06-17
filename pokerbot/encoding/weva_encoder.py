"""Warm-up Expected Value-based Abstraction (WEVA) for preflop hole-card clustering."""

from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

if TYPE_CHECKING:
    import torch

from pokerbot.engine.actions import NUM_ACTIONS
from pokerbot.engine.cards import Card, Rank, Suit

logger = logging.getLogger(__name__)

WEVA_WARMUP_ITERATIONS = 10
WEVA_DEFAULT_K = 10_000
WEVA_DEPTH_GAMMA = 0.85
WEVA_NUM_DEPTH_NODES = 4
WEVA_MAX_NODES = 4
WEVA_PREFLOP_ACTIONS = 3
WEVA_DEFAULT_ROLLOUTS = 128
WEVA_PRODUCTION_NUM_ROOTS = 1326  # hole_only full combo set

# Lossless suit-isomorphic preflop abstraction (Pluribus-faithful). Preflop, suits are fully
# interchangeable (no board) so collapsing the 1326 hole combos to the 169 canonical rank-classes is
# EXACTLY lossless — unlike the K=10k WEVA EV-clustering, which merges strategically-distinct hands and
# was the certified +180 bb/100 preflop leak. When POKERBOT_PREFLOP_LOSSLESS=1, the preflop card bucket
# becomes ``position*169 + canonical_preflop_class`` (≤1014) and the preflop KEY uses the LOSSLESS betting
# history (preflop trees are shallow → bounded) instead of the bucketed betting state. Postflop is
# untouched. Read at import (single source of truth referenced via this module by the encode/serve paths,
# so tests monkeypatch ``weva_encoder._PREFLOP_LOSSLESS`` and clear the bucket-table lru_cache).
_PREFLOP_LOSSLESS = os.environ.get("POKERBOT_PREFLOP_LOSSLESS", "0").lower() not in (
    "",
    "0",
    "false",
    "no",
)
# OPT-IN second stage: also make the PREFLOP BETTING lossless (full preflop action history) instead of the
# bounded betting_state. Pluribus-exact, but the preflop history combinatorics explode the infoset count
# (~10^7-10^8 vs ~10^6 for cards-only) -> overflows the hogwild table AND is under-trainable in 200 iters.
# DEFAULT OFF: lossless CARDS + bounded betting fixes the proven card-merge leak and is SMALLER than the
# deployed 10k-WEVA blueprint (1014 < 10000 preflop card-contexts), so it fits the existing capacity.
_PREFLOP_LOSSLESS_BETTING = os.environ.get("POKERBOT_PREFLOP_LOSSLESS_BETTING", "0").lower() not in (
    "",
    "0",
    "false",
    "no",
)
PREFLOP_NUM_CLASSES = 169  # 13 pairs + 78 suited + 78 offsuit
PREFLOP_LOSSLESS_NUM = 6 * PREFLOP_NUM_CLASSES  # 1014: 169 classes x 6 position enums


def canonical_preflop_class(c0: int, c1: int) -> int:
    """Lossless suit-isomorphic preflop class in ``[0, 168]`` for two card indices (0..51).

    Card index = ``(rank-2)*4 + suit`` (see ``Card.index``), so ``rank_ordinal = idx // 4`` (0=deuce …
    12=ace) and ``suit = idx % 4``. Layout (169 = 13 + 78 + 78):
      * pocket pair  -> ``rank_ordinal``                          (0..12)
      * suited       -> ``13 + tri(hr, lr)``                      (13..90)
      * offsuit      -> ``13 + 78 + tri(hr, lr)``                 (91..168)
    where ``tri(hr, lr) = hr*(hr-1)//2 + lr`` uniquely indexes the C(13,2)=78 rank-pairs with
    ``0 <= lr < hr <= 12``. Suit identity is discarded (lossless preflop); position is added by the caller.
    """
    r0, s0 = c0 // 4, c0 % 4
    r1, s1 = c1 // 4, c1 % 4
    if r0 == r1:
        return r0
    hr, lr = (r0, r1) if r0 > r1 else (r1, r0)
    tri = hr * (hr - 1) // 2 + lr
    return (13 + tri) if s0 == s1 else (13 + 78 + tri)


FeatureMode = Literal["hole_only", "hole_plus_context"]


@dataclass(frozen=True, slots=True)
class WEVAContext:
    """Public context appended to hole-card features for clustering."""

    hero_position: int = 0
    num_players: int = 6
    action_depth_bin: int = 0

    def context_vector(self) -> np.ndarray:
        pos = np.zeros(6, dtype=np.float32)
        pos[self.hero_position % 6] = 1.0
        return np.concatenate(
            [
                pos,
                np.array(
                    [self.num_players / 6.0, self.action_depth_bin / 3.0],
                    dtype=np.float32,
                ),
            ]
        )


@dataclass(frozen=True, slots=True)
class RootInfoSet:
    """Root information set: hole cards plus optional public context."""

    hole: tuple[Card, Card]
    context: WEVAContext = field(default_factory=WEVAContext)

    @property
    def hole_key(self) -> tuple[int, int]:
        i0, i1 = sorted(c.index for c in self.hole)
        return i0, i1

    def feature_suffix(self) -> np.ndarray:
        return self.context.context_vector()


def enumerate_hole_combos() -> list[tuple[Card, Card]]:
    """All 1326 unordered hole-card combinations."""
    deck = [Card(Rank(r), Suit(s)) for r in range(2, 15) for s in range(4)]
    return list(combinations(deck, 2))


def expand_root_info_sets(mode: FeatureMode) -> list[RootInfoSet]:
    combos = enumerate_hole_combos()
    if mode == "hole_only":
        return [RootInfoSet(hole=c) for c in combos]

    roots: list[RootInfoSet] = []
    for hole in combos:
        for pos in range(6):
            for n_players in range(2, 7):
                for depth_bin in range(4):
                    roots.append(
                        RootInfoSet(
                            hole=hole,
                            context=WEVAContext(
                                hero_position=pos,
                                num_players=n_players,
                                action_depth_bin=depth_bin,
                            ),
                        )
                    )
    return roots


def _preflop_strength(hole: tuple[Card, Card]) -> float:
    """Fast preflop strength proxy in [0, 1] for CFR terminal payoffs."""
    r0, r1 = int(hole[0].rank), int(hole[1].rank)
    hi, lo = max(r0, r1), min(r0, r1)
    suited = int(hole[0].suit) == int(hole[1].suit)
    pair = r0 == r1
    if pair:
        return 0.50 + (hi - 2) / 26.0
    base = (hi + lo) / 28.0
    suited_bonus = 0.04 if suited else 0.0
    connector_bonus = 0.02 if (hi - lo) <= 2 else 0.0
    return float(np.clip(base + suited_bonus + connector_bonus, 0.05, 0.95))


class PreflopRootGame:
    """Lightweight tabular preflop game for WEVA warm-up CFR."""

    def __init__(
        self,
        num_roots: int,
        num_depths: int = WEVA_NUM_DEPTH_NODES,
        num_actions: int = WEVA_PREFLOP_ACTIONS,
        device: torch.device | None = None,
    ) -> None:
        import torch

        self.num_roots = num_roots
        self.num_depths = num_depths
        self.num_actions = num_actions
        self.device = device or torch.device("cpu")

    def hand_utilities(self, strengths: torch.Tensor) -> torch.Tensor:
        """Terminal chip EV proxy per root (shape: num_roots)."""
        return (strengths - 0.5) * 4.0


class TabularCFR:
    """Vectorized tabular CFR over root info sets (PyTorch tensors)."""

    def __init__(
        self,
        game: PreflopRootGame,
        strengths: torch.Tensor,
        warmup_iterations: int = WEVA_WARMUP_ITERATIONS,
    ) -> None:
        import torch

        self.game = game
        self.strengths = strengths.to(game.device)
        self.w = warmup_iterations
        n, d, a = game.num_roots, game.num_depths, game.num_actions
        self.regrets = torch.zeros(n, d, a, device=game.device)
        self.strategy_sum = torch.zeros(n, d, a, device=game.device)

    def _regret_matching(self, regrets: torch.Tensor) -> torch.Tensor:
        import torch

        pos = regrets.clamp(min=0.0)
        totals = pos.sum(dim=-1, keepdim=True)
        uniform = torch.full_like(pos, 1.0 / regrets.shape[-1])
        return torch.where(totals > 0, pos / totals.clamp(min=1e-8), uniform)

    def _action_evs(self, depth: int) -> torch.Tensor:
        import torch

        n = self.game.num_roots
        fold_ev = torch.zeros(n, device=self.game.device)
        call_ev = self.strengths * (1.0 + 0.15 * depth)
        raise_ev = self.strengths * (1.35 + 0.10 * depth) - 0.25 * (1.0 - self.strengths)
        return torch.stack([fold_ev, call_ev, raise_ev], dim=-1)

    def run(self) -> torch.Tensor:
        """Run W warm-up iterations; return average strategy (N, D, A)."""
        for _ in range(self.w):
            strat = self._regret_matching(self.regrets)
            self.strategy_sum += strat
            for d in range(self.game.num_depths):
                action_evs = self._action_evs(d)
                node_val = (strat[:, d, :] * action_evs).sum(dim=-1, keepdim=True)
                self.regrets[:, d, :] += action_evs - node_val
        return self.strategy_sum / float(self.w)

    @property
    def average_strategy(self) -> torch.Tensor:
        return self.strategy_sum / max(self.w, 1)


class EVExtractor:
    """Extract depth-weighted EV features from warm-up average strategy."""

    def __init__(
        self,
        gamma: float = WEVA_DEPTH_GAMMA,
        num_depths: int = WEVA_NUM_DEPTH_NODES,
    ) -> None:
        self.gamma = gamma
        self.num_depths = num_depths
        self.depth_weights = np.array([gamma**d for d in range(num_depths)], dtype=np.float32)

    def extract(
        self,
        avg_strategy: torch.Tensor,
        strengths: torch.Tensor,
        roots: list[RootInfoSet],
        feature_mode: FeatureMode,
    ) -> np.ndarray:
        """Build ev-nd feature matrix (num_roots, feature_dim)."""
        import torch

        device = avg_strategy.device
        n = avg_strategy.shape[0]
        fold_ev = torch.zeros(n, device=device)
        ev_by_depth: list[torch.Tensor] = []
        for d in range(self.num_depths):
            call_ev = strengths * (1.0 + 0.15 * d)
            raise_ev = strengths * (1.35 + 0.10 * d) - 0.25 * (1.0 - strengths)
            action_evs = torch.stack([fold_ev, call_ev, raise_ev], dim=-1)
            ev_d = (avg_strategy[:, d, :] * action_evs).sum(dim=-1)
            ev_by_depth.append(ev_d)
        ev_stack = torch.stack(ev_by_depth, dim=1).cpu().numpy()
        weighted = ev_stack * self.depth_weights[np.newaxis, :]

        if feature_mode == "hole_only":
            return weighted.astype(np.float32)

        ctx_matrix = np.stack([r.feature_suffix() for r in roots], axis=0)
        return np.concatenate([weighted, ctx_matrix], axis=1).astype(np.float32)


class WEVAIndex:
    """FAISS k-means++ index with sklearn fallback."""

    def __init__(self, k: int = WEVA_DEFAULT_K) -> None:
        self.k = k
        self.centroids: np.ndarray | None = None
        self._faiss_index = None

    def fit(self, features: np.ndarray) -> None:
        n, d = features.shape
        k = min(self.k, n)
        self.k = k
        features_f32 = np.ascontiguousarray(features, dtype=np.float32)

        try:
            import faiss

            km = faiss.Kmeans(d, k, niter=20, verbose=False, seed=42)
            km.train(features_f32)
            self.centroids = km.centroids.copy()
            index = faiss.IndexFlatL2(d)
            index.add(self.centroids)
            self._faiss_index = index
            logger.info("WEVA clustering via FAISS k-means (K=%d, d=%d)", k, d)
        except ImportError:
            logger.warning("FAISS unavailable; falling back to sklearn KMeans")
            from sklearn.cluster import KMeans

            km = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=42)
            km.fit(features_f32)
            self.centroids = km.cluster_centers_.astype(np.float32)
            self._faiss_index = None

    def assign(self, features: np.ndarray) -> np.ndarray:
        if self.centroids is None:
            raise RuntimeError("WEVAIndex not fitted")
        features_f32 = np.ascontiguousarray(features, dtype=np.float32)
        if self._faiss_index is not None:
            import faiss

            _, ids = self._faiss_index.search(features_f32, 1)
            return ids[:, 0].astype(np.int64)
        dists = ((features_f32[:, None, :] - self.centroids[None, :, :]) ** 2).sum(axis=-1)
        return dists.argmin(axis=1).astype(np.int64)

    def assign_nearest(self, query: np.ndarray) -> int:
        return int(self.assign(query.reshape(1, -1))[0])

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        assert self.centroids is not None
        np.save(directory / "centroids.npy", self.centroids)
        meta = {"k": self.k, "dim": int(self.centroids.shape[1])}
        (directory / "index_meta.json").write_text(json.dumps(meta), encoding="utf-8")
        try:
            import faiss

            if self._faiss_index is not None:
                faiss.write_index(self._faiss_index, str(directory / "weva_index.faiss"))
        except ImportError:
            pass

    @classmethod
    def load(cls, directory: Path) -> WEVAIndex:
        centroids = np.load(directory / "centroids.npy")
        obj = cls(k=int(centroids.shape[0]))
        obj.centroids = centroids.astype(np.float32)
        faiss_path = directory / "weva_index.faiss"
        if faiss_path.exists():
            try:
                import faiss

                obj._faiss_index = faiss.read_index(str(faiss_path))
            except ImportError:
                obj._faiss_index = None
        else:
            obj._faiss_index = None
        return obj


@dataclass
class WEVABuilder:
    """Build WEVA bucket index from preflop CFR warm-up + MC rollout leaf payoffs."""

    k: int = WEVA_DEFAULT_K
    warmup_iterations: int = WEVA_WARMUP_ITERATIONS
    gamma: float = WEVA_DEPTH_GAMMA
    feature_mode: FeatureMode = "hole_only"
    device: str = "cpu"
    num_rollouts: int = WEVA_DEFAULT_ROLLOUTS
    cfr_seed: int = 42
    rollout_seed: int = 42
    max_nodes: int = WEVA_MAX_NODES
    max_roots: int | None = None

    def build(self) -> tuple[WEVAIndex, list[RootInfoSet], np.ndarray, np.ndarray, dict[str, float]]:
        from pokerbot.encoding.weva_cfr import RolloutConfig, run_weva_cfr_pipeline

        roots = expand_root_info_sets(self.feature_mode)
        if self.max_roots is not None and len(roots) > self.max_roots:
            roots = roots[: self.max_roots]
        rollout_cfg = RolloutConfig(
            num_rollouts=self.num_rollouts,
            seed=self.rollout_seed,
            cfr_seed=self.cfr_seed,
            warmup_iterations=self.warmup_iterations,
            num_depths=self.max_nodes,
        )
        features, calib, _avg_strat = run_weva_cfr_pipeline(
            roots,
            rollout_cfg,
            gamma=self.gamma,
            max_nodes=self.max_nodes,
            feature_mode=self.feature_mode,
        )

        index = WEVAIndex(k=self.k)
        index.fit(features)
        bucket_ids = index.assign(features)

        return index, roots, bucket_ids, features, calib

    def build_config_dict(self) -> dict[str, Any]:
        """Every parameter that affects persisted WEVA artifact output."""
        return {
            "k": self.k,
            "warmup_iterations": self.warmup_iterations,
            "gamma": self.gamma,
            "feature_mode": self.feature_mode,
            "num_rollouts": self.num_rollouts,
            "cfr_seed": self.cfr_seed,
            "rollout_seed": self.rollout_seed,
            "max_nodes": self.max_nodes,
            "max_roots": self.max_roots,
        }

    def build_fingerprint(self) -> str:
        return compute_weva_build_fingerprint(self.build_config_dict())

    def build_scope(self) -> Literal["production", "dev"]:
        if self.max_roots is not None:
            return "dev"
        if self.feature_mode == "hole_only":
            roots = expand_root_info_sets("hole_only")
            return "production" if len(roots) == WEVA_PRODUCTION_NUM_ROOTS else "dev"
        return "production"

    def artifacts_current(self, output_dir: Path) -> bool:
        return weva_artifacts_match_config(output_dir, self.build_config_dict())

    def save_artifacts(self, output_dir: Path, *, force: bool = False) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)
        fingerprint = self.build_fingerprint()
        if not force and self.artifacts_current(output_dir):
            logger.info(
                "WEVA artifacts up-to-date at %s (build_fingerprint=%s)",
                output_dir,
                fingerprint,
            )
            return output_dir
        if not force and (output_dir / "meta.json").exists():
            old = json.loads((output_dir / "meta.json").read_text(encoding="utf-8"))
            logger.warning(
                "WEVA build config changed (%s -> %s); rebuilding %s",
                old.get("build_fingerprint"),
                fingerprint,
                output_dir,
            )

        index, roots, bucket_ids, features, calib = self.build()
        index.save(output_dir)
        np.save(output_dir / "features.npy", features.astype(np.float32))

        rows = []
        for i, (root, bid) in enumerate(zip(roots, bucket_ids)):
            rows.append(
                {
                    "root_id": i,
                    "hole_i": root.hole_key[0],
                    "hole_j": root.hole_key[1],
                    "hero_position": root.context.hero_position,
                    "num_players": root.context.num_players,
                    "action_depth_bin": root.context.action_depth_bin,
                    "bucket_id": int(bid),
                }
            )
        try:
            import pandas as pd

            pd.DataFrame(rows).to_parquet(output_dir / "root_index.parquet", index=False)
            pd.DataFrame(rows).to_parquet(output_dir / "bucket_map.parquet", index=False)
        except ImportError:
            (output_dir / "root_index.json").write_text(json.dumps(rows), encoding="utf-8")

        meta = {
            "W": self.warmup_iterations,
            "K": index.k,
            "gamma": self.gamma,
            "feature_mode": self.feature_mode,
            "num_roots": len(roots),
            "max_roots": self.max_roots,
            "build_scope": self.build_scope(),
            "build_fingerprint": fingerprint,
            "build_config": self.build_config_dict(),
            "feature_dim": int(index.centroids.shape[1]) if index.centroids is not None else 0,
            "num_actions_policy": NUM_ACTIONS,
            "num_rollouts": self.num_rollouts,
            "cfr_seed": self.cfr_seed,
            "rollout_seed": self.rollout_seed,
            "max_nodes": self.max_nodes,
            "cfr_backend": "preflop_cfr_mc_rollout",
            **calib,
        }
        (output_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        logger.info("Saved WEVA artifacts to %s (build_fingerprint=%s)", output_dir, fingerprint)
        return output_dir


def compute_weva_build_fingerprint(config: dict[str, Any]) -> str:
    """Deterministic hash of build config; stored in meta.json for staleness detection."""
    payload = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def weva_artifacts_complete(directory: Path) -> bool:
    return (
        (directory / "centroids.npy").exists()
        and (directory / "features.npy").exists()
        and (directory / "meta.json").exists()
    )


def weva_artifacts_match_config(directory: Path, config: dict[str, Any]) -> bool:
    """True when on-disk artifacts were built with exactly this config."""
    if not weva_artifacts_complete(directory):
        return False
    meta = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
    return meta.get("build_fingerprint") == compute_weva_build_fingerprint(config)


def is_production_weva_index(index_path: Path) -> bool:
    meta_path = index_path / "meta.json"
    if not meta_path.exists():
        return False
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta.get("build_scope") != "production":
        return False
    if meta.get("feature_mode") != "hole_only":
        return False
    if int(meta.get("num_roots", 0)) != WEVA_PRODUCTION_NUM_ROOTS:
        return False
    if meta.get("max_roots") is not None:
        return False
    return True


def assert_production_weva_index(index_path: Path, *, context: str) -> None:
    """Reject dev-truncated WEVA indexes for downstream training (Phase 2+)."""
    if is_production_weva_index(index_path):
        return
    meta_path = index_path / "meta.json"
    detail = "missing meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        detail = (
            f"build_scope={meta.get('build_scope')!r}, "
            f"num_roots={meta.get('num_roots')}, "
            f"max_roots={meta.get('max_roots')}, "
            f"K={meta.get('K')}"
        )
    raise ValueError(
        f"WEVA index at {index_path} is not a production build ({detail}). "
        f"Dev/truncated indexes (--max-roots, --quick-weva) are for pipeline smoke tests only. "
        f"For {context}, run:\n"
        f"  python scripts/build_weva_index.py --output {index_path} --k 10000 --rollouts 128\n"
        f"(no --max-roots)"
    )


class WEVAMapper:
    """Runtime lookup: hole cards (+ context) -> WEVA bucket id."""

    _DEFAULT: WEVAMapper | None = None

    def __init__(
        self,
        index: WEVAIndex,
        feature_mode: FeatureMode = "hole_only",
        features: np.ndarray | None = None,
        root_index: list[dict] | None = None,
    ) -> None:
        self.index = index
        self.feature_mode = feature_mode
        self.features = features
        self._root_key_to_row: dict[tuple, int] = {}
        if root_index:
            for row in root_index:
                if "hole_i" in row and "hole_j" in row:
                    key = (
                        int(row["hole_i"]),
                        int(row["hole_j"]),
                        int(row.get("hero_position", 0)),
                        int(row.get("num_players", 6)),
                        int(row.get("action_depth_bin", 0)),
                    )
                    self._root_key_to_row[key] = int(row["root_id"])

    @classmethod
    def from_directory(cls, directory: Path, device: str = "cpu") -> WEVAMapper:
        meta_path = directory / "meta.json"
        feature_mode: FeatureMode = "hole_only"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            feature_mode = meta.get("feature_mode", feature_mode)
        index = WEVAIndex.load(directory)

        features = None
        feat_path = directory / "features.npy"
        if feat_path.exists():
            features = np.load(feat_path)

        root_index: list[dict] = []
        idx_path = directory / "root_index.parquet"
        if idx_path.exists():
            import pandas as pd

            root_index = pd.read_parquet(idx_path).to_dict(orient="records")
        else:
            legacy = directory / "bucket_map.parquet"
            if legacy.exists():
                import pandas as pd

                root_index = pd.read_parquet(legacy).to_dict(orient="records")

        return cls(
            index=index,
            feature_mode=feature_mode,
            features=features,
            root_index=root_index or None,
        )

    @classmethod
    def default(cls) -> WEVAMapper:
        if cls._DEFAULT is None:
            path = Path("data/weva")
            if not weva_artifacts_complete(path):
                logger.warning("WEVA index incomplete at %s; rebuilding minimal dev index", path)
                WEVABuilder(
                    k=100,
                    feature_mode="hole_only",
                    device="cpu",
                    num_rollouts=8,
                    max_roots=64,
                ).save_artifacts(path)
            cls._DEFAULT = cls.from_directory(path)
        return cls._DEFAULT

    def _lookup_feature(self, root: RootInfoSet) -> np.ndarray:
        if self.features is None:
            raise RuntimeError(
                "WEVA feature cache missing; rebuild index with build_weva_index.py"
            )
        key = (
            root.hole_key[0],
            root.hole_key[1],
            root.context.hero_position,
            root.context.num_players,
            root.context.action_depth_bin,
        )
        if key in self._root_key_to_row:
            return self.features[self._root_key_to_row[key]]

        hi, hj = root.hole_key
        same_hole_rows = [
            self._root_key_to_row[k]
            for k in self._root_key_to_row
            if k[0] == hi and k[1] == hj
        ]
        if same_hole_rows:
            feat = self.features[same_hole_rows[0]].copy()
            if self.feature_mode == "hole_plus_context":
                suffix = root.context.context_vector()
                if feat.shape[0] >= len(suffix):
                    feat[-len(suffix) :] = suffix
            return feat

        return self.features[self.index.assign_nearest(self.features.mean(axis=0))]

    def lookup(
        self,
        hole: tuple[Card, Card],
        position: int = 0,
        context: WEVAContext | None = None,
    ) -> int:
        if hole[0].index > hole[1].index:
            hole = (hole[1], hole[0])
        if _PREFLOP_LOSSLESS:
            # Identity: lossless 169-class x position; no WEVA index/feature cache needed.
            cls = canonical_preflop_class(hole[0].index, hole[1].index)
            return (position % 6) * PREFLOP_NUM_CLASSES + cls
        ctx = context or WEVAContext(hero_position=position)
        root = RootInfoSet(hole=hole, context=ctx)
        feat = self._lookup_feature(root)
        return self.index.assign_nearest(feat)

    def lookup_from_cards(self, c0: Card, c1: Card, hero_position: int = 0) -> int:
        return self.lookup((c0, c1), position=hero_position)


def holes_from_card_tensor(cards: np.ndarray) -> tuple[Card, Card] | None:
    """Decode hero hole cards from legacy (9, 4, 13) card tensor slots 0-1."""
    cards = np.asarray(cards)

    def _slot_card(slot: int) -> Card | None:
        plane = cards[slot]
        coords = np.argwhere(plane > 0)
        if coords.size == 0:
            return None
        suit_i, rank_i = int(coords[0, 0]), int(coords[0, 1])
        return Card(Rank(rank_i + 2), Suit(suit_i))

    c0, c1 = _slot_card(0), _slot_card(1)
    if c0 is None or c1 is None:
        return None
    return c0, c1


def weva_id_from_tensors(
    cards: np.ndarray,
    position: np.ndarray | None = None,
    mapper: WEVAMapper | None = None,
) -> int:
    hole = holes_from_card_tensor(cards)
    if hole is None:
        return 0
    pos = int(np.argmax(position)) if position is not None else 0
    m = mapper or WEVAMapper.default()
    return m.lookup(hole, position=pos)


def load_weva_bucket_count(index_path: Path) -> int:
    meta_path = index_path / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"WEVA meta.json not found at {index_path}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    return int(meta["K"])
