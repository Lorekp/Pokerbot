"""Clone data pipeline — skill-bin scoring + a skill-augmented trajectory dataset.

Two jobs (design components #2/#3 support, the conflict-free part — the heavy IRC/CoinPoker ingesters
are thin scripts that call into here):

1. **Skill-bin scoring** (:func:`composite_skill_score` / :func:`skill_bin`): the per-player STRENGTH
   axis. The composite score is ``s = z(PFR) - z(VPIP-PFR) + z(AF)`` (tighter-aggressive ⇒ higher),
   optionally augmented with a showdown bb/100 / won-wagered signal so a winning LAG and a fish LAG do
   not share a bin (design risk #3). ``skill_bin`` maps the score to a decile in [0, NUM_SKILL_BINS).
   The z-normalization uses the IRC population means/stds from ``irc_population_summary.json`` so the
   axis is calibrated to the real field (no magic numbers).

2. **Skill-augmented dataset** (:class:`SkillTrajectoryDataset`): a thin wrapper over
   :class:`pokerbot.training.trajectory_dataset.OpponentTrajectoryDataset` that carries ONE extra
   per-row ``skill_bin`` column (per the design's "+1 skill_bin column") plus a ``data_source`` tag,
   and a ``collate`` that builds the ``(B, NUM_PLAYERS)`` per-seat ``skill_bins`` + ``(B,)``
   ``table_skill`` tensors the :class:`HumanCloneNet` consumes. Reuses ``trajectory_to_batch``
   verbatim for everything except the two skill tensors.

The actual IRC re-stream (``scripts/ingest_irc_trajectories.py``) and CoinPoker reader
(``scripts/ingest_coinpoker_trajectories.py``) are documented TODOs that build ``TrajectoryRecord``s
via ``HandTrajectoryBuilder`` and write them with ``trajectory_dataset.append_trajectories`` + the
``skill_bin`` / ``data_source`` metadata columns this module reads back. :func:`build_engine_smoke_shard`
provides a SELF-CONTAINED, dependency-free shard (real engine hands, synthetic skill bins) so the whole
clone pipeline is smoke-testable without the 1 GB tarball.
"""

from __future__ import annotations

import json
import math
import random
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import torch

from pokerbot.engine.game import NUM_PLAYERS
from pokerbot.training.tourney_selfplay.human_clone import NUM_SKILL_BINS, NUM_TABLE_SKILL_BINS

_SUMMARY_PATH = (
    Path(__file__).resolve().parents[3]
    / "data"
    / "player_profiles"
    / "irc_population_summary.json"
)

# Data-source tags (design A2): every row is tagged so the trainer can weight / hold out by source.
SOURCE_IRC = "irc"
SOURCE_COINPOKER = "coinpoker"
SOURCE_ENGINE_SMOKE = "engine_smoke"


@lru_cache(maxsize=1)
def _population_moments() -> dict[str, tuple[float, float]]:
    """(mean, std) for VPIP / PFR / AF from the IRC summary, for z-normalizing the skill axis.

    The summary stores p10/p50/p90 (not std); we recover an approximate std from the 10-90 spread
    (``(p90 - p10) / 2.563`` — the normal-quantile width), which is robust enough for a decile bin.
    Falls back to fixed loose-passive moments if the file is absent (never raises).
    """
    try:
        data = json.loads(_SUMMARY_PATH.read_text(encoding="utf-8"))
        pp = data["population_prior"]
    except (OSError, ValueError, KeyError):
        # Loose-passive defaults (population VPIP .497 / PFR .140 / AF .88).
        return {"vpip": (0.497, 0.18), "pfr": (0.140, 0.10), "af": (0.88, 0.50)}

    def moments(key: str) -> tuple[float, float]:
        d = pp.get(key, {})
        mean = float(d.get("mean", 0.0))
        p10 = float(d.get("p10", mean))
        p90 = float(d.get("p90", mean))
        std = max((p90 - p10) / 2.563, 1e-6)
        return mean, std

    return {
        "vpip": moments("vpip"),
        "pfr": moments("pfr"),
        "af": moments("aggression_factor"),
    }


def _z(value: float, key: str) -> float:
    mean, std = _population_moments()[key]
    return (float(value) - mean) / std


def composite_skill_score(
    vpip: float,
    pfr: float,
    af: float,
    *,
    showdown_bb100: float | None = None,
    showdown_weight: float = 0.5,
) -> float:
    """The composite STRENGTH score ``s = z(PFR) - z(VPIP-PFR) + z(AF)`` (design A1).

    Tighter-aggressive ⇒ higher. ``VPIP-PFR`` is the "gap" (loose-passive calling) whose z-score is
    SUBTRACTED. An optional ``showdown_bb100`` (a measured win-rate signal) is z-normalized by a
    nominal 30 bb/100 spread and added with ``showdown_weight`` so a winning player out-ranks a
    same-style loser (design risk #3 mitigation).
    """
    gap = max(0.0, float(vpip) - float(pfr))
    # z(VPIP-PFR): use the VPIP std as the gap's scale (the gap is dominated by VPIP variance).
    _, vpip_std = _population_moments()["vpip"]
    z_gap = gap / max(vpip_std, 1e-6)
    s = _z(pfr, "pfr") - z_gap + _z(af, "af")
    if showdown_bb100 is not None:
        s += showdown_weight * (float(showdown_bb100) / 30.0)
    return float(s)


# The decile cut-points for the composite score, assuming an approximately-standard-normal axis (the
# z-scores roughly center on 0). 10 bins ⇒ 9 inner cuts at the standard-normal deciles.
_DECILE_CUTS = tuple(
    float(_q) for _q in (-1.2816, -0.8416, -0.5244, -0.2533, 0.0, 0.2533, 0.5244, 0.8416, 1.2816)
)


def skill_bin(score: float, *, num_bins: int = NUM_SKILL_BINS) -> int:
    """Map a composite skill score to a decile bin in ``[0, num_bins)`` (0 = weakest)."""
    b = 0
    for cut in _DECILE_CUTS[: num_bins - 1]:
        if score >= cut:
            b += 1
    return min(b, num_bins - 1)


def table_skill_bin(opponent_bins: list[int], *, num_bins: int = NUM_TABLE_SKILL_BINS) -> int:
    """Discretize the TABLE-skill level = the rounded mean of the seated opponents' skill bins."""
    if not opponent_bins:
        return num_bins // 2
    mean = sum(opponent_bins) / len(opponent_bins)
    return int(min(max(round(mean), 0), num_bins - 1))


# -- the skill-augmented dataset -------------------------------------------------------------------


class SkillTrajectoryDataset(torch.utils.data.Dataset):
    """Trajectory shards + a per-row ``skill_bin`` (per-seat) and ``data_source`` tag.

    Wraps the existing :class:`OpponentTrajectoryDataset` for the trajectory tensors; reads the two
    extra columns this module's ingesters write. ``collate`` returns the base batch dict PLUS:
        * ``skill_bins``  (B, NUM_PLAYERS) long — per-absolute-seat skill bin.
        * ``table_skill`` (B,) long — the discretized table-skill level.
    so :meth:`HumanCloneNet.forward(..., skill_bins=, table_skill=)` consumes it directly.
    """

    def __init__(self, parquet_glob: str | Path, max_samples: int | None = None) -> None:
        from pokerbot.training.trajectory_dataset import OpponentTrajectoryDataset

        self._base = OpponentTrajectoryDataset(parquet_glob, max_samples=max_samples)
        # Pull the per-row skill columns (default to a neutral bin when a shard predates the column).
        self._skill_bins: list[list[int]] = []
        self._sources: list[str] = []
        for row in self._base.rows:
            sb = row.get("skill_bins")
            if isinstance(sb, str):
                sb = json.loads(sb)
            if not isinstance(sb, list) or len(sb) != NUM_PLAYERS:
                sb = [NUM_SKILL_BINS // 2] * NUM_PLAYERS
            self._skill_bins.append([int(x) for x in sb])
            self._sources.append(str(row.get("data_source", SOURCE_IRC)))

    def __len__(self) -> int:
        return len(self._base)

    def __getitem__(self, idx: int):
        return self._base[idx], self._skill_bins[idx]

    def collate(self, batch: list) -> dict[str, torch.Tensor]:
        from pokerbot.opponent.sequence_encoder import trajectory_to_batch

        trajs = [b[0] for b in batch]
        skills = [b[1] for b in batch]
        out = trajectory_to_batch(trajs)
        skill_bins = torch.tensor(skills, dtype=torch.long)            # (B, NUM_PLAYERS)
        # table-skill = rounded mean of the non-hero seats' bins per row.
        table = []
        for row, traj in zip(skills, trajs):
            opp = [row[s] for s in range(NUM_PLAYERS) if s != traj.hero_seat]
            table.append(table_skill_bin(opp))
        out["skill_bins"] = skill_bins
        out["table_skill"] = torch.tensor(table, dtype=torch.long)
        return out


def write_skill_shard(
    trajectories: list,
    skill_bins_per_traj: list[list[int]],
    output_dir: Path,
    *,
    data_source: str = SOURCE_IRC,
    shard_name: str | None = None,
) -> Path:
    """Write trajectory records + the per-seat ``skill_bins`` column + a ``data_source`` tag.

    Reuses :func:`trajectory_dataset.append_trajectories` for the trajectory tensors; the skill column
    is injected as per-row metadata (one ``skill_bins`` list per record, NOT a single constant, so the
    ingester can label every seat). We therefore write the shard manually to keep the per-row variation.
    """
    import pandas as pd

    from pokerbot.opponent.sequence_encoder import trajectory_to_dict

    output_dir.mkdir(parents=True, exist_ok=True)
    existing = list(output_dir.glob("shard_*.parquet"))
    path = output_dir / (shard_name or f"shard_{len(existing):05d}.parquet")

    rows = []
    for traj, sb in zip(trajectories, skill_bins_per_traj):
        d = trajectory_to_dict(traj)
        rows.append(
            {
                **d,
                "token_types": json.dumps(d["token_types"]),
                "token_features": json.dumps(d["token_features"]),
                "actor_seats": json.dumps(d["actor_seats"]),
                "belief_targets": json.dumps(d["belief_targets"]),
                "policy_targets": json.dumps(d["policy_targets"]),
                "cards_known_mask": json.dumps(d["cards_known_mask"]),
                "action_known_mask": json.dumps(d["action_known_mask"]),
                "skill_bins": json.dumps([int(x) for x in sb]),
                "data_source": data_source,
            }
        )
    pd.DataFrame(rows).to_parquet(path, index=False)
    return path


def build_engine_smoke_shard(
    output_dir: Path, *, n_hands: int = 24, seed: int = 0
) -> Path:
    """Self-contained smoke shard: real engine hands + synthetic per-seat skill bins.

    Plays ``n_hands`` random 6-max hands through the engine, records each with
    :class:`HandTrajectoryBuilder` (the design's reuse seam), assigns each seat a random skill bin, and
    writes ONE skill-augmented Parquet shard. Lets the clone trainer smoke-test the full
    ingest→dataset→train loop with NO external data (the IRC tarball / CoinPoker JSONL paths are the
    production sources; this is the CI fixture).
    """
    from pokerbot.engine.actions import Action, ActionType
    from pokerbot.engine.game import GameState
    from pokerbot.opponent.sequence_encoder import HandTrajectoryBuilder

    rng = random.Random(seed)
    trajectories = []
    skills = []

    for h in range(n_hands):
        hero_seat = rng.randrange(NUM_PLAYERS)
        state = GameState.new_hand(button_seat=rng.randrange(NUM_PLAYERS), rng=rng)
        state.deal_hole_cards()
        builder = HandTrajectoryBuilder(hero_seat=hero_seat)
        builder.init_stacks(state)
        # Drive a short random betting sequence so the trajectory has STATE/ACTION/REWARD tokens.
        # The engine mutates ``state`` in place: apply_action advances the street when complete, but
        # within a street the actor must be walked via advance_to_next_actor.
        steps = 0
        while not state.is_terminal() and steps < 40:
            actor = state.action_seat
            legal = state.legal_actions(actor)
            if not legal:
                state.advance_to_next_actor()
                steps += 1
                continue
            builder.on_decision(state, actor)
            action = rng.choice(legal)
            prev_street = state.street
            builder.on_action(state, actor, action)
            state.apply_action(action)
            # If the street did not advance (same street, not terminal), walk to the next actor.
            if not state.is_terminal() and state.street == prev_street:
                state.advance_to_next_actor()
            steps += 1
        if state.is_terminal():
            builder.record_showdown(state)
        traj = builder.finalize()
        if not traj.token_types:
            continue
        trajectories.append(traj)
        skills.append([rng.randrange(NUM_SKILL_BINS) for _ in range(NUM_PLAYERS)])

    if not trajectories:
        raise RuntimeError("engine smoke produced no trajectories")
    return write_skill_shard(
        trajectories, skills, output_dir, data_source=SOURCE_ENGINE_SMOKE, shard_name="shard_00000.parquet"
    )


def held_out_player_split(
    player_ids: list[str], *, holdout_frac: float = 0.10, seed: int = 0
) -> tuple[set[str], set[str]]:
    """Leakage-free split BY PLAYER (design A3): ≥``holdout_frac`` of distinct players held out.

    Returns ``(train_players, heldout_players)``. The match-rate metric is computed only on heldout
    players' rows so a high top-1 match-rate cannot be memorization.
    """
    uniq = sorted(set(player_ids))
    rng = random.Random(seed)
    rng.shuffle(uniq)
    n_hold = max(1, math.ceil(len(uniq) * holdout_frac))
    held = set(uniq[:n_hold])
    train = set(uniq[n_hold:])
    return train, held
