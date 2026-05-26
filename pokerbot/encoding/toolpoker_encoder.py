"""ToolPoker unified solver API JSON encoding for Phase 1 SFT."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import numpy as np

from pokerbot.engine.actions import (
    ACTION_VOCAB,
    NUM_ACTIONS,
    action_to_index,
    parse_pokerbench_output,
)
from pokerbot.engine.cards import Card, parse_cards, parse_holding
from pokerbot.encoding.csv_encoder import (
    available_moves_to_legal_mask,
    encode_postflop_action,
    encode_prev_line,
    position_one_hot,
)
from pokerbot.data.sft_dataloader import LABEL_SOURCE_SOLVER
from pokerbot.encoding.weva_encoder import WEVAContext, WEVAMapper


@dataclass(frozen=True, slots=True)
class ToolPokerSolverRecord:
    weva_bucket_id: int
    actions: np.ndarray
    position: np.ndarray
    legal_mask: np.ndarray
    solver_label: int
    solver_policy: np.ndarray
    stage: str
    raw_json: str | None = None


_DIST_KEY_ALIASES: dict[str, str] = {
    "fold": "fold",
    "check": "check",
    "call": "call",
    "all_in": "all_in",
    "allin": "all_in",
    "all-in": "all_in",
    "all in": "all_in",
    "bet_0.33": "bet_0.33",
    "bet_0.50": "bet_0.50",
    "bet_0.66": "bet_0.66",
    "bet_0.75": "bet_0.75",
    "bet_1.00": "bet_1.00",
    "bet_1.50": "bet_1.50",
    "bet_2.00": "bet_2.00",
    "bet_0.5": "bet_0.50",
    "bet_0.75": "bet_0.75",
    "bet_1.0": "bet_1.00",
    "bet_1.5": "bet_1.50",
    "bet_2.0": "bet_2.00",
}


def _normalize_dist_key(key: str) -> str | None:
    k = key.strip().lower().replace("-", "_").replace(" ", "_")
    if k in _DIST_KEY_ALIASES:
        return _DIST_KEY_ALIASES[k]
    if k.startswith("bet_"):
        try:
            val = float(k.split("_", 1)[1])
            canonical = f"bet_{val:.2f}"
            if canonical in ACTION_VOCAB:
                return canonical
        except ValueError:
            pass
    return None


def action_distribution_to_policy(
    distribution: dict[str, float],
    legal_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Map ToolPoker action_distribution to 11-d soft policy vector."""
    policy = np.zeros(NUM_ACTIONS, dtype=np.float32)
    for key, prob in distribution.items():
        norm_key = _normalize_dist_key(str(key))
        if norm_key is None:
            continue
        try:
            idx = ACTION_VOCAB.index(norm_key)
            policy[idx] += float(prob)
        except ValueError:
            try:
                action = parse_pokerbench_output(norm_key.replace("_", " "))
                policy[action_to_index(action)] += float(prob)
            except ValueError:
                continue

    if legal_mask is not None:
        policy = policy * legal_mask.astype(np.float32)

    total = policy.sum()
    if total <= 0:
        if legal_mask is not None and legal_mask.any():
            policy = legal_mask.astype(np.float32) / legal_mask.sum()
        else:
            policy[:] = 1.0 / NUM_ACTIONS
    else:
        policy /= total
    return policy


def parse_toolpoker_solver_json(payload: dict[str, Any] | str) -> dict[str, Any]:
    """Parse ToolPoker unified API response; return inner data dict."""
    if isinstance(payload, str):
        payload = json.loads(payload)
    if not isinstance(payload, dict):
        raise ValueError("ToolPoker payload must be a JSON object")
    if not payload.get("success", True):
        err = payload.get("error", {})
        raise ValueError(f"ToolPoker solver error: {err}")
    data = payload.get("data", payload)
    if not isinstance(data, dict):
        raise ValueError("ToolPoker data field must be an object")
    return data


def _hole_from_data(data: dict[str, Any]) -> tuple[Card, Card]:
    hole_raw = data.get("hole_cards") or data.get("holding") or []
    if isinstance(hole_raw, str):
        return parse_holding(hole_raw.replace(" ", ""))
    if isinstance(hole_raw, (list, tuple)) and len(hole_raw) >= 2:
        c0 = hole_raw[0] if isinstance(hole_raw[0], Card) else Card.from_str(str(hole_raw[0]))
        c1 = hole_raw[1] if isinstance(hole_raw[1], Card) else Card.from_str(str(hole_raw[1]))
        return c0, c1
    raise ValueError("ToolPoker data missing hole_cards")


def _position_index(data: dict[str, Any]) -> int:
    from pokerbot.encoding.csv_encoder import POSITION_MAP

    pos = str(data.get("hero_position", "UTG")).strip().upper()
    return POSITION_MAP.get(pos, 0)


def _encode_actions_from_data(data: dict[str, Any], stage: str) -> np.ndarray:
    pot = float(data.get("pot_bb", data.get("pot_size", 1.5)) or 1.5)
    history = str(data.get("action_history", data.get("prev_line", "")))
    if stage == "preflop":
        return encode_prev_line(history, pot)
    preflop = encode_prev_line(str(data.get("preflop_action", "")), pot * 0.3)
    postflop = encode_postflop_action(
        history,
        pot,
        str(data.get("evaluation_at", stage)),
    )
    return np.maximum(preflop, postflop)


def encode_toolpoker_row(
    payload: dict[str, Any] | str,
    mapper: WEVAMapper | None = None,
) -> ToolPokerSolverRecord:
    """Encode ToolPoker solver JSON into training tensors and labels."""
    raw = payload if isinstance(payload, str) else json.dumps(payload)
    data = parse_toolpoker_solver_json(payload)

    hole = _hole_from_data(data)
    pos_idx = _position_index(data)
    stage = str(data.get("stage", "preflop")).lower()

    moves = data.get("available_actions") or data.get("available_moves") or []
    if isinstance(moves, str):
        moves = [moves]
    legal = available_moves_to_legal_mask([str(m) for m in moves])

    gto_action = str(data.get("gto_action", data.get("correct_decision", "fold")))
    solver_label = action_to_index(parse_pokerbench_output(gto_action.lower()))

    dist = data.get("action_distribution")
    if not isinstance(dist, dict) or len(dist) == 0:
        raise ValueError(
            "ToolPoker solver JSON must include non-empty action_distribution; "
            "argmax-only labels are not accepted for solver SFT."
        )
    solver_policy = action_distribution_to_policy(dist, legal)
    if float(solver_policy.sum()) <= 0:
        raise ValueError("action_distribution produced zero mass on legal actions")

    actions = _encode_actions_from_data(data, stage)
    position = position_one_hot(str(data.get("hero_position", "UTG")))

    ctx = WEVAContext(
        hero_position=pos_idx,
        num_players=int(data.get("num_players", 6)),
        action_depth_bin=int(data.get("action_depth_bin", 0)),
    )
    m = mapper or WEVAMapper.default()
    bucket_id = m.lookup(hole, position=pos_idx, context=ctx)

    return ToolPokerSolverRecord(
        weva_bucket_id=bucket_id,
        actions=actions,
        position=position,
        legal_mask=legal,
        solver_label=solver_label,
        solver_policy=solver_policy,
        stage=stage,
        raw_json=raw,
    )


def encode_toolpoker_to_batch_dict(record: ToolPokerSolverRecord) -> dict[str, Any]:
    """Convert record to plain dict for Parquet serialization."""
    return {
        "weva_bucket_id": record.weva_bucket_id,
        "actions": record.actions.tolist(),
        "position": record.position.tolist(),
        "legal_mask": record.legal_mask.tolist(),
        "solver_label": record.solver_label,
        "solver_policy": record.solver_policy.tolist(),
        "label_source": LABEL_SOURCE_SOLVER,
        "stage": record.stage,
        "raw_solver_json": record.raw_json,
    }
