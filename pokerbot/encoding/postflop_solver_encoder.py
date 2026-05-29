"""Encode postflop-solver labels into SFT Parquet rows."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import numpy as np

from pokerbot.engine.actions import action_to_index, parse_pokerbench_output
from pokerbot.engine.game import Position
from pokerbot.encoding.csv_encoder import available_moves_to_legal_mask, encode_prev_line, position_one_hot
from pokerbot.encoding.spot_scope import encode_spot_scope
from pokerbot.encoding.tensors import encode_state
from pokerbot.encoding.toolpoker_encoder import action_distribution_to_policy, solver_action_keys_to_legal_mask
from pokerbot.encoding.weva_encoder import WEVAContext, WEVAMapper
from pokerbot.solver.label_diff import canonicalize_action_distribution
from pokerbot.solver.postflop_solver_client import SolverSolveResult
from pokerbot.solver.spot_enumerator import SolverSpot

LABEL_SOURCE_POSTFLOP_SOLVER = "postflop_solver"
SOLVER_ENGINE = "postflop_solver"


@dataclass(frozen=True, slots=True)
class SolverLabelRecord:
    weva_bucket_id: int
    cards: np.ndarray
    actions: np.ndarray
    position: np.ndarray
    spot_scope: np.ndarray
    legal_mask: np.ndarray
    solver_label: int
    solver_policy: np.ndarray
    stage: str
    spot_class: str
    solver_converged: bool
    solver_exploitability_pct: float
    solver_iterations_used: int
    solver_engine: str
    oop_range: str
    ip_range: str
    bet_sizes: str
    raise_sizes: str
    raw_json: str


def _position_name(seat: int, state) -> str:
    return state.players[seat].position.name


def _available_moves_from_state(state, hero_seat: int) -> list[str]:
    legal = state.legal_actions(hero_seat)
    moves: list[str] = []
    for a in legal:
        label = a.label().lower()
        if label.startswith("bet ") or label.startswith("raise "):
            # Map to bucket keys via action_to_index round-trip
            idx = action_to_index(a)
            from pokerbot.engine.actions import ACTION_VOCAB

            moves.append(ACTION_VOCAB[idx].replace("_", " "))
        elif label == "all in":
            moves.append("all_in")
        else:
            moves.append(label)
    return moves


def _merge_solver_dist_with_legal(
    solver_dist: dict[str, float],
    legal_mask: np.ndarray,
) -> tuple[np.ndarray, int]:
    policy = action_distribution_to_policy(solver_dist, legal_mask)
    if float(policy.sum()) <= 0:
        raise ValueError("action_distribution produced zero mass on legal actions")
    solver_label = int(np.argmax(policy))
    return policy, solver_label


def encode_solver_spot_row(
    spot: SolverSpot,
    result: SolverSolveResult,
    mapper: WEVAMapper | None = None,
    *,
    bet_sizes: str,
    raise_sizes: str,
    target_exploitability_pct: float,
) -> SolverLabelRecord:
    if result.error:
        raise ValueError(f"solver error: {result.error}")
    if not result.converged:
        raise ValueError(
            f"solver did not converge (exploitability={result.exploitability_pct_of_pot:.3f}% "
            f"of pot, target={target_exploitability_pct}%)"
        )
    if not result.action_distribution:
        raise ValueError("empty action_distribution from solver")

    state = spot.state
    hero = spot.hero_seat
    state.players[hero].hole = spot.hero_hole
    state.action_seat = hero

    batch = encode_state(
        state,
        hero,
        mapper=mapper,
        spot_class=spot.spot_class.value,
        oop_range=spot.oop_range,
        ip_range=spot.ip_range,
    )
    if result.available_actions:
        legal_mask = solver_action_keys_to_legal_mask(result.available_actions)
    else:
        legal = _available_moves_from_state(state, hero)
        legal_mask = available_moves_to_legal_mask(legal)
    policy, solver_label = _merge_solver_dist_with_legal(result.action_distribution, legal_mask)

    pos_idx = int(state.players[hero].position)
    ctx = WEVAContext(hero_position=pos_idx, num_players=2, action_depth_bin=0)
    m = mapper or WEVAMapper.default()
    bucket_id = m.lookup(spot.hero_hole, position=pos_idx, context=ctx)

    exploit_pct = result.exploitability_pct_of_pot

    raw = {
        "spot_id": spot.spot_id,
        "solver_request": {
            "oop_range": spot.oop_range,
            "ip_range": spot.ip_range,
            "board": [c.to_str() for c in spot.board_cards],
            "starting_pot_bb": spot.starting_pot_bb,
            "effective_stack_bb": spot.effective_stack_bb,
            "bet_sizes": bet_sizes,
            "raise_sizes": raise_sizes,
            "target_exploitability_pct": target_exploitability_pct,
        },
        "solver_response": result.raw or {},
        "label_validity_scope": (
            f"GTO for HU postflop vs OOP range ({spot.oop_range}) "
            f"and IP range ({spot.ip_range}) with bet tree {bet_sizes} / {raise_sizes}. "
            "Not general full-range HU GTO unless ranges match live play."
        ),
        "hero_position": _position_name(hero, state),
        "hero_hole": [c.to_str() for c in spot.hero_hole],
        "action_distribution": canonicalize_action_distribution(result.action_distribution),
    }

    return SolverLabelRecord(
        weva_bucket_id=bucket_id,
        cards=batch["cards"],
        actions=batch["actions"],
        position=batch["position"],
        spot_scope=batch["spot_scope"],
        legal_mask=legal_mask,
        solver_label=solver_label,
        solver_policy=policy,
        stage="postflop",
        spot_class=spot.spot_class.value,
        solver_converged=result.converged,
        solver_exploitability_pct=exploit_pct,
        solver_iterations_used=result.iterations_used,
        solver_engine=SOLVER_ENGINE,
        oop_range=spot.oop_range,
        ip_range=spot.ip_range,
        bet_sizes=bet_sizes,
        raise_sizes=raise_sizes,
        raw_json=json.dumps(raw, sort_keys=True),
    )


def record_to_parquet_dict(record: SolverLabelRecord) -> dict[str, Any]:
    return {
        "weva_bucket_id": record.weva_bucket_id,
        "cards": record.cards.tolist(),
        "actions": record.actions.tolist(),
        "position": record.position.tolist(),
        "spot_scope": record.spot_scope.tolist(),
        "legal_mask": record.legal_mask.tolist(),
        "solver_label": record.solver_label,
        "solver_policy": record.solver_policy.tolist(),
        "label_source": LABEL_SOURCE_POSTFLOP_SOLVER,
        "stage": record.stage,
        "spot_class": record.spot_class,
        "solver_converged": record.solver_converged,
        "solver_exploitability_pct": record.solver_exploitability_pct,
        "solver_iterations_used": record.solver_iterations_used,
        "solver_engine": record.solver_engine,
        "oop_range": record.oop_range,
        "ip_range": record.ip_range,
        "bet_sizes": record.bet_sizes,
        "raise_sizes": record.raise_sizes,
        "raw_solver_json": record.raw_json,
    }
