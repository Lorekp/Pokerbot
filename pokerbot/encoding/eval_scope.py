"""Inference-time spot scope for policy encoding (eval / live play)."""

from __future__ import annotations

from typing import Any

import numpy as np

from pokerbot.engine.game import GameState, Street
from pokerbot.encoding.spot_scope import (
    SPOT_CLASS_ORDER,
    encode_spot_scope,
    fuse_position_scope,
)
from pokerbot.encoding.tensors import encode_state
from pokerbot.encoding.weva_encoder import WEVAMapper

# Training distribution (prod60).
TRAINING_SPOT_CLASS_COUNTS = {
    "hu_postflop_2p": 40,
    "pseudo_hu_bunching": 20,
}
TRAINING_SPOT_CLASS_FRACTION = {
    k: v / sum(TRAINING_SPOT_CLASS_COUNTS.values())
    for k, v in TRAINING_SPOT_CLASS_COUNTS.items()
}

SCOPE_RANGE_PAIRS: dict[str, tuple[str, str]] = {
    "hu_postflop_2p": ("QQ+,AK", "TT-22,AQs+"),
    "pseudo_hu_bunching": ("TT+,AQs+", "99-22,ATs+"),
}

GATE_ISOLATED_HARNESS_SCOPE_NOTE = (
    "Phase 1 exploitability gate uses an isolated 2-active-seat harness with four "
    "auto-fold seats. That construction can only exercise hu_postflop_2p policy "
    "scope (~67% of prod60 training labels). pseudo_hu_bunching (~33%) is never "
    "produced or evaluated by this harness — not a silent vanilla default."
)

_POSTFLOP = frozenset({Street.FLOP, Street.TURN, Street.RIVER})


def is_gate_isolated_hu_shape(state: GameState) -> bool:
    """True when exactly two players live and four are folded (gate BR harness)."""
    live = sum(1 for p in state.players if not p.folded)
    folded = sum(1 for p in state.players if p.folded)
    return live == 2 and folded == 4


def classify_phase1_gate_scope(state: GameState, target_seat: int) -> tuple[bool, str]:
    """
    Structural scope for Phase 1 gate exploitability collection.

    Isolated harness states are vanilla hu_postflop_2p only (see GATE_ISOLATED_HARNESS_SCOPE_NOTE).
    Full-ring states may classify as pseudo_hu_bunching when bunching validity passes.
    """
    from pokerbot.solver.spot_validity import is_two_player_valid_state, spot_node_from_state

    live = {p.seat for p in state.players if not p.folded}
    if state.street.name == "PREFLOP":
        return False, "preflop_out_of_scope"
    if state.street == Street.SHOWDOWN:
        return False, "showdown_terminal"
    if len(live) != 2:
        return False, f"multiway_live_{len(live)}"
    if target_seat not in live:
        return False, "target_not_live"

    if is_gate_isolated_hu_shape(state):
        validity = is_two_player_valid_state(state, target_seat, bunching=False)
        if not validity.valid:
            return False, validity.reason
        return True, "hu_postflop_2p"

    for bunching in (True, False):
        validity = is_two_player_valid_state(state, target_seat, bunching=bunching)
        if validity.valid:
            node = spot_node_from_state(state, target_seat, bunching=bunching)
            if node.spot_class.value in SPOT_CLASS_ORDER:
                return True, node.spot_class.value
    return False, "not_two_player_valid_postflop"


def infer_policy_spot_class(state: GameState, hero_seat: int) -> str | None:
    """
    Infer spot_class for policy encoding at inference.

    Returns None when the state is not a postflop HU decision for the hero.
    """
    from pokerbot.solver.spot_validity import is_two_player_valid_state

    if state.street not in _POSTFLOP:
        return None
    if state.street == Street.SHOWDOWN:
        return None

    if is_gate_isolated_hu_shape(state):
        validity = is_two_player_valid_state(state, hero_seat, bunching=False)
        return "hu_postflop_2p" if validity.valid else None

    for bunching, spot_class in ((True, "pseudo_hu_bunching"), (False, "hu_postflop_2p")):
        if is_two_player_valid_state(state, hero_seat, bunching=bunching).valid:
            return spot_class
    return None


def scope_ranges_for_class(spot_class: str) -> tuple[str, str]:
    return SCOPE_RANGE_PAIRS.get(spot_class, ("", ""))


def encode_state_for_policy(
    state: GameState,
    hero_seat: int,
    mapper: WEVAMapper | None = None,
    *,
    spot_class: str | None = None,
    require_scope: bool = False,
) -> dict[str, np.ndarray | int]:
    """
    Encode game state for policy forward pass with spot scope propagated.

    When ``require_scope`` is True, raises SystemExit if the resolved spot_class
    does not produce a non-zero spot_scope vector (training/eval precondition).
    """
    resolved = spot_class or infer_policy_spot_class(state, hero_seat) or ""
    oop_range, ip_range = scope_ranges_for_class(resolved)
    enc = encode_state(
        state,
        hero_seat,
        mapper=mapper,
        spot_class=resolved,
        oop_range=oop_range,
        ip_range=ip_range,
    )
    if require_scope:
        require_encoded_spot_scope(enc, context=f"hero_seat={hero_seat} spot_class={resolved!r}")
    return enc


def require_encoded_spot_scope(enc: dict[str, Any], *, context: str = "") -> None:
    """Fail loudly when spot_scope was not propagated (all-zero vector)."""
    scope = np.asarray(enc.get("spot_scope", []), dtype=np.float32).reshape(-1)
    if scope.size == 0 or float(scope.sum()) == 0.0:
        suffix = f" ({context})" if context else ""
        raise SystemExit(
            "Policy eval scope precondition FAILED: spot_scope is all-zero"
            f"{suffix}. Propagate spot_class through encode_state_for_policy; "
            "do not call encode_state() with empty spot_class at inference."
        )


def require_gate_policy_scope_encoding(*, device: str = "cpu") -> dict[str, object]:
    """
    Precondition: gate eval path must produce training-matched one-hot scope vectors.

    Verifies hu_postflop_2p and pseudo_hu_bunching encodings match training-time vectors.
    """
    from pokerbot.solver.spot_enumerator import (
        apply_flop_board,
        build_btn_open_bb_call,
        load_solver_config,
        pick_hero_hand,
    )
    from pokerbot.engine.cards import Card

    config = load_solver_config("config/solver_labels.yaml")
    board = [Card.from_str(b) for b in ["7c", "5d", "2h"]]
    state, oop, ip, _ = build_btn_open_bb_call(100.0)
    apply_flop_board(state, board)
    hero_hole = pick_hero_hand(config.hero_hands, board, __import__("random").Random(0))
    state.players[oop].hole = hero_hole

    vanilla_enc = encode_state_for_policy(
        state, oop, spot_class="hu_postflop_2p", require_scope=True
    )
    bunch_enc = encode_state_for_policy(
        state, oop, spot_class="pseudo_hu_bunching", require_scope=True
    )
    expected_vanilla = encode_spot_scope("hu_postflop_2p")
    expected_bunch = encode_spot_scope("pseudo_hu_bunching")

    checks = {
        "vanilla_matches_training": bool(
            np.allclose(vanilla_enc["spot_scope"], expected_vanilla)
        ),
        "bunching_matches_training": bool(
            np.allclose(bunch_enc["spot_scope"], expected_bunch)
        ),
        "vanilla_one_hot": vanilla_enc["spot_scope"].tolist(),
        "bunching_one_hot": bunch_enc["spot_scope"].tolist(),
        "gate_isolated_note": GATE_ISOLATED_HARNESS_SCOPE_NOTE,
        "training_spot_class_fraction": TRAINING_SPOT_CLASS_FRACTION,
        "gate_evaluates_fraction": {
            "hu_postflop_2p": 1.0,
            "pseudo_hu_bunching": 0.0,
        },
        "training_unevaluated_fraction": {
            "pseudo_hu_bunching": TRAINING_SPOT_CLASS_FRACTION["pseudo_hu_bunching"],
        },
    }
    if not checks["vanilla_matches_training"] or not checks["bunching_matches_training"]:
        raise SystemExit(
            "Policy eval scope precondition FAILED: encode_spot_scope one-hot mismatch "
            f"vs training. checks={checks}"
        )
    _ = device
    return checks
