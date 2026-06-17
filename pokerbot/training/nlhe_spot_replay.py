"""Reconstruct HU NLHE JAX states from solver Parquet rows (Gate A replay)."""

from __future__ import annotations

import json
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from pokerbot.data.sft_dataloader import SolverAnnotatedParquetDataset
from pokerbot.engine.actions import NUM_ACTIONS, Action, ActionType, action_to_index
from pokerbot.engine.cards import Card
from pokerbot.engine.game import GameState, Position
from pokerbot.engine.jax_engine import (
    ALL_IN,
    BET_BUCKET_START,
    CALL,
    CHECK,
    CHIP_UNIT,
    FOLD,
    MAX_BOARD,
    MAX_HISTORY,
    NUM_PLAYERS,
    SEAT_EMPTY,
    STREET_FLOP,
    init_hand,
)
from pokerbot.engine.jax_weva_encode import NLHEEncodeBundle, encode_nlhe_weva_bundle
from pokerbot.engine.nlhe_bridge import env_state_from_numpy, game_state_from_jax
from pokerbot.encoding.weva_encoder import WEVAMapper
from pokerbot.solver.spot_enumerator import (
    apply_flop_board,
    build_btn_open_bb_call,
    build_sb_vs_bb_limp_pot,
    seat_for_position,
)

BUTTON_SEAT = 3
# Matches solver Parquet / SFT encoding (BTN seat 3, BB seat 5).
HU_SOLVER_OCCUPANCY = (False, False, False, True, False, True)
HU_OCCUPANCY = HU_SOLVER_OCCUPANCY

_SPOT_PREFLOP: dict[str, str] = {
    "btn_vs_bb_flop": "btn_open_bb_call",
    "bunching_postflop": "btn_open_bb_call",
    "sb_vs_bb_flop": "fold_to_sb_bb_limp",
}


def _action_to_jax(action: Action) -> tuple[int, int]:
    """Return (jax_action_code, chip_amount)."""
    idx = action_to_index(action)
    chips = int(round(action.amount_bb * CHIP_UNIT))
    if idx == 0:
        return FOLD, 0
    if idx == 1:
        return CHECK, 0
    if idx == 2:
        return CALL, chips
    if idx == NUM_ACTIONS - 1:
        return ALL_IN, chips
    return BET_BUCKET_START + max(0, idx - 3), chips


def infer_stack_depth_bb(raw: dict[str, Any]) -> float:
    """Infer label stack depth from solver_request (prod6000 uses 40bb)."""
    req = raw.get("solver_request") or {}
    eff = float(req.get("effective_stack_bb", 100.0))
    # BTN open + BB call: both put 3.5bb in before flop decision.
    return eff + 3.5


def game_state_from_parquet_record(rec: dict[str, Any]) -> tuple[GameState, int]:
    """Rebuild Python ``GameState`` at the solver label decision point."""
    raw = json.loads(rec["raw_solver_json"]) if isinstance(rec.get("raw_solver_json"), str) else rec["raw_solver_json"]
    spot_id = str(raw.get("spot_id", ""))
    preflop = _SPOT_PREFLOP.get(spot_id)
    if preflop is None:
        raise ValueError(f"Unsupported spot_id for replay: {spot_id!r}")

    stack_bb = infer_stack_depth_bb(raw)
    if preflop == "btn_open_bb_call":
        state, oop, ip, _pot = build_btn_open_bb_call(stack_bb)
    else:
        state, oop, ip, _pot = build_sb_vs_bb_limp_pot(stack_bb)

    board = [Card.from_str(c) for c in raw["solver_request"]["board"]]
    apply_flop_board(state, board)

    hero_pos = str(raw.get("hero_position", "BB")).strip().upper()
    hero_seat = seat_for_position(BUTTON_SEAT, Position[hero_pos])
    hero_hole = tuple(Card.from_str(c) for c in raw["hero_hole"][:2])
    state.players[hero_seat].hole = hero_hole
    other = ip if hero_seat == oop else oop
    state.players[other].hole = (Card.from_str("2c"), Card.from_str("3d"))

    if hero_seat == ip:
        state.apply_action(Action(ActionType.CHECK), seat=oop)
    state.action_seat = hero_seat
    return state, hero_seat


def jax_state_from_game_state(
    gs: GameState,
    *,
    occupancy: tuple[bool, ...] = HU_OCCUPANCY,
    button: int = BUTTON_SEAT,
) -> Any:
    """Materialize a JAX ``EnvState`` snapshot from ``GameState`` (fixed cards, no deal)."""
    from pokerbot.engine.jax_engine import EnvState
    from pokerbot.engine.side_pots_jax import SEAT_ALLIN, SEAT_FOLDED, SEAT_LIVE

    occ = np.asarray(occupancy, dtype=bool)
    stacks = np.zeros(NUM_PLAYERS, dtype=np.int32)
    committed_street = np.zeros(NUM_PLAYERS, dtype=np.int32)
    committed_total = np.zeros(NUM_PLAYERS, dtype=np.int32)
    seat_status = np.full(NUM_PLAYERS, SEAT_EMPTY, dtype=np.int8)
    holes = np.full((NUM_PLAYERS, 2), -1, dtype=np.int32)
    used_cards: list[int] = []

    for seat in range(NUM_PLAYERS):
        p = gs.players[seat]
        stacks[seat] = int(round(p.stack_bb * CHIP_UNIT))
        committed_street[seat] = int(round(p.committed_street_bb * CHIP_UNIT))
        committed_total[seat] = int(round(p.committed_total_bb * CHIP_UNIT))
        if p.hole:
            holes[seat, 0] = p.hole[0].index
            holes[seat, 1] = p.hole[1].index
            used_cards.extend([holes[seat, 0], holes[seat, 1]])
        if not occ[seat]:
            # Out-of-hand seats still carry stack/features in solver labels (folded, not empty).
            seat_status[seat] = SEAT_FOLDED if p.folded else SEAT_EMPTY
            continue
        if p.folded:
            seat_status[seat] = SEAT_FOLDED
            continue
        seat_status[seat] = SEAT_ALLIN if p.all_in else SEAT_LIVE

    board = np.full(MAX_BOARD, -1, dtype=np.int32)
    for i, c in enumerate(gs.board):
        board[i] = c.index
        used_cards.append(c.index)

    deck = np.array([i for i in range(52) if i not in set(used_cards)], dtype=np.int32)
    if deck.size < 52 - len(used_cards):
        pad = np.array([i for i in range(52) if i not in deck and i not in used_cards], dtype=np.int32)
        deck = np.concatenate([deck, pad])[: 52 - len(used_cards)]
    while deck.size < 52:
        deck = np.concatenate([deck, np.array([deck.size % 52], dtype=np.int32)])

    pot_street = int(committed_street.sum())
    pot_dead = int(round(gs.pot_bb * CHIP_UNIT)) - pot_street
    pot_dead = max(pot_dead, 0)

    street_j = STREET_FLOP
    current_bet = int(round(gs.current_bet_bb * CHIP_UNIT))
    min_raise = int(CHIP_UNIT)  # BB

    hist_seat = np.full(MAX_HISTORY, -1, dtype=np.int32)
    hist_action = np.full(MAX_HISTORY, -1, dtype=np.int32)
    hist_amount = np.zeros(MAX_HISTORY, dtype=np.int32)
    for i, (seat, action) in enumerate(gs.action_history()):
        if i >= MAX_HISTORY:
            break
        code, amt = _action_to_jax(action)
        hist_seat[i] = seat
        hist_action[i] = code
        hist_amount[i] = amt

    acted = np.zeros(NUM_PLAYERS, dtype=bool)
    for seat, _ in gs.action_history():
        if 0 <= seat < NUM_PLAYERS:
            acted[seat] = True

    d = {
        "holes": holes,
        "board": board,
        "seat_status": seat_status,
        "stacks": stacks,
        "committed_street": committed_street,
        "committed_total": committed_total,
        "street": np.int32(street_j),
        "pot_dead": np.int32(pot_dead),
        "current_bet": np.int32(current_bet),
        "min_raise": np.int32(min_raise),
        "button": np.int32(button),
        "action_seat": np.int32(gs.action_seat),
        "last_aggressor": np.int32(gs.last_aggressor if gs.last_aggressor is not None else -1),
        "acted": acted,
        "last_raise_full": np.bool_(True),
        "deck": deck[:52].astype(np.int32),
        "deck_ptr": np.int32(len(used_cards)),
        "rng_key": np.array([0, 0], dtype=np.uint32),
        "hist_seat": hist_seat,
        "hist_action": hist_action,
        "hist_amount": hist_amount,
        "hist_len": np.int32(min(len(list(gs.action_history())), MAX_HISTORY)),
        "done": np.bool_(False),
        "rewards": np.zeros(NUM_PLAYERS, dtype=np.int32),
        "pending_deal": np.bool_(False),
    }
    return env_state_from_numpy(d)


def jax_state_from_parquet_record(rec: dict[str, Any]) -> tuple[Any, int]:
    gs, hero_seat = game_state_from_parquet_record(rec)
    return jax_state_from_game_state(gs, occupancy=HU_SOLVER_OCCUPANCY, button=BUTTON_SEAT), hero_seat


def encode_bundle_from_parquet_record(
    rec: dict[str, Any],
    mapper: WEVAMapper | None = None,
) -> tuple[NLHEEncodeBundle, np.ndarray]:
    """Encode via GameState replay (parity path for Gate A / SFT init checks)."""
    from pokerbot.encoding.spot_scope import fuse_position_scope
    from pokerbot.encoding.tensors import encode_state

    m = mapper or WEVAMapper.default()
    gs, hero_seat = game_state_from_parquet_record(rec)
    enc = encode_state(
        gs,
        hero_seat,
        mapper=m,
        spot_class=str(rec.get("spot_class", "")),
        oop_range=str(rec.get("oop_range", "")),
        ip_range=str(rec.get("ip_range", "")),
    )
    policy_position = fuse_position_scope(enc["position"], enc["spot_scope"])
    bundle = NLHEEncodeBundle(
        weva_id=int(enc["weva_id"]),
        cards=np.asarray(enc["cards"], dtype=np.float32),
        actions=np.asarray(enc["actions"], dtype=np.float32),
        position=np.asarray(enc["position"], dtype=np.float32),
        spot_scope=np.asarray(enc["spot_scope"], dtype=np.float32),
        policy_position=np.asarray(policy_position, dtype=np.float32),
    )
    mask = np.asarray(rec["legal_mask"], dtype=bool)
    return bundle, mask


def verify_parquet_encode_replay(
    rec: dict[str, Any],
    *,
    mapper: WEVAMapper | None = None,
    atol: float = 1e-4,
) -> dict[str, Any]:
    """Check GameState replay encoding matches stored Parquet tensors."""
    bundle, _mask = encode_bundle_from_parquet_record(rec, mapper=mapper)
    stored_cards = np.asarray(rec["cards"], dtype=np.float32)
    stored_actions = np.asarray(rec["actions"], dtype=np.float32)
    return {
        "weva_ok": int(bundle.weva_id) == int(rec["weva_bucket_id"]),
        "cards_ok": bool(np.allclose(bundle.cards, stored_cards, atol=atol)),
        "actions_ok": bool(np.allclose(bundle.actions, stored_actions, atol=atol)),
        "weva_id": int(bundle.weva_id),
        "stored_weva": int(rec["weva_bucket_id"]),
    }


def load_heldout_records(
    vanilla: str = "data/solver/heldout_prod600_vanilla.parquet",
    bunching: str = "data/solver/heldout_prod600_bunching.parquet",
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in (vanilla, bunching):
        ds = SolverAnnotatedParquetDataset(path)
        rows.extend(ds.records)
    return rows


def _spot_kwargs_from_record(rec: dict[str, Any]) -> dict[str, str]:
    return {
        "spot_class": str(rec.get("spot_class", "")),
        "oop_range": str(rec.get("oop_range", "")),
        "ip_range": str(rec.get("ip_range", "")),
    }


def check_gate_a_training_encode_parity_exhaustive(
    *,
    heldout_vanilla: str = "data/solver/heldout_prod600_vanilla.parquet",
    heldout_bunching: str = "data/solver/heldout_prod600_bunching.parquet",
    atol: float = 1e-4,
) -> dict[str, Any]:
    """
    Exhaustive parity: Gate A slow path (GameState replay) vs training fast path (JAX).

    Every row Gate A scores must emit the same ``NLHEEncodeBundle`` the trainer net consumes.
    """
    from pokerbot.engine.jax_weva_fast import default_weva_bucket_table, encode_nlhe_bundle_fast

    table = default_weva_bucket_table()
    mapper = WEVAMapper.default()
    mismatches: list[dict[str, Any]] = []
    replay_errors: list[dict[str, Any]] = []
    rows_checked = 0

    for source, path in (("vanilla", heldout_vanilla), ("bunching", heldout_bunching)):
        ds = SolverAnnotatedParquetDataset(path)
        for i in range(len(ds)):
            rec = ds.records[i]
            rows_checked += 1
            spot_kw = _spot_kwargs_from_record(rec)
            try:
                gate_a, _mask = encode_bundle_from_parquet_record(rec, mapper=mapper)
                state, hero_seat = jax_state_from_parquet_record(rec)
                train_fast = encode_nlhe_bundle_fast(
                    state, hero_seat, weva_table=table, **spot_kw
                )
                train_bridge = encode_nlhe_weva_bundle(state, hero_seat, mapper=mapper, **spot_kw)
            except Exception as exc:
                replay_errors.append(
                    {"source": source, "row": i, "error": str(exc)[:300]}
                )
                continue

            fp_gate = gate_a.fingerprint()
            fp_fast = train_fast.fingerprint()
            fp_bridge = train_bridge.fingerprint()
            tensor_ok = (
                gate_a.weva_id == train_fast.weva_id
                and np.allclose(gate_a.cards, train_fast.cards, atol=atol, rtol=0.0)
                and np.allclose(gate_a.actions, train_fast.actions, atol=atol, rtol=0.0)
                and np.allclose(
                    gate_a.policy_position, train_fast.policy_position, atol=atol, rtol=0.0
                )
            )
            if fp_gate != fp_fast or not tensor_ok:
                mismatches.append(
                    {
                        "source": source,
                        "row": i,
                        "spot_class": spot_kw["spot_class"],
                        "fp_gate_a": fp_gate,
                        "fp_train_fast": fp_fast,
                        "fp_jax_bridge": fp_bridge,
                        "weva_gate": gate_a.weva_id,
                        "weva_fast": train_fast.weva_id,
                        "fp_match_bridge_fast": fp_bridge == fp_fast,
                    }
                )

    return {
        "passed": rows_checked > 0 and not mismatches and not replay_errors,
        "rows_checked": rows_checked,
        "n_mismatches": len(mismatches),
        "n_replay_errors": len(replay_errors),
        "mismatches": mismatches,
        "replay_errors": replay_errors,
        "read": (
            "Gate A uses encode_bundle_from_parquet_record; training uses "
            "encode_nlhe_bundle_fast on jax_state_from_parquet_record. "
            "Fingerprints and tensors must match on every held-out row."
        ),
    }
