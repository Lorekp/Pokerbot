"""Structured tensor encoding from PokerBench CSV rows."""

from __future__ import annotations

import ast
import re
from typing import Any

import numpy as np

from pokerbot.engine.actions import NUM_ACTIONS, action_to_index, parse_pokerbench_output
from pokerbot.engine.cards import Card, parse_cards
from pokerbot.encoding.tensors import (
    CARD_PLANES,
    CARD_RANKS,
    MAX_ACTIONS_PER_STREET,
    NUM_CARD_SLOTS,
    NUM_PLAYERS,
    NUM_STREETS,
    SPEC,
    encode_card_plane,
)

POSITION_MAP = {
    "UTG": 0,
    "HJ": 1,
    "CO": 2,
    "BTN": 3,
    "SB": 4,
    "BB": 5,
    "IP": 3,
    "OOP": 5,
}

_PREFLOP_TOKEN = re.compile(
    r"(UTG|HJ|CO|BTN|SB|BB)/([\d.]+bb|call|fold|check|allin|all in|raise|bet[\d.]*)",
    re.I,
)
_POSTFLOP_TOKEN = re.compile(
    r"(OOP|IP|UTG|HJ|CO|BTN|SB|BB)_(CHECK|CALL|FOLD|BET|RAISE|ALLIN|ALL_IN)(?:_([\d.]+))?",
    re.I,
)


def position_one_hot(name: str) -> np.ndarray:
    pos = np.zeros(NUM_PLAYERS, dtype=np.float32)
    idx = POSITION_MAP.get(name.strip().upper(), 0)
    pos[idx] = 1.0
    return pos


def encode_holding(holding: str) -> np.ndarray:
    tensor = np.zeros(SPEC.card_shape, dtype=np.float32)
    if not holding:
        return tensor
    cards = parse_cards(holding.replace(" ", ""))
    for i, card in enumerate(cards[:2]):
        tensor[i] = encode_card_plane(card)
    return tensor


def encode_board(flop: str, turn: str = "", river: str = "") -> np.ndarray:
    tensor = np.zeros(SPEC.card_shape, dtype=np.float32)
    board_str = (flop or "") + (turn or "") + (river or "")
    if not board_str:
        return tensor
    cards = parse_cards(board_str.replace(" ", ""))
    for i, card in enumerate(cards[:5]):
        slot = 2 + i
        if slot < NUM_CARD_SLOTS:
            tensor[slot] = encode_card_plane(card)
    return tensor


def merge_card_tensors(hole: np.ndarray, board: np.ndarray) -> np.ndarray:
    out = hole.copy()
    for slot in range(2, NUM_CARD_SLOTS):
        if board[slot].any():
            out[slot] = board[slot]
    return out


def _pos_to_seat(name: str) -> int:
    return POSITION_MAP.get(name.upper(), 0)


def encode_prev_line(prev_line: str, pot_size: float) -> np.ndarray:
    tensor = np.zeros(SPEC.action_shape, dtype=np.float32)
    if not prev_line:
        tensor[:, :, 6] = pot_size / 200.0
        return tensor

    tokens = prev_line.split("/")
    i = 0
    street = 0
    while i < len(tokens):
        pos_name = tokens[i].upper()
        if pos_name not in POSITION_MAP:
            i += 1
            continue
        seat = _pos_to_seat(pos_name)
        i += 1
        if i >= len(tokens):
            break
        act = tokens[i].lower().replace("bb", "").strip()
        feat = np.zeros(MAX_ACTIONS_PER_STREET, dtype=np.float32)
        feat[0] = 1.0
        amount = 0.0
        if act in ("fold",):
            feat[1] = 1.0
        elif act in ("call", "check"):
            feat[2] = 1.0
        elif act in ("allin", "all in", "all-in"):
            feat[3] = 1.0
            amount = 100.0
        else:
            try:
                amount = float(act)
                feat[3] = 1.0
            except ValueError:
                feat[2] = 1.0
        feat[4] = min(amount / 100.0, 1.0)
        feat[6] = seat / NUM_PLAYERS
        tensor[seat, street, :] = np.maximum(tensor[seat, street, :], feat)
        i += 1

    tensor[:, :, 6] = pot_size / 200.0
    return tensor


def encode_postflop_action(postflop_action: str, pot_size: float, evaluation_at: str = "") -> np.ndarray:
    tensor = np.zeros(SPEC.action_shape, dtype=np.float32)
    street_map = {"flop": 1, "turn": 2, "river": 3, "preflop": 0}
    current_street = street_map.get((evaluation_at or "flop").lower(), 1)

    if postflop_action:
        for part in postflop_action.split("/"):
            part = part.strip()
            if part.lower().startswith("dealcards"):
                if current_street < 3:
                    current_street += 1
                continue
            m = _POSTFLOP_TOKEN.match(part.replace(" ", "_"))
            if not m:
                continue
            pos_name, act_name, amount_str = m.group(1), m.group(2), m.group(3)
            seat = _pos_to_seat(pos_name)
            feat = np.zeros(MAX_ACTIONS_PER_STREET, dtype=np.float32)
            feat[0] = 1.0
            act_u = act_name.upper()
            amount = float(amount_str) if amount_str else 0.0
            if act_u == "FOLD":
                feat[1] = 1.0
            elif act_u in ("CHECK", "CALL"):
                feat[2] = 1.0
            else:
                feat[3] = 1.0
            feat[4] = min(amount / 100.0, 1.0)
            feat[6] = seat / NUM_PLAYERS
            st = min(current_street, NUM_STREETS - 1)
            tensor[seat, st, :] = np.maximum(tensor[seat, st, :], feat)

    tensor[:, :, 6] = pot_size / 200.0
    return tensor


def _parse_available_moves(raw: Any) -> list[str]:
    if raw is None or raw == "":
        return []
    if isinstance(raw, list):
        return [str(x) for x in raw]
    text = str(raw).strip()
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, list):
            return [str(x) for x in parsed]
    except (SyntaxError, ValueError):
        pass
    return [text]


def available_moves_to_legal_mask(moves: list[str]) -> np.ndarray:
    mask = np.zeros(NUM_ACTIONS, dtype=bool)
    if not moves:
        mask[:] = True
        return mask

    for move in moves:
        m = move.strip().lower()
        try:
            if m.startswith("bet "):
                action = parse_pokerbench_output(f"bet {m.split()[1]}")
            elif m.startswith("raise "):
                action = parse_pokerbench_output(f"raise {m.split()[1]}")
            elif m.startswith("bet") and " " not in m:
                action = parse_pokerbench_output(f"bet {m.replace('bet', '').strip()}")
            else:
                action = parse_pokerbench_output(m)
            mask[action_to_index(action)] = True
        except ValueError:
            if "check" in m:
                mask[1] = True
            if "call" in m:
                mask[2] = True
            if "fold" in m:
                mask[0] = True
            if "all" in m:
                mask[NUM_ACTIONS - 1] = True
            if "bet" in m or "raise" in m:
                mask[3:10] = True

    if not mask.any():
        mask[:] = True
    return mask


def normalize_decision_label(decision: str) -> str:
    d = decision.strip()
    if not d:
        return "fold"
    lower = d.lower()
    if lower == "allin":
        return "all in"
    if lower.endswith("bb"):
        amount = float(lower[:-2])
        return f"raise {amount:g}"
    if d[0].isupper() and " " in d:
        parts = d.split(None, 1)
        return f"{parts[0].lower()} {parts[1]}"
    if d[0].isupper():
        return lower
    if lower.startswith("bet "):
        return f"bet {float(lower.split()[1]):g}"
    if lower.startswith("raise "):
        return f"raise {float(lower.split()[1]):g}"
    return lower


def decision_to_label_index(decision: str) -> int:
    d = normalize_decision_label(decision)
    try:
        return action_to_index(parse_pokerbench_output(d))
    except ValueError:
        if d.startswith("bet"):
            return action_to_index(parse_pokerbench_output(d.replace("bet", "bet ", 1) if " " not in d else d))
        raise


def encode_preflop_row(row: dict[str, Any]) -> dict[str, np.ndarray]:
    hole = encode_holding(str(row.get("hero_holding", "")))
    actions = encode_prev_line(str(row.get("prev_line", "")), float(row.get("pot_size", 1.5) or 1.5))
    position = position_one_hot(str(row.get("hero_pos", "UTG")))
    moves = _parse_available_moves(row.get("available_moves"))
    legal = available_moves_to_legal_mask(moves)
    label = decision_to_label_index(str(row.get("correct_decision", "fold")))
    return {
        "cards": hole,
        "actions": actions,
        "position": position,
        "legal_mask": legal,
        "label": label,
    }


def encode_postflop_row(row: dict[str, Any]) -> dict[str, np.ndarray]:
    hole = encode_holding(str(row.get("holding", "")))
    board = encode_board(
        str(row.get("board_flop", "")),
        str(row.get("board_turn", "") or ""),
        str(row.get("board_river", "") or ""),
    )
    cards = merge_card_tensors(hole, board)
    pot = float(row.get("pot_size", 10.0) or 10.0)
    preflop = encode_prev_line(str(row.get("preflop_action", "")), pot * 0.3)
    postflop = encode_postflop_action(
        str(row.get("postflop_action", "")),
        pot,
        str(row.get("evaluation_at", "Flop")),
    )
    actions = np.maximum(preflop, postflop)
    position = position_one_hot(str(row.get("hero_position", "IP")))
    moves = _parse_available_moves(row.get("available_moves"))
    legal = available_moves_to_legal_mask(moves)
    decision = str(row.get("correct_decision", "check"))
    label = decision_to_label_index(decision.lower() if decision[0].isupper() else decision)
    return {
        "cards": cards,
        "actions": actions,
        "position": position,
        "legal_mask": legal,
        "label": label,
    }


def encode_csv_row(row: dict[str, Any], stage: str) -> dict[str, np.ndarray]:
    if stage == "preflop":
        return encode_preflop_row(row)
    return encode_postflop_row(row)
