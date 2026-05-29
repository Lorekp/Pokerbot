"""Obs v3 encoder: role-canonical seat tokens + chronological history."""

from __future__ import annotations

import numpy as np
import torch

from aof_rl.env import ACTION_ALL_IN, ACTION_FOLD, AoFHandState
from aof_rl.game_state import (
    ROLE_INDEX,
    ROLE_ORDER,
    Action,
    AoFGameState,
    SeatRole,
    SeatState,
    action_order_labels,
    hand_label_index,
    normalize_hand_state,
    seat_label_for_role,
    seats_by_id,
)
from aof_rl.obs_v3 import (
    GLOBAL_FEAT_DIM,
    HERO_HAND_DIM,
    HISTORY_TOKEN_DIM,
    MAX_HISTORY,
    MAX_SEATS,
    NUM_ACTIONS,
    SEAT_TOKEN_DIM,
    ObsV3,
)
from aof_rl.spot_diagnostics import classify_decision_spot

ACT_STATE_LABELS = ("pending", "folded", "all_in", "acted_fold", "acted_all_in")
SPOT_KIND_SCALAR = {
    "open": 0.0,
    "call_clean": 0.2,
    "call_behind": 0.4,
    "facing_two": 0.6,
    "facing_three": 0.8,
    "terminal_3way": 1.0,
    "terminal_4way": 1.0,
    "other": 0.5,
}


def _write_one_hot(x: np.ndarray, off: int, size: int, idx: int) -> int:
    if 0 <= idx < size:
        x[off + idx] = 1.0
    return off + size


def encode_hero_hand_features(hero_cards: tuple[int, int]) -> np.ndarray:
    """40-dim: per-card rank OH (26) + meta (5) + 169 index + norms + suit OH (8)."""
    x = np.zeros(HERO_HAND_DIM, dtype=np.float32)
    c1, c2 = hero_cards
    r1, s1 = c1 // 4, c1 % 4
    r2, s2 = c2 // 4, c2 % 4
    if 0 <= r1 < 13:
        x[r1] = 1.0
    if 0 <= r2 < 13:
        x[13 + r2] = 1.0
    off = 26
    suited = 1.0 if s1 == s2 else 0.0
    pair = 1.0 if r1 == r2 else 0.0
    hi, lo = max(r1, r2), min(r1, r2)
    gap = (hi - lo - 1) / 12.0 if hi > lo else 0.0
    x[off] = suited
    x[off + 1] = pair
    x[off + 2] = hi / 12.0
    x[off + 3] = lo / 12.0
    x[off + 4] = gap
    off = 31
    x[off] = hand_label_index(hero_cards) / 168.0
    x[off + 1] = c1 / 51.0
    x[off + 2] = c2 / 51.0
    x[34] = s1 / 3.0
    x[35] = s2 / 3.0
    x[36] = 1.0 if abs(r1 - r2) == 1 else 0.0
    x[37] = 1.0 if hi >= 12 else 0.0
    return x


def _seat_state_for_label(
    label: str | None,
    action_order: tuple[str, ...],
    by_id: dict[int, SeatState],
) -> SeatState | None:
    if label is None:
        return None
    try:
        sid = action_order.index(label)
    except ValueError:
        return None
    return by_id.get(sid)


def encode_seat_token(
    seat: SeatState | None,
    *,
    stack_bb: float,
    role: SeatRole,
) -> np.ndarray:
    x = np.zeros(SEAT_TOKEN_DIM, dtype=np.float32)
    if seat is None:
        return x
    off = 0
    off = _write_one_hot(x, off, len(ROLE_ORDER), ROLE_INDEX[role])
    off = _write_one_hot(x, off, 5, min(seat.shover_rank, 4))
    act_idx = (
        ACT_STATE_LABELS.index(seat.state)
        if seat.state in ACT_STATE_LABELS
        else 0
    )
    off = _write_one_hot(x, off, len(ACT_STATE_LABELS), act_idx)
    denom = max(stack_bb, 0.5)
    x[off] = seat.committed_bb / denom
    x[off + 1] = seat.stack_remaining_bb / denom
    x[off + 2] = seat.blind_posted_bb / denom
    off += 3
    x[off] = 1.0 if seat.is_hero else 0.0
    off += 1
    for bit in range(4):
        if seat.eligible_pot_mask & (1 << bit):
            x[off + bit] = 1.0
    off += 4
    x[off] = seat.seat_id / max(MAX_SEATS - 1, 1)
    return x


def encode_history_token(
    seat_id: int,
    action: Action,
    *,
    shover_order: int,
    committed_frac: float,
    players_left_norm: float,
    role: SeatRole,
) -> np.ndarray:
    x = np.zeros(HISTORY_TOKEN_DIM, dtype=np.float32)
    off = 0
    off = _write_one_hot(x, off, len(ROLE_ORDER), ROLE_INDEX[role])
    act_idx = 0 if action == "FOLD" else 1
    off = _write_one_hot(x, off, 2, act_idx)
    x[off] = shover_order / 3.0
    x[off + 1] = committed_frac
    x[off + 2] = players_left_norm
    off += 3
    x[off] = seat_id / max(MAX_SEATS - 1, 1)
    return x


def _prior_all_ins(game: AoFGameState) -> list[tuple[int, Action]]:
    hero_id = game.hero_seat_id
    return [(sid, act) for sid, act in game.action_history if sid != hero_id and act == "ALL_IN"]


def encode_global_features(
    game: AoFGameState,
    *,
    n_prior_shovers: int,
    players_behind: int,
    facing_shove: bool,
    spot_kind: str,
) -> np.ndarray:
    x = np.zeros(GLOBAL_FEAT_DIM, dtype=np.float32)
    stack = max(game.effective_stack_bb, 0.5)
    x[0] = game.table_size / 4.0
    x[1] = stack / 10.0
    x[2] = game.small_blind_bb / stack
    x[3] = game.big_blind_bb / stack
    x[4] = n_prior_shovers / 3.0
    x[5] = 1.0 if facing_shove else 0.0
    x[6] = 1.0 if n_prior_shovers >= 2 else 0.0
    x[7] = 1.0 if n_prior_shovers >= 3 else 0.0
    x[8] = players_behind / max(game.table_size, 1)
    x[9] = 1.0 if game.equal_stacks else 0.0
    x[10] = 1.0 if game.side_pots_present else 0.0
    x[11] = game.flat_rake_bb / max(stack, 0.5)
    x[12] = game.pct_rake
    x[13] = game.rake_cap_bb / max(stack, 0.5)
    by_id = seats_by_id(game)
    hero = by_id.get(game.hero_seat_id) or next(
        (s for s in game.seats if s.is_hero), None
    )
    if hero is not None:
        x[14] = hero.committed_bb / stack
        x[15] = hero.blind_posted_bb / stack
    x[16] = SPOT_KIND_SCALAR.get(spot_kind, 0.5)
    x[17] = game.hero_seat_id / max(game.table_size - 1, 1)
    return x


def _role_for_history_seat(
    seat_id: int, action_order: tuple[str, ...]
) -> SeatRole:
    if seat_id < 0 or seat_id >= len(action_order):
        return "first_to_act"
    from aof_rl.game_state import primary_role_for_label

    return primary_role_for_label(action_order[seat_id], action_order)


def encode_obs_v3(game: AoFGameState, *, device: torch.device | None = None) -> ObsV3:
    if game.side_pots_present and not game.equal_stacks:
        from aof_rl.game_state import AoFEncodeError

        raise AoFEncodeError("side pots with unequal stacks are not supported")

    action_order = action_order_labels(game)
    by_id = seats_by_id(game)
    stack = max(game.effective_stack_bb, 0.5)

    hero_hand = encode_hero_hand_features(game.hero_cards)

    prior_all_ins = _prior_all_ins(game)
    n_prior = len(prior_all_ins)
    players_behind = max(0, game.table_size - game.hero_seat_id - 1)
    facing = n_prior > 0 or any(a == "ALL_IN" for _, a in game.action_history)

    # spot kind from synthetic hand-less classify: use history shape
    if n_prior >= 3:
        spot_kind = "facing_three"
    elif n_prior >= 2:
        spot_kind = "facing_two"
    elif n_prior == 1:
        spot_kind = "call_clean"
    else:
        spot_kind = "open"

    global_feats = encode_global_features(
        game,
        n_prior_shovers=n_prior,
        players_behind=players_behind,
        facing_shove=facing,
        spot_kind=spot_kind,
    )

    seat_tokens = np.zeros((MAX_SEATS, SEAT_TOKEN_DIM), dtype=np.float32)
    seat_mask = np.zeros(MAX_SEATS, dtype=np.float32)
    for role in ROLE_ORDER:
        slot = ROLE_INDEX[role]
        label = seat_label_for_role(role, action_order)
        occupant = _seat_state_for_label(label, action_order, by_id)
        if label is not None and occupant is not None:
            seat_mask[slot] = 1.0
            seat_tokens[slot] = encode_seat_token(
                occupant, stack_bb=stack, role=role
            )

    history_tokens = np.zeros((MAX_HISTORY, HISTORY_TOKEN_DIM), dtype=np.float32)
    history_mask = np.zeros(MAX_HISTORY, dtype=np.float32)
    shover_order = 0
    for i, (seat_id, action) in enumerate(prior_all_ins[:MAX_HISTORY]):
        shover_order += 1
        seat = by_id.get(seat_id)
        committed_frac = (seat.committed_bb / stack) if seat else 1.0
        players_left = max(0, game.table_size - seat_id - 1) / max(game.table_size, 1)
        role = _role_for_history_seat(seat_id, action_order)
        history_mask[i] = 1.0
        history_tokens[i] = encode_history_token(
            seat_id,
            action,
            shover_order=shover_order,
            committed_frac=committed_frac,
            players_left_norm=players_left,
            role=role,
        )

    legal = np.zeros(NUM_ACTIONS, dtype=np.float32)
    legal[ACTION_FOLD] = 1.0
    legal[ACTION_ALL_IN] = 1.0

    def _t(a: np.ndarray, device: torch.device | None = None) -> torch.Tensor:
        t = torch.from_numpy(a.astype(np.float32, copy=False))
        if device is not None and device.type != "cpu":
            return t.to(device, non_blocking=True)
        return t

    return ObsV3(
        hero_hand=_t(hero_hand, device),
        global_feats=_t(global_feats, device),
        seat_tokens=_t(seat_tokens, device),
        seat_mask=_t(seat_mask, device),
        history_tokens=_t(history_tokens, device),
        history_mask=_t(history_mask, device),
        legal_mask=_t(legal, device),
    )


def encode_obs_v3_from_hand(
    state: AoFHandState, hero: str, *, device: torch.device | None = None
) -> ObsV3:
    game = normalize_hand_state(state, hero)
    obs = encode_obs_v3(game, device=device)
    kind = classify_decision_spot(state, hero)
    gf = obs.global_feats.clone()
    gf[16] = SPOT_KIND_SCALAR.get(kind, 0.5)
    return ObsV3(
        hero_hand=obs.hero_hand,
        global_feats=gf,
        seat_tokens=obs.seat_tokens,
        seat_mask=obs.seat_mask,
        history_tokens=obs.history_tokens,
        history_mask=obs.history_mask,
        legal_mask=obs.legal_mask,
    )
