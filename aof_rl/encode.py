"""Fixed-size observation vectors for AoF RL (v1 + v2)."""

from __future__ import annotations

import numpy as np

from aof import AdvisorTeacher, NashTeacher
from aof_rl.env import AoFHandState
from aof_rl.spot_diagnostics import all_ins_before_hero, classify_decision_spot

OBS_VERSION_V1 = 1
OBS_VERSION_V2 = 2
OBS_VERSION_V3 = 3

# 13 ranks x 4 suits x 2 cards (binary presence per card slot)
NUM_CARD_BITS = 104
NUM_SEATS = 4

TEACHER_DIM = 5  # ev_primary, ev_alt, edge, p_fold, static-profile action index

# --- v1 layout (unchanged for checkpoint compatibility) ---
BASE_OBS_DIM_V1 = NUM_CARD_BITS + NUM_SEATS + 4
OBS_DIM_V1 = BASE_OBS_DIM_V1 + TEACHER_DIM

BASE_OBS_DIM = BASE_OBS_DIM_V1
OBS_DIM = OBS_DIM_V1

# --- v2 layout ---
TEACHER_PATH_LABELS: tuple[str, ...] = (
    "static_profile_open",
    "static_profile_clean_call",
    "static_profile_call_behind",
    "static_profile_facing_two",
    "static_profile_facing_two_conditional",
    "static_profile_facing_two_fallback",
    "static_profile_facing_two_supported",
    "static_profile_facing_two_missing",
    "static_profile_facing_three_missing",
    "static_profile_facing_three_supported",
    "static_profile_facing_three_fallback",
    "static_profile_unsupported",
    "unknown",
)
PROVENANCE_LABELS: tuple[str, ...] = (
    "solver_hu_dcfr",
    "hand_tuned",
    "static_profile",
    "conditional_call_chart",
    "fallback_static",
    "unknown",
)
TEACHER_PATH_DIM = len(TEACHER_PATH_LABELS)
PROVENANCE_DIM = len(PROVENANCE_LABELS)
_SHOVER_NONE_SLOT = NUM_SEATS  # index 4 in 5-wide one-hot

_CONTEXT_DIM_V2 = (
    NUM_CARD_BITS
    + NUM_SEATS
    + (NUM_SEATS + 1)
    + (NUM_SEATS + 1)
    + 2
    + NUM_SEATS
    + 6
)
BASE_OBS_DIM_V2 = _CONTEXT_DIM_V2 + TEACHER_PATH_DIM + PROVENANCE_DIM
OBS_DIM_V2 = BASE_OBS_DIM_V2 + TEACHER_DIM

RANKS = "23456789TJQKA"
SUITS = "cdhs"
_RANK_IDX = {r: i for i, r in enumerate(RANKS)}
_SUIT_IDX = {s: i for i, s in enumerate(SUITS)}


def obs_dim(version: int = OBS_VERSION_V1) -> int:
    if version == OBS_VERSION_V1:
        return OBS_DIM_V1
    if version == OBS_VERSION_V2:
        return OBS_DIM_V2
    if version == OBS_VERSION_V3:
        raise ValueError(
            "obs v3 is structured (ObsV3); use obs_v3_dims() from aof_rl.obs_v3"
        )
    raise ValueError(f"Unsupported obs version {version}")


def _card_indices(card: str) -> tuple[int, int]:
    return _RANK_IDX[card[0].upper()], _SUIT_IDX[card[1].lower()]


def _hand_label(cards: tuple[str, str]) -> str:
    r1, s1 = cards[0][0].upper(), cards[0][1].lower()
    r2, s2 = cards[1][0].upper(), cards[1][1].lower()
    i1, i2 = RANKS.index(r1), RANKS.index(r2)
    suited = s1 == s2
    if r1 == r2:
        return f"{r1}{r2}"
    hi, lo = (r1, r2) if i1 > i2 else (r2, r1)
    suf = "s" if suited else "o"
    return f"{hi}{lo}{suf}"


def _write_seat_one_hot(
    x: np.ndarray,
    offset: int,
    seat: str | None,
    seats: tuple[str, ...],
    *,
    none_slot: int,
) -> int:
    width = none_slot + 1
    if seat and seat in seats:
        idx = seats.index(seat)
        if idx < none_slot:
            x[offset + idx] = 1.0
            return offset + width
    x[offset + none_slot] = 1.0
    return offset + width


def _one_hot_index(labels: tuple[str, ...], key: str) -> int:
    try:
        return labels.index(key)
    except ValueError:
        return labels.index("unknown")


def encode_card_bits(state: AoFHandState, hero: str) -> np.ndarray:
    bits = np.zeros(NUM_CARD_BITS, dtype=np.float32)
    c1, c2 = state.holes[hero]
    for card in (c1, c2):
        ri, si = _card_indices(card)
        bits[ri * 4 + si] = 1.0
    return bits


def encode_base_v1(state: AoFHandState, hero: str) -> np.ndarray:
    """Observation without teacher (v1 layout)."""
    x = np.zeros(BASE_OBS_DIM_V1, dtype=np.float32)
    off = 0
    x[off : off + NUM_CARD_BITS] = encode_card_bits(state, hero)
    off += NUM_CARD_BITS
    if hero in state.seats:
        idx = state.seats.index(hero)
        if idx < NUM_SEATS:
            x[off + idx] = 1.0
    off += NUM_SEATS
    x[off] = state.table_size / 4.0
    x[off + 1] = state.stack_bb / 10.0
    x[off + 2] = 1.0 if state.facing_shove() else 0.0
    x[off + 3] = state.invested.get(hero, 0.0) / max(state.stack_bb, 0.5)
    return x


def encode_base_v2(
    state: AoFHandState,
    hero: str,
    *,
    teacher_path: str = "unknown",
    provenance: str = "unknown",
) -> np.ndarray:
    """Rich preflop context for multiway / facing-two spots."""
    x = np.zeros(BASE_OBS_DIM_V2, dtype=np.float32)
    off = 0
    x[off : off + NUM_CARD_BITS] = encode_card_bits(state, hero)
    off += NUM_CARD_BITS

    if hero in state.seats:
        idx = state.seats.index(hero)
        if idx < NUM_SEATS:
            x[off + idx] = 1.0
    off += NUM_SEATS

    prior = all_ins_before_hero(state, hero)
    first = prior[0] if prior else (state.shover if state.facing_shove() else None)
    second = prior[1] if len(prior) >= 2 else None
    off = _write_seat_one_hot(x, off, first, state.seats, none_slot=_SHOVER_NONE_SLOT)
    off = _write_seat_one_hot(x, off, second, state.seats, none_slot=_SHOVER_NONE_SLOT)

    n_all_in = sum(
        1 for s in state.seats if s in state.all_in and s not in state.folded
    )
    x[off] = n_all_in / max(state.table_size, 1)
    x[off + 1] = len(state.players_behind(hero)) / max(state.table_size, 1)
    off += 2

    stack = max(state.stack_bb, 0.5)
    for i, seat in enumerate(state.seats[:NUM_SEATS]):
        x[off + i] = state.invested.get(seat, 0.0) / stack
    off += NUM_SEATS

    x[off] = 1.0 if state.facing_shove() else 0.0
    x[off + 1] = 1.0 if len(prior) >= 2 else 0.0
    x[off + 2] = state.table_size / 4.0
    x[off + 3] = state.stack_bb / 10.0
    x[off + 4] = state.invested.get(hero, 0.0) / stack
    posted = state.invested.get(hero, 0.0)
    if hero == "SB":
        x[off + 5] = min(0.5, stack) / stack
    elif hero == "BB":
        x[off + 5] = min(1.0, stack) / stack
    else:
        x[off + 5] = posted / stack
    off += 6

    tp = _one_hot_index(TEACHER_PATH_LABELS, teacher_path)
    x[off + tp] = 1.0
    off += TEACHER_PATH_DIM
    pr = _one_hot_index(PROVENANCE_LABELS, provenance)
    x[off + pr] = 1.0
    return x


def encode_teacher(teacher: AdvisorTeacher | NashTeacher) -> np.ndarray:
    """Static-profile teacher features (not equilibrium)."""
    t = np.zeros(TEACHER_DIM, dtype=np.float32)
    t[0] = float(np.clip(teacher.ev_primary / 10.0, -2.0, 2.0))
    t[1] = float(np.clip(teacher.ev_alt / 10.0, -2.0, 2.0))
    t[2] = float(np.clip(teacher.edge_bb / 5.0, 0.0, 2.0))
    t[3] = float(teacher.p_fold if teacher.p_fold is not None else 0.0)
    t[4] = float(teacher.action_index)
    return t


def encode_obs_v1(state: AoFHandState, hero: str, teacher: AdvisorTeacher) -> np.ndarray:
    base = encode_base_v1(state, hero)
    return np.concatenate([base, encode_teacher(teacher)])


def encode_obs_v2(
    state: AoFHandState,
    hero: str,
    teacher: AdvisorTeacher,
    *,
    teacher_path: str = "unknown",
    provenance: str = "unknown",
) -> np.ndarray:
    base = encode_base_v2(
        state, hero, teacher_path=teacher_path, provenance=provenance
    )
    return np.concatenate([base, encode_teacher(teacher)])


def encode_obs(
    state: AoFHandState,
    hero: str,
    teacher: AdvisorTeacher | NashTeacher,
    *,
    version: int = OBS_VERSION_V1,
    teacher_path: str | None = None,
    provenance: str | None = None,
):
    if version == OBS_VERSION_V1:
        return encode_obs_v1(state, hero, teacher)
    if version == OBS_VERSION_V2:
        path = teacher_path or "unknown"
        prov = provenance or "unknown"
        return encode_obs_v2(
            state, hero, teacher, teacher_path=path, provenance=prov
        )
    if version == OBS_VERSION_V3:
        from aof_rl.encode_v3 import encode_obs_v3_from_hand

        return encode_obs_v3_from_hand(state, hero)
    raise ValueError(f"Unsupported obs version {version}")


def hand_label_for(state: AoFHandState, hero: str) -> str:
    return _hand_label(state.holes[hero])


def shover_seats_for_encode(state: AoFHandState, hero: str) -> tuple[str | None, str | None]:
    """First and second shover seats for tests / tooling."""
    prior = all_ins_before_hero(state, hero)
    first = prior[0] if prior else (state.shover if state.facing_shove() else None)
    second = prior[1] if len(prior) >= 2 else None
    return first, second


def default_teacher_path_for_state(state: AoFHandState, hero: str) -> str:
    from aof_rl.spot_diagnostics import teacher_path_for_spot

    kind = classify_decision_spot(state, hero)
    return teacher_path_for_spot(kind, teacher_version=1)
