"""Sequence encoding for MADT opponent transformer trajectories."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any

import numpy as np
import torch

from pokerbot.engine.actions import NUM_ACTIONS, Action, action_to_index
from pokerbot.engine.cards import Card
from pokerbot.engine.game import NUM_PLAYERS, GameState, Street
from pokerbot.encoding.tensors import encode_card_plane
from pokerbot.encoding.weva_encoder import enumerate_hole_combos

MAX_SEQ_LEN = 256
STATE_FEATURE_DIM = 304
ACTION_FEATURE_DIM = 18  # seat one-hot (6) + action one-hot (11) + amount (1)
REWARD_FEATURE_DIM = 1
TOKEN_FEATURE_DIM = max(STATE_FEATURE_DIM, ACTION_FEATURE_DIM, REWARD_FEATURE_DIM)
NUM_BELIEF_COMBOS = 1326
PAD_TOKEN_ID = 0


class TokenType(IntEnum):
    PAD = 0
    STATE = 1
    ACTION = 2
    REWARD = 3


# Precomputed combo index lookup (canonical sorted card indices).
_COMBO_LIST = enumerate_hole_combos()
_COMBO_INDEX: dict[tuple[int, int], int] = {
    tuple(sorted(c.index for c in combo)): i for i, combo in enumerate(_COMBO_LIST)
}


def cards_to_combo_index(hole: tuple[Card, Card]) -> int:
    """Map unordered hole cards to combo index in [0, 1326)."""
    key = tuple(sorted(c.index for c in hole))
    return _COMBO_INDEX[key]


def combo_index_to_cards(idx: int) -> tuple[Card, Card]:
    """Reverse lookup for debugging/tests."""
    return _COMBO_LIST[idx]


def build_card_removal_mask(
    hero_hole: tuple[Card, Card] | None,
    board: list[Card],
    device: torch.device | None = None,
) -> torch.Tensor:
    """Mask impossible hole-card combos given known cards (True = legal)."""
    dead = {c.index for c in board}
    if hero_hole:
        dead.update(c.index for c in hero_hole)
    mask = torch.ones(NUM_BELIEF_COMBOS, dtype=torch.bool, device=device)
    for i, combo in enumerate(_COMBO_LIST):
        if combo[0].index in dead or combo[1].index in dead:
            mask[i] = False
    return mask


def _street_onehot(street: Street) -> np.ndarray:
    idx = {
        Street.PREFLOP: 0,
        Street.FLOP: 1,
        Street.TURN: 2,
        Street.RIVER: 3,
        Street.SHOWDOWN: 4,
    }.get(street, 4)
    out = np.zeros(5, dtype=np.float32)
    out[idx] = 1.0
    return out


def encode_public_state(state: GameState, actor_seat: int, hero_seat: int) -> np.ndarray:
    """Public observation at a decision point (no opponent private cards)."""
    parts: list[np.ndarray] = []
    parts.append(_street_onehot(state.street))
    parts.append(np.array([state.pot_bb / 200.0, state.current_bet_bb / 100.0], dtype=np.float32))

    seat_feats = np.zeros(NUM_PLAYERS * 5, dtype=np.float32)
    for i, p in enumerate(state.players):
        base = i * 5
        seat_feats[base] = p.stack_bb / 100.0
        seat_feats[base + 1] = p.committed_street_bb / max(state.current_bet_bb, 1.0)
        seat_feats[base + 2] = float(p.folded)
        seat_feats[base + 3] = float(p.all_in)
        seat_feats[base + 4] = float(int(p.position)) / 5.0
    parts.append(seat_feats)

    board_flat = np.zeros(5 * 4 * 13, dtype=np.float32)
    for bi, card in enumerate(state.board[:5]):
        plane = encode_card_plane(card).flatten()
        board_flat[bi * 52 : (bi + 1) * 52] = plane
    parts.append(board_flat)

    actor_oh = np.zeros(NUM_PLAYERS, dtype=np.float32)
    actor_oh[actor_seat % NUM_PLAYERS] = 1.0
    parts.append(actor_oh)

    rel = (actor_seat - hero_seat) % NUM_PLAYERS
    parts.append(np.array([rel / NUM_PLAYERS], dtype=np.float32))

    feat = np.concatenate(parts)
    assert feat.shape[0] == STATE_FEATURE_DIM, feat.shape
    return feat


def encode_action_token(seat: int, action: Action) -> np.ndarray:
    """Action event features."""
    feat = np.zeros(ACTION_FEATURE_DIM, dtype=np.float32)
    feat[seat % NUM_PLAYERS] = 1.0
    idx = action_to_index(action)
    feat[NUM_PLAYERS + idx] = 1.0
    feat[NUM_PLAYERS + NUM_ACTIONS] = min(action.amount_bb / 100.0, 1.0)
    return feat


def encode_reward_token(reward_bb: float) -> np.ndarray:
    return np.array([np.clip(reward_bb / 10.0, -1.0, 1.0)], dtype=np.float32)


def _pad_feature(feat: np.ndarray, dim: int = TOKEN_FEATURE_DIM) -> np.ndarray:
    out = np.zeros(dim, dtype=np.float32)
    out[: feat.shape[0]] = feat
    return out


@dataclass
class TrajectoryRecord:
    """Single hand trajectory with labels for opponent transformer training."""

    token_types: list[int] = field(default_factory=list)
    token_features: list[np.ndarray] = field(default_factory=list)
    actor_seats: list[int] = field(default_factory=list)
    hero_seat: int = 0
    belief_targets: dict[int, int] = field(default_factory=dict)
    policy_targets: dict[int, int] = field(default_factory=dict)
    cards_known: dict[int, bool] = field(default_factory=dict)
    action_known: dict[int, bool] = field(default_factory=dict)
    showdown: bool = False


@dataclass
class HandTrajectoryBuilder:
    """Accumulates causal (state, action, reward) tokens during a hand."""

    hero_seat: int
    max_len: int = MAX_SEQ_LEN
    token_types: list[int] = field(default_factory=list)
    token_features: list[np.ndarray] = field(default_factory=list)
    actor_seats: list[int] = field(default_factory=list)
    _prev_stacks: dict[int, float] = field(default_factory=dict)
    _pending_state: np.ndarray | None = field(default=None, repr=False)
    _pending_actor: int = field(default=0, repr=False)
    belief_targets: dict[int, int] = field(default_factory=dict)
    policy_targets: dict[int, int] = field(default_factory=dict)
    cards_known: dict[int, bool] = field(default_factory=dict)
    action_known: dict[int, bool] = field(default_factory=dict)

    def reset(self, hero_seat: int | None = None) -> None:
        if hero_seat is not None:
            self.hero_seat = hero_seat
        self.token_types.clear()
        self.token_features.clear()
        self.actor_seats.clear()
        self._prev_stacks.clear()
        self._pending_state = None
        self._pending_actor = 0
        self.belief_targets.clear()
        self.policy_targets.clear()
        self.cards_known.clear()
        self.action_known.clear()

    def _append(self, token_type: TokenType, feat: np.ndarray, actor_seat: int) -> None:
        if len(self.token_types) >= self.max_len:
            return
        self.token_types.append(int(token_type))
        self.token_features.append(_pad_feature(feat))
        self.actor_seats.append(actor_seat)

    def on_decision(self, state: GameState, actor_seat: int) -> None:
        """Emit state token when a player is to act."""
        if len(self.token_types) >= self.max_len:
            return
        self._pending_state = encode_public_state(state, actor_seat, self.hero_seat)
        self._pending_actor = actor_seat
        self._append(TokenType.STATE, self._pending_state, actor_seat)

    def on_action(self, state: GameState, seat: int, action: Action) -> None:
        """Emit action and reward tokens after a player acts."""
        if len(self.token_types) >= self.max_len:
            return
        self._append(TokenType.ACTION, encode_action_token(seat, action), seat)

        prev = self._prev_stacks.get(seat, state.players[seat].stack_bb + action.amount_bb)
        reward = state.players[seat].stack_bb - prev
        self._prev_stacks[seat] = state.players[seat].stack_bb
        self._append(TokenType.REWARD, encode_reward_token(reward), seat)

        if seat != self.hero_seat:
            self.policy_targets[seat] = action_to_index(action)
            self.action_known[seat] = True

    def init_stacks(self, state: GameState) -> None:
        for p in state.players:
            self._prev_stacks[p.seat] = p.stack_bb

    def record_showdown(self, state: GameState) -> None:
        """Record belief targets for opponents whose hole cards are visible."""
        for p in state.players:
            if p.seat == self.hero_seat or p.hole is None or p.folded:
                continue
            self.belief_targets[p.seat] = cards_to_combo_index(p.hole)
            self.cards_known[p.seat] = True

    def finalize(self) -> TrajectoryRecord:
        return TrajectoryRecord(
            token_types=list(self.token_types),
            token_features=list(self.token_features),
            actor_seats=list(self.actor_seats),
            hero_seat=self.hero_seat,
            belief_targets=dict(self.belief_targets),
            policy_targets=dict(self.policy_targets),
            cards_known=dict(self.cards_known),
            action_known=dict(self.action_known),
        )


def opponent_seats(hero_seat: int) -> list[int]:
    """Five non-hero seats in fixed order."""
    return [s for s in range(NUM_PLAYERS) if s != hero_seat]


def trajectory_to_batch(
    trajectories: list[TrajectoryRecord],
    device: torch.device | None = None,
) -> dict[str, torch.Tensor]:
    """Dynamic padding and causal attention mask construction."""
    if not trajectories:
        raise ValueError("Empty trajectory list")

    batch_size = len(trajectories)
    max_len = min(max(len(t.token_types) for t in trajectories), MAX_SEQ_LEN)

    token_types = torch.zeros(batch_size, max_len, dtype=torch.long, device=device)
    token_features = torch.zeros(batch_size, max_len, TOKEN_FEATURE_DIM, dtype=torch.float32, device=device)
    attn_mask = torch.zeros(batch_size, max_len, dtype=torch.bool, device=device)
    actor_seats = torch.zeros(batch_size, max_len, dtype=torch.long, device=device)
    hero_seats = torch.tensor([t.hero_seat for t in trajectories], dtype=torch.long, device=device)

    belief_targets = torch.full((batch_size, 5), -1, dtype=torch.long, device=device)
    policy_targets = torch.full((batch_size, 5), -1, dtype=torch.long, device=device)
    cards_known_mask = torch.zeros(batch_size, 5, dtype=torch.float32, device=device)
    action_known_mask = torch.zeros(batch_size, 5, dtype=torch.float32, device=device)
    opp_seat_map = torch.zeros(batch_size, 5, dtype=torch.long, device=device)

    for b, traj in enumerate(trajectories):
        seq_len = min(len(traj.token_types), max_len)
        token_types[b, :seq_len] = torch.tensor(traj.token_types[:seq_len], dtype=torch.long)
        feats = np.stack(traj.token_features[:seq_len])
        token_features[b, :seq_len] = torch.from_numpy(feats)
        attn_mask[b, :seq_len] = True
        actor_seats[b, :seq_len] = torch.tensor(traj.actor_seats[:seq_len], dtype=torch.long)

        opps = opponent_seats(traj.hero_seat)
        for oi, seat in enumerate(opps):
            opp_seat_map[b, oi] = seat
            if seat in traj.belief_targets:
                belief_targets[b, oi] = traj.belief_targets[seat]
            if seat in traj.policy_targets:
                policy_targets[b, oi] = traj.policy_targets[seat]
            if traj.cards_known.get(seat):
                cards_known_mask[b, oi] = 1.0
            if traj.action_known.get(seat):
                action_known_mask[b, oi] = 1.0

    causal = torch.triu(torch.ones(max_len, max_len, dtype=torch.bool, device=device), diagonal=1)

    return {
        "token_types": token_types,
        "token_features": token_features,
        "attn_mask": attn_mask,
        "causal_mask": causal,
        "hero_seat": hero_seats,
        "belief_targets": belief_targets,
        "policy_targets": policy_targets,
        "cards_known_mask": cards_known_mask,
        "action_known_mask": action_known_mask,
        "opp_seat_map": opp_seat_map,
    }


def trajectory_to_dict(traj: TrajectoryRecord) -> dict[str, Any]:
    """Serialize trajectory for Parquet storage."""
    opps = opponent_seats(traj.hero_seat)
    belief = [-1] * 5
    policy = [-1] * 5
    cards_known = [0.0] * 5
    action_known = [0.0] * 5
    for oi, seat in enumerate(opps):
        if seat in traj.belief_targets:
            belief[oi] = traj.belief_targets[seat]
        if seat in traj.policy_targets:
            policy[oi] = traj.policy_targets[seat]
        cards_known[oi] = float(traj.cards_known.get(seat, False))
        action_known[oi] = float(traj.action_known.get(seat, False))

    return {
        "token_types": traj.token_types,
        "token_features": [f.tolist() for f in traj.token_features],
        "actor_seats": traj.actor_seats,
        "hero_seat": traj.hero_seat,
        "belief_targets": belief,
        "policy_targets": policy,
        "cards_known_mask": cards_known,
        "action_known_mask": action_known,
    }


def dict_to_trajectory(row: dict[str, Any]) -> TrajectoryRecord:
    """Deserialize from Parquet row."""
    opps = opponent_seats(int(row["hero_seat"]))
    belief_targets = {}
    policy_targets = {}
    cards_known = {}
    action_known = {}
    for oi, seat in enumerate(opps):
        b = row["belief_targets"][oi]
        p = row["policy_targets"][oi]
        if b >= 0:
            belief_targets[seat] = int(b)
        if p >= 0:
            policy_targets[seat] = int(p)
        if row["cards_known_mask"][oi]:
            cards_known[seat] = True
        if row["action_known_mask"][oi]:
            action_known[seat] = True

    feats = [np.array(f, dtype=np.float32) for f in row["token_features"]]
    return TrajectoryRecord(
        token_types=list(row["token_types"]),
        token_features=feats,
        actor_seats=list(row["actor_seats"]),
        hero_seat=int(row["hero_seat"]),
        belief_targets=belief_targets,
        policy_targets=policy_targets,
        cards_known=cards_known,
        action_known=action_known,
    )
