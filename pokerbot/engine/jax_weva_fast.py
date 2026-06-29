"""Fast NLHE encoding from JAX ``EnvState`` (no Python ``GameState`` round-trip)."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from pokerbot.engine.jax_engine import (
    ALL_IN,
    BET_BUCKET_START,
    CALL,
    CHECK,
    CHIP_UNIT,
    FOLD,
    SEAT_EMPTY,
    EnvState,
    NUM_PLAYERS,
)
from pokerbot.engine.jax_weva_encode import NLHEEncodeBundle
from pokerbot.encoding.spot_scope import (
    NUM_SPOT_SCOPE,
    POLICY_POSITION_BASE_DIM,
    POLICY_POSITION_DIM,
    SPOT_CLASS_TO_INDEX,
    fuse_position_scope,
)
from pokerbot.encoding.tensors import NUM_CARD_SLOTS, SPEC
from pokerbot.encoding import weva_encoder
from pokerbot.encoding.weva_encoder import WEVAMapper

# Position enum order used by nlhe_bridge._position_for_seat
_POS_BY_OFFSET = np.array([3, 4, 5, 0, 1, 2], dtype=np.int32)  # BTN, SB, BB, UTG, HJ, CO

# Python-int copies of the seat-status enums (the engine exports them as JAX int8 scalars). Comparing
# host ints against these directly keeps the per-node encode comparisons in pure Python (no eager JAX).
_SEAT_EMPTY_I = int(SEAT_EMPTY)


def _seat_status_ints() -> tuple[int, int]:
    from pokerbot.engine.side_pots_jax import SEAT_ALLIN, SEAT_FOLDED

    return int(SEAT_ALLIN), int(SEAT_FOLDED)


_SEAT_ALLIN_I, _SEAT_FOLDED_I = _seat_status_ints()


def build_weva_bucket_table(mapper: WEVAMapper | None = None) -> np.ndarray:
    """O(1) WEVA bucket lookup: table[lo_idx, hi_idx, position_enum] -> bucket id.

    Under POKERBOT_PREFLOP_LOSSLESS the entry is the lossless ``pos*169 + canonical_preflop_class`` and no
    WEVA k-means index is needed (the lossy clustering is bypassed entirely preflop)."""
    table = np.zeros((52, 52, 6), dtype=np.int32)
    if weva_encoder._PREFLOP_LOSSLESS:
        n = weva_encoder.PREFLOP_NUM_CLASSES
        for c0 in range(52):
            for c1 in range(c0 + 1, 52):
                cls = weva_encoder.canonical_preflop_class(c0, c1)
                for pos in range(6):
                    table[c0, c1, pos] = pos * n + cls
        return table

    from pokerbot.engine.cards import Card, Rank, Suit

    m = mapper or WEVAMapper.default()
    for c0 in range(52):
        for c1 in range(c0 + 1, 52):
            card0 = Card(Rank(c0 // 4 + 2), Suit(c0 % 4))
            card1 = Card(Rank(c1 // 4 + 2), Suit(c1 % 4))
            for pos in range(6):
                table[c0, c1, pos] = int(m.lookup((card0, card1), position=pos))
    return table


@lru_cache(maxsize=1)
def default_weva_bucket_table() -> np.ndarray:
    return build_weva_bucket_table()


@lru_cache(maxsize=8)
def _template_committed_street_bb(button: int) -> np.ndarray:
    """Match ``game_state_from_jax`` leaving EMPTY seats' blind defaults from ``new_hand``."""
    from pokerbot.engine.game import GameState

    gs = GameState.new_hand(button_seat=button)
    return np.array([p.committed_street_bb for p in gs.players], dtype=np.float32)


def _weva_bucket_from_holes(
    h0: int,
    h1: int,
    position_enum: int,
    table: np.ndarray,
) -> int:
    if h0 < 0 or h1 < 0:
        return 0
    lo, hi = (h0, h1) if h0 <= h1 else (h1, h0)
    return int(table[lo, hi, position_enum])


def _scatter_card_plane(tensor: np.ndarray, slot: int, card_idx: int) -> None:
    if card_idx < 0:
        return
    suit = card_idx % 4
    rank = card_idx // 4
    tensor[slot, suit, rank] = 1.0


def encode_cards_numpy(
    holes: np.ndarray,
    board: np.ndarray,
    hero_seat: int,
) -> np.ndarray:
    """Match ``encode_cards`` (9, 4, 13) without GameState."""
    tensor = np.zeros(SPEC.card_shape, dtype=np.float32)
    h0, h1 = int(holes[hero_seat, 0]), int(holes[hero_seat, 1])
    _scatter_card_plane(tensor, 0, h0)
    _scatter_card_plane(tensor, 1, h1)
    slot = 2
    for i in range(board.shape[0]):
        c = int(board[i])
        if c >= 0 and slot < NUM_CARD_SLOTS:
            _scatter_card_plane(tensor, slot, c)
            slot += 1
    return tensor


def _bucket(x: float, edges: tuple[float, ...]) -> int:
    """Index of the bin x falls into (0..len(edges)). Small, monotone, deterministic."""
    b = 0
    for e in edges:
        if x >= e:
            b += 1
        else:
            break
    return b


# Bucket edges for the bounded betting-state key (tunable coarseness vs precision).
_POT_EDGES = (3.0, 6.0, 12.0, 25.0, 50.0, 100.0, 200.0)        # total pot in BB
_TOCALL_EDGES = (0.01, 0.34, 0.67, 1.01, 2.01)                  # amount-to-call as fraction of pot
_SPR_EDGES = (1.0, 2.5, 5.0, 10.0, 20.0)                        # effective-stack-to-pot ratio
# Effective STARTING-stack depth (absolute BB) for the depth-aware PREFLOP key (A2). 4 buckets for
# CoinPoker cash: short [..,50) (40-50bb buy-ins + lost-down tail), mid [50,90), full [90,140) (the
# 100bb buy-in cluster), deep [140,..) (run-ups). NOTE: the 100bb MAX buy-in bounds the START depth, NOT
# the running stack — winners accumulate above 100bb and stay deep until they lose back (no mid-session
# cash-out), and losers reloading at <=100bb feed that persistent deep tail, so a WINNING bot genuinely
# plays 130-200bb deep-vs-deep. (Multiway all-ins make a 200-300bb POT, not >100bb DEPTH — pot is keyed
# separately by _POT_EDGES/SPR.) Absolute (not pot-relative): preflop pot ~1.5bb so SPR is uninformative.
# Only consulted when POKERBOT_DEPTH_BUCKETS.
_DEPTH_EDGES = (50.0, 90.0, 140.0)


def effective_depth_bb(state: EnvState | Any) -> float:
    """Effective STARTING stack depth in BB = min over non-folded seats of (stacks + committed_total).
    Chips only move stack->committed within a hand, so the sum is each seat's depth at hand start and the
    min (the binding stack) is CONSTANT across the hand — a preflop infoset stays in ONE depth bucket."""
    stacks = np.asarray(state.stacks, dtype=np.float64)
    committed_total = np.asarray(state.committed_total, dtype=np.float64)
    seat_status = np.asarray(state.seat_status)
    totals = stacks + committed_total
    live = [totals[s] for s in range(len(totals))
            if int(seat_status[s]) not in (_SEAT_EMPTY_I, _SEAT_FOLDED_I) and totals[s] > 0]
    chips = min(live) if live else (float(totals.max()) if totals.size else 100.0 * CHIP_UNIT)
    return float(chips) / float(CHIP_UNIT)


def depth_bucket(state: EnvState | Any) -> int:
    """Discrete effective-starting-depth bucket in [0, len(_DEPTH_EDGES)] for the depth-aware preflop key."""
    return _bucket(effective_depth_bb(state), _DEPTH_EDGES)


def betting_abstraction_key(state: EnvState | Any, seat: int) -> bytes:
    """BOUNDED betting-STATE descriptor (Pluribus-style betting abstraction) for the tabular key.

    Distinct action *sequences* that reach the same strategic situation are deliberately MERGED — the
    opposite of ``betting_history_key`` (which is lossless and therefore unbounded). Keying on the
    abstracted public betting state — (street, pot bucket, facing-bet bucket, SPR bucket, position,
    facing-aggression) — bounds the infoset count so the table SATURATES and each infoset is revisited
    enough to train, making convergence feasible. Deterministic ⇒ shard merge stays bit-identical;
    the merges it induces are intended abstraction, surfaced (not silently) by the collision meter.
    """
    committed = np.asarray(state.committed_street)
    seat_status = np.asarray(state.seat_status)
    stacks = np.asarray(state.stacks)
    pot = int(np.asarray(state.pot_dead)) + int(committed.sum())
    cur = int(np.asarray(state.current_bet))
    to_call = max(0, cur - int(committed[seat]))
    street = int(np.asarray(state.street))
    button = int(np.asarray(state.button))
    last_agg = int(np.asarray(state.last_aggressor))
    chip = float(CHIP_UNIT)
    pot_bb = pot / chip
    # Effective stack = smallest live stack (HU heads-up the binding stack); SPR vs the pot.
    live = [int(stacks[s]) for s in range(len(stacks)) if int(seat_status[s]) not in (_SEAT_EMPTY_I, _SEAT_FOLDED_I)]
    eff = (min(live) if live else 0) / chip
    pot_b = _bucket(pot_bb, _POT_EDGES)
    tocall_b = _bucket(to_call / max(pot, 1), _TOCALL_EDGES)
    spr_b = _bucket(eff / max(pot_bb, 1e-6), _SPR_EDGES)
    is_button = 1 if seat == button else 0
    facing_agg = 1 if (last_agg >= 0 and last_agg != seat) else 0
    return bytes((street & 7, pot_b, tocall_b, spr_b, is_button, facing_agg))


def betting_history_key(state: EnvState | Any) -> bytes:
    """Lossless bytes of the discrete betting sequence — the part the action TENSOR throws away.

    ``encode_actions_numpy`` max-pools the whole history into one per-seat slot (street index hardcoded
    to 0, bet sizes capped at 100bb), so distinct betting sequences can collide to the same action
    plane. The net keeps that lossy tensor (SFT-aligned), but the tabular CFR KEY must distinguish
    genuinely-different infosets, so ``fingerprint()`` also hashes this exact ``(seat, action, amount)``
    sequence. It is bounded by the engine's discrete action set (no unbounded blow-up) and deterministic
    (merge stays bit-identical). Independent of hero seat — the betting record is public.
    """
    hl = int(np.asarray(state.hist_len))
    if hl <= 0:
        return b"\x00\x00"
    seat = np.asarray(state.hist_seat)[:hl].astype(np.int16, copy=False)
    act = np.asarray(state.hist_action)[:hl].astype(np.int16, copy=False)
    # (seat, action-index) sequence only — NOT the raw amount. The amount is fully determined by the
    # action index + the deterministic state, so it is redundant for identifying the infoset; and
    # including it would UN-bound action abstraction (one abstracted size, e.g. ½-pot, maps to many
    # raw amounts across pot sizes → many keys). Dropping it keeps the key collision-free AND lets
    # the bet-size abstraction actually bound the betting dimension.
    return hl.to_bytes(2, "little") + seat.tobytes() + act.tobytes()


def encode_actions_numpy(
    *,
    holes: np.ndarray,
    stacks: np.ndarray,
    committed_street: np.ndarray,
    seat_status: np.ndarray,
    button: int,
    hero_seat: int,
    hist_seat: np.ndarray,
    hist_action: np.ndarray,
    hist_amount: np.ndarray,
    hist_len: int,
    pot_dead: int,
    current_bet: int,
) -> np.ndarray:
    """Match ``encode_actions`` including street-0 history packing."""
    tensor = np.zeros(SPEC.action_shape, dtype=np.float32)
    chip = float(CHIP_UNIT)
    pot_bb = float(pot_dead) / chip + float(np.sum(committed_street)) / chip
    current_bet_bb = float(current_bet) / chip
    street = 0

    for i in range(min(hist_len, hist_seat.shape[0])):
        seat = int(hist_seat[i])
        if seat < 0:
            continue
        act = int(hist_action[i])
        amt = float(hist_amount[i]) / chip
        feat = np.zeros(12, dtype=np.float32)
        feat[0] = 1.0
        if act == FOLD:
            feat[1] = 1.0
        elif act in (CHECK, CALL):
            feat[2] = 1.0
        else:
            feat[3] = 1.0
        feat[4] = min(amt / 100.0, 1.0)
        feat[5] = 1.0 if seat == hero_seat else 0.0
        feat[6] = seat / NUM_PLAYERS
        tensor[seat, street, :] = np.maximum(tensor[seat, street, :], feat)

    template_comm = _template_committed_street_bb(button)
    for seat in range(NUM_PLAYERS):
        pos_enum = int(_POS_BY_OFFSET[(seat - button) % NUM_PLAYERS])
        # Compare against python-int seat-status constants (the jax-engine values are JAX scalars; a
        # raw ``int(x) == JAX_SCALAR`` triggers an eager JAX ``==`` + ``bool`` per seat per node).
        status_i = int(seat_status[seat])
        empty = status_i == _SEAT_EMPTY_I
        folded = empty or status_i == _SEAT_FOLDED_I
        all_in = status_i == _SEAT_ALLIN_I
        stack_bb = 0.0 if empty else float(stacks[seat]) / chip
        comm_st = template_comm[seat] if empty else float(committed_street[seat]) / chip
        tensor[seat, 0, 7] = stack_bb / 100.0
        tensor[seat, 0, 8] = 1.0 if folded else 0.0
        tensor[seat, 0, 9] = 1.0 if all_in else 0.0
        tensor[seat, 0, 10] = comm_st / max(current_bet_bb, 1.0)
        tensor[seat, 0, 11] = float(pos_enum)

    tensor[:, :, 6] = pot_bb / 200.0
    return tensor


def encode_position_numpy(button: int, hero_seat: int) -> np.ndarray:
    pos = np.zeros(NUM_PLAYERS, dtype=np.float32)
    pos_enum = int(_POS_BY_OFFSET[(hero_seat - button) % NUM_PLAYERS])
    pos[pos_enum] = 1.0
    return pos


def encode_spot_scope_numpy(spot_class: str = "") -> np.ndarray:
    vec = np.zeros(NUM_SPOT_SCOPE, dtype=np.float32)
    idx = SPOT_CLASS_TO_INDEX.get(str(spot_class).strip())
    if idx is not None:
        vec[idx] = 1.0
    return vec


def structural_infoset_key(state: EnvState | Any, seat: int) -> tuple:
    """Cheap cache key for encode inputs (matches fast-path state fields + hero seat)."""
    holes = np.asarray(state.holes)
    board = np.asarray(state.board)
    stacks = np.asarray(state.stacks)
    committed_street = np.asarray(state.committed_street)
    seat_status = np.asarray(state.seat_status)
    hist_seat = np.asarray(state.hist_seat)
    hist_action = np.asarray(state.hist_action)
    hist_amount = np.asarray(state.hist_amount)
    hist_len = int(np.asarray(state.hist_len))
    return (
        int(seat),
        int(np.asarray(state.button)),
        int(np.asarray(state.pot_dead)),
        int(np.asarray(state.current_bet)),
        int(np.asarray(state.street)),
        holes.tobytes(),
        board.tobytes(),
        stacks.tobytes(),
        committed_street.tobytes(),
        seat_status.tobytes(),
        hist_len,
        hist_seat[:hist_len].tobytes(),
        hist_action[:hist_len].tobytes(),
        hist_amount[:hist_len].tobytes(),
    )


def encode_nlhe_bundle_fast(
    state: EnvState | Any,
    seat: int,
    *,
    weva_table: np.ndarray | None = None,
    spot_class: str = "",
    oop_range: str = "",
    ip_range: str = "",
) -> NLHEEncodeBundle:
    """Vectorized-friendly encode; skips ``game_state_from_jax``."""
    holes = np.asarray(state.holes)
    board = np.asarray(state.board)
    stacks = np.asarray(state.stacks)
    committed_street = np.asarray(state.committed_street)
    seat_status = np.asarray(state.seat_status)
    button = int(np.asarray(state.button))
    hist_seat = np.asarray(state.hist_seat)
    hist_action = np.asarray(state.hist_action)
    hist_amount = np.asarray(state.hist_amount)
    hist_len = int(np.asarray(state.hist_len))
    pot_dead = int(np.asarray(state.pot_dead))
    current_bet = int(np.asarray(state.current_bet))

    table = weva_table if weva_table is not None else default_weva_bucket_table()
    pos_enum = int(_POS_BY_OFFSET[(seat - button) % NUM_PLAYERS])
    h0, h1 = int(holes[seat, 0]), int(holes[seat, 1])
    weva_id = _weva_bucket_from_holes(h0, h1, pos_enum, table)

    cards = encode_cards_numpy(holes, board, seat)
    actions = encode_actions_numpy(
        holes=holes,
        stacks=stacks,
        committed_street=committed_street,
        seat_status=seat_status,
        button=button,
        hero_seat=seat,
        hist_seat=hist_seat,
        hist_action=hist_action,
        hist_amount=hist_amount,
        hist_len=hist_len,
        pot_dead=pot_dead,
        current_bet=current_bet,
    )
    position = encode_position_numpy(button, seat)
    _ = oop_range, ip_range
    spot_scope = encode_spot_scope_numpy(spot_class)
    policy_position = fuse_position_scope(position, spot_scope)
    from pokerbot.engine.jax_weva_encode import _DEPTH_BUCKETS, compute_postflop_bucket

    bucket = compute_postflop_bucket(holes[seat], board)
    dbucket = depth_bucket(state) if _DEPTH_BUCKETS else -1  # hashed preflop only; -1 => not hashed
    return NLHEEncodeBundle(
        weva_id=weva_id,
        cards=cards,
        actions=actions,
        position=position,
        spot_scope=spot_scope,
        policy_position=policy_position,
        postflop_bucket_id=bucket,
        history_key=betting_history_key(state),
        betting_state=betting_abstraction_key(state, seat),
        depth_bucket=dbucket,
    )


_FP_FAST_EMPTY = np.zeros((1,), dtype=np.float32)


def fingerprint_fast(state: EnvState | Any, seat: int, *, weva_table: np.ndarray | None = None) -> bytes:
    """Bounded-mode FAST fingerprint: computes ONLY the bytes the bounded key uses — weva_id (or the
    postflop made-hand bucket) + the betting-abstraction key — and reuses ``NLHEEncodeBundle.fingerprint``
    so the digest is BYTE-IDENTICAL to the full encode. Skips the (9,4,13)+(6,4,12) card/action tensors,
    position, and lossless history key (none of which the bounded key reads) — the dominant hot-path cost
    in the convergence traversal (encode_actions_numpy alone is ~25% of per-node time, and unused here).

    Falls back to the full encode when betting-abstraction mode is OFF (the exact-mode key needs the
    tensors). Use the full ``encode_nlhe_bundle_fast`` when the bundle itself is needed (net distillation).
    """
    from pokerbot.engine.jax_weva_encode import (
        _BETTING_ABSTRACTION,
        _DEPTH_BUCKETS,
        NLHEEncodeBundle,
        compute_postflop_bucket,
    )

    if not _BETTING_ABSTRACTION:
        return encode_nlhe_bundle_fast(state, seat, weva_table=weva_table).fingerprint()

    holes = np.asarray(state.holes)
    board = np.asarray(state.board)
    button = int(np.asarray(state.button))
    table = weva_table if weva_table is not None else default_weva_bucket_table()
    pos_enum = int(_POS_BY_OFFSET[(seat - button) % NUM_PLAYERS])
    weva_id = _weva_bucket_from_holes(int(holes[seat, 0]), int(holes[seat, 1]), pos_enum, table)
    bucket = compute_postflop_bucket(holes[seat], board)
    betting_state = betting_abstraction_key(state, seat)
    # Lossless preflop BETTING (opt-in): the preflop key uses the lossless betting history instead of the
    # bucketed betting_state. Only when BOTH flags set, and only preflop (bucket < 0). The cards-only
    # default keeps betting_state, so history stays empty (the digest ignores it there).
    history_key = b""
    if weva_encoder._PREFLOP_LOSSLESS and weva_encoder._PREFLOP_LOSSLESS_BETTING and bucket < 0:
        history_key = betting_history_key(state)
    # Minimal bundle: bounded-mode .fingerprint() reads weva_id/postflop_bucket_id + (history_key preflop-
    # lossless | betting_state otherwise) — see NLHEEncodeBundle._digest.
    dbucket = depth_bucket(state) if _DEPTH_BUCKETS else -1  # serve-side twin of the train encode
    return NLHEEncodeBundle(
        weva_id=weva_id, cards=_FP_FAST_EMPTY, actions=_FP_FAST_EMPTY, position=_FP_FAST_EMPTY,
        spot_scope=_FP_FAST_EMPTY, policy_position=_FP_FAST_EMPTY, postflop_bucket_id=bucket,
        history_key=history_key, betting_state=betting_state, depth_bucket=dbucket,
    ).fingerprint()


def stack_env_states(states: list[EnvState]) -> dict[str, np.ndarray]:
    """Stack a list of ``EnvState`` into batched numpy arrays (leading dim B)."""
    if not states:
        raise ValueError("empty state list")
    fields = states[0]._fields
    out: dict[str, np.ndarray] = {}
    for f in fields:
        arrs = [np.asarray(getattr(s, f)) for s in states]
        out[f] = np.stack(arrs, axis=0)
    return out


def encode_nlhe_bundles_batch_numpy(
    batch: dict[str, np.ndarray],
    seats: np.ndarray,
    *,
    weva_table: np.ndarray | None = None,
    spot_class: str = "",
) -> list[NLHEEncodeBundle]:
    """Batch encode on CPU numpy (no Python GameState)."""
    table = weva_table if weva_table is not None else default_weva_bucket_table()
    b = batch["holes"].shape[0]
    bundles: list[NLHEEncodeBundle] = []
    for i in range(b):
        seat = int(seats[i])

        class _Slice:
            def __init__(self, arrays: dict[str, np.ndarray], idx: int) -> None:
                self._a = arrays
                self._i = idx

            def __getattr__(self, name: str) -> Any:
                v = self._a[name][self._i]
                if name in ("street", "button", "action_seat", "last_aggressor", "hist_len", "deck_ptr"):
                    return np.int32(v)
                if name in ("done", "pending_deal", "last_raise_full"):
                    return np.bool_(v)
                return v

        bundles.append(
            encode_nlhe_bundle_fast(_Slice(batch, i), seat, weva_table=table, spot_class=spot_class)
        )
    return bundles


# --- JAX batched path (jit + vmap) ---

def _encode_cards_jax(holes: Array, board: Array, hero_seat: Array) -> Array:
    """Batched cards (B, 9, 4, 13). hero_seat scalar per batch via vmap."""
    tensor = jnp.zeros((NUM_CARD_SLOTS, 4, 13), dtype=jnp.float32)

    def scatter(slot, card_idx, t):
        valid = card_idx >= 0
        suit = jnp.where(valid, card_idx % 4, 0)
        rank = jnp.where(valid, card_idx // 4, 0)
        t = t.at[slot, suit, rank].set(jnp.where(valid, 1.0, 0.0), mode="drop")
        return t

    h0 = holes[hero_seat, 0]
    h1 = holes[hero_seat, 1]
    tensor = scatter(0, h0, tensor)
    tensor = scatter(1, h1, tensor)

    def scan_board(i, t):
        c = board[i]
        slot = i + 2
        return scatter(slot, c, t), None

    tensor, _ = jax.lax.scan(scan_board, tensor, jnp.arange(board.shape[0]))
    return tensor


def _encode_actions_jax(
    stacks: Array,
    committed_street: Array,
    seat_status: Array,
    button: Array,
    hero_seat: Array,
    hist_seat: Array,
    hist_action: Array,
    hist_amount: Array,
    hist_len: Array,
    pot_dead: Array,
    current_bet: Array,
) -> Array:
    """Single-state action tensor (6, 4, 12) — matches numpy path."""
    tensor = jnp.zeros((NUM_PLAYERS, 4, 12), dtype=jnp.float32)
    chip = jnp.float32(CHIP_UNIT)
    pot_bb = pot_dead / chip + jnp.sum(committed_street) / chip
    current_bet_bb = current_bet / chip
    street = 0

    def body(i, t):
        seat = hist_seat[i]
        act = hist_action[i]
        amt = hist_amount[i] / chip
        valid = seat >= 0
        feat = jnp.zeros(12, dtype=jnp.float32)
        feat = feat.at[0].set(1.0)
        feat = feat.at[1].set(jnp.where(act == FOLD, 1.0, 0.0))
        feat = feat.at[2].set(jnp.where((act == CHECK) | (act == CALL), 1.0, 0.0))
        feat = feat.at[3].set(
            jnp.where((act != FOLD) & (act != CHECK) & (act != CALL), 1.0, 0.0)
        )
        feat = feat.at[4].set(jnp.minimum(amt / 100.0, 1.0))
        feat = feat.at[5].set(jnp.where(seat == hero_seat, 1.0, 0.0))
        feat = feat.at[6].set(seat / jnp.float32(NUM_PLAYERS))
        old = t[seat, street, :]
        new = jnp.where(valid, jnp.maximum(old, feat), old)
        t = t.at[seat, street, :].set(new)
        return t

    tensor = jax.lax.fori_loop(0, hist_len, body, tensor)

    from pokerbot.engine.side_pots_jax import SEAT_ALLIN, SEAT_FOLDED

    def seat_row(s, t):
        pos_enum = _POS_BY_OFFSET[(s - button) % NUM_PLAYERS]
        empty = seat_status[s] == SEAT_EMPTY
        folded = empty | (seat_status[s] == SEAT_FOLDED)
        all_in = seat_status[s] == SEAT_ALLIN
        stack_bb = jnp.where(empty, 0.0, stacks[s] / chip)
        comm_st = committed_street[s] / chip
        t = t.at[s, 0, 7].set(stack_bb / 100.0)
        t = t.at[s, 0, 8].set(jnp.where(folded, 1.0, 0.0))
        t = t.at[s, 0, 9].set(jnp.where(all_in, 1.0, 0.0))
        t = t.at[s, 0, 10].set(comm_st / jnp.maximum(current_bet_bb, 1.0))
        t = t.at[s, 0, 11].set(jnp.float32(pos_enum))
        return t

    tensor = jax.lax.fori_loop(0, NUM_PLAYERS, seat_row, tensor)
    tensor = tensor.at[:, :, 6].set(pot_bb / 200.0)
    return tensor


def _encode_position_jax(button: Array, hero_seat: Array) -> Array:
    pos_enum = _POS_BY_OFFSET[(hero_seat - button) % NUM_PLAYERS]
    return jax.nn.one_hot(pos_enum, NUM_PLAYERS, dtype=jnp.float32)


@jax.jit
def encode_tensors_jit(
    holes: Array,
    board: Array,
    stacks: Array,
    committed_street: Array,
    seat_status: Array,
    button: Array,
    hero_seat: Array,
    hist_seat: Array,
    hist_action: Array,
    hist_amount: Array,
    hist_len: Array,
    pot_dead: Array,
    current_bet: Array,
) -> tuple[Array, Array, Array]:
    cards = _encode_cards_jax(holes, board, hero_seat)
    actions = _encode_actions_jax(
        stacks,
        committed_street,
        seat_status,
        button,
        hero_seat,
        hist_seat,
        hist_action,
        hist_amount,
        hist_len,
        pot_dead,
        current_bet,
    )
    position = _encode_position_jax(button, hero_seat)
    return cards, actions, position


_batch_encode_jit = jax.jit(jax.vmap(encode_tensors_jit, in_axes=(0,) * 12))


def encode_nlhe_tensors_batch_jax(
    batch: dict[str, Array],
    seats: Array,
    weva_table: np.ndarray,
) -> tuple[Array, Array, Array, Array]:
    """
    Batched encode: returns (weva_ids, cards, actions, policy_position) as JAX arrays.

    ``weva_table`` is numpy (52,52,6); lookup done on host then transferred.
    """
    cards, actions, position = _batch_encode_jit(
        batch["holes"],
        batch["board"],
        batch["stacks"],
        batch["committed_street"],
        batch["seat_status"],
        batch["button"],
        seats.astype(jnp.int32),
        batch["hist_seat"],
        batch["hist_action"],
        batch["hist_amount"],
        batch["hist_len"],
        batch["pot_dead"],
        batch["current_bet"],
    )
    holes = np.asarray(batch["holes"])
    buttons = np.asarray(batch["button"])
    seats_np = np.asarray(seats)
    b = holes.shape[0]
    weva_ids = np.zeros(b, dtype=np.int32)
    for i in range(b):
        seat = int(seats_np[i])
        pos_enum = int(_POS_BY_OFFSET[(seat - int(buttons[i])) % NUM_PLAYERS])
        h0, h1 = int(holes[i, seat, 0]), int(holes[i, seat, 1])
        weva_ids[i] = _weva_bucket_from_holes(h0, h1, pos_enum, weva_table)
    scope = jnp.zeros((b, NUM_SPOT_SCOPE), dtype=jnp.float32)
    policy_position = jnp.concatenate([position, scope], axis=-1)
    return jnp.asarray(weva_ids), cards, actions, policy_position


class FastEncodeCache:
    """Memoize WEVA bucket by (lo, hi, pos) and optional bundle fingerprint cache."""

    def __init__(self, weva_table: np.ndarray | None = None, *, max_entries: int = 50_000) -> None:
        self.weva_table = weva_table if weva_table is not None else default_weva_bucket_table()
        self._weva: dict[tuple[int, int, int], int] = {}
        self._max = max_entries

    def weva_bucket(self, h0: int, h1: int, pos_enum: int) -> int:
        key = (min(h0, h1), max(h0, h1), pos_enum) if h0 >= 0 and h1 >= 0 else (-1, -1, pos_enum)
        if key in self._weva:
            return self._weva[key]
        bid = _weva_bucket_from_holes(h0, h1, pos_enum, self.weva_table)
        if len(self._weva) < self._max:
            self._weva[key] = bid
        return bid
