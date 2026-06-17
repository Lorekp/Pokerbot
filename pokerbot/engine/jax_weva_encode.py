"""WEVA-aligned NLHE infoset encoding (Phase 2b Gate 0)."""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from typing import Any

import numpy as np

# Opt-in postflop information abstraction for the tabular KEY (off by default → exact current
# behavior, all bit-identical gates preserved). Enable with POKERBOT_POSTFLOP_BUCKETS=1.
_POSTFLOP_BUCKETS_ENABLED = os.environ.get("POKERBOT_POSTFLOP_BUCKETS", "0").lower() not in (
    "",
    "0",
    "false",
    "no",
)
# Betting (action-sequence) abstraction: key on a BOUNDED betting STATE instead of the lossless
# action history, so the infoset count saturates and the blueprint converges. Off => exact/lossless.
_BETTING_ABSTRACTION = os.environ.get("POKERBOT_BETTING_ABSTRACTION", "0").lower() not in (
    "",
    "0",
    "false",
    "no",
)
_GLOBAL_BUCKETER: Any = None
# Inline card-abstraction mode: "fast" (one made-hand eval/node, ~2us — the per-node-affordable
# default) vs "ehs" (Monte-Carlo EHS, accurate but ~ms/cold-key — only for offline precompute).
_POSTFLOP_MODE = os.environ.get("POKERBOT_POSTFLOP_MODE", "fast").lower()
_POSTFLOP_NUM_BUCKETS = int(os.environ.get("POKERBOT_POSTFLOP_NUM_BUCKETS", "200"))


def _get_bucketer() -> Any:
    global _GLOBAL_BUCKETER
    if _GLOBAL_BUCKETER is None:
        from pokerbot.encoding.postflop_bucketer import PostflopBucketer

        iters = int(os.environ.get("POKERBOT_POSTFLOP_EHS_ITERS", "500"))
        _GLOBAL_BUCKETER = PostflopBucketer(num_buckets=_POSTFLOP_NUM_BUCKETS, n_iter=iters)
    return _GLOBAL_BUCKETER


def compute_postflop_bucket(hole_ids: Any, board_ids: Any) -> int:
    """Bounded postflop bucket for (hero hole, board); -1 if abstraction off or preflop.

    Default mode 'fast' = one made-hand rank evaluation per node (cheap enough for the CFR hot path),
    bounding the postflop card dimension so the tables stop growing. 'ehs' = the slow MC bucketer.
    """
    if not _POSTFLOP_BUCKETS_ENABLED:
        return -1
    board = [int(c) for c in np.asarray(board_ids).tolist() if int(c) >= 0]
    if len(board) < 3:
        return -1  # preflop: weva_id already buckets the holes
    h = [int(c) for c in np.asarray(hole_ids).tolist()]
    if len(h) != 2 or any(c < 0 for c in h):
        return -1
    if _POSTFLOP_MODE == "fast":
        from pokerbot.encoding.postflop_bucketer import fast_made_hand_bucket

        return fast_made_hand_bucket((h[0], h[1]), tuple(board), _POSTFLOP_NUM_BUCKETS)
    return _get_bucketer().bucket((h[0], h[1]), tuple(board))

from pokerbot.engine.jax_engine import EnvState
from pokerbot.engine.nlhe_bridge import game_state_from_jax
from pokerbot.encoding.spot_scope import POLICY_POSITION_DIM, fuse_position_scope
from pokerbot.encoding.tensors import SPEC, encode_state
from pokerbot.encoding import weva_encoder
from pokerbot.encoding.weva_encoder import WEVAContext, WEVAMapper


def active_abstraction_config() -> dict:
    """The abstraction flags baked into THIS process at import. Stamp into trained checkpoints
    (provenance) and cross-check at serve via ``lbr.check_blueprint_serve_env`` — a mismatch silently
    breaks blueprint coverage (every lookup misses → uniform fallback), the train/serve skew that has
    bitten this pipeline before (postflop 0%-coverage)."""
    return {
        "betting_abstraction": bool(_BETTING_ABSTRACTION),
        "postflop_buckets": bool(_POSTFLOP_BUCKETS_ENABLED),
        "postflop_mode": _POSTFLOP_MODE,
        "postflop_num_buckets": int(_POSTFLOP_NUM_BUCKETS),
        "preflop_lossless": bool(weva_encoder._PREFLOP_LOSSLESS),
        "preflop_lossless_betting": bool(weva_encoder._PREFLOP_LOSSLESS_BETTING),
    }


@dataclass(frozen=True)
class NLHEEncodeBundle:
    weva_id: int
    cards: np.ndarray
    actions: np.ndarray
    position: np.ndarray
    spot_scope: np.ndarray
    policy_position: np.ndarray
    # Bounded postflop information-abstraction bucket for the tabular KEY (Pluribus-style). -1 when
    # the abstraction is off or preflop, in which case the key falls back to the exact card bytes.
    # The net (``flatten``) always uses the raw cards; only the tabular key is abstracted.
    postflop_bucket_id: int = -1
    # Lossless bytes of the discrete betting sequence (seat, action, amount per step). The action
    # TENSOR is max-pooled/lossy (SFT-aligned net input), so without this distinct betting histories
    # would collide to one tabular key and silently merge regrets. Hashed into the key below; bounded
    # by the engine's discrete action set. Empty => no history (e.g. legacy callers); see
    # ``betting_history_key`` in jax_weva_fast.
    history_key: bytes = b""
    # BOUNDED betting-state descriptor (street, pot/facing-size/SPR buckets, position, facing-agg).
    # Used as the betting dimension of the key ONLY in betting-abstraction mode (the lossless
    # history_key is unbounded → never converges). See ``betting_abstraction_key`` in jax_weva_fast.
    betting_state: bytes = b""

    def _digest(self, person: bytes) -> bytes:
        h = hashlib.blake2b(digest_size=16, person=person)
        if _BETTING_ABSTRACTION:
            # FULLY BOUNDED key (Pluribus-style): card bucket + abstracted betting STATE; no raw
            # cards/actions tensor, no lossless history → the infoset count saturates and trains.
            # PREFLOP: weva_id (hole bucket). POSTFLOP: the made-hand bucket ALONE — NOT weva_id too,
            # or the hole (≤1326) × bucket (200) product un-bounds the card dimension (the bug that
            # kept it from saturating).
            if self.postflop_bucket_id >= 0:
                h.update(b"B")
                h.update(int(self.postflop_bucket_id).to_bytes(4, "little", signed=True))
                h.update(b"S")
                h.update(self.betting_state)
            else:
                # PREFLOP. weva_id = lossless 169-class bucket when POKERBOT_PREFLOP_LOSSLESS (recovers the
                # strategically-distinct open/3-bet hands the WEVA k-means merged = the proven +180 leak),
                # else the WEVA bucket. Betting dimension = LOSSLESS history ONLY when the opt-in
                # POKERBOT_PREFLOP_LOSSLESS_BETTING is also set (Pluribus-exact, but the preflop history
                # combinatorics explode the table); otherwise the bounded betting_state (tractable default).
                h.update(b"w")
                h.update(int(self.weva_id).to_bytes(4, "little", signed=True))
                if weva_encoder._PREFLOP_LOSSLESS and weva_encoder._PREFLOP_LOSSLESS_BETTING:
                    h.update(b"H")
                    h.update(self.history_key)
                else:
                    h.update(b"S")
                    h.update(self.betting_state)
            return h.digest()
        h.update(b"w")
        h.update(int(self.weva_id).to_bytes(4, "little", signed=True))
        if self.postflop_bucket_id >= 0:
            h.update(b"B")
            h.update(int(self.postflop_bucket_id).to_bytes(4, "little", signed=True))
        else:
            h.update(b"C")
            h.update(np.round(self.cards, decimals=5).tobytes())
        h.update(np.round(self.actions, decimals=5).tobytes())
        h.update(np.round(self.policy_position, decimals=5).tobytes())
        # Lossless betting history — distinguishes infosets the lossy action tensor would merge.
        h.update(b"H")
        h.update(self.history_key)
        return h.digest()

    def fingerprint(self) -> bytes:
        """Compact 16-byte (128-bit) digest of the infoset key.

        The raw key components (the full card/action tensors) are ~3 KB; the tabular CFR tables
        store this key per entry, so the raw key dwarfs the 88-byte regret value (~36x). Hashing to
        a 128-bit digest cuts per-entry memory ~24x with negligible HASH-collision risk (birthday
        bound ~1e-19 even at 1e9 entries) and is deterministic, so the distributed merge stays
        bit-identical. The ``history_key`` makes it lossless over distinct betting sequences (the
        action tensor alone is not). Regret VALUES are untouched (no numerical risk).
        """
        return self._digest(b"")

    def collision_witness(self) -> bytes:
        """An INDEPENDENT 128-bit digest of the same infoset components (different personalization).

        Used only by the runtime collision meter: if two nodes share a ``fingerprint`` but differ
        here, that is a genuine representational collision (two independent 128-bit hashes both
        clashing on different pre-images is ~impossible), surfaced instead of silently merged.
        """
        return self._digest(b"collision-wit")

    def flatten(self) -> np.ndarray:
        """Fixed-length vector for Flax SD-CFR advantage net."""
        # Normalize the card-bucket id by its range so the net input scale is correct: lossless preflop
        # ids live in [0, 1014); the legacy WEVA range is 10k. (Served blueprint is the tabular table; this
        # only affects net distillation.)
        _wdenom = (
            float(weva_encoder.PREFLOP_LOSSLESS_NUM) if weva_encoder._PREFLOP_LOSSLESS else 10_000.0
        )
        parts = [
            np.array([float(self.weva_id) / _wdenom], dtype=np.float32),
            self.cards.astype(np.float32).reshape(-1),
            self.actions.astype(np.float32).reshape(-1),
            self.policy_position.astype(np.float32).reshape(-1),
        ]
        return np.concatenate(parts)


NLHE_FLAT_DIM = 1 + int(np.prod(SPEC.card_shape)) + int(np.prod(SPEC.action_shape)) + POLICY_POSITION_DIM


def encode_nlhe_weva_bundle(
    state: EnvState | Any,
    seat: int,
    mapper: WEVAMapper | None = None,
    *,
    spot_class: str = "",
    oop_range: str = "",
    ip_range: str = "",
) -> NLHEEncodeBundle:
    """Encode via Python ``encode_state`` on JAX→GameState bridge (parity path)."""
    gs = game_state_from_jax(
        state,
        spot_class=spot_class,
        oop_range=oop_range,
        ip_range=ip_range,
    )
    enc = encode_state(
        gs,
        seat,
        mapper=mapper,
        spot_class=spot_class,
        oop_range=oop_range,
        ip_range=ip_range,
    )
    policy_position = fuse_position_scope(enc["position"], enc["spot_scope"])
    holes = np.asarray(state.holes)
    bucket = compute_postflop_bucket(holes[seat], np.asarray(state.board))
    from pokerbot.engine.jax_weva_fast import betting_abstraction_key, betting_history_key

    return NLHEEncodeBundle(
        weva_id=int(enc["weva_id"]),
        cards=np.asarray(enc["cards"], dtype=np.float32),
        actions=np.asarray(enc["actions"], dtype=np.float32),
        position=np.asarray(enc["position"], dtype=np.float32),
        spot_scope=np.asarray(enc["spot_scope"], dtype=np.float32),
        policy_position=np.asarray(policy_position, dtype=np.float32),
        postflop_bucket_id=bucket,
        history_key=betting_history_key(state),
        betting_state=betting_abstraction_key(state, seat),
    )


def encode_nlhe_flat(
    state: EnvState | Any,
    seat: int,
    mapper: WEVAMapper | None = None,
    **kwargs: Any,
) -> np.ndarray:
    return encode_nlhe_weva_bundle(state, seat, mapper=mapper, **kwargs).flatten()


def opponent_weva_bucket(
    state: EnvState | Any,
    opponent_seat: int,
    mapper: WEVAMapper | None = None,
    *,
    num_players: int = 2,
) -> int:
    """Hidden world id for belief-weighted regret: opponent WEVA bucket."""
    gs = game_state_from_jax(state)
    opp = gs.players[opponent_seat]
    if opp.hole is None:
        return -1
    m = mapper or WEVAMapper.default()
    ctx = WEVAContext(
        hero_position=int(opp.position),
        num_players=num_players,
        action_depth_bin=0,
    )
    return int(m.lookup(opp.hole, position=int(opp.position), context=ctx))


def bundle_matches_parquet_row(
    bundle: NLHEEncodeBundle,
    row: dict[str, Any],
    *,
    atol: float = 1e-5,
) -> dict[str, Any]:
    """Compare encoded bundle to Parquet solver row tensors."""
    cards = np.asarray(row["cards"], dtype=np.float32)
    actions = np.asarray(row["actions"], dtype=np.float32)
    pos = np.asarray(row["position"], dtype=np.float32)
    scope = np.asarray(row.get("spot_scope"), dtype=np.float32)
    policy_pos = fuse_position_scope(pos, scope) if scope.size else pos
    weva_ok = int(row["weva_bucket_id"]) == bundle.weva_id
    cards_ok = bool(np.allclose(cards, bundle.cards, atol=atol, rtol=0.0))
    actions_ok = bool(np.allclose(actions, bundle.actions, atol=atol, rtol=0.0))
    pos_ok = bool(np.allclose(policy_pos, bundle.policy_position, atol=atol, rtol=0.0))
    return {
        "weva_ok": weva_ok,
        "cards_ok": cards_ok,
        "actions_ok": actions_ok,
        "position_ok": pos_ok,
        "all_ok": weva_ok and cards_ok and actions_ok and pos_ok,
    }
