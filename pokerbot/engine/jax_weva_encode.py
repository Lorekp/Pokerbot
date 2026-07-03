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
# Depth-aware blueprint (A2): fold an effective-STARTING-stack-depth bucket into the PREFLOP tabular key
# so a depth-randomized retrain can serve depth-specific preflop ranges (40-200bb), not just 100bb. Off
# (default) => the digest is BYTE-IDENTICAL to the non-depth blueprint (the byte is only hashed when ON and
# only preflop; postflop the existing SPR bucket already carries depth-relative-to-pot). Edges live in
# jax_weva_fast._DEPTH_EDGES. Train and serve MUST agree (stamped in active_abstraction_config + guarded).
_DEPTH_BUCKETS = os.environ.get("POKERBOT_DEPTH_BUCKETS", "0").lower() not in (
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


_GLOBAL_EMD_BUCKETER: Any = None
_GLOBAL_EMD_FROZEN: Any = None
_GLOBAL_EMD_FROZEN_PATH: str | None = None
_EMD_FROZEN_MISS_COUNT: int = 0


def _emd_fail_on_miss() -> bool:
    """When frozen lookup misses, return -1 instead of silently degrading to made-hand buckets."""
    return os.environ.get("POKERBOT_EMD_FROZEN_FAIL_ON_MISS", "1").lower() not in (
        "0",
        "",
        "false",
        "no",
        "off",
    )


def _emd_mirror_fallback() -> bool:
    """Explicit train/serve mirrored fallback to made-hand bucket on frozen miss (observable)."""
    return os.environ.get("POKERBOT_EMD_FROZEN_MIRROR_FALLBACK", "0").lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _emd_allow_silent_fallback() -> bool:
    """Legacy smoke-only path: silent made-hand fallback (discouraged for pilots/eval)."""
    return os.environ.get("POKERBOT_EMD_FROZEN_ALLOW_SILENT_FALLBACK", "0").lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def reset_emd_frozen_miss_count() -> int:
    """Reset miss counter (tests); returns previous count."""
    global _EMD_FROZEN_MISS_COUNT
    prev = _EMD_FROZEN_MISS_COUNT
    _EMD_FROZEN_MISS_COUNT = 0
    return prev


def emd_frozen_miss_count() -> int:
    return int(_EMD_FROZEN_MISS_COUNT)


def _get_emd_frozen_lookup() -> Any | None:
    """Load the native frozen EMD flop/turn lookup when the serve env provides it."""
    global _GLOBAL_EMD_FROZEN, _GLOBAL_EMD_FROZEN_PATH
    path = os.environ.get("POKERBOT_POSTFLOP_EMD_FROZEN")
    if not path:
        return None
    if _GLOBAL_EMD_FROZEN is None or _GLOBAL_EMD_FROZEN_PATH != path:
        try:
            import battery_native as bn  # noqa: PLC0415
        except Exception as e:  # noqa: BLE001
            try:
                from pokerbot.search import native_solver  # noqa: PLC0415

                bn = native_solver._try_load()
            except Exception:  # noqa: BLE001
                bn = None
            if bn is None:
                raise RuntimeError(
                    "POKERBOT_POSTFLOP_EMD_FROZEN is set, but battery_native could not be imported"
                ) from e
        if not hasattr(bn, "EmdFrozenLookup"):
            raise RuntimeError("battery_native.EmdFrozenLookup missing; rebuild the native EMD crate")
        _GLOBAL_EMD_FROZEN = bn.EmdFrozenLookup.load(path)
        _GLOBAL_EMD_FROZEN_PATH = path
    return _GLOBAL_EMD_FROZEN


def _get_emd_bucketer() -> Any:
    """Load the fitted potential-aware EMD centroids (POKERBOT_POSTFLOP_EMD_TABLE) for mode 'emd'.
    The centroids are fit offline by scripts/precompute_emd_buckets.py and serialized; serving and
    training MUST point at the same table (it's stamped, so the serve guard catches a mismatch)."""
    global _GLOBAL_EMD_BUCKETER
    if _GLOBAL_EMD_BUCKETER is None:
        from pokerbot.encoding.postflop_bucketer import EMDBucketer

        path = os.environ.get("POKERBOT_POSTFLOP_EMD_TABLE")
        if not path:
            raise RuntimeError(
                "POKERBOT_POSTFLOP_MODE=emd requires POKERBOT_POSTFLOP_EMD_TABLE=<centroids.json> "
                "(fit it offline with scripts/precompute_emd_buckets.py)")
        _GLOBAL_EMD_BUCKETER = EMDBucketer.load(path)
    return _GLOBAL_EMD_BUCKETER


def warn_if_made_hand_only(context: str = "training") -> bool:
    """Loudly flag the made-hand-only ('fast') postflop abstraction, which IGNORES DRAW EQUITY — a
    known strategic leak (a flush draw is bucketed by its weak current rank, so the blueprint can't
    size/continue it correctly). Call at the start of any production training run so a new blueprint is
    never SILENTLY shipped on the made-hand bucketer. Returns True iff the made-hand path is active.
    Set POKERBOT_ALLOW_MADE_HAND_BUCKETS=1 to acknowledge (e.g. a fast smoke run)."""
    mode = active_abstraction_config().get("postflop_mode", _POSTFLOP_MODE)  # single source of truth
    if mode != "fast":
        return False
    msg = (f"[postflop-abstraction] {context}: POKERBOT_POSTFLOP_MODE=fast = MADE-HAND-ONLY buckets "
           "(ignores draw equity). For a production blueprint use POKERBOT_POSTFLOP_MODE=emd (potential-"
           "aware) with POKERBOT_POSTFLOP_EMD_TABLE, or 'ehs'. Set POKERBOT_ALLOW_MADE_HAND_BUCKETS=1 to "
           "silence for a deliberate smoke run.")
    if os.environ.get("POKERBOT_ALLOW_MADE_HAND_BUCKETS", "0").lower() in ("0", "", "false", "no", "off"):
        import warnings
        warnings.warn(msg, RuntimeWarning, stacklevel=2)
    return True


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
    if _POSTFLOP_MODE == "emd":
        # Blueprint-v2 river OCHS (POKERBOT_RIVER_OCHS): the river is keyed by the OCHS
        # nearest-centroid bucket the TRAINER wrote river rows under — checked BEFORE the frozen
        # flop/turn lookup (which never covers the river) so the river cannot silently fall
        # through to the made-hand mirror bucket (same id space, WRONG semantics — a wrong-row
        # HIT, worse than a miss). Unset (the deployed bp_emd_k160_1 regime) => unchanged.
        if len(board) == 5:
            from pokerbot.encoding.river_ochs import get_river_ochs

            _ochs = get_river_ochs()
            if _ochs is not None:
                b = _ochs.bucket((h[0], h[1]), board)
                if b >= 0:
                    return b
        frozen = _get_emd_frozen_lookup()
        if frozen is not None:
            bucket = int(frozen.bucket([h[0], h[1]], list(board)))
            if bucket >= 0:
                return bucket
            global _EMD_FROZEN_MISS_COUNT
            _EMD_FROZEN_MISS_COUNT += 1
            if _emd_mirror_fallback() or _emd_allow_silent_fallback():
                from pokerbot.encoding.postflop_bucketer import fast_made_hand_bucket

                return fast_made_hand_bucket((h[0], h[1]), tuple(board), _POSTFLOP_NUM_BUCKETS)
            if _emd_fail_on_miss():
                return -1
            from pokerbot.encoding.postflop_bucketer import fast_made_hand_bucket

            return fast_made_hand_bucket((h[0], h[1]), tuple(board), _POSTFLOP_NUM_BUCKETS)
        return _get_emd_bucketer().bucket((h[0], h[1]), tuple(board))
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
    cfg = {
        "betting_abstraction": bool(_BETTING_ABSTRACTION),
        "postflop_buckets": bool(_POSTFLOP_BUCKETS_ENABLED),
        "postflop_mode": _POSTFLOP_MODE,
        "postflop_num_buckets": int(_POSTFLOP_NUM_BUCKETS),
        "preflop_lossless": bool(weva_encoder._PREFLOP_LOSSLESS),
        "preflop_lossless_betting": bool(weva_encoder._PREFLOP_LOSSLESS_BETTING),
    }
    if _POSTFLOP_MODE == "emd":
        frozen_path = os.environ.get("POKERBOT_POSTFLOP_EMD_FROZEN", "")
        cfg["postflop_emd_frozen_path"] = frozen_path
        cfg["postflop_emd_frozen_s3"] = os.environ.get("POKERBOT_POSTFLOP_EMD_FROZEN_S3", "")
        cfg["postflop_emd_table_id"] = os.environ.get("POKERBOT_POSTFLOP_EMD_TABLE_ID", "")
        cfg["postflop_emd_frozen_sha256"] = ""
        if frozen_path and os.path.exists(frozen_path):
            with open(frozen_path, "rb") as f:
                cfg["postflop_emd_frozen_sha256"] = hashlib.sha256(f.read()).hexdigest()
        cfg["postflop_emd_frozen_miss_count"] = int(_EMD_FROZEN_MISS_COUNT)
        cfg["postflop_emd_frozen_fail_on_miss"] = bool(_emd_fail_on_miss())
        cfg["postflop_emd_frozen_mirror_fallback"] = bool(_emd_mirror_fallback())
        # Blueprint-v2 river OCHS: emitted ONLY when enabled so the default stamp stays
        # byte-identical to pre-river-OCHS checkpoints (the depth_bucketed pattern). The serve
        # guard (check_blueprint_serve_env) also does an explicit two-sided default-False check.
        river_ochs_path = os.environ.get("POKERBOT_RIVER_OCHS", "")
        if river_ochs_path:
            cfg["river_ochs"] = True
            cfg["river_ochs_path"] = river_ochs_path
            cfg["river_ochs_sha256"] = ""
            if os.path.exists(river_ochs_path):
                with open(river_ochs_path, "rb") as f:
                    cfg["river_ochs_sha256"] = hashlib.sha256(f.read()).hexdigest()
    # Depth-aware blueprint (A2): emit the stamp fields ONLY when enabled, so the default (off) stamp stays
    # byte-identical to legacy non-depth checkpoints (and the golden config). The serve guard
    # (check_blueprint_serve_env) compares these with a `.get(default=False)` so a missing key reads as off.
    if _DEPTH_BUCKETS:
        from pokerbot.engine.jax_weva_fast import _DEPTH_EDGES  # lazy: avoids an import cycle
        cfg["depth_bucketed"] = True
        cfg["depth_edges"] = list(_DEPTH_EDGES)
    return cfg


def _made_hand_only_active() -> bool:
    """PRECISE predicate for the made-hand-only ('fast') postflop abstraction.

    True ONLY when postflop bucketing is actually ON and the mode is the made-hand 'fast' bucketer.
    If ``postflop_buckets`` is OFF there is no bucketing to be 'made-hand-only' — a lossless/non-bucketed
    run carries no draw-equity leak and must NOT be blocked. This is the predicate the HARD ship guard
    (``require_potential_aware_buckets``) uses; the advisory ``warn_if_made_hand_only`` keeps a LOOSER
    mode-only predicate (see its docstring) so its pinned test stays green."""
    cfg = active_abstraction_config()
    return bool(cfg["postflop_buckets"]) and cfg.get("postflop_mode") == "fast"


def require_potential_aware_buckets(context: str = "blueprint training") -> bool:
    """FAIL-CLOSED ship guard: refuse to train/save a production NLHE blueprint on the made-hand-only
    ('fast') postflop bucketer, which IGNORES DRAW EQUITY (the documented +188 bb/100 structural leak).

    - Not made-hand-only active (precise predicate) -> return False (nothing to guard).
    - Made-hand-only active + POKERBOT_ALLOW_MADE_HAND_BUCKETS acknowledged -> emit the visible
      RuntimeWarning (so a deliberate smoke is still loud) and return True.
    - Made-hand-only active + NOT acknowledged -> raise RuntimeError with an actionable message.
    """
    if not _made_hand_only_active():
        return False
    msg = (
        f"[postflop-abstraction] {context}: POKERBOT_POSTFLOP_MODE=fast = MADE-HAND-ONLY buckets "
        "(ignores draw equity = documented +188 bb/100 structural leak). For a production blueprint set "
        "POKERBOT_POSTFLOP_MODE=emd (potential-aware) with POKERBOT_POSTFLOP_EMD_TABLE=<centroids.json> "
        "(fit it offline via scripts/precompute_emd_buckets.py), or set POKERBOT_ALLOW_MADE_HAND_BUCKETS=1 "
        "to acknowledge a deliberate made-hand run."
    )
    if os.environ.get("POKERBOT_ALLOW_MADE_HAND_BUCKETS", "0").lower() in ("0", "", "false", "no", "off"):
        raise RuntimeError(msg)
    import warnings
    warnings.warn(msg, RuntimeWarning, stacklevel=2)
    return True


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
    # Effective-STARTING-stack-depth bucket (A2), hashed into the PREFLOP key ONLY when POKERBOT_DEPTH_BUCKETS
    # is set. -1 => not computed / depth-buckets off ⇒ not hashed (digest byte-identical). See depth_bucket().
    depth_bucket: int = -1

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
                # Depth-aware preflop key (A2): hash the effective-starting-depth bucket ONLY when enabled
                # and computed (>=0). Gated so a non-depth blueprint's preflop digest is byte-unchanged.
                if _DEPTH_BUCKETS and self.depth_bucket >= 0:
                    h.update(b"D")
                    h.update(int(self.depth_bucket).to_bytes(1, "little"))
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
    from pokerbot.engine.jax_weva_fast import betting_abstraction_key, betting_history_key, depth_bucket

    dbucket = depth_bucket(state) if _DEPTH_BUCKETS else -1
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
        depth_bucket=dbucket,
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
