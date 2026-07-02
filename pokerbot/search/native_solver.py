"""Native (Rust) drop-in for the np_engine hot path used by the Pluribus exploitation battery.

The bottleneck of the pluribus battery (profiled: ~70% of wall time) is the numpy engine state machine
(``apply_action_np`` / ``legal_action_mask_np`` / ``sample_chance_np`` / showdown ``compute_returns_np``),
called ~700k+ times per hand-pair from the CFR recursion (``pluribus_search._cfv``), the gadget leaf
playouts (``pluribus_rollout._playout``), and the AIVAT ``_leaf_q`` baseline. ``tools/battery_native`` is a
bit-identical Rust port (Tier A) of exactly those primitives.

This wrapper:
  * locates + imports the compiled ``battery_native`` extension (aliasing the cdylib to ``.pyd`` on first
    import, like ``aivat_native``), and runs a startup self-check;
  * exposes ``apply_action_np`` / ``legal_action_mask_np`` / ``sample_chance_np`` / ``is_chance_node_np`` /
    ``evaluate_seven`` with the SAME signatures + return types (numpy ``EnvState`` / arrays) as np_engine,
    so it is a true drop-in;
  * marshals the numpy ``EnvState`` <-> a flat ``int64`` buffer (one contiguous copy per call) and
    reconstructs the ``EnvState`` with the EXACT dtypes + 0-d-scalar shapes np_engine produces, so every
    downstream key (``_infoset_key`` / ``_leaf_key`` / blueprint fingerprint) is byte-unchanged;
  * falls back to the pure-Python np_engine when the extension is unavailable (``available()`` is False).

The RNG-bearing solver Monte-Carlo stream (numpy PCG64 in ``pluribus_search`` / the gadget) and all float
reductions stay in Python — only the integer-exact engine transitions + showdown scoring are native. The
exploit math (``opponent/*``) is never touched.
"""

from __future__ import annotations

import os
import warnings

import numpy as np

from pokerbot.engine.jax_engine import EnvState
from pokerbot.engine.np_engine import (
    _FIELD_DTYPES,
    _SCALAR_FIELDS,
    apply_action_np as _apply_action_py,
    compute_returns_np as _compute_returns_py,
    evaluate_seven as _evaluate_seven_py,
    is_chance_node_np as _is_chance_node_py,
    legal_action_mask_np as _legal_mask_py,
    sample_chance_np as _sample_chance_py,
)

# ---------------------------------------------------------------------------
# Extension loading (mirror aivat_native's locate+alias+import pattern)
# ---------------------------------------------------------------------------
_EXT = None
_AVAILABLE = False
_SELFCHECK_OK = False
# Runtime toggle: even when the extension is available + self-checked, the native path is used only
# when ENABLED. The solver / battery flip this via enable()/disable() so the flag-gated drop-in is
# opt-out-able at runtime without re-import. Default: ON iff available (set in _init after selfcheck).
_ENABLED = False
# SE8: the enable flag is PROCESS-GLOBAL (the hot modules dispatch through this module, not per-solver),
# so two solvers built with conflicting ``native=`` policies clobber each other — last writer wins. That
# is output-safe (the native core is byte/stat-equivalent to np_engine either way) but surprising in a
# mixed live process; track the last EXPLICIT policy and warn ONCE when it gets flipped, so the
# shared-policy behaviour is visible rather than silent.
_LAST_EXPLICIT_POLICY: bool | None = None
_POLICY_CLASH_WARNED = False


def _note_explicit_policy(on: bool) -> None:
    global _LAST_EXPLICIT_POLICY, _POLICY_CLASH_WARNED
    if (_LAST_EXPLICIT_POLICY is not None and _LAST_EXPLICIT_POLICY != on
            and not _POLICY_CLASH_WARNED):
        _POLICY_CLASH_WARNED = True
        warnings.warn(
            f"native engine policy flipped to {'ON' if on else 'OFF'} after being explicitly set to "
            f"{'ON' if _LAST_EXPLICIT_POLICY else 'OFF'}: the flag is PROCESS-GLOBAL, so all live "
            "solvers now share this policy (output is byte/stat-equivalent either way).",
            RuntimeWarning, stacklevel=3,
        )
    _LAST_EXPLICIT_POLICY = on


def enable() -> bool:
    """Turn on the native engine path (no-op if the extension isn't available). Returns is_enabled()."""
    global _ENABLED
    if _AVAILABLE:
        _ENABLED = True
    _note_explicit_policy(True)
    return _ENABLED


def disable() -> None:
    """Force the pure-Python np_engine path (used by the battery's --no-native and tests)."""
    global _ENABLED
    _ENABLED = False
    _note_explicit_policy(False)


def is_enabled() -> bool:
    return bool(_AVAILABLE and _ENABLED)


def _try_load():
    global _EXT
    if os.environ.get("POKERBOT_NO_NATIVE_BATTERY"):
        return None
    here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    rel = os.path.join(here, "tools", "battery_native", "target", "release")
    pyd = os.path.join(rel, "battery_native.pyd")
    dll = os.path.join(rel, "battery_native.dll")
    so = os.path.join(rel, "battery_native.so")
    try:
        import battery_native as ext  # already on sys.path
        # A stale root/ABI-tagged .pyd can shadow a freshly rebuilt release dll before the fallback
        # freshness check below gets a chance to copy dll -> pyd. If the release dll is newer than the
        # imported extension, ignore this import and fall through to the explicit target loader.
        ext_path = getattr(ext, "__file__", None)
        if ext_path and os.path.exists(dll) and os.path.exists(ext_path):
            if os.path.getmtime(dll) > os.path.getmtime(ext_path):
                raise ImportError(
                    f"stale battery_native extension {ext_path}; release dll is newer"
                )
        return ext
    except Exception as _e:  # noqa: BLE001
        if os.environ.get("POKERBOT_NATIVE_DEBUG"):
            import traceback as _tb
            print(f"[native_solver] import battery_native failed: {_e!r}\n{_tb.format_exc()}", flush=True)
        pass
    try:
        # Refresh the .pyd from the freshly built .dll when the dll is newer (or the pyd is missing).
        # Without the mtime check a stale .pyd silently shadows a rebuilt .dll (e.g. after adding the
        # SubgameSolver class), so the new symbols never load. cdylib output is .dll; CPython imports .pyd.
        if os.path.exists(dll):
            need_copy = (not os.path.exists(pyd)
                         or os.path.getmtime(dll) > os.path.getmtime(pyd))
            if need_copy:
                import shutil
                try:
                    shutil.copyfile(dll, pyd)
                except OSError:
                    pass  # pyd locked by another process already importing it; use the existing one
        elif not os.path.exists(pyd) and not os.path.exists(so):
            return None
        target = pyd if os.path.exists(pyd) else so
        import importlib.util
        import sys
        spec = importlib.util.spec_from_file_location("battery_native", target)
        if spec is None or spec.loader is None:
            return None
        mod = importlib.util.module_from_spec(spec)
        sys.modules["battery_native"] = mod
        spec.loader.exec_module(mod)
        return mod
    except Exception as _e2:  # noqa: BLE001
        if os.environ.get("POKERBOT_NATIVE_DEBUG"):
            import traceback as _tb
            print(f"[native_solver] importlib fallback failed: {_e2!r}\n{_tb.format_exc()}", flush=True)
        return None


# Field layout MUST match tools/battery_native/src/lib.rs::flat offsets exactly.
_NUM_PLAYERS = 6
_MAX_BOARD = 5
_MAX_HISTORY = 64

# Compute flat offsets (mirror lib.rs::flat).
_O_HOLES = 0
_O_BOARD = _O_HOLES + 12
_O_SEAT_STATUS = _O_BOARD + _MAX_BOARD
_O_STACKS = _O_SEAT_STATUS + _NUM_PLAYERS
_O_COMMITTED_STREET = _O_STACKS + _NUM_PLAYERS
_O_COMMITTED_TOTAL = _O_COMMITTED_STREET + _NUM_PLAYERS
_O_STREET = _O_COMMITTED_TOTAL + _NUM_PLAYERS
_O_POT_DEAD = _O_STREET + 1
_O_CURRENT_BET = _O_POT_DEAD + 1
_O_MIN_RAISE = _O_CURRENT_BET + 1
_O_BUTTON = _O_MIN_RAISE + 1
_O_ACTION_SEAT = _O_BUTTON + 1
_O_LAST_AGGRESSOR = _O_ACTION_SEAT + 1
_O_ACTED = _O_LAST_AGGRESSOR + 1
_O_LAST_RAISE_FULL = _O_ACTED + _NUM_PLAYERS
_O_DECK = _O_LAST_RAISE_FULL + 1
_O_DECK_PTR = _O_DECK + 52
_O_RNG_KEY = _O_DECK_PTR + 1
_O_HIST_SEAT = _O_RNG_KEY + 2
_O_HIST_ACTION = _O_HIST_SEAT + _MAX_HISTORY
_O_HIST_AMOUNT = _O_HIST_ACTION + _MAX_HISTORY
_O_HIST_LEN = _O_HIST_AMOUNT + _MAX_HISTORY
_O_DONE = _O_HIST_LEN + 1
_O_REWARDS = _O_DONE + 1
_O_PENDING_DEAL = _O_REWARDS + _NUM_PLAYERS
_FLAT_LEN = _O_PENDING_DEAL + 1


def _to_flat(state: EnvState) -> np.ndarray:
    """Pack a numpy EnvState into the flat int64 buffer the native core consumes.

    Hot path: the EnvState fields are already contiguous numpy arrays (from ``to_numpy_state`` /
    ``_replace``) with the engine's dtypes, so we assign them straight into the int64 buffer (numpy
    casts on assignment) WITHOUT per-field ``np.asarray``/``reshape`` — those allocations dominated the
    marshaling profile. Scalars are read via ``.item()`` (no intermediate array). ``holes`` is the only
    2-D field; ``.reshape(-1)`` is a free view."""
    b = np.empty(_FLAT_LEN, dtype=np.int64)
    b[_O_HOLES:_O_HOLES + 12] = state.holes.reshape(-1)
    b[_O_BOARD:_O_BOARD + _MAX_BOARD] = state.board
    b[_O_SEAT_STATUS:_O_SEAT_STATUS + _NUM_PLAYERS] = state.seat_status
    b[_O_STACKS:_O_STACKS + _NUM_PLAYERS] = state.stacks
    b[_O_COMMITTED_STREET:_O_COMMITTED_STREET + _NUM_PLAYERS] = state.committed_street
    b[_O_COMMITTED_TOTAL:_O_COMMITTED_TOTAL + _NUM_PLAYERS] = state.committed_total
    b[_O_STREET] = state.street.item()
    b[_O_POT_DEAD] = state.pot_dead.item()
    b[_O_CURRENT_BET] = state.current_bet.item()
    b[_O_MIN_RAISE] = state.min_raise.item()
    b[_O_BUTTON] = state.button.item()
    b[_O_ACTION_SEAT] = state.action_seat.item()
    b[_O_LAST_AGGRESSOR] = state.last_aggressor.item()
    b[_O_ACTED:_O_ACTED + _NUM_PLAYERS] = state.acted
    b[_O_LAST_RAISE_FULL] = state.last_raise_full.item()
    b[_O_DECK:_O_DECK + 52] = state.deck
    b[_O_DECK_PTR] = state.deck_ptr.item()
    b[_O_RNG_KEY:_O_RNG_KEY + 2] = state.rng_key.view(np.uint32)[:2]
    b[_O_HIST_SEAT:_O_HIST_SEAT + _MAX_HISTORY] = state.hist_seat
    b[_O_HIST_ACTION:_O_HIST_ACTION + _MAX_HISTORY] = state.hist_action
    b[_O_HIST_AMOUNT:_O_HIST_AMOUNT + _MAX_HISTORY] = state.hist_amount
    b[_O_HIST_LEN] = state.hist_len.item()
    b[_O_DONE] = state.done.item()
    b[_O_REWARDS:_O_REWARDS + _NUM_PLAYERS] = state.rewards
    b[_O_PENDING_DEAL] = state.pending_deal.item()
    return b


def _scalar(value, dtype) -> np.ndarray:
    return np.asarray(value, dtype=dtype)


def _from_flat(b) -> EnvState:
    """Reconstruct a numpy EnvState (exact dtypes + 0-d scalar shapes) from the flat buffer.

    Mirrors np_engine's field dtypes and which fields are 0-d scalars vs arrays so every downstream
    key/hash is byte-identical to the pure-Python engine output. ``b`` is a ``bytes`` of int64 LE
    (from the Rust core) or an int64 array."""
    if isinstance(b, (bytes, bytearray, memoryview)):
        b = np.frombuffer(b, dtype=np.int64)
    else:
        b = np.asarray(b, dtype=np.int64)
    holes = b[_O_HOLES:_O_HOLES + 12].astype(np.int32).reshape(_NUM_PLAYERS, 2)
    rng = np.array([b[_O_RNG_KEY] & 0xFFFFFFFF, b[_O_RNG_KEY + 1] & 0xFFFFFFFF], dtype=np.uint32)
    out = {
        "holes": holes,
        "board": b[_O_BOARD:_O_BOARD + _MAX_BOARD].astype(np.int32),
        "seat_status": b[_O_SEAT_STATUS:_O_SEAT_STATUS + _NUM_PLAYERS].astype(np.int8),
        "stacks": b[_O_STACKS:_O_STACKS + _NUM_PLAYERS].astype(np.int32),
        "committed_street": b[_O_COMMITTED_STREET:_O_COMMITTED_STREET + _NUM_PLAYERS].astype(np.int32),
        "committed_total": b[_O_COMMITTED_TOTAL:_O_COMMITTED_TOTAL + _NUM_PLAYERS].astype(np.int32),
        "street": _scalar(int(b[_O_STREET]), np.int32),
        "pot_dead": _scalar(int(b[_O_POT_DEAD]), np.int32),
        "current_bet": _scalar(int(b[_O_CURRENT_BET]), np.int32),
        "min_raise": _scalar(int(b[_O_MIN_RAISE]), np.int32),
        "button": _scalar(int(b[_O_BUTTON]), np.int32),
        "action_seat": _scalar(int(b[_O_ACTION_SEAT]), np.int32),
        "last_aggressor": _scalar(int(b[_O_LAST_AGGRESSOR]), np.int32),
        "acted": b[_O_ACTED:_O_ACTED + _NUM_PLAYERS].astype(np.bool_),
        "last_raise_full": _scalar(bool(b[_O_LAST_RAISE_FULL]), np.bool_),
        "deck": b[_O_DECK:_O_DECK + 52].astype(np.int32),
        "deck_ptr": _scalar(int(b[_O_DECK_PTR]), np.int32),
        "rng_key": rng,
        "hist_seat": b[_O_HIST_SEAT:_O_HIST_SEAT + _MAX_HISTORY].astype(np.int32),
        "hist_action": b[_O_HIST_ACTION:_O_HIST_ACTION + _MAX_HISTORY].astype(np.int32),
        "hist_amount": b[_O_HIST_AMOUNT:_O_HIST_AMOUNT + _MAX_HISTORY].astype(np.int32),
        "hist_len": _scalar(int(b[_O_HIST_LEN]), np.int32),
        "done": _scalar(bool(b[_O_DONE]), np.bool_),
        "rewards": b[_O_REWARDS:_O_REWARDS + _NUM_PLAYERS].astype(np.int32),
        "pending_deal": _scalar(bool(b[_O_PENDING_DEAL]), np.bool_),
    }
    return EnvState(**out)


# ---------------------------------------------------------------------------
# Public drop-in API (native when available, else pure-Python np_engine)
# ---------------------------------------------------------------------------


def apply_action_np(state: EnvState, action: int) -> EnvState:
    if not (_AVAILABLE and _ENABLED):
        return _apply_action_py(state, int(action))
    out = _EXT.apply_action(_to_flat(state).tobytes(), int(action))
    return _from_flat(out)


def sample_chance_np(state: EnvState) -> EnvState:
    if not (_AVAILABLE and _ENABLED):
        return _sample_chance_py(state)
    out = _EXT.sample_chance(_to_flat(state).tobytes())
    return _from_flat(out)


def legal_action_mask_np(state: EnvState, seat=None) -> np.ndarray:
    if not (_AVAILABLE and _ENABLED):
        return _legal_mask_py(state, seat)
    s = -1 if seat is None else int(seat)
    out = _EXT.legal_action_mask(_to_flat(state).tobytes(), s)
    return np.asarray(out, dtype=np.bool_)


def is_chance_node_np(state: EnvState) -> bool:
    if not (_AVAILABLE and _ENABLED):
        return _is_chance_node_py(state)
    return bool(_EXT.is_chance_node(_to_flat(state).tobytes()))


def evaluate_seven(cards: np.ndarray):
    if not (_AVAILABLE and _ENABLED):
        return _evaluate_seven_py(cards)
    c = np.ascontiguousarray(cards, dtype=np.int32).reshape(-1)
    return np.int32(_EXT.evaluate_seven([int(x) for x in c[:7]]))


def available() -> bool:
    return _AVAILABLE


# ---------------------------------------------------------------------------
# Native uniform-mode continuation-leaf playout (POKERBOT_NATIVE_ROLLOUT).
#
# Moves the ENTIRE pluribus_rollout._playout loop (uniform path) into the Rust core as one call,
# eliminating the per-step EnvState marshaling (_to_flat/_from_flat) that otherwise dominates. The
# fingerprint (blueprint lookup key) is computed natively, byte-identical to fingerprint_fast (gated
# + fuzz-verified). RNG is a fast native xoshiro seeded from base_seed — statistical equivalence,
# NOT a replay of numpy's PCG64 (the leaf value estimates the SAME expectation).
# Default OFF; the Python loop in pluribus_rollout is unchanged when this is not engaged.
# ---------------------------------------------------------------------------
_EMD_NATIVE_GATE_WARNED = False
_EMD_NATIVE_GATE_OPENED_LOGGED = False


def _emd_native_key_config() -> tuple | None:
    """``(blob_path, miss_made_hand)`` for the native EMD frozen-lookup keying, or ``None`` when the
    frozen blob is not configured / not on disk.

    ``miss_made_hand`` mirrors ``jax_weva_encode.compute_postflop_bucket``'s frozen-miss order
    EXACTLY (imported, not re-implemented): made-hand fallback when the mirror or legacy-silent
    fallback is on OR fail-on-miss is off; otherwise the Python path keys the miss as bucket=-1
    (the preflop 'w' branch) and the native side replicates that."""
    path = os.environ.get("POKERBOT_POSTFLOP_EMD_FROZEN") or ""
    if not path or not os.path.exists(path):
        return None
    from pokerbot.engine.jax_weva_encode import (  # noqa: PLC0415
        _emd_allow_silent_fallback,
        _emd_fail_on_miss,
        _emd_mirror_fallback,
    )
    miss_made_hand = bool(
        _emd_mirror_fallback() or _emd_allow_silent_fallback() or not _emd_fail_on_miss()
    )
    return path, miss_made_hand


def _native_supports_emd() -> bool:
    """True iff the loaded battery_native binary REPORTS EMD frozen-lookup keying support.

    A binary built before the EMD port has no ``native_caps`` symbol at all, so this fails closed
    on stale .pyds without any version arithmetic."""
    if _EXT is None:
        return False
    try:
        caps = set(_EXT.native_caps())
    except Exception:  # noqa: BLE001
        return False
    return {"emd_fingerprint", "emd_subgame", "emd_rollout"} <= caps


def _native_fingerprint_matches_serve_abstraction() -> bool:
    """FAIL-CLOSED gate for the native blueprint-consuming fast paths under the EMD serve regime.

    HISTORY: ``fingerprint.rs`` was originally a byte-identical port of the MADE-HAND ('fast')
    bucketer only; under ``POKERBOT_POSTFLOP_MODE=emd`` (the DEPLOYED regime since the 2026-07-02
    EMD-K160 promotion) its flop/turn keys were the wrong abstraction and COLLIDED INTO wrong rows
    (measured 12/15 flop, 14/15 turn), so this gate disabled the native fast paths and served the
    ~45x-slower Python search path.

    NOW: the native core carries the SAME frozen-lookup keying the trainer wrote the blueprint rows
    under (``fingerprint::fingerprint_fast_depth_emd``, parity-gated by
    ``scripts/probe_native_emd_parity.py``). The gate therefore OPENS under EMD iff BOTH hold:
      * the loaded binary reports EMD support (``native_caps`` — stale .pyds fail closed), AND
      * the frozen blob is configured and present on disk (``POKERBOT_POSTFLOP_EMD_FROZEN``).
    The engine builders then pass the blob path + miss semantics through the FFI so every native
    blueprint lookup keys EMD-correctly. On ANY mismatch (old binary, missing blob) the gate stays
    fail-closed and the Python (EMD-correct) path serves. POKERBOT_NATIVE_ALLOW_EMD_MISKEY=1 remains
    as the historical experiments-only override (it forces the gate open WITHOUT the EMD keying and
    should no longer be needed).
    """
    if os.environ.get("POKERBOT_POSTFLOP_MODE", "fast").lower() != "emd":
        return True
    if os.environ.get("POKERBOT_NATIVE_ALLOW_EMD_MISKEY", "0").lower() in ("1", "true", "yes", "on"):
        return True
    global _EMD_NATIVE_GATE_WARNED, _EMD_NATIVE_GATE_OPENED_LOGGED
    if _native_supports_emd() and _emd_native_key_config() is not None:
        if not _EMD_NATIVE_GATE_OPENED_LOGGED:
            _EMD_NATIVE_GATE_OPENED_LOGGED = True
            path, miss_made_hand = _emd_native_key_config()
            print("[native_solver] POKERBOT_POSTFLOP_MODE=emd: native fast paths ENABLED with EMD "
                  f"frozen-lookup keying (blob={path}, miss_made_hand={miss_made_hand}) — parity-"
                  "gated port of the trainer's keying.", flush=True)
        return True
    if not _EMD_NATIVE_GATE_WARNED:
        _EMD_NATIVE_GATE_WARNED = True
        print("[native_solver] POKERBOT_POSTFLOP_MODE=emd: native subgame/rollout fast paths DISABLED "
              "(this battery_native binary does not report EMD keying support, or "
              "POKERBOT_POSTFLOP_EMD_FROZEN is unset/missing) — serving through the Python search "
              "path (EMD-correct). Rebuild the crate (--features python) to re-enable. "
              "POKERBOT_NATIVE_ALLOW_EMD_MISKEY=1 overrides (experiments only).", flush=True)
    return False


def _emd_engine_kwargs() -> dict:
    """FFI kwargs plumbing the EMD keying into a native engine ctor; {} outside the EMD regime (or
    when the binary/blob can't support it — the availability gate already fails closed there)."""
    if os.environ.get("POKERBOT_POSTFLOP_MODE", "fast").lower() != "emd":
        return {}
    if not _native_supports_emd():
        return {}
    cfg = _emd_native_key_config()
    if cfg is None:
        return {}
    return {"emd_frozen_path": cfg[0], "emd_miss_made_hand": cfg[1]}


def native_rollout_available() -> bool:
    """True iff the native extension exposes the PlayoutEngine (built) AND is enabled."""
    return bool(_AVAILABLE and _ENABLED and _EXT is not None and hasattr(_EXT, "PlayoutEngine")
                and _native_fingerprint_matches_serve_abstraction())


# Per-adapter PlayoutEngine cache (built once per blueprint snapshot identity; the blueprint is large
# and the HashMap build is the only non-trivial setup cost). Keyed by id(adapter) with a weak hold.
_PLAYOUT_ENGINE_CACHE: dict[int, object] = {}


def _build_playout_engine(adapter, num_buckets: int, max_playout_depth: int):
    import numpy as _np

    allowed_pre = adapter._allowed_pre
    allowed_post = adapter._allowed_post
    if allowed_pre is None or allowed_post is None:
        return None  # native path requires client_bet_sizing (the eval config)
    bp = {
        bytes(k): [float(x) for x in _np.asarray(v, dtype=_np.float64).reshape(-1)[:11]]
        for k, v in adapter.snapshot.items()
    }
    try:
        return _EXT.PlayoutEngine(
            bp,
            [bool(x) for x in allowed_pre],
            [bool(x) for x in allowed_post],
            float(adapter.bias_mult),
            float(adapter.bias_floor),
            int(max_playout_depth),
            int(num_buckets),
            **_emd_engine_kwargs(),
        )
    except TypeError:
        # Stale native binary predating the EMD keying kwargs: never construct a mis-keyed engine
        # under the EMD regime — fall back to the Python loop (the availability gate should already
        # have failed closed; this is defense in depth).
        if _emd_engine_kwargs():
            return None
        raise


def native_uniform_leaf_value(
    env, hero: int, adapter, base_seed: int, rollout_samples: int,
    resample_board: bool, max_playout_depth: int, num_buckets: int,
) -> float:
    """Native uniform continuation-leaf value (hero bb). Builds/caches a PlayoutEngine per adapter.

    Returns None if the native path is unavailable or the adapter is not in the supported (bounded,
    client_bet_sizing) config, so the caller falls back to the Python loop."""
    if not native_rollout_available():
        return None
    ckey = (id(adapter), int(max_playout_depth), int(num_buckets))
    eng = _PLAYOUT_ENGINE_CACHE.get(ckey)
    if eng is None:
        eng = _build_playout_engine(adapter, num_buckets, max_playout_depth)
        if eng is None:
            return None
        _PLAYOUT_ENGINE_CACHE[ckey] = eng
    sb = _to_flat(env).tobytes()
    return float(eng.leaf_value(sb, int(hero), int(base_seed) & 0xFFFFFFFFFFFFFFFF,
                                int(rollout_samples), bool(resample_board)))


# ---------------------------------------------------------------------------
# Native depth-limited CFR subgame solve (POKERBOT_NATIVE_SUBGAME).
#
# Runs the ENTIRE pluribus_search._cfv loop (opponent-hole sampling + CFR+ recursion + blueprint-mixed
# opponent prior + continuation-gadget leaf) in Rust as one call per hero decision, eliminating the
# per-node _to_flat/_from_flat marshaling (~20% of a run) AND the interpreted CFR orchestration. Only
# the GTO-base path (opponent_model is None, uniform belief) is native; the exploit tail stays in
# Python (firewall). Statistically equivalent, not bit-identical (fast native RNG). Default OFF.
# ---------------------------------------------------------------------------
def native_subgame_available() -> bool:
    """True iff the native extension exposes SubgameSolver (built) AND the native path is enabled."""
    return bool(_AVAILABLE and _ENABLED and _EXT is not None and hasattr(_EXT, "SubgameSolver")
                and _native_fingerprint_matches_serve_abstraction())


_SUBGAME_ENGINE_CACHE: dict[tuple, object] = {}


def _build_subgame_engine(adapter, num_buckets: int, leaf_max_depth: int):
    import numpy as _np

    allowed_pre = adapter._allowed_pre
    allowed_post = adapter._allowed_post
    if allowed_pre is None or allowed_post is None:
        return None  # native path requires client_bet_sizing (the eval config)
    bp = {
        bytes(k): [float(x) for x in _np.asarray(v, dtype=_np.float64).reshape(-1)[:11]]
        for k, v in adapter.snapshot.items()
    }
    try:
        return _EXT.SubgameSolver(
            bp,
            [bool(x) for x in allowed_pre],
            [bool(x) for x in allowed_post],
            float(adapter.bias_mult),
            float(adapter.bias_floor),
            int(leaf_max_depth),
            int(num_buckets),
            **_emd_engine_kwargs(),
        )
    except TypeError:
        # Stale native binary predating the EMD keying kwargs (see _build_playout_engine).
        if _emd_engine_kwargs():
            return None
        raise


def native_subgame_solve(
    base_state, hero: int, adapter, *, public_seed: int, max_iterations: int, max_depth: int,
    blueprint_mix: float, rollout_samples: int, resample_board: bool, max_playout_depth: int,
    num_buckets: int, linear_averaging: bool = True, repair_deck: bool = False,
    two_sided: bool = False, decouple_sampler: bool = False, range_aware: bool = False,
    range_uniform_floor: float = 0.25, ensemble_streams: int = 1, leaf_cache_worlds: int = 1,
    avg_skip_frac: float = 0.0, leaf_biased_continuations: bool = False, leaf_bias_mult: float = 5.0,
):
    """Native root action-probability vector (length 11) for a depth-limited subgame solve, or None
    if the native path is unavailable / the adapter is not in the supported (bounded, client_bet_sizing)
    config — in which case the caller runs the Python CFR loop.

    ``linear_averaging`` mirrors PluribusReBeLConfig.linear_averaging (CFR+ recency-weighted strategy
    average); the native core must apply the SAME weighting or the python/native crosscheck diverges.

    ``decouple_sampler`` (Q2) gives the opponent-hole deal its own deterministic sub-stream; ``range_aware``
    (Q3) weights that deal by the blueprint continuation range mixed with ``range_uniform_floor`` uniform.

    ``leaf_cache_worlds`` (BUG-1 fix) makes the public-keyed leaf cache a running mean over the first M
    visiting hole-worlds (1 = legacy first-touch freeze, byte-identical); ``avg_skip_frac`` (BUG-5
    partial) starts strat_sum accumulation at iteration floor(frac * K) (0.0 = legacy, byte-identical).

    ``leaf_biased_continuations`` (R2) turns every depth-limit leaf into the Pluribus 4-way
    continuation-CHOICE node per live seat, solved by the same subgame CFR; the leaf playout applies
    ``leaf_bias_mult`` (Pluribus 5 / Modicum 10) to the chosen bias group. False (default) = the legacy
    uniform-draw leaf, byte-identical. GTO path only — the exploit FFI does not accept these knobs."""
    if not native_subgame_available():
        return None
    ckey = (id(adapter), int(max_playout_depth), int(num_buckets))
    eng = _SUBGAME_ENGINE_CACHE.get(ckey)
    if eng is None:
        eng = _build_subgame_engine(adapter, num_buckets, max_playout_depth)
        if eng is None:
            return None
        _SUBGAME_ENGINE_CACHE[ckey] = eng
    sb = _to_flat(base_state).tobytes()
    try:
        probs = eng.solve(
            sb, int(hero), int(public_seed) & 0xFFFFFFFFFFFFFFFF,
            int(max_iterations), int(max_depth), float(blueprint_mix),
            int(rollout_samples), bool(resample_board), bool(linear_averaging),
            bool(repair_deck), bool(two_sided),
            bool(decouple_sampler), bool(range_aware), float(range_uniform_floor),
            int(ensemble_streams),
            int(leaf_cache_worlds), float(avg_skip_frac),
            bool(leaf_biased_continuations), float(leaf_bias_mult),
        )
    except TypeError:
        # Stale native binary built before the linear_averaging / repair_deck / two_sided / decouple /
        # range_aware / leaf_cache_worlds / avg_skip_frac / leaf_biased_continuations args — fall back
        # to Python so a not-yet-rebuilt crate doesn't crash the solve (the Python path applies the same
        # weighting + repair_deck/two_sided + leaf_cache_worlds/avg_skip_frac/leaf_biased_continuations;
        # the Q2/Q3 world-sampling levers are native-only, so the fallback is the uniform-deal base arm).
        return None
    import numpy as _np

    return _np.asarray(probs, dtype=_np.float64)


# ---------------------------------------------------------------------------
# Native EXPLOIT subgame solve (POKERBOT_NATIVE_EXPLOIT).
#
# Same native CFR subgame solve as native_subgame_solve, but the opponent nodes apply a PRECOMPUTED
# pure-frozen exploit deviation (built in pokerbot.search.exploit_table, Python) instead of the GTO
# base. The exploit MATH (context_read / Dirichlet posterior / BOCPD / nash_reference / cap_eff) runs
# in Python; native only LOOKS UP the output frozen vector + the per-actor rnr scalar and applies the
# documented overwrite / blend at opponent nodes (the data-not-code firewall). Native ALSO exposes the
# root per-action CFVs it already computes (for the AIVAT decision-baseline). Default OFF; a stale .pyd
# lacking solve_exploit silently falls back to the Python _cfv loop.
# ---------------------------------------------------------------------------
def native_subgame_exploit_available() -> bool:
    """True iff the native SubgameSolver exposes solve_exploit (a rebuilt .pyd) AND the native path is
    enabled. Probes the bound method on the class so a stale binary silently falls back to Python."""
    if not (_AVAILABLE and _ENABLED and _EXT is not None and hasattr(_EXT, "SubgameSolver")):
        return False
    if not _native_fingerprint_matches_serve_abstraction():
        return False
    return hasattr(_EXT.SubgameSolver, "solve_exploit")


def native_subgame_solve_exploit(
    base_state, hero: int, adapter, *, public_seed: int, max_iterations: int, max_depth: int,
    blueprint_mix: float, rollout_samples: int, resample_board: bool, max_playout_depth: int,
    num_buckets: int, linear_averaging: bool, frozen_table, rnr, frozen_read_response: bool,
    rnr_restrict: bool, want_root_q: bool, repair_deck: bool = False,
    decouple_sampler: bool = False, range_aware: bool = False, range_uniform_floor: float = 0.25,
):
    """Native exploit subgame solve. Returns ``(probs[11], root_q[11] or None)`` or ``None`` when the
    native path is unavailable / the adapter is unsupported / the binary is stale (caller runs Python).

    ``frozen_table`` is the FrozenExploitTable (parallel keys_actor / keys_ctx / keys_mask + flat vals).
    ``rnr`` is the length-6 per-actor scalar list (NaN = overwrite-with-frozen). The native side
    recomputes the context KEY + legal_bits and applies the overwrite / blend against its OWN base."""
    if not native_subgame_exploit_available():
        return None
    ckey = (id(adapter), int(max_playout_depth), int(num_buckets))
    eng = _SUBGAME_ENGINE_CACHE.get(ckey)
    if eng is None:
        # The SAME engine (ctor unchanged — the table is passed PER-CALL, not baked), so the base
        # (model=None) cache key is shared and untouched.
        eng = _build_subgame_engine(adapter, num_buckets, max_playout_depth)
        if eng is None:
            return None
        _SUBGAME_ENGINE_CACHE[ckey] = eng
    sb = _to_flat(base_state).tobytes()
    keys_actor = [int(x) for x in frozen_table.keys_actor]
    keys_ctx = [int(x) for x in frozen_table.keys_ctx]
    keys_mask = [int(x) for x in frozen_table.keys_mask]
    vals = [float(x) for x in frozen_table.vals]
    rnr_list = [float(x) for x in rnr]
    try:
        out = eng.solve_exploit(
            sb, int(hero), int(public_seed) & 0xFFFFFFFFFFFFFFFF,
            int(max_iterations), int(max_depth), float(blueprint_mix),
            int(rollout_samples), bool(resample_board), bool(linear_averaging),
            keys_actor, keys_ctx, keys_mask, vals, rnr_list,
            bool(frozen_read_response), bool(rnr_restrict), bool(want_root_q),
            bool(repair_deck),
            bool(decouple_sampler), bool(range_aware), float(range_uniform_floor),
        )
    except (TypeError, AttributeError):
        # Stale native binary lacking solve_exploit (or a signature mismatch) -> fall back to Python.
        return None
    import numpy as _np

    probs, root_q = out
    probs = _np.asarray(probs, dtype=_np.float64)
    rq = _np.asarray(root_q, dtype=_np.float64) if (want_root_q and root_q) else None
    return probs, rq


# ---------------------------------------------------------------------------
# Startup self-check: native transitions must equal the Python engine on a battery of random hands.
# ---------------------------------------------------------------------------


def _selfcheck(n_hands: int = 40) -> bool:
    """Play random hands with BOTH engines in lockstep; assert every state field is bit-identical.

    Returns True iff native == pure-Python on every transition + every legal mask + chance deal +
    terminal reward. Run once at import (cheap) so a miscompiled / stale extension silently falls back
    instead of corrupting a battery run."""
    try:
        import random as _random

        from pokerbot.engine.game import GameState
        from pokerbot.engine.np_engine import to_numpy_state
        from pokerbot.search.state_bridge import build_subgame_root  # noqa: F401
    except Exception:  # noqa: BLE001
        return False

    def _eq_state(a: EnvState, b: EnvState) -> bool:
        for f in a._fields:
            if not np.array_equal(np.asarray(getattr(a, f)), np.asarray(getattr(b, f))):
                return False
        return True

    rng = _random.Random(12345)
    for _ in range(n_hands):
        gs = GameState.new_hand(button_seat=3, rng=_random.Random(rng.randint(0, 10**9)))
        # build a numpy root from the live GameState via the same bridge the solver uses.
        try:
            from pokerbot.search.state_bridge import build_subgame_root
            from pokerbot.search.pbs_tracker import PBSTracker  # noqa: F401
        except Exception:  # noqa: BLE001
            return False
        # Simplest robust seed: construct an EnvState by stepping the JAX init through to_numpy_state.
        try:
            from pokerbot.engine.jax_engine import init_hand
        except Exception:  # noqa: BLE001
            return False
        import jax  # noqa: F401  (only used by init_hand; battery env already has jax)
        key = jax.random.PRNGKey(rng.randint(0, 10**9))
        jstate = init_hand(key, button=3)
        env = to_numpy_state(jstate)
        depth = 0
        while not bool(env.done) and depth < 200:
            if is_chance_node_np(env):
                e_n = sample_chance_np(env)
                e_p = _sample_chance_py(env)
                if not _eq_state(e_n, e_p):
                    return False
                env = e_p
                continue
            seat = int(env.action_seat)
            if seat < 0:
                break
            m_n = legal_action_mask_np(env, seat)
            m_p = _legal_mask_py(env, seat)
            if not np.array_equal(m_n, m_p):
                return False
            legal = np.flatnonzero(m_p)
            if legal.size == 0:
                break
            a = int(rng.choice(list(legal)))
            e_n = apply_action_np(env, a)
            e_p = _apply_action_py(env, a)
            if not _eq_state(e_n, e_p):
                return False
            env = e_p
            depth += 1
    return True


def _init():
    global _EXT, _AVAILABLE, _SELFCHECK_OK, _ENABLED
    ext = _try_load()
    if ext is None:
        _AVAILABLE = False
        return
    _EXT = ext
    try:
        dims = ext.engine_dims()
        if tuple(dims[:5]) != (_NUM_PLAYERS, 11, _MAX_BOARD, _MAX_HISTORY, _FLAT_LEN):
            warnings.warn(
                f"[battery_native] dim/layout mismatch {dims} -> falling back to Python engine",
                RuntimeWarning, stacklevel=2,
            )
            _AVAILABLE = False
            return
    except Exception:  # noqa: BLE001
        _AVAILABLE = False
        return
    # Temporarily flag available+enabled so the self-check exercises the native path, then confirm.
    _AVAILABLE = True
    _ENABLED = True
    try:
        ok = _selfcheck()
    except Exception:  # noqa: BLE001
        ok = False
    _SELFCHECK_OK = bool(ok)
    if not ok:
        warnings.warn(
            "[battery_native] startup self-check failed -> falling back to pure-Python np_engine",
            RuntimeWarning, stacklevel=2,
        )
        _AVAILABLE = False
        _ENABLED = False
        return
    # Default: enabled iff available + self-check passed. Callers can disable() for the Python path.
    _ENABLED = True


_init()
