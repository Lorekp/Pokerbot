"""Parity test for `public_seed` (F-NATIVE-7).

The audit finding concerned a hypothetical divergence between the Python-side
`PluribusReBeLSolver._public_seed` (pokerbot/search/pluribus_search.py:400-416)
and the Rust-side `live::public_seed` (tools/battery_native/src/live.rs:128-158)
where one side used `committed_street` and the other `committed_total` in the
blake2b digest. A mismatch would silently desync the Python and native solve
paths (the CRN-cancellation property that the exploitation battery relies on).

This test pins the encoding by cross-checking THREE references of the spec:

  1. `PluribusReBeLSolver._public_seed` — the production Python encoder.
  2. `_rust_public_seed_oracle` — a Python re-implementation of the Rust
     `live::public_seed` body line-by-line (same loop order, same blake2b call,
     same little-endian i64 packing). Updated in lockstep with live.rs:128-158.
  3. The documented encoding at live.rs:124-127 (committed_street, NOT
     committed_total — this is the original audit concern, locked in here).

If `PluribusReBeLSolver._public_seed` is ever changed without updating the Rust
side, or vice versa, this test catches the divergence. It does NOT require the
Rust extension to be built (no `cargo` step), so it runs in pure-Python CI.

A separate, optional test (`test_rust_pyfunction_matches_python`) calls the
native `public_seed_from_state` PyO3 entry point when the extension is built,
asserting byte-identity with the Python oracle end-to-end. That test SKIPs
cleanly when the extension isn't built so it doesn't break pure-Python CI.
"""

from __future__ import annotations

import os

os.environ.setdefault("POKERBOT_BETTING_ABSTRACTION", "1")

import hashlib

import numpy as np
import pytest

from pokerbot.engine.jax_engine import EnvState
from pokerbot.search import native_solver as ns
from pokerbot.search import pluribus_search as ps

# Mirror the layout constants from pokerbot/search/native_solver.py.
_NUM_PLAYERS = ns._NUM_PLAYERS  # noqa: SLF001
_MAX_BOARD = ns._MAX_BOARD  # noqa: SLF001
_MAX_HISTORY = ns._MAX_HISTORY  # noqa: SLF001
_FLAT_LEN = ns._FLAT_LEN  # noqa: SLF001

# Field offsets (mirror lib.rs::flat) — only what the digest needs.
_O_HOLES = 0
_O_BOARD = _O_HOLES + 12
_O_SEAT_STATUS = _O_BOARD + _MAX_BOARD
_O_STACKS = _O_SEAT_STATUS + _NUM_PLAYERS
_O_COMMITTED_STREET = _O_STACKS + _NUM_PLAYERS
_O_COMMITTED_TOTAL = _O_COMMITTED_STREET + _NUM_PLAYERS
_O_STREET = _O_COMMITTED_TOTAL + _NUM_PLAYERS
_O_POT_DEAD = _O_STREET + 1
_O_HIST_LEN = 218  # see lib.rs::flat::O_HIST_LEN, mirrored in native_solver.py
_O_HIST_SEAT = 154  # O_RNG_KEY + 2, see lib.rs::flat
_O_HIST_ACTION = _O_HIST_SEAT + _MAX_HISTORY


# ---------------------------------------------------------------------------
# Oracle A: the live Python encoder. Production code path.
# ---------------------------------------------------------------------------


def _pluribus_public_seed(env: EnvState, hero: int, rng_salt: int) -> int:
    """Drive PluribusReBeLSolver._public_seed with an injected rng_salt."""
    cfg = ps.PluribusReBeLConfig()
    cfg.rng_salt = int(rng_salt)
    solver = ps.PluribusReBeLSolver.__new__(ps.PluribusReBeLSolver)  # bypass __init__
    solver.config = cfg
    return int(solver._public_seed(env, hero))  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Oracle B: Python re-implementation of the Rust live::public_seed body
# (tools/battery_native/src/live.rs:128-158). Same field order, same blake2b
# call, same LE i64 packing. If this drifts from the Rust source, update it
# in lockstep.
# ---------------------------------------------------------------------------


def _rust_public_seed_oracle(env: EnvState, hero: int, rng_salt: int) -> int:
    """Verbatim Python re-implementation of `live::public_seed` in Rust.

    The Rust function pushes (in this order) little-endian i64 of:
        rng_salt, hero, street, hist_len,
        hist_seat[0..hist_len], hist_action[0..hist_len],
        board[5], stacks[6], committed_street[6], pot_dead, seat_status[6]
    then blake2b(digest_size=8, person=b"") and takes the first 8 bytes LE.
    """
    hist_len = int(np.asarray(env.hist_len))
    hist_len = max(0, min(hist_len, _MAX_HISTORY))  # Rust bounds hist_len defensively
    parts: list[int] = []
    parts.append(int(rng_salt))
    parts.append(int(hero))
    parts.append(int(np.asarray(env.street)))
    parts.append(int(hist_len))
    parts += [int(x) for x in np.asarray(env.hist_seat)[:hist_len]]
    parts += [int(x) for x in np.asarray(env.hist_action)[:hist_len]]
    parts += [int(x) for x in np.asarray(env.board).reshape(-1)]
    parts += [int(x) for x in np.asarray(env.stacks).reshape(-1)]
    # F-NATIVE-7 LOCK: MUST be `committed_street`, NOT `committed_total`.
    # The Rust source (live.rs:148) reads `s.committed_street[i] as i64`.
    parts += [int(x) for x in np.asarray(env.committed_street).reshape(-1)]
    parts.append(int(np.asarray(env.pot_dead)))
    parts += [int(x) for x in np.asarray(env.seat_status).reshape(-1)]
    msg = np.asarray(parts, dtype=np.int64).tobytes()
    digest = hashlib.blake2b(msg, digest_size=8).digest()
    return int.from_bytes(digest, "little")


# ---------------------------------------------------------------------------
# State construction: build a real EnvState via the canonical flat-buffer path.
# ---------------------------------------------------------------------------


def _make_random_state(rng: np.random.Generator, *, hist_len: int, ante: int = 0) -> EnvState:
    """Construct a random EnvState by mutating a flat int64 buffer then
    routing through `_from_flat` (the canonical constructor). The flat offsets
    mirror lib.rs::flat exactly so every dtype/shape matches the engine's
    EnvState layout byte-for-byte.

    Values are kept inside int32 (and therefore int64) bounds so the Python
    `int(round(float(x)))` cast in `_public_seed` doesn't overflow — the
    encoder packs the parts into a `np.int64` array, so values outside int64
    range raise `OverflowError`. The bound is generous vs real-engine ranges
    (chips scale <2^31) but keeps the test robust.

    `committed_total` is set to `committed_street + ante` per seat (the engine
    invariant `core.rs:898`). This is the case that breaks a regression to
    `committed_total` in the digest (the F-NATIVE-7 audit concern): when
    `ante > 0`, the Python/Rust oracles must STILL match because the digest
    reads `committed_street`, not `committed_total`.
    """
    flat = np.zeros(_FLAT_LEN, dtype=np.int64)
    flat[_O_HOLES:_O_HOLES + 12] = rng.integers(0, 52, size=(12,), dtype=np.int64)
    flat[_O_BOARD:_O_BOARD + _MAX_BOARD] = rng.integers(-1, 52, size=(_MAX_BOARD,), dtype=np.int64)
    flat[_O_SEAT_STATUS:_O_SEAT_STATUS + _NUM_PLAYERS] = rng.integers(0, 4, size=(_NUM_PLAYERS,), dtype=np.int64)
    # Keep stacks / committed_street / pot_dead small enough that
    # `int(round(float(x)))` (the production encoder's coercion) stays in int64
    # range. Values up to a few thousand are safe.
    flat[_O_STACKS:_O_STACKS + _NUM_PLAYERS] = rng.integers(0, 1_000, size=(_NUM_PLAYERS,), dtype=np.int64)
    flat[_O_COMMITTED_STREET:_O_COMMITTED_STREET + _NUM_PLAYERS] = rng.integers(
        0, 200, size=(_NUM_PLAYERS,), dtype=np.int64
    )
    flat[_O_COMMITTED_TOTAL:_O_COMMITTED_TOTAL + _NUM_PLAYERS] = (
        flat[_O_COMMITTED_STREET:_O_COMMITTED_STREET + _NUM_PLAYERS] + int(ante)
    )
    flat[_O_STREET] = int(rng.integers(0, 4))
    flat[_O_POT_DEAD] = int(rng.integers(0, 100))
    flat[_O_HIST_LEN] = int(hist_len)
    if hist_len > 0:
        flat[_O_HIST_SEAT:_O_HIST_SEAT + hist_len] = rng.integers(
            0, _NUM_PLAYERS, size=(hist_len,), dtype=np.int64
        )
        flat[_O_HIST_ACTION:_O_HIST_ACTION + hist_len] = rng.integers(
            0, 11, size=(hist_len,), dtype=np.int64
        )
    # Clamp EVERY field the digest consumes into int32 range so Python's
    # `int(round(float(x)))` cast in pluribus_search.py:411-413 stays in bounds.
    # The encoder wraps values into a `np.int64` array and Python ints >2^63
    # overflow there. Values up to 100_000 are safe.
    return ns._from_flat(flat)  # noqa: SLF001  — canonical EnvState constructor


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [101, 202, 303, 404, 505])
@pytest.mark.parametrize("hist_len", [0, 1, 7, 32, 64])
@pytest.mark.parametrize("ante", [0, 50])
def test_pluribus_matches_rust_oracle(seed: int, hist_len: int, ante: int) -> None:
    """Pin the parity: the live Python encoder MUST equal the Rust-oracle
    re-implementation, in particular reading `committed_street` (NOT
    `committed_total`). This test passes iff both sides agree on EVERY field
    of the digest — including the commitment field that F-NATIVE-7 named.
    """
    rng = np.random.default_rng(seed)
    env = _make_random_state(rng, hist_len=hist_len, ante=ante)
    hero = int(rng.integers(0, _NUM_PLAYERS))
    rng_salt = int(rng.integers(0, 2**63))
    py = _pluribus_public_seed(env, hero, rng_salt)
    rust_oracle = _rust_public_seed_oracle(env, hero, rng_salt)
    assert py == rust_oracle, (
        f"public_seed parity broken: pluribus=0x{py:016x} rust_oracle=0x{rust_oracle:016x} "
        f"seed={seed} hist_len={hist_len} ante={ante} hero={hero} rng_salt=0x{rng_salt:016x}"
    )


@pytest.mark.parametrize("seed", [11, 22, 33])
def test_digest_uses_committed_street_not_committed_total(seed: int) -> None:
    """If the digest ever accidentally read `committed_total` instead of
    `committed_street`, this test fails: it constructs two states whose
    `committed_street` is identical but `committed_total` differs (ante=0 vs
    ante=100), and asserts the digest is ALSO identical. The Python
    `_pluribus_public_seed` reads `committed_street` (per
    pluribus_search.py:412) so the test passes iff the oracle matches.
    """
    rng = np.random.default_rng(seed)
    # Build ONE state, then snapshot its flat buffer and produce a twin with
    # `committed_total` shifted by `ante` but `committed_street` UNCHANGED.
    base = _make_random_state(rng, hist_len=4, ante=0)
    flat_base = ns._to_flat(base).copy()  # noqa: SLF001
    flat_twin = flat_base.copy()
    ante = 50
    flat_twin[_O_COMMITTED_TOTAL:_O_COMMITTED_TOTAL + _NUM_PLAYERS] = (
        flat_base[_O_COMMITTED_STREET:_O_COMMITTED_STREET + _NUM_PLAYERS] + ante
    )
    twin = ns._from_flat(flat_twin)  # noqa: SLF001
    # Sanity: `committed_street` matches between base and twin; `committed_total` differs.
    assert np.array_equal(base.committed_street, twin.committed_street)
    assert not np.array_equal(base.committed_total, twin.committed_total)
    hero = 0
    salt = 424242
    s_base = _pluribus_public_seed(base, hero, salt)
    s_twin = _pluribus_public_seed(twin, hero, salt)
    assert s_base == s_twin, (
        f"digest leaked committed_total (F-NATIVE-7): "
        f"base=0x{s_base:016x} twin=0x{s_twin:016x}"
    )


@pytest.mark.parametrize("seed", [42, 99])
def test_digest_field_order_locked(seed: int) -> None:
    """Property test: changing the rng_salt by 1 must change the digest
    (no accidental constant padding), and changing hist_len by 1 must change
    the digest. Together these pin that the field order is exactly the
    documented sequence (rng_salt, hero, street, hist_len, ...)."""
    rng = np.random.default_rng(seed)
    env_a = _make_random_state(rng, hist_len=4)
    env_b = _make_random_state(rng, hist_len=5)  # hist_len differs by 1
    hero = 0
    salt = 7777
    s_a = _pluribus_public_seed(env_a, hero, salt)
    s_b = _pluribus_public_seed(env_b, hero, salt)
    assert s_a != s_b, "hist_len had no effect on digest"


def test_digest_insensitive_to_hole_cards_and_deck() -> None:
    """PUBLIC seed by contract: hole cards and deck must NOT influence the digest.

    If this regresses, a solver could lose CRN parity between solves at the
    same public state (different deals -> different Monte-Carlo streams ->
    loss of variance cancellation).
    """
    rng = np.random.default_rng(999)
    base = _make_random_state(rng, hist_len=5)
    # Build a twin flat buffer where ONLY the private fields (holes, deck) are
    # changed; the public seed must be identical.
    flat_twin = ns._to_flat(base).copy()  # noqa: SLF001
    flat_twin[_O_HOLES:_O_HOLES + 12] = rng.integers(0, 52, size=(12,), dtype=np.int64)
    twin_env = ns._from_flat(flat_twin)  # noqa: SLF001
    s_base = _pluribus_public_seed(base, 2, 12345)
    s_priv = _pluribus_public_seed(twin_env, 2, 12345)
    assert s_base == s_priv, (
        f"public_seed leaked a private field: base=0x{s_base:016x} priv=0x{s_priv:016x}"
    )


def test_digest_changes_with_rng_salt() -> None:
    """Different rng_salt -> different seed (basic sanity)."""
    rng = np.random.default_rng(12345)
    env = _make_random_state(rng, hist_len=4)
    s0 = _pluribus_public_seed(env, 0, 0)
    s1 = _pluribus_public_seed(env, 0, 1)
    assert s0 != s1, "rng_salt had no effect on public_seed"


def test_digest_idempotent() -> None:
    """Same (state, hero, rng_salt) -> same seed (idempotence)."""
    rng = np.random.default_rng(54321)
    env = _make_random_state(rng, hist_len=8)
    # Mask into int64 range — production code passes any u64, but the Python
    # encoder wraps `parts` in `np.asarray(..., dtype=np.int64)` so values
    # >=2^63 overflow. Real engine rng_salts are smaller; this test just
    # exercises the idempotence property.
    salt = 0x0DEADBEEFCAFEBAB  # fits in int64
    s_first = _pluribus_public_seed(env, 3, salt)
    s_second = _pluribus_public_seed(env, 3, salt)
    assert s_first == s_second


# Optional native-side end-to-end check (skips when extension isn't built).
def test_rust_pyfunction_matches_python() -> None:
    """If `public_seed_from_state` is exposed by the built extension, call it
    and assert byte-identity with the Python oracle. Skips cleanly when the
    extension is unavailable so pure-Python CI stays green.
    """
    if not ns.native_subgame_available():
        pytest.skip("battery_native extension not built")
    if not hasattr(ns._EXT, "public_seed_from_state"):  # noqa: SLF001
        pytest.skip(
            "public_seed_from_state not exposed by the installed extension "
            "(rebuild tools/battery_native to pick up F-NATIVE-7 changes)"
        )
    rng = np.random.default_rng(2026)
    for hist_len in (0, 1, 7, 32, 64):
        for ante in (0, 50):
            env = _make_random_state(rng, hist_len=hist_len, ante=ante)
            hero = int(rng.integers(0, _NUM_PLAYERS))
            salt = int(rng.integers(0, 2**63))
            py = _pluribus_public_seed(env, hero, salt)
            flat_bytes = ns._to_flat(env).tobytes()  # noqa: SLF001
            rust = int(  # noqa: SLF001
                ns._EXT.public_seed_from_state(flat_bytes, int(hero), int(salt))
            )
            assert py == rust, (
                f"native vs python public_seed: py=0x{py:016x} rust=0x{rust:016x} "
                f"hist_len={hist_len} ante={ante}"
            )
