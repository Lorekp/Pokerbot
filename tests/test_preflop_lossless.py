"""Lossless-preflop abstraction (Pluribus-faithful Stage 2).

Covers the four contracts of the street-split key:
  1. ``canonical_preflop_class`` is the EXACT 169-class suit-isomorphic preflop collapse.
  2. The JAX bucket table under the flag = ``pos*169 + class`` (no WEVA index needed).
  3. ``fingerprint_fast`` stays BYTE-IDENTICAL to the full encode at every node (preflop uses the
     lossless history; postflop is untouched) — the hot-path correctness contract.
  4. ``check_blueprint_serve_env`` catches a train/serve flag skew via the snapshot stamp.
"""

from __future__ import annotations

import itertools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pokerbot.encoding import weva_encoder
from pokerbot.encoding.weva_encoder import (
    PREFLOP_LOSSLESS_NUM,
    PREFLOP_NUM_CLASSES,
    canonical_preflop_class,
)


# --------------------------------------------------------------------------- #
# 1. canonical_preflop_class — the lossless 169 collapse
# --------------------------------------------------------------------------- #
def _all_combos():
    return list(itertools.combinations(range(52), 2))


def test_canonical_class_range_and_count():
    classes = {canonical_preflop_class(a, b) for a, b in _all_combos()}
    assert classes == set(range(PREFLOP_NUM_CLASSES))  # exactly 169, contiguous 0..168
    assert PREFLOP_NUM_CLASSES == 169
    assert PREFLOP_LOSSLESS_NUM == 1014


def test_pairs_on_diagonal():
    # Pocket pair of rank-ordinal r (0=2 … 12=A) -> class r. Card index = rank_ord*4 + suit.
    for r in range(13):
        suits = [r * 4 + s for s in range(4)]
        for a, b in itertools.combinations(suits, 2):
            assert canonical_preflop_class(a, b) == r


def test_suit_isomorphism_lossless():
    # AhKh == AsKs (suited), AhKs == AdKc (offsuit), and suited != offsuit for the same rank pair.
    A, K = 12, 11  # rank ordinals (ace=12, king=11)
    ah_kh = canonical_preflop_class(A * 4 + 2, K * 4 + 2)   # both hearts
    as_ks = canonical_preflop_class(A * 4 + 3, K * 4 + 3)   # both spades
    ah_ks = canonical_preflop_class(A * 4 + 2, K * 4 + 3)   # hearts/spades (offsuit)
    ad_kc = canonical_preflop_class(A * 4 + 1, K * 4 + 0)   # diamonds/clubs (offsuit)
    assert ah_kh == as_ks                                   # suited suit-invariant
    assert ah_ks == ad_kc                                   # offsuit suit-invariant
    assert ah_kh != ah_ks                                   # suited and offsuit are DISTINCT classes
    assert 13 <= ah_kh <= 90                                # suited band
    assert 91 <= ah_ks <= 168                               # offsuit band


def test_class_is_order_independent():
    for a, b in _all_combos():
        assert canonical_preflop_class(a, b) == canonical_preflop_class(b, a)


# --------------------------------------------------------------------------- #
# 2/3. encode-path integration — table + byte-identity (flag ON)
# --------------------------------------------------------------------------- #
@pytest.fixture()
def lossless_on(monkeypatch):
    """Turn on betting-abstraction + postflop-buckets + preflop-lossless and clear the table cache."""
    from pokerbot.engine import jax_weva_encode as enc
    from pokerbot.engine import jax_weva_fast as fast

    monkeypatch.setattr(weva_encoder, "_PREFLOP_LOSSLESS", True)
    monkeypatch.setattr(enc, "_BETTING_ABSTRACTION", True)
    monkeypatch.setattr(enc, "_POSTFLOP_BUCKETS_ENABLED", True)
    fast.default_weva_bucket_table.cache_clear()
    yield
    fast.default_weva_bucket_table.cache_clear()


def test_lossless_bucket_table(lossless_on):
    from pokerbot.engine.jax_weva_fast import build_weva_bucket_table

    table = build_weva_bucket_table()  # no mapper → must NOT need a WEVA index under the flag
    assert table.shape == (52, 52, 6)
    for a, b in _all_combos():
        cls = canonical_preflop_class(a, b)
        for pos in range(6):
            assert int(table[a, b, pos]) == pos * PREFLOP_NUM_CLASSES + cls
    assert int(table.max()) < PREFLOP_LOSSLESS_NUM  # bounded at 1014


def _live_seats(state):
    return [s for s in range(6) if int(np.asarray(state.holes)[s, 0]) >= 0]


def test_fingerprint_fast_byte_identical(lossless_on):
    """The hot-path fingerprint must equal the full-encode digest at every node and seat, all streets."""
    from pokerbot.engine import jax_engine as je
    from pokerbot.engine.jax_weva_fast import (
        default_weva_bucket_table,
        encode_nlhe_bundle_fast,
        fingerprint_fast,
    )

    table = default_weva_bucket_table()
    occ = jnp.asarray((True,) * 6, dtype=bool)
    checked_preflop = checked_postflop = 0
    for seed in range(40):
        s = je.init_hand(jax.random.PRNGKey(seed), occupancy=occ)
        rng = np.random.default_rng(seed * 911 + 7)
        steps = 0
        while not bool(np.asarray(s.done)) and steps < 200:
            steps += 1
            if bool(np.asarray(je.is_chance_node(s))):
                s = je.sample_chance(s)
                continue
            for seat in _live_seats(s):
                fp_fast = fingerprint_fast(s, seat, weva_table=table)
                fp_full = encode_nlhe_bundle_fast(s, seat, weva_table=table).fingerprint()
                assert fp_fast == fp_full, f"seed={seed} seat={seat} street={int(np.asarray(s.street))}"
                if int(np.asarray(s.street)) == 0:
                    checked_preflop += 1
                else:
                    checked_postflop += 1
            mask = np.asarray(je.legal_action_mask(s))
            legal = np.nonzero(mask)[0]
            if legal.size == 0:
                break
            # Bias toward CHECK(1)/CALL(2) so hands reach the flop (random uniform folds out preflop) —
            # we need postflop nodes to exercise the bucketed branch of the split key.
            cont = [a for a in legal if a in (1, 2)]
            pick = cont[0] if (cont and rng.random() < 0.85) else int(rng.choice(legal))
            s = je.apply_action(s, jnp.int32(int(pick)))
    assert checked_preflop > 0 and checked_postflop > 0  # exercised both branches of the split key


def test_lossless_changes_preflop_key_vs_legacy(monkeypatch):
    """A preflop key under lossless must differ from the legacy bucketed-betting key (the whole point)."""
    from pokerbot.engine import jax_engine as je
    from pokerbot.engine import jax_weva_encode as enc
    from pokerbot.engine import jax_weva_fast as fast

    monkeypatch.setattr(enc, "_BETTING_ABSTRACTION", True)
    monkeypatch.setattr(enc, "_POSTFLOP_BUCKETS_ENABLED", True)
    occ = jnp.asarray((True,) * 6, dtype=bool)
    s = je.init_hand(jax.random.PRNGKey(3), occupancy=occ)
    while bool(np.asarray(je.is_chance_node(s))):
        s = je.sample_chance(s)
    seat = _live_seats(s)[0]

    monkeypatch.setattr(weva_encoder, "_PREFLOP_LOSSLESS", False)
    fast.default_weva_bucket_table.cache_clear()
    legacy = fast.fingerprint_fast(s, seat, weva_table=fast.default_weva_bucket_table())

    monkeypatch.setattr(weva_encoder, "_PREFLOP_LOSSLESS", True)
    fast.default_weva_bucket_table.cache_clear()
    lossless = fast.fingerprint_fast(s, seat, weva_table=fast.default_weva_bucket_table())
    fast.default_weva_bucket_table.cache_clear()
    assert legacy != lossless


# --------------------------------------------------------------------------- #
# 4. serve-skew guard
# --------------------------------------------------------------------------- #
def test_cards_only_default_keeps_bounded_betting(monkeypatch):
    """The DEFAULT lossless (cards only, betting flag OFF) must NOT emit the lossless history -- that
    bounded betting is what keeps the infoset count fittable (the SharedHogwildTable-full fix)."""
    from pokerbot.engine import jax_engine as je
    from pokerbot.engine import jax_weva_encode as enc
    from pokerbot.engine import jax_weva_fast as fast

    monkeypatch.setattr(enc, "_BETTING_ABSTRACTION", True)
    monkeypatch.setattr(enc, "_POSTFLOP_BUCKETS_ENABLED", True)
    monkeypatch.setattr(weva_encoder, "_PREFLOP_LOSSLESS", True)
    monkeypatch.setattr(weva_encoder, "_PREFLOP_LOSSLESS_BETTING", False)
    fast.default_weva_bucket_table.cache_clear()
    occ = jnp.asarray((True,) * 6, dtype=bool)
    s = je.init_hand(jax.random.PRNGKey(5), occupancy=occ)
    while bool(np.asarray(je.is_chance_node(s))):
        s = je.sample_chance(s)
    seat = _live_seats(s)[0]
    # cards-only digest must equal the full encode (which fills betting_state, not history).
    cards_only = fast.fingerprint_fast(s, seat, weva_table=fast.default_weva_bucket_table())
    assert cards_only == fast.encode_nlhe_bundle_fast(s, seat, weva_table=fast.default_weva_bucket_table()).fingerprint()

    # Turning the betting flag ON must change the preflop key (now keyed on lossless history).
    monkeypatch.setattr(weva_encoder, "_PREFLOP_LOSSLESS_BETTING", True)
    with_betting = fast.fingerprint_fast(s, seat, weva_table=fast.default_weva_bucket_table())
    fast.default_weva_bucket_table.cache_clear()
    assert with_betting != cards_only


def test_active_abstraction_config_reports_flag(monkeypatch):
    from pokerbot.engine import jax_weva_encode as enc

    monkeypatch.setattr(weva_encoder, "_PREFLOP_LOSSLESS", True)
    monkeypatch.setattr(weva_encoder, "_PREFLOP_LOSSLESS_BETTING", True)
    cfg = enc.active_abstraction_config()
    assert cfg["preflop_lossless"] is True
    assert cfg["preflop_lossless_betting"] is True


def test_serve_guard_rejects_flag_skew(monkeypatch):
    from pokerbot.engine import jax_weva_encode as enc
    from pokerbot.eval.lbr import check_blueprint_serve_env

    # Active serve env: postflop OK but preflop-lossless OFF; snapshot was trained with it ON.
    monkeypatch.setattr(enc, "_BETTING_ABSTRACTION", True)
    monkeypatch.setattr(enc, "_POSTFLOP_BUCKETS_ENABLED", True)
    monkeypatch.setattr(weva_encoder, "_PREFLOP_LOSSLESS", False)
    stamp = {"betting_abstraction": True, "postflop_buckets": True,
             "postflop_mode": "fast", "postflop_num_buckets": 200, "preflop_lossless": True}
    with pytest.raises(RuntimeError, match="preflop_lossless|stamp"):
        check_blueprint_serve_env(strict=True, expected=stamp)

    # Matching stamp → no raise.
    monkeypatch.setattr(weva_encoder, "_PREFLOP_LOSSLESS", True)
    out = check_blueprint_serve_env(strict=True, expected={**stamp})
    assert out["preflop_lossless"] is True
