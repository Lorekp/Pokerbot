"""Suit-aware abstraction-free info encoding (POKERBOT_INFO_SUITS), the neural-CFR preflop prerequisite.

The rank-only ``encode_info_state`` collapses AKs and AKo (169 -> 91 preflop classes), which is coarser
than the 169-class tabular blueprint. The opt-in suit-isomorphic tail fixes that. These tests pin the
two contracts: (1) BYTE-IDENTICAL when off (deployed search path unaffected); (2) suit-distinguishing
when on, while staying suit-ISOMORPHIC (relabeling suits leaves the feature unchanged).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pokerbot.engine import jax_engine as je
from pokerbot.engine.jax_engine import card_suit, encode_info_state


def _state_with_holes(c0: int, c1: int):
    st = je.reset_state(jax.random.PRNGKey(0))
    return st._replace(holes=st.holes.at[0].set(jnp.array([c0, c1], dtype=st.holes.dtype)))


def _ranks():
    return np.array([int(je.card_rank(jnp.int32(c))) for c in range(52)])


def _suited_offsuit_pair():
    """Return (c0, c1_suited_with_c0, c1_offsuit_same_rank) — same two RANKS, suited vs offsuit."""
    r = _ranks()
    c0 = 0                                                   # rank r[0], suit 0
    s0 = int(card_suit(jnp.int32(c0)))
    r1cards = [c for c in range(52) if r[c] != r[0]]
    c1_off = next(c for c in r1cards if int(card_suit(jnp.int32(c))) != s0)
    r1 = r[c1_off]
    c1_suited = next(c for c in range(52) if r[c] == r1 and int(card_suit(jnp.int32(c))) == s0)
    return c0, c1_suited, c1_off


def test_byte_identical_when_off(monkeypatch):
    """Flag OFF: a suit swap leaves the vector identical (current rank-only behavior preserved)."""
    monkeypatch.setattr(je, "_INFO_SUITS", False)
    c0, c1s, c1o = _suited_offsuit_pair()
    v_suited = np.array(encode_info_state(_state_with_holes(c0, c1s), jnp.int32(0)))
    v_offsuit = np.array(encode_info_state(_state_with_holes(c0, c1o), jnp.int32(0)))
    assert np.array_equal(v_suited, v_offsuit)              # rank-only -> identical (no suit leak)
    assert v_suited.shape == (je.INFO_FEATURE_DIM,)


def test_distinguishes_suitedness_when_on(monkeypatch):
    """Flag ON: AKs and AKo now differ -> the 169th class is recovered."""
    monkeypatch.setattr(je, "_INFO_SUITS", True)
    c0, c1s, c1o = _suited_offsuit_pair()
    v_suited = np.array(encode_info_state(_state_with_holes(c0, c1s), jnp.int32(0)))
    v_offsuit = np.array(encode_info_state(_state_with_holes(c0, c1o), jnp.int32(0)))
    assert not np.array_equal(v_suited, v_offsuit)


def test_additive_tail_only(monkeypatch):
    """ON vs OFF differ ONLY in the appended tail: the original feature region is untouched, so the
    suit block cannot perturb any existing feature the deployed path reads."""
    c0, c1s, _ = _suited_offsuit_pair()
    monkeypatch.setattr(je, "_INFO_SUITS", False)
    off = np.array(encode_info_state(_state_with_holes(c0, c1s), jnp.int32(0)))
    monkeypatch.setattr(je, "_INFO_SUITS", True)
    on = np.array(encode_info_state(_state_with_holes(c0, c1s), jnp.int32(0)))
    diff = np.nonzero(np.abs(on - off) > 1e-9)[0]
    assert diff.size > 0                                    # the tail did change something
    # everything before the first changed index is identical (the original feature block)
    first = int(diff.min())
    assert np.array_equal(on[:first], off[:first])


def test_suit_isomorphic(monkeypatch):
    """Relabeling suits (canonical permutation) leaves the suit features unchanged: AhKh ~ AsKs."""
    monkeypatch.setattr(je, "_INFO_SUITS", True)
    r = _ranks()
    # AKs in suit 0 vs AKs in suit 1: same ranks, both suited, different suit label -> must match.
    cA0 = next(c for c in range(52) if r[c] == 14 and int(card_suit(jnp.int32(c))) == 0)
    cK0 = next(c for c in range(52) if r[c] == 13 and int(card_suit(jnp.int32(c))) == 0)
    cA1 = next(c for c in range(52) if r[c] == 14 and int(card_suit(jnp.int32(c))) == 1)
    cK1 = next(c for c in range(52) if r[c] == 13 and int(card_suit(jnp.int32(c))) == 1)
    v0 = np.array(encode_info_state(_state_with_holes(cA0, cK0), jnp.int32(0)))
    v1 = np.array(encode_info_state(_state_with_holes(cA1, cK1), jnp.int32(0)))
    assert np.allclose(v0, v1)                              # suit-permutation invariant
