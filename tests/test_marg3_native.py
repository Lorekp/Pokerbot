"""Bit-identity tests for the native 3-max marginal cube kernel (``aof_cfr.marg3_native``).

The native kernel materializes the contiguous ``(w, prod=w*pc)`` cubes that
``fp_br._legal_marginals_3max`` builds via NumPy broadcasts; the order-sensitive ``np.sum`` reductions
stay in NumPy. Every native-built marginal MUST equal the pure-NumPy ``_legal_marginals_3max`` output
bit-for-bit (``np.array_equal``, NOT allclose) across many random push policies x all six keys x
several depths.

These tests pass on BOTH backends:
  * native present  -> the kernel is exercised and cross-checked against the NumPy oracle;
  * native absent   -> ``_legal_marginals_3max`` uses its pure-NumPy path, so the consistency checks
    are trivially satisfied and the dedicated native check is skipped.
"""

from __future__ import annotations

import numpy as np
import pytest

from aof_cfr import marg3_native
from aof_cfr.exploit_router import _payoffs_3max
from aof_cfr.fp_br import _legal_marginals_3max

_KEYS = ("btn", "sb_f", "sb_a", "bb_fa", "bb_af", "bb_aa")


def _rand_policy(rng):
    return rng.random(169), rng.random((2, 169)), rng.random((3, 169))


def test_legal_marginals_self_consistent_keys_subset():
    """Requesting a key subset returns byte-identical arrays to requesting all keys (the keys=
    fast path must not perturb the computed marginals — holds on either backend)."""
    st, dw = _payoffs_3max(8.0)
    rng = np.random.default_rng(1)
    btn, sb, bb = _rand_policy(rng)
    full = _legal_marginals_3max(st, dw, btn, sb, bb)
    for key in _KEYS:
        sub = _legal_marginals_3max(st, dw, btn, sb, bb, keys=(key,))
        assert set(sub) == {key}
        ep_f, ef_f, w_f = full[key]
        ep_s, ef_s, w_s = sub[key]
        assert np.array_equal(ep_f, ep_s)
        assert ef_f == ef_s
        assert np.array_equal(w_f, w_s)


@pytest.mark.skipif(not marg3_native.native_available(),
                    reason="marg3_native extension not built; NumPy fallback is the oracle itself")
def test_native_cubes_bit_identical_to_numpy():
    """The native cube kernel reproduces the NumPy marginals EXACTLY over a battery of policies."""
    res = marg3_native.verify_against_numpy(n_policies=30, depths=(6.0, 8.0, 10.0), verbose=False)
    assert res["_backend"] == "native"
    assert res["checked"] == 30 * len(_KEYS) * 3


@pytest.mark.skipif(not marg3_native.native_available(),
                    reason="marg3_native extension not built")
def test_native_path_matches_numpy_path_in_legal_marginals():
    """End-to-end: _legal_marginals_3max with the native path == a forced pure-NumPy recompute."""
    st, dw = _payoffs_3max(8.0)
    rng = np.random.default_rng(7)
    for _ in range(10):
        btn, sb, bb = _rand_policy(rng)
        native_out = _legal_marginals_3max(st, dw, btn, sb, bb)
        # Force the NumPy path by temporarily disabling the native module.
        marg3_native.reset_native_cache()
        saved = marg3_native._native, marg3_native._native_tried
        marg3_native._native, marg3_native._native_tried = None, True
        try:
            numpy_out = _legal_marginals_3max(st, dw, btn, sb, bb)
        finally:
            marg3_native._native, marg3_native._native_tried = saved
            marg3_native.reset_native_cache()
        for key in _KEYS:
            assert np.array_equal(native_out[key][0], numpy_out[key][0]), f"ep {key}"
            assert native_out[key][1] == numpy_out[key][1], f"ef {key}"
            assert np.array_equal(native_out[key][2], numpy_out[key][2]), f"w {key}"


def test_build_cubes_returns_none_without_native(monkeypatch):
    """build_cubes degrades to None (NumPy fallback) when the extension is absent."""
    monkeypatch.setattr(marg3_native, "_native", None)
    monkeypatch.setattr(marg3_native, "_native_tried", True)
    st, dw = _payoffs_3max(8.0)
    rng = np.random.default_rng(3)
    btn, sb, bb = _rand_policy(rng)
    assert marg3_native.build_cubes("btn", st, dw, btn, sb, bb) is None
    marg3_native.reset_native_cache()
