"""Bit-identity gate for the native 4-max marginal payoff kernel.

``aof_cfr.marg4_native.precompute_payoffs_native`` MUST reproduce
``aof_cfr.fp_4max_vec.precompute_payoffs`` byte-for-byte (``np.array_equal``) — it
feeds the validated equilibrium, so anything short of exact equality is a defect.

Skips cleanly when the Rust extension is not built (CI without the .pyd stays green).
"""

from __future__ import annotations

import numpy as np
import pytest

from aof_cfr.marg4_native import native_available, precompute_payoffs_native

pytestmark = pytest.mark.skipif(
    not native_available(),
    reason="marg4_native extension not built",
)


def _ref_and_native(n_deals: int, seed: int, depth: float):
    from aof_cfr.fp_4max_vec import precompute_payoffs, sample_deals
    from aof_cfr.pushfold_payoffs import flat_fee_default, game_model_default

    _, holes = sample_deals(n_deals, seed)
    # Reference path: force the pure-Python loop so we compare against it, not the
    # native fast-path it now wires in (precompute_payoffs would otherwise call native).
    seats = __import__("aof_ranges", fromlist=["active_seats"]).active_seats(4)
    from aof_cfr.exploit_mc import _terminal_returns
    from aof_cfr.fp_4max_vec import _OUTCOMES

    gm = game_model_default()
    fee = flat_fee_default()
    ref = np.zeros((len(_OUTCOMES), n_deals, 4), dtype=np.float64)
    for oi, o in enumerate(_OUTCOMES):
        folded = set(o["folded"])
        for d in range(n_deals):
            ref[oi, d, :] = _terminal_returns(4, seats, holes[d], folded, float(depth),
                                              game_model=gm)
    nat = precompute_payoffs_native(holes, float(depth), game_model=gm, fee_bb=fee)
    return ref, nat


@pytest.mark.parametrize("depth", [6.0, 8.0])
def test_native_matches_python_bit_identical(depth: float) -> None:
    ref, nat = _ref_and_native(n_deals=400, seed=4242, depth=depth)
    assert nat is not None
    assert nat.shape == ref.shape == (15, 400, 4)
    assert np.array_equal(ref, nat), f"native payoff tensor differs from Python at depth {depth}"


def test_native_matches_precompute_payoffs_wrapper(monkeypatch) -> None:
    # When the marg4 native path is explicitly enabled (and the default batched path disabled),
    # the wrapper must route to it and return exactly the native tensor. NOTE: by default
    # precompute_payoffs now uses the *batched* OMPEval kernel (statistically equivalent, different
    # RNG → NOT array_equal to this bit-identical kernel), so we force the marg4 path here.
    import aof_cfr.fp_4max_vec as fp
    from aof_cfr.fp_4max_vec import precompute_payoffs, sample_deals
    from aof_cfr.pushfold_payoffs import flat_fee_default, game_model_default

    monkeypatch.setenv("AOF_MARG4_NATIVE", "1")     # enable the marg4 bit-identical fast-path
    monkeypatch.setattr(fp, "_DISABLE_BATCH", True)  # and disable the (default) batched path
    _, holes = sample_deals(300, 999)
    gm = game_model_default()
    fee = flat_fee_default()
    via_wrapper = precompute_payoffs(holes, 8.0)  # routes through the marg4 native fast-path
    nat = precompute_payoffs_native(holes, 8.0, game_model=gm, fee_bb=fee)
    assert np.array_equal(via_wrapper, nat)


def test_empty_deals_returns_empty_tensor() -> None:
    out = precompute_payoffs_native([], 8.0, game_model=0, fee_bb=0.0)
    assert out is not None
    assert out.shape == (15, 0, 4)


def test_thread_parity_single_call() -> None:
    # rayon thread count must not change the result (deal-parallel, independent per deal).
    from aof_cfr.fp_4max_vec import sample_deals
    from aof_cfr.pushfold_payoffs import flat_fee_default, game_model_default

    _, holes = sample_deals(500, 13)
    gm = game_model_default()
    fee = flat_fee_default()
    base = precompute_payoffs_native(holes, 8.0, game_model=gm, fee_bb=fee, threads=1)
    for t in (2, 4):
        got = precompute_payoffs_native(holes, 8.0, game_model=gm, fee_bb=fee, threads=t)
        assert np.array_equal(base, got), f"thread parity failed at threads={t}"
