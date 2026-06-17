"""Batched native showdown kernel (tools/ompeval_cli mc_nway_shares_batch + the fp_4max_vec fast
path). The batch reuses the SAME OMPEval evaluator + _N_BOARDS as the pure-Python loop but a
different (fast) board RNG, so it is STATISTICALLY equivalent, not bit-identical: validate to MC
tolerance. Skips cleanly when the extension isn't built (CI without the native build stays green)."""

from __future__ import annotations

import numpy as np
import pytest

import aof_cfr.fp_4max_vec as fp
from aof_cfr.fp_4max_vec import precompute_payoffs, sample_deals

try:
    from tools.ompeval_cli.native import mc_nway_shares_batch_available
    _HAVE = mc_nway_shares_batch_available()
except Exception:  # noqa: BLE001
    _HAVE = False

pytestmark = pytest.mark.skipif(not _HAVE, reason="mc_nway_shares_batch native extension not built")


def _payoffs(holes, depth, batched: bool):
    saved = fp._DISABLE_BATCH
    fp._DISABLE_BATCH = not batched
    # the Python loop and batched path share exploit_mc._SHARE_CACHE — clear it so each path computes
    # its own showdown shares (otherwise the 2nd run reads back the 1st run's cached values).
    fp._SHARE_CACHE.clear()
    try:
        return precompute_payoffs(holes, depth)
    finally:
        fp._DISABLE_BATCH = saved
        fp._SHARE_CACHE.clear()


def test_batched_matches_python_within_mc_noise():
    """Tensor-level: averaged over deals, the batched payoffs match the Python loop to MC noise.
    Per-cell differences are real (different RNG stream) but must not be biased."""
    _, holes = sample_deals(500, 4242)
    old = _payoffs(holes, 8.0, batched=False)
    new = _payoffs(holes, 8.0, batched=True)
    assert old.shape == new.shape
    # per-outcome mean payoff (the quantity the marginals integrate) agrees tightly
    per_outcome = np.abs(old.mean(axis=1) - new.mean(axis=1))
    assert per_outcome.max() < 0.10, f"per-outcome mean diff too large: {per_outcome.max():.4f} bb"
    # the uncontested-steal outcomes (≤1 alive) are deterministic → must be exactly equal
    # (outcome 0 in _OUTCOMES is everyone-folds-to-one; pick outcomes with a single alive seat)
    from aof_cfr.fp_4max_vec import _OUTCOMES
    for oi, o in enumerate(_OUTCOMES):
        alive = [i for i in range(4) if i not in o["folded"]]
        if len(alive) <= 1:
            assert np.array_equal(old[oi], new[oi]), f"deterministic outcome {oi} differs"


def test_batched_seed_is_deterministic():
    """Same deals → same batched tensor (the per-showdown seed scheme is deterministic)."""
    _, holes = sample_deals(300, 99)
    a = _payoffs(holes, 8.0, batched=True)
    fp._SHARE_CACHE.clear()
    b = _payoffs(holes, 8.0, batched=True)
    assert np.array_equal(a, b)
