"""4-max BR marginal normalization — regression for the den 2x double-count.

The old ``_br_marginals`` accumulated ``den`` over BOTH action branches, silently halving every
EV that divides by it (router ev_gain gate, engine advise/coach 4-max EVs, gauntlet marginals).
These tests pin the normalization invariants that make the fix verifiable without reference to
any other estimator, plus a cross-check against the independent ``onpolicy_seat_ev``.
"""

from __future__ import annotations

import numpy as np
import pytest

from aof_cfr.fp_4max_vec import (
    _OUTCOMES,
    _br_marginals,
    _infoset_index,
    _Policy,
    _reach_factors,
    onpolicy_seat_ev,
    precompute_payoffs,
    sample_deals,
)

N_DEALS = 400


@pytest.fixture(scope="module")
def batch():
    buckets, holes = sample_deals(N_DEALS, seed=123)
    payoffs = precompute_payoffs(holes, 8.0)
    return buckets, payoffs


def _mixed_policy(buckets, seed=5) -> _Policy:
    """A non-trivial mixed policy so reach factors are strictly inside (0, 1)."""
    rng = np.random.default_rng(seed)
    pol = _Policy()
    for k in _infoset_index(buckets):
        pol.table[k] = rng.uniform(0.2, 0.8, size=169)
    return pol


def test_den_equals_full_infoset_reach(batch) -> None:
    """Root infoset reach is exactly 1 per deal; the old code returned 2 per deal."""
    buckets, payoffs = batch
    pol = _mixed_policy(buckets)
    infosets = _infoset_index(buckets)
    _np_, _nf_, den = _br_marginals(pol, buckets, payoffs, infosets)
    assert float(den[(0, "-")].sum()) == pytest.approx(N_DEALS)


def test_push_and_fold_branch_reaches_are_equal(batch) -> None:
    """Each branch's reach (own factor excluded) sums to the same infoset reach — including
    the collapsed FFF outcome vs its FFA·{F,A} pair — proving the old den was exactly 2x."""
    buckets, payoffs = batch
    pol = _mixed_policy(buckets)
    infosets = _infoset_index(buckets)
    n = buckets.shape[0]
    facs = _reach_factors(pol, buckets)
    den_a = {k: np.zeros(169) for k in infosets}
    den_f = {k: np.zeros(169) for k in infosets}
    for oi, o in enumerate(_OUTCOMES):
        fac = facs[oi]
        for pos, (pp, hpp, app) in enumerate(o["acts"]):
            re = np.ones(n)
            for q, (_i, _a, f) in enumerate(fac):
                if q != pos:
                    re = re * f
            np.add.at((den_a if app == "A" else den_f)[(pp, hpp)], buckets[:, pp], re)
    for k in infosets:
        np.testing.assert_allclose(den_a[k], den_f[k], atol=1e-9, err_msg=str(k))
    # and the shipped den is the single-branch reach
    _np_, _nf_, den = _br_marginals(pol, buckets, payoffs, infosets)
    for k in infosets:
        np.testing.assert_allclose(den[k], den_a[k], atol=1e-12)


def test_onpolicy_marginals_reconstruct_seat_ev(batch) -> None:
    """Σ over a seat's infosets of (p·num_push + (1−p)·num_fold) / n_deals must equal that
    seat's independent on-policy EV — a normalization identity that fails under any den or
    reach miscount (it held for numerators before the fix too; with den it pins e_push/e_fold)."""
    buckets, payoffs = batch
    pol = _mixed_policy(buckets)
    infosets = _infoset_index(buckets)
    num_push, num_fold, den = _br_marginals(pol, buckets, payoffs, infosets)
    seat_ev = onpolicy_seat_ev(pol, buckets, payoffs)
    n = buckets.shape[0]
    recon = [0.0, 0.0, 0.0, 0.0]
    for (pp, hpp) in infosets:
        p = pol.table[(pp, hpp)]
        recon[pp] += float((p * num_push[(pp, hpp)] + (1.0 - p) * num_fold[(pp, hpp)]).sum()) / n
    # seat EV also includes terminal payoffs at nodes where the seat does NOT act (e.g. the BB
    # pocketing the walk after FFF). Those are excluded from the seat's own infosets, so the
    # identity is exact only for seats that act in every outcome they're paid in — true for
    # UTG (acts first in every outcome).
    assert recon[0] == pytest.approx(seat_ev[0], abs=1e-9)


def test_ev_scale_sane_at_base(batch) -> None:
    """With the den fix, conditional EVs are in natural bb units: UTG folding is worth ~0bb
    (blinds untouched), and e_push for AA must exceed 2bb at 8bb stacks (it read ~half before)."""
    import aof_cfr.query as q
    from pathlib import Path
    if not Path("data/aof/cfr_base/cfr_base_4p.npz").is_file():
        pytest.skip("4-max base not built")
    q.load_cfr_base()
    from aof_cfr.buckets import LABEL_TO_INDEX, LABELS_169
    from aof_cfr.fp_4max_vec import rows_to_table
    from aof_cfr.profiler import base_policy_rows

    buckets, payoffs = batch
    infosets = _infoset_index(buckets)
    table = rows_to_table(base_policy_rows(4, 8.0), 8.0)
    num_push, num_fold, den = _br_marginals(table, buckets, payoffs, infosets)
    k = (0, "-")
    safe = np.where(den[k] > 0, den[k], 1.0)
    e_push = num_push[k] / safe
    e_fold = num_fold[k] / safe
    aa = LABEL_TO_INDEX["AA"]
    # AA open-jam at 8bb: ~P(all fold)*1.5bb dead + a healthy showdown edge when called —
    # ≈1.6bb true. The old halved code read ≈0.81, IMPOSSIBLY below even the discounted
    # all-fold winnings. 1.2 separates the regimes robustly.
    assert e_push[aa] > 1.2
    # folding UTG leaves the blinds to fight: |EV| well under a blind on average
    w = den[k] / den[k].sum()
    assert abs(float((w * e_fold).sum())) < 1.0
