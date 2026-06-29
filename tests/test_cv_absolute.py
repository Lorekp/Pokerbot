"""Validate the control-variate absolute-EV decomposition (pokerbot/eval/cv_absolute.py) on SYNTHETIC
correlated draws -- no heavy poker compute, so the estimator's statistics are proven before it is wired
into the (expensive) rotated eval.

Model of a rotated-duplicate per-deck seat-mean:
    f_i      ~ N(0, sf^2)   the deck's shared luck (hand-set + board) -- common to both heroes on deck i
    S_i = mu_S + f_i + eps_S_i     search hero  (eps = the search-specific divergence + residual noise)
    B_i = mu_B + f_i + eps_B_i     blueprint hero on the SAME deck (CRN) -- shares f_i
Because S and B share f_i, the paired diff S-B removes the dominant shared-luck variance. The cheap
blueprint sample is drawn from the SAME distribution (its own independent f'_j) -> pins E[B] = mu_B.
"""

from __future__ import annotations

import numpy as np

from pokerbot.eval.cv_absolute import CVResult, decks_for_target_ci, decompose_absolute


def _draw(rng, mu_S, mu_B, n, *, sf=8.0, se=2.0):
    """n decks of CRN-paired (S, B): a shared luck factor f plus hero-specific residuals."""
    f = rng.normal(0.0, sf, size=n)
    S = mu_S + f + rng.normal(0.0, se, size=n)
    B = mu_B + f + rng.normal(0.0, se, size=n)
    return S, B


def test_unbiased_and_reduces_variance():
    rng = np.random.default_rng(0)
    mu_S, mu_B = 3.0, 1.0           # true search EV +3 bb/hand, blueprint +1 (search adds +2)
    S, B = _draw(rng, mu_S, mu_B, 400)
    # disjoint cheap blueprint sample (independent decks), large + cheap
    _, Bc = _draw(rng, mu_S, mu_B, 4000)

    res = decompose_absolute(S, B, Bc)
    assert isinstance(res, CVResult)
    # 1) UNBIASED: the CV mean covers the true mu_S well within its 95% CI
    lo, hi = res.ci95
    assert lo <= mu_S <= hi, f"true {mu_S} outside CV CI {res.ci95}"
    # 2) VARIANCE REDUCED: high shared-luck correlation -> the CV SE is materially tighter than naive
    assert res.corr > 0.9, f"expected high CRN corr, got {res.corr:.3f}"
    assert res.var_reduction > 3.0, f"expected >3x reduction, got {res.var_reduction:.2f}x"
    assert res.se < res.naive_se
    # 3) both estimators target the same quantity (naive also ~ mu_S, just noisier)
    assert abs(res.naive_mean - mu_S) < 4 * res.naive_se


def test_unbiasedness_over_many_seeds():
    """Across independent replications the CV mean must be centered on mu_S (no systematic bias)."""
    mu_S, mu_B = 2.5, 0.5
    means = []
    for seed in range(200):
        rng = np.random.default_rng(1000 + seed)
        S, B = _draw(rng, mu_S, mu_B, 120)
        _, Bc = _draw(rng, mu_S, mu_B, 1200)
        means.append(decompose_absolute(S, B, Bc).mean)
    grand = float(np.mean(means))
    se_of_mean = float(np.std(means, ddof=1) / np.sqrt(len(means)))
    assert abs(grand - mu_S) < 4 * se_of_mean, f"bias: grand mean {grand:.4f} vs {mu_S} (se {se_of_mean:.4f})"


def test_beta_one_also_unbiased_but_weaker():
    rng = np.random.default_rng(7)
    mu_S, mu_B = 3.0, 1.0
    S, B = _draw(rng, mu_S, mu_B, 400)
    _, Bc = _draw(rng, mu_S, mu_B, 4000)
    opt = decompose_absolute(S, B, Bc, optimal_beta=True)
    one = decompose_absolute(S, B, Bc, optimal_beta=False)
    assert abs(one.beta - 1.0) < 1e-12
    # both unbiased
    assert one.ci95[0] <= mu_S <= one.ci95[1]
    # optimal beta is never worse than beta=1 (it minimizes the residual variance)
    assert opt.se <= one.se + 1e-9


def test_zero_correlation_gives_no_reduction():
    """If the baseline shares NO luck with the hero (e.g. an always-fold bot vs the board variance), the CV
    must not claim a spurious reduction -- it degenerates to ~naive (modulo the cheap-term cost)."""
    rng = np.random.default_rng(3)
    # independent S and B (no shared f) -> corr ~ 0
    S = 3.0 + rng.normal(0.0, 8.0, size=500)
    B = 1.0 + rng.normal(0.0, 8.0, size=500)
    _, Bc = _draw(rng, 3.0, 1.0, 5000)
    res = decompose_absolute(S, B, Bc, optimal_beta=True)
    assert abs(res.corr) < 0.15
    # optimal beta ~ 0 when uncorrelated -> falls back toward the naive estimator, no big spurious gain
    assert res.var_reduction < 1.6


def test_decks_for_target_ci_monotone():
    # tighter target or higher variance -> more decks
    n_loose = decks_for_target_ci(per_deck_var=64.0, target_halfwidth_bb_per_100=300.0)
    n_tight = decks_for_target_ci(per_deck_var=64.0, target_halfwidth_bb_per_100=100.0)
    n_noisier = decks_for_target_ci(per_deck_var=256.0, target_halfwidth_bb_per_100=100.0)
    assert n_tight > n_loose
    assert n_noisier > n_tight
    assert n_loose >= 1
