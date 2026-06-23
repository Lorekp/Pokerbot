r"""Control-variate decomposition for the ABSOLUTE rotated-duplicate win-rate of an EXPENSIVE hero.

Why this exists
---------------
``run_field_absolute_rotated`` measures a hero's absolute bb/hand vs a homogeneous field by seat-rotated
duplicate dealing. Rotation cancels the dominant *seat-assignment* luck (the hero holds each of the deck's
hands once), but it does NOT cancel three remaining variance sources: (a) the *set* of hands the deck deals
(varies deck-to-deck), (b) board runouts, (c) villain stochasticity. For the **search** hero each decision
costs ~1-1.5 s, so driving the standard error low enough to exclude zero against a small-edge field (a
solid-reg table) needs thousands of decks -> a cloud-scale job. AIVAT does not help on the search path (its
chance term is absent and the gadget-Q baseline is miscalibrated -> *negative* variance reduction, see
``run_field_absolute_rotated``).

The decomposition
-----------------
Split the estimand using a CHEAP, highly-correlated baseline policy (the blueprint-only hero -- same engine,
no search, ~10-50x cheaper) evaluated on the SAME decks under common random numbers (CRN: identical deck
seed -> identical boards + villain RNG):

    E[S]  =  E[S - B]            +  E[B]
             \__ paired diff __/    \_ baseline _/
             low variance (CRN:      measured CHEAPLY to a tight
             S,B share cards+board,  CI with MANY no-search decks
             differ only where
             search != blueprint)

Estimator (with a DISJOINT cheap baseline sample so the two terms are independent):

    mu_hat = (mean(S_paired) - mean(B_paired)) + mean(B_cheap)
    Var    = Var(S - B)/n_paired  +  Var(B_cheap)/n_cheap

vs the naive ``mean(S_paired)`` with ``Var(S)/n_paired``. When corr(S, B) is high the paired-diff variance
``Var(S-B) = Var(S) + Var(B) - 2 Cov(S, B)`` collapses, and the baseline term is driven down by the large
cheap sample -- so the *expensive* search-deck count needed for a target CI drops by ``Var(S)/Var(S-B)``,
typically 3-8x. This is an unbiased control variate (the baseline's own mean is estimated, not assumed), so
it cannot bias the headline -- it only sharpens it.

Unbiasedness requires the cheap baseline sample be drawn from the SAME deck distribution as the paired set
(same field, same rotation, same deck-seed generator) and be statistically independent of it (disjoint deck
seeds). ``decompose_absolute`` enforces neither at runtime -- the caller wires the arms -- so the contract is
documented here and exercised by ``tests/test_cv_absolute.py`` (unbiasedness + a measured variance reduction
on synthetic correlated draws).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CVResult:
    """Variance-reduced absolute mean (per-deck seat-mean units, e.g. bb/hand) + its provenance.

    ``mean`` / ``se`` are the control-variate estimate; ``naive_mean`` / ``naive_se`` are the plain
    ``mean(S_paired)`` an unaided run would report from the SAME expensive decks. ``var_reduction`` is the
    factor by which the expensive-deck count could shrink for the same CI (``(naive_se/se)**2``); ``corr``
    is the realized search<->baseline correlation that drives it. ``decks_paired`` / ``decks_cheap`` are the
    two sample sizes."""

    mean: float
    se: float
    naive_mean: float
    naive_se: float
    var_reduction: float
    corr: float
    beta: float
    decks_paired: int
    decks_cheap: int

    @property
    def ci95(self) -> tuple[float, float]:
        return (self.mean - 1.96 * self.se, self.mean + 1.96 * self.se)

    @property
    def naive_ci95(self) -> tuple[float, float]:
        return (self.naive_mean - 1.96 * self.naive_se, self.naive_mean + 1.96 * self.naive_se)


def decompose_absolute(
    search_paired,
    blueprint_paired,
    blueprint_cheap,
    *,
    optimal_beta: bool = True,
) -> CVResult:
    """Control-variate estimate of ``E[search]`` from a small CRN-paired (search, blueprint) sample plus a
    large disjoint cheap blueprint sample.

    Parameters
    ----------
    search_paired, blueprint_paired
        Per-deck seat-means for the SAME expensive decks under CRN (``search_paired[i]`` and
        ``blueprint_paired[i]`` are the hero-search and hero-blueprint results on deck ``i`` with identical
        boards + villain RNG). Must be equal length >= 2.
    blueprint_cheap
        Per-deck seat-means for the blueprint hero on a DISJOINT (independent) set of cheap no-search decks,
        used to pin ``E[blueprint]`` precisely. Length >= 2.
    optimal_beta
        If True (default) use the variance-minimizing control coefficient ``beta* = Cov(S,B)/Var(B)`` on the
        paired term: ``mean(S) - beta*(mean(B_paired) - mean(B_cheap))``. With ``beta=1`` the diff is the
        plain ``S - B``; ``beta*`` is never worse and is the textbook control-variate optimum. Unbiased for
        any fixed beta because ``E[B_paired] = E[B_cheap]``.

    Returns
    -------
    CVResult
        The variance-reduced mean + SE, the naive baseline for comparison, and the realized reduction.
    """
    S = np.asarray(search_paired, dtype=float).ravel()
    B = np.asarray(blueprint_paired, dtype=float).ravel()
    Bc = np.asarray(blueprint_cheap, dtype=float).ravel()
    n = S.size
    m = Bc.size
    if n != B.size:
        raise ValueError(f"search_paired ({n}) and blueprint_paired ({B.size}) must align (CRN pairs)")
    if n < 2 or m < 2:
        raise ValueError("need >=2 paired and >=2 cheap decks for a variance estimate")

    var_B = float(np.var(B, ddof=1))
    cov_SB = float(np.cov(S, B, ddof=1)[0, 1])
    beta = (cov_SB / var_B) if (optimal_beta and var_B > 0.0) else 1.0

    # Unbiased: E[S] - beta*(E[B_paired]-E[B_cheap]) = E[S] since E[B_paired]=E[B_cheap].
    mean = float(S.mean() - beta * (B.mean() - Bc.mean()))

    # Independent terms (paired set disjoint from cheap set): variances add.
    #   term1 = mean(S - beta*B)  over the paired decks
    #   term2 = beta*mean(B_cheap)
    resid = S - beta * B
    var_term1 = float(np.var(resid, ddof=1)) / n
    var_term2 = (beta ** 2) * float(np.var(Bc, ddof=1)) / m
    se = float(np.sqrt(var_term1 + var_term2))

    naive_mean = float(S.mean())
    naive_se = float(np.sqrt(np.var(S, ddof=1) / n))

    std_S = float(np.std(S, ddof=1))
    std_B = float(np.std(B, ddof=1))
    corr = (cov_SB / (std_S * std_B)) if (std_S > 0.0 and std_B > 0.0) else 0.0
    var_reduction = ((naive_se / se) ** 2) if se > 0.0 else float("inf")

    return CVResult(
        mean=mean, se=se, naive_mean=naive_mean, naive_se=naive_se,
        var_reduction=var_reduction, corr=corr, beta=float(beta),
        decks_paired=n, decks_cheap=m,
    )


def run_cv_absolute(field, *, search_factory, baseline_factory, paired_decks: int, cheap_decks: int,
                    seed: int = 0, cheap_seed_offset: int = 777, arm: str = "off",
                    run_fn=None, checkpoint_dir: str | None = None, resume: bool = False,
                    field_tag: str | None = None, **rotated_kwargs) -> tuple[CVResult, dict]:
    """Orchestrate the control-variate measurement of an EXPENSIVE hero's absolute tier win-rate.

    Runs three seat-rotated passes and combines them via :func:`decompose_absolute`:
      1. the EXPENSIVE ``search_factory`` hero on ``paired_decks`` decks at ``seed`` (the headline arm),
      2. the CHEAP ``baseline_factory`` hero on the SAME ``seed`` -> identical deck seeds -> CRN-paired,
      3. the CHEAP ``baseline_factory`` hero on ``cheap_decks`` DISJOINT decks (``seed+cheap_seed_offset``)
         to pin ``E[baseline]`` precisely.

    Because the deck seed is a pure function of ``(seed, deck_index)`` (see ``run_field_absolute_rotated``),
    passes 1 and 2 share boards + villain RNG deck-for-deck with NO change to the eval internals — the CRN
    pairing is automatic. ``arm`` selects which per-deck series to combine: ``"off"`` (pure base, the
    "beats a <tier>" headline) or ``"on"`` (base + exploit). Returns ``(CVResult, meta)`` where ``meta``
    carries the three raw aggregate means for cross-checking. ``run_fn`` is injectable for testing."""
    if arm not in ("off", "on"):
        raise ValueError("arm must be 'off' or 'on'")
    if run_fn is None:
        from pokerbot.eval.exploitation_battery import run_field_absolute_rotated as run_fn
    key = "off_deck_seat_means" if arm == "off" else "on_deck_seat_means"

    # Each of the three passes gets its OWN per-deck OFF checkpoint stream. Passes (ii) and (iii) share
    # deck INDICES but differ by effective seed (seed vs seed+offset) -> different deck_seeds -> different
    # OFF values, so the pass_key MUST encode hero-role + effective seed. field_tag namespaces multi-field
    # runs. checkpoint_dir=None (default) -> no checkpointing (the **rotated_kwargs path is unchanged).
    from pathlib import Path

    def _ckpt(pass_key: str) -> dict:
        if checkpoint_dir is None:
            return {}
        fn = f"{field_tag + '.' if field_tag else ''}{pass_key}.jsonl"
        return {"checkpoint_path": str(Path(checkpoint_dir) / fn), "resume": resume}

    s = run_fn(field, decks=paired_decks, seed=seed, hero_factory=search_factory,
               **_ckpt(f"search@{seed}"), **rotated_kwargs)
    b = run_fn(field, decks=paired_decks, seed=seed, hero_factory=baseline_factory,
               **_ckpt(f"base@{seed}"), **rotated_kwargs)
    bc = run_fn(field, decks=cheap_decks, seed=seed + cheap_seed_offset,
                hero_factory=baseline_factory,
                **_ckpt(f"base@{seed + cheap_seed_offset}"), **rotated_kwargs)

    if key not in s or key not in b or key not in bc:
        raise KeyError(f"run_fn result missing {key!r}; the eval must expose per-deck seat-means "
                       "(run_field_absolute_rotated does as of 2026-06-23)")
    res = decompose_absolute(s[key], b[key], bc[key])
    agg = "off_base" if arm == "off" else "on_exploit"
    meta = {
        "arm": arm,
        "search_raw_mean_bb": s[agg]["mean_bb"], "search_raw_se": s[agg]["se"],
        "baseline_paired_mean_bb": b[agg]["mean_bb"], "baseline_cheap_mean_bb": bc[agg]["mean_bb"],
        "paired_decks": paired_decks, "cheap_decks": cheap_decks, "seed": seed,
    }
    return res, meta


def decks_for_target_ci(per_deck_var: float, target_halfwidth_bb_per_100: float,
                        *, seats: int = 6) -> int:
    """How many decks to reach a target 95% half-width (in bb/100) given a per-deck seat-mean variance
    (bb/hand^2). Useful for sizing the expensive arm AFTER ``decompose_absolute`` reports the realized
    paired-diff variance. ``n = (1.96*sqrt(var) / hw_bb_per_hand)^2`` with ``hw_bb_per_hand = hw_bb_per_100/100``."""
    if target_halfwidth_bb_per_100 <= 0.0:
        raise ValueError("target half-width must be > 0")
    hw = target_halfwidth_bb_per_100 / 100.0
    return int(np.ceil((1.96 * np.sqrt(max(per_deck_var, 0.0)) / hw) ** 2))
