"""Adaptive opponent model — fast flaw-spotting + never-stale reads.

The persistent DB ([[opponent_db]]) stores LIFETIME additive counts with a GTO-base prior. That has
two exploitation weaknesses this module fixes:

  1. **Slow to spot flaws.** A new opponent is assumed to play the GTO base until ~8 observations
     override it, so we leave money on the table against the typical (exploitable) field early.
     Fix: an empirical-Bayes **population prior** — shrink toward how the FIELD actually plays
     (loose-passive / over-folds-to-jams per the population research), not toward Nash. We start
     exploiting the average opponent from hand one, then converge to their specific tendencies.

  2. **Can't adapt.** Lifetime pooling anchors the read to an opponent's past self, so when they
     improve or switch gears the model lags forever. Fix: **dual-rate exponential forgetting** — a
     SLOW EWMA (long half-life ≈ a stable lifetime read) and a FAST EWMA (short half-life ≈ recent
     behaviour). When the fast rate diverges from the slow rate we've detected a **change**; we trust
     the recent read and flag ``adapting``. Old behaviour decays out automatically, so the system
     keeps learning forever.

Output is the same per-(seat,history) aggregate push rate the range-width-shift estimator consumes,
so this is a drop-in for ``live_exploit.decide(estimator=...)`` via :meth:`AdaptiveOpponentModel.as_estimator`.
State is small (4 floats per (identity, infoset)) and JSON-serialisable for persistence.

Play-money / own-account / bot-arena use only.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from aof_cfr.infoset import aggregate_infoset_key, encode_infoset_key

# defaults (tunable). Half-lives are in OBSERVATIONS at that infoset, not hands.
SLOW_HALFLIFE = 250.0     # a stable lifetime-ish read
FAST_HALFLIFE = 12.0      # recent behaviour; ~this many obs to track a gear change
PRIOR_STRENGTH = 6.0      # pseudo-obs of the population prior (empirical-Bayes shrinkage)
ADAPT_THRESHOLD = 0.20    # |fast - slow| push-rate gap that flags a change
MIN_FAST_N = 4.0          # effective recent obs before we trust the fast signal


def _lam(halflife: float) -> float:
    return 0.5 ** (1.0 / max(halflife, 1e-9))


@dataclass
class _Dual:
    """Per-(identity, infoset) dual-rate exponential-forgetting sufficient stats."""
    ws: float = 0.0   # slow effective weight
    ss: float = 0.0   # slow effective pushes
    wf: float = 0.0   # fast effective weight
    sf: float = 0.0   # fast effective pushes

    def update(self, pushed: float, lam_s: float, lam_f: float) -> None:
        x = 1.0 if pushed else 0.0
        self.ws = lam_s * self.ws + 1.0
        self.ss = lam_s * self.ss + x
        self.wf = lam_f * self.wf + 1.0
        self.sf = lam_f * self.sf + x


class PopulationPrior:
    """Per-(seat, history) aggregate push rate the FIELD plays — the cold-start exploit target.

    Default models the documented short-stack population: over-folds vs jams (so we steal wider) and
    slightly under-shoves first-in. Pass ``flat_rate`` for a constant prior (tests / no artifacts),
    or a ``table`` dict ``{(num_players, seat, hist): rate}`` fitted from real data.
    """

    def __init__(self, *, overfold_vs_jam: float = 0.15, undershove: float = 0.08,
                 flat_rate: float | None = None, table: dict | None = None):
        self.overfold_vs_jam = float(overfold_vs_jam)
        self.undershove = float(undershove)
        self.flat_rate = flat_rate
        self.table = table or {}
        self._agg_cache: dict = {}

    def _base_agg(self, num_players: int, seat: str, hist: str, stack_bb: float) -> float:
        ck = (num_players, seat, hist, round(float(stack_bb), 3))
        if ck not in self._agg_cache:
            from aof_cfr.profiler import _COMBO_W, _COMBO_W_SUM, _base_arrays_seathist, _query_base_push
            arrs = _base_arrays_seathist(num_players, stack_bb, _query_base_push)
            base_p = arrs.get((seat, hist))
            self._agg_cache[ck] = (float((_COMBO_W * base_p).sum() / _COMBO_W_SUM)
                                   if base_p is not None else 0.0)
        return self._agg_cache[ck]

    def push_rate(self, num_players: int, seat: str, hist: str, stack_bb: float) -> float:
        if self.flat_rate is not None:
            return float(self.flat_rate)
        if (num_players, seat, hist) in self.table:
            return float(self.table[(num_players, seat, hist)])
        base = self._base_agg(num_players, seat, hist, stack_bb)
        facing_jam = "a" in (hist or "").lower()
        adj = (1.0 - self.overfold_vs_jam) if facing_jam else (1.0 - self.undershove)
        return float(np.clip(base * adj, 0.0, 1.0))


@dataclass
class AdaptiveOpponentModel:
    """Recency-weighted, population-prior-shrunk per-opponent push-rate model with change detection."""
    prior: PopulationPrior = field(default_factory=PopulationPrior)
    slow_halflife: float = SLOW_HALFLIFE
    fast_halflife: float = FAST_HALFLIFE
    prior_strength: float = PRIOR_STRENGTH
    adapt_threshold: float = ADAPT_THRESHOLD
    min_fast_n: float = MIN_FAST_N
    _d: dict = field(default_factory=dict)        # (identity, agg_key) -> _Dual

    def __post_init__(self):
        self._lam_s = _lam(self.slow_halflife)
        self._lam_f = _lam(self.fast_halflife)

    # -- feed --
    def observe(self, identity: str, num_players: int, seat: str, hist: str, stack_bb: float,
                pushed: bool) -> None:
        if not identity:
            return
        key = (identity, aggregate_infoset_key(seat, hist, stack_bb))
        self._d.setdefault(key, _Dual()).update(bool(pushed), self._lam_s, self._lam_f)

    def seed_from_db(self, db, identity: str, num_players: int, stack_bb: float) -> None:
        """Cold-start a returning opponent: load lifetime aggregate counts as the SLOW estimate (the
        fast estimate stays empty, so if they've changed since we last saw them the fast signal will
        catch it fresh). Idempotent-ish: call once when an opponent re-appears."""
        from aof_cfr.profiler import _base_arrays_seathist, _query_base_push
        for (seat, hist) in _base_arrays_seathist(num_players, stack_bb, _query_base_push):
            push, total = db.cell(identity, aggregate_infoset_key(seat, hist, stack_bb))
            if total > 0:
                key = (identity, aggregate_infoset_key(seat, hist, stack_bb))
                d = self._d.setdefault(key, _Dual())
                d.ws, d.ss = float(total), float(push)     # seed slow with lifetime aggregate

    # -- read --
    def estimate(self, identity: str, num_players: int, seat: str, hist: str, stack_bb: float) -> dict:
        prior_p = self.prior.push_rate(num_players, seat, hist, stack_bb)
        d = self._d.get((identity, aggregate_infoset_key(seat, hist, stack_bb)))
        if d is None or d.ws <= 0:
            return {"rate": prior_p, "slow": prior_p, "fast": None, "eff_n": 0.0,
                    "confidence": 0.0, "adapting": False}
        S = self.prior_strength
        slow = d.ss / d.ws
        post = (d.ss + S * prior_p) / (d.ws + S)            # slow shrunk toward the population prior
        fast = d.sf / d.wf if d.wf > 0 else None
        adapting = fast is not None and d.wf >= self.min_fast_n and abs(fast - slow) >= self.adapt_threshold
        rate = post
        if adapting:                                        # trust the recent read on a detected change
            g = d.wf / (d.wf + S)
            rate = post + g * (fast - post)
        eff = d.wf if adapting else d.ws
        return {"rate": float(np.clip(rate, 0.0, 1.0)), "slow": float(slow),
                "fast": (None if fast is None else float(fast)), "eff_n": float(eff),
                "confidence": float(eff / (eff + S)), "adapting": bool(adapting)}

    def push_rate(self, identity, num_players, seat, hist, stack_bb) -> float:
        return self.estimate(identity, num_players, seat, hist, stack_bb)["rate"]

    def is_adapting(self, identity, num_players, seat, hist, stack_bb) -> bool:
        return self.estimate(identity, num_players, seat, hist, stack_bb)["adapting"]

    # -- estimator hook for live_exploit.decide(estimator=...) --
    def as_estimator(self):
        """Return a callable matching ``estimate_policy_rows_shift``'s signature that uses THIS model
        instead of the DB cell. Reconstructs a per-bucket policy from the model's aggregate push rate
        via the same range-width shift (premiums preserved)."""
        def _est(db, identity, num_players, stack_bb, *, prior_strength=None, base_push_fn=None):
            return estimate_policy_rows_adaptive(self, identity, num_players, stack_bb,
                                                 base_push_fn=base_push_fn)
        return _est

    # -- persistence --
    def to_dict(self) -> dict:
        return {f"{ident}\x1f{key}": [d.ws, d.ss, d.wf, d.sf] for (ident, key), d in self._d.items()}

    def load_dict(self, data: dict) -> "AdaptiveOpponentModel":
        for k, v in (data or {}).items():
            ident, key = k.split("\x1f", 1)
            self._d[(ident, key)] = _Dual(*[float(x) for x in v])
        return self


def estimate_policy_rows_adaptive(model: AdaptiveOpponentModel, identity: str, num_players: int,
                                  stack_bb: float, *, base_push_fn=None) -> dict:
    """Per-infoset rows ``{key: [p_fold, p_push]}`` from the adaptive model — drop-in replacement for
    ``profiler.estimate_policy_rows_shift`` (same output contract)."""
    from aof_cfr.buckets import LABELS_169
    from aof_cfr.profiler import _base_arrays_seathist, _query_base_push, _range_width_shift
    bp = base_push_fn or _query_base_push
    rows: dict = {}
    for (seat, hist), base_p in _base_arrays_seathist(num_players, stack_bb, bp).items():
        rate = model.push_rate(identity, num_players, seat, hist, stack_bb)
        est_p = _range_width_shift(base_p, rate)
        for bi, lab in enumerate(LABELS_169):
            p = float(est_p[bi])
            rows[encode_infoset_key(seat, lab, hist, stack_bb)] = np.array([1.0 - p, p], dtype=np.float64)
    return rows
