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

import json
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from aof_cfr.infoset import aggregate_infoset_key, encode_infoset_key

# defaults (tunable). Half-lives are in OBSERVATIONS at that infoset, not hands.
SLOW_HALFLIFE = 250.0     # a stable lifetime-ish read
FAST_HALFLIFE = 12.0      # recent behaviour; ~this many obs to track a gear change
PRIOR_STRENGTH = 6.0      # pseudo-obs of the population prior (empirical-Bayes shrinkage)
ADAPT_THRESHOLD = 0.20    # |fast - slow| push-rate gap that flags a change
MIN_FAST_N = 4.0          # effective recent obs before we trust the fast signal
POOL_PRIOR_STRENGTH = 10.0  # s2: depth-pooled (seat,hist) rate -> population-prior shrinkage
POOL_WEIGHT = 0.5           # rho: pooled decayed obs count at half weight toward confidence
# Instability memory (drives the router's Restricted-Nash-Response cap, exploit_router._route_epilogue):
# once an opponent's fast read diverges from its slow read (a gear change / an ADAPTING counter-
# exploiter), the opponent is flagged unstable and the flag DECAYS slowly, so a proven adapter stays
# clamped between its move windows instead of letting us re-commit to a pure BR they will punish.
INSTAB_DECAY = 0.997        # per-qualifying-obs decay of the instability memory (~230 obs half-life)


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
    population_confidence: float = 0.0   # cold-start trust in the population prior (0 = unknown->base)
    instab_decay: float = INSTAB_DECAY   # slow decay of the per-opponent instability memory
    path: object = None                  # optional json for persistence
    _d: dict = field(default_factory=dict)        # (identity, agg_key) -> _Dual
    _instab: dict = field(default_factory=dict)   # identity -> instability memory in [0,1]
    _seeded: set = field(default_factory=set)     # (identity, n, depth) already seeded from the DB

    def __post_init__(self):
        self._lam_s = _lam(self.slow_halflife)
        self._lam_f = _lam(self.fast_halflife)
        # Guards _d/_seeded: multiple tables observing the same opponent concurrently used to
        # race the read-modify-write of the dual EWMAs. Persistence is whole-file single-writer
        # (one engine process owns adaptive_model.json; the durable multi-process store is the
        # opponent DB, which has delta-merge saves).
        import threading
        self._mu = threading.RLock()

    # -- feed --
    def observe(self, identity: str, num_players: int, seat: str, hist: str, stack_bb: float,
                pushed: bool) -> None:
        self.observe_key(identity, aggregate_infoset_key(seat, hist, stack_bb), pushed)

    def observe_key(self, identity: str, agg_key: str, pushed: bool) -> None:
        """Feed one observation by its aggregate infoset key (what the DB / ingest already produce)."""
        if not identity:
            return
        with self._mu:
            d = self._d.setdefault((identity, agg_key), _Dual())
            d.update(bool(pushed), self._lam_s, self._lam_f)
            # Instability memory: the fast read diverging from the slow read at this infoset means the
            # opponent is changing (or actively counter-adapting to us). Jump the memory up to any new
            # divergence immediately; otherwise decay it slowly so a proven adapter stays flagged.
            if d.wf >= self.min_fast_n and d.ws > 0:
                div = min(abs(d.sf / d.wf - d.ss / d.ws) / max(self.adapt_threshold, 1e-9), 1.0)
                cur = self._instab.get(identity, 0.0)
                self._instab[identity] = div if div >= cur else self.instab_decay * cur

    def seed_from_db(self, db, identity: str, num_players: int, stack_bb: float) -> None:
        """Cold-start a returning opponent: load lifetime aggregate counts as the SLOW estimate (the
        fast estimate stays empty, so a since-last-session change is caught fresh). Idempotent and
        non-clobbering: only seeds infosets the live model hasn't already touched; runs the DB scan
        once per (identity, players, depth)."""
        if not identity:
            return
        with self._mu:
            tag = (identity, num_players, round(float(stack_bb), 3))
            if tag in self._seeded:
                return
            self._seeded.add(tag)
        from aof_cfr.profiler import _base_arrays_seathist, _query_base_push
        pairs = list(_base_arrays_seathist(num_players, stack_bb, _query_base_push))
        with self._mu:
            for (seat, hist) in pairs:
                agg = aggregate_infoset_key(seat, hist, stack_bb)
                if (identity, agg) in self._d:
                    continue                               # don't clobber a live read
                push, total = db.cell(identity, agg)
                if total > 0:
                    d = self._d.setdefault((identity, agg), _Dual())
                    d.ws, d.ss = float(total), float(push)  # seed slow with lifetime aggregate

    def _depth_pool(self, identity: str, seat: str, hist: str, stack_bb: float) -> tuple[float, float]:
        """Decayed (weight, pushes) summed over the OTHER depth buckets of this (seat, hist)."""
        from aof_cfr.infoset import STACK_DEPTH_BUCKETS
        own_key = aggregate_infoset_key(seat, hist, stack_bb)
        ws = ss = 0.0
        with self._mu:
            for db_ in STACK_DEPTH_BUCKETS:
                k = aggregate_infoset_key(seat, hist, float(db_))
                if k == own_key:
                    continue
                dd = self._d.get((identity, k))
                if dd is not None:
                    ws += dd.ws
                    ss += dd.ss
        return ws, ss

    # -- read --
    def estimate(self, identity: str, num_players: int, seat: str, hist: str, stack_bb: float) -> dict:
        """Hierarchical read: own-depth cell -> depth-pooled (seat,hist) -> population prior.

        With no cross-depth data this collapses exactly to the original two-level posterior;
        the change-detection (fast vs slow) stays strictly on the own-depth cell."""
        prior_p = self.prior.push_rate(num_players, seat, hist, stack_bb)
        with self._mu:
            d = self._d.get((identity, aggregate_infoset_key(seat, hist, stack_bb)))
            if d is not None:
                d = _Dual(d.ws, d.ss, d.wf, d.sf)   # read a consistent copy
        ws_p, ss_p = self._depth_pool(identity, seat, hist, stack_bb)
        if (d is None or d.ws <= 0) and ws_p <= 0:
            return {"rate": prior_p, "slow": prior_p, "fast": None, "eff_n": 0.0,
                    "confidence": 0.0, "adapting": False}
        S = self.prior_strength
        m_pool = (ss_p + POOL_PRIOR_STRENGTH * prior_p) / (ws_p + POOL_PRIOR_STRENGTH)
        ws_c, ss_c = (d.ws, d.ss) if d is not None else (0.0, 0.0)
        slow = ss_c / ws_c if ws_c > 0 else m_pool
        post = (ss_c + S * m_pool) / (ws_c + S)             # cell shrunk toward the depth pool
        fast = (d.sf / d.wf) if (d is not None and d.wf > 0) else None
        adapting = (fast is not None and d.wf >= self.min_fast_n
                    and abs(fast - slow) >= self.adapt_threshold)
        rate = post
        if adapting:                                        # trust the recent read on a detected change
            g = d.wf / (d.wf + S)
            rate = post + g * (fast - post)
        eff = (d.wf if adapting else ws_c) + POOL_WEIGHT * ws_p
        return {"rate": float(np.clip(rate, 0.0, 1.0)), "slow": float(slow),
                "fast": (None if fast is None else float(fast)), "eff_n": float(eff),
                "confidence": float(eff / (eff + S)), "adapting": bool(adapting)}

    def opponent_confidence(self, identity: str) -> float:
        """Per-opponent scalar for the router gate. Effective decayed sample over all infosets ->
        confidence, floored by ``population_confidence`` (so cold-start exploitation of the field
        prior can fire before any per-opponent data). Mirrors the DB ``n_obs`` confidence, but the
        weight DECAYS, so a long-absent / changed opponent's confidence relaxes back down."""
        if not identity:
            return 0.0
        with self._mu:
            eff = sum(d.ws for (i, _k), d in self._d.items() if i == identity)
        personal = eff / (eff + self.prior_strength) if eff > 0 else 0.0
        return float(max(self.population_confidence, personal))

    def opponent_eff_n(self, identity: str) -> float:
        """The opponent's TRUE effective (decayed) sample size — the same ``eff`` that backs
        ``opponent_confidence`` but WITHOUT the population-confidence floor. The router's LCB
        pessimism uses this directly instead of inverting the floored confidence with a mismatched
        prior strength (see exploit_router._eff_n_for)."""
        if not identity:
            return 0.0
        with self._mu:
            return float(sum(d.ws for (i, _k), d in self._d.items() if i == identity))

    def opponent_stability(self, identity: str) -> float:
        """Per-opponent read stability in [0, 1] for the router's Restricted-Nash-Response cap.

        1.0 = a stable, stationary read (the fast EWMA agrees with the slow EWMA, or we have no
        fast data yet) — the exploit may commit fully (a static leak has no downside). It falls
        toward 0 as the opponent is seen to ADAPT (fast diverges from slow), with hysteresis so a
        counter-exploiter we have already caught moving stays clamped between its move windows.
        The router multiplies the exploit-cap by this, so a non-stationary opponent only ever pulls
        us a bounded fraction toward the pure best response they would otherwise punish."""
        if not identity:
            return 1.0
        with self._mu:
            return float(1.0 - min(max(self._instab.get(identity, 0.0), 0.0), 1.0))

    def push_rate(self, identity, num_players, seat, hist, stack_bb) -> float:
        return self.estimate(identity, num_players, seat, hist, stack_bb)["rate"]

    def is_adapting(self, identity, num_players, seat, hist, stack_bb) -> bool:
        return self.estimate(identity, num_players, seat, hist, stack_bb)["adapting"]

    # -- estimator hook for live_exploit.decide(estimator=...) --
    def as_estimator(self):
        """Return a callable matching ``estimate_policy_rows_shift``'s signature that uses THIS model
        instead of the DB cell. Reconstructs a per-bucket policy from the model's aggregate push rate
        via the same range-width shift (premiums preserved); exact-bucket showdown observations from
        the durable DB override the reconstruction where they exist."""
        def _est(db, identity, num_players, stack_bb, *, prior_strength=None, base_push_fn=None):
            return estimate_policy_rows_adaptive(self, identity, num_players, stack_bb,
                                                 base_push_fn=base_push_fn, db=db)
        return _est

    # -- persistence --
    def to_dict(self) -> dict:
        with self._mu:
            return {f"{ident}\x1f{key}": [d.ws, d.ss, d.wf, d.sf] for (ident, key), d in self._d.items()}

    def load_dict(self, data: dict) -> "AdaptiveOpponentModel":
        with self._mu:
            for k, v in (data or {}).items():
                ident, key = k.split("\x1f", 1)
                self._d[(ident, key)] = _Dual(*[float(x) for x in v])
        return self

    # -- on-disk persistence (atomic + fsync; crash-safe; single-writer per process) --
    def load(self) -> "AdaptiveOpponentModel":
        if self.path and Path(self.path).is_file():
            try:
                raw = json.loads(Path(self.path).read_text(encoding="utf-8"))
                from aof_cfr.infoset import infoset_key_spec_hash
                stored = raw.get("infoset_key_spec_hash")
                if stored is not None and stored != infoset_key_spec_hash():
                    bad = Path(self.path).with_suffix(".json.corrupt")
                    os.replace(self.path, bad)
                    print(f"AdaptiveOpponentModel: key-spec mismatch in {self.path}; "
                          f"backed up to {bad}, starting fresh")
                    return self
                self.load_dict(raw.get("duals", {}))
                with self._mu:
                    self._instab.update({k: float(v) for k, v in (raw.get("instab", {}) or {}).items()})
            except (OSError, ValueError):
                pass
        return self

    def save(self) -> None:
        if not self.path:
            return
        from aof_cfr.infoset import infoset_key_spec_hash
        p = Path(self.path)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(p.suffix + f".tmp{os.getpid()}-{threading.get_ident()}")
        with self._mu:
            instab = dict(self._instab)
        payload = {"schema_version": 1, "infoset_key_spec_hash": infoset_key_spec_hash(),
                   "duals": self.to_dict(), "instab": instab}
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, p)


def estimate_policy_rows_adaptive(model: AdaptiveOpponentModel, identity: str, num_players: int,
                                  stack_bb: float, *, base_push_fn=None, db=None) -> dict:
    """Per-infoset rows ``{key: [p_fold, p_push]}`` from the adaptive model — drop-in replacement for
    ``profiler.estimate_policy_rows_shift`` (same output contract). Pass the durable ``db`` to let
    exact-bucket showdown observations override the range-width reconstruction where observed."""
    from aof_cfr.buckets import LABELS_169
    from aof_cfr.profiler import (
        _base_arrays_seathist, _query_base_push, _range_width_shift, exact_bucket_override)
    bp = base_push_fn or _query_base_push
    cells = db.cells_for(identity) if db is not None and identity else {}
    s = float(model.prior_strength)
    rows: dict = {}
    for (seat, hist), base_p in _base_arrays_seathist(num_players, stack_bb, bp).items():
        rate = model.push_rate(identity, num_players, seat, hist, stack_bb)
        est_p = _range_width_shift(base_p, rate)
        if cells:
            est_p = exact_bucket_override(est_p, cells, seat, hist, stack_bb, s)
        for bi, lab in enumerate(LABELS_169):
            p = float(est_p[bi])
            rows[encode_infoset_key(seat, lab, hist, stack_bb)] = np.array([1.0 - p, p], dtype=np.float64)
    return rows
