"""Statistical opponent profiling (VPIP, PFR, AF, etc.)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field

# Fast-timescale decay — env-overridable for the G4 adaptation tuning sweep (must match
# bayes_exploit.GAMMA_FAST, which reads the same variable at import).
_GAMMA_FAST_DEFAULT = float(os.environ.get("POKERBOT_GAMMA_FAST", "0.90"))


@dataclass
class OpponentStats:
    hands: int = 0
    vpip: int = 0  # voluntarily put $ in pot preflop
    pfr: int = 0  # preflop raise
    three_bet: int = 0
    fold_to_cbet: int = 0
    cbet_faced: int = 0
    aggression_bets_raises: int = 0
    aggression_calls: int = 0
    wtsd: int = 0  # went to showdown
    saw_flop: int = 0
    # Per-format hand split (schema v7). The store is UNIFIED per player (one human, format-stable read),
    # but each hand is counted by format so reporting / a future ICM-aware model can segment cash vs MTT
    # without fragmenting the live posterior. ``hands`` stays the grand total (hands_cash+hands_tourney).
    hands_cash: int = 0
    hands_tourney: int = 0
    # Per-context Dirichlet sufficient stats: {context_key -> K=4 outcome-bin counts}
    # (see pokerbot.opponent.context). Empty for legacy/cold profiles ⇒ all scalar math below and
    # archetype()/confidence() are unaffected; the contextual Bayesian model uses it when present.
    ctx_counts: dict[str, list[int]] = field(default_factory=dict)
    # Dual-timescale DISCOUNTED counts (non-stationarity): fast (γ≈0.90, ~10-hand window) tracks gear-
    # changes; slow (γ≈0.99, ~100-hand) is the low-variance lifetime view. Decayed toward the
    # population-prior floor so unseen actions never collapse to zero. ctx_last = the `hands` index at
    # each context's last update (for LAZY decay). Empty for v2/legacy ⇒ bayes falls back to ctx_counts.
    ctx_fast: dict[str, list[float]] = field(default_factory=dict)
    ctx_slow: dict[str, list[float]] = field(default_factory=dict)
    ctx_last: dict[str, int] = field(default_factory=dict)
    # BOCPD-style MAP run-length per context: # of hands since the last detected regime change. Drives a
    # smooth post-change confidence ramp in bayes_exploit. Empty for v3/legacy ⇒ bayes falls back to the
    # binary fast/slow switch. ``ctx_shift_active`` is a TRANSIENT (not persisted) rising-edge latch.
    ctx_runlen: dict[str, int] = field(default_factory=dict)
    ctx_shift_active: dict[str, bool] = field(default_factory=dict)
    # SHOWDOWN-CONDITIONED bluff-ratio Beta counts (schema v5): {context_key -> [n_bluff, n_value]},
    # discounted (γ≈0.98, lazy decay like ctx_fast/slow but toward ZERO — the Beta prior is applied at
    # READ time by bayes_exploit.bluff_read, not embedded here). An entry is added only when a
    # villain's aggressive action was later GROUND-TRUTHED by a showdown reveal: hand strength at the
    # action < 0.45 ⇒ bluff, > 0.60 ⇒ value, middle ⇒ skipped (thin-value ambiguity stays out of the
    # posterior). Parallel keyspace over the LOCKED 22-context spec — context.py/spec_hash untouched.
    # Empty for v4/legacy profiles ⇒ bluff_read returns None ⇒ frequency-only exploitation (unchanged).
    ctx_bluff: dict[str, list[float]] = field(default_factory=dict)
    ctx_bluff_last: dict[str, int] = field(default_factory=dict)
    # Server-computed HUD stats (CoinPoker), authoritative over the player's WHOLE history:
    # {vpip, pfr, wtsd, wsd, threeBet, foldToThreeBet, cbet, foldToCbet, steal, ...} as the site's
    # values. A high-confidence PRIOR captured live (seed_from_hud) — kept SEPARATE from the
    # count-based fields above so it never distorts per-hand derivation. Empty ⇒ none captured.
    hud: dict = field(default_factory=dict)

    def add_ctx(self, context_key: str, bin_increments) -> None:
        """Element-wise fold a hand's per-context bin increments into the running counts."""
        from pokerbot.opponent.context import NUM_BINS

        row = self.ctx_counts.get(context_key)
        if row is None:
            row = [0] * NUM_BINS
            self.ctx_counts[context_key] = row
        for i, v in enumerate(bin_increments):
            if i < len(row):
                row[i] += int(v)

    def decay_and_add(
        self,
        context_key: str,
        bin_increments,
        prior_per_bin,
        hands_now: int,
        *,
        gamma_fast: float = _GAMMA_FAST_DEFAULT,
        gamma_slow: float = 0.99,
    ) -> None:
        """Stabilized discounted update for BOTH timescales (non-stationarity). Lazily decays over
        Δ = hands_now − last_update toward the population-prior floor, then adds this hand's bin counts:
            α ← γ^Δ·α + (1−γ^Δ)·prior_per_bin ;  α += bins
        The prior floor prevents zero-collapse of never-seen actions. ``prior_per_bin`` is the
        population pseudocount vector (sums to the prior strength)."""
        from pokerbot.opponent.context import NUM_BINS

        bins = list(bin_increments) + [0] * NUM_BINS
        prior = list(prior_per_bin) + [0.0] * NUM_BINS
        delta = max(0, int(hands_now) - int(self.ctx_last.get(context_key, hands_now)))
        for store, gamma in ((self.ctx_fast, gamma_fast), (self.ctx_slow, gamma_slow)):
            row = store.get(context_key)
            if row is None:
                row = list(prior[:NUM_BINS])  # seed at the population floor
            g = gamma ** delta
            for i in range(NUM_BINS):
                row[i] = g * row[i] + (1.0 - g) * prior[i] + float(bins[i])
            store[context_key] = row
        self.ctx_last[context_key] = int(hands_now)
        # BOCPD-style MAP run-length: reset to 0 on the RISING EDGE of a regime change (fast/slow
        # diverge for the first time), then INCREMENT every hand thereafter — even while the slow model
        # is still catching up (TV stays high) — so the post-change confidence ramp recovers promptly
        # rather than stalling for the slow half-life. A single counter + a transient edge latch.
        nf, ns = self.ctx_fast[context_key], self.ctx_slow[context_key]
        nfs, nss = sum(nf), sum(ns)
        if nfs <= 0 or nss <= 0:
            self.ctx_runlen[context_key] = 0
        else:
            from pokerbot.opponent.bayes_exploit import CHANGE_TV

            tv = 0.5 * sum(abs(nf[i] / nfs - ns[i] / nss) for i in range(NUM_BINS))
            shifting = tv > CHANGE_TV
            if shifting and not self.ctx_shift_active.get(context_key, False):
                self.ctx_runlen[context_key] = 0          # rising edge = a fresh change
            else:
                self.ctx_runlen[context_key] = self.ctx_runlen.get(context_key, 0) + 1
            self.ctx_shift_active[context_key] = shifting

    def bluff_decay_and_add(self, context_key: str, is_bluff: bool, hands_now: int,
                            *, weight: float = 1.0, gamma: float = 0.98) -> None:
        """Fold one showdown-ground-truthed aggressive action into the discounted bluff/value Beta
        counts. Lazy decay over Δ = hands_now − last_update toward zero (recency: a player who
        stops bluffing stops being called down within ~35 effective hands). ``weight`` < 1 down-
        weights low-confidence captures (live OCR parse confidence)."""
        row = self.ctx_bluff.get(context_key) or [0.0, 0.0]
        delta = max(0, int(hands_now) - int(self.ctx_bluff_last.get(context_key, hands_now)))
        g = gamma ** delta
        row = [g * row[0], g * row[1]]
        row[0 if is_bluff else 1] += float(weight)
        self.ctx_bluff[context_key] = row
        self.ctx_bluff_last[context_key] = int(hands_now)

    def ctx_total(self, context_key: str) -> int:
        """Total observed decisions at a context (the DBR sample size n for confidence)."""
        return int(sum(self.ctx_counts.get(context_key, ())))

    @property
    def vpip_pct(self) -> float:
        return self.vpip / max(self.hands, 1)

    @property
    def pfr_pct(self) -> float:
        return self.pfr / max(self.hands, 1)

    @property
    def aggression_factor(self) -> float:
        return self.aggression_bets_raises / max(self.aggression_calls, 1)

    @property
    def fold_to_cbet_pct(self) -> float:
        return self.fold_to_cbet / max(self.cbet_faced, 1)

    @property
    def aggression_frequency(self) -> float | None:
        """Postflop aggression frequency: (bets+raises) / (bets+raises + calls), in [0, 1].
        Returns ``None`` when the denominator is zero (UNDEFINED — a fresh profile with no
        postflop actions). Bug B1 fix: previously returned a silent 0.0 here which the
        archetype path read as "passive" — a station with zero observations was being mis-tagged
        as a NIT (the inverse). Tests cover this asymmetry at ``tests/test_opponent_property.py``.
        """
        b = int(self.aggression_bets_raises)
        c = int(self.aggression_calls)
        if (b + c) <= 0:
            return None
        return float(b) / float(b + c)

    @property
    def aggression_ratio(self) -> float:
        """Legacy "aggression ratio" (B+R / calls) — kept distinct from ``aggression_frequency``
        which is a normalized percentage. The two are equivalent when denominator > 0 but
        the ratio diverges as calls→0; downstream callers can pick whichever fits their math."""
        return self.aggression_bets_raises / max(self.aggression_calls, 1)

    @property
    def three_bet_pct(self) -> float | None:
        """3-bet percentage: three_bet / pfr, in [0, 1]. Returns ``None`` when the villain has not
        PFR'd at all (no opportunities ⇒ denominator zero). Bug B3 fix: previously returned a
        silent 0.0 which a zero-PFR villain carried forever."""
        pf = int(self.pfr)
        if pf <= 0:
            return None
        return min(1.0, float(self.three_bet) / float(pf))

    @property
    def wtsd_pct(self) -> float | None:
        """Went-to-showdown percentage: wtsd / saw_flop, in [0, 1]. Returns ``None`` when the
        villain has not seen a flop yet (no opportunities ⇒ denominator zero). Bug B5 fix:
        previously returned a silent 0.0 which made every cold profile read as "never saw showdown"."""
        sf = int(self.saw_flop)
        if sf <= 0:
            return None
        return min(1.0, float(self.wtsd) / float(sf))

    def archetype(self) -> str:
        # FIX B1 (table-toughness read): the 5-hand window let 1/5 VPIP tag a villain and 0/5 brand a
        # NIT, so a fresh play-money table read "tough" hand-one. Require a real local sample (>=12)
        # before ANY tag, and a larger one (>=20) before the high-impact "nit" verdict. A server HUD with
        # a wide VPIP-PFR gap (a loose-passive limp/flat field — the population signal) overrides the
        # count window directly as a calling_station, the dominant play-money archetype.
        hud = getattr(self, "hud", None)
        if hud:
            from pokerbot.opponent.value_sizing import _pct
            v, pf = _pct(hud, "vpip"), _pct(hud, "pfr")
            if v is not None and pf is not None and (v - pf) >= 0.25:
                return "calling_station"
        if self.hands < 12:
            return "unknown"
        if self.hands >= 20 and self.vpip_pct < 0.15 and self.pfr_pct < 0.12:
            return "nit"
        if self.vpip_pct > 0.45 and self.aggression_factor < 1.5:
            return "calling_station"
        if self.vpip_pct > 0.35 and self.aggression_factor > 2.5:
            return "lag"
        if 0.18 <= self.vpip_pct <= 0.28 and self.pfr_pct >= 0.15:
            return "tag"
        return "unknown"


@dataclass
class OpponentTracker:
    """Track stats per villain seat/id."""

    profiles: dict[int, OpponentStats] = field(default_factory=dict)

    def get(self, player_id: int) -> OpponentStats:
        if player_id not in self.profiles:
            self.profiles[player_id] = OpponentStats()
        return self.profiles[player_id]

    def record_hand(
        self,
        player_id: int,
        *,
        vpip: bool = False,
        pfr: bool = False,
        three_bet: bool = False,
        saw_flop: bool = False,
        wtsd: bool = False,
        bet_or_raise: int = 0,
        calls: int = 0,
        fold_to_cbet: bool = False,
        cbet_faced: bool = False,
    ) -> None:
        s = self.get(player_id)
        s.hands += 1
        s.vpip += int(vpip)
        s.pfr += int(pfr)
        s.three_bet += int(three_bet)
        s.saw_flop += int(saw_flop)
        s.wtsd += int(wtsd)
        s.aggression_bets_raises += bet_or_raise
        s.aggression_calls += calls
        s.fold_to_cbet += int(fold_to_cbet)
        s.cbet_faced += int(cbet_faced)

    def confidence(self, player_id: int) -> float:
        """0-1 confidence based on sample size."""
        hands = self.get(player_id).hands
        return min(1.0, hands / 50.0)
