"""Rank two 6-max NLHE BLUEPRINT TABLES by win-rate (bb/100) with a PAIRED (B-A) CI.

(Units note: everything here is bb/100 = bb per 100 hands. mbb/game = bb/hand * 1000 = bb/100 * 10 --
this module reports bb/100 throughout; multiply by 10 if you want mbb/game.)

Why this module exists (the 6-max signal problem)
-------------------------------------------------
For 3+ players there is no unique Nash and no meaningful single exploitability scalar, so LBR /
best-response CANNOT reliably rank one blueprint against another (our 6-max LBR even RISES as the
average strategy sharpens -- expected, not a bug). The field (Pluribus, Brown & Sandholm Science 2019)
never used exploitability: it ranked strength by mbb/game vs a FIXED opponent pool with seat-rotated
duplicate dealing and reported a CI, never a bare point estimate. This module is that signal for us:

  * ``rank_head_to_head(A, B)`` -- A as hero vs a homogeneous field of B, and B as hero vs a field of A,
    on the SAME decks (common random numbers). delta = (B-vs-field-of-A) - (A-vs-field-of-B); positive =>
    B extracts more when surrounded by A than A does when surrounded by B => B is the stronger table.
    Cheap A/B, but 6-max is NON-TRANSITIVE so it is a complement, not the sole verdict.
  * ``rank_vs_pool(A, B)`` -- A and B each measured vs a FIXED realistic CoinPoker-mirror field
    (population-weighted over {responsive_station, responsive_reg, hs_nit, pro_balanced}), on the SAME
    decks. The Pluribus recipe; absolute, comparable, and dominated by the value-extraction target.
  * ``decide(...)`` -- gate a change on BOTH signals agreeing (CI excludes 0 in the same direction);
    DISAGREEMENT is the non-transitivity flag.

How the variance reduction works (no broken AIVAT, no search)
-------------------------------------------------------------
We compare the raw blueprint TABLES (pure-tabular play via ``TabularBlueprintAdapter``), NOT the deployed
search agent -- a training change (DCFR vs CFR+, more iters, a richer abstraction) produces a different
TABLE, and ranking the table directly is the cheap, clean question. The search path's AIVAT is broken
(negative variance reduction) and irrelevant here. We reuse the gold-standard SEAT-ROTATED DUPLICATE
estimator (``exploitation_battery.run_field_absolute_rotated``), which replays each deck with the hero in
every seat (cancelling hole-card-allocation luck) and emits the per-deck seat-mean. Because the deck seed
is a pure function of ``(seed, deck_index)``, running it twice (hero=A, hero=B) at the SAME seed yields
CRN-PAIRED per-deck arrays: identical boards + villain RNG, so the per-deck difference ``b_d - a_d``
cancels nearly all card / board / villain variance and only the A<->B strategy divergence survives. The
paired CI over those per-deck deltas is far tighter than two independent absolute measurements.

A snapshot that equals itself (A == B) replays BYTE-IDENTICALLY on every deck, so every per-deck delta is
exactly 0 -- ranking a blueprint against itself can NEVER fabricate a winner (the no-false-positive
property, asserted in tests/test_blueprint_ranking.py).

Firewall-safe: a new file that only READS the engine + existing eval bots; it never touches the engine,
AoF, or NUM_PLAYERS.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

import numpy as np

from pokerbot.engine.game import NUM_PLAYERS


# --------------------------------------------------------------------------------------------------
# The pure-tabular blueprint player (serves as both the hero-under-test and a fixed-blueprint opponent)
# --------------------------------------------------------------------------------------------------
class TabularBlueprintAgent:
    """SAMPLES a trained blueprint's mixed strategy at every decision via ``TabularBlueprintAdapter``
    (no search). Implements the seat-rotated-harness HERO contract (``reseed`` / ``set_profile`` /
    ``reset_hand`` no-ops) AND the plain field-bot contract (``__call__(state) -> Action``), so ONE
    class is used as the blueprint under test (hero) and as a fixed-blueprint opponent (a homogeneous
    field of the other version, for head-to-head).

    ``snapshot`` is the raw ``{fingerprint(bytes) -> strat row}`` table (what ``nlcfr_to_snapshot`` and
    ``validate_blueprint_native._load`` produce); the adapter legal-masks + renormalizes and falls back
    to legal-uniform on a fingerprint miss. We COUNT misses (``fallback_count`` / ``decision_count``) so
    a driver can warn when an abstraction mismatch is silently degrading the measurement (the serve-skew
    lesson) -- but ranking A vs B under the SAME abstraction stays fair even with a uniform tail.
    """

    def __init__(self, snapshot, rng: random.Random | None = None, *, client_bet_sizing: bool = True):
        from pokerbot.search.tabular_blueprint_adapter import TabularBlueprintAdapter

        self.rng = rng or random.Random(0)
        self.adapter = TabularBlueprintAdapter(snapshot, client_bet_sizing=client_bet_sizing)
        self.decision_count = 0
        self.fallback_count = 0

    def __call__(self, state):
        from pokerbot.search.state_bridge import game_state_to_env
        from pokerbot.training.self_play import _index_to_action as _i2a

        self.decision_count += 1
        try:
            seat = state.action_seat
            env = game_state_to_env(state, seat)
            p = self.adapter.blueprint(env, seat)
            s = float(p.sum())
            if s > 0.0 and np.isfinite(p).all():
                idx = self.rng.choices(range(len(p)), weights=(p / s).tolist())[0]
                return _i2a(state, seat, int(idx))
        except Exception:  # noqa: BLE001 -- a bridge/lookup failure must degrade gracefully, never crash a run
            pass
        # Legal-uniform fallback (adapter miss or bridge failure): sample a legal action directly.
        self.fallback_count += 1
        legal = list(state.legal_actions())
        return self.rng.choice(legal)

    # --- seat-rotated-harness hero contract (all no-ops for a stateless pure-tabular blueprint) -------
    def reseed(self, seed: int) -> None:
        """Reset the sampling RNG to a pure function of the deck seed (the harness calls this per deck).
        This is what makes A == B replay byte-identically -> every per-deck delta is exactly 0."""
        self.rng = random.Random(int(seed))

    def set_profile(self, *_a, **_k) -> None:
        return None

    def set_opponent_profiles(self, *_a, **_k) -> None:
        return None

    def set_exploit_enabled(self, *_a, **_k) -> None:
        return None

    def reset_hand(self, *_a, **_k) -> None:
        return None


# --------------------------------------------------------------------------------------------------
# Paired (B - A) statistics over CRN per-deck seat-means
# --------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class PairedResult:
    """A paired (B - A) win-rate difference over CRN-matched per-deck seat-means.

    All bb/100 (= mbb/game / 10). ``delta`` is mean(b_d - a_d) * 100; ``ci95`` its 95% half-width.
    ``a`` / ``b`` are the two ABSOLUTE per-arm win-rates (context). ``var_reduction`` is how much the
    CRN pairing tightened the difference vs treating the arms as independent (Var_indep / Var_paired).
    """

    n: int
    delta_bb_per_100: float
    se_bb_per_100: float
    ci95_bb_per_100: float
    t_stat: float
    a_bb_per_100: float
    b_bb_per_100: float
    var_reduction: float
    detail: dict = field(default_factory=dict)

    @property
    def excludes_zero(self) -> bool:
        """True iff the 95% CI for the (B-A) difference is entirely on one side of 0."""
        return abs(self.delta_bb_per_100) > self.ci95_bb_per_100

    @property
    def b_stronger(self) -> bool:
        """True iff B beats A with the whole 95% CI above 0."""
        return (self.delta_bb_per_100 - self.ci95_bb_per_100) > 0.0

    @property
    def a_stronger(self) -> bool:
        """True iff A beats B with the whole 95% CI below 0."""
        return (self.delta_bb_per_100 + self.ci95_bb_per_100) < 0.0

    def summary(self) -> str:
        return (f"delta(B-A) = {self.delta_bb_per_100:+.2f} +- {self.ci95_bb_per_100:.2f} bb/100  "
                f"(A {self.a_bb_per_100:+.2f}, B {self.b_bb_per_100:+.2f}; n={self.n}, "
                f"t={self.t_stat:+.2f}, VR {self.var_reduction:.1f}x)")


def paired_delta(a_means, b_means, *, detail: dict | None = None) -> PairedResult:
    """Paired (B - A) bb/100 + 95% CI from two CRN-matched per-deck seat-mean arrays (bb/hand units).

    ``a_means[i]`` and ``b_means[i]`` are the SAME deck ``i`` under common random numbers. The paired
    difference ``d_i = b_i - a_i`` cancels the shared card/board/villain variance; we report
    mean(d)*100 +- 1.96*se(d)*100 and the realized variance reduction vs an unpaired comparison."""
    a = np.asarray(a_means, dtype=np.float64).ravel()
    b = np.asarray(b_means, dtype=np.float64).ravel()
    if a.shape != b.shape:
        raise ValueError(f"a_means ({a.size}) and b_means ({b.size}) must be CRN-aligned (equal length)")
    n = a.size
    if n < 2:
        raise ValueError("need >= 2 paired decks for a variance estimate")

    d = b - a
    mean_d = float(d.mean())
    se_d = float(d.std(ddof=1) / math.sqrt(n))
    # Variance reduction from pairing: an UNPAIRED comparison of two independent n-sample means would have
    # se^2 = Var(a)/n + Var(b)/n; the paired diff has Var(b-a)/n. The ratio is the CRN lift (>1 = pairing
    # helped; ~1 = arms uncorrelated; the strategies that share decks should be highly correlated).
    var_indep = float(a.var(ddof=1) + b.var(ddof=1)) / n
    var_paired = float(d.var(ddof=1)) / n
    vr = (var_indep / var_paired) if var_paired > 0.0 else float("inf")
    t = (mean_d / se_d) if se_d > 0.0 else 0.0

    det = {"decks": n, "mean_delta_bb_per_hand": mean_d, "se_bb_per_hand": se_d}
    if detail:
        det.update(detail)
    return PairedResult(
        n=n,
        delta_bb_per_100=mean_d * 100.0,
        se_bb_per_100=se_d * 100.0,
        ci95_bb_per_100=1.96 * se_d * 100.0,
        t_stat=t,
        a_bb_per_100=float(a.mean()) * 100.0,
        b_bb_per_100=float(b.mean()) * 100.0,
        var_reduction=vr,
        detail=det,
    )


# --------------------------------------------------------------------------------------------------
# The curated CoinPoker-mirror opponent pool (population-weighted IRC archetypes)
# --------------------------------------------------------------------------------------------------
def _combined_bot_builder(name: str, rng: random.Random | None = None, params: dict | None = None):
    """Build a pool/field opponent by name. Routes the size-elastic ``responsive_*`` bots to their own
    factory and everything else (hs_nit, pro_balanced, blueprint, ...) to ``build_bot``. The reserved
    name ``__snapshot__`` is handled by the head-to-head builder (see ``_snapshot_bot_builder``)."""
    from pokerbot.eval.bots import build_bot
    from pokerbot.eval.responsive_station import _RESPONSIVE_NAMES, build_responsive_bot

    if name in _RESPONSIVE_NAMES:
        return build_responsive_bot(name, rng, params)
    return build_bot(name, rng, params=params)


def default_pool_roster() -> list[tuple[str, float]]:
    """The realistic CoinPoker-mirror roster: ``[(bot_name, weight), ...]`` with weights summing to 1,
    derived from the IRC population archetype distribution (``field_sampler.archetype_probabilities``)
    mapped onto the faithful eval bots. The dominant calling-station mass routes to the size-elastic
    ``responsive_station`` (the value target); the reg tail to ``responsive_reg``; the nit to ``hs_nit``;
    the carved pro slice to the near-equilibrium ``pro_balanced`` (the counter-exploit-risk seat). Falls
    back to a fixed station-dominated roster if the population summary is unavailable."""
    try:
        from pokerbot.eval.field_sampler import archetype_probabilities

        p = archetype_probabilities()
        roster = {
            "responsive_station": p.get("calling_station", 0.0),
            "responsive_reg": p.get("lag", 0.0) + p.get("tag", 0.0),
            "hs_nit": p.get("nit", 0.0),
            "pro_balanced": p.get("pro", 0.0),
        }
        total = sum(roster.values())
        if total > 0.0:
            return [(k, v / total) for k, v in roster.items() if v > 0.0]
    except Exception:  # noqa: BLE001 -- never let a missing summary file break ranking; use the default
        pass
    # Station-dominated default mirroring the canonical co-rotated ring (3 station / 1 reg / 1 nit + pro).
    return [("responsive_station", 0.60), ("responsive_reg", 0.20),
            ("hs_nit", 0.10), ("pro_balanced", 0.10)]


# --------------------------------------------------------------------------------------------------
# Serve-fidelity guard: a blueprint that misses its abstraction plays UNIFORM, and uniform-vs-uniform
# silently reports delta ~ 0 -- the worst failure mode (a real change looks like "no difference"). The
# encoder bakes POKERBOT_BETTING_ABSTRACTION / POKERBOT_POSTFLOP_BUCKETS / POKERBOT_PREFLOP_LOSSLESS at
# IMPORT, so they must be set (to match the snapshot's training stamp) BEFORE pokerbot is imported.
# --------------------------------------------------------------------------------------------------
def serve_coverage_probe(snapshot, *, n_hands: int = 40, seed: int = 0, n_seats: int = NUM_PLAYERS):
    """Play ``n_hands`` with ``snapshot`` as the hero vs calling stations and return the POSTFLOP blueprint
    hit-rate (flop+turn+river hits / lookups). Returns ``None`` if coverage instrumentation is off
    (``POKERBOT_BP_COVERAGE`` unset) -- it cannot be measured. A hit-rate near 0 means the serve
    abstraction does not match the snapshot's training abstraction, so the snapshot plays uniform and any
    ranking is meaningless."""
    from pokerbot.engine.game import play_hand
    from pokerbot.eval import lbr
    from pokerbot.eval.bots import build_bot

    if not getattr(lbr, "_BP_COVERAGE_ON", False):
        return None
    lbr.bp_coverage_reset()
    hero = TabularBlueprintAgent(snapshot, random.Random(seed))
    for h in range(n_hands):
        pol = {0: hero}
        for s in range(1, n_seats):
            pol[s] = build_bot("calling_station", random.Random(100 + s))
        play_hand(pol, rng=random.Random(seed * 1000 + h))
    rep = lbr.bp_coverage_report()
    hits = sum(rep[s]["hits"] for s in ("flop", "turn", "river") if s in rep)
    miss = sum(rep[s]["misses"] for s in ("flop", "turn", "river") if s in rep)
    tot = hits + miss
    if not tot:  # no postflop nodes sampled (shouldn't happen vs calling stations) -> can't judge
        print("[rank] WARNING: serve-coverage probe sampled 0 postflop nodes; cannot verify coverage",
              flush=True)
        return None
    return hits / tot


def _assert_serve_coverage(snapshot, label: str, *, floor: float | None) -> None:
    """Fail loud if a NON-EMPTY snapshot serves below ``floor`` postflop coverage (a uniform-play
    measurement). Skips an EMPTY snapshot (an intentional uniform baseline) and skips silently when
    instrumentation is off (warning once so the operator knows the guard is dark)."""
    if floor is None or not snapshot:
        return
    cov = serve_coverage_probe(snapshot)
    if cov is None:
        print(f"[rank] WARNING: serve-coverage guard DISABLED for {label} "
              "(set POKERBOT_BP_COVERAGE=1 before importing pokerbot to enable it)", flush=True)
        return
    if cov < floor:
        raise RuntimeError(
            f"blueprint {label} serves only {cov*100:.1f}% postflop coverage (< {floor*100:.0f}% floor): "
            "the snapshot MISSES under the active abstraction and plays UNIFORM, so any ranking would be "
            "uniform-vs-uniform noise. Set POKERBOT_BETTING_ABSTRACTION / POKERBOT_POSTFLOP_BUCKETS / "
            "POKERBOT_PREFLOP_LOSSLESS (and num_buckets/mode) to MATCH the snapshot's training stamp BEFORE "
            "importing pokerbot (the encoder bakes them at import).")
    print(f"[rank] serve coverage {label}: {cov*100:.1f}% postflop (OK >= {floor*100:.0f}%)", flush=True)


# --------------------------------------------------------------------------------------------------
# The rotated-duplicate driver (one arm = one hero snapshot vs one homogeneous field)
# --------------------------------------------------------------------------------------------------
def _rotated_seat_means(hero_snapshot, *, villain_name=None, villain_snapshot=None,
                        decks: int, seed: int, n_seats: int = NUM_PLAYERS, button_seat: int = 3,
                        progress_every: int = 0, progress_cb=None) -> np.ndarray:
    """Run ONE rotated-duplicate arm and return the per-deck OFF seat-means (bb/hand) as an array.

    The hero is a pure-tabular ``TabularBlueprintAgent(hero_snapshot)``. The field is HOMOGENEOUS
    (``run_field_absolute_rotated`` requires it): either ``n_seats-1`` copies of a named archetype
    (``villain_name``, vs-pool) or of a fixed blueprint (``villain_snapshot``, head-to-head). The deck
    seed is a pure function of ``(seed, deck_index)`` so two arms at the same ``seed`` are CRN-paired
    deck-for-deck -- the caller pairs the returned arrays with :func:`paired_delta`."""
    from pokerbot.eval.exploitation_battery import run_field_absolute_rotated

    if (villain_name is None) == (villain_snapshot is None):
        raise ValueError("pass exactly one of villain_name / villain_snapshot")

    def hero_factory(*, exploit: bool):  # exploit is ignored: we rank the RAW table (OFF arm only)
        return TabularBlueprintAgent(hero_snapshot)

    if villain_snapshot is not None:
        spec_name = "__snapshot__"

        def bot_builder(name, rng=None, params=None):
            return TabularBlueprintAgent(villain_snapshot, rng)
    else:
        spec_name = villain_name
        bot_builder = _combined_bot_builder

    field = [(spec_name, None)] * (n_seats - 1)
    res = run_field_absolute_rotated(
        field, decks=decks, seed=seed, hero_factory=hero_factory,
        allin_ev=True, learn=False, n_seats=n_seats, button_seat=button_seat,
        bot_builder=bot_builder, off_only=True,
        progress_every=progress_every, progress_cb=progress_cb,
    )
    return np.asarray(res["off_deck_seat_means"], dtype=np.float64)


# --------------------------------------------------------------------------------------------------
# The two ranking signals
# --------------------------------------------------------------------------------------------------
def rank_head_to_head(snap_a, snap_b, *, decks: int = 400, seed: int = 0,
                      n_seats: int = NUM_PLAYERS, button_seat: int = 3,
                      coverage_floor: float | None = None, progress_cb=None) -> PairedResult:
    """Head-to-head: A as hero vs a homogeneous field of B, and B as hero vs a field of A, on the SAME
    decks (CRN). Returns the paired delta = (B-vs-field-of-A) - (A-vs-field-of-B); ``b_stronger`` =>
    B extracts more surrounded by A than A does surrounded by B. CAVEAT: this confounds hero strength with
    how EXPLOITABLE the two tables are AS OPPONENTS (A faces B-bots, B faces A-bots), so a positive delta
    can reflect "A is a softer opponent" as much as "B is a better hero" -- which is exactly why
    :func:`decide` gates on this AND the fixed-common-field :func:`rank_vs_pool` agreeing (6-max is
    non-transitive). ``coverage_floor`` (e.g. 0.5) fails loud if either snapshot serves
    uniform (abstraction mismatch); ``None`` (default) skips the guard (used by the {}-vs-{} self-tests)."""
    _assert_serve_coverage(snap_a, "A", floor=coverage_floor)
    _assert_serve_coverage(snap_b, "B", floor=coverage_floor)
    a_arm = _rotated_seat_means(snap_a, villain_snapshot=snap_b, decks=decks, seed=seed,
                                n_seats=n_seats, button_seat=button_seat,
                                progress_cb=_tag_cb(progress_cb, "h2h:A-vs-fieldB"))
    b_arm = _rotated_seat_means(snap_b, villain_snapshot=snap_a, decks=decks, seed=seed,
                                n_seats=n_seats, button_seat=button_seat,
                                progress_cb=_tag_cb(progress_cb, "h2h:B-vs-fieldA"))
    return paired_delta(a_arm, b_arm, detail={"signal": "head_to_head", "seed": seed, "decks": decks})


def rank_vs_pool(snap_a, snap_b, *, roster: list[tuple[str, float]] | None = None,
                 decks: int = 400, seed: int = 0, n_seats: int = NUM_PLAYERS, button_seat: int = 3,
                 coverage_floor: float | None = None, progress_cb=None) -> tuple[PairedResult, dict]:
    """Vs-fixed-pool: A and B each measured vs a FIXED realistic CoinPoker-mirror field, on the SAME
    decks (CRN), reported as a population-weighted paired (B-A) bb/100. Returns ``(combined, per_arch)``
    where ``per_arch[name] = (PairedResult, weight)`` is the per-archetype breakdown.

    Each archetype is a HOMOGENEOUS sub-field (5 copies); the combined number is the per-deck
    weight-sum over archetypes (weights from ``default_pool_roster``), so the paired CI correctly
    accounts for the cross-archetype deck correlation (all archetypes share the same deck seeds).
    ``coverage_floor`` (e.g. 0.5) fails loud if either snapshot serves uniform (abstraction mismatch)."""
    _assert_serve_coverage(snap_a, "A", floor=coverage_floor)
    _assert_serve_coverage(snap_b, "B", floor=coverage_floor)
    roster = roster or default_pool_roster()
    wsum = sum(w for _n, w in roster)
    if wsum <= 0.0:
        raise ValueError("roster weights must sum to > 0")
    roster = [(n, w / wsum) for n, w in roster]  # normalize defensively

    per_arch: dict[str, tuple[PairedResult, float]] = {}
    comb_a = np.zeros(decks, dtype=np.float64)
    comb_b = np.zeros(decks, dtype=np.float64)
    for name, w in roster:
        a_arm = _rotated_seat_means(snap_a, villain_name=name, decks=decks, seed=seed,
                                    n_seats=n_seats, button_seat=button_seat,
                                    progress_cb=_tag_cb(progress_cb, f"pool:A-vs-{name}"))
        b_arm = _rotated_seat_means(snap_b, villain_name=name, decks=decks, seed=seed,
                                    n_seats=n_seats, button_seat=button_seat,
                                    progress_cb=_tag_cb(progress_cb, f"pool:B-vs-{name}"))
        per_arch[name] = (paired_delta(a_arm, b_arm, detail={"archetype": name, "weight": w}), w)
        comb_a += w * a_arm
        comb_b += w * b_arm

    combined = paired_delta(comb_a, comb_b,
                            detail={"signal": "vs_pool", "seed": seed, "decks": decks,
                                    "roster": [(n, round(w, 4)) for n, w in roster]})
    return combined, per_arch


# --------------------------------------------------------------------------------------------------
# The gate
# --------------------------------------------------------------------------------------------------
def decide(h2h: PairedResult, pool: PairedResult, *, label_a: str = "A", label_b: str = "B") -> tuple[str, dict]:
    """Combine the two signals into a verdict. A change is accepted ONLY when BOTH the head-to-head and
    the vs-pool 95% CI exclude 0 in the SAME direction (the research's gate-on-both rule). When the two
    signals exclude 0 in OPPOSITE directions, that is the 6-max non-transitivity flag, surfaced loudly."""
    meta = {
        "head_to_head": h2h.summary(),
        "vs_pool": pool.summary(),
        "h2h_excludes_zero": h2h.excludes_zero,
        "pool_excludes_zero": pool.excludes_zero,
    }
    if h2h.b_stronger and pool.b_stronger:
        return f"{label_b} STRONGER (both signals, 95% CI excludes 0)", meta
    if h2h.a_stronger and pool.a_stronger:
        return f"{label_a} STRONGER (both signals, 95% CI excludes 0)", meta
    if h2h.excludes_zero and pool.excludes_zero and (h2h.b_stronger != pool.b_stronger):
        return ("DISAGREEMENT (non-transitivity): head-to-head and vs-pool point OPPOSITE ways -- "
                "neither is unambiguously stronger; do not ship on one signal", meta)
    return ("INCONCLUSIVE (>= 1 signal's 95% CI includes 0) -- run more decks or the change is too small "
            "to distinguish", meta)


# --------------------------------------------------------------------------------------------------
def _tag_cb(progress_cb, tag: str):
    """Wrap a ``(done, total)`` progress callback to also carry which arm is running (for the driver's
    live log). ``None`` stays ``None`` (no-op fast path)."""
    if progress_cb is None:
        return None

    def _cb(done, total):
        progress_cb(done, total, tag)

    return _cb
