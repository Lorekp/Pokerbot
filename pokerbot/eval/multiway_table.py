"""6-max multiway table evaluation — bb/100 with 95% CIs, seat rotation, duplicate dealing.

``eval/head_to_head.py`` is effectively HU (one opponent policy fills every non-hero seat). For the
Pluribus 6-max foundation we need hero vs a FIELD of (possibly distinct) pool bots at a full ring, with
the variance reduction the literature mandates (AIVAT is added separately; this provides the duplicate /
seat-rotation control). Exact exploitability is intractable for 6-max (LBR is HU-only) so win-rate ±CI is
the metric, exactly as Pluribus reported.

Variance reduction:
  * **Seat rotation** — hero plays every seat, cancelling positional EV asymmetry (the BTN earns, the SB
    bleeds; without rotation a bot's bb/100 is dominated by which seat it sat in).
  * **Duplicate dealing** — ``engine.game.play_hand``'s ``rng`` only drives the deck (bots carry their own
    rng), so the SAME seed deals the SAME cards. We replay each deck once per hero seat with the field
    otherwise fixed, so hero faces every position on identical cards — the standard duplicate-poker control.

Report ``bb_per_100`` with a 95% CI (``1.96 * stderr``). NEVER a bare point estimate (the lesson from the
earlier LBR over-claim): a noisy +X bb/100 with a CI that straddles 0 is not a win.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

import numpy as np

from pokerbot.engine.game import NUM_PLAYERS, play_hand
from pokerbot.eval.bots import BOT_REGISTRY
from pokerbot.eval.aivat import baseline_from_leaf_net, decision_correction
from pokerbot.eval.chance_term import chance_corrections_for_hand


@dataclass
class MultiwayResult:
    hands: int
    mean_bb: float
    stderr_bb: float
    bb_per_100: float
    ci95_bb_per_100: float
    detail: dict = field(default_factory=dict)

    @property
    def beats_zero(self) -> bool:
        """True iff the 95% CI for bb/100 is entirely above 0 (a statistically real win)."""
        return (self.bb_per_100 - self.ci95_bb_per_100) > 0.0

    def summary(self) -> str:
        return f"{self.bb_per_100:+.2f} ± {self.ci95_bb_per_100:.2f} bb/100  (n={self.hands})"


def _reset_if_supported(policy, hero_seat: int):
    """SearchAgent-style stateful heroes need a per-hand reset; stateless bots/blueprint don't."""
    fn = getattr(policy, "reset_hand", None)
    if callable(fn):
        fn(hero_seat)


def _default_field(seed: int) -> list:
    """A representative 6-seat field spanning the archetypes (cycled to fill the ring)."""
    names = ["calling_station", "lag", "nit", "tag", "random", "human_like"]
    return [BOT_REGISTRY[names[i % len(names)]](random.Random(seed + 101 * i)) for i in range(NUM_PLAYERS)]


def _hero_aivat_correction(hero_policy) -> float:
    """Sum the hero's per-decision AIVAT control variate for the just-played hand (0 if unsupported).

    Uses the SearchAgent decision capture: each entry is ``(probs, leaf_q_row, taken_idx)`` where the
    leaf_q_row is the continuation-gadget baseline (NOT a poisoned net). Zero-mean over the hero's own
    action randomization, so subtracting it is unbiased and lower-variance. Board-reveal (chance) terms
    are a documented follow-up — the decision term alone is already a valid unbiased reduction.
    """
    pop = getattr(hero_policy, "pop_decisions_raw", None)
    if not callable(pop):
        return 0.0
    total = 0.0
    for probs, bv, idx in pop():
        if bv is None:
            continue
        total += decision_correction(probs, baseline_from_leaf_net(bv), int(idx))
    return total


def _hero_chance_correction(hero_policy, final_state, hero_seat, *, include_flop, flop_samples, seed) -> float:
    """Sum the hero's per-reveal AIVAT chance (board-luck) control variate for the just-played hand.

    Sourced from the hero's OWN continuation-gadget value function (``solver.adapter`` + ``solver.config
    .leaf``) — never a value net — so the gadget config is identical to the search leaf. Zero-mean over the
    board deal => unbiased. 0 for heroes without a gadget solver (plain bots: no chance term needed/possible).
    """
    solver = getattr(hero_policy, "solver", None)
    adapter = getattr(solver, "adapter", None)
    leaf_cfg = getattr(getattr(solver, "config", None), "leaf", None)
    if adapter is None or leaf_cfg is None:
        return 0.0
    cc = chance_corrections_for_hand(
        final_state, int(hero_seat), adapter, leaf_cfg,
        include_flop=include_flop, flop_samples=flop_samples, base_seed=seed,
    )
    return float(sum(cc))


def run_multiway(
    hero_policy,
    *,
    field_bots: list | None = None,
    hands: int = 600,
    seed: int = 42,
    duplicate: bool = True,
    aivat: bool = False,
    aivat_chance: bool = True,
    aivat_flop: bool = False,
    aivat_flop_samples: int = 24,
) -> MultiwayResult:
    """Hero vs a 6-max field; returns bb/100 ± 95% CI over ``hands`` hero-hands.

    ``field_bots`` is a length-``NUM_PLAYERS`` list of bot callables assigned to absolute seats; hero
    REPLACES whichever seat it occupies. Default = a mixed archetype field. With ``duplicate=True`` each
    deck is played once per hero seat (hero-hands = groups * NUM_PLAYERS), so ``hands`` is rounded up to a
    multiple of NUM_PLAYERS.

    ``aivat=True`` (hero must support ``enable_decision_capture``/``pop_decisions_raw``, e.g. SearchAgent)
    applies the AIVAT decision control variate (gadget-baseline, ~44x fewer hands target) and reports
    bb/100±CI from the VARIANCE-REDUCED series; the raw series + variance_reduction are in ``detail``.

    ``aivat_chance=True`` (default; needs a hero with a continuation-gadget ``solver.adapter``) ALSO subtracts
    the AIVAT chance (board-reveal) control variate — the larger half of AIVAT's reduction. Turn+river reveals
    are enumerated exactly (provably unbiased); the flop term is sampled and applied only with
    ``aivat_flop=True`` (``aivat_flop_samples`` flops). ``detail`` reports both the combined
    ``variance_reduction`` and ``variance_reduction_decision_only`` so the chance term's lift is isolatable.
    """
    field_bots = field_bots or _default_field(seed)
    if len(field_bots) != NUM_PLAYERS:
        raise ValueError(f"field_bots must have {NUM_PLAYERS} entries, got {len(field_bots)}")

    use_aivat = aivat and callable(getattr(hero_policy, "enable_decision_capture", None))
    if use_aivat:
        # AIVAT's decision CV is zero-mean over the hero's OWN randomization => hero must SAMPLE, not argmax.
        hero_policy.enable_decision_capture(rng=random.Random(seed ^ 0xA1A7))
    # Chance (board-reveal) CV: applied whenever AIVAT is on AND the hero has a gadget value function.
    use_chance = use_aivat and aivat_chance and getattr(getattr(hero_policy, "solver", None), "adapter", None) is not None

    raw: list[float] = []
    adj: list[float] = []
    dec_sum: list[float] = []   # decision-term corrections per hand (for decision-only vs +chance reporting)
    chn_sum: list[float] = []   # chance-term corrections per hand
    seat_sums = np.zeros(NUM_PLAYERS)
    seat_counts = np.zeros(NUM_PLAYERS)

    def _play(hero_seat: int, g_or_h: int) -> None:
        rng = random.Random(seed + g_or_h)
        policies = {s: field_bots[s] for s in range(NUM_PLAYERS)}
        policies[hero_seat] = hero_policy
        _reset_if_supported(hero_policy, hero_seat)
        final_state, d = play_hand(policies, rng=rng)
        v = float(d.get(hero_seat, 0.0))
        c_dec = _hero_aivat_correction(hero_policy) if use_aivat else 0.0
        c_chn = (
            _hero_chance_correction(
                hero_policy, final_state, hero_seat,
                include_flop=aivat_flop, flop_samples=aivat_flop_samples,
                seed=seed + 7 * g_or_h + hero_seat,
            )
            if use_chance else 0.0
        )
        raw.append(v)
        adj.append(v - c_dec - c_chn)
        dec_sum.append(c_dec)
        chn_sum.append(c_chn)
        seat_sums[hero_seat] += v
        seat_counts[hero_seat] += 1

    if duplicate:
        groups = max(1, math.ceil(hands / NUM_PLAYERS))
        for g in range(groups):
            for hero_seat in range(NUM_PLAYERS):
                _play(hero_seat, g)  # SAME deck (seed+g) across the NUM_PLAYERS rotations of this group
    else:
        for h in range(hands):
            _play(h % NUM_PLAYERS, h)

    raw_arr = np.asarray(raw, dtype=np.float64)
    adj_arr = np.asarray(adj, dtype=np.float64)
    n = raw_arr.shape[0]
    # Primary estimate uses the variance-reduced series when AIVAT is on (unbiased; tighter CI).
    primary = adj_arr if use_aivat else raw_arr
    mean = float(primary.mean()) if n else 0.0
    stderr = float(primary.std(ddof=1) / math.sqrt(n)) if n > 1 else 0.0
    per_seat = {
        s: float(seat_sums[s] / seat_counts[s]) * 100 if seat_counts[s] else 0.0
        for s in range(NUM_PLAYERS)
    }
    detail = {"duplicate": duplicate, "bb_per_100_by_hero_seat": per_seat, "seed": seed, "aivat": use_aivat}
    if use_aivat and n > 1:
        rstd = float(raw_arr.std(ddof=1))
        astd = float(adj_arr.std(ddof=1))
        detail["raw_bb_per_100"] = float(raw_arr.mean()) * 100.0
        detail["variance_reduction"] = (1.0 - (astd / rstd) ** 2) if rstd > 0 else 0.0
        detail["aivat_chance"] = use_chance
        # Decision-only adjusted series (subtract ONLY the decision term) to isolate the chance term's lift.
        dec_arr = np.asarray(dec_sum, dtype=np.float64)
        chn_arr = np.asarray(chn_sum, dtype=np.float64)
        adj_dec_only = raw_arr - dec_arr
        dstd = float(adj_dec_only.std(ddof=1))
        detail["variance_reduction_decision_only"] = (1.0 - (dstd / rstd) ** 2) if rstd > 0 else 0.0
        detail["adj_mean_bb_per_100"] = float(adj_arr.mean()) * 100.0  # ~ raw (unbiasedness check)
        detail["mean_chance_correction_bb"] = float(chn_arr.mean())
    return MultiwayResult(
        hands=n,
        mean_bb=mean,
        stderr_bb=stderr,
        bb_per_100=mean * 100.0,
        ci95_bb_per_100=1.96 * stderr * 100.0,
        detail=detail,
    )
