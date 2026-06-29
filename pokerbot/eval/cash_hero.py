"""C4 — ``cash_hero``: the DEPLOYED-stack offline hero that actually FIRES the HUD value-sizing overlay.

The deployed cash edge is value-EXTRACTION (thin value-betting / the HUD value-SIZING + bet-frequency
overlays) on top of a measured-strong GTO base, WITHOUT becoming counter-exploitable by a pro. The
single missing offline seam (the "G3" gap in docs/cash_eval_design.md) is that ``apply_live_value_sizing``
(root_mixer.py:594 — the headline HUD value-SIZING lever) is wired ONLY into ``live_advisor_bridge.py``;
the offline ``SearchAgent`` serve never fires it, so the headline CV cash run measured the GTO BASE, not
the exploitation uplift the cash thesis rests on. This module is the one place that fires the overlay
offline: it wraps the production ``SearchAgent`` (Pluribus search on the deployed tabular blueprint) and,
on the post-search policy, composes ``apply_live_value_sizing`` exactly as the live bridge does
(``search -> apply_live_value_sizing -> sample``). It is NOTHING but the hero M1/M2/M4/C5 serve.

WHY a subclass of ``SearchAgent`` (vs the monkeypatch in scripts/eval/sizing_channel_ab.py): the
sampling + AIVAT decision-capture + leaf-Q baseline all live INSIDE ``SearchAgent.__call__`` after the
policy is computed, so the overlay MUST run before that sample. ``SearchAgent`` already exposes a
post-search policy hook — ``_apply_root_mixer`` (search_agent.py:300), invoked at line 258 when
``root_exploit_mixer`` is on and profiles are set. ``_CashSearchAgent`` enables that hook and overrides
it to apply ONLY ``apply_live_value_sizing`` (NOT ``apply_root_exploit_mixer`` — the spec's
``root_exploit_mixer=False`` semantics: the root-mixer frequency tilt is OUT; the value-SIZING overlay
is the deployed cash lever). The solver opponent-tail (``set_opponent_profiles`` -> ``solver.opponent_model``,
search_agent.py:284) is wired identically to the live path. All sampling/capture is reused verbatim, so
the CRN reseed contract and AIVAT capture are byte-unchanged from ProductionHero.

CLEAN ON/OFF A/B via a SINGLE boolean (``make_cash_hero(..., exploit=...)``):
  * ``exploit=False`` -> ``set_opponent_profiles(None)`` -> the solver tail is inert AND the overlay hook
    sees no profiles -> bare ``SearchAgent`` = the measured-strong GTO base (the OFF arm).
  * ``exploit=True`` -> ``set_opponent_profiles(profiles)`` (solver tail) AND, on the post-search probs,
    ``apply_live_value_sizing(state, seat, probs, profiles, stability_tracker=..., out=capture)``.

The overlay is itself gated (the SAME firewalls the live bridge uses): it only fires vs a confirmed
all-station, non-lag table, and its WSD value-cut guard fails CLOSED without showdown-weakness evidence
(value_sizing.py:241). So a CALLING-STATION villain must have its server HUD (vpip/pfr/wtsd/wsd/
foldToCbet) seeded onto ``OpponentStats.hud`` for the overlay to express — exactly the live capture path,
mirroring exploitation_battery._POP_HUD. ``set_profile`` seeds that HUD automatically when a bare
(HUD-less) station profile is passed; an already-HUD'd profile is left untouched.

Non-degrading tier: this hero NEVER applies ``skill_tiers.apply_leaks`` — it plays the full deployed
strength so difficulty never confounds the measured edge.

The per-decision overlay ``out`` capture is drained into a running mean ``bet_shift`` (the M6 low-variance
cap-binding proxy: the change in the hero's expected bet SIZE in pot fractions from the pre-overlay search
base to the post-overlay mix). Purely observational — it never alters the served policy.

Serve-skew safety (the #1 cash correctness risk): this hero serves the blueprint, so it is built via the
SAME ``_load_blueprint_snapshot`` path whose STRICT stamp check (bots.py:678) is a second backstop; the
C6 ``assert_cash_serve`` gate runs FIRST in any driver. ``make_cash_hero`` ALSO calls the serve-fidelity
guard itself (``check_blueprint_serve_env(expected=snap['abstraction'], strict=True)``) when the passed
snapshot carries a stamp, so a fingerprint MISMATCH FAILS CLOSED here too — a mismatch must never silently
mis-measure (that was the -559 bb/100 "config artifact").

Firewall: NEW file. No edits to jax_engine.py/np_engine.py/game.py/core.rs/battery_native; NUM_PLAYERS=6
untouched; no AoF dirs. Reuse-only: SearchAgent, apply_live_value_sizing, value_sizing.*, rnr_cap
StabilityTracker, OpponentStats, the exploitation_battery population HUD.
"""

from __future__ import annotations

import random

import numpy as np

from pokerbot.engine.game import GameState
from pokerbot.eval.search_agent import SearchAgent

N_SEATS = 6

# Per-vocab commitment proxy (matches exploitation_battery._COMMIT / production_hero._COMMIT) — the
# checkpoint-free fallback AIVAT baseline when the solver does not provide a leaf-Q row.
_COMMIT = np.array([0.0, 0.0, 0.30, 0.50, 0.55, 0.60, 0.65, 0.75, 0.85, 0.95, 1.0], dtype=np.float64)


def _crude_baseline(state: GameState, seat: int) -> np.ndarray:
    """Cheap hand-strength x commitment baseline (mirrors production_hero._crude_baseline)."""
    from pokerbot.agent import _hero_hand_strength

    hs = _hero_hand_strength(state, seat)
    hs = 0.5 if hs is None else float(np.clip(hs, 0.0, 1.0))
    return (hs - 0.5) * 2.0 * _COMMIT


class _CashSearchAgent(SearchAgent):
    """``SearchAgent`` whose post-search policy hook fires the LIVE HUD value-SIZING overlay (instead of
    the root-mixer frequency tilt). Everything else — solve, sampling, AIVAT decision-capture, the leaf-Q
    baseline — is inherited verbatim, so the CRN/AIVAT contracts are byte-identical to the base class.

    ``root_exploit_mixer`` is forced True on this subclass so ``SearchAgent.__call__`` reaches the hook
    (search_agent.py:258); the hook itself (``_apply_root_mixer``) is overridden to apply ONLY the
    value-sizing overlay — there is NO ``apply_root_exploit_mixer`` call, matching the spec's
    ``root_exploit_mixer=False`` semantics (the deployed cash lever is the value-SIZE overlay)."""

    def __init__(self, *args, stability_tracker=None, **kwargs) -> None:
        kwargs["root_exploit_mixer"] = True   # reach the post-search hook; the hook is overridden below
        super().__init__(*args, **kwargs)
        self._stability_tracker = stability_tracker
        # Per-decision overlay diagnostics (drained by the wrapper into the running bet_shift mean). The
        # capture is observational-only: apply_live_value_sizing(out=...) never alters the returned policy.
        self._overlay_caps: list[dict] = []

    def _apply_root_mixer(self, state: GameState, seat: int, probs: np.ndarray) -> np.ndarray:
        """Post-search policy hook (overrides SearchAgent._apply_root_mixer): apply the LIVE HUD
        value-SIZING overlay on the search base, mask + renormalize exactly as live_advisor_bridge does
        (live_advisor_bridge.py:663). Capture the per-decision overlay diagnostics for the bet_shift
        proxy. Returns ``probs`` unchanged when the overlay does not fire (gated to a confirmed station,
        non-lag table; value-cut guard fails closed without WSD evidence)."""
        from pokerbot.opponent.root_mixer import apply_live_value_sizing
        from pokerbot.training.self_play import _legal_mask

        cap: dict = {}
        mixed = apply_live_value_sizing(
            state, seat, probs, self._opponent_profiles,
            stability_tracker=self._stability_tracker, out=cap)
        try:
            self._overlay_caps.append(dict(cap))
        except Exception:  # noqa: BLE001 — diagnostics must never break the serve
            pass
        legal = np.asarray(_legal_mask(state, seat), dtype=bool)
        mixed = np.asarray(mixed, dtype=np.float64) * legal
        s = float(mixed.sum())
        if s > 1e-12:
            return mixed / s
        return probs   # degenerate (overlay zeroed the legal mass) -> keep the un-overlaid search base

    def drain_overlay_caps(self) -> list[dict]:
        d, self._overlay_caps = self._overlay_caps, []
        return d


class CashHero:
    """The deployed cash stack as a battery/CV/rotated hero.

    Protocol the rotated + CV estimators rely on (see exploitation_battery._reseed,
    run_field_absolute_rotated): ``__call__(state) -> Action``; ``hero_seat`` attr SET BEFORE
    ``on_new_hand()`` (a search hero plays seat-0's cards from every seat otherwise,
    exploitation_battery.py:1014); ``set_profile(stats)``; ``reseed(seed)``; ``on_new_hand()``;
    ``pop_decisions()`` + ``pop_chance_corrections()`` + ``bet_freq`` (AIVAT/battery surface);
    ``bet_shift`` (the M6 low-variance proxy = mean overlay size-shift, exposed via ``mean_bet_shift``).

    The single ``exploit`` boolean gives a CRN-clean ON/OFF A/B: OFF == bare SearchAgent (profiles None),
    ON == solver tail + the value-SIZE overlay. Both arms share the per-hand seed, so a decision the
    exploit does not change is sampled identically.
    """

    def __init__(self, agent: _CashSearchAgent, rng: random.Random, *, exploit: bool,
                 hero_seat: int = 0, n_seats: int = N_SEATS, value_fn=None,
                 seed_station_hud: bool = True) -> None:
        self.agent = agent
        self.rng = rng
        self.exploit = exploit
        self.hero_seat = hero_seat
        self.n_seats = n_seats
        self.value_fn = value_fn
        # When True (default), set_profile seeds a representative station server-HUD onto a HUD-less
        # station profile so the value-cut-guarded overlay can FIRE (it fails closed without WSD
        # evidence). A profile that already carries a HUD is never overwritten.
        self.seed_station_hud = seed_station_hud
        self._profile = None
        self._dec: list[tuple] = []
        self.bet_mass = 0.0
        self.n_dec = 0
        # Running bet_shift accumulator (M6): mean |overlay size-shift in pot fractions| over the
        # decisions where the overlay FIRED since construction. Drained per __call__ from the agent.
        self._shift_sum = 0.0
        self._shift_n = 0
        self.n_overlay_fires = 0          # count of decisions where the overlay actually moved the policy
        self.last_overlay_caps: list[dict] = []   # per-decision overlay diagnostics from the last __call__
        # Seeded sampling => AIVAT decision correction is zero-mean and CRN-shared across arms.
        if hasattr(agent, "enable_decision_capture"):
            agent.enable_decision_capture(rng)

    # ---- battery / AIVAT surface ------------------------------------------------------------------
    @property
    def bet_freq(self) -> float:
        return self.bet_mass / self.n_dec if self.n_dec else 0.0

    @property
    def mean_bet_shift(self) -> float:
        """M6 low-variance proxy: mean |overlay size-shift (pot fractions)| over FIRED decisions; 0.0 if
        the overlay never fired (the OFF arm, or a table where it correctly refused to escalate)."""
        return (self._shift_sum / self._shift_n) if self._shift_n else 0.0

    # alias kept for callers that probe ``bet_shift`` (the design's M6 column name)
    @property
    def bet_shift(self) -> float:
        return self.mean_bet_shift

    def reseed(self, seed: int) -> None:
        """CRN per-hand reseed: both arms share the seed so an unchanged decision plays identically."""
        self.rng = random.Random(seed)

    def _seed_station_hud_if_needed(self, profile) -> None:
        """Mirror exploitation_battery._POP_HUD seeding: a confirmed-station profile with no server HUD
        gets a representative calling-station HUD so station_payoff_score has the WSD evidence the
        value-cut guard requires (value_sizing.py:241 fails closed without it). Only touches a bare
        single OpponentStats; per-seat dicts and already-HUD'd profiles are left as-is."""
        if not self.seed_station_hud or profile is None or isinstance(profile, dict):
            return
        if getattr(profile, "hud", None):
            return
        try:
            from pokerbot.eval.exploitation_battery import population_hud

            hud = population_hud("calling_station")
            if hud:
                profile.hud = hud
        except Exception:  # noqa: BLE001 — seeding is best-effort; absence => overlay simply won't fire
            pass

    def _apply_profile(self) -> None:
        """Wire the profile into the solver tail + the overlay hook. The OFF arm is the PURE base:
        ``set_opponent_profiles(None)`` so the solver tail is inert AND the overlay hook sees no
        profiles (returns the search base unchanged). ON broadcasts a single OpponentStats to every
        seat (the homogeneous-field default) or passes a per-seat {seat: stats} map verbatim."""
        if self._profile is None or not self.exploit:
            prof = None
        elif isinstance(self._profile, dict):
            prof = self._profile               # per-seat map (live / het field) passed through verbatim
        else:
            prof = {i: self._profile for i in range(self.n_seats)}
        self.agent.set_opponent_profiles(prof)

    def set_profile(self, profile) -> None:
        self._seed_station_hud_if_needed(profile)
        self._profile = profile
        self._apply_profile()

    def on_new_hand(self) -> None:
        # hero_seat MUST already be set by the caller's _reseed (a search hero plays seat-0's cards from
        # every physical seat otherwise). reset_hand rebuilds solver/tracker state -> re-apply profile +
        # re-arm capture, exactly like ProductionHero.on_new_hand.
        if hasattr(self.agent, "reset_hand"):
            self.agent.reset_hand(self.hero_seat)
        self._apply_profile()
        if hasattr(self.agent, "enable_decision_capture"):
            self.agent.enable_decision_capture(self.rng)

    def pop_decisions(self) -> list[tuple]:
        d, self._dec = self._dec, []
        return d

    def pop_chance_corrections(self) -> list[float]:
        return []

    def __call__(self, state: GameState):
        from pokerbot.training.self_play import _legal_mask

        seat = state.action_seat
        action = self.agent(state)
        # Drain the overlay diagnostics this decision produced (the value-SIZE overlay hook appends one
        # cap per hero decision when profiles are set; OFF arm produces none). Accumulate the FIRED
        # size-shift into the running M6 proxy and expose the caps for introspection.
        caps = self.agent.drain_overlay_caps()
        self.last_overlay_caps = caps
        for cap in caps:
            if cap.get("fired"):
                self._shift_sum += abs(float(cap.get("bet_shift", 0.0) or 0.0))
                self._shift_n += 1
                self.n_overlay_fires += 1
        # AIVAT / bet-freq decision capture (mirrors ProductionHero.__call__).
        raw = self.agent.pop_decisions_raw() if hasattr(self.agent, "pop_decisions_raw") else []
        for dec in raw:
            if len(dec) == 3 and dec[1] is not None:
                probs, bv, idx = dec
                bv = np.asarray(bv, dtype=np.float64).reshape(-1)
            else:
                probs, idx = (dec[0], dec[-1])
                if self.value_fn is not None:
                    legal = np.asarray(_legal_mask(state, seat), dtype=bool)
                    bv = np.asarray(self.value_fn(state, seat, legal), dtype=np.float64).reshape(-1)
                else:
                    bv = _crude_baseline(state, seat)
            probs = np.asarray(probs, dtype=np.float64).reshape(-1)
            self._dec.append((probs.copy(), bv, int(idx)))
            self.bet_mass += float(probs[3:].sum())
            self.n_dec += 1
        return action


def make_cash_hero(snap, exploit: bool, *, stability_tracker=None, rng=None,
                   hero_seat: int = 0, n_seats: int = N_SEATS, time_budget_ms: int = 2000,
                   iterations: int | None = None, pluribus_leaf_select: str = "uniform",
                   search_preflop: bool = True, aivat_leaf_q: bool = False,
                   seed_station_hud: bool = True, seed: int = 0,
                   verify_serve: bool = True) -> CashHero:
    """Build ONE deployed-cash hero arm.

    ``snap`` is EITHER the raw blueprint strategy table (``{key: probs}``) OR the full stamped snapshot
    dict (``{'strat_sum': ..., 'abstraction': {...}}``) as loaded by ``bots._load_blueprint_snapshot``.
    When the full stamped dict is given AND ``verify_serve`` is True, the serve-fidelity guard runs HERE
    (``check_blueprint_serve_env(expected=snap['abstraction'], strict=True)``) so a fingerprint MISMATCH
    FAILS CLOSED — a mismatch must never silently mis-measure (the -559 bb/100 artifact). Drivers should
    still call the C6 ``assert_cash_serve`` gate FIRST; this is the second backstop.

    ``exploit`` is the SINGLE ON/OFF boolean: False => bare SearchAgent (the measured-strong base);
    True => solver opponent-tail + the LIVE HUD value-SIZING overlay on the post-search policy.

    ``iterations`` (default None) fixes the CFR iteration count so each solve is a pure function of
    (public state, opponent model, config) — required for ON/OFF CRN determinism in a fixed-budget A/B
    (a wall-clock budget makes the iteration count load-dependent and breaks reproducibility). When set,
    the time budget is disabled.

    ``aivat_leaf_q`` defaults False: AIVAT gives NEGATIVE variance reduction on the search path, so the
    rigorous leaf-Q baseline is never needed for the rotated/CV cash estimators (and it is ~44% of the
    pluribus wall time). The cheap crude baseline is used for the secondary per-arm columns; the headline
    on-off uplift uses raw chips and is byte-unchanged either way.
    """
    bp = _resolve_snapshot(snap, verify_serve=verify_serve)
    agent = _CashSearchAgent(
        use_pluribus_search=True,
        blueprint_snapshot=bp,
        pluribus_leaf_select=pluribus_leaf_select,
        time_budget_ms=time_budget_ms,
        transformer_checkpoint=None,        # no MADT belief net on the eval/CV path (pure blueprint serve)
        value_checkpoint=None,
        hero_seat=hero_seat,
        search_preflop=search_preflop,      # the deployed serve eliminates the preflop LBR hole
        aivat_leaf_q=aivat_leaf_q,
        stability_tracker=stability_tracker,
    )
    if iterations:
        # FIXED CFR iterations => deterministic solves => CRN-clean ON/OFF arms.
        agent.solver.config.max_iterations = iterations
        agent.solver.config.time_budget_ms = 10 ** 9
    return CashHero(agent, random.Random(seed), exploit=exploit, hero_seat=hero_seat,
                    n_seats=n_seats, seed_station_hud=seed_station_hud)


def _resolve_snapshot(snap, *, verify_serve: bool):
    """Return the raw strategy table from ``snap`` (raw table or full stamped dict), running the
    serve-fidelity guard against the stamp when present. A stamped snapshot served under the wrong
    encoder flags misses every affected node -> ~0% coverage -> degenerate play -> garbage measurement;
    the strict check raises so that CANNOT happen silently here."""
    if isinstance(snap, dict) and "strat_sum" in snap:
        stamp = snap.get("abstraction")
        if verify_serve:
            from pokerbot.eval.lbr import check_blueprint_serve_env

            # strict iff the snapshot is stamped (the lossless era); an unstamped legacy table warns.
            check_blueprint_serve_env(expected=stamp, strict=stamp is not None)
        return snap["strat_sum"]
    return snap


def hero_factory(snap, *, stability_tracker=None, **kwargs):
    """Return a ``factory(exploit: bool) -> CashHero`` closure satisfying the rotated/CV reseed contract
    (run_field_absolute_rotated / het_cv_absolute call ``hero_factory(exploit=...)`` to build each arm).

    The snapshot is resolved + serve-verified ONCE here (so the strict stamp check runs a single time
    and every arm shares the identical raw table — no per-arm re-load skew); each arm then gets its own
    fresh agent. ``stability_tracker`` (a single rnr_cap.StabilityTracker, or a per-villain-stability
    feed) is shared across arms so a caught counter-adapter clamps both."""
    bp = _resolve_snapshot(snap, verify_serve=kwargs.pop("verify_serve", True))

    def factory(*, exploit: bool) -> CashHero:
        # verify_serve already done on the shared table -> skip the per-arm re-check (bp is the raw table).
        return make_cash_hero(bp, exploit, stability_tracker=stability_tracker,
                              verify_serve=False, **kwargs)

    return factory
