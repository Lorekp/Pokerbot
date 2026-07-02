"""Pluribus-faithful depth-limited search: tabular blueprint + continuation-strategy leaf gadget.

This is the P1 solver that replaces ``TurboReBeLSolver``'s poisoned neural stack (value net +
``AverageStrategyNet`` blueprint) with:
  * a TABULAR blueprint (``tabular_blueprint_adapter``) for the opponent-node prior, and
  * a CONTINUATION-GADGET leaf (``pluribus_rollout``) at the depth limit — NO value net.

Depth limit = the END of the current betting round (Pluribus): ``_cfv`` recurses through the current
street's betting and treats the next chance node (street deal) as the leaf, where the continuation gadget
estimates the value. Preflop is served directly from the blueprint (``search_preflop=False`` default —
Pluribus skips preflop search; the mixer handles any preflop exploitation outside the solver).

It MIRRORS ``TurboReBeLSolver``'s public surface so it is a drop-in via a ``SearchAgent`` flag:
``solve(root, pbs) -> SolveResult``; attributes ``opponent_model`` / ``gift_register``; method ``_leaf_q``.

FIREWALL: the exploitation deviation is applied ONLY at opponent nodes, via the SAME tail lifted verbatim
from ``turbo_rebel_search`` (``apply_context_bias`` / ``apply_opponent_bias`` / ``context_read`` /
``_context_key_from_env`` / ``DEFAULT_MAX_ALPHA`` / ``GiftRegister.capped_max_alpha``). The exploit math,
``context_spec_hash``, the 11->4 action-bin map, and the gift/bounded-eps machinery are byte-untouched. With
``opponent_model=None`` the hero output is independent of any profile (zero-deviation bit-identity).
"""

from __future__ import annotations

import hashlib
import os
import time
from dataclasses import dataclass, field

import numpy as np

from pokerbot.engine.actions import BET_SIZE_BUCKETS, pseudo_harmonic_bucket
from pokerbot.engine.jax_engine import BET_BUCKET_START, CHIP_UNIT, NUM_ACTIONS
from pokerbot.engine.np_engine import (
    to_numpy_state,
    SEAT_FOLDED,
    SEAT_LIVE,
)
# Hot-path engine primitives via the native dispatch layer (Rust core when enabled, else np_engine).
from pokerbot.search.native_solver import (
    apply_action_np,
    is_chance_node_np,
    legal_action_mask_np,
    sample_chance_np,
)
from pokerbot.search.isomorphism import cached_canonicalize_key
from pokerbot.search.pbs_tracker import PBSTracker
from pokerbot.search.state_bridge import SubgameRoot
from pokerbot.search.tabular_blueprint_adapter import TabularBlueprintAdapter
from pokerbot.search.pluribus_rollout import (
    ContinuationLeafConfig,
    _reshuffle_undealt,
    biased_profile_leaf_value,
    continuation_leaf_value,
)
from pokerbot.search.tabular_blueprint_adapter import NUM_CONTINUATIONS

# Exploit tail + helpers lifted VERBATIM from the existing solver (firewall: unchanged math/keys).
from pokerbot.search.turbo_rebel_search import (
    SolveResult,
    _blend_bias,  # noqa: F401  (re-exported for parity / tests)
    _regret_match,
    apply_context_bias,
    apply_opponent_bias,
    _context_key_from_env,
)
from pokerbot.opponent.bayes_exploit import DEFAULT_MAX_ALPHA as _CTX_MAX_ALPHA
from pokerbot.opponent.rnr_cap import cap_eff as _rnr_cap_eff  # RNR restriction of the solver BR (#28)
from pokerbot.opponent.rnr_cap import fish_relaxed_n as _rnr_fish_relaxed_n  # no-punisher wipe-floor relax
from pokerbot.opponent.bayes_exploit import context_read
from pokerbot.opponent.context import action_bin as _action_bin


@dataclass
class PluribusReBeLConfig:
    max_iterations: int = 200
    time_budget_ms: int = 2000
    max_depth: int = 48               # safety cap on betting-round recursion depth (not the leaf limit)
    blueprint_mix: float = 0.5        # opponent node: (1-mix)*regret_match + mix*blueprint
    # STALE-DECK FIX (2026-07-01): _set_world overwrote opponent holes but left env.deck frozen, so an
    # all-in line (which bypasses the reshuffling leaf gadget) replayed ONE fixed board every world/iter
    # -> the search best-responded to a single frozen runout -> catastrophic preflop over-jamming (~0.92
    # all-in on AKs vs a blueprint 0.083). When True, _set_world rebuilds the undealt deck tail per world
    # (deterministic per-world seed => CRN preserved). Genuine correctness fix, but it changes sampled
    # boards everywhere (postflop + exploit path) and the NATIVE core is not yet patched, so it ships
    # default-OFF (byte-identical + Python/native parity) until native is patched + field/rank-validated,
    # then flipped ON.
    repair_deck: bool = False
    # TWO-SIDED SEARCH (2026-07-01): when True the intra-round CFR is a proper multi-player subgame solve
    # -- the traverser cycles across live seats so EVERY player updates regrets (opponents learn to fold to
    # jams), instead of the hero one-sidedly best-responding to a fixed opponent (which degenerates to
    # over-aggression, the Pluribus "fixed-strategy-beyond-leaf is highly exploitable" failure). Applies to
    # GTO/base play only; the EXPLOIT path (opponent_model set) intentionally stays a one-sided BR to the
    # MODELED opponent. Default OFF (byte-identical one-sided) until validated + ported to native.
    two_sided: bool = False
    # DECOUPLE THE OPPONENT-HOLE SAMPLER's RNG (2026-07-01): opponent-hole sampling and in-tree
    # action/rollout sampling currently share ONE per-solve stream. sample_opponent_holes consumes a
    # VARIABLE (CDF-dependent) number of draws per world, so it desyncs the action stream -> higher-
    # variance CFR (the diagnosed cause of the native-vs-Python fixed-point divergence: native's
    # fixed-count Fisher-Yates draws are lower-variance). When True, world/hole sampling runs on its own
    # deterministic sub-stream so the action stream is unperturbed. Default OFF (byte-identical).
    # Threaded into the NATIVE fast path too (native's fixed-count Fisher-Yates is already low-variance in
    # the UNIFORM case, so native's effect is small until range_aware makes the draw count data-dependent).
    decouple_sampler: bool = False
    # Q3 — RANGE-AWARE OPPONENT-HOLE SAMPLING (native lever). When True, the native solve weights each
    # opponent's sampled hole by the blueprint CONTINUATION probability at the solve root (1 - P(FOLD)),
    # mixed with a uniform robustness floor, so the hero best-responds to a REALISTIC (tighter) opponent
    # that would not have folded the action so far -> faster convergence + a more realistic BR. Default OFF
    # => uniform opponent-hole deal (byte-identical). Native-only for now (the Python _cfv loop samples via
    # PBSTracker.sample_opponent_holes, which is uniform in the eval base arm).
    range_aware: bool = False
    # Q3 — UNIFORM ROBUSTNESS FLOOR for range_aware. sample_prob = (1-floor)*range + floor*uniform. Mirrors
    # the RNR restricted-response cap: a PRO who bluffs off-range is still sampled with weight >= floor, so
    # the hero keeps defending off-range/bluff hands. floor too LOW = exploitable by an off-range bluffer;
    # floor=1.0 == range_aware OFF (pure uniform). Only read when range_aware.
    range_uniform_floor: float = 0.25
    # ENSEMBLE PARALLELISM (2026-07-01, native GTO path only). Run this many INDEPENDENT CFR streams in
    # parallel inside the native solve (rayon), each with its OWN regret/strategy tables + its OWN RNG
    # (seed = public_seed ^ (stream_idx*ODD)), then MERGE by summing the per-stream ROOT strategy_sum before
    # extracting the served policy. The idle cores (the sequential native solve uses ~1 of N) pack ~N x more
    # effective search into the SAME wall-clock -> faster convergence at the live budget (variance reduction:
    # averaging N independent CFR averages). No shared mutable state => embarrassingly parallel => ~linear
    # core scaling. `<= 1` => single-stream, byte/stat-identical to the pre-ensemble native binary. Threaded
    # ONLY into the native GTO fast path (opponent_model is None); the exploit path stays single-stream
    # (firewalled follow-up). Default 1 => production unchanged. See tools/battery_native/src/subgame.rs.
    ensemble_streams: int = 1
    # BUG-1 FIX (2026-07-02) — LEAF CACHE HOLE-WORLD INTEGRATION. The leaf value cache (_leaf_value) is
    # keyed by PUBLIC state only, but the env reaching a leaf carries THAT ITERATION's sampled opponent
    # holes — so the value computed under the FIRST visiting hole-world was frozen and reused for all
    # remaining ~K iterations (a single-hole-world draw, not an expectation over the range: an arbitrary
    # per-leaf bias the CFR best-responds to; the frozen-villain-hands sibling of the stale-deck bug).
    # When > 1, the cached leaf is a RUNNING MEAN over the first `leaf_cache_worlds` VISITING worlds:
    # each cache hit while n < M re-evaluates the leaf under the CURRENT iteration's sampled holes (with
    # a deterministic per-(leaf, n) seed — CRN/pure-function contract preserved) and returns the
    # incremental mean; after M samples the mean freezes. Default 1 => first-touch freeze, byte-identical
    # to today (world 0 uses EXACTLY the legacy seed string). Applies to whichever path runs _cfv
    # (GTO and exploit alike — it only changes the leaf's world-integration, never the exploit math);
    # the NATIVE exploit fast path (solve_exploit) instead FORCES the no-op value (firewall: the
    # certified native-exploit serve is byte-identical regardless of this knob). Mirrored in the native
    # GTO solve (subgame.rs leaf_cache_worlds).
    # ADOPTED 2026-07-02: default 4 (full-game H2H, 3 seeds, 3300 decks pooled:
    # +146.3 chips/deck CI95 [+55.9, +236.8] vs the legacy first-touch freeze).
    leaf_cache_worlds: int = 4
    # BUG-5 (partial, 2026-07-02) — DELAYED AVERAGING (DeepStack skips the first 50-98% of iterations
    # from the served average; Supremus weights iteration t by max{0, t-100}). strat_sum accumulation
    # (and the AIVAT root-CFV capture, which must stay CONSISTENT with the served average) begins only
    # at iteration t >= floor(avg_skip_frac * K), where K is the solve's iteration target. Regret updates
    # are NEVER skipped, and the CFR+ plus-clamp is untouched (that is a separate experiment). Default
    # 0.0 => accumulate from t=0, byte-identical to today. Mirrored in the native GTO solve (subgame.rs
    # avg_skip_frac); the NATIVE exploit fast path FORCES 0.0 (firewall).
    avg_skip_frac: float = 0.0
    # R2 (2026-07-02) — PLURIBUS BIASED CONTINUATION LEAVES. When True, every depth-limit leaf becomes
    # a 4-way CONTINUATION-CHOICE node for each player still able to act (SEAT_LIVE): (0) unaltered
    # blueprint, (1) fold-biased x leaf_bias_mult, (2) call-biased, (3) raise-biased — the Pluribus
    # depth-limited construction (Science 2019 supp; Modicum: k=1 pure-blueprint leaves are WORSE than
    # no search at all). The choice is made by the SAME CFR that solves the subgame ("essentially just
    # another action"): the traverser BRANCHES its 4 continuations at the leaf (regret update +
    # strategy-sum, exactly like an action node); every other live seat SAMPLES from its own learned
    # 4-way strategy via a draw that is a PURE FUNCTION of (choice infoset, iteration) — identical
    # across all leaves in the same infoset within an iteration, per Pluribus. BOTH hero and opponents
    # choose (opponent-only choice makes the searcher defensive/lower-EV — documented). Leaf value =
    # playout under the chosen JOINT profile (per-seat bias applied to the blueprint probs before
    # sampling; hero's continuation biases hero's own playout actions too). Default False => the
    # legacy uniform-draw leaf, byte-identical. GTO path only (mirrors two_sided gating:
    # opponent_model must be None); the NATIVE exploit path forces OFF and its FFI does not expose the
    # knob (firewall, defense in depth). Mirrored in the native GTO solve (subgame.rs
    # leaf_biased_continuations). NOTE: applies to the "uniform" leaf-select regime (the deployed
    # live/native leaf); with select="gadget" the Python loop's per-leaf 4x4 maximin matrix is a
    # RELATED but DIFFERENT construction (per-leaf matrix game, opponents collapsed to one field, no
    # cross-leaf learning) — this flag replaces the leaf evaluation entirely when active, superseding
    # the select mode.
    leaf_biased_continuations: bool = False
    # R2 — bias multiplier for the biased continuations (target-group mass x this, renormalized).
    # Pluribus used 5, Modicum 10 (the adapter-level default 10.0 still drives the legacy gadget /
    # uniform leaf tilts — this knob applies ONLY to the biased-continuation leaf). Only read when
    # leaf_biased_continuations is True.
    leaf_bias_mult: float = 5.0
    client_bet_sizing: bool = True    # use the SAME action abstraction as training (blueprint keys hit)
    search_preflop: bool = False      # Pluribus: serve preflop from blueprint, search postflop only
    # A1 stack-agnostic preflop: the preflop blueprint is keyed on (hole, position) only and trained at
    # 100bb, so it is the one engine component that is NOT depth-general (the postflop key + rollout leaf
    # already key on real stacks / SPR). Trust it ONLY when the effective stack is within this band of its
    # training depth; OFF-band (deep/short tables), fall through to the depth-general search below instead
    # of serving an off-depth range. None => always trust (legacy fixed-100bb behaviour). Standard 100bb
    # tables stay on the fast blueprint path bit-for-bit; only genuinely off-depth tables divert to search.
    preflop_blueprint_band_bb: tuple[float, float] | None = (80.0, 130.0)
    # TOURNAMENT SHALLOW-STACK PRIOR (push/fold). At N>6 there is no trained blueprint, so the search
    # root prior is the uniform-over-legal fallback (the fingerprint misses). For a SHALLOW PREFLOP spot
    # — folded-to-hero first-in, or facing exactly one all-in jam — within ``pushfold_prior_band_bb`` of
    # effective stack, serve the CERTIFIED ante-aware ICM open-shove / call-vs-jam Nash range
    # (pokerbot.search.pushfold_prior, reusing pokerbot.tournament.open_shove_icm) as the root prior
    # INSTEAD of uniform. Generalizes 2..7. ICM-aware when ``pushfold_payouts`` is supplied; with None it
    # is winner-take-all (chip-EV push/fold Nash) — still a real position/ante-aware range. None band =>
    # the prior is DISABLED (default OFF => byte-identical to today; nothing changes on the 6-max path or
    # any spot the certified push/fold abstraction does not cover, where the prior abstains and the
    # uniform fallback stands). Set to (0.0, 20.0) to activate the shallow band.
    pushfold_prior_band_bb: tuple[float, float] | None = None
    # Remaining prize structure (best first) for the ICM push/fold prior; None => winner-take-all (chip-EV).
    pushfold_payouts: tuple[float, ...] | None = None
    leaf: ContinuationLeafConfig = field(default_factory=ContinuationLeafConfig)
    # Deterministic per-decision RNG: each solve() seeds its Monte-Carlo stream from a hash of the
    # ROOT'S PUBLIC STATE (and each leaf from its canonical leaf key), so a decision is a pure function
    # of (public state, opponent model, config) — independent of how many solves came before. This is
    # what lets the exploitation battery's CRN A/B cancel the search's internal rollout variance: the
    # ON and OFF arms draw identical search randomness wherever their situations match, so the uplift
    # isolates genuine exploit-induced deviations. False restores the legacy running-stream behavior.
    per_decision_seed: bool = True
    rng_salt: int = 0                 # mixed into the public-state seed (vary for independent replicas)
    # CFR+ recency-weighted (linear) strategy averaging: accumulate iteration t's strategy with weight
    # (t+1) instead of a flat 1.0. This is the averaging half of CFR+ (Tammelin 2014); without it the
    # served average over-weights the noisy early iterates and converges slower. The final extraction
    # already normalizes by the accumulated sum, so only the RELATIVE per-iteration weight matters.
    # Default ON (matches the "CFR+" the solver advertises); set False for the legacy uniform average.
    # Mirrored in the native (Rust) subgame solve so the statistical-equivalence crosscheck stays green.
    # Env override POKERBOT_LINEAR_AVERAGING=0/1 (read at import) lets an eval A/B both arms without code
    # changes (the field default IS the env when present); explicit constructor arg still wins.
    linear_averaging: bool = field(
        default_factory=lambda: os.environ.get("POKERBOT_LINEAR_AVERAGING", "1") not in ("0", "false", "False", ""))
    # DBR-style frozen-read response. Two measured facts motivate this: (1) opponent regrets are
    # never updated in this solver (only hero nodes update), so the "regret_match" half of the
    # opponent mix is permanently UNIFORM; (2) the tabular blueprint has ~0% serve-time coverage
    # postflop (adapter falls back to uniform), so the blueprint half is uniform too — the hero has
    # been best-responding to a half-uniform phantom (measured value/bluff separation vs a pure
    # station: ≈0 at 10 iters). When True and a context read exists at an opponent node, the modeled
    # policy is FROZEN to the read's posterior bin distribution expanded over legal actions — the
    # only information-bearing opponent model postflop — and hero CFR solves a true best response to
    # it. Gated on the read ⇒ zero-deviation bit-identity (opponent_model=None) and the exploit-OFF
    # arm are byte-unchanged. Exploit math/keys untouched (the read is consumed, not modified).
    frozen_read_response: bool = False
    # RNR restriction of the SOLVER-internal best-response (#28). The opponent-node tail models the
    # villain deviating from blueprint and hero CFR best-responds to it; left unbounded (the flat
    # _CTX_MAX_ALPHA=0.7, or the full frozen posterior), that BR crushes a STATIC reg but is maximally
    # COUNTER-exploitable by an ADAPTIVE pro. When True, the modeled deviation is bounded by the N- and
    # stability-dependent RNR cap cap_eff(stability, n) (rnr_cap.py): a stable read keeps the N-ceiling
    # (HU/3-max 1.0 => unchanged; multiway lower, no minimax), a caught adapter collapses toward
    # CAP_FLOOR. The hero's response is then a RESTRICTED BR (exposed to <= ~rnr*eps). The caller sets
    # ``opponent_stability`` (per-seat, from a StabilityTracker) and ``opponent_n_live`` (table size).
    # Default OFF => byte-identical (the modeled deviation is the legacy un-restricted cap); the
    # opponent_model=None GTO path is untouched either way (the whole tail is gated on a usable read).
    rnr_restrict: bool = False
    # Native (Rust) engine hot path. None => auto (use the bit-identical native core IFF it imported
    # cleanly AND its startup self-check passed; else silent pure-Python fallback). True => force-enable
    # (still no-op if unavailable); False => force the pure-Python np_engine. The native core is a
    # bit-identical drop-in for apply_action/legal_mask/sample_chance/showdown, so the solve output is
    # byte-unchanged either way — it only changes speed. See pokerbot/search/native_solver.py.
    native: bool | None = None
    # STARVATION-ROBUST SERVE (blend-to-blueprint when budget-bound). None => follow the process env
    # POKERBOT_SEARCH_BLUEPRINT_BLEND (default ON live / set OFF for the eval+LBR battery). True/False =>
    # force per-solver. When ON and a solve is budget-bound (iterations << max_iterations), the served
    # ROOT policy is a convex combination of the noisy CFR average and the trusted root blueprint, the
    # blueprint weight scaling with the convergence shortfall (more starved => lean more on the prior).
    # A fully-converged solve serves the pure search policy (byte-identical). Never changes the per-node
    # CFR traversal — only the final extracted root policy. See _BLUEPRINT_BLEND above.
    blueprint_blend: bool | None = None
    # Cap on the blueprint weight at TOTAL starvation (0 iterations of the target ran). The realized weight
    # is blueprint_blend_max * shortfall where shortfall = 1 - iters/max_iterations in [0,1]; so even a fully
    # starved serve keeps (1 - blueprint_blend_max) of the search signal. 1.0 => a 0-iteration serve is the
    # pure blueprint (the trusted floor); lower keeps more search. Convex either way.
    blueprint_blend_max: float = 1.0
    # SINGLE-TABLE ITERATION FLOOR. When the live driver knows only ONE table is demanding action
    # (single_table=True at solve() time, or a per-solver default here), the solve is allowed up to
    # ``single_table_iter_floor`` iterations AND ``single_table_budget_ms`` wall-clock before the budget
    # bound fires — single-tabling has spare time, so let the solve CONVERGE rather than serving a starved
    # average. Under multi-table load (single_table=False) the per-table time_budget_ms / max_iterations are
    # UNCHANGED (no slowdown). None => disabled (legacy: time_budget_ms / max_iterations always bind).
    single_table_iter_floor: int | None = None
    single_table_budget_ms: int | None = None
    # MEASUREMENT-ONLY: force the budget-bound blend to apply at a FIXED iteration count, treating a solve
    # that ran ``force_budget_bound`` iterations as if it had been wall-clock-starved at that many (shortfall
    # = 1 - force_budget_bound/max_iterations). This lets an LBR / determinism harness compare blend ON vs
    # OFF with the iteration count held IDENTICAL between arms (no wall-clock jitter confound), so the only
    # difference measured is the blend itself. None => off (production: starvation is detected by the real
    # wall clock). Requires blueprint_blend active; never set in live serve.
    force_budget_bound: int | None = None
    # SNAPSHOT INTEROP VALIDATION (defense in depth vs SEARCH-AUDIT F1/F2/F3).
    # ``validate_snapshot_at_init`` (default True): call ``eval.lbr.validate_snapshot`` on the
    # blueprint snapshot at construction time. Catches empty dicts (F1) and string-keyed legacy
    # snapshots (F2) before the solver starts serving uniform-legal at every postflop node.
    # ``strict_snapshot`` (default True): if True the validator raises on any anomaly; if False
    # it warns and returns. Live serve wants strict=True (fail closed). Battery / dev paths can
    # set strict=False to run a no-snapshot smoke without code changes. Both flags default ON; an
    # empty snapshot is NEVER silently accepted in production.
    validate_snapshot_at_init: bool = True
    strict_snapshot: bool = True


# POKERBOT_NATIVE_SUBGAME: when truthy AND the native SubgameSolver is built AND opponent_model is None
# (GTO base), the whole CFR subgame solve runs in Rust (one call/decision, no per-node marshaling). The
# result is STATISTICALLY EQUIVALENT (fast native RNG), not bit-identical. Default OFF => the Python
# `_cfv` loop below runs unchanged. The exploit tail (opponent_model != None) is NEVER native (firewall).
_NATIVE_SUBGAME = os.environ.get("POKERBOT_NATIVE_SUBGAME", "0") not in ("0", "", "false", "False")


# POKERBOT_NATIVE_EXPLOIT: when truthy AND the native SubgameSolver exposes solve_exploit (a rebuilt
# .pyd) AND the solve is on the FROZEN-read exploit path (opponent_model set + config.frozen_read_response
# + a frozen table built + no off-tree insert), run the whole CFR subgame solve in Rust — opponent nodes
# apply a PRECOMPUTED pure-frozen deviation (the exploit MATH stays in Python: context_read / Dirichlet
# posterior / BOCPD / nash_reference / cap_eff run in exploit_table.build_frozen_table; native only LOOKS
# UP the output frozen vector + the per-actor rnr scalar — the data-not-code firewall). Native ALSO
# exposes the root per-action CFVs for the AIVAT decision baseline, so the AIVAT gate relaxes too. The
# non-pure tails (frozen_read_response False -> archetype apply_opponent_bias / apply_context_bias, which
# are hole-card dependent) are NEVER native: the gate excludes them and the Python loop runs unchanged.
# STATISTICALLY EQUIVALENT (fast native RNG), not bit-identical. Default OFF => the Python `_cfv` loop
# (with the full exploit tail) runs unchanged.
_NATIVE_EXPLOIT = os.environ.get("POKERBOT_NATIVE_EXPLOIT", "0") not in ("0", "", "false", "False")


# POKERBOT_OFFTREE_RESOLVE: Pluribus-style real-time response to an OFF-TREE opponent bet size. When the
# villain we are facing made a bet whose pot-fraction is NOT well-represented by the static keep-list bet
# abstraction (its pseudo-harmonic-mapped bucket differs from the observed fraction by more than
# _OFFTREE_REL_TOL relative), we ENABLE one extra bet-size bucket — the one whose fraction is closest to
# the observed size — in the ROOT subgame's hero action set FOR THAT SOLVE ONLY (capped at one extra
# child per decision). The solver then branches a hero response sized to the true off-tree action instead
# of reasoning only about the rounded bucket. With the env unset the augmentation never runs and the path
# is BYTE-IDENTICAL to today (the augmented post mask is identically the static mask). No engine / golden /
# abstraction / opponent-store change: this reuses the existing 7 bet buckets (NUM_ACTIONS unchanged), it
# only un-masks one that the static keep-list dropped. Falls back to the static Tier-1 pseudo-harmonic
# translation when the time budget is tight (the insert is skipped if the decision is already budget-bound).
_OFFTREE_RESOLVE = os.environ.get("POKERBOT_OFFTREE_RESOLVE", "0") not in ("0", "", "false", "False")
_OFFTREE_REL_TOL = 0.15  # relative pot-fraction mismatch above which the observed size enters the solve


# POKERBOT_SEARCH_BLUEPRINT_BLEND: starvation-robust serve (Pluribus/Modicum principle). When a live solve
# is BUDGET-BOUND (the wall-clock time_budget_ms cut it short at iterations << max_iterations, see S2), the
# served CFR AVERAGE strategy is UNDER-CONVERGED -> a noisy argmax that flips frame-to-frame under load (the
# trash-raise amplifier). Instead of serving the noisy average, blend the served ROOT policy toward the
# TRUSTED BLUEPRINT prior with weight proportional to the convergence shortfall: a fully-converged solve
# (ran all iterations) serves the PURE search policy (byte-identical to today); a fully-starved solve leans
# hard on the blueprint. The blend is a proper convex combination over the LEGAL mask, renormalized. DEFAULT
# ON (live robustness) but the eval / LBR battery sets it OFF so the certified-exploitability path and the
# CRN A/B stay byte-identical (a converged eval solve never blends anyway, but an env-OFF default keeps the
# code path provably inert). Only BUDGET-BOUND serves are ever changed.
_BLUEPRINT_BLEND = os.environ.get("POKERBOT_SEARCH_BLUEPRINT_BLEND", "1") not in ("0", "", "false", "False")


def blueprint_blend_enabled() -> bool:
    return _BLUEPRINT_BLEND


def set_blueprint_blend(on: bool) -> None:
    """Runtime toggle for the budget-bound blueprint blend (live driver / tests / eval battery)."""
    global _BLUEPRINT_BLEND
    _BLUEPRINT_BLEND = bool(on)


def offtree_resolve_enabled() -> bool:
    return _OFFTREE_RESOLVE


def set_offtree_resolve(on: bool) -> None:
    """Runtime toggle for the off-tree re-solve trigger (used by tests / the live driver)."""
    global _OFFTREE_RESOLVE
    _OFFTREE_RESOLVE = bool(on)


def native_subgame_enabled() -> bool:
    return _NATIVE_SUBGAME


def set_native_subgame(on: bool) -> None:
    """Runtime toggle for the native CFR subgame fast path (used by the validation harness)."""
    global _NATIVE_SUBGAME
    _NATIVE_SUBGAME = bool(on)


def native_exploit_enabled() -> bool:
    return _NATIVE_EXPLOIT


def set_native_exploit(on: bool) -> None:
    """Runtime toggle for the native EXPLOIT subgame fast path (used by the validation harness)."""
    global _NATIVE_EXPLOIT
    _NATIVE_EXPLOIT = bool(on)


def wire_native_exploit_serve(solver) -> bool:
    """LIVE/production wiring for the NATIVE exploit fast path (POKERBOT_LIVE_NATIVE_EXPLOIT).

    The frozen-read-response serve decision (2026-07-02): exploit decisions (opponent_model set) have
    been the last starved search surface — the dynamic Python exploit tail bought only ~68 iters in the
    800ms live budget while GTO decisions run the full K=2000 natively. The dynamic tail is provably
    STATIC within a solve (opponent regrets never update one-sided => uniform regret-match; the
    blueprint half is cached; context_read is a pure function of (fixed stats, public-state ctx key)),
    so freezing the read-response into tables at solve start loses no within-solve adaptivity. What
    changes is the modeled-opponent FUNCTIONAL FORM: the frozen serve is the read's posterior expanded
    over legal actions (RNR-blended), not the alpha-capped ``apply_context_bias`` tilt — the
    DBR response the ``frozen_read_response`` config documents (the dynamic tail best-responds to a
    half-uniform phantom postflop). Firewall: ALL exploit math (context_read / Dirichlet / BOCPD /
    cap_eff — the RNR cap binds BEFORE freezing) runs in Python inside
    ``exploit_table.build_frozen_table`` per decision; native only READS the frozen vectors + per-actor
    rnr scalars (data-not-code), and its Rust side hard-forces two_sided/leaf_biased/avg_skip/
    leaf_worlds to the certified no-op values.

    Gate semantics: DEFAULT OFF — opt-in via POKERBOT_LIVE_NATIVE_EXPLOIT=1 (an explicit env value
    always wins; under POKERBOT_TEST_MODE it is OFF like everything else). WHY OFF: the 2026-07-02
    FIDELITY GATE (artifacts/native_exploit_fidelity_gate.json — 15 game-flow exploit states, real
    learned profiles, FIXED equal K=1000) measured the frozen-native serve GENUINELY DIVERGING from
    the dynamic Python serve: action flip rate 0.667, mean policy L1 0.77 (decomposed: the frozen
    SEMANTICS alone flip 0.40 / L1 0.52 — the different functional form above — the rest is the
    certified native-RNG + uniform-leaf component). The dynamic response IS static within a solve,
    so freezing loses no adaptivity, but the frozen response is a DIFFERENT modeled opponent; per
    the divergence rule the default cannot flip on speed alone. It flips only after a VALUE A/B
    (adaptive-harness exploit-ON, this gate ON vs OFF, same seeds — note scripts/engine_ab_h2h.py
    CANNOT express exploit arms: it never sets opponent_model) certifies which fixed point earns
    more. Speed when opted in: mean 1525 iters in 382ms (worst 685ms) at the live 800ms budget vs
    the dynamic tail's ~62 iters (worst wall 1042ms) — ~25x iterations-in-budget.

    When opted in AND the native solve_exploit surface is available (built .pyd + EMD capability
    gate), sets the solver's ``config.frozen_read_response = True`` and flips the process-global
    native-exploit gate. Returns True iff wired. Fail-safe: any unavailability/exception leaves the
    solver on the dynamic Python exploit tail byte-unchanged."""
    want = os.environ.get("POKERBOT_LIVE_NATIVE_EXPLOIT", "0").lower() not in (
        "0", "", "false", "no", "off")
    if not want:
        return False
    try:
        from pokerbot.search import native_solver as _ns

        if not _ns.native_subgame_exploit_available():
            return False
        solver.config.frozen_read_response = True
        set_native_exploit(True)
        return True
    except Exception:  # noqa: BLE001  (never let the enable kill a serve path)
        return False


# NATIVE BUDGET FIT (live wiring 2026-07-02). A native subgame solve is one non-interruptible FFI
# call, and per-iteration cost varies ~20x across roots (a big multiway flop subgame at K=2000 costs
# ~1.7s on the live box; a river heads-up spot ~40ms). Under a live-scale wall budget
# (eff_budget_ms < _NATIVE_FIT_BUDGET_MS) the solve is split into a K=_NATIVE_FIT_PROBE_ITERS probe
# call that measures THIS root's rate, then one main call fitted to the remaining budget (same public
# seed => statistically one longer solve). Budgets >= the threshold (eval/battery/measurement configs
# use 1e9+ or no wall) skip the fit entirely — single full-K call, byte-identical to before.
_NATIVE_FIT_BUDGET_MS = 10_000
_NATIVE_FIT_PROBE_ITERS = 200


def _abstraction_masks(client_bet_sizing: bool):
    if not client_bet_sizing:
        return None, None
    from pokerbot.training.sd_cfr_nlhe import build_allowed_action_mask

    return build_allowed_action_mask((4, 5, 6)), build_allowed_action_mask((0, 1, 3, 4, 5))


class PluribusReBeLSolver:
    """Depth-limited CFR+ on np_engine with a continuation-gadget leaf and a tabular blueprint prior."""

    def __init__(
        self,
        blueprint_snapshot: dict[bytes, np.ndarray],
        config: PluribusReBeLConfig | None = None,
        rng: np.random.Generator | None = None,
        opponent_model: dict | None = None,
    ) -> None:
        self.config = config or PluribusReBeLConfig()
        # Apply the native-engine policy: auto (None) leaves the import-time default (on iff available
        # + self-checked); True force-enables; False forces pure-Python. Global toggle (the hot modules
        # dispatch through native_solver), so the most-recently-constructed solver's policy wins — fine
        # for the battery, which builds the ON/OFF arms with the same flag.
        from pokerbot.search import native_solver as _ns
        if self.config.native is True:
            _ns.enable()
        elif self.config.native is False:
            _ns.disable()
        self.rng = rng or np.random.default_rng(0)
        # Parity with TurboReBeLSolver: None => regret-matched/blueprint opponents (unchanged GTO).
        self.opponent_model = opponent_model
        self.gift_register = None  # set by the live driver / battery; None => static Pmax cap.
        # RNR restriction inputs (#28), set by the caller alongside opponent_model when config.rnr_restrict
        # is on. ``opponent_stability``: {seat -> read stability in [0,1]} from a StabilityTracker (a caught
        # adapter -> 0). ``opponent_n_live``: the live-seat count for the N-dependent ceiling. Both None =>
        # stability 1.0 / N inferred from the model size (rnr still bounds multiway via cap_eff).
        self.opponent_stability: dict | None = None
        self.opponent_n_live: int | None = None
        self.adapter = TabularBlueprintAdapter(
            blueprint_snapshot, client_bet_sizing=self.config.client_bet_sizing
        )
        self._allowed_pre, self._allowed_post = _abstraction_masks(self.config.client_bet_sizing)
        # F1/F2 defense in depth: an empty or string-keyed blueprint snapshot silently serves
        # uniform-legal at every postflop node (the LBR fingerprint lookup returns None, and the
        # adapter falls back to ``mask / max(mask.sum(), 1)``). Fail closed in production; the
        # toy / dev / battery paths can opt out by passing ``validate_snapshot_at_init=False``
        # on the config if they genuinely need an empty prior.
        # TEST_MODE bypass: when POKERBOT_TEST_MODE=1 (set by tests/conftest.py) the validator
        # logs a warning instead of raising, so the many unit tests that pass an empty {} snapshot
        # continue to run. Production never sets this env var; the bypass is unreachable there.
        if getattr(self.config, "validate_snapshot_at_init", True):
            from pokerbot.eval.lbr import validate_snapshot
            _test_mode = os.environ.get("POKERBOT_TEST_MODE", "").lower() in ("1", "true", "yes")
            # When POKERBOT_TEST_MODE=1, default to PERMISSIVE validation (warn instead of raise) so
            # the dozens of unit tests that pass an empty {} blueprint snapshot continue to run.
            # Production never sets this env var; the bypass is unreachable there.
            if _test_mode:
                _strict = False
            else:
                _strict = bool(getattr(self.config, "strict_snapshot", True))
            validate_snapshot(
                blueprint_snapshot,
                strict=_strict,
                context="PluribusReBeLSolver.__init__",
            )
        # Per-solve off-tree augmentation: when POKERBOT_OFFTREE_RESOLVE fires, this holds the post mask
        # to use AT THE ROOT INFOSET ONLY (the static _allowed_post + one extra bet bucket matching the
        # observed off-tree villain size). None => no augmentation (default / interior nodes), so _abstract
        # is byte-identical to today. Keyed by the root hero infoset so the extra child appears only where
        # the hero is actually responding to the off-tree bet, never at deeper recursion nodes.
        self._root_extra_key: str | None = None
        self._root_extra_post: np.ndarray | None = None
        self._offtree_inserted = False  # diagnostic: did an extra child enter the last solve
        self._regrets: dict[str, np.ndarray] = {}
        self._strategy_sum: dict[str, np.ndarray] = {}
        self._bp_cache: dict[str, np.ndarray] = {}  # blueprint probs by infoset key (within a solve)
        # Leaf value cache: key -> (sum, n) running mean over the first config.leaf_cache_worlds
        # VISITING hole-worlds (BUG-1 fix; (v, 1) first-touch-frozen at the default of 1).
        self._leaf_cache: dict = {}
        self._leaf_eval_count = 0
        self._leaf_cache_hits = 0
        self._leaf_counter = 0  # per-leaf CRN offset so distinct leaves get distinct board runouts
        self._leaf_cont_weights: dict[int, np.ndarray] | None = None  # model-aware leaf (per solve)
        self._budget_bound_count = 0  # per-solver count of decisions cut short by time_budget_ms (S2)
        # One-time warning latch: a NATIVE solve (one non-interruptible FFI call) whose wall exceeded
        # its budget. Separate from _budget_bound_count (which keeps its certified S2 semantics:
        # Python-loop solves cut short mid-iteration).
        self._native_over_budget_warned = False
        self._two_sided_active = False  # set per-solve; True => cycle traverser (config.two_sided, GTO path)
        self._traversers = []
        # R2 biased continuation leaves: set per-solve (config.leaf_biased_continuations, GTO path only —
        # mirrors the two_sided gating). False (default) => the legacy leaf path, byte-identical.
        self._leaf_biased_active = False
        # BUG-5 delayed averaging: iterations t < this accumulate NO strat_sum / AIVAT root CFVs. Set
        # per-solve from config.avg_skip_frac * the solve's iteration target; 0 (default) = legacy.
        self._avg_skip_iters = 0
        self._hole_rng = None  # opponent-hole sampler stream (== self.rng unless config.decouple_sampler)
        # AIVAT decision-baseline capture (default OFF => byte-identical hot loop + native fast path).
        # When armed (search_agent.enable_decision_capture with aivat_leaf_q), the solve accumulates the
        # ROOT per-action CFVs it ACTUALLY used to form the served strategy (depth-0 hero node) and
        # _leaf_q returns their (CFR+-weighted) average -- a baseline CONSISTENT with the served policy,
        # which is what makes AIVAT's decision term a POSITIVE variance reducer on the search path (the
        # old 1-ply gadget _leaf_q was inconsistent -> anti-correlated -> negative VR).
        self._aivat_capture = False
        self._root_av_sum: np.ndarray | None = None   # sum_t weight_t * action_values_t at the root
        self._root_av_wsum = 0.0                       # sum_t weight_t (the normalizer)

    def _blend_active(self) -> bool:
        """Resolve the blueprint-blend flag: the per-solver config override (True/False) wins; None
        follows the process env (POKERBOT_SEARCH_BLUEPRINT_BLEND). Read at solve time so set_blueprint_blend()
        and a runtime config change both take effect without rebuilding the solver."""
        cb = self.config.blueprint_blend
        return _BLUEPRINT_BLEND if cb is None else bool(cb)

    # ------------------------------------------------------------------ keys / masks
    def _abstract(self, mask: np.ndarray, street: int, allowed_post: np.ndarray | None = None) -> np.ndarray:
        if self._allowed_pre is None:
            return mask
        if street == 0:
            allowed = self._allowed_pre
        else:
            allowed = allowed_post if allowed_post is not None else self._allowed_post
        return mask & allowed

    def _legal(self, env, seat: int) -> np.ndarray:
        m = legal_action_mask_np(env, seat).astype(bool)
        # Off-tree re-solve: AT THE ROOT INFOSET ONLY, use the augmented post mask (static keep-list +
        # the one extra bucket matching the observed off-tree villain size) so the hero gets a response
        # child sized to the true bet. Identified by the cached root infoset key; every other node (and
        # every solve when the augmentation never fired) falls through to the static _allowed_post,
        # byte-identical to today.
        allowed_post = None
        if self._root_extra_post is not None and int(env.street) != 0:
            if self._infoset_key(env, seat) == self._root_extra_key:
                allowed_post = self._root_extra_post
        return self._abstract(m, int(env.street), allowed_post)

    @staticmethod
    def _effective_stack_bb(env) -> float:
        """Effective starting stack in BB = min over in-hand (non-folded) seats of (stack + committed_total).
        Chips only move from stack to committed within a hand, so the sum is each seat's depth at hand
        start; the min is the effective depth that caps the action. Used by the A1 preflop-blueprint band."""
        totals = np.asarray(env.stacks, dtype=np.float64) + np.asarray(env.committed_total, dtype=np.float64)
        live = (np.asarray(env.seat_status) != SEAT_FOLDED) & (totals > 0)
        chips = totals[live].min() if live.any() else (totals.max() if totals.size else 100.0 * CHIP_UNIT)
        return float(chips) / float(CHIP_UNIT)

    def _offtree_extra_bucket(self, env, hero: int) -> int | None:
        """Off-tree re-solve trigger. Inspect the bet the hero is FACING at this root and derive its
        pot-fraction ``frac_obs = villain_bet_to / pot_before_villain_bet``. The static abstraction's
        Tier-1 translation rounds that to ``mapped = pseudo_harmonic_bucket(frac_obs)`` over the FULL
        7-bucket menu (``bucket_index_for_bet``). When that rounded bucket's pot-fraction differs from the
        observed fraction by more than ``_OFFTREE_REL_TOL`` (relative) — i.e.
        ``|frac_obs - BET_SIZE_BUCKETS[mapped]| / frac_obs > _OFFTREE_REL_TOL`` — the observed size is
        genuinely OFF-TREE and is being mis-served. We then INSERT the observed size into the solve by
        un-masking the single bet bucket whose fraction is closest to ``frac_obs`` and that the static
        postflop keep-list currently DROPS (cap: one extra child / decision). This gives the hero a
        response sized to the true off-tree action that the keep-list otherwise denied. Return None when
        there is no facing bet, the size is on-tree (within tolerance), or every nearby bucket is already
        served (nothing to add). Pure read of the root env; never mutates it."""
        if self._allowed_post is None:
            return None
        to_call = int(env.current_bet) - int(np.asarray(env.committed_street)[hero])
        if to_call <= 0:
            return None  # not facing a bet -> nothing off-tree to respond to
        # villain bet-TO this street = current_bet; pot BEFORE that bet = current pot minus the chips the
        # villain put in to make it. We approximate the villain's street contribution as current_bet (a
        # clean open) which is exact for the dominant single-raise facing spot; clamp the base to > 0.
        pot_now = float(int(env.pot_dead) + int(np.asarray(env.committed_street).sum())) / float(CHIP_UNIT)
        villain_to = float(int(env.current_bet)) / float(CHIP_UNIT)
        pot_before = pot_now - villain_to
        if pot_before <= 1e-9:
            return None
        frac_obs = villain_to / pot_before
        if frac_obs <= 0.0:
            return None
        # The observed size's best representation in the FULL 7-bucket menu vs in the SERVED keep-list.
        # ``nearest_full`` is the true-size leaf (the bucket the abstraction WOULD use if unrestricted);
        # ``nearest_served`` is what the static keep-list actually rounds the hero's response toward. When
        # the keep-list DROPS the bucket the size belongs in, nearest_served is materially farther — that
        # gap is the off-tree leak. Trigger when the served rounding is worse than the full-menu rounding
        # by more than _OFFTREE_REL_TOL (relative to the observed fraction), and the true-size bucket is
        # one the keep-list dropped. The Tier-1 ``pseudo_harmonic_bucket`` over the full menu identifies
        # the intended bucket (anti-exploitation), matching bucket_index_for_bet on the serve path.
        nearest_full = pseudo_harmonic_bucket(frac_obs, BET_SIZE_BUCKETS)
        if bool(self._allowed_post[BET_BUCKET_START + nearest_full]):
            return None  # the intended bucket is already served -> no keep-list leak, nothing off-tree
        served_idx = [i for i in range(len(BET_SIZE_BUCKETS))
                      if bool(self._allowed_post[BET_BUCKET_START + i])]
        if not served_idx:
            return None
        nearest_served = min(served_idx, key=lambda i: abs(frac_obs - BET_SIZE_BUCKETS[i]))
        d_full = abs(frac_obs - BET_SIZE_BUCKETS[nearest_full])
        d_served = abs(frac_obs - BET_SIZE_BUCKETS[nearest_served])
        if (d_served - d_full) / frac_obs <= _OFFTREE_REL_TOL:
            return None  # the keep-list rounding is no worse than tolerance -> on-tree, no insert
        return int(nearest_full)  # un-mask the dropped true-size bucket (one extra child this solve)

    def _infoset_key(self, env, seat: int) -> str:
        hist_len = int(np.asarray(env.hist_len))
        seats = tuple(int(x) for x in np.asarray(env.hist_seat)[:hist_len])
        actions = tuple(int(x) for x in np.asarray(env.hist_action)[:hist_len])
        board = tuple(int(x) for x in np.asarray(env.board) if int(x) >= 0)
        pot = float(np.asarray(env.pot_dead)) + float(np.asarray(env.committed_street).sum())
        stacks = np.asarray(env.stacks, dtype=np.float64)
        folded = np.asarray(env.seat_status == SEAT_FOLDED, dtype=bool)
        active_mean = float(stacks[~folded].mean()) if (~folded).any() else float(stacks.mean())
        spr_bucket = int(pot / max(active_mean, 1.0) * 4)
        return f"{seat}|{int(np.asarray(env.street))}|{board}|{seats}|{actions}|spr{spr_bucket}"

    def _public_seed(self, env, hero: int) -> int:
        """Stable 64-bit seed from the PUBLIC state at a solve root (board, history, street, stacks,
        commitments, statuses — never hole cards). Two solves at the same public situation get the
        same Monte-Carlo stream regardless of process history; that is the CRN-cancellation property
        the exploitation battery needs (and plain determinism hygiene). blake2b, not Python hash()
        (which is salted per-process)."""
        hist_len = int(np.asarray(env.hist_len))
        parts = [int(self.config.rng_salt), int(hero), int(np.asarray(env.street)), hist_len]
        parts += [int(x) for x in np.asarray(env.hist_seat)[:hist_len]]
        parts += [int(x) for x in np.asarray(env.hist_action)[:hist_len]]
        parts += [int(x) for x in np.asarray(env.board).reshape(-1)]
        parts += [int(round(float(x))) for x in np.asarray(env.stacks).reshape(-1)]
        parts += [int(round(float(x))) for x in np.asarray(env.committed_street).reshape(-1)]
        parts.append(int(round(float(np.asarray(env.pot_dead)))))
        parts += [int(x) for x in np.asarray(env.seat_status).reshape(-1)]
        digest = hashlib.blake2b(np.asarray(parts, dtype=np.int64).tobytes(), digest_size=8).digest()
        return int.from_bytes(digest, "little")

    def _leaf_key(self, env) -> str:
        hist_len = int(np.asarray(env.hist_len))
        action_seq = tuple(int(x) for x in np.asarray(env.hist_action)[:hist_len])
        board = tuple(int(x) for x in np.asarray(env.board) if int(x) >= 0)
        street = int(np.asarray(env.street))
        pot = float(np.asarray(env.pot_dead)) + float(np.asarray(env.committed_street).sum())
        return cached_canonicalize_key(board, street, action_seq + (int(pot * 2),))

    def _get_regrets(self, key: str) -> np.ndarray:
        r = self._regrets.get(key)
        if r is None:
            r = np.zeros(NUM_ACTIONS, dtype=np.float64)
            self._regrets[key] = r
            self._strategy_sum[key] = np.zeros(NUM_ACTIONS, dtype=np.float64)
        return r

    # ------------------------------------------------------------------ opponent strategy (exploit tail)
    def _cached_blueprint(self, env, actor: int, key: str) -> np.ndarray:
        """Blueprint probs memoized by infoset key — the same opponent infoset recurs across the solve's
        iterations, and the fingerprint_fast + snapshot lookup is the per-node hot path."""
        bp = self._bp_cache.get(key)
        if bp is None:
            bp = self.adapter.blueprint(env, actor)
            self._bp_cache[key] = bp
        return bp

    def _rnr_n_live(self) -> int:
        """Table size N for the RNR ceiling (rnr_cap.cap_eff). Prefer the caller-set live-seat count;
        else infer from the model size (modeled villains + hero), clamped to [2, 6]. Larger N => a lower
        ceiling (multiway has no minimax safety), so an unknown count defaults conservatively, never below 2."""
        if self.opponent_n_live is not None:
            n_full = max(2, int(self.opponent_n_live))
        elif self.opponent_model:
            n_full = max(2, min(6, len(self.opponent_model) + 1))
        else:
            return 2
        # Optionally relax to the HU ceiling on a NO-PUNISHER (all-confirmed-fish) table (default-OFF env).
        return _rnr_fish_relaxed_n(n_full, self.opponent_model)

    def _opponent_strategy(
        self, env, actor: int, legal: np.ndarray, regrets: np.ndarray, key: str
    ) -> np.ndarray:
        strategy = _regret_match(regrets, legal.astype(np.float64))
        bp = self._cached_blueprint(env, actor, key) * legal
        s = bp.sum()
        if s > 1e-12:
            mix = float(self.config.blueprint_mix)
            strategy = (1.0 - mix) * strategy + mix * (bp / s)
            ss = strategy.sum()
            strategy = strategy / ss if ss > 1e-12 else legal.astype(np.float64) / max(legal.sum(), 1)
        # Opponent-conditioned safe exploitation (lifted tail; None/low-conf => unchanged GTO).
        if self.opponent_model is not None:
            stats = self.opponent_model.get(actor)
            if stats is not None:
                read = context_read(stats, _context_key_from_env(env, actor))
                if read is not None:
                    max_alpha = _CTX_MAX_ALPHA
                    if self.gift_register is not None:
                        max_alpha = self.gift_register.capped_max_alpha(actor, _CTX_MAX_ALPHA)
                    # RNR restriction (#28): bound the modeled deviation by cap_eff(stability, n) so hero's
                    # response is a RESTRICTED best-response. ``rnr`` (None when the flag is off) also gates
                    # the frozen-posterior blend below. The gift relaxation composes UNDER it (min), so the
                    # RNR cap is the hard outer bound. Default-OFF => max_alpha is the legacy value unchanged.
                    rnr = None
                    if self.config.rnr_restrict:
                        stab = (1.0 if self.opponent_stability is None
                                else float(self.opponent_stability.get(actor, 1.0)))
                        rnr = _rnr_cap_eff(stab, self._rnr_n_live())
                        max_alpha = min(max_alpha, rnr)
                    if self.config.frozen_read_response:
                        # Frozen modeled policy built FROM the read posterior itself. Postflop the
                        # tabular blueprint has ~zero serve-time coverage (uniform fallback) and
                        # opponent regrets never update (uniform regret-match), so the only
                        # information-bearing opponent model is the Bayesian read: expand its bin
                        # posterior over the legal actions (uniform within a bin, renormalized over
                        # realizable bins) and let hero CFR best-respond to THAT. The posterior is
                        # already prior-shrunk + confidence-aware, so it needs no alpha re-dilution.
                        frozen = np.zeros(NUM_ACTIONS, dtype=np.float64)
                        bin_n = np.zeros(read.post.shape[0], dtype=np.float64)
                        legal_idx = np.flatnonzero(legal)
                        for a in legal_idx:
                            bin_n[_action_bin(int(a))] += 1.0
                        for a in legal_idx:
                            b = _action_bin(int(a))
                            frozen[a] = read.post[b] / bin_n[b]
                        fs = frozen.sum()
                        if fs > 1e-12:
                            frozen = frozen / fs
                            if rnr is None:
                                strategy = frozen
                            else:
                                # RNR-blend the (confidence-aware but stability/N-blind) frozen posterior
                                # toward the regret-match/blueprint base: (1-rnr)*base + rnr*frozen. With
                                # rnr=1 (HU stable read) this is the pure posterior — byte-identical to the
                                # un-restricted frozen path; multiway / caught adapters pull toward the base.
                                mixed = (1.0 - rnr) * strategy + rnr * frozen
                                ms = mixed.sum()
                                strategy = mixed / ms if ms > 1e-12 else frozen
                        else:
                            strategy = apply_context_bias(strategy, read, legal, max_alpha=max_alpha)
                    else:
                        strategy = apply_context_bias(strategy, read, legal, max_alpha=max_alpha)
                else:
                    strategy = apply_opponent_bias(strategy, stats, legal)
        return strategy

    # ------------------------------------------------------------------ leaf
    def _leaf_value(self, env, hero: int, t: int = 0) -> float:
        """Hero bb value at a depth-limit leaf via the continuation gadget, cached by canonical key.

        ``t`` = the CFR iteration index, read ONLY by the R2 biased-continuation branch (the choice
        nodes weight their strategy-sum like action nodes and the non-traverser draw is a pure function
        of (infoset, t)); the legacy path below ignores it — byte-identical default.

        BUG-1 FIX (config.leaf_cache_worlds): the cache entry is a ``(sum, n)`` RUNNING MEAN over the
        first M VISITING hole-worlds. The env at each visit carries THAT iteration's sampled opponent
        holes, so while n < M every hit re-evaluates the gadget under the CURRENT world (deterministic
        per-(leaf, n) seed) and the mean integrates M independent hole-worlds instead of freezing the
        first-touch draw; n >= M freezes the mean. M=1 (default) is exactly the legacy first-touch
        cache, byte-identical (world 0 uses the EXACT legacy seed string and sum/1 == v)."""
        if bool(env.done):
            return float(env.rewards[hero]) / CHIP_UNIT
        key = self._leaf_key(env)
        if self._two_sided_active:
            key = (key, int(hero))  # two-sided: the leaf value is per-traverser, so key the cache on the seat
        # R2 — Pluribus biased continuation leaves (gated, default OFF => the legacy path below runs
        # byte-identically).
        if self._leaf_biased_active:
            return self._leaf_value_biased(env, hero, key, t)
        m_worlds = max(1, int(self.config.leaf_cache_worlds))
        hit = self._leaf_cache.get(key)
        if hit is not None:
            leaf_sum, leaf_n = hit
            if leaf_n >= m_worlds:
                self._leaf_cache_hits += 1
                return leaf_sum / leaf_n
        else:
            leaf_sum, leaf_n = 0.0, 0
        leaf_cfg = self.config.leaf
        if self.config.per_decision_seed:
            # Deterministic per-leaf CRN: seed from the canonical leaf key, so the SAME leaf situation
            # integrates the SAME board runouts in any solve, any process, any arm of a CRN A/B —
            # while distinct leaves still get distinct runouts (distinct keys). World 0 keeps the EXACT
            # legacy salt (byte-identity at the default M=1); worlds n>=1 mix the world index in, so the
            # nth re-evaluation of a leaf is still a pure function of (leaf key, n).
            salt = (f"{leaf_cfg.base_seed}|{key}" if leaf_n == 0
                    else f"{leaf_cfg.base_seed}|{key}|w{leaf_n}")
            digest = hashlib.blake2b(salt.encode(), digest_size=8).digest()
            seed = int.from_bytes(digest, "little")
        else:
            # Legacy: a running counter so distinct leaves get distinct runouts (history-dependent).
            self._leaf_counter += 1
            seed = leaf_cfg.base_seed + 100003 * self._leaf_counter
        cfg = ContinuationLeafConfig(
            select=leaf_cfg.select,
            rollout_samples=leaf_cfg.rollout_samples,
            gadget_iters=leaf_cfg.gadget_iters,
            max_playout_depth=leaf_cfg.max_playout_depth,
            base_seed=seed,
            resample_board=leaf_cfg.resample_board,
        )
        v = continuation_leaf_value(env, hero, self.adapter, cfg,
                                    cont_weights=self._leaf_cont_weights)
        # n=0: the sum IS v (not 0.0+v, which could flip a -0.0) so the M=1 default path caches and
        # returns EXACTLY the legacy first-touch value.
        leaf_sum = v if leaf_n == 0 else leaf_sum + v
        leaf_n += 1
        self._leaf_cache[key] = (leaf_sum, leaf_n)
        self._leaf_eval_count += 1
        return leaf_sum / leaf_n

    # ------------------------------------------------------------------ R2 biased continuation leaves
    # 4-way legal mask for a continuation-CHOICE node: the first NUM_CONTINUATIONS "actions" are the
    # continuations (0=blueprint, 1=fold-biased, 2=call-biased, 3=raise-biased); the rest dead. Lets
    # the choice node reuse _regret_match / the NUM_ACTIONS table rows in the SAME regret/strategy maps.
    _CONT_LEGAL = np.array(
        [1.0] * NUM_CONTINUATIONS + [0.0] * (NUM_ACTIONS - NUM_CONTINUATIONS), dtype=np.float64
    )

    def _cont_choice_key(self, env, seat: int) -> str:
        """The continuation-choice infoset key for ``seat`` at leaf ``env``: keyed EXACTLY like every
        other subgame infoset (seat + public state + betting history + spr) plus a prefix that can
        never collide with an action infoset (action keys start with the seat digit). NOTE (honest
        scope): like ALL subgame infoset keys in this engine it carries no private-hand component (the
        documented BUG-2/R3 limitation) — hero's hand is fixed per solve so hero's choice is exactly
        Pluribus-keyed; a two-sided opponent's choice is range-average across its resampled worlds,
        the same approximation its ACTION infosets already make."""
        return "cont!" + self._infoset_key(env, seat)

    def _cont_choice_u(self, ck: str, t: int) -> float:
        """Deterministic u in [0,1) for a non-traverser's continuation draw: a PURE FUNCTION of
        (choice infoset, iteration t) — never the shared traversal rng — so the same infoset gets the
        SAME continuation everywhere within an iteration (Pluribus's per-infoset consistency) and the
        solve stays CRN-stable across processes/arms."""
        digest = hashlib.blake2b(
            f"{self.config.leaf.base_seed}|contchoice|{ck}|{t}".encode(), digest_size=8
        ).digest()
        return int.from_bytes(digest, "little") / float(1 << 64)

    def _leaf_value_biased(self, env, hero: int, base_key, t: int) -> float:
        """R2 — the biased-continuation LEAF NODE: a 4-way continuation-choice decision for every seat
        that can still act (SEAT_LIVE), solved by the SAME CFR as the rest of the subgame (mirrors
        subgame.rs::leaf_value_biased exactly):

          * every live NON-traverser seat SAMPLES its continuation from its own learned 4-way strategy
            (regret-matched over its choice-infoset regrets; uniform until it has traversed — under
            one-sided solving non-traversers never update, matching that mode's semantics), via the
            pure-function draw above;
          * the TRAVERSER (``hero`` here is the current traverser) BRANCHES all 4 continuations,
            evaluates each joint profile, and does the standard CFR+ regret update + (avg-gated,
            linear-weighted) strategy-sum accumulation on its choice infoset — "essentially just
            another action" (Pluribus); BOTH hero and opponents choose;
          * a traverser that cannot act beyond the leaf (folded / all-in) has no choice: the value is
            the single playout under the others' sampled profile.

        Folded/all-in seats get NO choice node (absent from ``choices``): a folded seat has no stake
        and an all-in seat has no remaining decisions, so a continuation cannot affect either."""
        st = np.asarray(env.seat_status)
        choices: dict[int, int] = {}
        for s in range(st.shape[0]):
            if s == hero or int(st[s]) != int(SEAT_LIVE):
                continue
            ck = self._cont_choice_key(env, s)
            strat = _regret_match(self._get_regrets(ck), self._CONT_LEGAL)
            cum = np.cumsum(strat[:NUM_CONTINUATIONS])
            u = self._cont_choice_u(ck, t)
            choices[s] = int(min(np.searchsorted(cum, u, side="right"), NUM_CONTINUATIONS - 1))
        if int(st[hero]) != int(SEAT_LIVE):
            return self._biased_profile_value(env, hero, base_key, choices)
        # Traverser's choice node: branch all 4 continuations (hero's own continuation biases hero's
        # own playout actions too).
        ck = self._cont_choice_key(env, hero)
        regrets = self._get_regrets(ck)
        strat = _regret_match(regrets, self._CONT_LEGAL)
        cv = np.zeros(NUM_CONTINUATIONS, dtype=np.float64)
        for c in range(NUM_CONTINUATIONS):
            choices[hero] = c
            cv[c] = self._biased_profile_value(env, hero, base_key, choices)
        ev = float(np.dot(strat[:NUM_CONTINUATIONS], cv))
        for c in range(NUM_CONTINUATIONS):
            regrets[c] = max(regrets[c] + cv[c] - ev, 0.0)
        # Strategy-sum accumulation: identical gating/weighting to the action nodes (delayed averaging
        # + CFR+ linear weight), so the choice node is a first-class citizen of the served average.
        if t >= self._avg_skip_iters:
            weight = float(t + 1) if self.config.linear_averaging else 1.0
            self._strategy_sum[ck][:NUM_CONTINUATIONS] += weight * strat[:NUM_CONTINUATIONS]
        return ev

    def _biased_profile_value(self, env, hero: int, base_key, choices: dict[int, int]) -> float:
        """Leaf value under the JOINT continuation profile ``choices``, cached + M-world-integrated
        exactly like the legacy leaf.

        LEAF-CACHE KEY DESIGN (mirrors subgame.rs::biased_profile_value): the cache key is
        ``(base_key, "cont_prof", sorted profile)`` — ``base_key`` already carries the two-sided
        traverser id — so a cached value is valid ONLY for the exact continuation profile it was
        computed under, and the ``leaf_cache_worlds`` running mean applies PER (leaf, profile) entry.
        Worst-case profile space is 4^n_live per public leaf, but entries are created LAZILY on visits
        only: per leaf visit the traverser branches its 4 choices while every other seat's choice is
        SAMPLED (one profile per visit) — Pluribus's compression insight — so the map grows by at most
        4 entries per leaf visit and the 4^n blowup never materializes."""
        prof = tuple(sorted(choices.items()))
        ckey = (base_key, "cont_prof", prof)
        m_worlds = max(1, int(self.config.leaf_cache_worlds))
        hit = self._leaf_cache.get(ckey)
        if hit is not None:
            leaf_sum, leaf_n = hit
            if leaf_n >= m_worlds:
                self._leaf_cache_hits += 1
                return leaf_sum / leaf_n
        else:
            leaf_sum, leaf_n = 0.0, 0
        leaf_cfg = self.config.leaf
        if self.config.per_decision_seed:
            salt = (f"{leaf_cfg.base_seed}|{ckey}" if leaf_n == 0
                    else f"{leaf_cfg.base_seed}|{ckey}|w{leaf_n}")
            digest = hashlib.blake2b(salt.encode(), digest_size=8).digest()
            seed = int.from_bytes(digest, "little")
        else:
            self._leaf_counter += 1
            seed = leaf_cfg.base_seed + 100003 * self._leaf_counter
        cfg = ContinuationLeafConfig(
            select=leaf_cfg.select,  # not read by the fixed-profile playout; carried for parity
            rollout_samples=leaf_cfg.rollout_samples,
            gadget_iters=leaf_cfg.gadget_iters,
            max_playout_depth=leaf_cfg.max_playout_depth,
            base_seed=seed,
            resample_board=leaf_cfg.resample_board,
        )
        v = biased_profile_leaf_value(env, hero, self.adapter, cfg, choices,
                                      float(self.config.leaf_bias_mult))
        leaf_sum = v if leaf_n == 0 else leaf_sum + v
        leaf_n += 1
        self._leaf_cache[ckey] = (leaf_sum, leaf_n)
        self._leaf_eval_count += 1
        return leaf_sum / leaf_n

    def _build_leaf_cont_weights(self, hero: int) -> dict[int, np.ndarray] | None:
        """MODEL-AWARE LEAF: per-seat continuation-draw weights from the opponent model's aggregate
        read, mapped bins->continuations as [blueprint floor, fold, passive, aggressive] and
        normalized. Built ONCE per solve (the model is fixed within a decision). None when no seat
        has a usable read => the leaf draws via the legacy uniform path, bit-identical (this is the
        zero-deviation gate: opponent_model=None can never change a leaf value)."""
        if self.opponent_model is None:
            return None
        from pokerbot.opponent.bayes_exploit import aggregate_read

        out: dict[int, np.ndarray] = {}
        for seat, stats in self.opponent_model.items():
            if seat == hero or stats is None:
                continue
            read = aggregate_read(stats)
            if read is None:
                continue
            post = np.asarray(read.post, dtype=np.float64)
            # bins [fold, passive, raise-small, raise-large] -> continuations
            # [blueprint, fold, call, raise]; the blueprint floor keeps the playout anchored and
            # caps how hard a sparse read can skew the leaf world. conf scales the tilt.
            conf = float(np.clip(read.conf, 0.0, 1.0))
            w = np.array([
                0.25,
                conf * float(post[0]) + (1.0 - conf) * 0.25,
                conf * float(post[1]) + (1.0 - conf) * 0.25,
                conf * float(post[2] + post[3]) + (1.0 - conf) * 0.25,
            ], dtype=np.float64)
            out[int(seat)] = w / w.sum()
        return out or None

    # ------------------------------------------------------------------ CFR recursion
    def _cfv(self, env, hero: int, depth: int, t: int = 0) -> float:
        # ``t`` = the (0-based) CFR iteration index, threaded down only to weight the strategy-sum
        # accumulation (CFR+ linear averaging); it does not affect regrets or the traversal.
        if bool(env.done):
            return float(env.rewards[hero]) / CHIP_UNIT
        # Depth limit = end of current betting round: the next chance node (street deal) is the leaf.
        # ``t`` is threaded into the leaf for the R2 biased-continuation choice nodes (legacy leaf
        # ignores it — byte-identical default).
        if is_chance_node_np(env):
            return self._leaf_value(env, hero, t)
        if depth >= self.config.max_depth:
            return self._leaf_value(env, hero, t)

        actor = int(env.action_seat)
        if actor < 0:
            return self._leaf_value(env, hero, t)
        legal = self._legal(env, actor)
        if not legal.any():
            # E1 fix: an empty legal mask at a non-terminal street is an engine-state invariant
            # violation (legal_action_mask always returns at least one action at a non-terminal
            # street). Returning a terminal reward here silently substituted a wrong leaf for a
            # live solve; raise so the caller sees the corruption immediately and the LBR / eval
            # harness never silently reports a metric on a corrupt branch.
            street = int(env.street)
            raise RuntimeError(
                f"PluribusReBeLSolver._cfv: empty legal mask at non-terminal street={street} "
                f"actor={actor} hero={hero} — engine invariant violated. "
                "Aborting solve to avoid a corrupt CFR update."
            )

        key = self._infoset_key(env, actor)
        regrets = self._get_regrets(key)

        if actor != hero:
            # NON-TRAVERSER node. TWO-SIDED (GTO, no opponent model): this seat plays its OWN CURRENT
            # regret-matched strategy -- its regrets ARE updated when it is the traverser on other
            # iterations, so it learns (e.g. to fold to a jam) and the solve converges to a subgame
            # equilibrium rather than a one-sided best-response to a fixed loose opponent. ONE-SIDED /
            # EXPLOIT path (opponent_model set): keep the fixed modeled opponent (best-responding to a
            # KNOWN station is the intended exploitative behaviour).
            if self._two_sided_active:
                strategy = _regret_match(regrets, legal.astype(np.float64))
            else:
                strategy = self._opponent_strategy(env, actor, legal, regrets, key)
            lp = strategy[legal]
            tot = lp.sum()
            if tot <= 1e-12:
                a = int(np.flatnonzero(legal)[0])
            else:
                a = int(self.rng.choice(np.flatnonzero(legal), p=lp / tot))
            return self._cfv(apply_action_np(env, a), hero, depth + 1, t)

        # Traverser (hero) node: branch all (abstracted-)legal actions, CFR+ regret update.
        strategy = _regret_match(regrets, legal.astype(np.float64))
        action_values = np.zeros(NUM_ACTIONS, dtype=np.float64)
        for a in np.flatnonzero(legal):
            action_values[a] = self._cfv(apply_action_np(env, int(a)), hero, depth + 1, t)
        ev = float(np.dot(strategy, action_values))
        for a in np.flatnonzero(legal):
            regrets[a] = max(regrets[a] + action_values[a] - ev, 0.0)
        # BUG-5 (partial) — DELAYED AVERAGING (config.avg_skip_frac): the SERVED average excludes
        # iterations t < _avg_skip_iters (= floor(avg_skip_frac * K), 0 = legacy). Regret updates above
        # are NEVER skipped. The AIVAT root-CFV capture is gated IDENTICALLY so the decision baseline
        # stays CONSISTENT with the served average (its whole design contract).
        if t >= self._avg_skip_iters:
            # CFR+ linear averaging: weight by (t+1) so later (better-converged) iterates dominate the
            # served average. Flat weight 1.0 = legacy uniform average (extraction normalizes by the sum).
            weight = float(t + 1) if self.config.linear_averaging else 1.0
            self._strategy_sum[key] += weight * strategy
            # AIVAT decision-baseline capture (gated; OFF => this block is skipped, hot loop byte-identical).
            # depth==0 is the ROOT decision node the search is solving for; accumulate the per-action CFVs it
            # used, CFR+-weighted exactly like the served strategy, so _leaf_q can return a baseline that is
            # CONSISTENT with the served average policy (the fix for the search-path AIVAT negative-VR bug).
            if depth == 0 and self._aivat_capture:
                if self._root_av_sum is None:
                    self._root_av_sum = weight * action_values.copy()
                else:
                    self._root_av_sum += weight * action_values
                self._root_av_wsum += weight
        return ev

    # ------------------------------------------------------------------ public surface
    def _set_world(self, base_state, sampled_holes: dict[int, tuple[int, int]]):
        """Return a copy of ``base_state`` with sampled opponent holes written into env.holes.

        STALE-DECK FIX (config.repair_deck): after writing the sampled holes, rebuild the UNDEALT deck
        tail from the unseen complement so every world integrates over RANDOM board runouts. Without it an
        all-in line replays the ONE frozen board from base_state in every world/iteration and the search
        best-responds to that single runout (the preflop over-jamming bug). The per-world reshuffle seed is
        a deterministic function of the sampled holes, so a given decision still samples the SAME worlds
        across processes/CRN arms (pure-function contract preserved)."""
        holes = np.asarray(base_state.holes).copy()
        for seat, (c0, c1) in sampled_holes.items():
            holes[seat, 0] = int(c0)
            holes[seat, 1] = int(c1)
        state = base_state._replace(holes=holes)
        if self.config.repair_deck:
            seed = 0
            for s in sorted(sampled_holes):
                c0, c1 = sampled_holes[s]
                seed = (seed * 131 + int(s) * 2704 + int(c0) * 52 + int(c1)) & 0xFFFFFFFF
            state = _reshuffle_undealt(state, np.random.default_rng(seed))
        return state

    def _pushfold_prior(self, base_state, hero: int, legal: np.ndarray) -> np.ndarray | None:
        """Certified ICM push/fold ROOT prior for a shallow preflop spot (tournament N>6 lever), or None.

        Gated on ``pushfold_prior_band_bb`` being set (default None => disabled => byte-identical). Returns
        None whenever the spot is not a certified push/fold one (postflop / limped / multiway opened / deep
        / >1 jam / hole unknown), so the caller keeps its existing blueprint-or-uniform prior unchanged."""
        band = self.config.pushfold_prior_band_bb
        if band is None:
            return None
        from pokerbot.search.pushfold_prior import pushfold_root_prior

        return pushfold_root_prior(
            base_state, hero, legal,
            payouts=(list(self.config.pushfold_payouts)
                     if self.config.pushfold_payouts is not None else None),
            band_bb=band,
        )

    def _root_blueprint(self, base_state, hero: int, legal: np.ndarray) -> np.ndarray:
        # Tournament shallow-stack lever: prefer the certified ICM push/fold range over the (uniform at
        # N>6) blueprint adapter when in the push/fold band. Abstains (None) outside that regime.
        pf = self._pushfold_prior(base_state, hero, legal)
        if pf is not None:
            return pf
        bp = self.adapter.blueprint(base_state, hero) * legal
        s = bp.sum()
        return bp / s if s > 1e-12 else legal.astype(np.float64) / max(legal.sum(), 1)

    def solve(self, root: SubgameRoot, pbs: PBSTracker, cancel_event=None,
              single_table: bool | None = None) -> SolveResult:
        # ``cancel_event`` (optional threading.Event): if set mid-solve (e.g. by a live watchdog or the
        # multitable manager when the table moves on), the iteration loop breaks early and serves the
        # best average so far. None => only the internal time_budget_ms bounds the solve.
        # ``single_table`` (optional): when True the live driver knows exactly ONE table is demanding
        # action, so this solve gets the SINGLE-TABLE iteration floor / extended wall-clock budget (let it
        # converge — single-tabling has spare time). When False/None, the per-table time_budget_ms /
        # max_iterations bind UNCHANGED (no multi-table slowdown). None falls back to the per-solver config
        # default (single_table_iter_floor / single_table_budget_ms gate whether the floor is active at all).
        t0 = time.perf_counter()
        cfg = self.config
        hero = int(root.hero_seat)
        # SINGLE-TABLE ITERATION FLOOR: under single-table play we have spare wall-clock, so let the solve
        # run more iterations / longer before the budget bound fires (converge -> less starvation noise).
        # Multi-table (single_table is False) or floor-disabled (config None) => the plain per-table budgets.
        eff_max_iters = cfg.max_iterations
        eff_budget_ms = cfg.time_budget_ms
        if single_table and cfg.single_table_iter_floor is not None:
            eff_max_iters = max(eff_max_iters, int(cfg.single_table_iter_floor))
        if single_table and cfg.single_table_budget_ms is not None:
            eff_budget_ms = max(eff_budget_ms, int(cfg.single_table_budget_ms))
        # MEASUREMENT-ONLY deterministic starvation: run exactly force_budget_bound iterations (time budget
        # out of the way) and treat the serve as budget-bound at that count. Holds the iteration count fixed
        # across blend ON/OFF arms so an LBR/determinism harness isolates the blend with no wall-clock jitter.
        force_bb = cfg.force_budget_bound
        if force_bb is not None and force_bb > 0:
            eff_max_iters = int(force_bb)
            eff_budget_ms = 10 ** 12  # wall clock never bounds -> exactly eff_max_iters iterations run
        # BUG-5 delayed averaging: strat_sum (and AIVAT root-CFV) accumulation starts at iteration
        # floor(avg_skip_frac * K), K = this solve's iteration target. 0.0 (default) => 0 => legacy.
        self._avg_skip_iters = int(
            float(np.clip(getattr(cfg, "avg_skip_frac", 0.0), 0.0, 1.0)) * eff_max_iters
        )
        # Each decision is an independent depth-limited subgame solve: reset regrets/strategy + the
        # per-solve caches so there is no cross-hand contamination and memory stays bounded over a long eval.
        self._regrets.clear()
        self._strategy_sum.clear()
        self._bp_cache.clear()
        self._leaf_cache.clear()
        self._root_av_sum = None      # reset the AIVAT root-CFV accumulator (per-decision; no carryover)
        self._root_av_wsum = 0.0
        base_state = to_numpy_state(root.env)
        if cfg.per_decision_seed:
            # Fresh stream per decision, seeded from the public state: the solve is a pure function of
            # (public state, opponent model, config) — see PluribusReBeLConfig.per_decision_seed.
            self.rng = np.random.default_rng(self._public_seed(base_state, hero))
        # Opponent-hole sampler stream: its OWN deterministic sub-stream when decouple_sampler (so its
        # variable draw count can't desync the action/rollout stream), else the shared self.rng (identical).
        if cfg.decouple_sampler:
            _hseed = (int(self._public_seed(base_state, hero)) ^ 0x9E3779B97F4A7C15) & ((1 << 64) - 1)
            self._hole_rng = np.random.default_rng(_hseed)
        else:
            self._hole_rng = self.rng
        self._leaf_cont_weights = self._build_leaf_cont_weights(hero)

        # R2 BIASED CONTINUATION LEAVES (config.leaf_biased_continuations, GTO/base only — mirrors the
        # two_sided gating): the exploit path (opponent_model set) keeps its certified leaf, and the
        # model-aware leaf weights (_leaf_cont_weights, which require an opponent model) can never
        # coexist with this flag. False (default) => the legacy leaf, byte-identical.
        self._leaf_biased_active = bool(
            cfg.leaf_biased_continuations
            and self.opponent_model is None
            and self._leaf_cont_weights is None
        )

        # TWO-SIDED SUBGAME SOLVE (config.two_sided, GTO/base only): cycle the traverser across the seats
        # that can still act (SEAT_LIVE) so EVERY player updates regrets and learns (e.g. to fold to a jam)
        # -> a subgame equilibrium instead of the hero one-sidedly best-responding to a fixed opponent. The
        # EXPLOIT path (opponent_model set) stays one-sided by design. OFF => traverser is always the hero
        # (byte-identical to the legacy solve). The hero is always included so its own average still builds.
        self._two_sided_active = bool(cfg.two_sided and self.opponent_model is None)
        if self._two_sided_active:
            _st = np.asarray(base_state.seat_status)
            self._traversers = [int(s) for s in range(_st.shape[0]) if int(_st[s]) == int(SEAT_LIVE)]
            if hero not in self._traversers:
                self._traversers = [hero]
        else:
            self._traversers = [hero]

        # OFF-TREE RE-SOLVE (POKERBOT_OFFTREE_RESOLVE): if the villain we are facing bet a genuinely
        # off-tree size, augment the ROOT hero infoset's action set with the one in-abstraction bucket
        # closest to the observed size (capped at one extra child). Set BEFORE the first _legal() so the
        # root legal mask and every in-loop _legal at the root see the extra child. None => byte-identical
        # to today. Budget fallback: if THIS solver has already been budget-bound on a prior decision,
        # adding a child risks blowing the wall clock -> skip the insert and serve the static Tier-1
        # pseudo-harmonic translation. Postflop only (preflop is served from the blueprint above).
        self._root_extra_key = None
        self._root_extra_post = None
        self._offtree_inserted = False
        if (_OFFTREE_RESOLVE and self._allowed_post is not None
                and int(base_state.street) != 0 and self._budget_bound_count == 0):
            extra = self._offtree_extra_bucket(base_state, hero)
            if extra is not None:
                aug = self._allowed_post.copy()
                aug[BET_BUCKET_START + extra] = True
                self._root_extra_key = self._infoset_key(base_state, hero)
                self._root_extra_post = aug
                self._offtree_inserted = True

        legal = self._legal(base_state, hero)
        if not legal.any():
            probs = np.zeros(NUM_ACTIONS, dtype=np.float32)
            return SolveResult(action_probs=probs, ev=0.0, iterations=0, metadata={})

        # TOURNAMENT SHALLOW-STACK PRIOR: a folded-to-hero / facing-one-jam SHALLOW preflop spot is an
        # exact-game push/fold node. Serve the CERTIFIED ante-aware ICM open-shove / call-vs-jam Nash
        # range (pushfold_prior, reusing tournament.open_shove_icm) directly — at N>6 the blueprint is a
        # miss (uniform) and depth-limited search reasons from that flat prior over a coarse postflop
        # leaf, so the certified push/fold range is strictly the stronger serve here. Gated on
        # ``pushfold_prior_band_bb`` (default None => never fires => byte-identical). Abstains (None)
        # outside the certified push/fold regime, falling through to the existing serve paths below.
        if int(base_state.street) == 0 and self.config.pushfold_prior_band_bb is not None:
            pf = self._pushfold_prior(base_state, hero, legal)
            if pf is not None:
                probs = (pf * legal)
                probs = probs / max(probs.sum(), 1e-12)
                return SolveResult(
                    action_probs=probs.astype(np.float32), ev=0.0, iterations=0,
                    metadata={"served": "pushfold_icm_prior",
                              "eff_stack_bb": round(self._effective_stack_bb(base_state), 1)},
                )

        # Pluribus: preflop served directly from the blueprint (no search) — but ONLY when the table is
        # within the blueprint's trained depth band (A1). Off-band (deep/short), fall through to the
        # depth-general search so we never serve an off-depth 100bb preflop range.
        if int(base_state.street) == 0 and not cfg.search_preflop:
            band = cfg.preflop_blueprint_band_bb
            eff_bb = self._effective_stack_bb(base_state)
            if band is None or band[0] <= eff_bb <= band[1]:
                probs = self._root_blueprint(base_state, hero, legal).astype(np.float32)
                return SolveResult(
                    action_probs=probs, ev=0.0, iterations=0,
                    metadata={"served": "blueprint_preflop", "eff_stack_bb": round(eff_bb, 1)},
                )
            # else: off-depth preflop -> depth-general search (pot-relative buckets + real-stack leaf).

        # NATIVE FAST PATH (opt-in): run the entire CFR subgame solve in Rust (one call, no per-node
        # EnvState marshaling). Gated on opponent_model is None (firewall: the exploit tail stays in
        # Python) and the model-aware leaf being off; the native solve samples opponent holes under a
        # UNIFORM belief (the eval base arm). Statistically equivalent to the Python loop below.
        #
        # LIVE SEMANTICS (2026-07-02 live native wiring):
        #   * ITERATIONS: the native call runs ``eff_max_iters`` — cfg.max_iterations everywhere it
        #     always did (eval/battery callers never pass single_table), PLUS the live single-table
        #     convergence floor (single_table=True) and the measurement-only force_budget_bound count,
        #     both of which the old ``cfg.max_iterations`` pass-through silently ignored.
        #   * CANCEL/BUDGET FIT: each native call is a non-interruptible FFI call — cancel_event and
        #     the wall clock cannot fire INSIDE it. Under a live-scale budget (eff_budget_ms below
        #     _NATIVE_FIT_BUDGET_MS) the solve is therefore split into a small PROBE call (K=200,
        #     measures this root's per-iteration cost — subgame size varies ~20x across spots) and one
        #     MAIN call whose iteration count is fitted to the remaining budget (capped at
        #     eff_max_iters). Both calls use the SAME public seed, so probe+main is statistically a
        #     single K_main-iteration solve (the main call replays the probe's stream prefix); the
        #     cancel event is checked between the calls. Worst-case wall = ~eff_budget_ms + rate noise;
        #     worst-case non-cancellable window = one call <= ~the remaining budget. With a huge /
        #     absent budget (eval, batteries, measurement: >= _NATIVE_FIT_BUDGET_MS) the fit is skipped
        #     and the single full-K call is byte-identical to the pre-wiring behavior.
        #     NOTE: a budget-fitted native serve (K_main < eff_max_iters) reports budget_bound=True but
        #     deliberately does NOT trigger the starvation blueprint blend — the blend exists for the
        #     ~tens-of-iterations Python starved average; a fitted native solve is >= the probe K (200),
        #     far past the noise regime the blend was built for.
        if (self.opponent_model is None and self._leaf_cont_weights is None
                and native_subgame_enabled() and not self._offtree_inserted
                and not self._aivat_capture):  # AIVAT capture needs the Python loop's per-action root CFVs
            from pokerbot.engine.jax_weva_encode import _POSTFLOP_NUM_BUCKETS
            from pokerbot.search import native_solver as _ns

            _pub_seed = self._public_seed(base_state, hero)

            def _native_call(_k: int):
                return _ns.native_subgame_solve(
                    base_state, hero, self.adapter,
                    public_seed=_pub_seed,
                    max_iterations=int(_k), max_depth=cfg.max_depth,
                    blueprint_mix=cfg.blueprint_mix,
                    rollout_samples=cfg.leaf.rollout_samples,
                    resample_board=cfg.leaf.resample_board,
                    max_playout_depth=cfg.leaf.max_playout_depth,
                    num_buckets=int(_POSTFLOP_NUM_BUCKETS),
                    linear_averaging=bool(cfg.linear_averaging),
                    repair_deck=bool(cfg.repair_deck),
                    # two_sided is GTO-only; this branch is already gated on opponent_model is None, so
                    # mirror _two_sided_active = bool(cfg.two_sided and opponent_model is None) exactly.
                    two_sided=bool(cfg.two_sided and self.opponent_model is None),
                    # Q2/Q3 world-sampling levers (native): decouple the hole sampler + range-aware
                    # weighting.
                    decouple_sampler=bool(cfg.decouple_sampler),
                    range_aware=bool(cfg.range_aware),
                    range_uniform_floor=float(cfg.range_uniform_floor),
                    # ENSEMBLE: N independent parallel CFR streams merged at the root (native rayon
                    # fan-out).
                    ensemble_streams=int(cfg.ensemble_streams),
                    # BUG-1 leaf hole-world integration (1 = legacy first-touch freeze) + BUG-5 delayed
                    # averaging (0.0 = legacy). Native mirrors the Python semantics exactly; both no-op
                    # at the defaults.
                    leaf_cache_worlds=int(cfg.leaf_cache_worlds),
                    avg_skip_frac=float(cfg.avg_skip_frac),
                    # R2 biased continuation leaves (GTO-only; this branch is gated on opponent_model
                    # is None, mirroring _leaf_biased_active). False (default) => the legacy uniform
                    # native leaf, byte-identical.
                    leaf_biased_continuations=bool(cfg.leaf_biased_continuations
                                                   and self.opponent_model is None),
                    leaf_bias_mult=float(cfg.leaf_bias_mult),
                )

            _fit = eff_budget_ms < _NATIVE_FIT_BUDGET_MS
            k_probe = min(_NATIVE_FIT_PROBE_ITERS, int(eff_max_iters)) if _fit else int(eff_max_iters)
            probs_n = _native_call(k_probe)
            iters_used = int(k_probe)
            if probs_n is not None and _fit and k_probe < int(eff_max_iters):
                probe_ms = (time.perf_counter() - t0) * 1000.0
                # Cancel between the FFI calls (the audit-fixed live cancel wiring): if the user already
                # acted, serve the probe solve (K>=200 — still far better-converged than the ~tens of
                # iterations the cancelled Python loop would have banked).
                if not (cancel_event is not None and cancel_event.is_set()):
                    rate = max(probe_ms / float(k_probe), 1e-6)  # ms per iteration AT THIS ROOT
                    remaining_ms = float(eff_budget_ms) - probe_ms
                    k_main = min(int(eff_max_iters), int((remaining_ms * 0.9) / rate))
                    if k_main > k_probe:
                        probs_full = _native_call(k_main)
                        if probs_full is not None:
                            probs_n, iters_used = probs_full, int(k_main)
            if probs_n is not None:
                probs = probs_n * legal
                probs = probs / max(probs.sum(), 1e-12)
                budget_ms = (time.perf_counter() - t0) * 1000.0
                # Budget visibility (S2 parity with the Python loop): the wall spent must be visible to
                # the live decision log (the coach reads budget_used_ms). A native serve that EXCEEDS
                # its wall budget despite the fit is a loud one-time warning: the calls are
                # non-interruptible, so an overrun is the operator's cue to lower the iteration target
                # (or investigate CPU contention), not a silent latency regression.
                if budget_ms >= eff_budget_ms * 1.5 and not self._native_over_budget_warned:
                    self._native_over_budget_warned = True
                    print(f"[search] NATIVE solve exceeded its wall budget: {budget_ms:.0f}ms >= "
                          f"1.5x {eff_budget_ms}ms at K={iters_used} (non-interruptible FFI calls) — "
                          f"lower POKERBOT_SEARCH_MAX_ITERS or investigate CPU contention", flush=True)
                return SolveResult(
                    action_probs=probs.astype(np.float32), ev=0.0,
                    iterations=iters_used,
                    metadata={"served": "search_native",
                              "eff_max_iters": int(eff_max_iters),
                              "single_table": bool(single_table) if single_table is not None else None,
                              "budget_used_ms": budget_ms,
                              # True iff the budget fit served fewer iterations than the target (never
                              # triggers the blueprint blend on this path — see the header note).
                              "budget_bound": bool(iters_used < int(eff_max_iters))},
                )

        # NATIVE EXPLOIT FAST PATH (opt-in, POKERBOT_NATIVE_EXPLOIT): the exploit analogue of the GTO
        # native path. Engaged ONLY on the PURE frozen-read branch — opponent_model set AND
        # config.frozen_read_response (so _opponent_strategy is a pure function of public state) AND a
        # frozen table builds AND no off-tree insert (the augmented root mask is not modeled native-side).
        # The exploit MATH runs in Python (exploit_table.build_frozen_table -> context_read pipeline);
        # native only looks up the precomputed frozen vector + the per-actor rnr scalar and applies the
        # documented overwrite / blend (firewall). When AIVAT capture is armed, native also returns the
        # root per-action CFVs, so the AIVAT gate (``not self._aivat_capture``) is LIFTED here — the
        # chance-term v stays Python (RNG-gadget boundary), only the decision baseline comes from native.
        # The non-pure tails (frozen_read_response False) are NEVER native: the condition below excludes
        # them so the Python loop runs unchanged.
        #
        # LEAF MODEL: like the native GTO path, this uses the native UNIFORM continuation-gadget leaf
        # (it does NOT apply the model-aware per-seat continuation weights `_leaf_cont_weights`). The
        # opponent-NODE frozen deviation — the dominant exploit lever — is applied exactly; the leaf
        # world is the uniform base arm. This is the statistical-equivalence target: the cert compares
        # native-exploit vs the Python `_cfv` exploit baseline run with the SAME (uniform) leaf. The flag
        # is default-OFF, so the default live path (with the model-aware leaf) is unchanged.
        if (self.opponent_model is not None
                and native_exploit_enabled() and bool(cfg.frozen_read_response)
                and not self._offtree_inserted):
            from pokerbot.engine.jax_weva_encode import _POSTFLOP_NUM_BUCKETS
            from pokerbot.search import exploit_table as _et
            from pokerbot.search import native_solver as _ns

            _tb0 = time.perf_counter()
            ftab = _et.build_frozen_table(self, base_state, hero) if _ns.native_subgame_exploit_available() else None
            _table_build_ms = (time.perf_counter() - _tb0) * 1000.0
            if ftab is not None:
                # LIVE BUDGET FIT (2026-07-02 exploit wiring, mirrors the GTO native path above): the
                # native exploit solve is one non-interruptible FFI call, so under a live-scale wall
                # budget (eff_budget_ms < _NATIVE_FIT_BUDGET_MS) split it into a K=200 probe that
                # measures THIS root's per-iteration cost (the probe wall also absorbs the one-time
                # Python table build above — conservative, fits FEWER main iterations) and one main
                # call fitted to the remaining budget (same public seed => statistically one longer
                # solve; cancel checked between the calls). Eval/battery/cert callers use huge budgets
                # (>= the threshold) so the fit is skipped and the single full-K call is byte-identical
                # to the pre-wiring behavior. eff_max_iters (not cfg.max_iterations) so the live
                # single-table convergence floor reaches the exploit path exactly like the GTO path.
                _pub_seed = self._public_seed(base_state, hero)

                def _native_exploit_call(_k: int):
                    return _ns.native_subgame_solve_exploit(
                        base_state, hero, self.adapter,
                        public_seed=_pub_seed,
                        max_iterations=int(_k), max_depth=cfg.max_depth,
                        blueprint_mix=cfg.blueprint_mix,
                        rollout_samples=cfg.leaf.rollout_samples,
                        resample_board=cfg.leaf.resample_board,
                        max_playout_depth=cfg.leaf.max_playout_depth,
                        num_buckets=int(_POSTFLOP_NUM_BUCKETS),
                        linear_averaging=bool(cfg.linear_averaging),
                        frozen_table=ftab, rnr=ftab.rnr,
                        frozen_read_response=bool(ftab.frozen_read_response),
                        rnr_restrict=bool(cfg.rnr_restrict),
                        want_root_q=bool(self._aivat_capture),
                        # repair_deck applies to the exploit path too (Python _set_world reshuffles for
                        # every solve); two_sided stays OFF here (firewall: exploit is one-sided).
                        repair_deck=bool(cfg.repair_deck),
                        # Q2/Q3 world-sampling levers (native): they change ONLY the sampled worlds, not
                        # the frozen-read exploit math (firewall-safe).
                        decouple_sampler=bool(cfg.decouple_sampler),
                        range_aware=bool(cfg.range_aware),
                        range_uniform_floor=float(cfg.range_uniform_floor),
                        # NOTE: leaf_cache_worlds / avg_skip_frac / leaf_biased_continuations (+ its
                        # leaf_bias_mult) are deliberately NOT forwarded — the native solve_exploit
                        # hard-codes their no-op values and its FFI does not even accept them (firewall:
                        # the certified exploit serve is byte-identical regardless of those GTO-path knobs).
                    )

                _fit = eff_budget_ms < _NATIVE_FIT_BUDGET_MS
                k_probe = min(_NATIVE_FIT_PROBE_ITERS, int(eff_max_iters)) if _fit else int(eff_max_iters)
                out = _native_exploit_call(k_probe)
                iters_used = int(k_probe)
                if out is not None and _fit and k_probe < int(eff_max_iters):
                    probe_ms = (time.perf_counter() - t0) * 1000.0
                    # Cancel between the FFI calls (audit-fixed live cancel wiring): if the user already
                    # acted, serve the probe solve (K>=200 beats the ~tens of Python-loop iterations a
                    # cancelled dynamic solve would have banked).
                    if not (cancel_event is not None and cancel_event.is_set()):
                        rate = max(probe_ms / float(k_probe), 1e-6)  # ms/iter AT THIS ROOT (+ table amort.)
                        remaining_ms = float(eff_budget_ms) - probe_ms
                        k_main = min(int(eff_max_iters), int((remaining_ms * 0.9) / rate))
                        if k_main > k_probe:
                            out_full = _native_exploit_call(k_main)
                            if out_full is not None:
                                out, iters_used = out_full, int(k_main)
                if out is not None:
                    probs_n, root_q = out
                    probs = probs_n * legal
                    probs = probs / max(probs.sum(), 1e-12)
                    # AIVAT: feed the native root-CFV average into the decision baseline (so _leaf_q
                    # returns it). _root_av_wsum=1.0 because root_q is ALREADY the CFR+-weighted average.
                    if self._aivat_capture and root_q is not None:
                        self._root_av_sum = np.asarray(root_q, dtype=np.float64)
                        self._root_av_wsum = 1.0
                    budget_ms = (time.perf_counter() - t0) * 1000.0
                    if budget_ms >= eff_budget_ms * 1.5 and not self._native_over_budget_warned:
                        self._native_over_budget_warned = True
                        print(f"[search] NATIVE exploit solve exceeded its wall budget: "
                              f"{budget_ms:.0f}ms >= 1.5x {eff_budget_ms}ms at K={iters_used} "
                              f"(non-interruptible FFI calls; table build {_table_build_ms:.0f}ms) — "
                              f"lower POKERBOT_SEARCH_MAX_ITERS or investigate CPU contention",
                              flush=True)
                    return SolveResult(
                        action_probs=probs.astype(np.float32), ev=0.0,
                        iterations=iters_used,
                        metadata={"served": "search_native_exploit",
                                  "eff_max_iters": int(eff_max_iters),
                                  "single_table": bool(single_table) if single_table is not None else None,
                                  "budget_used_ms": budget_ms,
                                  "table_build_ms": _table_build_ms,
                                  # True iff the budget fit served fewer iterations than the target
                                  # (never triggers the blueprint blend — same rationale as the GTO
                                  # native path header note).
                                  "budget_bound": bool(iters_used < int(eff_max_iters))},
                    )

        root_key = self._infoset_key(base_state, hero)
        iterations = 0
        budget_bound = False
        while iterations < eff_max_iters:
            if cancel_event is not None and cancel_event.is_set():
                break
            if (time.perf_counter() - t0) * 1000.0 >= eff_budget_ms:
                # Wall-clock budget bound before max_iterations -> the realized average strategy is
                # machine-speed dependent, breaking the "pure function of public state" determinism/CRN
                # contract. Count it PER SOLVER (not a one-shot process flag) and surface the count in
                # metadata so the live decision log sees how often decisions are non-reproducible under
                # load (set a high time_budget_ms / drive purely by max_iterations for a deterministic serve).
                budget_bound = True
                self._budget_bound_count += 1
                if self._budget_bound_count == 1:
                    print(f"[search] time_budget_ms={eff_budget_ms} bound at {iterations}/"
                          f"{eff_max_iters} iters — decisions non-reproducible under load "
                          f"(per-solver count tracked in SolveResult.metadata)", flush=True)
                break
            sampled_holes = pbs.sample_opponent_holes(self._hole_rng)
            world = self._set_world(base_state, sampled_holes)
            # Traverser cycles across live seats when two-sided (== hero when one-sided/exploit). The hero's
            # served average (strategy_sum[root_key]) builds only on hero-traverser iterations, so two-sided
            # needs proportionally more iterations for the same hero convergence (native makes that cheap).
            traverser = self._traversers[iterations % len(self._traversers)]
            self._cfv(world, traverser, 0, iterations)
            iterations += 1

        ss = self._strategy_sum.get(root_key)
        if ss is not None and ss.sum() > 1e-12:
            avg = ss * legal
            probs = avg / max(avg.sum(), 1e-12)
        else:
            probs = _regret_match(self._regrets.get(root_key, np.zeros(NUM_ACTIONS)), legal.astype(np.float64))
        probs = probs * legal
        probs = probs / max(probs.sum(), 1e-12)

        # STARVATION-ROBUST SERVE: when this solve was BUDGET-BOUND (cut short before eff_max_iters), the
        # average above is under-converged and its argmax flips frame-to-frame under load. Blend the served
        # ROOT policy toward the trusted root blueprint, the blueprint weight scaling with the convergence
        # SHORTFALL (shortfall = 1 - iters/eff_max_iters in [0,1]; more starved => lean more on the prior).
        # Proper convex combination over the LEGAL mask, renormalized. A fully-converged solve never enters
        # this branch (budget_bound False) => the pure search policy is served byte-identically. Gated by
        # _blend_active() so the eval / LBR battery (env OFF) is provably inert. The exploit tail already
        # shaped ``probs`` at the opponent nodes; this blend is at hero's ROOT serve only and pulls toward
        # the GTO blueprint floor (it can only REDUCE exploitability vs the noisy starved average).
        blueprint_weight = 0.0
        # Measurement mode: a forced fixed-iteration solve is treated as budget-bound at that count, so the
        # blend fires deterministically (shortfall relative to the real max_iterations target).
        served_budget_bound = budget_bound or (force_bb is not None and force_bb > 0)
        if served_budget_bound and self._blend_active():
            target_iters = cfg.max_iterations if (force_bb is not None and force_bb > 0) else eff_max_iters
            shortfall = 1.0 - (float(iterations) / float(target_iters)) if target_iters > 0 else 1.0
            shortfall = float(np.clip(shortfall, 0.0, 1.0))
            blueprint_weight = float(np.clip(self.config.blueprint_blend_max, 0.0, 1.0)) * shortfall
            if blueprint_weight > 1e-9:
                bp_root = self._root_blueprint(base_state, hero, legal)
                blended = (1.0 - blueprint_weight) * probs + blueprint_weight * bp_root
                blended = blended * legal  # convex combo of two legal-supported dists stays legal-supported
                s = blended.sum()
                probs = blended / s if s > 1e-12 else probs

        budget_ms = (time.perf_counter() - t0) * 1000.0
        return SolveResult(
            action_probs=probs.astype(np.float32),
            ev=0.0,
            iterations=iterations,
            metadata={
                "leaf_eval_count": float(self._leaf_eval_count),
                "leaf_cache_hits": float(self._leaf_cache_hits),
                "blueprint_blend_weight": float(blueprint_weight),
                "eff_max_iters": int(eff_max_iters),
                "single_table": bool(single_table) if single_table is not None else None,
                "budget_used_ms": budget_ms,
                "budget_bound": budget_bound,
                "budget_bound_count": float(self._budget_bound_count),
                "offtree_inserted": bool(self._offtree_inserted),
                "served": "search",
            },
        )

    # ------------------------------------------------------------------ AIVAT baseline
    def _leaf_q(self, env) -> np.ndarray:
        """Per-action AIVAT baseline at the just-solved ROOT decision (NOT used in the CFR hot loop).

        PREFERRED (the fix): when the solve captured the ROOT per-action CFVs (``_aivat_capture`` armed),
        return their CFR+-weighted average -- the values the search ACTUALLY used to form the served
        strategy, so the baseline is CONSISTENT with the served policy and AIVAT's decision term is a
        POSITIVE variance reducer. FALLBACK (capture off, or a non-search serve such as preflop-blueprint
        / pushfold where no CFR loop ran): the legacy 1-ply continuation-gadget value of each child --
        which is inconsistent with the multi-ply served strategy and gave NEGATIVE VR on the search path
        (kept only as a graceful fallback; never the poisoned value net). Illegal entries are 0; hero is
        the acting seat at ``env``. Accepts a JAX or numpy env (AIVAT capture passes the JAX ``root.env``)."""
        if self._root_av_sum is not None and self._root_av_wsum > 0.0:
            return self._root_av_sum / self._root_av_wsum
        env = to_numpy_state(env)
        hero = int(env.action_seat)
        out = np.zeros(NUM_ACTIONS, dtype=np.float64)
        if hero < 0:
            return out
        legal = self._legal(env, hero)
        for a in np.flatnonzero(legal):
            child = apply_action_np(env, int(a))
            out[a] = self._leaf_value(child, hero)
        return out
