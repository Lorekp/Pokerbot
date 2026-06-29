"""Depth-limited TurboReBeL subgame solver with SSMIG and batched leaf evaluation."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import jax.numpy as jnp
import numpy as np

from pokerbot.engine.jax_engine import (
    NUM_ACTIONS,
    NUM_PLAYERS,
    SEAT_FOLDED,
    encode_history_state,
    encode_info_state,
    legal_action_mask,
    step_state,
)
from pokerbot.opponent.bayes_exploit import DEFAULT_MAX_ALPHA as _CTX_MAX_ALPHA
from pokerbot.opponent.bayes_exploit import context_read
from pokerbot.opponent.rnr_cap import cap_eff as _rnr_cap_eff  # RNR restriction of the solver BR (#28)
from pokerbot.opponent.rnr_cap import fish_relaxed_n as _rnr_fish_relaxed_n  # no-punisher wipe-floor relax
from pokerbot.opponent.context import context_key_rs as _ctx_key_rs
from pokerbot.opponent.context import is_aggressive_action as _is_aggressive_action
from pokerbot.search.isomorphism import cached_canonicalize_key, canonicalize_history_encoding
from pokerbot.search.leaf_evaluator import HistoryValueEvaluator
from pokerbot.search.pbs_tracker import PBSTracker
from pokerbot.search.state_bridge import SubgameRoot, env_to_numpy
from pokerbot.training.neural_cfr import AverageStrategyNet


@dataclass
class TurboReBeLConfig:
    max_depth: int = 4
    max_iterations: int = 200
    ssmig_inner_iters: int = 10
    time_budget_ms: int = 2000
    leaf_batch_size: int = 64
    ssmig_resample_every: int = 1
    blueprint_mix: float = 0.5
    # RNR restriction of the solver-internal BR (#28) — see PluribusReBeLConfig.rnr_restrict. When True,
    # the opponent-node deviation cap is bounded by cap_eff(stability, n) (the caller sets
    # ``opponent_stability`` / ``opponent_n_live``). Default OFF => byte-identical (legacy flat cap).
    rnr_restrict: bool = False


@dataclass
class SolveResult:
    action_probs: np.ndarray
    ev: float
    iterations: int
    metadata: dict[str, float] = field(default_factory=dict)


def _regret_match(regrets: np.ndarray, legal: np.ndarray) -> np.ndarray:
    pos = np.maximum(regrets, 0.0) * legal
    s = pos.sum()
    if s > 0:
        return pos / s
    return legal.astype(np.float64) / max(legal.sum(), 1)


def _blend_bias(strategy: np.ndarray, tilt: np.ndarray, alpha: float, legal: np.ndarray) -> np.ndarray:
    """Nash-anchored safe-mix tail shared by the archetype and contextual exploit paths:
    ``mixed = (1-alpha)*GTO + alpha*normalize(GTO * tilt)``, renormalized over legal actions.
    ``alpha`` (confidence-gated upstream) bounds the exploitability exposed; alpha=0 ⇒ unchanged."""
    n = strategy.shape[0]
    t = np.asarray(tilt, dtype=np.float64)[:n]
    biased = strategy * t * legal
    s = biased.sum()
    if s <= 0:
        return strategy
    biased = biased / s
    mixed = ((1.0 - float(alpha)) * strategy + float(alpha) * biased) * legal
    ms = mixed.sum()
    return mixed / ms if ms > 0 else strategy


def apply_opponent_bias(
    strategy: np.ndarray, stats, legal: np.ndarray, *, max_alpha: float = 0.5
) -> np.ndarray:
    """Tilt a regret-matched/blueprint OPPONENT strategy toward their observed ARCHETYPE (the
    fallback when no per-context Bayesian read is available), so the hero best-response becomes
    exploitative. Nash-anchored + sample-size-confidence-gated: 'unknown'/'tag'/sparse ⇒ unchanged.
    Action indices: 0=fold, 2=call, 3+=bet/raise/all-in."""
    arch = stats.archetype()
    conf = min(1.0, max(0, stats.hands) / 50.0)
    if arch in ("unknown", "tag") or conf <= 0.0:
        return strategy
    # Data-proportional exploitation: deviate MORE the more exploitable the read actually is (a
    # 2%-VPIP hyper-nit is punished harder than a borderline one). extremity in [0.3, 1.0].
    if arch == "nit":
        extremity = (0.15 - stats.vpip_pct) / 0.15
    elif arch == "calling_station":
        extremity = (stats.vpip_pct - 0.45) / 0.45
    elif arch == "lag":
        extremity = (stats.aggression_factor - 2.5) / 3.0
    else:
        extremity = 0.0
    extremity = float(min(1.0, max(0.3, extremity + 0.3)))
    n = strategy.shape[0]
    tilt = np.ones(n, dtype=np.float64)
    fold_i, call_i = 0, 2
    agg = np.zeros(n, dtype=bool)
    if n > 3:
        agg[3:] = True  # bet / raise / all-in
    if arch == "nit":  # folds too much -> we should apply more pressure
        tilt[fold_i] *= 1.8
        tilt[agg] *= 0.7
    elif arch == "calling_station":  # won't fold, calls too much -> value-bet, don't bluff
        tilt[call_i] *= 1.8
        tilt[fold_i] *= 0.4
        tilt[agg] *= 0.85
    elif arch == "lag":  # over-aggressive -> let them barrel, trap
        tilt[agg] *= 1.5
        tilt[call_i] *= 1.2
        tilt[fold_i] *= 0.8
    return _blend_bias(strategy, tilt, max_alpha * conf * extremity, legal)


def apply_context_bias(
    strategy: np.ndarray, read, legal: np.ndarray, *, max_alpha: float = _CTX_MAX_ALPHA
) -> np.ndarray:
    """Tilt the OPPONENT strategy from a per-context Bayesian read (bayes_exploit.ContextRead). The
    read's posterior-vs-population deviation drives a per-action tilt; its per-context confidence
    drives alpha. Strictly preferred over the archetype path when contextual data exists."""
    tilt, alpha = read.search_tilt(max_alpha=max_alpha)
    return _blend_bias(strategy, tilt, alpha, legal)


_CTX_BOARD_LEN = (0, 3, 4, 5)


def _context_key_from_env(env, actor: int) -> str:
    """Serve-side per-context key from the live env — the SAME spec as offline ingestion
    (pokerbot.opponent.context), so the DB lookup matches what was recorded. Card id ⇒ rank/suit via
    the engine convention (rank = id//4 + 2, suit = id%4)."""
    street = int(np.asarray(env.street))
    n_board = _CTX_BOARD_LEN[street] if 0 <= street < len(_CTX_BOARD_LEN) else 0
    board = np.asarray(env.board)
    ranks, suits = [], []
    for i in range(n_board):
        c = int(board[i])
        if c < 0:
            continue
        ranks.append(c // 4 + 2)
        suits.append(c % 4)
    n_aggr = _facing_count_from_env(env, street, actor)
    return _ctx_key_rs(street, ranks, suits, n_aggr)


def _facing_count_from_env(env, street: int, actor: int) -> int:
    """Raw per-street aggression count for the facing bucket, computed IDENTICALLY to offline ingest.
    Preflop the whole flat history is the preflop action sequence ⇒ count its aggressive entries
    (full open/3bet/4bet depth). Postflop the flat env can't recover raise-depth without replay, so
    return the binary to_call signal (``context.facing_token`` caps postflop at 1 anyway)."""
    if street == 0:
        hist_len = int(np.asarray(env.hist_len))
        actions = np.asarray(env.hist_action)[:hist_len]
        return int(sum(1 for a in actions if _is_aggressive_action(int(a))))
    to_call = int(np.asarray(env.current_bet)) - int(np.asarray(env.committed_street)[actor])
    return 0 if to_call <= 0 else 1


class TurboReBeLSolver:
    """Depth-limited CFR+ with single-sample multi-iteration generation."""

    def __init__(
        self,
        leaf_eval: HistoryValueEvaluator,
        config: TurboReBeLConfig | None = None,
        rng: np.random.Generator | None = None,
        opponent_model: dict | None = None,
    ) -> None:
        self.leaf_eval = leaf_eval
        self.config = config or TurboReBeLConfig()
        self.rng = rng or np.random.default_rng(0)
        # Optional seat -> OpponentStats for opponent-conditioned safe exploitation. None (default)
        # => the solve assumes regret-matched/blueprint opponents (unchanged). The exploitation bias
        # that consumes this is built later (see docs/exploitation-design.md).
        self.opponent_model = opponent_model
        # RNR restriction inputs (#28), set by the caller when config.rnr_restrict is on (see
        # PluribusReBeLSolver). None => stability 1.0 / N inferred from the model size.
        self.opponent_stability: dict | None = None
        self.opponent_n_live: int | None = None
        # Optional GiftRegister (pokerbot.opponent.safety_budget): banked opponent gifts relax the
        # per-context alpha cap (gift-modulated bounded-ε). None ⇒ the static Pmax cap (unchanged).
        self.gift_register = None
        self._regrets: dict[str, np.ndarray] = {}
        self._strategy_sum: dict[str, np.ndarray] = {}
        self._leaf_cache: dict[str, np.ndarray] = {}
        self._pending_leaf_keys: list[str] = []
        self._pending_leaf_hists: list[np.ndarray] = []
        self._leaf_eval_count = 0
        self._leaf_cache_hits = 0
        self._frozen_opp_avg: dict[str, np.ndarray] = {}

        self._blueprint_module = None
        self._blueprint_params = None
        if getattr(leaf_eval, "blueprint_params", None) is not None:
            hidden = int(getattr(leaf_eval, "hidden_dim", 256))
            self._blueprint_module = AverageStrategyNet(hidden)
            self._blueprint_params = leaf_eval.blueprint_params

    def _infoset_key(self, env, seat: int) -> str:
        hist_len = int(np.asarray(env.hist_len))
        seats = tuple(int(x) for x in np.asarray(env.hist_seat[:hist_len]))
        actions = tuple(int(x) for x in np.asarray(env.hist_action[:hist_len]))
        board = tuple(int(x) for x in np.asarray(env.board) if int(x) >= 0)
        pot = float(np.asarray(env.pot_dead + jnp.sum(env.committed_street)))
        stacks = np.asarray(env.stacks, dtype=np.float64)
        folded = np.asarray(env.seat_status == SEAT_FOLDED, dtype=bool)
        active_mean = float(stacks[~folded].mean()) if (~folded).any() else float(stacks.mean())
        spr_bucket = int(pot / max(active_mean, 1.0) * 4)
        return f"{seat}|{int(np.asarray(env.street))}|{board}|{seats}|{actions}|spr{spr_bucket}"

    def _get_regrets(self, key: str, legal: np.ndarray) -> np.ndarray:
        if key not in self._regrets:
            self._regrets[key] = np.zeros(NUM_ACTIONS, dtype=np.float64)
            self._strategy_sum[key] = np.zeros(NUM_ACTIONS, dtype=np.float64)
        return self._regrets[key]

    def _leaf_cache_key(self, env) -> str:
        hist_len = int(np.asarray(env.hist_len))
        action_seq = tuple(int(x) for x in np.asarray(env.hist_action[:hist_len]))
        board = tuple(int(x) for x in np.asarray(env.board) if int(x) >= 0)
        street = int(np.asarray(env.street))
        pot = float(np.asarray(env.pot_dead + jnp.sum(env.committed_street)))
        stacks = np.asarray(env.stacks, dtype=np.float64)
        folded = np.asarray(env.seat_status == SEAT_FOLDED, dtype=bool)
        active_mean = float(stacks[~folded].mean()) if (~folded).any() else float(stacks.mean())
        pot_bucket = int(pot * 2)
        spr_bucket = int(pot / max(active_mean, 1.0) * 4)
        return cached_canonicalize_key(board, street, action_seq + (pot_bucket, spr_bucket))

    def _queue_leaf(self, env) -> str:
        cache_key = self._leaf_cache_key(env)
        if cache_key in self._leaf_cache:
            self._leaf_cache_hits += 1
            return cache_key
        hist = np.asarray(encode_history_state(env), dtype=np.float32)
        pot = float(np.asarray(env.pot_dead + jnp.sum(env.committed_street)))
        stacks = np.asarray(env.stacks, dtype=np.float64)
        folded = np.asarray(env.seat_status == SEAT_FOLDED, dtype=bool)
        active_mean = float(stacks[~folded].mean()) if (~folded).any() else float(stacks.mean())
        spr_bucket = int(pot / max(active_mean, 1.0) * 4)
        hist = canonicalize_history_encoding(hist, pot_bb=pot, spr_bucket=spr_bucket)
        self._pending_leaf_keys.append(cache_key)
        self._pending_leaf_hists.append(hist)
        if len(self._pending_leaf_hists) >= self.config.leaf_batch_size:
            self._flush_leaf_batch()
        return cache_key

    def _flush_leaf_batch(self) -> None:
        if not self._pending_leaf_hists:
            return
        batch = np.stack(self._pending_leaf_hists, axis=0)
        qs = self.leaf_eval.evaluate_batch(batch)
        for key, q in zip(self._pending_leaf_keys, qs, strict=True):
            self._leaf_cache[key] = q
            self._leaf_eval_count += 1
        self._pending_leaf_keys.clear()
        self._pending_leaf_hists.clear()

    def _leaf_q(self, env) -> np.ndarray:
        cache_key = self._queue_leaf(env)
        if cache_key in self._leaf_cache:
            return self._leaf_cache[cache_key]
        self._flush_leaf_batch()
        return self._leaf_cache[cache_key]

    def _rnr_n_live(self) -> int:
        """Table size N for the RNR ceiling (rnr_cap.cap_eff). Caller-set live-seat count, else inferred
        from the model size (villains + hero), clamped to [2, 6]. Mirrors PluribusReBeLSolver._rnr_n_live."""
        if self.opponent_n_live is not None:
            n_full = max(2, int(self.opponent_n_live))
        elif self.opponent_model:
            n_full = max(2, min(6, len(self.opponent_model) + 1))
        else:
            return 2
        # Optionally relax to the HU ceiling on a NO-PUNISHER (all-confirmed-fish) table (default-OFF env).
        return _rnr_fish_relaxed_n(n_full, self.opponent_model)

    def _opponent_strategy(
        self,
        env,
        actor: int,
        legal: np.ndarray,
        regrets: np.ndarray,
        *,
        use_frozen: bool = False,
    ) -> np.ndarray:
        key = self._infoset_key(env, actor)
        if use_frozen and key in self._frozen_opp_avg:
            strategy = self._frozen_opp_avg[key].copy() * legal
            s = strategy.sum()
            if s > 0:
                return strategy / s

        strategy = _regret_match(regrets, legal)
        if self._blueprint_module is not None and self._blueprint_params is not None:
            info = encode_info_state(env, jnp.int32(actor))
            mask = legal_action_mask(env, jnp.int32(actor))
            bp = np.asarray(self._blueprint_module.apply(self._blueprint_params, info, mask))
            mix = float(self.config.blueprint_mix)
            strategy = (1.0 - mix) * strategy + mix * bp * legal
            s = strategy.sum()
            if s > 0:
                strategy = strategy / s
            else:
                strategy = legal.astype(np.float64) / max(legal.sum(), 1)
        # Opponent-conditioned safe exploitation: bias the assumed opponent strategy toward their
        # persistent-DB read. Prefer the per-context Bayesian posterior (DBR, confidence-gated);
        # fall back to the discrete archetype when that context has no data. None / no-profile /
        # low-confidence => unchanged (GTO).
        if self.opponent_model is not None:
            stats = self.opponent_model.get(actor)
            if stats is not None:
                read = context_read(stats, _context_key_from_env(env, actor))
                if read is not None:
                    # Gift-modulated bounded-ε: a banked-gift budget relaxes the Pmax cap for THIS
                    # opponent (exploit demonstrable leakers harder). None / zero gift ⇒ base cap.
                    max_alpha = _CTX_MAX_ALPHA
                    if self.gift_register is not None:
                        max_alpha = self.gift_register.capped_max_alpha(actor, _CTX_MAX_ALPHA)
                    # RNR restriction (#28): bound the modeled deviation by cap_eff(stability, n) so the
                    # hero's response is a RESTRICTED BR. Gift composes UNDER it. Default-OFF => unchanged.
                    if self.config.rnr_restrict:
                        stab = (1.0 if self.opponent_stability is None
                                else float(self.opponent_stability.get(actor, 1.0)))
                        max_alpha = min(max_alpha, _rnr_cap_eff(stab, self._rnr_n_live()))
                    strategy = apply_context_bias(strategy, read, legal, max_alpha=max_alpha)
                else:
                    strategy = apply_opponent_bias(strategy, stats, legal)
        return strategy

    def _cfv(
        self,
        env,
        seat: int,
        hero_seat: int,
        depth: int,
        reach: float,
        sampled_holes: dict[int, tuple[int, int]],
    ) -> float:
        if bool(np.asarray(env.done)) or depth >= self.config.max_depth:
            legal = np.asarray(legal_action_mask(env, jnp.int32(seat)), dtype=bool)
            q = self._leaf_q(env)
            return float(np.max(q[legal])) if legal.any() else 0.0

        actor = int(np.asarray(env.action_seat))
        legal = np.asarray(legal_action_mask(env, jnp.int32(actor)), dtype=bool)
        if not legal.any():
            return 0.0

        key = self._infoset_key(env, actor)
        regrets = self._get_regrets(key, legal)

        if actor != hero_seat:
            strategy = self._opponent_strategy(env, actor, legal, regrets, use_frozen=getattr(self, "_ssmig_use_frozen", False))
            a_idx = int(self.rng.choice(np.where(legal)[0], p=strategy[legal] / strategy[legal].sum()))
            holes = np.asarray(env.holes).copy()
            if actor in sampled_holes:
                c0, c1 = sampled_holes[actor]
                holes[actor, 0] = c0
                holes[actor, 1] = c1
                env = env._replace(holes=jnp.asarray(holes))
            next_env = step_state(env, jnp.int32(a_idx))
            return self._cfv(next_env, seat, hero_seat, depth + 1, reach, sampled_holes)

        strategy = _regret_match(regrets, legal)
        action_values = np.zeros(NUM_ACTIONS, dtype=np.float64)
        for a in np.where(legal)[0]:
            next_env = step_state(env, jnp.int32(int(a)))
            action_values[a] = self._cfv(next_env, seat, hero_seat, depth + 1, reach, sampled_holes)

        ev = float(np.dot(strategy, action_values))
        for a in np.where(legal)[0]:
            regrets[a] += action_values[a] - ev
            regrets[a] = max(regrets[a], 0.0)
        self._strategy_sum[key] += reach * strategy
        return ev

    def solve(self, root: SubgameRoot, pbs: PBSTracker, cancel_event=None) -> SolveResult:
        """Run time-budgeted SSMIG CFR+ from root.

        ``cancel_event`` (optional threading.Event): if set mid-solve (e.g. by a live watchdog or
        the multitable manager when the table moves on), the iteration loop breaks early and
        serves the best average so far. Mirrors PluribusReBeLSolver.solve (the live bridge now
        passes the same event to both). None => only the internal time_budget_ms bounds the solve.
        """
        t0 = time.perf_counter()
        cfg = self.config
        hero = root.hero_seat
        iterations = 0
        leaf_ms = 0.0
        cache_hits_start = self._leaf_cache_hits
        self._leaf_eval_count = 0

        self._traversal_ms = 0.0
        sampled_holes = pbs.sample_opponent_holes(self.rng)

        def _cancelled() -> bool:
            return cancel_event is not None and cancel_event.is_set()

        while iterations < cfg.max_iterations:
            # B1 fix: cancel_event is now respected on the outer loop. The inner SSMIG loop also
            # rechecks (line below) so a cancel mid-batch aborts within one or two iterations,
            # not the full ``ssmig_inner_iters`` count. Same priority order as Pluribus: cancel
            # wins over time budget (a cancelled solve should return FAST).
            if _cancelled():
                break
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            if elapsed_ms >= cfg.time_budget_ms:
                break

            self._frozen_opp_avg.clear()
            for key, sums in self._strategy_sum.items():
                if sums.sum() > 1e-12:
                    self._frozen_opp_avg[key] = (sums / sums.sum()).astype(np.float64)

            for inner in range(cfg.ssmig_inner_iters):
                if _cancelled():
                    break
                if (time.perf_counter() - t0) * 1000.0 >= cfg.time_budget_ms:
                    break
                if inner % max(cfg.ssmig_resample_every, 1) == 0:
                    sampled_holes = pbs.sample_opponent_holes(self.rng)

                self._ssmig_use_frozen = inner > 0
                env = env_to_numpy(root.env)
                holes = np.asarray(env.holes)
                for seat, (c0, c1) in sampled_holes.items():
                    holes = np.array(holes)
                    holes[seat, 0] = c0
                    holes[seat, 1] = c1
                env = env._replace(holes=jnp.asarray(holes))
                iter_before = time.perf_counter()
                self._cfv(env, hero, hero, 0, 1.0, sampled_holes)
                self._traversal_ms += (time.perf_counter() - iter_before) * 1000.0
                iterations += 1
                if iterations >= cfg.max_iterations:
                    break

            if _cancelled():
                break
            if (time.perf_counter() - t0) * 1000.0 >= cfg.time_budget_ms * 0.8:
                break

        self._flush_leaf_batch()

        root_key = self._infoset_key(root.env, hero)
        legal = root.legal_mask.astype(bool)
        if root_key in self._strategy_sum and self._strategy_sum[root_key].sum() > 0:
            avg = self._strategy_sum[root_key].copy()
            avg = avg * legal
            probs = avg / max(avg.sum(), 1e-12)
        else:
            regrets = self._regrets.get(root_key, np.zeros(NUM_ACTIONS))
            probs = _regret_match(regrets, legal)

        probs = probs * legal
        probs = probs / max(probs.sum(), 1e-12)

        ev = float(np.sum(probs * self._leaf_q(root.env))) if legal.any() else 0.0
        budget_ms = (time.perf_counter() - t0) * 1000.0
        self._flush_leaf_batch()
        traversal_ms = self._traversal_ms
        leaf_ms_est = max(budget_ms - traversal_ms, 0.0)

        return SolveResult(
            action_probs=probs.astype(np.float32),
            ev=ev,
            iterations=iterations,
            metadata={
                "leaf_eval_ms": leaf_ms_est,
                "leaf_eval_count": float(self._leaf_eval_count),
                "cache_hits": float(self._leaf_cache_hits - cache_hits_start),
                "budget_used_ms": budget_ms,
                "traversal_ms": traversal_ms,
                "traversal_fraction": float(traversal_ms / max(budget_ms, 1e-6)),
                "native_cfr_recommended": traversal_ms / max(budget_ms, 1e-6) > 0.4,
                "leaf_ms_per_iter": float(leaf_ms_est / max(iterations, 1)),
                "decision_latency_ms_p50": float(budget_ms),
                "decision_latency_ms_p95": float(budget_ms),
            },
        )
