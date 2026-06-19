"""Local Best Response (LBR) — a tractable LOWER BOUND on HUNL exploitability.

This is the real metric used in the literature (Lisy & Bowling, AAAI 2017,
"Equilibrium Approximation Quality of Current No-Limit Poker Bots"), NOT the
heuristic exploiter ensemble in ``pokerbot/eval/exploitability.py`` (which is a
relative sanity tool only) nor the aggressive-exploiter proxy in
``pokerbot/eval/hu_lbr_proxy.py``.

How it works
------------
LBR plays one seat; the *target* policy plays the other. LBR maintains a Bayesian
belief over the target's hole cards, updated from the target's known strategy at
every target decision (range tracking). At each LBR decision it values a small set
of candidate actions (fold / call-check / a pot-ish bet / all-in) using:
  - fold equity: the share of the target range that folds to a bet (from the
    target's own strategy at the hypothetical post-bet node), and
  - showdown equity: LBR's hand vs the (continuing) target range via rollout.
LBR then plays the argmax-EV action. Because LBR plays a *fully specified legal
strategy*, the actual EV it achieves against the target (estimated by Monte-Carlo
over many hands) is a guaranteed LOWER BOUND on the target's true exploitability.

Everything runs on the pure-NumPy engine (``np_engine``) and keys the blueprint
with the SAME ``fingerprint_fast`` used at training time, so the policy queried
here is byte-identical to the trained/served blueprint.

Units: results are reported in bb/hand, mbb/g (= bb/hand * 1000), and bb/100.
A POSITIVE value means LBR beats the target by that much => the target is at
least that exploitable. Reference points (Lisy & Bowling): always-fold ~750
mbb/g; 2014-2016 ACPC bots 400-1000 mbb/g (restricted LBR); DeepStack ~0
(LBR could not exploit it).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np

# Bounded abstraction must be configured BEFORE the encoders are imported so the
# fingerprint matches the trained table. Callers that build a blueprint policy are
# expected to have set these already; we assert-friendly default them here too.
os.environ.setdefault("POKERBOT_BETTING_ABSTRACTION", "1")
os.environ.setdefault("POKERBOT_POSTFLOP_BUCKETS", "1")
os.environ.setdefault("POKERBOT_POSTFLOP_MODE", "fast")
os.environ.setdefault("POKERBOT_POSTFLOP_NUM_BUCKETS", "200")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from pokerbot.engine.jax_engine import CHIP_UNIT, init_hand  # noqa: E402
from pokerbot.engine.np_engine import (  # noqa: E402
    ALL_IN,
    BET_BUCKET_START,
    CALL,
    CHECK,
    FOLD,
    NUM_ACTIONS,
    SEAT_LIVE,
    apply_action_np,
    evaluate_seven,
    is_chance_node_np,
    legal_action_mask_np,
    sample_chance_np,
    to_numpy_state,
)

from phevaluator import evaluate_cards as _phe  # noqa: E402  (fast C eval; ordering == our fixed engine)

_CARD_STR = ["23456789TJQKA"[c // 4] + "cdhs"[c % 4] for c in range(52)]

_HU_OCC = np.array([False, False, False, True, False, True], dtype=bool)
_HU_LIVE = (3, 5)
_POT_BET_INDEX = BET_BUCKET_START + 4  # frac 1.0 == full pot (BET_SIZE_BUCKETS[4])


# --------------------------------------------------------------------------- #
# Configuration + result
# --------------------------------------------------------------------------- #
@dataclass
class LBRConfig:
    hands: int = 400
    seed: int = 0
    max_depth: int = 80
    range_cap: int = 120          # max belief combos carried (weighted-subsample beyond this)
    equity_rollouts: int = 16     # runout samples per combo for showdown equity
    bet_candidates: tuple[int, ...] = (_POT_BET_INDEX, ALL_IN)  # LBR's own bet menu


@dataclass
class LBRResult:
    ev_bb_per_hand: float
    mbb_per_g: float
    bb_per_100: float
    hands: int
    stderr_bb_per_hand: float
    note: str = (
        "Local Best Response lower bound on exploitability (Lisy & Bowling 2017). "
        "Positive => target exploitable by at least this much. Lower bound, not exact."
    )
    detail: dict = field(default_factory=dict)


PolicyFn = "callable"  # (env_np, seat:int) -> np.ndarray shape (NUM_ACTIONS,), sums to 1 over legal


# --------------------------------------------------------------------------- #
# Blueprint + calibration policies  (all operate on numpy EnvState)
# --------------------------------------------------------------------------- #
# Stage-1b blueprint coverage diagnostic (opt-in via POKERBOT_BP_COVERAGE). When enabled, every
# make_blueprint_policy lookup records hit (trained row actually used) vs miss (uniform-legal fallback),
# per street, into a process-global counter. The basestrength driver reads bp_coverage_report() to decide
# whether the search ceiling is blueprint COVERAGE (retrain) or search CONFIG. Disabled ⇒ byte-identical
# (a single bool check per lookup), so it is always safe to leave the code in.
_BP_COVERAGE_ON = os.environ.get("POKERBOT_BP_COVERAGE", "0") not in ("0", "", "false", "False")
_BP_COVERAGE: dict[int, list[int]] = {}  # street_int -> [hits, misses]
_STREET_NAMES = {0: "preflop", 1: "flop", 2: "turn", 3: "river"}


def bp_coverage_reset() -> None:
    _BP_COVERAGE.clear()


def bp_coverage_report() -> dict:
    """Per-street blueprint hit-rate since the last reset. ``{}`` if disabled or no lookups yet."""
    out: dict = {}
    for s in sorted(_BP_COVERAGE):
        hits, misses = _BP_COVERAGE[s]
        tot = hits + misses
        out[_STREET_NAMES.get(s, str(s))] = {
            "hits": hits, "misses": misses, "hit_rate": (hits / tot if tot else 0.0)}
    return out


def check_blueprint_serve_env(*, strict: bool = False, expected: dict | None = None) -> dict:
    """Verify the abstraction is configured so blueprint serve keys MATCH the trained table.

    ``jax_weva_encode`` bakes ``POKERBOT_BETTING_ABSTRACTION`` / ``POKERBOT_POSTFLOP_BUCKETS`` /
    ``POKERBOT_PREFLOP_LOSSLESS`` (and the mode/bucket-count) into module constants AT IMPORT, so they
    must be set BEFORE pokerbot is imported. If they are off, every postflop blueprint lookup misses and
    the search silently falls back to a uniform postflop policy (~0% postflop coverage) — the train/serve
    skew the Stage-1b diagnostic caught. The same applies to ``preflop_lossless``: serving with the flag
    set differently from training keys EVERY preflop node to a miss. Returns the active config; warns
    (default) or raises (``strict``) when misconfigured. ``expected`` (the abstraction dict STAMPED into
    the trained snapshot, see ``active_abstraction_config``) is cross-checked field-by-field when given.
    The policy NET is invariant to these flags (it always uses raw cards), so enabling them is safe for
    the SFT/hybrid path too — only the tabular blueprint key is abstracted."""
    from pokerbot.engine import jax_weva_encode as _enc

    cfg = _enc.active_abstraction_config()
    problems: list[str] = []
    if not (cfg["betting_abstraction"] and cfg["postflop_buckets"]):
        problems.append(
            "set POKERBOT_BETTING_ABSTRACTION=1 and POKERBOT_POSTFLOP_BUCKETS=1 (mode=fast, "
            "num_buckets=200) BEFORE importing pokerbot — the encoder bakes them at import, else "
            "postflop blueprint coverage is ~0% and search falls back to a uniform postflop policy")
    if expected is not None:
        mism = {k: (expected[k], cfg.get(k)) for k in expected if cfg.get(k) != expected[k]}
        if mism:
            problems.append(
                f"serve abstraction != trained-snapshot stamp (expected,active): {mism} — every "
                "affected node misses; set the matching POKERBOT_* flags before import")
        # Depth-bucket guard: a PRE-depth stamp has no "depth_bucketed" key, so the field-by-field loop
        # above can't catch an OLD blueprint served under POKERBOT_DEPTH_BUCKETS=1 (every preflop key would
        # then hash a depth byte the table never saw -> 0% preflop coverage). Check it explicitly with a
        # default so a missing key is treated as False on both sides.
        if bool(expected.get("depth_bucketed", False)) != bool(cfg.get("depth_bucketed", False)):
            problems.append(
                f"depth_bucketed mismatch (trained={expected.get('depth_bucketed', False)}, "
                f"serve={cfg.get('depth_bucketed', False)}) — set POKERBOT_DEPTH_BUCKETS to match the "
                "snapshot, else every preflop node misses")
    if problems:
        msg = f"blueprint serve MISCONFIGURED: {'; '.join(problems)}. active={cfg}"
        if strict:
            raise RuntimeError(msg)
        import warnings

        warnings.warn(msg, RuntimeWarning, stacklevel=2)
    return cfg


def make_blueprint_policy(snapshot: dict[bytes, np.ndarray], *, client_bet_sizing: bool = True):
    """Return a numpy-native policy_fn(env, seat)->probs matching the trained table keys.

    Replicates ``TabularBlueprintBot._probs_for_env`` but with the numpy engine and
    ``fingerprint_fast`` (the exact key used during training), so lookups hit.
    """
    from pokerbot.engine.jax_weva_fast import default_weva_bucket_table
    # Gated native fingerprint: ``fingerprint_native.fingerprint_fast`` dispatches to the Rust core
    # ONLY when POKERBOT_FINGERPRINT_NATIVE is truthy AND the .pyd is built (byte-identical, validated
    # by tools/fingerprint_native/verify_parity.py); otherwise it delegates to the Python oracle, so
    # default serve behavior is unchanged. Same dict-key bytes either way -> no coverage skew.
    from pokerbot.encoding.fingerprint_native import fingerprint_fast
    from pokerbot.training.sd_cfr_nlhe_hogwild import _apply_abstraction
    from pokerbot.training.sd_cfr_nlhe import build_allowed_action_mask

    weva = default_weva_bucket_table()
    allowed_pre = build_allowed_action_mask((4, 5, 6)) if client_bet_sizing else None
    allowed_post = build_allowed_action_mask((0, 1, 3, 4, 5)) if client_bet_sizing else None

    def policy(env, seat: int) -> np.ndarray:
        mask = legal_action_mask_np(env, seat).astype(bool)
        if client_bet_sizing:
            mask = _apply_abstraction(mask, int(env.street), allowed_pre, allowed_post).astype(bool)
        key = fingerprint_fast(env, seat, weva_table=weva)
        row = snapshot.get(key)
        if row is not None:
            p = np.asarray(row, dtype=np.float64) * mask
            tot = p.sum()
            if tot > 1e-12:
                if _BP_COVERAGE_ON:
                    _BP_COVERAGE.setdefault(int(env.street), [0, 0])[0] += 1  # hit: row used
                return p / tot
        if _BP_COVERAGE_ON:
            _BP_COVERAGE.setdefault(int(env.street), [0, 0])[1] += 1  # miss: uniform-legal fallback
        m = mask.astype(np.float64)
        return m / max(m.sum(), 1.0)

    return policy


def uniform_policy(env, seat: int) -> np.ndarray:
    m = legal_action_mask_np(env, seat).astype(np.float64)
    return m / max(m.sum(), 1.0)


def always_fold_policy(env, seat: int) -> np.ndarray:
    mask = legal_action_mask_np(env, seat).astype(bool)
    p = np.zeros(NUM_ACTIONS, dtype=np.float64)
    if mask[FOLD]:
        p[FOLD] = 1.0
    elif mask[CHECK]:
        p[CHECK] = 1.0
    else:
        p[np.argmax(mask)] = 1.0
    return p


def make_nit_policy(call_thresh: float = 0.0):
    """Tight value bot: bet/call only strong made hands, else check/fold. Calibration aid."""
    def policy(env, seat: int) -> np.ndarray:
        mask = legal_action_mask_np(env, seat).astype(bool)
        p = np.zeros(NUM_ACTIONS, dtype=np.float64)
        hole = env.holes[seat]
        r0, r1 = int(hole[0]) // 4 + 2, int(hole[1]) // 4 + 2
        strong = (r0 == r1) or (max(r0, r1) >= 13)
        if strong and mask[BET_BUCKET_START + 4]:
            p[BET_BUCKET_START + 4] = 1.0
        elif strong and mask[CALL]:
            p[CALL] = 1.0
        elif mask[CHECK]:
            p[CHECK] = 1.0
        elif mask[FOLD]:
            p[FOLD] = 1.0
        else:
            p[np.argmax(mask)] = 1.0
        s = p.sum()
        return p / s if s > 0 else uniform_policy(env, seat)
    return policy


# --------------------------------------------------------------------------- #
# Belief over the target's hole cards + showdown equity
# --------------------------------------------------------------------------- #
def _board_cards(env) -> np.ndarray:
    b = np.asarray(env.board)
    return b[b >= 0]


def _all_combos(dead: set[int]) -> np.ndarray:
    live = [c for c in range(52) if c not in dead]
    combos = [(a, b) for i, a in enumerate(live) for b in live[i + 1:]]
    return np.array(combos, dtype=np.int32)


def _set_holes(env, seat: int, c0: int, c1: int):
    holes = env.holes.copy()
    holes[seat] = (c0, c1)
    return env._replace(holes=holes)


def _equity_vs_combos(lbr_hole: np.ndarray, combos: np.ndarray, board: np.ndarray,
                      rng: np.random.Generator, rollouts: int) -> np.ndarray:
    """P(LBR hand wins) per combo, averaged over sampled runouts. {0,0.5,1} per sample.

    Uses the C-backed phevaluator (lower rank == stronger), whose ordering is identical to the
    (now-fixed) engine evaluator — verified 0 disagreements across all 7462 strength classes.
    """
    n = combos.shape[0]
    if n == 0:
        return np.zeros(0, dtype=np.float64)
    need = 5 - board.shape[0]
    lbr_str = [_CARD_STR[int(lbr_hole[0])], _CARD_STR[int(lbr_hole[1])]]
    board_str = [_CARD_STR[int(c)] for c in board]
    eq = np.zeros(n, dtype=np.float64)
    base_dead = set(int(x) for x in board.tolist()) | {int(lbr_hole[0]), int(lbr_hole[1])}

    if need == 0:
        lh = _phe(*(lbr_str + board_str))
        for i in range(n):
            oh = _phe(_CARD_STR[int(combos[i, 0])], _CARD_STR[int(combos[i, 1])], *board_str)
            eq[i] = 1.0 if lh < oh else (0.5 if lh == oh else 0.0)
        return eq

    for i in range(n):
        c0, c1 = int(combos[i, 0]), int(combos[i, 1])
        deck = [c for c in range(52) if c not in (base_dead | {c0, c1})]
        opp_str = [_CARD_STR[c0], _CARD_STR[c1]]
        wins = 0.0
        for _ in range(rollouts):
            pick = rng.choice(len(deck), size=need, replace=False)
            full = board_str + [_CARD_STR[deck[d]] for d in pick]
            lh = _phe(*(lbr_str + full))
            oh = _phe(*(opp_str + full))
            wins += 1.0 if lh < oh else (0.5 if lh == oh else 0.0)
        eq[i] = wins / rollouts
    return eq


# --------------------------------------------------------------------------- #
# LBR decision
# --------------------------------------------------------------------------- #
def _lbr_action(env, lbr_seat: int, bp_seat: int, combos: np.ndarray, weights: np.ndarray,
                policy_fn, cfg: LBRConfig, rng: np.random.Generator) -> int:
    mask = legal_action_mask_np(env, lbr_seat).astype(bool)
    lbr_hole = np.asarray(env.holes[lbr_seat], dtype=np.int32)
    board = _board_cards(env)

    lbr_inv = float(env.committed_total[lbr_seat])
    bp_inv = float(env.committed_total[bp_seat])
    committed_street = float(env.committed_street[lbr_seat])
    to_call = max(0.0, float(env.current_bet) - committed_street)

    # Showdown equity of LBR vs the current belief (used by call + bet valuations).
    if combos.shape[0] == 0:
        eq_arr = np.zeros(0)
        eq_mean = 0.5
    else:
        eq_arr = _equity_vs_combos(lbr_hole, combos, board, rng, cfg.equity_rollouts)
        eq_mean = float(np.dot(weights, eq_arr))

    best_a, best_v = None, -1e18

    # FOLD
    if mask[FOLD]:
        v = -lbr_inv
        if v > best_v:
            best_v, best_a = v, FOLD

    # CHECK / CALL  -> assume the remaining streets check down to showdown.
    if mask[CALL] or mask[CHECK]:
        lbr_inv_after = lbr_inv + (to_call if mask[CALL] else 0.0)
        pot_after = lbr_inv_after + bp_inv
        v = eq_mean * pot_after - lbr_inv_after
        a = CALL if mask[CALL] else CHECK
        if v > best_v:
            best_v, best_a = v, a

    # BET candidates: value via per-combo fold equity (from the target's own strategy) + equity.
    if combos.shape[0] > 0:
        for a in cfg.bet_candidates:
            if not mask[a]:
                continue
            env_bet = apply_action_np(env, a)
            lbr_inv_after = float(env_bet.committed_total[lbr_seat])
            call_value = eq_arr * (2.0 * lbr_inv_after) - lbr_inv_after  # bp matches -> showdown
            fold_value = bp_inv  # bp folds -> win what it already committed
            # Only model a bp fold/call response if the bet actually leaves bp a decision. If LBR's bet
            # closes the action (all-in confrontation -> env_bet is terminal/chance with the board
            # auto-dealt), there's no fold equity and querying the policy on the advanced board would
            # feed duplicate cards to the evaluator; value it directly as a showdown.
            bp_to_act = ((not bool(env_bet.done)) and (not is_chance_node_np(env_bet))
                         and int(env_bet.action_seat) == bp_seat)
            if bp_to_act:
                fold_p = np.empty(combos.shape[0], dtype=np.float64)
                for i in range(combos.shape[0]):
                    env_c = _set_holes(env_bet, bp_seat, int(combos[i, 0]), int(combos[i, 1]))
                    fold_p[i] = float(policy_fn(env_c, bp_seat)[FOLD])
                per_combo = fold_p * fold_value + (1.0 - fold_p) * call_value
            else:
                per_combo = call_value
            v = float(np.dot(weights, per_combo))
            if v > best_v:
                best_v, best_a = v, a

    if best_a is None:
        best_a = int(np.argmax(mask))
    return int(best_a)


# --------------------------------------------------------------------------- #
# Main driver
# --------------------------------------------------------------------------- #
def lbr_exploitability(policy_fn, cfg: LBRConfig | None = None) -> LBRResult:
    """Monte-Carlo LBR lower bound on exploitability of ``policy_fn`` in HU NLHE."""
    cfg = cfg or LBRConfig()
    key = jax.random.PRNGKey(cfg.seed)
    rng = np.random.default_rng(cfg.seed + 1)
    occ = jnp.asarray(_HU_OCC)

    results = np.zeros(cfg.hands, dtype=np.float64)
    forced_uniform_target_acts = 0
    target_acts = 0

    for h in range(cfg.hands):
        key, k = jax.random.split(key)
        # Button MUST land on a LIVE seat (3 or 5). With button=None, init_hand draws a random seat
        # over all 6, so ~2/3 of hands put the button on a DEAD seat -- and encode_info_state one-hots
        # state.button over 6 seats (jax_engine.py:585), so a dead-seat button lights an input position
        # the target never saw in training (HU training deals button in {3,5} only). That off-distribution
        # serve makes a position-aware target misplay catastrophically and inflates the LBR bound. Deal
        # the button on a live seat, rotated DECOUPLED from the LBR/target assignment (h//2 vs h%2) so all
        # four (who-is-LBR) x (who-has-button) combinations are sampled equally -> unbiased position.
        button = jnp.int32(_HU_LIVE[(h // 2) % 2])
        env = to_numpy_state(init_hand(k, button=button, occupancy=occ))
        lbr_seat = _HU_LIVE[h % 2]
        bp_seat = _HU_LIVE[1 - (h % 2)]

        # LBR belief over the target's (bp) hole cards: uniform over combos not blocked
        # by LBR's cards or the (currently empty) board.
        dead = {int(env.holes[lbr_seat][0]), int(env.holes[lbr_seat][1])}
        combos = _all_combos(dead)
        if combos.shape[0] > cfg.range_cap:  # cap support up front so per-combo loops stay bounded
            combos = combos[rng.choice(combos.shape[0], size=cfg.range_cap, replace=False)]
        weights = np.full(combos.shape[0], 1.0 / combos.shape[0], dtype=np.float64)

        depth = 0
        while depth < cfg.max_depth and not bool(env.done):
            if is_chance_node_np(env):
                env = sample_chance_np(env)
                # prune combos that collide with newly revealed board cards
                board_set = set(int(x) for x in _board_cards(env).tolist())
                keep = np.array([
                    (int(combos[i, 0]) not in board_set and int(combos[i, 1]) not in board_set)
                    for i in range(combos.shape[0])
                ], dtype=bool) if combos.shape[0] else np.zeros(0, dtype=bool)
                if combos.shape[0]:
                    combos, weights = combos[keep], weights[keep]
                    s = weights.sum()
                    weights = weights / s if s > 1e-12 else np.full(combos.shape[0], 1.0 / max(combos.shape[0], 1))
                continue
            if bool(env.done):
                break
            seat = int(env.action_seat)
            # Keep the belief consistent before any policy/equity query: drop combos blocked by the
            # current board or LBR's own cards (impossible holdings). Guards phevaluator against
            # duplicate-card inputs regardless of how/when board cards were revealed.
            if combos.shape[0]:
                blocked = set(int(x) for x in _board_cards(env).tolist()) | {
                    int(env.holes[lbr_seat][0]), int(env.holes[lbr_seat][1])}
                keep = np.array([int(combos[i, 0]) not in blocked and int(combos[i, 1]) not in blocked
                                 for i in range(combos.shape[0])], dtype=bool)
                if not keep.all():
                    combos, weights = combos[keep], weights[keep]
                    s = weights.sum()
                    weights = (weights / s if s > 1e-12
                               else np.full(combos.shape[0], 1.0 / max(combos.shape[0], 1)))
            if seat == bp_seat:
                # target acts with its real dealt holes; sample from its policy
                probs = policy_fn(env, bp_seat)
                target_acts += 1
                legal = np.flatnonzero(probs > 0)
                if legal.size == 0:
                    legal = np.flatnonzero(legal_action_mask_np(env, bp_seat))
                    a = int(rng.choice(legal))
                    forced_uniform_target_acts += 1
                else:
                    a = int(rng.choice(legal, p=probs[legal] / probs[legal].sum()))
                # Bayes-update LBR belief from the observed action.
                if combos.shape[0]:
                    upd = np.empty(combos.shape[0], dtype=np.float64)
                    for i in range(combos.shape[0]):
                        env_c = _set_holes(env, bp_seat, int(combos[i, 0]), int(combos[i, 1]))
                        upd[i] = float(policy_fn(env_c, bp_seat)[a])
                    weights = weights * upd
                    s = weights.sum()
                    if s > 1e-12:
                        weights = weights / s
                    else:
                        weights = np.full(combos.shape[0], 1.0 / combos.shape[0])
                    # cap belief support for speed (weighted subsample)
                    if combos.shape[0] > cfg.range_cap:
                        idx = rng.choice(combos.shape[0], size=cfg.range_cap, replace=False, p=weights)
                        combos, weights = combos[idx], weights[idx]
                        weights = weights / weights.sum()
                env = apply_action_np(env, a)
            else:
                a = _lbr_action(env, lbr_seat, bp_seat, combos, weights, policy_fn, cfg, rng)
                env = apply_action_np(env, a)
            depth += 1

        if bool(env.done):
            results[h] = float(env.rewards[lbr_seat]) / CHIP_UNIT

    ev = float(results.mean())
    stderr = float(results.std(ddof=1) / np.sqrt(cfg.hands)) if cfg.hands > 1 else 0.0
    return LBRResult(
        ev_bb_per_hand=ev,
        mbb_per_g=ev * 1000.0,
        bb_per_100=ev * 100.0,
        hands=cfg.hands,
        stderr_bb_per_hand=stderr,
        detail={
            "target_decisions": target_acts,
            "forced_uniform_target_decisions": forced_uniform_target_acts,
            "range_cap": cfg.range_cap,
            "equity_rollouts": cfg.equity_rollouts,
        },
    )
