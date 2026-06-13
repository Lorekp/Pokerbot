"""Shared best-response and Nash-gap machinery for multiway push/fold FP solvers."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from aof_cfr.pushfold_payoffs import TERMINAL, StaticPayoffs, build_static_payoffs, mix
from aof_cfr.trio_payoff_tensor import trio_deal_weights

AGGRESS = 1


@dataclass
class StabilityTracker:
    gap_tol: float = 0.5
    policy_tol: float = 0.001
    window: int = 5
    gap_floor_mbb_g: float | None = None
    _gaps: list[float] = field(default_factory=list)
    _btn_delta: list[float] = field(default_factory=list)
    _prev_btn: np.ndarray | None = None

    def observe(self, gap_mbb_g: float, btn_push: np.ndarray) -> bool:
        if self._prev_btn is not None:
            self._btn_delta.append(float(np.max(np.abs(btn_push - self._prev_btn))))
        self._prev_btn = btn_push.copy()
        self._gaps.append(gap_mbb_g)
        if len(self._gaps) < self.window:
            return False
        if self.gap_floor_mbb_g is not None and gap_mbb_g > self.gap_floor_mbb_g:
            return False
        gap_stable = (
            max(self._gaps[-self.window :]) - min(self._gaps[-self.window :]) < self.gap_tol
        )
        pol_stable = (
            max(self._btn_delta[-self.window :]) < self.policy_tol if self._btn_delta else False
        )
        return gap_stable and pol_stable


def nash_gap_3max(
    st: StaticPayoffs,
    deal_w: np.ndarray,
    btn_push: np.ndarray,
    sb_push: np.ndarray,
    bb_push: np.ndarray,
) -> tuple[float, list[float]]:
    """CLAIRVOYANT Nash gap (bb/hand) — DIAGNOSTIC ONLY, do not use as exploitability.

    Takes ``max(fold, push)`` per ``(i, j, k)`` cell, i.e. chooses the action AFTER
    seeing both opponents' hole cards, then averages. By Jensen this upper-bounds the
    legal gap (E[max] >= max(E)) and over-states exploitability ~3x on this game. The
    correct, legal (imperfect-information) metric is :func:`nash_gap_3max_legal`, which
    marginalizes opponents' hidden hands before choosing the action. Retained only to
    reproduce historical numbers and for clairvoyant-vs-legal comparisons.
    """
    t = TERMINAL
    pbtn = btn_push[:, None, None]
    psf = sb_push[0, :, None]
    psa = sb_push[1, :, None]
    pfa = bb_push[0, None, :]
    paf = bb_push[1, None, :]
    paa = bb_push[2, None, :]

    u_sb_fa1 = mix(pfa, t.sb_win_uncontested_after_btn_fold, st.sb_bb_call1)
    u_btn_after_sb_f = mix(paf, t.btn_win_uncontested_after_sb_fold, st.btn_bb_call0)
    u_btn_after_sb_a = mix(paa, st.btn_sb_call0, st.three0)
    u_btn_push0 = mix(psa, u_btn_after_sb_f, u_btn_after_sb_a)
    btn_val = np.sum(deal_w * ((1.0 - pbtn) * t.btn_fold + pbtn * u_btn_push0))
    btn_br = np.sum(deal_w * np.maximum(t.btn_fold, u_btn_push0))
    btn_imp = float(btn_br - btn_val)

    sb_f_val = mix(psf, t.sb_fold_after_btn_fold, u_sb_fa1)
    sb_f_br = np.maximum(t.sb_fold_after_btn_fold, u_sb_fa1)
    u_sb_aa1 = mix(paa, st.btn_sb_call1, st.three1)
    sb_a_val = mix(psa, t.sb_fold_after_btn_jam, u_sb_aa1)
    sb_a_br = np.maximum(t.sb_fold_after_btn_jam, u_sb_aa1)
    sb_val = np.sum(deal_w * ((1.0 - pbtn) * sb_f_val + pbtn * sb_a_val))
    sb_br = np.sum(deal_w * ((1.0 - pbtn) * sb_f_br + pbtn * sb_a_br))
    sb_imp = float(sb_br - sb_val)

    bb_fa_val = mix(pfa, t.bb_fold, st.sb_bb_call2)
    bb_af_val = mix(paf, t.bb_fold, st.btn_bb_call2)
    bb_aa_val = mix(paa, t.bb_fold, st.three2)
    bb_fa_br = np.maximum(t.bb_fold, st.sb_bb_call2)
    bb_af_br = np.maximum(t.bb_fold, st.btn_bb_call2)
    bb_aa_br = np.maximum(t.bb_fold, st.three2)
    bb_val = np.sum(
        deal_w * ((1.0 - pbtn) * psf * bb_fa_val + pbtn * (1.0 - psa) * bb_af_val + pbtn * psa * bb_aa_val)
    )
    bb_br = np.sum(
        deal_w * ((1.0 - pbtn) * psf * bb_fa_br + pbtn * (1.0 - psa) * bb_af_br + pbtn * psa * bb_aa_br)
    )
    bb_imp = float(bb_br - bb_val)

    imps = [max(0.0, btn_imp), max(0.0, sb_imp), max(0.0, bb_imp)]
    return float(sum(imps)), imps


def _legal_marginals_3max(
    st: StaticPayoffs,
    deal_w: np.ndarray,
    btn_push: np.ndarray,
    sb_push: np.ndarray,
    bb_push: np.ndarray,
    *,
    keys: tuple[str, ...] | frozenset[str] | None = None,
) -> dict[str, tuple[np.ndarray, float, np.ndarray]]:
    """Single source for the LEGAL (imperfect-information) 3-max best response.

    For every (seat, history) returns ``(E_push, E_fold, reach_w)`` where, per own-hand
    bucket, ``E_push`` is the expected push/call value AFTER marginalizing the opponents'
    hidden hands (deal-weighted over the two non-own axes, with the same reach
    conditioning the clairvoyant BR uses), ``E_fold`` is the constant fold value, and
    ``reach_w`` is the bucket's reach weight. The legal best response is pure: push iff
    ``E_push > E_fold``. ``best_responses_3max_legal`` / ``nash_gap_3max_legal`` /
    ``per_hand_legal_improvement_3max`` all read from here so they cannot drift.

    ``keys`` (optional) restricts the computation to the requested marginal keys. Each
    marginal is computed independently from byte-identical expressions, so skipping the
    unrequested ones leaves the returned ones bit-for-bit unchanged — the arena's
    ``routed_arrays`` only ever consumes the agent's own (1-3) marginals, so this avoids
    the dominant cost of materializing the five unused 169^3 cubes per refresh.
    """
    t = TERMINAL
    want = None if keys is None else frozenset(keys)

    def need(k: str) -> bool:
        return want is None or k in want

    pbtn = btn_push[:, None, None]
    psf = sb_push[0, :, None]
    psa = sb_push[1, :, None]
    pfa = bb_push[0, None, :]
    paf = bb_push[1, None, :]
    paa = bb_push[2, None, :]
    eps = 1e-12

    def _marg(push_cell: np.ndarray, weight: np.ndarray, keep_axis: int) -> tuple[np.ndarray, np.ndarray]:
        shape = np.broadcast_shapes(np.shape(push_cell), np.shape(weight))
        w = np.broadcast_to(weight, shape)
        pc = np.broadcast_to(push_cell, shape)
        red = tuple(a for a in (0, 1, 2) if a != keep_axis)
        den = np.sum(w, axis=red)
        e_push = np.where(den > eps, np.sum(w * pc, axis=red) / np.maximum(den, eps), 0.0)
        return e_push, den

    # Native fast path: a fused Rust kernel materializes the contiguous (w, prod=w*pc) cubes
    # bit-identically (same per-element IEEE-754 op order as the NumPy broadcasts above), and we do the
    # ORDER-SENSITIVE np.sum reductions here on those exact buffers — so the output equals the pure-NumPy
    # path byte-for-byte (verified with np.array_equal in aof_cfr.marg3_native.verify_against_numpy).
    # Absent the extension, _native_marg returns None and we fall through to the NumPy mix/_marg path.
    _KEEP = {"btn": 0, "sb_f": 1, "sb_a": 1, "bb_fa": 2, "bb_af": 2, "bb_aa": 2}

    def _native_marg(key: str):
        try:
            from aof_cfr import marg3_native
        except Exception:
            return None
        built = marg3_native.build_cubes(key, st, deal_w, btn_push, sb_push, bb_push)
        if built is None:
            return None
        w_cube, prod_cube = built
        red = tuple(a for a in (0, 1, 2) if a != _KEEP[key])
        den = np.sum(w_cube, axis=red)
        e_push = np.where(den > eps, np.sum(prod_cube, axis=red) / np.maximum(den, eps), 0.0)
        return e_push, den

    out: dict[str, tuple[np.ndarray, float, np.ndarray]] = {}

    # BTN (axis 0): unconditional reach.
    if need("btn"):
        nat = _native_marg("btn")
        if nat is not None:
            btn_ep, btn_w = nat
        else:
            u_btn_after_sb_f = mix(paf, t.btn_win_uncontested_after_sb_fold, st.btn_bb_call0)
            u_btn_after_sb_a = mix(paa, st.btn_sb_call0, st.three0)
            u_btn_push0 = mix(psa, u_btn_after_sb_f, u_btn_after_sb_a)
            btn_ep, btn_w = _marg(u_btn_push0, deal_w, 0)
        out["btn"] = (btn_ep, float(t.btn_fold), btn_w)

    # SB (axis 1): hist F on BTN fold, hist A on BTN jam.
    if need("sb_f"):
        nat = _native_marg("sb_f")
        if nat is not None:
            sb_f_ep, sb_f_w = nat
        else:
            u_sb_fa1 = mix(pfa, t.sb_win_uncontested_after_btn_fold, st.sb_bb_call1)
            sb_f_ep, sb_f_w = _marg(u_sb_fa1, deal_w * (1.0 - pbtn), 1)
        out["sb_f"] = (sb_f_ep, float(t.sb_fold_after_btn_fold), sb_f_w)
    if need("sb_a"):
        nat = _native_marg("sb_a")
        if nat is not None:
            sb_a_ep, sb_a_w = nat
        else:
            u_sb_aa1 = mix(paa, st.btn_sb_call1, st.three1)
            sb_a_ep, sb_a_w = _marg(u_sb_aa1, deal_w * pbtn, 1)
        out["sb_a"] = (sb_a_ep, float(t.sb_fold_after_btn_jam), sb_a_w)

    # BB (axis 2): FA / AF / AA call EVs.
    if need("bb_fa"):
        nat = _native_marg("bb_fa")
        if nat is not None:
            bb_fa_ep, bb_fa_w = nat
        else:
            bb_fa_ep, bb_fa_w = _marg(st.sb_bb_call2, deal_w * (1.0 - pbtn) * psf, 2)
        out["bb_fa"] = (bb_fa_ep, float(t.bb_fold), bb_fa_w)
    if need("bb_af"):
        nat = _native_marg("bb_af")
        if nat is not None:
            bb_af_ep, bb_af_w = nat
        else:
            bb_af_ep, bb_af_w = _marg(st.btn_bb_call2, deal_w * pbtn * (1.0 - psa), 2)
        out["bb_af"] = (bb_af_ep, float(t.bb_fold), bb_af_w)
    if need("bb_aa"):
        nat = _native_marg("bb_aa")
        if nat is not None:
            bb_aa_ep, bb_aa_w = nat
        else:
            bb_aa_ep, bb_aa_w = _marg(st.three2, deal_w * pbtn * psa, 2)
        out["bb_aa"] = (bb_aa_ep, float(t.bb_fold), bb_aa_w)

    return out


def seat_values_3max(
    st: StaticPayoffs,
    deal_w: np.ndarray,
    btn_push: np.ndarray,
    sb_push: np.ndarray,
    bb_push: np.ndarray,
) -> list[float]:
    """Exact on-policy expected chip-EV (bb/hand) per seat [BTN, SB, BB] for a full table.

    These are the policies' own values (no best-response max), so they're correct regardless
    of the legal/clairvoyant distinction. Used for head-to-head bb/100 evaluation.
    """
    t = TERMINAL
    pbtn = btn_push[:, None, None]
    psf = sb_push[0, :, None]
    psa = sb_push[1, :, None]
    pfa = bb_push[0, None, :]
    paf = bb_push[1, None, :]
    paa = bb_push[2, None, :]

    u_sb_fa1 = mix(pfa, t.sb_win_uncontested_after_btn_fold, st.sb_bb_call1)
    u_btn_after_sb_f = mix(paf, t.btn_win_uncontested_after_sb_fold, st.btn_bb_call0)
    u_btn_after_sb_a = mix(paa, st.btn_sb_call0, st.three0)
    u_btn_push0 = mix(psa, u_btn_after_sb_f, u_btn_after_sb_a)
    btn_val = float(np.sum(deal_w * ((1.0 - pbtn) * t.btn_fold + pbtn * u_btn_push0)))

    sb_f_val = mix(psf, t.sb_fold_after_btn_fold, u_sb_fa1)
    u_sb_aa1 = mix(paa, st.btn_sb_call1, st.three1)
    sb_a_val = mix(psa, t.sb_fold_after_btn_jam, u_sb_aa1)
    sb_val = float(np.sum(deal_w * ((1.0 - pbtn) * sb_f_val + pbtn * sb_a_val)))

    bb_fa_val = mix(pfa, t.bb_fold, st.sb_bb_call2)
    bb_af_val = mix(paf, t.bb_fold, st.btn_bb_call2)
    bb_aa_val = mix(paa, t.bb_fold, st.three2)
    bb_val = float(np.sum(
        deal_w * ((1.0 - pbtn) * psf * bb_fa_val + pbtn * (1.0 - psa) * bb_af_val + pbtn * psa * bb_aa_val)
    ))
    return [btn_val, sb_val, bb_val]


def best_responses_3max_legal(
    st: StaticPayoffs,
    deal_w: np.ndarray,
    btn_push: np.ndarray,
    sb_push: np.ndarray,
    bb_push: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Legal (marginalized) pure best response: push iff E[push|own hand] > E[fold].

    Returns ``(br_btn (n,), br_sb (2, n), br_bb (3, n))`` of 0/1 push frequencies. This
    is the correct FP iteration target — unlike :func:`best_responses_3max`, the action
    cannot depend on opponents' hidden cards.
    """
    m = _legal_marginals_3max(st, deal_w, btn_push, sb_push, bb_push)
    n = int(btn_push.shape[0])
    br_btn = (m["btn"][0] > m["btn"][1]).astype(np.float64)
    br_sb = np.zeros((2, n), dtype=np.float64)
    br_sb[0] = (m["sb_f"][0] > m["sb_f"][1]).astype(np.float64)
    br_sb[1] = (m["sb_a"][0] > m["sb_a"][1]).astype(np.float64)
    br_bb = np.zeros((3, n), dtype=np.float64)
    br_bb[0] = (m["bb_fa"][0] > m["bb_fa"][1]).astype(np.float64)
    br_bb[1] = (m["bb_af"][0] > m["bb_af"][1]).astype(np.float64)
    br_bb[2] = (m["bb_aa"][0] > m["bb_aa"][1]).astype(np.float64)
    return br_btn, br_sb, br_bb


def per_hand_legal_improvement_3max(
    st: StaticPayoffs,
    deal_w: np.ndarray,
    btn_push: np.ndarray,
    sb_push: np.ndarray,
    bb_push: np.ndarray,
) -> dict[str, np.ndarray]:
    """Per-hand, per-seat LEGAL BR improvement (bb): the exploitable EV left at each
    own-hand infoset by playing the current frequency instead of the legal pure BR.

    ``imp = max(E_fold, E_push) - ((1-p) E_fold + p E_push)`` per bucket; ``"*_w"`` are
    reach weights so ``sum(imp * w)`` reconciles to :func:`nash_gap_3max_legal`'s
    per-seat improvements. Shapes: ``btn (n,)``, ``sb (2, n)``, ``bb (3, n)``.
    """
    m = _legal_marginals_3max(st, deal_w, btn_push, sb_push, bb_push)

    def _imp(key: str, p: np.ndarray) -> np.ndarray:
        ep, ef, _w = m[key]
        return np.maximum(ef, ep) - ((1.0 - p) * ef + p * ep)

    return {
        "btn": _imp("btn", btn_push),
        "sb": np.stack([_imp("sb_f", sb_push[0]), _imp("sb_a", sb_push[1])], axis=0),
        "bb": np.stack(
            [_imp("bb_fa", bb_push[0]), _imp("bb_af", bb_push[1]), _imp("bb_aa", bb_push[2])],
            axis=0,
        ),
        "btn_w": m["btn"][2],
        "sb_w": np.stack([m["sb_f"][2], m["sb_a"][2]], axis=0),
        "bb_w": np.stack([m["bb_fa"][2], m["bb_af"][2], m["bb_aa"][2]], axis=0),
    }


def nash_gap_3max_legal(
    st: StaticPayoffs,
    deal_w: np.ndarray,
    btn_push: np.ndarray,
    sb_push: np.ndarray,
    bb_push: np.ndarray,
) -> tuple[float, list[float]]:
    """Legal (imperfect-information) Nash gap (bb/hand) and per-player improvements.

    The true exploitability: each player best-responds with a legal strategy that
    marginalizes opponents' hidden hands. Use this, not :func:`nash_gap_3max`.
    """
    ph = per_hand_legal_improvement_3max(st, deal_w, btn_push, sb_push, bb_push)
    btn_imp = float(np.sum(ph["btn"] * ph["btn_w"]))
    sb_imp = float(np.sum(ph["sb"] * ph["sb_w"]))
    bb_imp = float(np.sum(ph["bb"] * ph["bb_w"]))
    imps = [max(0.0, btn_imp), max(0.0, sb_imp), max(0.0, bb_imp)]
    return float(sum(imps)), imps


def per_hand_br_improvement_3max(
    st: StaticPayoffs,
    deal_w: np.ndarray,
    btn_push: np.ndarray,
    sb_push: np.ndarray,
    bb_push: np.ndarray,
) -> dict[str, np.ndarray]:
    """CLAIRVOYANT per-hand BR improvement (bb) — DIAGNOSTIC; see ``nash_gap_3max``.

    For each (seat, history, hand) returns the bb-EV an opponent gains from this seat
    playing its current frequency at that hand instead of the pure best response,
    conditional on reaching that node holding that hand. The value/BR expressions and
    conditional weights are byte-identical to ``nash_gap_3max`` / the BR functions; only
    the reduction differs (sum over the two non-own axes, normalized by the reduced
    weight). ``"*_w"`` entries are the reduced reach weights so callers can reconcile:
    ``sum(imp * w)`` over a seat's (history, hand) equals that seat's ``nash_gap_3max``
    improvement. BTN's own hand is axis 0, SB's axis 1, BB's axis 2.
    """
    t = TERMINAL
    n = int(btn_push.shape[0])
    pbtn = btn_push[:, None, None]
    psf = sb_push[0, :, None]
    psa = sb_push[1, :, None]
    pfa = bb_push[0, None, :]
    paf = bb_push[1, None, :]
    paa = bb_push[2, None, :]
    eps = 1e-12

    def _reduce(diff: np.ndarray, weight: np.ndarray, keep_axis: int) -> tuple[np.ndarray, np.ndarray]:
        red_axes = tuple(a for a in (0, 1, 2) if a != keep_axis)
        w = np.broadcast_to(weight, diff.shape)
        num = np.sum(w * diff, axis=red_axes)
        den = np.sum(w, axis=red_axes)
        imp = np.where(den > eps, num / np.maximum(den, eps), 0.0)
        return imp, den

    # BTN (axis 0): unconditional reach (weight = deal_w).
    u_btn_after_sb_f = mix(paf, t.btn_win_uncontested_after_sb_fold, st.btn_bb_call0)
    u_btn_after_sb_a = mix(paa, st.btn_sb_call0, st.three0)
    u_btn_push0 = mix(psa, u_btn_after_sb_f, u_btn_after_sb_a)
    btn_diff = np.maximum(t.btn_fold, u_btn_push0) - ((1.0 - pbtn) * t.btn_fold + pbtn * u_btn_push0)
    btn_imp, btn_w = _reduce(btn_diff, deal_w, keep_axis=0)

    # SB (axis 1): hist F reached on BTN fold, hist A on BTN jam.
    u_sb_fa1 = mix(pfa, t.sb_win_uncontested_after_btn_fold, st.sb_bb_call1)
    sb_f_diff = np.maximum(t.sb_fold_after_btn_fold, u_sb_fa1) - mix(
        psf, t.sb_fold_after_btn_fold, u_sb_fa1
    )
    sb_f_imp, sb_f_w = _reduce(sb_f_diff, deal_w * (1.0 - pbtn), keep_axis=1)
    u_sb_aa1 = mix(paa, st.btn_sb_call1, st.three1)
    sb_a_diff = np.maximum(t.sb_fold_after_btn_jam, u_sb_aa1) - mix(
        psa, t.sb_fold_after_btn_jam, u_sb_aa1
    )
    sb_a_imp, sb_a_w = _reduce(sb_a_diff, deal_w * pbtn, keep_axis=1)

    # BB (axis 2): FA / AF / AA histories with their reach weights.
    bb_fa_diff = np.maximum(t.bb_fold, st.sb_bb_call2) - mix(pfa, t.bb_fold, st.sb_bb_call2)
    bb_af_diff = np.maximum(t.bb_fold, st.btn_bb_call2) - mix(paf, t.bb_fold, st.btn_bb_call2)
    bb_aa_diff = np.maximum(t.bb_fold, st.three2) - mix(paa, t.bb_fold, st.three2)
    bb_fa_imp, bb_fa_w = _reduce(bb_fa_diff, deal_w * (1.0 - pbtn) * psf, keep_axis=2)
    bb_af_imp, bb_af_w = _reduce(bb_af_diff, deal_w * pbtn * (1.0 - psa), keep_axis=2)
    bb_aa_imp, bb_aa_w = _reduce(bb_aa_diff, deal_w * pbtn * psa, keep_axis=2)

    return {
        "btn": btn_imp,
        "sb": np.stack([sb_f_imp, sb_a_imp], axis=0),
        "bb": np.stack([bb_fa_imp, bb_af_imp, bb_aa_imp], axis=0),
        "btn_w": btn_w,
        "sb_w": np.stack([sb_f_w, sb_a_w], axis=0),
        "bb_w": np.stack([bb_fa_w, bb_af_w, bb_aa_w], axis=0),
    }


def best_response_btn_3max(
    st: StaticPayoffs,
    deal_w: np.ndarray,
    sb_push: np.ndarray,
    bb_push: np.ndarray,
) -> np.ndarray:
    t = TERMINAL
    psa = sb_push[1, :, None]
    paf = bb_push[1, None, :]
    paa = bb_push[2, None, :]
    u_btn_after_sb_f = mix(paf, t.btn_win_uncontested_after_sb_fold, st.btn_bb_call0)
    u_btn_after_sb_a = mix(paa, st.btn_sb_call0, st.three0)
    u_btn_push0 = mix(psa, u_btn_after_sb_f, u_btn_after_sb_a)
    push_better = u_btn_push0 > t.btn_fold
    num = np.sum(deal_w * push_better, axis=(1, 2))
    den = np.sum(deal_w, axis=(1, 2))
    return np.clip(num / np.maximum(den, 1e-12), 0.0, 1.0)


def best_response_sb_3max(
    st: StaticPayoffs,
    deal_w: np.ndarray,
    btn_push: np.ndarray,
    bb_push: np.ndarray,
) -> np.ndarray:
    t = TERMINAL
    n = btn_push.shape[0]
    out = np.zeros((2, n), dtype=np.float64)
    pbtn = btn_push[:, None, None]
    pfa = bb_push[0, None, :]
    paa = bb_push[2, None, :]

    u_sb_fa1 = mix(pfa, t.sb_win_uncontested_after_btn_fold, st.sb_bb_call1)
    push_f = u_sb_fa1 > t.sb_fold_after_btn_fold
    w_f = deal_w * (1.0 - pbtn)
    for j in range(n):
        mask = w_f[:, j, :]
        wt = mask.sum()
        if wt > 1e-12:
            out[0, j] = float((mask * push_f[:, j, :]).sum() / wt)

    u_sb_aa1 = mix(paa, st.btn_sb_call1, st.three1)
    push_a = u_sb_aa1 > t.sb_fold_after_btn_jam
    w_a = deal_w * pbtn
    for j in range(n):
        mask = w_a[:, j, :]
        wt = mask.sum()
        if wt > 1e-12:
            out[1, j] = float((mask * push_a[:, j, :]).sum() / wt)
    return np.clip(out, 0.0, 1.0)


def best_response_bb_3max(
    st: StaticPayoffs,
    deal_w: np.ndarray,
    btn_push: np.ndarray,
    sb_push: np.ndarray,
) -> np.ndarray:
    t = TERMINAL
    n = btn_push.shape[0]
    out = np.zeros((3, n), dtype=np.float64)
    pbtn = btn_push[:, None, None]
    psf = sb_push[0, :, None]
    psa = sb_push[1, :, None]

    push_fa = st.sb_bb_call2 > t.bb_fold
    w_fa = deal_w * (1.0 - pbtn) * psf
    push_af = st.btn_bb_call2 > t.bb_fold
    w_af = deal_w * pbtn * (1.0 - psa)
    push_aa = st.three2 > t.bb_fold
    w_aa = deal_w * pbtn * psa

    for hist, push_m, w_m in (
        (0, push_fa, w_fa),
        (1, push_af, w_af),
        (2, push_aa, w_aa),
    ):
        for k in range(n):
            wt = w_m[:, :, k].sum()
            if wt > 1e-12:
                out[hist, k] = float((w_m[:, :, k] * push_m[:, :, k]).sum() / wt)
    return np.clip(out, 0.0, 1.0)


def best_responses_3max(
    st: StaticPayoffs,
    deal_w: np.ndarray,
    btn_push: np.ndarray,
    sb_push: np.ndarray,
    bb_push: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        best_response_btn_3max(st, deal_w, sb_push, bb_push),
        best_response_sb_3max(st, deal_w, btn_push, bb_push),
        best_response_bb_3max(st, deal_w, btn_push, sb_push),
    )


def _always_policy_3max(mode: str, n: int = 169) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    v = 1.0 if mode == "push" else 0.0
    btn = np.full(n, v, dtype=np.float64)
    sb = np.full((2, n), v, dtype=np.float64)
    bb = np.full((3, n), v, dtype=np.float64)
    return btn, sb, bb


def trivial_baselines_3max(
    st: StaticPayoffs, deal_w: np.ndarray
) -> dict[str, float]:
    """CLAIRVOYANT trivial baselines (mbb/g) — DIAGNOSTIC; see ``trivial_baselines_3max_legal``."""
    gaps = {}
    for name, mode in (("always_push", "push"), ("always_fold", "fold")):
        btn, sb, bb = _always_policy_3max(mode, n=deal_w.shape[0])
        gap, _ = nash_gap_3max(st, deal_w, btn, sb, bb)
        gaps[name] = gap * 1000.0
    return gaps


def trivial_baselines_3max_legal(
    st: StaticPayoffs, deal_w: np.ndarray
) -> dict[str, float]:
    """Legal (marginalized) trivial baselines (mbb/g) for always-push / always-fold."""
    gaps = {}
    for name, mode in (("always_push", "push"), ("always_fold", "fold")):
        btn, sb, bb = _always_policy_3max(mode, n=deal_w.shape[0])
        gap, _ = nash_gap_3max_legal(st, deal_w, btn, sb, bb)
        gaps[name] = gap * 1000.0
    return gaps


def load_3max_payoffs(
    shares: np.ndarray,
    pair_eq: np.ndarray,
    deal_w: np.ndarray | None = None,
    *,
    stack_bb: float = 8.0,
    pair_eq_source: str = "tensor",
) -> tuple[StaticPayoffs, np.ndarray]:
    if deal_w is None:
        deal_w = trio_deal_weights(shares.shape[0])
    pe = resolve_pair_eq_for_static(shares.shape[0], pair_eq, source=pair_eq_source)
    return build_static_payoffs(shares, pe, stack_bb=stack_bb), deal_w


def resolve_pair_eq_for_static(
    n: int,
    pair_eq_tensor: np.ndarray,
    *,
    source: str = "tensor",
) -> np.ndarray:
    """Select pair_eq volume for build_static_payoffs (isolated experiments)."""
    if source == "tensor":
        return np.asarray(pair_eq_tensor, dtype=np.float64)
    if source == "hu_expand":
        from aof_cfr.trio_pair_eq_table import hu_pair_eq_values, require_pair_eq_table

        hu = require_pair_eq_table()
        out = np.empty((n, n, n, 3), dtype=np.float64)
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    out[i, j, k] = hu_pair_eq_values(i, j, k, hu)
        return out
    if source == "exact_native":
        from aof_cfr.trio_pair_eq_table import load_or_build_exact_pair_eq_cube

        return load_or_build_exact_pair_eq_cube(n=n)
    raise ValueError(f"unknown pair_eq_source={source!r}")
