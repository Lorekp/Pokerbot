"""Routed-policy arrays for the arena — one analytic BR pass, vectorized over (history, hand).

Generalizes ``scripts/gauntlet_tournament.py::_routed_arrays`` with a pluggable estimator and
confidence function so the SAME code path evaluates:

  * the legacy frozen-profile arm (``estimate_policy_rows_shift`` + DB n_obs confidence), and
  * the SHIPPING arm (``AdaptiveOpponentModel.as_estimator()`` + its population-floored,
    decay-aware confidence) — the configuration ``aof_engine.AoFEngine`` actually wires, which
    had never been through the gauntlet before this suite.

Also exposes the BASE marginals per (seat, history) — the exact per-action value function used as
the AIVAT decision-term baseline (any baseline is unbiased; the analytic one is maximally
correlated with the outcome, which is what buys the variance reduction).
"""

from __future__ import annotations

import numpy as np

from aof_cfr.buckets import LABELS_169
from aof_cfr.profiler import base_policy_rows, estimate_policy_rows_shift
from aof_ranges import active_seats


def hists_for(n: int, seat: str):
    """Decision histories a seat actually faces (phantom uncontested-walk node excluded)."""
    from aof_cfr.fp_4max import _histories_before
    seats = active_seats(n)
    si = seats.index(seat)
    phantom = "F" * si if si == n - 1 else None
    return tuple(h for h in _histories_before(si) if h != phantom)


def _seat_arr(rows, n, seat, stack_bb):
    from aof_cfr.exploit_router import _seat_array, _seat_array_hu
    return _seat_array_hu(rows, seat, stack_bb) if n == 2 else _seat_array(rows, seat, stack_bb)


def _agent_base_array(n, agent_seat, stack_bb) -> np.ndarray:
    import aof_cfr.query as q
    return np.array(
        [[q.base_strategy(n, agent_seat, h, lab, stack_bb) for lab in LABELS_169]
         for h in hists_for(n, agent_seat)],
        dtype=np.float64,
    )


def marginals_for(n: int, agent_seat: str, stack_bb: float, seat_arrays: dict) -> dict:
    """(e_push(169,), e_fold) per history for ``agent_seat`` given per-seat push arrays."""
    if n == 2:
        from aof_cfr.exploit_router import _MARG_KEY_HU
        from aof_cfr.hu_br_legal import _legal_marginals_hu, load_hu_eq
        m = _legal_marginals_hu(seat_arrays["SB"], seat_arrays["BB"], stack_bb, load_hu_eq())
        return {h: m[_MARG_KEY_HU[(agent_seat, h)]][:2] for h in hists_for(n, agent_seat)}
    if n == 3:
        from aof_cfr.exploit_router import _MARG_KEY, _payoffs_3max
        from aof_cfr.fp_br import _legal_marginals_3max
        st, dw = _payoffs_3max(stack_bb)
        hs = hists_for(n, agent_seat)
        need = tuple(_MARG_KEY[(agent_seat, h)] for h in hs)
        m = _legal_marginals_3max(st, dw, seat_arrays["BTN"], seat_arrays["SB"],
                                  seat_arrays["BB"], keys=need)
        return {h: m[_MARG_KEY[(agent_seat, h)]][:2] for h in hs}
    # 4-max: vectorized over the cached deal batch; seat_arrays is a rows->table policy instead.
    raise ValueError("use marginals_4max for n=4")


def marginals_4max(agent_seat: str, stack_bb: float, table, *, n_deals=8000, seed=12345) -> dict:
    from aof_cfr.exploit_router import _SEATS_4MAX, _router_batch
    from aof_cfr.fp_4max_vec import _br_marginals
    buckets, payoffs, infosets = _router_batch(stack_bb, n_deals, seed)
    num_push, num_fold, den = _br_marginals(table, buckets, payoffs, infosets)
    idx = _SEATS_4MAX.index(agent_seat)
    out = {}
    for hist in hists_for(4, agent_seat):
        d = den[(idx, hist)]
        safe = np.where(d > 0, d, 1.0)
        out[hist] = (num_push[(idx, hist)] / safe, num_fold[(idx, hist)] / safe)
    return out


def base_marginals(n: int, agent_seat: str, stack_bb: float) -> dict:
    """AIVAT baseline: the agent's (e_push, e_fold) per history under an all-BASE table."""
    base_rows = base_policy_rows(n, stack_bb)
    if n == 4:
        from aof_cfr.fp_4max_vec import rows_to_table
        return marginals_4max(agent_seat, stack_bb, rows_to_table(base_rows, stack_bb))
    arrays = {s: _seat_arr(base_rows, n, s, stack_bb) for s in active_seats(n)}
    return marginals_for(n, agent_seat, stack_bb, arrays)


def routed_arrays(db, n, agent_seat, stack_bb, opp_at_seat, *, min_conf, margin,
                  estimator=None, confidence_fn=None,
                  confidence_mode: str | None = None, gate_mode: str | None = None) -> dict:
    """``{history: (p_push(169,), is_exploit(169,), gain(169,))}`` for the agent.

    Vectorized mirror of the router's gating over every (history, hand) at once.
    ``estimator``/``confidence_fn`` default to the legacy DB shift path; pass the adaptive
    model's hooks for the shipping configuration. ``confidence_mode``/``gate_mode`` default to
    the router's shipping defaults — pass ("min", "point") to reproduce the legacy gauntlet
    behavior (the frozen pre-Phase-3 baseline)."""
    from aof_cfr.exploit_router import (
        DEFAULT_CONFIDENCE_MODE, DEFAULT_GATE_MODE, _confidence, _kappa, _n_eff, lcb_thresholds)

    confidence_mode = confidence_mode or DEFAULT_CONFIDENCE_MODE
    gate_mode = gate_mode or DEFAULT_GATE_MODE
    est = estimator or (lambda d, ident, np_, s: estimate_policy_rows_shift(d, ident, np_, s))
    if confidence_fn is None:
        confidence_fn = lambda ident: _confidence(db, ident, 8.0)  # noqa: E731
    prior_strength = 8.0
    base_rows = base_policy_rows(n, stack_bb)
    hists = hists_for(n, agent_seat)

    # per-opponent estimate objects + confidences (None ident -> base, no read)
    seats = active_seats(n)
    est_by_seat, conf_by_seat = {}, {}
    legacy_confs = []
    for seat in seats:
        if seat == agent_seat:
            continue
        ident = opp_at_seat.get(seat)
        if ident:
            est_by_seat[seat] = est(db, ident, n, stack_bb)
            conf_by_seat[seat] = float(confidence_fn(ident))
            legacy_confs.append(conf_by_seat[seat])
        else:
            legacy_confs.append(0.0)

    if n == 4:
        from aof_cfr.exploit_router import _SEATS_4MAX
        from aof_cfr.fp_4max_vec import rows_to_table

        base_tab = rows_to_table(base_rows, stack_bb)

        def _marg(subset: dict, kappas: dict | None = None):
            from aof_cfr.fp_4max_vec import _Policy
            tab = _Policy(dict(base_tab.table))
            for seat, rows in subset.items():
                si = _SEATS_4MAX.index(seat)
                et = rows_to_table(rows, stack_bb)
                for k, arr in et.table.items():
                    if k[0] == si and k in tab.table:
                        kk = 1.0 if kappas is None else kappas[seat]
                        tab.table[k] = base_tab.table[k] + kk * (arr - base_tab.table[k])
            return marginals_4max(agent_seat, stack_bb, tab)

        def _opp_arrs(seat, rows):
            si = _SEATS_4MAX.index(seat)
            et = rows_to_table(rows, stack_bb)
            pairs = [(base_tab.table[k], arr) for k, arr in et.table.items()
                     if k[0] == si and k in base_tab.table]
            return [b for b, _e in pairs], [e for _b, e in pairs]
    else:
        base_arrs = {s: _seat_arr(base_rows, n, s, stack_bb) for s in seats}

        def _marg(subset: dict, kappas: dict | None = None):
            arrs = dict(base_arrs)
            for seat, rows in subset.items():
                e = _seat_arr(rows, n, seat, stack_bb)
                kk = 1.0 if kappas is None else kappas[seat]
                arrs[seat] = base_arrs[seat] + kk * (e - base_arrs[seat])
            return marginals_for(n, agent_seat, stack_bb, arrs)

        def _opp_arrs(seat, rows):
            return [base_arrs[seat]], [_seat_arr(rows, n, seat, stack_bb)]

    joint = _marg(est_by_seat)
    base_seat = _agent_base_array(n, agent_seat, stack_bb)

    # decision confidence per (hist, hand)
    if confidence_mode == "min" or not est_by_seat:
        conf = min(legacy_confs) if legacy_confs else 0.0
        conf_arr = {hist: np.full(len(LABELS_169), conf) for hist in hists}
    else:
        solo = {seat: _marg({seat: est_by_seat[seat]}) for seat in est_by_seat}
        conf_arr = {}
        for hi, hist in enumerate(hists):
            base_p = base_seat[hi]
            num = np.zeros(len(LABELS_169))
            den = np.zeros(len(LABELS_169))
            for seat, m_o in solo.items():
                ep_o, ef_o = m_o[hist]
                g = np.maximum(np.maximum(ef_o, ep_o)
                               - ((1.0 - base_p) * ef_o + base_p * ep_o), 0.0)
                num += g * conf_by_seat[seat]
                den += g
            fallback = max(conf_by_seat.values(), default=0.0)
            conf_arr[hist] = np.where(den > 1e-12, num / np.maximum(den, 1e-12), fallback)

    # gate
    out = {}
    if gate_mode == "point" or not est_by_seat:
        for hi, hist in enumerate(hists):
            e_push, e_fold = joint[hist]
            base_p = base_seat[hi]
            exploit_p = (e_push > e_fold).astype(np.float64)
            gain = np.maximum(e_fold, e_push) - ((1.0 - base_p) * e_fold + base_p * e_push)
            c = conf_arr[hist]
            gated = (c >= min_conf) & (gain >= margin)
            p = np.where(gated, (1.0 - c) * base_p + c * exploit_p, base_p)
            out[hist] = (p, gated, gain)
        return out

    kappas = {seat: _kappa(*_opp_arrs(seat, rows), _n_eff(conf_by_seat[seat], prior_strength),
                           prior_strength)
              for seat, rows in est_by_seat.items()}
    lcb = _marg(est_by_seat, kappas)
    eff_min_conf, eff_margin = lcb_thresholds(min_conf, margin)
    for hi, hist in enumerate(hists):
        e_push, e_fold = joint[hist]
        ep_l, ef_l = lcb[hist]
        base_p = base_seat[hi]
        exploit_p = (e_push > e_fold).astype(np.float64)
        gain = np.maximum(e_fold, e_push) - ((1.0 - base_p) * e_fold + base_p * e_push)
        v_act = np.where(exploit_p >= 0.5, ep_l, ef_l * np.ones_like(ep_l))
        gain_lcb = v_act - ((1.0 - base_p) * ef_l + base_p * ep_l)
        c = conf_arr[hist]
        gated = (c >= eff_min_conf) & (gain_lcb >= eff_margin)
        p = np.where(gated, (1.0 - c) * base_p + c * exploit_p, base_p)
        out[hist] = (p, gated, gain)
    return out
