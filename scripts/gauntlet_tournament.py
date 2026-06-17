"""Gauntlet tournament: prove the exploitation uplift in bb/100 (train/test split).

DEPRECATED for certification — superseded by ``aof_cfr.evalsuite`` (scripts/run_aof_eval_suite.py),
which adds multi-seed CIs, seat rotation, an AIVAT decision term, the SHIPPING adaptive-online
arm, and fixes this script's field-composition bug (``bots[i % len(bots)]`` keyed by SLOT means
HU "mixed" fields only ever played the overfolder, and 3-max never fielded depth_blind). This
script is kept byte-compatible so its historical runs/gauntlet_tournament.json and tests remain
reproducible; do NOT quote its numbers for new work.

The agent first OBSERVES a field of bots (near-GTO "Pluribus-level" and leaky archetypes) for a
warmup phase, building a persistent profile of every opponent. It then precomputes its routed
exploit policy ONCE from that frozen profile (the confidence-gated legal best response), and a
paired test phase measures

    exploit_uplift_bb100 = bb/100(DB-on, exploit policy) - bb/100(GTO-only base)

over the same test deals. This mirrors the real loop ("after observing player X, exploit X") and
is fast: the heavy analytic BR (fp_br._legal_marginals_3max, the HU/4-max analogues) runs once per
agent seat, not per hand. Sanity: uplift ~ 0 and 0 exploit spots vs an all-GTO field.

Run:  python scripts/gauntlet_tournament.py
      python scripts/gauntlet_tournament.py --players 3 --stack-bb 8 --hands 5000
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from aof_cfr.buckets import LABEL_TO_INDEX, LABELS_169, combo_to_bucket
from aof_cfr.combo_deal import FULL_DECK, card_id
from aof_cfr.exploit_mc import _terminal_returns
from aof_cfr.exploit_router import DEFAULT_MIN_CONFIDENCE, MARGINAL_EDGE_BB, _confidence
from aof_cfr.infoset import aggregate_infoset_key
from aof_cfr.live_exploit import record_observed_decision
from aof_cfr.profiler import base_policy_rows, estimate_policy_rows_shift
from aof_ranges import active_seats

ACTION_FOLD, ACTION_ALL_IN = 0, 1
PREMIUM = {"AA", "KK", "QQ", "JJ", "AKs", "AKo"}


def _hists(n, seat):
    """Decision histories a seat actually faces (action-order + table-size dependent).

    Canonical source: fp_4max._histories_before + the phantom (uncontested last seat) skip,
    so it matches the infoset enumeration the solvers / BR marginals use. E.g. BTN faces "-" in
    3-max (acts first) but "F"/"A" in 4-max (acts after UTG).
    """
    from aof_cfr.fp_4max import _histories_before
    seats = active_seats(n)
    si = seats.index(seat)
    phantom = "F" * si if si == n - 1 else None
    return tuple(h for h in _histories_before(si) if h != phantom)


def _seat_arr(rows, n, seat, stack_bb):
    from aof_cfr.exploit_router import _seat_array, _seat_array_hu
    return _seat_array_hu(rows, seat, stack_bb) if n == 2 else _seat_array(rows, seat, stack_bb)


@dataclass
class Bot:
    name: str
    push_fn: object  # f(q, n, pos, hist, hand, stack_bb) -> p_push
    leaky: bool


def _gto(q, n, pos, hist, hand, stack_bb):
    return q.base_strategy(n, pos, hist, hand, stack_bb)


def _overfolder(q, n, pos, hist, hand, stack_bb):
    if "A" in hist:                                  # facing a jam: call premiums only
        return 1.0 if hand in PREMIUM else 0.0
    return q.base_strategy(n, pos, hist, hand, stack_bb)


def _calling_station(q, n, pos, hist, hand, stack_bb):
    if "A" in hist:                                  # facing a jam: call far too wide
        return 0.85
    return q.base_strategy(n, pos, hist, hand, stack_bb)


def _depth_blind(q, n, pos, hist, hand, stack_bb):
    return q.base_strategy(n, pos, hist, hand, 8.0)  # ignores true depth (leaks off 8bb)


GTO = Bot("gto", _gto, leaky=False)
OVERFOLDER = Bot("overfolder", _overfolder, leaky=True)
CALLING_STATION = Bot("calling_station", _calling_station, leaky=True)
DEPTH_BLIND = Bot("depth_blind", _depth_blind, leaky=True)

FIELDS = {
    "all_leaky": [OVERFOLDER, CALLING_STATION, DEPTH_BLIND],
    "mixed": [OVERFOLDER, GTO, CALLING_STATION],
    "all_gto": [GTO, GTO, GTO],
}


def _deal(n, rng):
    deck = list(FULL_DECK)
    rng.shuffle(deck)
    return [(deck[2 * i], deck[2 * i + 1]) for i in range(n)]


def _min_opp_conf(db, n, agent_seat, opp_at_seat):
    """Min calibrated confidence over the profiled opponents (0 if any unknown)."""
    confs = [(_confidence(db, opp_at_seat.get(s), 8.0) if opp_at_seat.get(s) else 0.0)
             for s in active_seats(n) if s != agent_seat]
    return min(confs) if confs else 0.0


def _profiled_seat_arrays(db, n, agent_seat, stack_bb, opp_at_seat, base_rows):
    """Per-seat push arrays for the 2/3-max marginal fns (agent=base, opp=range-width shift)."""
    arrays = {}
    for seat in active_seats(n):
        if seat != agent_seat and opp_at_seat.get(seat):
            est = estimate_policy_rows_shift(db, opp_at_seat[seat], n, stack_bb)
            arrays[seat] = _seat_arr(est, n, seat, stack_bb)
        else:
            arrays[seat] = _seat_arr(base_rows, n, seat, stack_bb)
    return arrays


def _agent_base_array(n, agent_seat, stack_bb):
    """Agent's GTO base push freq per (history, hand): shape (n_hists, 169). Works for all n."""
    import aof_cfr.query as q
    return np.array(
        [[q.base_strategy(n, agent_seat, h, lab, stack_bb) for lab in LABELS_169]
         for h in _hists(n, agent_seat)],
        dtype=np.float64,
    )


def _routed_arrays(db, n, agent_seat, stack_bb, opp_at_seat, *, min_conf, margin):
    """Precompute the agent's routed push prob + exploit mask per (history, hand) at ``agent_seat``.

    One analytic BR pass (not per hand). Returns ``{history: (push(169,), is_exploit(169,))}``.
    """
    base_rows = base_policy_rows(n, stack_bb)
    conf = _min_opp_conf(db, n, agent_seat, opp_at_seat)
    if n == 2:
        from aof_cfr.exploit_router import _MARG_KEY_HU
        from aof_cfr.hu_br_legal import _legal_marginals_hu, load_hu_eq
        arrays = _profiled_seat_arrays(db, n, agent_seat, stack_bb, opp_at_seat, base_rows)
        m = _legal_marginals_hu(arrays["SB"], arrays["BB"], stack_bb, load_hu_eq())
        marg = {h: m[_MARG_KEY_HU[(agent_seat, h)]] for h in _hists(n, agent_seat)}
    elif n == 3:
        from aof_cfr.exploit_router import _MARG_KEY, _payoffs_3max
        from aof_cfr.fp_br import _legal_marginals_3max
        arrays = _profiled_seat_arrays(db, n, agent_seat, stack_bb, opp_at_seat, base_rows)
        st, dw = _payoffs_3max(stack_bb)
        mm = _legal_marginals_3max(st, dw, arrays["BTN"], arrays["SB"], arrays["BB"])
        marg = {h: mm[_MARG_KEY[(agent_seat, h)]] for h in _hists(n, agent_seat)}
    else:
        marg = _marg_4max(db, agent_seat, stack_bb, opp_at_seat)

    out = {}
    base_seat = _agent_base_array(n, agent_seat, stack_bb)
    for hi, hist in enumerate(_hists(n, agent_seat)):
        e_push, e_fold, _w = marg[hist]
        base_p = base_seat[hi]
        exploit_p = (e_push > e_fold).astype(np.float64)
        base_ev = (1.0 - base_p) * e_fold + base_p * e_push
        gain = np.maximum(e_fold, e_push) - base_ev
        gated = (conf >= min_conf) & (gain >= margin)
        p = np.where(gated, (1.0 - conf) * base_p + conf * exploit_p, base_p)
        out[hist] = (p, gated)
    return out


def _marg_4max(db, agent_seat, stack_bb, opp_at_seat, *, n_deals=8000, seed=12345):
    """4-max BR marginals per (history) at agent_seat from the frozen profile (one MC pass)."""
    from aof_cfr.exploit_router import _SEATS_4MAX, _router_batch
    from aof_cfr.fp_4max_vec import _br_marginals, rows_to_table
    buckets, payoffs, infosets = _router_batch(stack_bb, n_deals, seed)
    table = rows_to_table(base_policy_rows(4, stack_bb), stack_bb)
    for si, seat in enumerate(_SEATS_4MAX):
        ident = opp_at_seat.get(seat)
        if seat != agent_seat and ident:
            est = rows_to_table(estimate_policy_rows_shift(db, ident, 4, stack_bb), stack_bb)
            for (i, h) in list(table.table):
                if i == si and (i, h) in est.table:
                    table.table[(i, h)] = est.table[(i, h)]
    num_push, num_fold, den = _br_marginals(table, buckets, payoffs, infosets)
    idx = _SEATS_4MAX.index(agent_seat)
    out = {}
    for hist in _hists(4, agent_seat):
        d = den[(idx, hist)]
        safe = np.where(d > 0, d, 1.0)
        out[hist] = (num_push[(idx, hist)] / safe, num_fold[(idx, hist)] / safe, d)
    return out


def _simulate(n, seats, bots, stack_bb, n_hands, seed, *, db, routed):
    """Play hands; record bot decisions into ``db`` if given; agent uses ``routed`` else base.

    Returns (total_zero_sum_bb, n_exploit_decisions).
    """
    import aof_cfr.query as q
    total = 0.0
    n_exploit = 0
    for h in range(n_hands):
        agent_seat = seats[h % n]
        non_agent = [s for s in seats if s != agent_seat]
        seat_bot = {non_agent[i]: bots[i % len(bots)] for i in range(len(non_agent))}
        seat_id = {non_agent[i]: f"bot:{i}" for i in range(len(non_agent))}
        dr = random.Random((seed * 1_000_003) ^ h)
        cr = random.Random((seed * 7_654_321) ^ (h + 1))
        cards = _deal(n, dr)
        folded: set[int] = set()
        all_in: set[int] = set()
        for idx, seat in enumerate(seats):
            r = cr.random()
            if len([i for i in range(n) if i not in folded]) == 1:
                continue
            hist = "".join("A" if j in all_in else "F" for j in range(idx)) or "-"
            hand = combo_to_bucket(*cards[idx])
            if seat == agent_seat:
                if routed is not None and hist in routed[agent_seat]:
                    p_arr, ex_arr = routed[agent_seat][hist]
                    bi = LABEL_TO_INDEX[hand]
                    p = float(p_arr[bi])
                    if bool(ex_arr[bi]):
                        n_exploit += 1
                else:
                    p = q.base_strategy(n, seat, hist, hand, stack_bb)
            else:
                p = seat_bot[seat].push_fn(q, n, seat, hist, hand, stack_bb)
            action = ACTION_ALL_IN if r < p else ACTION_FOLD
            if db is not None and seat != agent_seat:
                record_observed_decision(db, seat_id[seat], seat, hist, stack_bb,
                                         pushed=action == ACTION_ALL_IN)
            (all_in if action == ACTION_ALL_IN else folded).add(idx)
        holes = [(card_id(c0), card_id(c1)) for (c0, c1) in cards]
        payoffs = _terminal_returns(n, seats, holes, folded, stack_bb)
        total += float(payoffs[seats.index(agent_seat)] - sum(payoffs) / n)
    return total, n_exploit


def run_tournament(n, stack_bb, n_hands, field, seed, *, db_on, warmup_hands=None):
    import aof_cfr.query as q
    q.load_cfr_base()
    seats = active_seats(n)
    bots = list(field)
    routed = None
    if db_on:
        from aof_cfr.opponent_db import AoFOpponentDB
        db = AoFOpponentDB(path=Path("data/aof_player_profiles/_gauntlet_session.json"))
        db.counts.clear(); db.seen_hands.clear()
        warmup = warmup_hands if warmup_hands is not None else max(300, n_hands // 2)
        _simulate(n, seats, bots, stack_bb, warmup, seed, db=db, routed=None)  # observe only
        routed = {}
        for a_seat in seats:
            non_agent = [s for s in seats if s != a_seat]
            opp = {non_agent[i]: f"bot:{i}" for i in range(len(non_agent))}
            routed[a_seat] = _routed_arrays(db, n, a_seat, stack_bb, opp,
                                            min_conf=DEFAULT_MIN_CONFIDENCE, margin=MARGINAL_EDGE_BB)
    total, n_exploit = _simulate(n, seats, bots, stack_bb, n_hands, seed + 99, db=None, routed=routed)
    return {"bb100": total / n_hands * 100.0, "n_exploit_decisions": n_exploit}


def run_paired(n, stack_bb, n_hands, field, seed, warmup_hands=None):
    gto = run_tournament(n, stack_bb, n_hands, field, seed, db_on=False)
    on = run_tournament(n, stack_bb, n_hands, field, seed, db_on=True, warmup_hands=warmup_hands)
    return {
        "players": n, "stack_bb": stack_bb, "hands": n_hands, "seed": seed,
        "field": [b.name for b in field],
        "bb100_gto_only": round(gto["bb100"], 2),
        "bb100_db_on": round(on["bb100"], 2),
        "exploit_uplift_bb100": round(on["bb100"] - gto["bb100"], 2),
        "n_exploit_decisions": on["n_exploit_decisions"],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--players", type=int, default=0, help="0 = sweep 2,3,4")
    ap.add_argument("--stack-bb", type=float, default=0.0, help="0 = sweep 4,8,12,20")
    ap.add_argument("--hands", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--field", default="", help="all_leaky|mixed|all_gto (default: all three)")
    args = ap.parse_args()

    players = [args.players] if args.players else [2, 3, 4]
    depths = [args.stack_bb] if args.stack_bb else [4.0, 8.0, 12.0, 20.0]
    fields = [args.field] if args.field else list(FIELDS)

    results = []
    print(f"{'n':>2} {'depth':>6} {'field':>16} {'gto_only':>9} {'db_on':>8} "
          f"{'uplift':>8} {'exploits':>9}")
    for n in players:
        for d in depths:
            for fname in fields:
                res = run_paired(n, d, args.hands, FIELDS[fname], args.seed)
                results.append(res)
                print(f"{n:>2} {d:>6.0f} {fname:>16} {res['bb100_gto_only']:>9.2f} "
                      f"{res['bb100_db_on']:>8.2f} {res['exploit_uplift_bb100']:>8.2f} "
                      f"{res['n_exploit_decisions']:>9}")

    out = _ROOT / "runs" / "gauntlet_tournament.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
