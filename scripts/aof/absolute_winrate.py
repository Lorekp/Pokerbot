"""HONEST absolute bb/100 for the AoF base — seat-rotated, rake-inclusive, exact.

Unlike ``scripts/eval_gauntlet.py`` (reports only the positional-winner seat) and the eval
suite (``paired_arena`` subtracts the table mean -> reports edge-over-field, which factors the
common rake drag OUT), this script reports the number that actually decides "are we losing
money": the average of all three seat values for our agent, played in every seat, WITH the
flat showdown fee included in the pot.

Exact analytic 3-max values via ``fp_br.seat_values_3max`` over the bucket deal (zero sampling
variance). Sweeps stack depth x rake x opponent field, for our agent playing the GTO base vs
playing a pure best response (the "exploitation ON" upper bound).

Run:  python scripts/aof/absolute_winrate.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np

from aof_cfr.buckets import LABELS_169

PREMIUM = {"AA", "KK", "QQ", "JJ", "AKs", "AKo"}
PREM = np.array([1.0 if h in PREMIUM else 0.0 for h in LABELS_169])


def _set_fee(fee_bb: float) -> None:
    """Force the payoff layer to a specific flat fee (clears the module cache)."""
    from aof_cfr import pushfold_payoffs as pf
    os.environ["AOF_FLAT_FEE_BB"] = str(fee_bb)
    os.environ["AOF_GAME_MODEL"] = "2"
    pf._GM_CACHE.clear()


_BASE_CACHE: dict = {}


def _base_arrays(stack_bb: float):
    """Policy rows are fee-independent NUMBERS; load once under the stamped fee (0.12) to pass
    the base stamp guard, then cache so we never reload at a mismatched runtime fee."""
    key = round(float(stack_bb), 3)
    if key in _BASE_CACHE:
        return _BASE_CACHE[key]
    _set_fee(0.12)  # match the npz stamp so load_cfr_base() accepts it
    import aof_cfr.query as q
    q.load_cfr_base()
    btn = np.array([q.base_strategy(3, "BTN", "-", h, stack_bb) for h in LABELS_169])
    sb = np.stack([[q.base_strategy(3, "SB", hh, h, stack_bb) for h in LABELS_169] for hh in ("F", "A")])
    bb = np.stack([[q.base_strategy(3, "BB", hh, h, stack_bb) for h in LABELS_169]
                   for hh in ("FA", "AF", "AA")])
    _BASE_CACHE[key] = (btn, sb, bb)
    return btn, sb, bb


def _nit_arrays():
    """Only-premium jammer at every infoset (open AND facing a jam)."""
    btn = PREM.copy()
    sb = np.stack([PREM.copy(), PREM.copy()])
    bb = np.stack([PREM.copy(), PREM.copy(), PREM.copy()])
    return btn, sb, bb


def _station_arrays(base_btn, base_sb, base_bb):
    """Opens base; calls a jam far too wide (0.85)."""
    btn = base_btn.copy()
    sb = base_sb.copy(); bb = base_bb.copy()
    sb[1] = 0.85
    bb[:] = 0.85
    return btn, sb, bb


def _best_response(st, dw, opp_btn, opp_sb, opp_bb):
    """Pure legal best-response hero rows at every seat vs a fixed opponent field.

    Each seat's action values depend only on the OTHER two seats, so we read the hero
    seat's marginals from a table where the other two seats hold the opponent field.
    """
    from aof_cfr.fp_br import _legal_marginals_3max
    m = _legal_marginals_3max(st, dw, opp_btn, opp_sb, opp_bb)

    def br(key):
        ep, ef, _w = m[key]
        ef = np.full_like(ep, ef) if np.ndim(ef) == 0 else ef
        return (ep > ef).astype(np.float64)

    br_btn = br("btn")
    br_sb = np.stack([br("sb_f"), br("sb_a")])
    br_bb = np.stack([br("bb_fa"), br("bb_af"), br("bb_aa")])
    return br_btn, br_sb, br_bb


def _rotated_value(st, dw, hero, field, *, hero_is_br=False):
    """Average bb/100 for hero played in all three seats vs ``field`` opponents.

    hero/field are (btn, sb, bb) tuples. For each seat we place hero's row at that seat,
    the field rows at the other two, and read that seat's exact value.
    """
    from aof_cfr.fp_br import seat_values_3max
    h_btn, h_sb, h_bb = hero
    f_btn, f_sb, f_bb = field
    vals = []
    # BTN
    vals.append(seat_values_3max(st, dw, h_btn, f_sb, f_bb)[0])
    # SB
    vals.append(seat_values_3max(st, dw, f_btn, h_sb, f_bb)[1])
    # BB
    vals.append(seat_values_3max(st, dw, f_btn, f_sb, h_bb)[2])
    return float(np.mean(vals)) * 100.0, [v * 100.0 for v in vals]


def evaluate(stack_bb: float, fee_bb: float) -> dict:
    _set_fee(fee_bb)
    from aof_cfr.fp_br import load_3max_payoffs, seat_values_3max
    from aof_cfr.trio_payoff_tensor import DEFAULT_TENSOR, load_trio_tensor
    shares, pair_eq, deal_w = load_trio_tensor(DEFAULT_TENSOR)
    st, dw = load_3max_payoffs(shares, pair_eq, deal_w, stack_bb=stack_bb)

    base = _base_arrays(stack_bb)
    nit = _nit_arrays()
    station = _station_arrays(*base)

    # Sanity: all-GTO table average over seats == the per-seat rake split (small negative).
    all_gto = list(seat_values_3max(st, dw, *base))
    rake_drag = float(np.mean(all_gto)) * 100.0

    out = {"stack_bb": stack_bb, "fee_bb": fee_bb, "all_gto_rake_drag": rake_drag}

    # vs GTO field: our base (should ~match rake drag) and our BR (should be ~0 deviation).
    br_vs_gto = _best_response(st, dw, *base)
    out["base_vs_gto"], _ = _rotated_value(st, dw, base, base)
    out["br_vs_gto"], _ = _rotated_value(st, dw, br_vs_gto, base)

    # vs NIT field (the user's observation: opponents only jam premium).
    br_vs_nit = _best_response(st, dw, *nit)
    out["base_vs_nit"], out["base_vs_nit_seats"] = _rotated_value(st, dw, base, nit)
    out["br_vs_nit"], out["br_vs_nit_seats"] = _rotated_value(st, dw, br_vs_nit, nit)

    # vs CALLING-STATION field.
    br_vs_stn = _best_response(st, dw, *station)
    out["base_vs_station"], _ = _rotated_value(st, dw, base, station)
    out["br_vs_station"], _ = _rotated_value(st, dw, br_vs_stn, station)
    return out


# ----------------------------------------------------------------------------------------------
# 4-max (UTG, BTN, SB, BB) — Monte-Carlo deal batch (small residual deal variance; equity is
# expected over boards in the payoff precompute). Same seat-rotation + best-response logic.
# ----------------------------------------------------------------------------------------------
N_DEALS_4MAX = 15000
SEED_4MAX = 99
_DEALS_CACHE: dict = {}
_PAY_CACHE: dict = {}     # stack -> (payoffs@fee0, slope) ; payoff is EXACTLY linear in the fee
_BASE4_CACHE: dict = {}   # stack -> base rows dict (loaded under the stamped fee 0.12)


def _deals_4max():
    key = (N_DEALS_4MAX, SEED_4MAX)
    if key not in _DEALS_CACHE:
        from aof_cfr.fp_4max_vec import sample_deals
        _DEALS_CACHE[key] = sample_deals(N_DEALS_4MAX, SEED_4MAX)
    return _DEALS_CACHE[key]


def _payoffs_4max(stack_bb: float, fee_bb: float):
    """Payoffs at any fee via exact linear interpolation (pot is linear in fee, payoff in pot)."""
    from aof_cfr.fp_4max_vec import precompute_payoffs
    key = round(float(stack_bb), 3)
    if key not in _PAY_CACHE:
        _, holes = _deals_4max()
        _set_fee(0.0); p0 = precompute_payoffs(holes, stack_bb, game_model=2)
        _set_fee(1.0); p1 = precompute_payoffs(holes, stack_bb, game_model=2)
        _PAY_CACHE[key] = (p0, p1 - p0)
    p0, slope = _PAY_CACHE[key]
    return p0 + float(fee_bb) * slope


def _base_table_4max(stack_bb: float):
    from aof_cfr.fp_4max_vec import rows_to_table
    from aof_cfr.profiler import base_policy_rows
    key = round(float(stack_bb), 3)
    if key not in _BASE4_CACHE:
        _set_fee(0.12)  # match the npz stamp so load_cfr_base() accepts it
        _BASE4_CACHE[key] = base_policy_rows(4, stack_bb)
    return rows_to_table(_BASE4_CACHE[key], stack_bb)


def _nit_table_4max(infosets):
    from aof_cfr.fp_4max_vec import _Policy
    t = _Policy()
    for k in infosets:
        t.table[k] = PREM.copy()
    return t


def _station_table_4max(base_table, infosets):
    from aof_cfr.fp_4max_vec import _Policy
    t = _Policy()
    for (s, h) in infosets:
        t.table[(s, h)] = np.full(len(LABELS_169), 0.85) if "A" in h else base_table.get(s, h).copy()
    return t


def _br_table_4max(field_table, buckets, payoffs, infosets):
    from aof_cfr.fp_4max_vec import _Policy, _br_marginals
    np_push, np_fold, _den = _br_marginals(field_table, buckets, payoffs, infosets)
    br = _Policy()
    for k in infosets:
        br.table[k] = (np_push[k] > np_fold[k]).astype(np.float64)
    return br


def _rotated_value_4max(infosets, buckets, payoffs, hero_table, field_table):
    """Average bb/100 for hero played in all 4 seats vs ``field_table`` opponents."""
    from aof_cfr.fp_4max_vec import _Policy, onpolicy_seat_ev
    vals = []
    for i in range(4):
        mixed = _Policy()
        for (s, h) in infosets:
            src = hero_table if s == i else field_table
            mixed.table[(s, h)] = src.get(s, h)
        vals.append(onpolicy_seat_ev(mixed, buckets, payoffs)[i])
    return float(np.mean(vals)) * 100.0, [v * 100.0 for v in vals]


def evaluate_4max(stack_bb: float, fee_bb: float) -> dict:
    """SAMPLE-SPLIT to debias the MC best response: the BR action is chosen on deal-set A and
    every value (base AND BR) is scored on the disjoint set B. Without this, BR-vs-GTO reads a
    spurious +2-4 bb/100 (max-of-noisy-estimates), the same bias ``legal_nash_gap_4max`` splits
    out. base/* are unbiased either way; scoring all arms on B keeps base-vs-BR comparable."""
    from aof_cfr.fp_4max_vec import _infoset_index, onpolicy_seat_ev
    base_tab = _base_table_4max(stack_bb)          # loads base rows under fee 0.12
    buckets, _holes = _deals_4max()
    payoffs = _payoffs_4max(stack_bb, fee_bb)
    infosets = _infoset_index(buckets)
    nit = _nit_table_4max(infosets)
    station = _station_table_4max(base_tab, infosets)

    n = buckets.shape[0]
    m = n // 2
    bA, pA = buckets[:m], payoffs[:, :m, :]        # choose BR action here
    bB, pB = buckets[m:], payoffs[:, m:, :]        # score every arm here

    rake_drag = float(np.mean(onpolicy_seat_ev(base_tab, bB, pB))) * 100.0
    out = {"stack_bb": stack_bb, "fee_bb": fee_bb, "all_gto_rake_drag": rake_drag}

    def arm(field):
        br = _br_table_4max(field, bA, pA, infosets)         # BR on A
        base_v, base_s = _rotated_value_4max(infosets, bB, pB, base_tab, field)  # score on B
        br_v, br_s = _rotated_value_4max(infosets, bB, pB, br, field)
        return base_v, br_v, base_s, br_s

    out["base_vs_gto"], out["br_vs_gto"], _, _ = arm(base_tab)
    out["base_vs_nit"], out["br_vs_nit"], out["base_vs_nit_seats"], out["br_vs_nit_seats"] = arm(nit)
    out["base_vs_station"], out["br_vs_station"], _, _ = arm(station)
    return out


def _print_sweep(title, eval_fn, fees, stacks, seat_names):
    print(title)
    for fee, label in fees.items():
        print(f"################  FEE = {fee:.2f} bb  [{label}]  ################")
        print(f"{'stack':>6} | {'rakeDrag':>9} | {'base/GTO':>9} {'BR/GTO':>8} | "
              f"{'base/NIT':>9} {'BR/NIT':>8} | {'base/STN':>9} {'BR/STN':>8}")
        for s in stacks:
            r = eval_fn(s, fee)
            print(f"{int(s):>5}b | {r['all_gto_rake_drag']:>+9.2f} | "
                  f"{r['base_vs_gto']:>+9.2f} {r['br_vs_gto']:>+8.2f} | "
                  f"{r['base_vs_nit']:>+9.2f} {r['br_vs_nit']:>+8.2f} | "
                  f"{r['base_vs_station']:>+9.2f} {r['br_vs_station']:>+8.2f}")
        print()
    r = eval_fn(8.0, 0.16)
    print("Per-seat detail @ 8bb, fee 0.16 (LIVE 0.10/0.25), vs NIT field:")
    for seat, b, e in zip(seat_names, r["base_vs_nit_seats"], r["br_vs_nit_seats"]):
        print(f"  {seat:<4}:  base {b:+7.2f}   BR(exploit) {e:+7.2f}   uplift {e - b:+6.2f} bb/100")
    print()


def main() -> int:
    from aof_ranges import active_seats
    fees = {0.0: "no-rake", 0.12: "0.25/0.50 (base solved)", 0.16: "0.10/0.25 (LIVE)",
            0.22: "0.10/0.25 (DC2 capture)"}
    stacks = (6.0, 8.0, 10.0, 12.0)

    _print_sweep(
        "HONEST absolute bb/100 — seat-rotated (avg of all 3 seats), rake INCLUDED.\n"
        "Positive = winning chips; negative = losing chips. 3-max, EXACT analytic.\n",
        evaluate, fees, stacks, ("BTN", "SB", "BB"))

    _print_sweep(
        f"================ 4-max (UTG/BTN/SB/BB), MC {N_DEALS_4MAX} deals, seed {SEED_4MAX} "
        "(small deal variance) ================\n"
        "Seat-rotated (avg of all 4 seats), rake INCLUDED.\n",
        evaluate_4max, fees, stacks, [s for s in active_seats(4)])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


if __name__ == "__main__":
    raise SystemExit(main())
