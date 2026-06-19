"""Self-contained multiway push/fold SNG simulator -- end-to-end ROI validation.

Plays full 3-handed final-table push/fold sit-n-gos (the bubble regime where ICM bites hardest:
3 left, top 2 paid) to completion and measures each strategy's real-money ROI. Isolated from the
cash engine and ``aof_*``; uses only the certified tournament solvers + the AoF card evaluator.

Correctness guards: concrete cards are dealt from a real deck and showdowns are resolved by an
actual board eval with proper SIDE POTS; a chip-conservation invariant is asserted every hand. This
is the measurement that shows the ICM engine doesn't just play unexploitably but *wins money* -- it
ladders past opponents on the bubble (folds where chip-EV players bust, shoves where they can't call).
"""

from __future__ import annotations

import numpy as np

from aof_equity_grids import _eval7
from pokerbot.tournament.decide import decide_pushfold
from pokerbot.tournament.pushfold_icm import _hu_grid, hand_to_label

_, _LABELS, _ = _hu_grid()
_LABEL_TO_IDX = {l: i for i, l in enumerate(_LABELS)}
_RANKS = "23456789TJQKA"
_SUITS = "cdhs"


def _card_str(cid: int) -> str:
    return _RANKS[cid // 4] + _SUITS[cid % 4]


def _bucket(c0: int, c1: int) -> int:
    return _LABEL_TO_IDX[hand_to_label(_card_str(c0) + _card_str(c1))]


def _settle_allin(contenders, committed, holes, board, dead=0.0) -> dict[int, float]:
    """Side-pot-correct award of an all-in among ``contenders`` (seats), each with ``committed``
    chips, given hole cards ``holes`` and a 5-card ``board``. ``dead`` = folded players' committed
    chips, which join the main pot (all contenders eligible). Returns chips won per seat (lower
    ``_eval7`` rank wins; ties split)."""
    ranks = {s: _eval7(holes[s][0], holes[s][1], *board) for s in contenders}
    won = {s: 0.0 for s in contenders}
    levels = sorted(set(committed[s] for s in contenders))
    prev = 0.0
    for li, lv in enumerate(levels):
        eligible = [s for s in contenders if committed[s] >= lv]
        layer = (lv - prev) * len(eligible) + (dead if li == 0 else 0.0)   # dead -> main pot
        best = min(ranks[s] for s in eligible)
        winners = [s for s in eligible if ranks[s] == best]
        for s in winners:
            won[s] += layer / len(winners)
        prev = lv
    return won


def _strategy_action(kind, stacks, payouts, hero, hand, *, blinds, ante, behind=None, jammer=None,
                     cache=None, field):
    """Return 'jam'/'fold' (first-in) or 'call'/'fold' (facing a jam) for a strategy ``kind``."""
    if kind == "icm":
        key = (tuple(round(s, 1) for s in stacks), hero, tuple(behind or []), jammer)
        if cache is not None and key in cache:
            d = cache[key]
        else:
            d = decide_pushfold(stacks, payouts, hero, hand, blinds=blinds, behind=behind,
                                jammer=jammer, ante=ante, pushfold_max_bb=1e9, iters=1500)
            if cache is not None:
                cache[key] = d
        if d.action in ("jam", "call"):
            return d.action
        return "fold"
    if kind == "exploit":
        # RNR-capped best response to the station's KNOWN ranges (accumulate vs gamblers, but
        # ICM-capped so it still ladders). HU-based on the effective depth; cached.
        from pokerbot.tournament.exploit_pushfold import hu_exploit_call, hu_exploit_jam
        d = round(min(s for s in stacks if s > 0), 1)
        ekey = ("ex", d, jammer is None)
        if cache is not None and ekey in cache:
            rng_tbl = cache[ekey]
        else:
            if jammer is None:
                rng_tbl = hu_exploit_jam([d, d], [1, 0], 0, 1, field["call"], rnr=0.35, iters=1200).jam
            else:
                rng_tbl = hu_exploit_call([d, d], [1, 0], 0, 1, field["jam"], rnr=0.35, iters=1200).jam
            if cache is not None:
                cache[ekey] = rng_tbl
        b = _LABEL_TO_IDX[hand]
        act = "jam" if jammer is None else "call"
        return act if rng_tbl[b] > 0.5 else "fold"
    # field strategies use fixed ranges keyed by hand bucket
    b = _LABEL_TO_IDX[hand]
    if jammer is None:
        return "jam" if field["jam"][b] > 0.5 else "fold"
    return "call" if field["call"][b] > 0.5 else "fold"


def play_hand(stacks, order, kinds, payouts, blinds_bb, ante, rng, cache, field):
    """Play one push/fold hand among the seats in ``order`` (action order). Mutates ``stacks``.
    ``order[0]`` posts SB, ``order[-1]`` posts BB (heads-up: button posts SB)."""
    n = len(order)
    deck = list(range(52))
    rng.shuffle(deck)
    holes = {s: (deck[2 * k], deck[2 * k + 1]) for k, s in enumerate(order)}
    board_cards = deck[2 * n:2 * n + 5]
    sb_seat, bb_seat = order[-2] if n > 2 else order[0], order[-1]
    blinds = {sb_seat: min(0.5, stacks[sb_seat]), bb_seat: min(1.0, stacks[bb_seat])}

    committed = {s: min(ante, stacks[s]) + blinds.get(s, 0.0) for s in order}
    for s in order:
        stacks[s] -= committed[s]
    pot_dead = sum(committed.values())

    jammers, folded = [], set()
    for s in order:                                   # one sequential pass: jam/fold or call/fold
        if s in folded:
            continue
        hand = _LABELS[_bucket(*holes[s])]
        if not jammers:                               # first-in: jam or fold
            behind = [o for o in order if o != s and o not in folded and order.index(o) > order.index(s)]
            act = _strategy_action(kinds[s], stacks_snapshot(stacks, committed), payouts, s, hand,
                                   blinds=blinds, ante=ante, behind=behind, cache=cache, field=field)
            if act == "jam":
                add = stacks[s]; stacks[s] = 0.0; committed[s] += add; jammers.append(s)
            else:
                folded.add(s)
        else:                                         # facing a jam: call or fold
            act = _strategy_action(kinds[s], stacks_snapshot(stacks, committed), payouts, s, hand,
                                   blinds=blinds, ante=ante, jammer=jammers[0], cache=cache, field=field)
            if act == "call":
                add = stacks[s]; stacks[s] = 0.0; committed[s] += add; jammers.append(s)
            else:
                folded.add(s)

    contenders = jammers
    if len(contenders) <= 1:                          # uncontested: last aggressor (or BB) wins pot
        winner = contenders[0] if contenders else _last_unfolded(order, folded)
        stacks[winner] += sum(committed.values())
    else:
        dead = sum(committed[s] for s in order if s not in contenders)
        won = _settle_allin(contenders, {s: committed[s] for s in contenders}, holes, board_cards, dead)
        for s, amt in won.items():
            stacks[s] += amt


def stacks_snapshot(stacks, committed):
    """Stacks as the strategy should see them: behind chips + own committed (full stack pre-jam)."""
    return [stacks[i] + committed.get(i, 0.0) for i in range(len(stacks))]


def _last_unfolded(order, folded):
    for s in reversed(order):
        if s not in folded:
            return s
    return order[-1]


def simulate_sng(kinds, payouts, ante, rng, field, cache, start_bb=12.0):
    """One 3-handed push/fold SNG to completion. Returns $ payout per seat (by finish)."""
    stacks = [start_bb] * 3
    total0 = sum(stacks)
    alive = [0, 1, 2]
    button = int(rng.integers(0, 3))
    finish = []
    for _ in range(2000):
        live = [s for s in alive if stacks[s] > 1e-9]
        if len(live) <= 1:
            break
        order = _action_order(button, live)
        pre = list(stacks)                            # pre-hand stacks (for finish order on co-bust)
        play_hand(stacks, order, kinds, payouts, None, ante, rng, cache, field)
        assert abs(sum(stacks) - total0) < 1e-6, "chip conservation violated"
        newly = [s for s in alive if stacks[s] <= 1e-9 and s not in finish]
        newly.sort(key=lambda s: pre[s])              # shorter pre-hand stack finishes LOWER (no seat bias)
        for s in newly:
            finish.append(s)
            alive.remove(s)
        button = _next_button(button, [s for s in alive])
    survivors = [s for s in range(3) if s not in finish]
    rank = survivors + finish[::-1]                   # 1st = last survivor, then reverse bust order
    pay = {seat: payouts[i] if i < len(payouts) else 0.0 for i, seat in enumerate(rank)}
    return [pay[s] for s in range(3)]


def _action_order(button, live):
    live = sorted(live)
    idx = live.index(button) if button in live else 0
    return [live[(idx + k) % len(live)] for k in range(len(live))]


def _next_button(button, alive):
    for k in range(1, 4):
        nb = (button + k) % 3
        if nb in alive:
            return nb
    return alive[0]
