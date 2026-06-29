"""PKO (progressive-knockout) SNG evaluator -- the chip-EV != $-EV demonstration.

The push/fold SNG (``sim.py``) measures prize-ladder ROI but is blind to bounties, so it cannot show the
defining PKO property: **covering an all-in is worth more than its chip-EV because winning collects the
victim's bounty cash**, so the correct PKO strategy calls WIDER than the ICM-only (prize) threshold.

This module adds per-player bounty tracking (``pokerbot.tournament.bounty.BountyState``: a KO pays the
eliminator ``0.5*victim`` cash + grows their own bounty by ``0.5*victim`` -- the CoinPoker progressive
rule) on top of the SAME certified push/fold machinery (``sim._settle_allin`` side-pot award,
``sim.play_hand``-style sequential jam/call, real cards, chip-conservation). Two strategy arms:

  * ``icm``  -- the certified prize-only ICM push/fold (``decide_pushfold``), bounty-blind (the baseline).
  * ``pko``  -- jams like ``icm`` but on the CALL side calls iff hero's showdown equity vs the jammer's
    range >= ``bounty.min_equity_to_call`` WITH the jammer's live bounty (a lower threshold -> wider calls).

Total realized $ per seat = prize finish $ + realized bounty cash. If the ``pko`` arm's total $ beats the
``icm`` arm's (CI-separated) while the bounty pool stays conserved, chip-EV != $-EV is demonstrated.

Run the measurement: ``python scripts/tournament/pko_bounty_edge.py [N] [iters]``.
"""

from __future__ import annotations

import numpy as np

from pokerbot.tournament.bounty import BountyState, min_equity_to_call
from pokerbot.tournament.pushfold_icm import _hu_grid
from pokerbot.tournament.sim import (
    _action_order,
    _bucket,
    _next_button,
    _settle_allin,
    _strategy_action,
    stacks_snapshot,
)

# Equity grid: EQ[i][j] = HU equity of hand i vs hand j; W[j] = combo weight of hand j; _LABELS rank->label.
_EQ, _LABELS, _W = _hu_grid()
_LABEL_TO_IDX = {l: i for i, l in enumerate(_LABELS)}


def equity_vs_range(hand_label: str, jam_range: np.ndarray) -> float:
    """Hero's showdown equity (hand vs the opponent's jam RANGE), combo-weighted. ``jam_range`` is the
    169-length {0,1}/[0,1] indicator the field jams. Empty range -> 0.5 (no read)."""
    i = _LABEL_TO_IDX[hand_label]
    w = _W * np.asarray(jam_range, dtype=float)
    tot = float(w.sum())
    if tot <= 0.0:
        return 0.5
    return float((_EQ[i] * w).sum() / tot)


def _pko_call(stacks_snap, prize_payouts, hero, hand, jammer, villain_bounty, field) -> str:
    """The bounty-aware CALL decision: call iff hero's equity vs the jammer's jam range clears the
    bounty-adjusted indifference threshold. A positive ``villain_bounty`` LOWERS ``min_equity_to_call``
    (the bounty cash sweetens the call), so this calls strictly wider than the prize-only (bounty=0)
    threshold -- the chip-EV != $-EV lever."""
    hero_eq = equity_vs_range(hand, field["jam"])
    p_star = min_equity_to_call(stacks_snap, prize_payouts, hero, jammer, villain_bounty=villain_bounty)
    return "call" if hero_eq >= p_star else "fold"


def _play_hand_pko(stacks, order, kinds, prize_payouts, ante, rng, cache, field, bounties):
    """One push/fold hand WITH bounty-aware calling + KO attribution. Mutates ``stacks``; returns the list
    of ``(eliminator, victim)`` knockouts settled this hand. Mirrors ``sim.play_hand`` exactly except the
    ``pko`` kind's call side consults the live bounty, and the contested showdown reports who busted whom."""
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

    jammers, folded = [], set()
    for s in order:
        if s in folded:
            continue
        hand = _LABELS[_bucket(*holes[s])]
        snap = stacks_snapshot(stacks, committed)
        if not jammers:                                       # first-in: jam or fold (pko jams like icm)
            behind = [o for o in order if o != s and o not in folded and order.index(o) > order.index(s)]
            kind = "icm" if kinds[s] == "pko" else kinds[s]   # bounty lever is on the CALL side
            act = _strategy_action(kind, snap, prize_payouts, s, hand, blinds=blinds, ante=ante,
                                   behind=behind, cache=cache, field=field, rng=rng)
            if act == "jam":
                add = stacks[s]; stacks[s] = 0.0; committed[s] += add; jammers.append(s)
            else:
                folded.add(s)
        else:                                                 # facing a jam: call or fold
            if kinds[s] == "pko":
                act = _pko_call(snap, prize_payouts, s, hand, jammers[0], bounties.live[jammers[0]], field)
            else:
                act = _strategy_action(kinds[s], snap, prize_payouts, s, hand, blinds=blinds, ante=ante,
                                       jammer=jammers[0], cache=cache, field=field, rng=rng)
            if act == "call":
                add = stacks[s]; stacks[s] = 0.0; committed[s] += add; jammers.append(s)
            else:
                folded.add(s)

    contenders = jammers
    if len(contenders) <= 1:                                  # uncontested: aggressor (or last unfolded) wins
        if contenders:
            winner = contenders[0]
        else:
            from pokerbot.tournament.sim import _last_unfolded
            winner = _last_unfolded(order, folded)
        stacks[winner] += sum(committed.values())
    else:
        dead = sum(committed[s] for s in order if s not in contenders)
        won = _settle_allin(contenders, {s: committed[s] for s in contenders}, holes, board_cards, dead)
        for s, amt in won.items():
            stacks[s] += amt
        winner = max(contenders, key=lambda s: stacks[s])     # the hand winner = most chips after settle
    # KO attribution (UNIFIED over both branches): ANY seat that busted this hand -- lost an all-in OR
    # blinded/ante'd off its last chips uncontested -- was knocked out by the pot winner. Attributing the
    # blind-out KO too is what keeps the bounty pool fully realizing (a missed blind-out left its bounty
    # live forever). Side-pot mis-attribution (a short victim formally KO'd by a mid stack) is possible but
    # pool-conserving; refine to per-layer winners if exact KO credit ever matters.
    kos = [(winner, s) for s in order if s != winner and stacks[s] <= 1e-9]
    return kos


def simulate_pko_sng(kinds, prize_payouts, entry_bounty, ante, rng, field, cache, start_bb=12.0):
    """One 3-handed PKO push/fold SNG to completion. Returns total realized $ per seat = prize finish $
    + realized bounty cash. ``entry_bounty`` is each entrant's starting bounty (currency); the bounty pool
    is conserved end-to-end (asserted)."""
    stacks = [float(start_bb)] * 3
    total0 = sum(stacks)
    bounties = BountyState.uniform(3, float(entry_bounty))
    alive = [0, 1, 2]
    button = int(rng.integers(0, 3))
    finish: list[int] = []
    for _ in range(2000):
        live = [s for s in alive if stacks[s] > 1e-9]
        if len(live) <= 1:
            break
        order = _action_order(button, live)
        pre = list(stacks)
        kos = _play_hand_pko(stacks, order, kinds, prize_payouts, ante, rng, cache, field, bounties)
        assert abs(sum(stacks) - total0) < 1e-6, "chip conservation violated"
        for elim, victim in kos:
            bounties.award_ko(elim, victim)
        newly = [s for s in alive if stacks[s] <= 1e-9 and s not in finish]
        newly.sort(key=lambda s: pre[s])
        for s in newly:
            finish.append(s)
            alive.remove(s)
        button = _next_button(button, [s for s in alive])

    survivors = [s for s in range(3) if s not in finish]
    rank = survivors + finish[::-1]                           # 1st = last survivor, then reverse bust order
    bounties.finalize_winner(rank[0])                         # winner is paid their own outstanding bounty
    assert bounties.conserved(), "bounty pool not conserved"
    prize = {seat: (prize_payouts[i] if i < len(prize_payouts) else 0.0) for i, seat in enumerate(rank)}
    return [float(prize[s] + bounties.realized[s]) for s in range(3)]
