"""Tournament (MTT) support for the 6/7-max NLHE bot.

This package is the **tournament layer**: it makes the chip-EV cash engine play correct $-EV
tournament poker. It is deliberately separate from the cash engine and from the ``aof_*`` push/fold
engine -- tournament code may *import* from those (read-only reuse of equity grids, buckets, depth
tiers) but never edits them.

Layers (load-bearing idea: ICM(value) ~= chip-EV when deep / far from the money -> a chip-EV
blueprint plays the body and the ICM layer bends it near the money):

  * :mod:`~pokerbot.tournament.icm` -- the Malmuth-Harville ICM primitive (``icm_equity`` /
    ``bubble_factor`` / ``icm_delta``); also ported to native Rust (``battery_native.icm_equity`` ...,
    gated by ``tests/test_native_icm_equiv.py``) for the in-loop solve hot path.
  * :mod:`~pokerbot.tournament.payouts` -- the empirically-captured payout model: format
    classification, ``places_paid``, per-rank structure, and the prize vector that feeds ICM
    (NORMAL / PKO / SATELLITE).
  * :mod:`~pokerbot.tournament.structure` -- the blind/ante ladder (CoinPoker MTT), final-table
    rollback, re-buy/late-reg windows, and the training spot sampler.
  * :mod:`~pokerbot.tournament.bubble` -- Bubble Protection (B(N) pre-ITM refund -> a flattened
    bubble; protected players play WIDER near the money).
  * :mod:`~pokerbot.tournament.bounty` -- PKO progressive-bounty accounting + per-stack bounty value
    (chip-EV != $-EV: call wider to collect bounties).
  * :mod:`~pokerbot.tournament.satellite` -- the seat-locked step payoff (extreme ICM; fold once a
    seat is mathematically locked).
  * :mod:`~pokerbot.tournament.icm_overlay` / :mod:`~pokerbot.tournament.decide` -- the serve-time
    overlay (rigorous ICM action value + the deal-time chip-EV toggle) and the short-stack router.
  * :mod:`~pokerbot.tournament.anchor` -- the always-fold calibration anchor (6-max +125, 7-max +203.6).
"""

from __future__ import annotations

from pokerbot.tournament.anchor import (
    always_fold_exploitability,
    sevenmax_tournament_anchor,
    sixmax_anchor,
)
from pokerbot.tournament.bounty import (
    BountyState,
    bounty_cash_on_win,
    min_equity_to_call,
    pko_call_dollar_ev,
)
from pokerbot.tournament.bubble import (
    bubble_seats,
    protected_bubble_factor,
    protected_payouts,
)
from pokerbot.tournament.icm import bubble_factor, icm_delta, icm_equity
from pokerbot.tournament.icm_overlay import (
    chip_ev_action_value,
    icm_action_value,
    icm_tilt,
)
from pokerbot.tournament.payouts import (
    TournamentFormat,
    classify,
    payout_structure,
    payout_vector,
    places_paid,
    prize_pool,
)
from pokerbot.tournament.satellite import (
    satellite_payouts,
    seat_locked,
    should_fold_for_seat,
)
from pokerbot.tournament.serve_payouts import (
    ResolvedPayouts,
    resolve_from_prize_list,
    resolve_payouts,
    try_resolve,
)
from pokerbot.tournament.live_adapter import (
    deep_icm_live_override,
    field_icm_live_override,
    pko_call_override,
    satellite_fold_override,
)
from pokerbot.tournament.structure import (
    BlindLevel,
    ante_for_bb,
    blind_ladder,
    effective_bb_at_level,
    final_table_rollback,
    level_at,
)

__all__ = [
    # ICM primitive
    "icm_equity", "bubble_factor", "icm_delta",
    # payout model
    "TournamentFormat", "classify", "places_paid", "payout_structure", "payout_vector", "prize_pool",
    # blind/ante structure
    "BlindLevel", "blind_ladder", "ante_for_bb", "level_at", "effective_bb_at_level",
    "final_table_rollback",
    # bubble protection
    "bubble_seats", "protected_payouts", "protected_bubble_factor",
    # PKO bounty
    "BountyState", "min_equity_to_call", "pko_call_dollar_ev", "bounty_cash_on_win",
    # satellite
    "satellite_payouts", "seat_locked", "should_fold_for_seat",
    # overlay / deal toggle
    "icm_action_value", "icm_tilt", "chip_ev_action_value",
    # live payout resolution
    "ResolvedPayouts", "resolve_payouts", "resolve_from_prize_list", "try_resolve",
    # live override gates (thin adapters over the certified math)
    "deep_icm_live_override", "pko_call_override", "field_icm_live_override",
    "satellite_fold_override",
    # calibration anchor
    "always_fold_exploitability", "sixmax_anchor", "sevenmax_tournament_anchor",
]
