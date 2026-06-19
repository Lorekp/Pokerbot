"""Tournament (MTT) support for the 6/7-max NLHE bot.

This package is the **ICM layer** for tournament play. It is deliberately separate from the
cash engine and from the ``aof_*`` push/fold engine: tournament code may *import* from those
(read-only reuse of equity grids, buckets, depth tiers) but never edits them.

Modules:
  * :mod:`pokerbot.tournament.icm` -- the shared ICM primitive (Malmuth-Harville equity,
    bubble factor, and the ``icm_delta`` chip->$ wrap that every consumer calls).

See ``artifacts/task-output.txt`` (7-player MTT plan) for the
full design. The load-bearing idea: ICM(value) approximately equals chip-EV when deep / far from
the money (bubble factor -> 1), so a chip-EV blueprint plays the body and the ICM layer bends it
near the money.
"""

from __future__ import annotations

from pokerbot.tournament.icm import bubble_factor, icm_delta, icm_equity

__all__ = ["icm_equity", "bubble_factor", "icm_delta"]
