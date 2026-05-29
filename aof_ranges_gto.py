"""
HU solver-derived push/fold tables only (from ``hu_preflop_dcfr``).

Does **not** export 3/4-max tables. For default multiway tables use ``aof_ranges``
(chart pool). For tight archive comparison use ``aof_ranges_tight``.
"""

from __future__ import annotations

from aof_ranges import hu_solve_manifest, lookup_call, lookup_shove, range_provenance

__all__ = [
    "hu_solve_manifest",
    "lookup_call",
    "lookup_shove",
    "range_provenance",
]
