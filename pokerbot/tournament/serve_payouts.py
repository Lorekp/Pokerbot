"""Serve-time payout resolver -- the bridge from a live tournament's lobby fields to the best-first
currency payout vector the ICM serve path consumes.

The certified model layer (:mod:`pokerbot.tournament.payouts`) already turns ``(format, entrants,
buy_in)`` into a per-rank currency ladder via :func:`classify` + :func:`payout_vector`. What was missing
is the thin *serve-time* orchestration: take whatever the live lobby/config exposes, resolve a single
:class:`ResolvedPayouts` (the best-first ladder + PKO bounty economics + format flags), and **cache it
per table** so the ``(entrants -> ladder)`` reconstruction runs once rather than every hand. That is this
module.

The contract that flows downstream (verified against the consumers):
  * :func:`pokerbot.tournament.icm.icm_equity` / :func:`pokerbot.tournament.live_adapter.tournament_decide`
    / :func:`pokerbot.tournament.icm_exploit.state_exploit_attenuation` all take ``payouts`` = a
    **best-first currency ladder** and pad/truncate it to ``len(stacks)`` internally. So we store the
    FULL ladder (length ``places_paid``); the "remaining field shrinks as players bust" is handled by ICM
    through the *current stacks*, NOT by truncating the vector here. The vector only needs re-resolving
    when ``entrants`` itself changes (late-reg re-entries grow the pool + places_paid).
  * "payouts unknown -> 1.0 (full exploit)" is the correct *pure* answer from
    :func:`state_exploit_attenuation`, so the live gate -- not this module -- owns the separate safety
    policy "tournament AND payouts unknown -> disable exploit". :func:`try_resolve` returning ``None`` is
    exactly the signal that gate keys on.

Pure orchestration over already-certified functions; numpy-free, stdlib + the payouts model only. Live
runs are multi-table across Frida threads, so :class:`PayoutBook` is ``RLock``-guarded (same precedent as
``PlayerProfileStore``). See ``tests/test_serve_payouts.py``.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Optional, Sequence

from pokerbot.tournament.payouts import (
    TournamentFormat,
    bounty_pool,
    buyin_split,
    classify,
    payout_vector,
    places_paid,
)

_log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ResolvedPayouts:
    """A live table's resolved tournament payout state, ready for the ICM serve path.

    ``payouts`` is the best-first CURRENCY ladder (length ``places``); for PKO it is the PRIZE half only
    -- the bounty economics ride alongside in ``bounty_pool`` / ``per_entry_bounty`` (consumed by
    :mod:`pokerbot.tournament.bounty`). ``entrants`` is the field size the pool/places were resolved at;
    a change to it is what invalidates the cached ladder.
    """

    fmt: TournamentFormat
    payouts: tuple[float, ...]      # best-first currency ladder (PKO: prize half)
    places: int                     # paid places (== len(payouts))
    entrants: int                   # field size the pool/places were resolved at
    bounty_pool: float = 0.0        # PKO total bounty pool ($); 0 otherwise
    per_entry_bounty: float = 0.0   # PKO starting per-entry bounty value ($); 0 otherwise
    seat_value: float = 0.0         # SATELLITE per-seat value ($); 0 otherwise

    @property
    def is_pko(self) -> bool:
        return self.fmt is TournamentFormat.PKO

    @property
    def is_satellite(self) -> bool:
        return self.fmt is TournamentFormat.SATELLITE

    def top(self, n: int) -> tuple[float, ...]:
        """Best-first ladder padded/truncated to exactly ``n`` entries (unpaid ranks -> 0.0).

        ICM does this internally, so passing the full :attr:`payouts` is usually correct; this helper is
        for callers that want an exactly-``n``-long vector (e.g. one entry per seat at the table).
        """
        if n <= 0:
            return ()
        pay = self.payouts[:n]
        if len(pay) < n:
            pay = pay + (0.0,) * (n - len(pay))
        return pay


def resolve_payouts(
    *,
    tournament_type: Optional[str] = None,
    is_satellite: bool = False,
    bounty: bool = False,
    entrants: Optional[int] = None,
    buy_in: Optional[float] = None,
    prize_pool: Optional[float] = None,
    rake: float = 0.08,
    guarantee: float = 0.0,
    reentries: int = 0,
    places: Optional[int] = None,
    seats: Optional[int] = None,
    seat_value: Optional[float] = None,
    frac: Optional[float] = None,
    min_paid: int = 2,
) -> ResolvedPayouts:
    """Resolve a live tournament's lobby fields into a :class:`ResolvedPayouts`.

    Format is classified from ``tournament_type`` / ``is_satellite`` / ``bounty`` (see
    :func:`pokerbot.tournament.payouts.classify`). Then:

    * NORMAL / PKO: ``places = places or places_paid(fmt, entrants)`` and the currency ladder comes from
      :func:`payout_vector` (needs either ``prize_pool``, or ``entrants`` + ``buy_in``). For PKO the
      bounty pool + per-entry bounty are computed alongside the PRIZE ladder.
    * SATELLITE: a flat ``[seat_value] * seats`` ladder (every paid finisher wins an identical seat);
      ``seat_value`` defaults to ``buy_in`` (the seat-into-the-target scale -- only ratios matter for the
      fold-once-locked logic, and the flat shape is what :mod:`pokerbot.tournament.satellite` expects).

    Raises ``ValueError`` when the inputs are insufficient for the classified format (mirrors
    :func:`payout_vector`) -- callers wanting a soft failure use :func:`try_resolve`.
    """
    fmt = classify(tournament_type=tournament_type, is_satellite=is_satellite, bounty=bounty)

    if fmt is TournamentFormat.SATELLITE:
        if seats is None:
            raise ValueError("SATELLITE payouts need `seats` (the guaranteed seat count)")
        sv = float(seat_value) if seat_value is not None else (
            float(buy_in) if buy_in is not None else None
        )
        if sv is None:
            raise ValueError("SATELLITE payouts need `seat_value` (or `buy_in` as its proxy)")
        s = int(seats)
        if entrants is not None:
            s = min(s, int(entrants))
        ladder = payout_vector(fmt, seats=s, seat_value=sv, entrants=entrants)
        return ResolvedPayouts(
            fmt=fmt, payouts=tuple(float(x) for x in ladder), places=len(ladder),
            entrants=int(entrants) if entrants is not None else s, seat_value=sv,
        )

    # NORMAL / PKO -- a cash ladder.
    p = places if places is not None else (
        places_paid(fmt, entrants, frac=frac, min_paid=min_paid) if entrants is not None else None
    )
    if p is None:
        raise ValueError("NORMAL/PKO payouts need `entrants` (or an explicit `places`)")

    ladder = payout_vector(
        fmt, entrants=entrants, places=p, prize_pool=prize_pool, buy_in=buy_in,
        rake=rake, guarantee=guarantee, reentries=reentries, frac=frac, min_paid=min_paid,
    )

    bpool = 0.0
    per_entry_bounty = 0.0
    if fmt is TournamentFormat.PKO and buy_in is not None and entrants is not None:
        bpool = bounty_pool(fmt, entrants, buy_in, rake=rake, reentries=reentries)
        per_entry_bounty = buyin_split(fmt, buy_in, rake=rake)["bounty_contrib"]

    return ResolvedPayouts(
        fmt=fmt, payouts=tuple(float(x) for x in ladder), places=len(ladder),
        entrants=int(entrants) if entrants is not None else p,
        bounty_pool=float(bpool), per_entry_bounty=float(per_entry_bounty),
    )


def resolve_from_prize_list(
    prize_list: Sequence,
    *,
    fmt: TournamentFormat = TournamentFormat.NORMAL,
    entrants: Optional[int] = None,
    bounty_pool: float = 0.0,
    per_entry_bounty: float = 0.0,
    seat_value: float = 0.0,
) -> Optional[ResolvedPayouts]:
    """Build a :class:`ResolvedPayouts` directly from a CAPTURED live ladder -- the highest-fidelity
    path (no model approximation).

    ``prize_list`` is a ``tournamentlobby.prize_list`` TourneyPrizeListBean as parsed by
    :mod:`pokerbot.tourney.aggregate` -- a list of ``{"rank": int, "amount": float}`` (``pct`` ignored),
    or a bare list of amounts. Entries are sorted best-first by ``rank`` (or taken in order for bare
    amounts), zero/None amounts dropped from the tail. Returns ``None`` if nothing usable is present, so
    the caller falls back to model reconstruction / the chip-EV path.
    """
    if not prize_list:
        return None
    pairs: list[tuple[int, float]] = []
    for i, e in enumerate(prize_list):
        if isinstance(e, dict):
            amt = e.get("amount")
            rank = e.get("rank", i + 1)
        else:
            amt, rank = e, i + 1
        if amt is None:
            continue
        try:
            pairs.append((int(rank), float(amt)))
        except (TypeError, ValueError):
            continue
    if not pairs:
        return None
    pairs.sort(key=lambda p: p[0])
    ladder = [a for _, a in pairs]
    while ladder and ladder[-1] <= 0.0:        # trim a zero/None tail (unpaid ranks)
        ladder.pop()
    if not ladder:
        return None
    return ResolvedPayouts(
        fmt=fmt, payouts=tuple(ladder), places=len(ladder),
        entrants=int(entrants) if entrants is not None else len(ladder),
        bounty_pool=float(bounty_pool), per_entry_bounty=float(per_entry_bounty),
        seat_value=float(seat_value),
    )


def try_resolve(**kwargs) -> Optional[ResolvedPayouts]:
    """:func:`resolve_payouts` that returns ``None`` (with a debug log) instead of raising.

    This is the live gate's signal: ``None`` => payouts unknown => the caller stays on the safe path
    (chip-EV push/fold, exploit disabled) rather than feeding a bogus ladder into ICM.
    """
    try:
        return resolve_payouts(**kwargs)
    except (ValueError, KeyError, TypeError) as exc:  # insufficient/ill-formed lobby fields
        _log.debug("payout resolve failed (%s); serve will use the chip-EV fallback", exc)
        return None


def remaining_payout_vector(payouts: Sequence[float] | ResolvedPayouts, n: int) -> tuple[float, ...]:
    """Best-first ladder for ``n`` competing seats (top-``n`` padded with 0.0).

    Accepts a raw ladder or a :class:`ResolvedPayouts`. Convenience for callers that want exactly one
    payout entry per live seat; ICM itself already pads/truncates, so this is optional sugar.
    """
    ladder = payouts.payouts if isinstance(payouts, ResolvedPayouts) else tuple(float(x) for x in payouts)
    if n <= 0:
        return ()
    pay = ladder[:n]
    if len(pay) < n:
        pay = pay + (0.0,) * (n - len(pay))
    return pay


class PayoutBook:
    """Thread-safe per-table cache of :class:`ResolvedPayouts`.

    Live tournament tables run concurrently on Frida message threads, so resolution is ``RLock``-guarded
    (same precedent as the live ``PlayerProfileStore``). A table's ladder is resolved once and reused
    until its resolve inputs change (e.g. ``entrants`` grows during late registration), at which point it
    is recomputed. Only successful resolves are cached -- a ``None`` (insufficient inputs) is never cached
    so a later, better-populated lobby read can still succeed.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._by_table: dict[str, ResolvedPayouts] = {}
        self._keys: dict[str, tuple] = {}

    @staticmethod
    def _key(kwargs: dict) -> tuple:
        # A hashable, order-stable snapshot of the resolve inputs for change detection.
        return tuple(sorted((k, v) for k, v in kwargs.items() if v is not None))

    def resolve(self, table_id: str, **kwargs) -> Optional[ResolvedPayouts]:
        """Resolve (and cache) the ladder for ``table_id``; returns the cached value when inputs are
        unchanged. Returns ``None`` if resolution fails (the cache is left untouched)."""
        key = self._key(kwargs)
        with self._lock:
            if self._keys.get(table_id) == key and table_id in self._by_table:
                return self._by_table[table_id]
        rp = try_resolve(**kwargs)  # resolve OUTSIDE the lock (icm/payout work is non-trivial)
        if rp is None:
            return None
        with self._lock:
            self._by_table[table_id] = rp
            self._keys[table_id] = key
        return rp

    def get(self, table_id: str) -> Optional[ResolvedPayouts]:
        """Last successfully resolved ladder for ``table_id`` (``None`` if never resolved)."""
        with self._lock:
            return self._by_table.get(table_id)

    def clear(self, table_id: Optional[str] = None) -> None:
        """Drop the cached ladder for ``table_id`` (or all tables when ``table_id is None``)."""
        with self._lock:
            if table_id is None:
                self._by_table.clear()
                self._keys.clear()
            else:
                self._by_table.pop(table_id, None)
                self._keys.pop(table_id, None)
