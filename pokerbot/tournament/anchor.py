"""7-max (and general n-max) calibration anchor: the always-fold exploitability lower bound.

A native belief-tracking LBR at 7 seats is gated on the (deferred) 7-seat tournament blueprint --
``lbr.rs`` and the JAX engine are seat-pinned at 6, and no 7-max blueprint exists yet to best-respond
to. But the most important calibration anchor -- the **always-fold** exploitability -- has an EXACT
closed form that needs no blueprint, because the best response to a table of pure folders is trivial:
**always raise and steal the dead money** (they fold to any size, so the raise comes back and the
exploiter scoops every other seat's committed chips). Seat-rotated over a full orbit, the exploiter's
win rate is therefore exactly the average dead money it collects:

    per-hand value (bb) = [ (n-1)*(sb+bb)  +  n*(n-1)*ante ] / n
    anchor (bb/100)     = 100 * per-hand value

  Derivation (n seats, exploiter rotates through every seat once):
    * n-2 non-blind seats: win sb + bb + (n-1)*ante   (the two blinds + every other ante)
    * the SB seat:         win bb + (n-1)*ante
    * the BB seat:         win sb + (n-1)*ante
    summed and divided by n gives the closed form above.

This is validated against the empirically-measured 6-max harness anchor (+128 bb/100): the formula
gives +125.0 bb/100 at 6-max no-ante, inside the documented [110,145] band -- so the formula is a
faithful idealization (the ~3 bb/100 residual is the harness LBR's postflop/belief overhead, which
vanishes vs a pure folder). The 7-max tournament anchor (ante = 0.125 bb) is **+203.6 bb/100** --
higher than 6-max because there are more folders to steal from and the ante adds dead money.

The full belief-LBR at 7-max (a guardrail / behavioral-sanity lower bound, NOT a target -- multiplayer
has no Nash-exploitability guarantee) lands with the 7-seat blueprint retrain (Workstream C training).
"""

from __future__ import annotations

# The 6-max no-ante anchor measured by the native LBR harness, with its calibration band.
SIXMAX_NOANTE_BAND = (110.0, 145.0)
SIXMAX_NOANTE_EMPIRICAL = 128.0
# CoinPoker MTT per-player ante is round(BB/8) ~= 0.125 bb (see pokerbot.tournament.structure).
MTT_ANTE_BB = 0.125


def always_fold_exploitability(n_seats: int, *, ante_bb: float = 0.0,
                               sb_bb: float = 0.5, bb_bb: float = 1.0) -> float:
    """Exact always-fold exploitability (bb/100): the win rate an always-steal best response extracts
    from ``n_seats - 1`` pure folders, seat-rotated over a full orbit. See the module docstring for
    the derivation. ``ante_bb`` is the per-player ante in bb (0 = cash, 0.125 = CoinPoker MTT)."""
    if n_seats < 2:
        raise ValueError(f"n_seats must be >= 2, got {n_seats}")
    n = n_seats
    per_hand = ((n - 1) * (sb_bb + bb_bb) + n * (n - 1) * ante_bb) / n
    return 100.0 * per_hand


def sixmax_anchor() -> float:
    """The 6-max cash (no-ante) always-fold anchor: +125.0 bb/100 (matches the +128 harness band)."""
    return always_fold_exploitability(6, ante_bb=0.0)


def sevenmax_tournament_anchor(ante_bb: float = MTT_ANTE_BB) -> float:
    """The 7-max tournament always-fold anchor (per-player ante 0.125 bb): +203.6 bb/100."""
    return always_fold_exploitability(7, ante_bb=ante_bb)


def print_anchors() -> None:  # pragma: no cover - convenience CLI
    print(f"6-max  no-ante     : {sixmax_anchor():7.2f} bb/100  (empirical harness {SIXMAX_NOANTE_EMPIRICAL})")
    print(f"7-max  ante=0.125  : {sevenmax_tournament_anchor():7.2f} bb/100")
    for n in range(2, 8):
        print(f"  n={n} no-ante: {always_fold_exploitability(n):7.2f}   ante=0.125: "
              f"{always_fold_exploitability(n, ante_bb=MTT_ANTE_BB):7.2f}")


if __name__ == "__main__":  # pragma: no cover
    print_anchors()
