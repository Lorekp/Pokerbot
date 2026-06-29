"""Seat remapping for the 7-max neural serve bridge — the <7-live sub-table case ONLY.

WHY THIS EXISTS (and why it is usually a no-op)
-----------------------------------------------
The 7-max tournament net (``checkpoints/sd_cfr_7max_neural/seven_v1.pkl``) is trained by
``cfr_env.nlhe_7max_env_spec`` with ALL SEVEN seats live in their NATURAL absolute order (occupancy =
``[True]*7``; the button rotates over every seat). ``encode_info_state`` one-hots the ABSOLUTE seat and
button index sized to the table width. So unlike the heads-up net (which pins its two live seats at the
fixed training pair {3,5} and therefore ALWAYS needs the ``api/hu_neural_endgame._remap_env_to_hu_seats``
permutation), a genuinely-7-live tournament root needs NO permutation at all: the served absolute seat
indices already coincide with the trained ones (IDENTITY remap). ``neural_serve.SevenMaxNeuralPolicy``
serves that natural case directly.

This module covers the OTHER case: a tournament table with FEWER than 7 live players (a short / broken
table, a hand folded to a sub-set we want to consult the net on, or a final-table down to k<7). The
trained net never saw a 7-wide state with holes at arbitrary positions, but its seat one-hots are still
defined over all 7 absolute indices, so we must place the ``k`` live seats onto a chosen set of trained
absolute indices and mark the remaining ``7-k`` as ``SEAT_EMPTY``. ``place_live_seats`` does exactly that:
it SCATTERS every per-seat array (holes / seat_status / stacks / committed_street / committed_total /
acted / rewards) and RELABELS every scalar seat reference (button / action_seat / last_aggressor) and the
per-action ``hist_seat`` indices through the placement map, mirroring ``_remap_env_to_hu_seats``
(hu_neural_endgame.py:76-119). A full (7-live) table passes ``live_seats = list(range(7))`` and gets the
IDENTITY map back (every field byte-identical) — proven in ``tests/test_seven_neural_serve.py``.

Seat-agnostic fields (board, street, pot_dead, current_bet, min_raise, deck, deck_ptr, rng_key,
hist_action, hist_amount, hist_len, done, last_raise_full, pending_deal) pass through untouched.

NOTE: ``encode_history_state``'s ``NUM_PLAYERS=6`` hardcode (jax_engine.py:684) is a TRAINING-side
history encoder and does NOT touch the served ``AverageStrategyNet`` (which consumes only
``encode_info_state`` + ``legal_action_mask``); it is irrelevant to this serve path.
"""

from __future__ import annotations

import numpy as np

# Number of seats the 7-max net was trained on (cfr_env.nlhe_7max_env_spec, N7 = 7).
SEVEN: int = 7


def _placement_perm(live_seats: list[int], n: int) -> np.ndarray:
    """Build ``perm`` with ``perm[old_index] = trained_absolute_index`` for an n-wide table.

    ``live_seats`` lists the trained absolute indices the k current live seats should land on, in the
    SAME order those live seats occupy the input env's 0..k-1 (D-order) positions. The remaining empty
    input positions (k..n-1) are scattered onto the leftover trained indices in ascending order, so the
    map is always a true bijection of 0..n-1 — no per-seat datum is dropped (asserted in the test).
    """
    live_seats = [int(s) for s in live_seats]
    if not all(0 <= s < n for s in live_seats):
        raise ValueError(f"live_seats {live_seats} out of range for n={n}")
    if len(set(live_seats)) != len(live_seats):
        raise ValueError(f"live_seats {live_seats} contains duplicates")
    if len(live_seats) > n:
        raise ValueError(f"live_seats {live_seats} longer than table width {n}")
    leftover = [s for s in range(n) if s not in set(live_seats)]
    targets = live_seats + leftover  # targets[old] = new ; len == n
    perm = np.asarray(targets, dtype=np.int64)
    assert sorted(perm.tolist()) == list(range(n)), "placement must be a bijection of 0..n-1"
    return perm


def place_live_seats(env_np, live_seats: list[int]):
    """Place a numpy ``EnvState`` whose live players occupy D-order 0..k-1 onto trained absolute seats.

    Args:
        env_np: a host (numpy) ``EnvState`` (e.g. from ``state_bridge.env_to_numpy``) whose ``k`` live
            seats sit at the leading positions 0..k-1 (the sub-table layout the live bridge emits).
        live_seats: the trained absolute indices to place those ``k`` live seats on, in order. For a
            genuinely-full table pass ``list(range(7))`` -> IDENTITY remap (every field unchanged).

    Returns:
        ``(remapped_env, mapping)`` where ``remapped_env`` is a new ``EnvState`` with seat-indexed arrays
        scattered and scalar seat references relabelled, and ``mapping`` is ``{old_index: new_index}``
        (the placement bijection) so the caller can translate the hero's seat:
        ``hero_trained = mapping[hero_old]``.
    """
    from pokerbot.engine.jax_engine import EnvState

    env_arr = np.asarray(env_np.seat_status)
    n = env_arr.shape[0]
    perm = _placement_perm(live_seats, n)  # perm[old] = new

    def scatter_rows(arr):
        a = np.asarray(arr)
        out = np.empty_like(a)
        out[perm] = a  # out[new] = a[old]
        return out

    def relabel_seat(x):
        xi = int(x)
        return int(perm[xi]) if 0 <= xi < n else xi  # -1 (no last_aggressor / terminal) passes through

    hist_seat = np.asarray(env_np.hist_seat).copy()
    valid = (hist_seat >= 0) & (hist_seat < n)
    hist_seat[valid] = perm[hist_seat[valid]]

    remapped = env_np._replace(
        holes=scatter_rows(env_np.holes),
        seat_status=scatter_rows(env_np.seat_status),
        stacks=scatter_rows(env_np.stacks),
        committed_street=scatter_rows(env_np.committed_street),
        committed_total=scatter_rows(env_np.committed_total),
        acted=scatter_rows(env_np.acted),
        rewards=scatter_rows(env_np.rewards),
        button=np.asarray(relabel_seat(env_np.button), dtype=np.asarray(env_np.button).dtype),
        action_seat=np.asarray(relabel_seat(env_np.action_seat),
                               dtype=np.asarray(env_np.action_seat).dtype),
        last_aggressor=np.asarray(relabel_seat(env_np.last_aggressor),
                                  dtype=np.asarray(env_np.last_aggressor).dtype),
        hist_seat=hist_seat,
    )
    assert isinstance(remapped, EnvState)
    mapping = {int(old): int(perm[old]) for old in range(n)}
    return remapped, mapping
