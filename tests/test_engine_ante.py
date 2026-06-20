"""Per-player ante support in jax_engine (for tournament HU). Antes are dead money: they inflate the
pot for bet sizing (pot_dead) AND go into committed_total so the winner collects them at showdown and
chips are conserved -- but they are NOT part of committed_street (an ante doesn't reduce what a seat
owes to call). ante=0 (the default) must be byte-identical to the pre-ante engine.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from pokerbot.engine import jax_engine as je

_HU_OCC = jnp.array([i in (3, 5) for i in range(je.NUM_PLAYERS)])  # heads-up: seats 3,5 live


def _hu_init(ante: int, seed: int = 0):
    # button=3 -> in HU the button is the SB and acts first preflop.
    return je.init_hand(jax.random.PRNGKey(seed), button=jnp.int32(3), occupancy=_HU_OCC,
                        ante=jnp.int32(ante))


def test_ante_posting_accounting():
    A = 25  # 0.25 BB at CHIP_UNIT=100
    st = _hu_init(A)
    cs = np.asarray(st.committed_street)
    ct = np.asarray(st.committed_total)
    # Identify SB/BB from the posted blinds (this engine makes the button the BB in HU; the SB is the
    # seat after the button and acts first preflop). Don't hard-code physical seats.
    sb = int(np.argwhere(cs == je.SB_CHIPS)[0, 0])
    bb = int(np.argwhere(cs == je.BB_CHIPS)[0, 0])
    # committed_total = blind + ante (so the ante is in the distributed pot)
    assert ct[sb] == je.SB_CHIPS + A and ct[bb] == je.BB_CHIPS + A
    # dead money bucket holds every live seat's ante
    assert int(st.pot_dead) == 2 * A
    # the pot the net sizes against includes the antes
    assert int(je._pot_for_sizing(st)) == je.SB_CHIPS + je.BB_CHIPS + 2 * A
    # dead seats post nothing
    for d in (0, 1, 2, 4):
        assert ct[d] == 0 and cs[d] == 0


def test_ante_zero_is_byte_identical():
    a = je.init_hand(jax.random.PRNGKey(7), button=jnp.int32(3), occupancy=_HU_OCC)
    b = je.init_hand(jax.random.PRNGKey(7), button=jnp.int32(3), occupancy=_HU_OCC, ante=0)
    la, lb = jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b)
    assert len(la) == len(lb)
    for x, y in zip(la, lb):
        assert np.array_equal(np.asarray(x), np.asarray(y))


def test_ante_chip_conservation_and_winner_collects_ante():
    A = 25
    st = _hu_init(A)
    sb = int(st.action_seat)            # HU first actor = button/SB
    bb = 5 if sb == 3 else 3
    st2 = je.step_state(st, jnp.int32(je.FOLD))   # SB folds preflop -> BB wins
    assert bool(st2.done)
    r = np.asarray(st2.rewards)
    assert r.sum() == 0                 # zero-sum (chips conserved)
    # BB collects SB's blind + SB's ante; BB's own blind-overlay and ante are returned in the pot.
    assert r[sb] == -(je.SB_CHIPS + A)
    assert r[bb] == (je.SB_CHIPS + A)


def test_ante_clamped_to_short_stack():
    # The ANTE must not push a stack negative. Isolate the ante clamp: give the first-acting seat (the
    # SB, seat after the button) a stack just above the SB blind but below SB+ante, so only the ante
    # would overdraw. (Blinds themselves can overdraw a sub-blind stack -- a pre-existing engine quirk
    # irrelevant at the >=5bb tournament-HU depths -- so we keep the stack above the blind here.)
    stacks = jnp.array([0, 0, 0, je.SB_CHIPS + 10, 0, je.BB_CHIPS * 20], dtype=jnp.int32)
    st = je.init_hand(jax.random.PRNGKey(0), button=jnp.int32(5), stacks=stacks, occupancy=_HU_OCC,
                      ante=jnp.int32(50))
    assert int(np.asarray(st.stacks).min()) >= 0      # nobody negative from the ante


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-v"]))
