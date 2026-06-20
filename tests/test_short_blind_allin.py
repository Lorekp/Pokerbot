"""BUG-B2 / BUG-B3 gate: short blind posts are clamped + flagged all-in; the standing bet reflects a
sub-BB all-in blind.

Before the fix ``_post_blinds`` subtracted the FULL blind from the stack (driving a short stack
NEGATIVE, which corrupts ``legal_action_mask``'s ``stack > to_call`` test and side-pot eligibility) and
hard-set ``current_bet = BB`` regardless of a short all-in BB (so others over-paid ``to_call``). The fix
clamps each blind to the poster's stack, flags an exhausted live seat ``SEAT_ALLIN``, and sets
``current_bet`` to the largest actual street commitment.

CRITICAL invariant: for stacks >= 1BB and ante=0 the new code is BYTE-IDENTICAL to the old engine —
this is what keeps ``tests/test_np_engine_equiv.py`` green without a golden regeneration.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from pokerbot.engine import jax_engine as je
from pokerbot.engine import np_engine as ne

CU = je.CHIP_UNIT          # 100 chips per BB
SB = je.SB_CHIPS           # 50
BB = je.BB_CHIPS           # 100
ALLIN = int(je.SEAT_ALLIN)
LIVE = int(je.SEAT_LIVE)


def _init(stacks, button=0, ante=0, occ=None):
    key = jax.random.PRNGKey(0)
    occ_arr = None if occ is None else jnp.asarray(occ, dtype=bool)
    return je.init_hand(
        key,
        button=jnp.int32(button),
        stacks=jnp.asarray(stacks, dtype=jnp.int32),
        occupancy=occ_arr,
        ante=ante,
    )


def test_short_sb_and_bb_are_clamped_and_allin():
    # 6 live seats, button=0 => SB=seat1, BB=seat2. Give SB 0.3BB and BB 0.7BB (both short).
    stacks = [100 * CU, 30, 70, 100 * CU, 100 * CU, 100 * CU]
    s = _init(stacks, button=0)
    ss = np.asarray(s.seat_status)
    stk = np.asarray(s.stacks)
    com = np.asarray(s.committed_street)

    assert ss[1] == ALLIN and ss[2] == ALLIN, f"short blinds must be all-in: {ss}"
    assert stk[1] == 0 and stk[2] == 0, f"exhausted stacks must be 0, got {stk[1]},{stk[2]}"
    assert (stk >= 0).all(), f"no stack may go negative: {stk}"
    assert com[1] == 30 and com[2] == 70, f"committed must equal the clamped post: {com}"
    # B3: standing bet = largest commitment (70), not a phantom full BB (100).
    assert int(s.current_bet) == 70, f"current_bet must reflect the short all-in BB: {int(s.current_bet)}"
    assert int(s.min_raise) == BB, f"min raise stays a full BB: {int(s.min_raise)}"
    # A full-stack live seat (UTG=seat3) owes only the short standing bet to call.
    to_call_utg = int(s.current_bet) - int(com[3])
    assert to_call_utg == 70, f"UTG to_call must be 70 (short BB), not {to_call_utg}"


def test_bb_exactly_one_bb_is_allin_but_current_bet_unchanged():
    # BB seat (seat2) has EXACTLY 1BB -> posts it all and is all-in; current_bet is still BB.
    stacks = [100 * CU, 100 * CU, BB, 100 * CU, 100 * CU, 100 * CU]
    s = _init(stacks, button=0)
    ss = np.asarray(s.seat_status)
    assert ss[2] == ALLIN, f"BB with exactly 1BB must be all-in: {ss}"
    assert int(s.current_bet) == BB
    assert int(np.asarray(s.stacks)[2]) == 0


def test_full_stacks_byte_identical_baseline():
    # Stacks >= 1BB, ante=0 => the fix must be a no-op vs the classic engine.
    s = _init([100 * CU] * 6, button=0)
    ss = np.asarray(s.seat_status)
    com = np.asarray(s.committed_street)
    assert (ss == LIVE).all(), f"no seat should be all-in at 100bb: {ss}"
    assert com[1] == SB and com[2] == BB, f"normal blinds: {com}"
    assert int(s.current_bet) == BB and int(s.min_raise) == BB
    assert (np.asarray(s.stacks)[1] == 100 * CU - SB) and (np.asarray(s.stacks)[2] == 100 * CU - BB)


def test_to_numpy_state_preserves_short_blind_allin():
    # The np_engine bridge (to_numpy_state) must carry the all-in status + short current_bet exactly.
    stacks = [100 * CU, 30, 70, 100 * CU, 100 * CU, 100 * CU]
    s_jax = _init(stacks, button=0)
    s_np = ne.to_numpy_state(s_jax)
    for field in je.EnvState._fields:
        a = np.asarray(getattr(s_jax, field))
        b = np.asarray(getattr(s_np, field))
        assert np.array_equal(a, b), f"to_numpy_state diverged on {field}: {a} vs {b}"


def test_ante_exhausts_short_stack_to_allin():
    # A seat (non-blind) with a stack equal to the ante goes all-in on the ante (dead money).
    # button=0 => SB=1, BB=2; seat 4 is a plain live seat. Give it exactly the ante.
    ante = 13
    stacks = [100 * CU, 100 * CU, 100 * CU, 100 * CU, ante, 100 * CU]
    s = _init(stacks, button=0, ante=ante)
    ss = np.asarray(s.seat_status)
    assert ss[4] == ALLIN, f"seat with stack==ante must be all-in after the ante: {ss}"
    assert int(np.asarray(s.stacks)[4]) == 0
    # ante is dead money: it inflates pot_dead + committed_total, not committed_street.
    assert int(np.asarray(s.committed_street)[4]) == 0
    assert int(np.asarray(s.committed_total)[4]) == ante
