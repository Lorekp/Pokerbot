"""Deep-ICM evaluator P2 — the FALSIFIABLE held-out headline (workflow wexmo5p7v).

Does ICM-grading a deep (>20bb) near-money decision beat chip-EV? The trap (caught in design): scoring both
arms under the SAME icm_action_value that DEFINES the ICM arm as its argmax makes delta>=0 a tautology. The
FIX = a held-out SCORE table independent of the CHOOSE table:

  per near-money <=4-live flop spot:
    CHOOSE table  = build_action_outcomes(seed A, R_c rollouts)
    SCORE table   = build_action_outcomes(seed B disjoint, R_s > R_c rollouts)   # independent, higher fidelity
    a_icm  = decide_deep(deal_struck=False) over CHOOSE     # ICM valuation
    a_chip = decide_deep(deal_struck=True)  over CHOOSE     # chip-EV valuation
    delta_$ = icm_action_value(SCORE[a_icm]) - icm_action_value(SCORE[a_chip])   # scored on HELD-OUT data

Because a_icm/a_chip are chosen on a DIFFERENT table than they are scored on, delta_$ CAN be negative -> it
measures REALIZED $-ICM saved by the overlay's choice, not a positivity identity. Report FLIP-RATE first
(delta_$ is exactly 0 on non-flip spots), then mean delta_$ +/- 95% CI over flips. NEGATIVE CONTROL:
shuffle the ICM payout labels -> delta_$ must collapse to ~0 (a real instrument must fail this).

Run: python scripts/tournament/deep_icm_grade.py [N_spots]
"""
from __future__ import annotations

import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from pokerbot.engine.game import ActionType, GameState
from pokerbot.tournament.action_outcomes import build_action_outcomes
from pokerbot.tournament.decide import decide_deep
from pokerbot.tournament.icm import bubble_factor
from pokerbot.tournament.icm_overlay import icm_action_value

PAYOUTS = [50.0, 30.0, 20.0]                 # 6 players, top-3 paid -> two pay jumps (bubble at 4th)
# Near-money stack ladder (bb): a couple of short stacks make the bubble bite for the mid stacks.
# DEEP (default): 22-38bb -> deep postflop, the regime under test. SHORT (--short): 4-18bb -> the
# positive control: ICM SHOULD flip decisions here (busting is one all-in away), so a NON-zero flip-rate
# validates the instrument and confirms the deep-0% is a real DEPTH effect, not a harness bug.
DEEP_STACKS = [38.0, 32.0, 27.0, 22.0, 13.0, 7.0]
SHORT_STACKS = [18.0, 15.0, 13.0, 11.0, 7.0, 4.0]
NEAR_MONEY_STACKS = DEEP_STACKS               # overridden by --short in main()


def _near_money_flop_node(rng: random.Random):
    """A 6-handed deep near-money FLOP node: set the bubble stack ladder, deal, play preflop passively to
    a flop with >=2 live, hero = the player to act. Returns (state, hero, handstart_stacks) or None."""
    state = GameState.new_hand(rng=rng)
    perm = list(range(6))
    rng.shuffle(perm)
    for i, p in enumerate(state.players):
        p.stack_bb = NEAR_MONEY_STACKS[perm[i]]
    handstart = [float(p.stack_bb) for p in state.players]
    state.deal_hole_cards()
    for _ in range(30):
        if state.is_terminal() or len(state.board) >= 3 or not state.actors_remaining():
            break
        seat = state.action_seat
        if state.players[seat].folded or state.players[seat].all_in:
            state.advance_to_next_actor()
            continue
        legal = state.legal_actions()
        act = next((a for a in legal if a.action_type in (ActionType.CHECK, ActionType.CALL)),
                   legal[0] if legal else None)
        if act is None:
            break
        state.apply_action(act)
        if not state.is_terminal():
            state.advance_to_next_actor()
    if state.is_terminal() or len(state.board) < 3 or not state.actors_remaining():
        return None
    return state, state.action_seat, handstart


def _icm_stacks(state, handstart):
    # Tournament stacks for ICM = hand-START stacks (the chip_delta from the rollout is the full-hand net,
    # so stacks_handstart + delta = post-hand stacks). A busted-out (folded pre-set) seat keeps its stack.
    return list(handstart)


def grade_spot(state, hero, handstart, payouts, rng) -> dict | None:
    legal = state.legal_actions()
    if len(legal) < 2:
        return None
    # Candidates MUST include the STACK-THREATENING actions (all-in + biggest bet) — that is where ICM
    # diverges from chip-EV (busting costs $-equity near the money). fold/check/small-bet alone never flip.
    fold = next((a for a in legal if a.action_type == ActionType.FOLD), None)
    call = next((a for a in legal if a.action_type in (ActionType.CHECK, ActionType.CALL)), None)
    allin = next((a for a in legal if a.action_type == ActionType.ALL_IN), None)
    bets = [a for a in legal if a.action_type in (ActionType.BET, ActionType.RAISE)]
    cands, seen = [], set()
    for a in (fold, call, (bets[-1] if bets else None), (bets[0] if bets else None), allin):
        if a is not None and id(a) not in seen:
            seen.add(id(a))
            cands.append(a)
    if len(cands) < 2:
        return None
    stacks = _icm_stacks(state, handstart)
    # require real bubble pressure at this spot (else ICM == chip-EV trivially)
    live = [i for i in range(len(stacks)) if not state.players[i].folded]
    bf = max((bubble_factor(stacks, payouts, hero, v) for v in live if v != hero), default=1.0)
    choose = build_action_outcomes(state, cands, rollouts=20, n_runouts=24, rng=random.Random(rng.getrandbits(32)))
    score = build_action_outcomes(state, cands, rollouts=60, n_runouts=48, rng=random.Random(rng.getrandbits(32)))
    a_icm = decide_deep(stacks, payouts, hero, choose, deal_struck=False).action
    a_chip = decide_deep(stacks, payouts, hero, choose, deal_struck=True).action
    if a_icm not in score or a_chip not in score:
        return None
    d = (icm_action_value(stacks, score[a_icm], payouts, hero)
         - icm_action_value(stacks, score[a_chip], payouts, hero))
    # negative control: shuffle the payout labels -> the ICM/chip distinction should vanish
    shuf = list(payouts)
    random.Random(rng.getrandbits(32)).shuffle(shuf)
    a_icm_s = decide_deep(stacks, shuf, hero, choose, deal_struck=False).action
    a_chip_s = decide_deep(stacks, shuf, hero, choose, deal_struck=True).action
    d_ctrl = (icm_action_value(stacks, score.get(a_icm_s, score[a_icm]), shuf, hero)
              - icm_action_value(stacks, score.get(a_chip_s, score[a_chip]), shuf, hero))
    return {"flip": a_icm != a_chip, "delta": float(d), "bubble_factor": float(bf), "delta_ctrl": float(d_ctrl)}


def main(argv) -> int:
    global NEAR_MONEY_STACKS
    args = [a for a in argv[1:] if a != "--short"]
    if "--short" in argv:
        NEAR_MONEY_STACKS = SHORT_STACKS      # POSITIVE CONTROL: ICM should flip at short stacks
    n = int(args[0]) if args else 120
    rng = random.Random(7)
    t0 = time.time()
    spots, flips, deltas, deltas_ctrl, bfs = 0, 0, [], [], []
    for _ in range(n * 3):                       # oversample; many spots end before a live flop
        if spots >= n:
            break
        s = _near_money_flop_node(rng)
        if s is None:
            continue
        g = grade_spot(*s, payouts=PAYOUTS, rng=rng)
        if g is None:
            continue
        spots += 1
        bfs.append(g["bubble_factor"])
        deltas_ctrl.append(g["delta_ctrl"])
        if g["flip"]:
            flips += 1
            deltas.append(g["delta"])
    import statistics as st
    flip_rate = flips / spots if spots else 0.0
    mean_d = st.fmean(deltas) if deltas else 0.0
    se_d = (st.pstdev(deltas) / (len(deltas) ** 0.5)) if len(deltas) > 1 else 0.0
    mean_ctrl = st.fmean(deltas_ctrl) if deltas_ctrl else 0.0
    se_ctrl = (st.pstdev(deltas_ctrl) / (len(deltas_ctrl) ** 0.5)) if len(deltas_ctrl) > 1 else 0.0
    out = {
        "spots": spots, "flip_rate": flip_rate, "n_flips": flips, "mean_bubble_factor": st.fmean(bfs) if bfs else 0.0,
        "held_out_delta_mean": mean_d, "held_out_delta_se": se_d, "held_out_delta_lcb95": mean_d - 1.96 * se_d,
        "icm_beats_chipev_on_flips_ci": bool(mean_d - 1.96 * se_d > 0.0),
        "negative_control_delta_mean": mean_ctrl, "negative_control_delta_se": se_ctrl,
        "negative_control_passes": bool(abs(mean_ctrl) < 1.96 * se_ctrl + 1e-9),   # control straddles 0
        "elapsed_s": time.time() - t0,
    }
    art = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                       "artifacts", "deep_icm_grade.json")
    os.makedirs(os.path.dirname(art), exist_ok=True)
    with open(art, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print(f"Deep-ICM held-out grading: {spots} near-money spots (mean bubble_factor {out['mean_bubble_factor']:.2f})")
    print(f"  FLIP-RATE (a_icm != a_chip): {flip_rate:.1%}  ({flips}/{spots})")
    print(f"  held-out delta_$ on flips  : {mean_d:+.4f} +/- {se_d:.4f}  (95% LCB {out['held_out_delta_lcb95']:+.4f})"
          f"  -> ICM beats chip-EV: {out['icm_beats_chipev_on_flips_ci']}")
    print(f"  NEGATIVE CONTROL (shuffled): {mean_ctrl:+.4f} +/- {se_ctrl:.4f}  -> collapses to ~0: {out['negative_control_passes']}")
    print(f"wrote {art}  ({out['elapsed_s']:.0f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
