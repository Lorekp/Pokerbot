"""DC1/DC2: pin the pot convention + fee incidence from a captured AoF hand.

Run:  python scripts/analyze_aof_capture_dc1.py runs/aof_capture/aof_20260612_132755.jsonl
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass
    path = Path(sys.argv[1] if len(sys.argv) > 1 else "runs/aof_capture/aof_20260612_132755.jsonl")
    events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

    by_hand: dict[str, list[tuple[str, dict]]] = defaultdict(list)
    for ev in events:
        data = json.loads(ev["data"])
        by_hand[str(data.get("gameHandId"))].append((ev["event"], data))

    sb, bb = 0.25, 0.50
    for hand_id, evs in by_hand.items():
        names = {e for e, _ in evs}
        if "table-show-winners" not in names:
            continue
        print(f"\n=== hand {hand_id} (bb={bb}) ===")
        commits: dict[str, float] = {}
        start: dict[str, float] = {}
        pots: list[float] = []
        for e, d in evs:
            if e == "table-user-action":
                u, cap = d["userName"], d["caption"]
                commits[u] = max(commits.get(u, 0.0), float(d["betAmout"]))
                start.setdefault(u, float(d["userChips"]) + float(d["betAmout"]))
                print(f"  action {cap:>7s} {u:<12s} betTotal={d['betAmout']:.2f} chipsLeft={d['userChips']:.2f}")
            elif e == "table-update-pot-info":
                pots.append(float(d["totalPotAmount"]))
            elif e == "table-show-winners":
                gross = float(d["cumulativePotAmount"])
                net = float(d["cumulativePotAmountWithoutRake"])
                print(f"  winners: pot_field={gross:.2f} withoutRake_field={net:.2f} (field rake={gross - net:.2f})")
                for w in d["winnersData"]:
                    print(f"    {w['userName']:<12s} profitLoss={w['cumulativeProfitLoss']:+.2f}")
                winners = {w["userName"]: float(w["cumulativeProfitLoss"]) for w in d["winnersData"]}
            elif e == "table-seats-update" and any(s.get("lastAction") == "AllIn" for s in d["seatResponseDataList"]):
                final = {s["userName"]: (float(s["userChips"]), s.get("lastAction", "")) for s in d["seatResponseDataList"]}

        pot = pots[-1] if pots else float("nan")
        total_commit = sum(commits.values())
        print(f"  pot stream: {' -> '.join(f'{p:.2f}' for p in pots)}")
        print(f"  sum(commits) = {total_commit:.2f}  vs final pot = {pot:.2f}  match={abs(total_commit - pot) < 0.005}")

        # payouts: final chips - (start chips - commit)
        paid_out = 0.0
        for u, (chips, last) in final.items():
            if u in commits:
                s0 = start.get(u, chips + commits[u])
                recv = chips - (s0 - commits[u])
                if abs(recv) > 0.004:
                    print(f"  payout {u:<12s} start={s0:.2f} commit={commits[u]:.2f} end={chips:.2f} received={recv:.2f}")
                paid_out += max(recv, 0.0)
        retained = pot - paid_out
        print(f"  RETAINED (pot - payouts) = {retained:.2f} chips = {retained / bb:.3f} BB")

        # pot-convention check: all-in players vs folded posters
        allin = [u for u, (c, la) in final.items() if la == "AllIn" and commits.get(u, 0) > 0]
        dead = {u: commits[u] for u in commits if u not in allin}
        k = len(allin)
        stack = 4.0
        v2 = k * stack + sum(dead.values())
        v1 = k * stack + sb + bb + sum(v for u, v in dead.items() if v not in (sb, bb))
        print(f"  all-in: {allin} (k={k})  dead posts: {dead}")
        print(f"  v2 predicts pot = k*stack + folded posts = {v2:.2f}   actual {pot:.2f}  -> {'MATCH' if abs(v2 - pot) < 0.005 else 'no'}")
        print(f"  v1 (blinds double-counted on top) predicts ~{k * stack + sb + bb + sum(dead.values()):.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
