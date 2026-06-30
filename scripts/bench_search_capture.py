"""Capture representative postflop (and preflop) subgame roots by replaying hands, so the
benchmark/profile/flamegraph drivers can re-solve a FIXED set of real serve decisions deterministically.

Serve env MUST be set before importing pokerbot (the encoder bakes the abstraction at import). The caller
sets POKERBOT_BETTING_ABSTRACTION/POSTFLOP_BUCKETS/PREFLOP_LOSSLESS/POSTFLOP_NUM_BUCKETS.

Writes a pickle of {hero_seat, env(numpy, picklable), pbs_vector, street} root snapshots to artifacts/.
"""
from __future__ import annotations

import os
import pickle
import random
import sys

import numpy as np

# Coverage diagnostic ON so we can report serve coverage of the captured roots.
os.environ.setdefault("POKERBOT_BP_COVERAGE", "1")

from pokerbot.engine import game
from pokerbot.engine.game import GameState
from pokerbot.engine.np_engine import to_numpy_state
from pokerbot.eval.bots import build_bot
from pokerbot.eval.search_agent import SearchAgent
from pokerbot.search.state_bridge import build_subgame_root


def load_snapshot(path: str) -> dict:
    obj = pickle.load(open(path, "rb"))
    if isinstance(obj, dict) and "strat_sum" in obj:
        return {bytes(k): np.asarray(v) for k, v in obj["strat_sum"].items()}
    return obj


class CapturingHero(SearchAgent):
    """A SearchAgent that records every (postflop) subgame root it is asked to solve, then plays it."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.captured: list[dict] = []
        self.capture_streets = {1, 2, 3}  # flop/turn/river by default

    def __call__(self, state: GameState):
        seat = state.action_seat
        hero = self._state.hero_seat
        if len(self._state.trajectory.token_types) == 0:
            self._init_pbs(state)
        self._state.trajectory.on_decision(state, seat)
        self._sync_pbs(state, seat=None, action=None)
        pbs = self._state.pbs
        root = build_subgame_root(state, pbs, hero)
        street = int(getattr(state.street, "value", state.street))
        # Map Street enum to int via env
        env_np = to_numpy_state(root.env)
        st = int(np.asarray(env_np.street))
        if st in self.capture_streets:
            self.captured.append({
                "hero_seat": hero,
                "env": env_np,            # numpy EnvState (picklable namedtuple of np arrays)
                "pbs_vector": np.asarray(root.pbs_vector),
                "street": st,
            })
        # Defer to the real solve so the hand keeps progressing realistically.
        return super().__call__(state)


def main():
    snap_path = "checkpoints/serve_blueprint_6max.pkl"
    out_path = sys.argv[1] if len(sys.argv) > 1 else "artifacts/search_roots.pkl"
    n_hands = int(sys.argv[2]) if len(sys.argv) > 2 else 120
    want = int(sys.argv[3]) if len(sys.argv) > 3 else 60

    snap = load_snapshot(snap_path)
    print(f"loaded snapshot: {len(snap)} infosets")

    rng = random.Random(1234)
    hero_seat = 0
    hero = CapturingHero(
        use_pluribus_search=True,
        blueprint_snapshot=snap,
        search_preflop=True,
        hero_seat=hero_seat,
        pluribus_leaf_select="uniform",  # CHEAP leaf for capture only; we re-solve w/ gadget later
    )
    hero.solver.config.max_iterations = 4     # keep capture cheap; we re-solve at target iters later
    hero.solver.config.time_budget_ms = 10**9
    hero.solver.config.leaf.rollout_samples = 1
    # Stations at the other seats.
    others = {s: build_bot("calling_station", rng) for s in range(1, 6)}

    captured = []
    for h in range(n_hands):
        hero.reset_hand(hero_seat)
        policies = {hero_seat: hero}
        policies.update(others)
        state = GameState.new_hand(rng=rng)
        try:
            game.play_hand(policies, state=state, rng=rng)
        except Exception as e:  # noqa: BLE001
            print(f"hand {h} aborted: {e!r}")
        captured.extend(hero.captured)
        hero.captured = []
        if h % 10 == 0:
            print(f"  hand {h}: captured {len(captured)} roots so far", flush=True)
        if len(captured) >= want:
            break
    captured = captured[:want]
    print(f"captured {len(captured)} postflop roots over <= {n_hands} hands")

    # Coverage report on the captured roots.
    try:
        from pokerbot.eval.lbr import bp_coverage_report
        rep = bp_coverage_report()
        print("bp_coverage_report:", rep)
    except Exception as e:  # noqa: BLE001
        print("coverage report unavailable:", repr(e))

    by_street = {}
    for c in captured:
        by_street[c["street"]] = by_street.get(c["street"], 0) + 1
    print("captured-by-street:", by_street)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(captured, f)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
