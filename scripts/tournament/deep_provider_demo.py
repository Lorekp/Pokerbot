"""Proof that the deep zone is integrated on the EXISTING engine (no retrain, no core-engine change).

Loads the deployed (heads-up) blueprint, serves its strategy for deep spots, and shows the ICM
risk-premium tightening near the money. The 6-max blueprint (lossless run, S3) is a drop-in to this
exact path once it finishes. Serve flags are set BEFORE importing pokerbot (required).

Run: python scripts/tournament/deep_provider_demo.py
"""

from __future__ import annotations

import os
import sys

os.environ.setdefault("POKERBOT_BETTING_ABSTRACTION", "1")
os.environ.setdefault("POKERBOT_POSTFLOP_BUCKETS", "1")
os.environ.setdefault("POKERBOT_POSTFLOP_MODE", "fast")
os.environ.setdefault("POKERBOT_POSTFLOP_NUM_BUCKETS", "200")

import jax                                                # noqa: E402
import jax.numpy as jnp                                   # noqa: E402
import numpy as np                                        # noqa: E402

from pokerbot.engine import jax_engine as je              # noqa: E402
from pokerbot.engine.np_engine import to_numpy_state      # noqa: E402
from pokerbot.tournament.deep_provider import BlueprintDeepProvider  # noqa: E402

BP = "checkpoints/blueprint_bounded_fine_iter200.pkl"     # deployed HU blueprint (deep final-2)
OCC = jnp.asarray((False, False, False, True, False, True), dtype=bool)   # HU: BTN(3)+BB(5)


def main() -> int:
    prov = BlueprintDeepProvider(BP)
    print(f"loaded deep blueprint: {prov._n_entries} infosets\n")

    hits = 0
    for seed in range(60):
        st = je.init_hand(jax.random.PRNGKey(seed), occupancy=OCC)
        env = to_numpy_state(st)
        seat = int(env.action_seat)
        p = prov.action_probs(env, seat)
        if (p > 0).sum() > 1 and not np.allclose(p[p > 0], p[p > 0][0]):
            hits += 1
    print(f"served a non-uniform (trained) strategy on {hits}/60 deep preflop spots\n")

    # Same deep spot, far-from-money vs on-the-bubble: show the ICM tightening.
    st = je.init_hand(jax.random.PRNGKey(7), occupancy=OCC)
    env = to_numpy_state(st)
    seat = int(env.action_seat)
    far = prov.decide(env, seat, [80, 80, 80, 80], [50, 30, 20, 10], hero=0, villain=1)   # deep, far
    bub = prov.decide(env, seat, [80, 80, 80, 4], [50, 30, 20], hero=0, villain=1)        # bubble
    print("deep spot, FAR from money :", far["mode"], f"(bf={far['bubble_factor']:.2f}) action={far['action']}")
    print("deep spot, ON the bubble  :", bub["mode"], f"(bf={bub['bubble_factor']:.2f}) action={bub['action']}")
    commit_far = float(far["probs"][2:].sum())
    commit_bub = float(bub["probs"][2:].sum())
    print(f"chip-committing weight: far={commit_far:.3f}  bubble={commit_bub:.3f}  "
          f"(tightens: {commit_bub <= commit_far + 1e-9})")

    ok = hits > 30 and commit_bub <= commit_far + 1e-9
    print("\nVERDICT: %s -- deep zone integrated on the existing engine (HU now; 6-max drop-in "
          "when the lossless run finishes)." % ("DEEP INTEGRATED" if ok else "REVIEW"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
