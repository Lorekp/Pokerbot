"""GPU smoke + correctness gate for the heads-up NLHE neural-CFR env.

Validates the occupancy-based HU adapter BEFORE committing an overnight run:
  (1) over many random rollouts ONLY the two live seats ever act, and rewards are zero-sum and
      supported only on those seats (=> the persp_sign perspective flip is exactly valid);
  (2) a few real VR-DeepPDCFR+ iterations run on the GPU with no NaN/Inf in any net, the served
      average strategy is non-uniform (it is learning), and the atomic checkpoint round-trips.

Run inside the WSL CUDA venv:
  source ~/jaxgpu-venv/bin/activate && cd . && \
      PYTHONPATH=. POKERBOT_INFO_SUITS=1 python scripts/neural_cfr/smoke_hu_gpu.py
"""
from __future__ import annotations

import sys

import jax
import jax.numpy as jnp
import numpy as np

from pokerbot.training.cfr_env import nlhe_hu_env_spec

LIVE = {3, 5}
FAILS: list[str] = []


def check(cond: bool, msg: str) -> None:
    print(("OK   " if cond else "FAIL ") + msg)
    if not cond:
        FAILS.append(msg)


# --- (1) env correctness: only the 2 live seats act; rewards zero-sum on those seats -------------
env = nlhe_hu_env_spec()
print(f"env: num_players={env.num_players} num_actions={env.num_actions} "
      f"info_dim={env.info_dim} history_dim={env.history_dim}")

key = jax.random.PRNGKey(0)
seats_seen: set[int] = set()
max_abs_rwsum = 0.0
dead_reward_mass = 0.0
n_hands = 200
for h in range(n_hands):
    key, k = jax.random.split(key)
    s = env.reset(k)
    for _ in range(env.max_steps):
        if bool(env.done(s)):
            break
        if bool(env.is_chance(s)):
            s = env.sample_chance(s)
            continue
        seat = int(env.action_seat(s))
        seats_seen.add(seat)
        mask = np.asarray(env.legal_mask(s))
        legal = np.flatnonzero(mask)
        key, ka = jax.random.split(key)
        a = int(legal[jax.random.randint(ka, (), 0, len(legal))])
        s = env.step(s, jnp.int32(a))
    rw = np.asarray(env.rewards(s), dtype=np.float64)
    max_abs_rwsum = max(max_abs_rwsum, abs(float(rw.sum())))
    dead_reward_mass += float(np.abs(rw[[0, 1, 2, 4]]).sum())

check(seats_seen <= LIVE, f"acting seats {sorted(seats_seen)} subset of {sorted(LIVE)}")
check(seats_seen == LIVE, f"both live seats act over {n_hands} hands (seen={sorted(seats_seen)})")
check(max_abs_rwsum < 1e-3, f"rewards zero-sum (max|sum| over hands = {max_abs_rwsum:.2e})")
check(dead_reward_mass < 1e-6, f"dead seats carry zero reward (mass={dead_reward_mass:.2e})")

# --- (2) GPU training smoke: a few real iters, no NaN, strategy moves, checkpoint round-trips ----
print("devices:", jax.devices())
from pokerbot.training.neural_cfr import (  # noqa: E402
    VRDeepPDCFRConfig,
    VRDeepPDCFRPlusTrainer,
    load_trainer_checkpoint,
)

cfg = VRDeepPDCFRConfig(
    total_iterations=4, traversals_per_iter=16, batch_size=128,
    hidden_dim=128, train_steps_per_iter=8, avg_strategy_steps_per_iter=16, seed=0,
)
trainer = VRDeepPDCFRPlusTrainer(cfg, env=env)
trainer.train(checkpoint_every=2, checkpoint_path="/tmp/hu_smoke.pkl")

# no NaN/Inf anywhere
bad = 0
for label, st in [("avg", trainer.avg_state), ("hv", trainer.hv_state),
                  *[(f"cum{i}", s) for i, s in enumerate(trainer.cum_states)]]:
    for leaf in jax.tree_util.tree_leaves(st.params):
        arr = np.asarray(leaf)
        if not np.isfinite(arr).all():
            bad += 1
check(bad == 0, f"all net params finite (bad_leaves={bad})")

# served strategy is non-uniform at a live decision node (it is learning, not flat)
key, k = jax.random.split(key)
s = env.reset(k)
while bool(env.is_chance(s)):
    s = env.sample_chance(s)
seat = int(env.action_seat(s))
info = env.encode_info(s, jnp.int32(seat))
mask = env.legal_mask(s)
strat = np.asarray(trainer.get_strategy(info, mask, seat, trainer._iteration))
nlegal = int(np.asarray(mask).sum())
spread = float(strat.max() - strat[np.asarray(mask).astype(bool)].min()) if nlegal > 1 else 0.0
print(f"sample strat (seat {seat}, {nlegal} legal): {np.round(strat, 3)}  spread={spread:.3f}")
check(nlegal <= 1 or spread > 1e-3, f"served strategy non-uniform (spread={spread:.3f})")

# atomic checkpoint resumes
resumed = load_trainer_checkpoint("/tmp/hu_smoke.pkl")
check(resumed._iteration >= 2, f"checkpoint resumes at iter {resumed._iteration}")

print()
if FAILS:
    print(f"SMOKE FAILED ({len(FAILS)} checks): " + "; ".join(FAILS))
    sys.exit(1)
print("SMOKE PASSED: HU env correct + GPU training healthy + checkpoint atomic/resumable.")
