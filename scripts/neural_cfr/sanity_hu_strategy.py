"""Sanity-check a TRAINED heads-up neural-CFR checkpoint: does it play like poker?

Loads the served average-strategy net, plays K random HU hands, and at each FIRST preflop
decision (the SB/button open) reads the served action distribution. Buckets by 169 starting-hand
class and checks the one thing a real preflop strategy must satisfy: premium hands are aggressive
and rarely fold; trash folds and rarely jams. Also reports non-uniformity + a strength/aggression
rank correlation. Pure inference, no training, no engine edits.

Run (WSL Debian):
  source ~/jaxgpu-venv/bin/activate
  cd .
  PYTHONPATH=. POKERBOT_INFO_SUITS=1 python scripts/neural_cfr/sanity_hu_strategy.py
"""

from __future__ import annotations

import sys

import jax
import jax.numpy as jnp
import numpy as np

from pokerbot.training.cfr_env import nlhe_hu_env_spec
from pokerbot.training.neural_cfr import AverageStrategyNet, load_network_params

CKPT = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/sd_cfr_hu_neural/hu_vrpdcfr.pkl"
K = int(sys.argv[2]) if len(sys.argv) > 2 else 4000

# Action layout (jax_engine): 0=FOLD 1=CHECK 2=CALL 3..9=bet buckets 10=ALL_IN
FOLD = 0
AGGR_START = 3  # bets + all-in are "aggressive"

RANKCH = {14: "A", 13: "K", 12: "Q", 11: "J", 10: "T", 9: "9", 8: "8",
          7: "7", 6: "6", 5: "5", 4: "4", 3: "3", 2: "2"}

PREMIUM = {"AA", "KK", "QQ", "JJ", "AKs", "AKo", "AQs"}
TRASH = {"72o", "82o", "62o", "52o", "42o", "32o", "93o", "83o", "94o", "73o"}


def hand_class(c0: int, c1: int) -> tuple[str, bool]:
    r0, s0 = c0 // 4 + 2, c0 % 4
    r1, s1 = c1 // 4 + 2, c1 % 4
    hi, lo = max(r0, r1), min(r0, r1)
    if hi == lo:
        return RANKCH[hi] + RANKCH[lo], True  # pair
    suited = s0 == s1
    return RANKCH[hi] + RANKCH[lo] + ("s" if suited else "o"), suited


def main() -> int:
    payload = load_network_params(CKPT)
    hidden = int(payload.get("hidden_dim", 256))
    params = payload["average_strategy_params"]
    net = AverageStrategyNet(hidden)

    # --- params finite ---
    leaves = jax.tree_util.tree_leaves(params)
    bad = sum(int(jnp.sum(~jnp.isfinite(x))) for x in leaves)
    n_w = sum(int(x.size) for x in leaves)
    print(f"checkpoint: {CKPT}")
    print(f"  hidden_dim={hidden}  info_suits={payload.get('info_suits')}  "
          f"iteration={payload.get('iteration', '?')}  params={n_w:,}  nonfinite={bad}")
    if bad:
        print("FAIL: non-finite params"); return 1

    env = nlhe_hu_env_spec()

    # --- batch K first-decision nodes ---
    keys = jax.random.split(jax.random.PRNGKey(12345), K)
    states = jax.vmap(env.reset)(keys)
    seats = jax.vmap(env.action_seat)(states).astype(jnp.int32)
    infos = jax.vmap(env.encode_info)(states, seats)
    masks = jax.vmap(env.legal_mask)(states)
    probs = jax.vmap(lambda i, m: net.apply(params, i, m))(infos, masks)
    # acting seat's hole cards
    holes = jax.vmap(lambda st, s: st.holes[s])(states, seats)  # (K,2)

    seats = np.asarray(seats); probs = np.asarray(probs)
    masks = np.asarray(masks); holes = np.asarray(holes)

    # --- env sanity: only the 2 live seats act ---
    seen = set(int(s) for s in seats)
    print(f"  acting seats seen = {sorted(seen)} (expect subset of {{3,5}})")
    if not seen <= {3, 5}:
        print("FAIL: a dead seat acted"); return 1

    # --- non-uniformity ---
    nlegal = masks.sum(1)
    maxp = probs.max(1)
    unif = 1.0 / np.maximum(nlegal, 1)
    spread = float(np.mean(maxp - unif))
    print(f"  mean legal actions={nlegal.mean():.2f}  mean(max_prob - uniform)={spread:+.4f} "
          f"(>0 => non-uniform)")
    if spread < 1e-3:
        print("FAIL: served strategy is ~uniform (net not trained / not loaded)"); return 1

    # --- aggregate by hand class ---
    fold = probs[:, FOLD]
    aggr = probs[:, AGGR_START:].sum(1)
    agg_by, fold_by, n_by = {}, {}, {}
    strength, agg_list = [], []
    for i in range(K):
        cls, _ = hand_class(int(holes[i, 0]), int(holes[i, 1]))
        agg_by.setdefault(cls, []).append(aggr[i])
        fold_by.setdefault(cls, []).append(fold[i])
        n_by[cls] = n_by.get(cls, 0) + 1
        r0, r1 = int(holes[i, 0]) // 4 + 2, int(holes[i, 1]) // 4 + 2
        pair = r0 == r1
        suited = (int(holes[i, 0]) % 4) == (int(holes[i, 1]) % 4)
        strength.append((20 if pair else 0) + max(r0, r1) + 0.5 * min(r0, r1) + (1.0 if suited else 0))
        agg_list.append(aggr[i])

    def group_mean(grp, table):
        vals = [v for c in grp for v in table.get(c, [])]
        return float(np.mean(vals)) if vals else float("nan")

    prem_a, trash_a = group_mean(PREMIUM, agg_by), group_mean(TRASH, agg_by)
    prem_f, trash_f = group_mean(PREMIUM, fold_by), group_mean(TRASH, fold_by)
    print(f"\n  PREMIUM {sorted(PREMIUM)}")
    print(f"    mean aggression={prem_a:.3f}  mean fold={prem_f:.3f}")
    print(f"  TRASH   {sorted(TRASH)}")
    print(f"    mean aggression={trash_a:.3f}  mean fold={trash_f:.3f}")

    # strength/aggression Spearman rank correlation
    s = np.asarray(strength); a = np.asarray(agg_list)
    rs = np.argsort(np.argsort(s)); ra = np.argsort(np.argsort(a))
    rho = float(np.corrcoef(rs, ra)[0, 1])
    print(f"\n  Spearman(strength, aggression) over {K} hands = {rho:+.3f} (want clearly >0)")

    # most/least aggressive classes with enough samples
    well = {c: float(np.mean(v)) for c, v in agg_by.items() if n_by[c] >= 8}
    top = sorted(well.items(), key=lambda kv: -kv[1])[:8]
    bot = sorted(well.items(), key=lambda kv: kv[1])[:8]
    print("  most aggressive classes:  " + ", ".join(f"{c}={p:.2f}" for c, p in top))
    print("  least aggressive classes: " + ", ".join(f"{c}={p:.2f}" for c, p in bot))

    ok = (prem_a > trash_a + 0.10) and (prem_f < trash_f) and (rho > 0.15)
    print("\n" + ("SANITY PASSED: plays like poker (premium aggressive, trash folds, "
                  "strength-correlated)." if ok else
                  "SANITY CONCERN: ordering weaker than expected -- inspect above."))
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
