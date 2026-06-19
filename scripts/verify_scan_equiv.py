"""Numerical-equivalence oracle for traversal-scan refactors.

Runs _batched_traversal_scans on FIXED inputs (params, keys, traverser) and hashes/serializes the
emitted (adv, strat, hist, is_trav, active) arrays. A refactor (e.g. the deferred next-node pipeline)
must reproduce these arrays to fp tolerance on the active rows -- a far sharper oracle than the
slow-to-converge Leduc gate. Use:
  python -m scripts.verify_scan_equiv dump   # writes golden from current code
  python -m scripts.verify_scan_equiv check  # compares current code against golden
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from pokerbot.training.cfr_env import leduc_env_spec, nlhe_env_spec
from pokerbot.training.leduc_gate import LEDUC_REWARD_SCALE
from pokerbot.training.neural_cfr import VRDeepPDCFRConfig, VRDeepPDCFRPlusTrainer

GOLD = Path("./runs/scan_equiv_gold.pkl")


def run_case(env, reward_scale, hidden, k, seed):
    cfg = VRDeepPDCFRConfig(hidden_dim=hidden, traversals_per_iter=k, batch_size=64,
                            buffer_capacity=10000, seed=seed, reward_scale=reward_scale)
    tr = VRDeepPDCFRPlusTrainer(cfg, env=env)
    cum_p = tuple(s.params for s in tr.cum_states)
    inst_p = tuple(s.params for s in tr.inst_states)
    hv_p = tr.hv_state.params
    rng = jax.random.PRNGKey(seed + 777)
    keys = jax.random.split(rng, k * 2)
    out = tr._batched_scans(keys[:k], keys[k:], jnp.int32(1), jnp.int32(3),
                            cum_p, inst_p, hv_p, cfg.alpha, cfg.epsilon,
                            cfg.reward_scale, cfg.hidden_dim, env)
    adv, strat, hist, is_trav, active = jax.device_get(out)
    return {"adv": adv, "strat": strat, "hist": hist,
            "is_trav": np.asarray(is_trav), "active": np.asarray(active)}


def all_cases():
    return {
        "nlhe": run_case(nlhe_env_spec(), 10000.0, 64, 16, seed=0),
        "leduc": run_case(leduc_env_spec(), LEDUC_REWARD_SCALE, 64, 16, seed=1),
    }


def _cmp(a, b, active, path, errs):
    if isinstance(a, dict):
        for key in a:
            _cmp(a[key], b[key], active, f"{path}.{key}", errs)
        return
    a = np.asarray(a); b = np.asarray(b)
    # only compare ACTIVE rows (masked-out rows are don't-care)
    m = active.astype(bool)
    af = a[m]; bf = b[m]
    if af.shape != bf.shape:
        errs.append(f"{path}: shape {af.shape} vs {bf.shape}")
        return
    if af.size == 0:
        return
    if np.issubdtype(af.dtype, np.floating):
        d = np.max(np.abs(af.astype(np.float64) - bf.astype(np.float64)))
        if d > 1e-4:
            errs.append(f"{path}: max abs diff {d:.3e}")
    else:
        if not np.array_equal(af, bf):
            errs.append(f"{path}: int/bool mismatch ({np.mean(af!=bf)*100:.1f}% rows)")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "check"
    cur = all_cases()
    if mode == "dump":
        GOLD.parent.mkdir(parents=True, exist_ok=True)
        with GOLD.open("wb") as f:
            pickle.dump(cur, f)
        print(f"wrote golden -> {GOLD}")
    else:
        with GOLD.open("rb") as f:
            gold = pickle.load(f)
        errs = []
        for game in cur:
            active = gold[game]["active"]
            for fld in ("adv", "strat", "hist"):
                _cmp(gold[game][fld], cur[game][fld], active, f"{game}.{fld}", errs)
            if not np.array_equal(gold[game]["active"], np.asarray(cur[game]["active"])):
                errs.append(f"{game}.active mismatch")
            if not np.array_equal(gold[game]["is_trav"][active.astype(bool)],
                                  np.asarray(cur[game]["is_trav"])[active.astype(bool)]):
                errs.append(f"{game}.is_trav mismatch")
        if errs:
            print("MISMATCH:")
            for e in errs:
                print("  -", e)
            sys.exit(1)
        print("EQUIVALENT: all active rows match golden within tol.")
