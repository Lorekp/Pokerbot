"""Local bit-identical check: single-threaded vs PersistentPool over 2 iterations (manual)."""
import numpy as np

from pokerbot.training.sd_cfr_nlhe import SDCFRNLHEConfig, SDCFRNLHETrainer
from pokerbot.training.sd_cfr_nlhe_pool import PersistentPool


def mk():
    cfg = SDCFRNLHEConfig(iterations=2, traversals_per_iter=5, deal_sweeps=1, seed=11,
                          max_traverse_depth=12, sft_checkpoint=None)
    return SDCFRNLHETrainer(cfg)


def cmp(a, b, name):
    ka, kb = set(a), set(b)
    if ka != kb:
        print(f"{name}: KEY MISMATCH ref={len(ka)} pool={len(kb)}")
        return False
    bad = sum(0 if np.array_equal(np.asarray(a[k]), np.asarray(b[k])) else 1 for k in ka)
    print(f"{name}: {len(ka)} keys, {bad} mismatches")
    return bad == 0


def params_max_abs_diff(p1, p2):
    from flax.core import unfreeze
    import jax
    f1, f2 = unfreeze(p1), unfreeze(p2)
    leaves1 = jax.tree_util.tree_leaves(f1)
    leaves2 = jax.tree_util.tree_leaves(f2)
    return max(float(np.max(np.abs(np.asarray(a) - np.asarray(b)))) for a, b in zip(leaves1, leaves2))


if __name__ == "__main__":
    ref = mk()
    losses_ref = []
    for n in (1, 2):
        np.random.seed(11 + n - 1)
        losses_ref.append(ref.train_iteration(float(n)))

    pool_t = mk()
    pool = PersistentPool(3)
    losses_pool = []
    try:
        for n in (1, 2):
            np.random.seed(11 + n - 1)
            losses_pool.append(pool.run_iteration(pool_t, float(n)))
    finally:
        pool.close()

    print("losses_ref =", [round(x, 6) for x in losses_ref])
    print("losses_pool=", [round(x, 6) for x in losses_pool])
    loss_ok = all(abs(a - b) < 1e-6 for a, b in zip(losses_ref, losses_pool))
    o1 = cmp(ref._cum_regret, pool_t._cum_regret, "cum_regret")
    o2 = cmp(dict(ref._strat_sum), dict(pool_t._strat_sum), "strat_sum")
    pdiff = params_max_abs_diff(ref.state.params, pool_t.state.params)
    print(f"net params max|diff| = {pdiff:.3e}")
    print("BIT_IDENTICAL:", o1 and o2 and loss_ok and pdiff < 1e-5)
