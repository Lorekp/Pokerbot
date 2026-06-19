"""Root-cause diagnostic for the VR-DeepPDCFR+ Leduc convergence failure.

Trains a SHORT Leduc run via the same path the gate uses, then at canonical Leduc
infosets prints side by side:
  * tabular CFR equilibrium action probs (the ruler),
  * the trained CURRENT regret-matching policy (advantage nets -> regret_matching),
  * the trained AVERAGE policy (AverageStrategyNet).

Also tracks over training (every N iters) the entropy/argmax of the current and
average policies at a fixed infoset, and brute-forces the EXACT counterfactual
instantaneous advantage at a chosen infoset to compare the sign+ranking against
what the InstantaneousAdvantageNet predicts (the core "F4" question).

Read-only on leduc_tabular_cfr.py / leduc_bridge.py (import only). Owns this file.

Run:  python -m scripts.neural_cfr.diagnose_leduc
"""

from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp

from pokerbot.training.cfr_env import leduc_env_spec
from pokerbot.training.leduc_gate import build_leduc_policy_fn, LEDUC_REWARD_SCALE
from pokerbot.training.neural_cfr import (
    AverageStrategyNet,
    CumulativeAdvantageNet,
    InstantaneousAdvantageNet,
    VRDeepPDCFRConfig,
    VRDeepPDCFRPlusTrainer,
    pdcfr_predicted_regrets,
    regret_matching,
)
from pokerbot.training.leduc_tabular_cfr import tabular_cfr, _all_leduc_deals
from pokerbot.engine.leduc_bridge import (
    PyLeducState,
    apply_action_py,
    jax_from_py,
    py_info_key,
    py_legal,
    py_state_from_deal,
)
from pokerbot.engine import leduc_jax as leduc

ACT_NAMES = ["FOLD", "CALL", "RAISE"]


def fmt(p):
    return "[" + " ".join(f"{x:5.3f}" for x in np.asarray(p, dtype=float)) + "]"


def entropy(p):
    p = np.asarray(p, dtype=float)
    p = p[p > 1e-12]
    return float(-(p * np.log(p)).sum())


# ---------------------------------------------------------------------------
# Canonical preflop infosets: first-to-act seat 0 holding J/Q/K, empty history.
# In Leduc, card ranks are 0,1,2. Build the root state from a deal and read it.
# ---------------------------------------------------------------------------

def preflop_root_for_card(seat0_card: int) -> PyLeducState:
    """A fresh hand where seat 0 holds seat0_card. Public not yet revealed."""
    # deal: (p0, p1), public. opponent card / public irrelevant for seat-0 preflop infoset.
    other = (seat0_card + 1) % 3
    public = (seat0_card + 2) % 3
    return py_state_from_deal((seat0_card, other), public)


def current_policy_fn(trainer: VRDeepPDCFRPlusTrainer):
    """Replicate the trainer's CURRENT (regret-matching) policy used inside traversal."""
    env = trainer.env
    hidden = trainer.config.hidden_dim
    na = env.num_actions
    alpha = trainer.config.alpha
    it = jnp.int32(max(1, trainer._iteration))
    cum_mod = CumulativeAdvantageNet(hidden, na)
    inst_mod = InstantaneousAdvantageNet(hidden, na)

    @jax.jit
    def _cur(cum_params, inst_params, info, mask):
        cum = cum_mod.apply(cum_params, info, mask)
        inst = inst_mod.apply(inst_params, info, mask)
        predicted = pdcfr_predicted_regrets(cum, inst, it, alpha)
        return regret_matching(predicted, mask), cum, inst, predicted

    def f(py_state: PyLeducState, seat: int):
        js = jax_from_py(py_state)
        mask = env.legal_mask(js)
        info = env.encode_info(js, jnp.int32(seat))
        cum_p = trainer.cum_states[seat].params
        inst_p = trainer.inst_states[seat].params
        pol, cum, inst, pred = _cur(cum_p, inst_p, info, mask)
        return (np.asarray(pol), np.asarray(cum), np.asarray(inst), np.asarray(pred),
                np.asarray(mask))

    return f


# ---------------------------------------------------------------------------
# Brute-force EXACT instantaneous (counterfactual) advantage at an infoset.
# This is what the InstantaneousAdvantageNet is supposed to regress toward (in
# expectation over deals + opponent, under the CURRENT policy sigma): the
# advantage v_bar(I,a) - v_node(I), counterfactually weighted, on the same
# reward_scale the trainer uses.
# ---------------------------------------------------------------------------

def exact_instant_advantage(infokey_seat: int, infokey: tuple, sigma_fn, reward_scale: float):
    """Counterfactual advantage per action at the infoset matching `infokey`.

    sigma_fn(state, seat) -> probs[3]. Enumerate all 90 deals; recurse the tree
    under sigma for both players; accumulate, at every visit to a state whose
    py_info_key == infokey, the counterfactual-weighted child values.
    Returns (adv[3], legal[3]) where adv[a] = sum_h cf_reach(h) * (v(h,a) - v(h))
    normalized by sum cf_reach, on the reward_scale (matches trainer targets up
    to the sampling/baseline scheme).
    """
    NA = leduc.NUM_ACTIONS

    val_cache: dict = {}

    def state_value(state: PyLeducState, player: int) -> float:
        """E[reward to `player`] under sigma from this state."""
        if state.done:
            return float(state.rewards[player])
        seat = state.acting
        mask = py_legal(state)
        sig = sigma_fn(state, seat)
        sig = sig * mask
        s = sig.sum()
        sig = sig / s if s > 0 else mask / mask.sum()
        v = 0.0
        for a in range(NA):
            if mask[a]:
                v += sig[a] * state_value(apply_action_py(state, a), player)
        return v

    acc = np.zeros(NA, dtype=np.float64)
    cf_total = 0.0

    def walk(state: PyLeducState, reach_opp_chance: float):
        nonlocal cf_total, acc
        if state.done:
            return
        seat = state.acting
        mask = py_legal(state)
        sig = sigma_fn(state, seat)
        sig = sig * mask
        ssum = sig.sum()
        sig = sig / ssum if ssum > 0 else mask / mask.sum()
        if seat == infokey_seat and py_info_key(state, seat) == infokey:
            v_node = state_value(state, seat)
            for a in range(NA):
                if mask[a]:
                    qa = state_value(apply_action_py(state, a), seat)
                    acc[a] += reach_opp_chance * (qa - v_node)
            cf_total += reach_opp_chance
        for a in range(NA):
            if mask[a]:
                # counterfactual: do NOT multiply by the acting seat's own prob when
                # descending toward the target seat's infoset; but for the OPPONENT's
                # reach we DO weight by their sigma.
                if seat == infokey_seat:
                    walk(apply_action_py(state, a), reach_opp_chance)
                else:
                    walk(apply_action_py(state, a), reach_opp_chance * sig[a])

    deals = _all_leduc_deals()
    chance = 1.0 / len(deals)
    for cards, public in deals:
        root = py_state_from_deal(cards, public)
        walk(root, chance)

    adv = acc / cf_total if cf_total > 0 else acc
    return adv / reward_scale, cf_total


# ---------------------------------------------------------------------------
# Main diagnosis
# ---------------------------------------------------------------------------

def main():
    import os
    iters = int(os.environ.get("DIAG_ITERS", "200"))
    trav = int(os.environ.get("DIAG_TRAV", "256"))
    steps = int(os.environ.get("DIAG_STEPS", "8"))
    print(f"=== VR-DeepPDCFR+ Leduc diagnosis: {iters} iters, trav={trav}, steps={steps} ===\n",
          flush=True)

    env = leduc_env_spec()
    cfg = VRDeepPDCFRConfig(
        total_iterations=iters, traversals_per_iter=trav, batch_size=256,
        hidden_dim=64, buffer_capacity=100_000, reward_scale=LEDUC_REWARD_SCALE,
        seed=0, train_steps_per_iter=steps,
    )
    trainer = VRDeepPDCFRPlusTrainer(cfg, env=env)

    # tabular equilibrium (the ruler). Skippable to avoid CPU contention during quick checks
    # (the equilibrium probs are stable; the first full run already printed them).
    if os.environ.get("DIAG_SKIP_TABULAR") == "1":
        print("skipping tabular CFR (DIAG_SKIP_TABULAR=1)", flush=True)
        tab_avg = {}
    else:
        print("computing tabular CFR equilibrium (800 iters)...", flush=True)
        tab_avg, tab_regret = tabular_cfr(iterations=800)

    # fixed tracking infoset: seat 0, card K (rank 2), preflop, empty history.
    track_state = preflop_root_for_card(2)
    track_seat = 0
    track_key = py_info_key(track_state, track_seat)
    print(f"tracking infoset (seat0, card=K preflop): key={track_key}\n", flush=True)

    cur_fn = current_policy_fn(trainer)

    # trend tracking
    print("--- trend (current vs average policy at seat0/K preflop) ---", flush=True)
    print(f"{'iter':>5} {'cur_policy':>22} {'cur_H':>6} {'avg_policy':>22} {'avg_H':>6}", flush=True)
    for t in range(1, iters + 1):
        trainer.train_iteration(t)
        if t % 25 == 0 or t == 1:
            cur_fn = current_policy_fn(trainer)  # rebind for updated iteration
            cur, cum, inst, pred, mask = cur_fn(track_state, track_seat)
            avg_module = AverageStrategyNet(cfg.hidden_dim, env.num_actions)
            pol_fn = build_leduc_policy_fn(avg_module, trainer.avg_state.params, env)
            avg = pol_fn(jax_from_py(track_state), track_seat)
            print(f"{t:5d} {fmt(cur):>22} {entropy(cur):6.3f} {fmt(avg):>22} {entropy(avg):6.3f}",
                  flush=True)

    print(flush=True)

    # final side-by-side at canonical infosets
    avg_module = AverageStrategyNet(cfg.hidden_dim, env.num_actions)
    pol_fn = build_leduc_policy_fn(avg_module, trainer.avg_state.params, env)
    cur_fn = current_policy_fn(trainer)

    print("=== FINAL side-by-side at canonical preflop infosets (seat 0) ===", flush=True)
    print(f"{'card':>4} {'tabular':>22} {'current(RM)':>22} {'average(net)':>22}", flush=True)
    for card, name in [(0, "J"), (1, "Q"), (2, "K")]:
        st = preflop_root_for_card(card)
        key = py_info_key(st, 0)
        tab = tab_avg.get(key)
        tab_s = fmt(tab) if tab is not None else "  (missing)"
        cur, *_ = cur_fn(st, 0)
        avg = pol_fn(jax_from_py(st), 0)
        print(f"{name:>4} {tab_s:>22} {fmt(cur):>22} {fmt(avg):>22}", flush=True)

    # a couple of postflop spots: seat 0, after CALL/CALL, public revealed.
    print("\n=== FINAL side-by-side at postflop infosets (seat 0, after call/call) ===",
          flush=True)
    print(f"{'card/pub':>10} {'tabular':>22} {'current(RM)':>22} {'average(net)':>22}",
          flush=True)
    for card, pub, name in [(2, 2, "K/K(pair)"), (2, 0, "K/J"), (0, 1, "J/Q")]:
        other = (card + 1) % 3 if (card + 1) % 3 != pub else (card + 2) % 3
        st = py_state_from_deal((card, other), pub)
        st = apply_action_py(st, leduc.CALL)  # seat0 call
        st = apply_action_py(st, leduc.CALL)  # seat1 call -> flop dealt, seat0 to act
        if st.acting != 0:
            # ensure seat 0 is to act; if not, skip
            print(f"{name:>10}  (seat0 not to act: acting={st.acting})", flush=True)
            continue
        key = py_info_key(st, 0)
        tab = tab_avg.get(key)
        tab_s = fmt(tab) if tab is not None else "  (missing)"
        cur, *_ = cur_fn(st, 0)
        avg = pol_fn(jax_from_py(st), 0)
        print(f"{name:>10} {tab_s:>22} {fmt(cur):>22} {fmt(avg):>22}", flush=True)

    # ------------------------------------------------------------------
    # F4 test: exact counterfactual advantage vs the InstantaneousAdvantageNet
    # at the tracking infoset, under the trainer's CURRENT policy.
    # ------------------------------------------------------------------
    print("\n=== F4 test: EXACT cf advantage vs net prediction (seat0/K preflop) ===",
          flush=True)

    def sigma_fn(state, seat):
        pol, *_ = cur_fn(state, seat)
        return pol

    adv_exact, cf_tot = exact_instant_advantage(track_seat, track_key, sigma_fn,
                                                cfg.reward_scale)
    cur, cum, inst, pred, mask = cur_fn(track_state, track_seat)
    print(f"legal mask                 : {mask.astype(int)}", flush=True)
    print(f"EXACT cf advantage (scaled): {fmt(adv_exact)}  (cf_total={cf_tot:.3f})", flush=True)
    print(f"InstantaneousAdvantageNet  : {fmt(inst)}", flush=True)
    print(f"CumulativeAdvantageNet     : {fmt(cum)}", flush=True)
    print(f"pdcfr predicted regrets    : {fmt(pred)}", flush=True)
    print(f"-> current policy (RM)     : {fmt(cur)}", flush=True)

    # sign/ranking agreement on legal actions
    legal_idx = [a for a in range(3) if mask[a]]
    ex_rank = sorted(legal_idx, key=lambda a: adv_exact[a], reverse=True)
    net_rank = sorted(legal_idx, key=lambda a: inst[a], reverse=True)
    print(f"exact best action          : {ACT_NAMES[ex_rank[0]]}", flush=True)
    print(f"net   best action          : {ACT_NAMES[net_rank[0]]}", flush=True)
    print(f"ranking match              : {ex_rank == net_rank}", flush=True)

    # tabular equilibrium current policy at this infoset for reference
    _tk = tab_avg.get(track_key)
    print(f"tabular equilibrium policy : {fmt(_tk) if _tk is not None else '(skipped)'}",
          flush=True)

    # exact infoset-BR exploitability of the served average strategy (gate number)
    from pokerbot.training.leduc_tabular_cfr import infoset_exploitability_mbb_from_policy
    print("\ncomputing infoset-BR exploitability of served avg strategy...", flush=True)
    expl, *_ = infoset_exploitability_mbb_from_policy(pol_fn, samples=90)
    print(f"NEURAL avg-strategy exploitability : {expl:.1f} mbb/g", flush=True)


if __name__ == "__main__":
    main()
