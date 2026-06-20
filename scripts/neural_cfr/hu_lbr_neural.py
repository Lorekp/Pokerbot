"""HU-LBR exploitability gate for a TRAINED neural-CFR checkpoint.

The existing ``pokerbot/eval/lbr.py`` is already a full heads-up Local Best Response (Lisy & Bowling
2017): it runs the HU game on ``np_engine`` (seats 3,5), tracks a Bayesian belief over the target's
hole cards, and reports a LOWER BOUND on the target's exploitability in mbb/g. It only needs a
``policy_fn(env_np, seat) -> probs``. ``make_blueprint_policy`` serves the TABULAR blueprint; this
script supplies the missing NEURAL serve path so we can measure our HU neural net's exploitability
with NO stack-depth mismatch and NO bet-size translation (unlike a Slumbot match at 200bb).

The numpy ``EnvState`` is the same NamedTuple as the jax one, so we rebuild a jax state field-by-field
and call the EXACT training encoder (``encode_info_state``) + the served ``AverageStrategyNet`` — i.e.
byte-identical to what the net saw during training (provided POKERBOT_INFO_SUITS matches; asserted).

Reference points (Lisy & Bowling / the lbr.py docstring): always-fold ~+750 mbb/g; 2014-16 ACPC bots
+400..1000; DeepStack ~0 (LBR could not exploit it). LOWER is better; positive = exploitable.

Run (WSL Debian) -- CPU jax is fastest for these single-example serves:
  source ~/jaxgpu-venv/bin/activate
  cd .
  PYTHONPATH=. POKERBOT_INFO_SUITS=1 JAX_PLATFORMS=cpu \
    python scripts/neural_cfr/hu_lbr_neural.py --hands 200 --seed 0
"""

from __future__ import annotations

import argparse
import json
import time

import jax
import jax.numpy as jnp
import numpy as np

from pokerbot.engine.jax_engine import (
    EnvState,
    encode_info_state,
    legal_action_mask,
)
from pokerbot.eval.lbr import (
    LBRConfig,
    always_fold_policy,
    lbr_exploitability,
    make_nit_policy,
    uniform_policy,
)
from pokerbot.training.neural_cfr import AverageStrategyNet, load_network_params


def make_neural_policy(checkpoint: str):
    """Return policy_fn(env_np, seat)->np probs serving the trained AverageStrategyNet."""
    payload = load_network_params(checkpoint)
    hidden = int(payload.get("hidden_dim", 256))
    params = payload["average_strategy_params"]
    net = AverageStrategyNet(hidden)

    # Skew guard: the served encode_info_state layout MUST match training (suit features on/off).
    trained_suits = payload.get("info_suits")
    if trained_suits is not None:
        import pokerbot.engine.jax_engine as _je
        if bool(trained_suits) != bool(_je._INFO_SUITS):
            raise RuntimeError(
                f"info-state encoding skew: checkpoint trained with POKERBOT_INFO_SUITS="
                f"{int(bool(trained_suits))} but serving with {int(bool(_je._INFO_SUITS))}. "
                f"Set POKERBOT_INFO_SUITS to match BEFORE importing pokerbot.")

    @jax.jit
    def _serve(state: EnvState, seat):
        info = encode_info_state(state, seat)
        mask = legal_action_mask(state, seat)
        probs = net.apply(params, info, mask)
        return probs * mask, mask

    fields = EnvState._fields

    def policy_fn(env_np, seat: int) -> np.ndarray:
        state = EnvState(**{f: jnp.asarray(getattr(env_np, f)) for f in fields})
        probs, mask = _serve(state, jnp.int32(int(seat)))
        p = np.asarray(probs, dtype=np.float64)
        m = np.asarray(mask, dtype=bool)
        p = p * m
        s = p.sum()
        if s > 1e-12:
            return p / s
        mm = m.astype(np.float64)
        return mm / max(mm.sum(), 1.0)

    return policy_fn, {"hidden_dim": hidden, "iteration": payload.get("iteration"),
                       "info_suits": payload.get("info_suits")}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/sd_cfr_hu_neural/hu_vrpdcfr.pkl")
    ap.add_argument("--policy", choices=["neural", "fold", "uniform", "nit"], default="neural",
                    help="target to measure. fold/uniform/nit are harness CALIBRATION references "
                         "(fold ~+750 mbb/g, uniform large+, nit moderate+).")
    ap.add_argument("--hands", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--range-cap", type=int, default=120)
    ap.add_argument("--equity-rollouts", type=int, default=16)
    ap.add_argument("--stack-bb", type=float, default=None,
                    help="deal at this effective stack depth (BB). Default = engine 100bb. Set e.g. 20 "
                         "to measure the TOURNAMENT-shallow regime the net actually plays.")
    ap.add_argument("--ante-bb", type=float, default=0.0,
                    help="per-player ante (BB), e.g. 0.125 for the tournament structure. Default 0.")
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    if args.policy == "neural":
        policy_fn, meta = make_neural_policy(args.checkpoint)
    elif args.policy == "fold":
        policy_fn, meta = always_fold_policy, {"policy": "always_fold (ref ~+750 mbb/g)"}
    elif args.policy == "uniform":
        policy_fn, meta = uniform_policy, {"policy": "uniform"}
    else:
        policy_fn, meta = make_nit_policy(), {"policy": "nit"}
    cfg = LBRConfig(hands=args.hands, seed=args.seed,
                    range_cap=args.range_cap, equity_rollouts=args.equity_rollouts,
                    stack_bb=args.stack_bb, ante_bb=args.ante_bb)

    print(f"checkpoint: {args.checkpoint}  meta={meta}")
    print(f"LBR config: hands={cfg.hands} seed={cfg.seed} range_cap={cfg.range_cap} "
          f"equity_rollouts={cfg.equity_rollouts}  (lower mbb/g = less exploitable)")
    t0 = time.time()
    res = lbr_exploitability(policy_fn, cfg)
    dt = time.time() - t0

    ci95 = 1.96 * res.stderr_bb_per_hand * 1000.0  # mbb/g
    print(f"\n=== HU-LBR exploitability ===")
    print(f"  mbb/g     = {res.mbb_per_g:+.1f}  +/- {ci95:.1f} (95% CI)")
    print(f"  bb/100    = {res.bb_per_100:+.3f}")
    print(f"  bb/hand   = {res.ev_bb_per_hand:+.5f}  (stderr {res.stderr_bb_per_hand:.5f})")
    print(f"  target decisions = {res.detail.get('target_decisions')}  "
          f"forced-uniform = {res.detail.get('forced_uniform_target_decisions')}")
    print(f"  wall = {dt:.1f}s  ({dt / max(cfg.hands, 1):.2f}s/hand)")
    print(f"\n  reference: always-fold ~+750 mbb/g; ACPC 2014-16 bots +400..1000; DeepStack ~0.")
    print(f"  POSITIVE => exploitable by at least this much (lower bound). NEGATIVE/near-0 => robust.")

    if args.output:
        with open(args.output, "w") as f:
            json.dump({"meta": meta, "mbb_per_g": res.mbb_per_g, "bb_per_100": res.bb_per_100,
                       "ev_bb_per_hand": res.ev_bb_per_hand, "stderr_bb_per_hand": res.stderr_bb_per_hand,
                       "ci95_mbb_g": ci95, "hands": cfg.hands, "seed": cfg.seed,
                       "detail": res.detail, "wall_s": dt}, f, indent=2)
        print(f"\n  wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
