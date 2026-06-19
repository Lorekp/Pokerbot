"""CLI for the Leduc exploitability gate -- the pre-cloud convergence instrument for VR-DeepPDCFR+.

Trains the REAL neural trainer on Leduc (via the pluggable EnvSpec) and measures the trained average
strategy's exact infoset-BR exploitability vs the tabular-CFR floor. This adjudicates F4 (the deferred
TD-bootstrap-vs-recursive VR estimator): converges toward the floor => faithful, scale on AWS;
plateaus far above => the estimator needs the recursive rewrite before any cloud spend.

Run: python scripts/neural_cfr/leduc_gate.py --iters 800 --train-steps 8
"""

from __future__ import annotations

import argparse
import json
import sys

from pokerbot.training.leduc_gate import run_leduc_gate

# PASS = neural exploitability within this multiple of the tabular floor (neural CFR never reaches
# tabular precision, but a faithful run should land within ~1-2 orders of magnitude, not 1000x).
PASS_RATIO = 30.0


def main(argv) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--iters", type=int, default=800)
    p.add_argument("--traversals", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--train-steps", type=int, default=8)
    p.add_argument("--samples", type=int, default=90)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--reference-iters", type=int, default=2000)
    p.add_argument("--pass-ratio", type=float, default=PASS_RATIO)
    p.add_argument("--log-every", type=int, default=None,
                   help="progress-log cadence in iters (default: ~20 lines over the run)")
    args = p.parse_args(argv[1:])

    r = run_leduc_gate(
        iters=args.iters, traversals=args.traversals, batch_size=args.batch_size, hidden=args.hidden,
        train_steps=args.train_steps, samples=args.samples, seed=args.seed,
        reference_iters=args.reference_iters, log_every=args.log_every, verbose=True,
    )
    verdict = "PASS" if r["ratio_to_floor"] <= args.pass_ratio else "REVIEW"
    print(json.dumps({**{k: round(v, 3) if isinstance(v, float) else v for k, v in r.items()},
                      "pass_ratio": args.pass_ratio, "verdict": verdict}))
    print(f"\nVERDICT: {verdict}  (neural {r['neural_exploitability_mbb']:.1f} mbb/g vs floor "
          f"{r['tabular_floor_mbb']:.1f}, ratio {r['ratio_to_floor']:.1f}x, pass<= {args.pass_ratio}x)")
    if verdict == "PASS":
        print("=> The neural estimator converges on Leduc -> the implementation (incl. the TD-bootstrap "
              "VR estimator) is faithful enough to scale on a small AWS test. BP stays the fallback.")
    else:
        print("=> Plateaus above the floor -> escalate F4: implement the recursive subtree-return "
              "(faithful OS-MCCFR) VR estimator before any cloud spend. Do not scale yet.")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
