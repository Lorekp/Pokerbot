#!/usr/bin/env python3
"""6-max empirical eval: LBR proxy + baseline head-to-head stub."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pokerbot.eval.hu_lbr_proxy import hu_lbr_proxy_ev
from pokerbot.training.sd_cfr_nlhe import SDCFRNLHEConfig, SDCFRNLHETrainer, load_nlhe_checkpoint


def main() -> None:
    p = argparse.ArgumentParser(description="6-max SD-CFR empirical eval")
    p.add_argument("--checkpoint", default="checkpoints/sd_cfr_6max_smoke.pkl")
    p.add_argument("--hands", type=int, default=24)
    p.add_argument("--output", default="logs/sd_cfr_6max_empirical.json")
    args = p.parse_args()

    cfg = SDCFRNLHEConfig(occupancy=(True, True, True, True, True, True))
    trainer = SDCFRNLHETrainer(cfg)
    if Path(args.checkpoint).exists():
        load_nlhe_checkpoint(trainer, Path(args.checkpoint))

    lbr = hu_lbr_proxy_ev(trainer, hands=args.hands)
    report = {
        "checkpoint": args.checkpoint,
        "lbr_proxy": {
            "exploiter_ev_bb_per_hand": lbr.exploiter_ev_bb_per_hand,
            "hands": lbr.hands,
            "note": lbr.note,
        },
        "baseline_h2h": "stub — wire static 6-max baseline bot for full empirical gate",
        "theory_sanity": "manual — position-aware aggression / preflop ranges eyeball",
        "no_nash_claim": True,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
