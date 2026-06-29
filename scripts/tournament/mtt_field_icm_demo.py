"""Demonstration: does FIELD-WIDE ICM awareness add $ROI in a multi-table MTT vs an ICM-naive (chip-EV) field?

Runs the multi-table simulator (pokerbot.tournament.mtt_sim) twice against the SAME chip-EV field:
  * an ICM-AWARE hero (tightens near the GLOBAL pay jump via field_bubble_factor), vs
  * a chip-EV hero (bubble-blind, plays the field's own strategy).
The uplift (ICM-aware ROI - chip-EV ROI) is the measured value of the multi-table field-ICM layer: the
ICM-aware hero should LADDER past an ICM-naive field that over-gambles on the bubble. Top-heavy payout (where
ICM pressure is strongest). Honest: the all-in-lottery hand model is a simplification, so this measures the
DIRECTION + rough magnitude of the field-ICM edge, not a live-faithful win rate.
"""

from __future__ import annotations

import argparse
import random

from pokerbot.tournament.mtt_sim import chip_ev_strategy, icm_aware_strategy, measure_roi


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--players", type=int, default=18)
    ap.add_argument("--sims", type=int, default=400)
    ap.add_argument("--bf-sims", type=int, default=150)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--icm-sens", type=float, default=0.14, help="ICM-aware tightening per (BF-1)")
    ap.add_argument("--field-jam", type=float, default=0.42, help="field's chip-EV jam threshold (higher=tighter)")
    ap.add_argument("--field-call", type=float, default=0.58, help="field's chip-EV call threshold")
    ap.add_argument("--realistic", action="store_true",
                    help="LIVE-FIDELITY hands: real hole cards + real HU matchup equity (vs the uniform "
                         "all-in-lottery abstraction) — the honest test that the field-ICM edge isn't a "
                         "simplified-model artifact.")
    args = ap.parse_args()

    # Top-heavy payout (~28% paid, steep) -> ICM pressure is real near the bubble.
    payouts = [38.0, 24.0, 16.0, 11.0, 7.0]
    # The hero shares the field's BASE thresholds; the ONLY difference is field-ICM awareness -> the uplift is
    # purely the value of field-wide ICM, not a different base style. A tighter (--field-jam ~0.55) field is a
    # disciplined reg-like-but-ICM-BLIND opponent (the realistic strong field, not an over-gambler).
    fj, fc = args.field_jam, args.field_call
    field = chip_ev_strategy(jam=fj, call=fc)            # chip-EV (ICM-blind) field — discipline set by fj/fc
    icm_hero = icm_aware_strategy(jam=fj, call=fc, sens=args.icm_sens)
    flat_hero = chip_ev_strategy(jam=fj, call=fc)        # control: hero == field (no ICM awareness)

    print(f"[mtt-icm-demo] players={args.players} sims={args.sims} bf_sims={args.bf_sims} "
          f"payout={payouts} (top-heavy, {len(payouts)}/{args.players} paid)", flush=True)

    icm = measure_roi(args.players, payouts, icm_hero, field, n_sims=args.sims,
                      rng=random.Random(args.seed), bf_sims=args.bf_sims, realistic=args.realistic)
    flat = measure_roi(args.players, payouts, flat_hero, field, n_sims=args.sims,
                       rng=random.Random(args.seed), bf_sims=1, realistic=args.realistic)

    def line(tag, r):
        print(f"  {tag:14s} ROI={r['roi']:+7.3f}  mean_payout={r['mean_payout']:.3f}  "
              f"ITM={r['itm_rate']:.3f}  mean_finish={r['mean_finish']:.2f}", flush=True)

    print("[mtt-icm-demo] === results (same chip-EV field both rows) ===", flush=True)
    line("ICM-aware", icm)
    line("chip-EV (ctrl)", flat)
    uplift = icm["roi"] - flat["roi"]
    print(f"[mtt-icm-demo] FIELD-ICM UPLIFT (ROI): {uplift:+.4f}  "
          f"({'field-ICM adds value' if uplift > 0 else 'no measured edge in this sim'})", flush=True)
    print(f"[mtt-icm-demo] ITM uplift: {icm['itm_rate'] - flat['itm_rate']:+.4f} "
          f"(ICM-aware ladders into the money more often if positive)", flush=True)


if __name__ == "__main__":
    main()
