#!/usr/bin/env python3
"""(Re)build data/player_profiles/irc_population_prior.json — per-context Dirichlet pseudocounts for
the loose-passive real-money/online population (IRC medians: VPIP~48%, PFR~10%, AF~0.65).

This hardcodes the known medians into per-context outcome-bin distributions (see
pokerbot.opponent.population_prior). If the IRC source tarball is restored,
scripts/analyze_irc_population.py is the empirical regenerator that can replace these.
"""

from __future__ import annotations

import json
from pathlib import Path

from pokerbot.opponent.context import context_spec_hash
from pokerbot.opponent.population_prior import (
    BIN_NAMES,
    PRIOR_PATH,
    PRIOR_STRENGTH,
    generate_prior,
)


def main() -> None:
    prior = generate_prior()
    payload = {
        "source": "hardcoded IRC medians (VPIP~48%, PFR~10%, AF~0.65 — loose-passive)",
        "context_spec_hash": context_spec_hash(),
        "prior_strength": PRIOR_STRENGTH,
        "bins": list(BIN_NAMES),
        "prior": prior,
    }
    out = Path(PRIOR_PATH)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    print(f"wrote {out} — {len(prior)} contexts, strength {PRIOR_STRENGTH}, spec {payload['context_spec_hash']}")


if __name__ == "__main__":
    main()
