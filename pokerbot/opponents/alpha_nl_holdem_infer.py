"""Standalone inference stub for AlphaNLHoldem subprocess (TF env optional)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

# When AlphaNLHoldem + TF are installed, replace body with real model load.
# This stub returns JSON actions for IPC testing.


def infer(obs: dict) -> dict:
    street = obs.get("street", "PREFLOP")
    pot = float(obs.get("pot_bb", 1.5))
    stack = float(obs.get("stack_bb", 50))
    if street == "PREFLOP":
        return {"action": "raise", "amount": 2.5}
    if pot > stack * 0.5:
        return {"action": "call", "amount": 0}
    return {"action": "check", "amount": 0}


def main() -> None:
    alpha_root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
    _ = alpha_root
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        obs = json.loads(line)
        result = infer(obs)
        print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
