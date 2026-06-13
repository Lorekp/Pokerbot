"""Fee re-solve chain: re-solve the AoF base with the OFFICIAL flat showdown fee in-model.

DC2 (capture 2026-06-12) pinned the incidence: the fee is deducted from CONTESTED pots at
showdown (folded players pay nothing, uncontested steals pay nothing). Magnitude per the
official CoinPoker schedule: 0.12BB at 0.25/0.50 Hold'em (user-directed).

  1. HU + 3-max grid re-solve   (game model v2, flat fee 0.12 — stamps manifest flat_fee_bb)
  2. 4-max MC batches (--force) (fee-stamped NPZs; _router_batch hard-rejects mismatches)
  3. 4-max base grid re-solve
  4. verify_aof_cfr_base        (artifact firewall, runtime defaults follow the manifest)
  5. eval suite tag post_fee012 (settlements charge the fee in-model -> gross == net)

Aborts on the first failing step (eval gate failures non-fatal once the artifact exists).
Log: runs/fee_chain.log
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
LOG = _ROOT / "runs" / "fee_chain.log"
FEE = "0.12"

STEPS = [
    ("re-solve HU + 3-max grid (v2, fee 0.12)",
     ["scripts/solve_cfr_base_grid.py", "--flat-fee-bb", FEE], None),
    ("regenerate 4-max MC batches (fee-stamped)",
     ["scripts/aof/precompute_4max_batches.py", "--force", "--game-model", "2",
      "--flat-fee-bb", FEE], None),
    ("re-solve 4-max base grid (v2, fee 0.12)",
     ["scripts/solve_4max_base.py", "--flat-fee-bb", FEE], None),
    ("verify base artifacts",
     ["scripts/verify_aof_cfr_base.py"], None),
    ("eval post_fee012",
     ["scripts/run_aof_eval_suite.py", "--tag", "post_fee012"], "post_fee012"),
]


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as log:
        def emit(s: str) -> None:
            print(s, end="" if s.endswith("\n") else "\n", flush=True)
            log.write(s if s.endswith("\n") else s + "\n")
            log.flush()

        for name, argv, eval_tag in STEPS:
            if eval_tag and (_ROOT / "runs" / "aof_eval" / eval_tag / "metrics.json").is_file():
                emit(f"=== {name} === SKIPPED (runs/aof_eval/{eval_tag}/metrics.json exists)")
                continue
            emit(f"\n=== {name} === {datetime.now().isoformat(timespec='seconds')}")
            t = time.monotonic()
            env = {**os.environ, "PYTHONIOENCODING": "utf-8:replace"}
            proc = subprocess.Popen([sys.executable, "-u", *argv], cwd=_ROOT, env=env,
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                    text=True, encoding="utf-8", errors="replace")
            for line in proc.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
            proc.wait()
            emit(f"--- step took {time.monotonic() - t:.0f}s, exit {proc.returncode}")
            if proc.returncode != 0:
                artifact_ok = bool(eval_tag) and (
                    _ROOT / "runs" / "aof_eval" / eval_tag / "metrics.json").is_file()
                if artifact_ok:
                    emit(f"NOTE: {name} reported failing gates (exit 1) but wrote its artifact "
                         "- continuing; analyze the gates from the metrics.json.")
                    continue
                emit(f"CHAIN ABORTED at: {name}")
                return 1
        emit(f"FEE CHAIN COMPLETE {datetime.now().isoformat(timespec='seconds')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
