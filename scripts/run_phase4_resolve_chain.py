"""Phase 3-6 verification + Phase 4 re-solve chain (run AFTER any in-flight eval completes).

  1. baseline eval (--legacy)     — pre-Phase-3 gating on the v1 base (the honest 'before')
  2. post_phase3_5_6 eval         — new gating/model, still on the v1 base
  3. HU + 3-max re-solve          — game model v2 (corrected pots + combo deal weights)
  4. 4-max MC batches (--force)   — v2 payoffs, version-stamped
  5. 4-max base grid re-solve     — v2
  6. verify_aof_cfr_base          — artifact firewall
  7. post_phase4 eval             — the full 'after' picture

Aborts on the first failing step. Log: runs/phase4_chain.log
"""

from __future__ import annotations

import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
LOG = _ROOT / "runs" / "phase4_chain.log"

# (name, argv, eval_tag_or_None). Eval steps: skipped when their metrics.json already exists
# (idempotent resume) and NON-FATAL on gate failures (exit 1 = "some gate failed", which the
# final comparison analyzes — only crashes before the artifact is written abort the chain).
STEPS = [
    ("eval baseline (v1 base, LEGACY gating)",
     ["scripts/run_aof_eval_suite.py", "--legacy", "--tag", "baseline_pre_phase3"],
     "baseline_pre_phase3"),
    ("eval post_phase3_5_6 (v1 base, new engine)",
     ["scripts/run_aof_eval_suite.py", "--tag", "post_phase3_5_6"],
     "post_phase3_5_6"),
    ("re-solve HU + 3-max grid (game model v2)",
     ["scripts/solve_cfr_base_grid.py"], None),
    ("regenerate 4-max MC batches (v2, stamped)",
     ["scripts/aof/precompute_4max_batches.py", "--force", "--game-model", "2"], None),
    ("re-solve 4-max base grid (v2)",
     ["scripts/solve_4max_base.py"], None),
    ("verify base artifacts",
     ["scripts/verify_aof_cfr_base.py"], None),
    ("eval post_phase4 (v2 base)",
     ["scripts/run_aof_eval_suite.py", "--tag", "post_phase4"],
     "post_phase4"),
]


def main() -> int:
    # Windows consoles default to cp1252, which cannot encode characters the children may
    # emit (the children themselves run with replace-errors; we must too).
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
            import os
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
                         "- continuing; the final comparison analyzes the gates.")
                    continue
                emit(f"CHAIN ABORTED at: {name}")
                return 1
        emit(f"CHAIN COMPLETE {datetime.now().isoformat(timespec='seconds')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
