"""Deep-stack tournament decision provider -- closes the >20bb zone on the EXISTING engine.

Above the push/fold zone, tournament play is raise-based and is the cash blueprint's domain. This
wraps the **already-deployed tabular blueprint** (no retrain, no core-engine change) into a deep
decision: it serves the blueprint's chip-EV strategy and applies an ICM risk-premium tightening
scaled by the live bubble factor -- so it is the strong cash engine far from the money (where chips
are $-linear) and tightens correctly near pay jumps.

Mechanism (proven end-to-end, 40/40 key hits on the HU blueprint): construct the numpy ``EnvState``
for the spot -> ``make_blueprint_policy`` returns the trained action distribution -> ICM-tilt -> act.
The local deployable blueprint is heads-up (the deep final-2 endgame); the 6-max blueprint (the
lossless run, in S3) is a DROP-IN to this exact path once it finishes -- same loader, same policy fn.

Serve flags (``POKERBOT_BETTING_ABSTRACTION`` etc.) MUST be set before importing pokerbot, or every
key misses (the train/serve skew :func:`check_blueprint_serve_env` guards) -- this module asserts it.
"""

from __future__ import annotations

import hashlib
import logging
import pickle
from pathlib import Path

import numpy as np

from pokerbot.tournament.icm import bubble_factor

_log = logging.getLogger(__name__)

# action vocab: 0=fold, 1=check, 2=call, 3..10 = bets/raises (chip-committing)
_COMMIT = np.array([i >= 2 for i in range(11)], dtype=bool)

# Only snapshots rooted under these project-owned directories may be loaded. This is the
# P0 pickle-RCE hardening called out in artifacts/REFUTE-TOURN-PROD.md §4.1: a path supplied
# by a non-trusted caller (e.g. the live bridge) cannot pull arbitrary bytes into pickle.
_SNAPSHOT_ROOTS = (
    Path(__file__).resolve().parent.parent.parent / "data",  # <repo>/data
    Path(__file__).resolve().parent.parent.parent / "checkpoints",  # <repo>/checkpoints
    Path(__file__).resolve().parent.parent.parent / "artifacts",  # <repo>/artifacts
)


def _verify_snapshot_integrity(snapshot_path: Path) -> None:
    """If `<snapshot>.sha256` sidecar exists, refuse to load on hash mismatch.

    Mirrors the live-babysitter pattern (MEMORY task #594). If no sidecar is present we
    log a single warning at construction time so an operator can add one for hardening
    without blocking local dev.
    """
    sidecar = snapshot_path.with_suffix(snapshot_path.suffix + ".sha256")
    if not sidecar.exists():
        _log.warning("[deep_provider] no SHA-256 sidecar at %s; load is unguarded", sidecar)
        return
    expected = sidecar.read_text(encoding="utf-8").strip().split()[0].lower()
    h = hashlib.sha256()
    with open(snapshot_path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    actual = h.hexdigest()
    if actual != expected:
        raise ValueError(
            f"[deep_provider] SHA-256 mismatch for {snapshot_path}: "
            f"expected={expected[:16]}.. got={actual[:16]}.. (refusing to unpickle)"
        )


class BlueprintDeepProvider:
    """Serves the deployed blueprint's strategy for a deep spot, ICM-graded by bubble factor."""

    def __init__(self, snapshot_path: str, *, client_bet_sizing: bool = True, strict: bool = False):
        from pokerbot.eval.lbr import check_blueprint_serve_env, make_blueprint_policy
        check_blueprint_serve_env(strict=strict)          # warn/raise on train/serve flag skew

        resolved = Path(snapshot_path).resolve()
        if not any(root.exists() and resolved.is_relative_to(root) for root in _SNAPSHOT_ROOTS):
            raise ValueError(
                f"[deep_provider] snapshot {resolved} is not under any project snapshot root "
                f"({[str(r) for r in _SNAPSHOT_ROOTS]}) - refusing to unpickle"
            )
        _verify_snapshot_integrity(resolved)
        with open(resolved, "rb") as fh:
            snap = pickle.load(fh)
        if isinstance(snap, dict) and "strat_sum" in snap:
            snap = snap["strat_sum"]
        self._n_entries = len(snap)
        self.policy = make_blueprint_policy(snap, client_bet_sizing=client_bet_sizing)

    def action_probs(self, env, seat: int) -> np.ndarray:
        return np.asarray(self.policy(env, seat), dtype=np.float64)

    def decide(self, env, seat: int, stacks, payouts, hero: int, villain: int) -> dict:
        """Deep decision: blueprint strategy, ICM-tightened by the live bubble factor.

        Far from the money (bubble factor ~1) returns the blueprint's chip-EV-optimal action
        unchanged; near a pay jump it shifts weight off chip-committing actions (call/bet/raise)
        toward fold/check, in proportion to the ICM pressure. Returns the chosen action + provenance.
        """
        probs = self.action_probs(env, seat)
        bf = bubble_factor(stacks, payouts, hero, villain)
        if not np.isfinite(bf):
            bf = 50.0
        if bf <= 1.05:                                    # chips $-linear -> trust the cash engine
            a = int(np.argmax(probs))
            return {"action": a, "probs": probs, "bubble_factor": bf, "mode": "chip-EV"}
        tilt = probs.copy()
        tilt[_COMMIT] /= bf                               # ICM risk premium: discount stack-committing lines
        s = tilt.sum()
        tilt = tilt / s if s > 1e-12 else probs
        return {"action": int(np.argmax(tilt)), "probs": tilt, "bubble_factor": bf, "mode": "ICM-tightened"}
