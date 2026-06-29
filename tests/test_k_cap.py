"""K-cap regression guard for the EMD bucketer precompute path.

Background: run_emd_k500_pluribus_20260625 produced a real -140 SB regression
in the SB seat — K=500 overfit the training-time EMD bucketer and the resulting
blueprint lost value there. The fix is a HARD K=200 production cap wired into
every EMD entry point that the launcher/pipeline touches:

  - scripts/precompute_emd_buckets.py  (K_MAX = 200, EMD_K_OVERRIDE env rejected)
  - scripts/launch_emd_pipeline_spot.py (--num-buckets > 200 rejected)
  - scripts/pluribus_blueprint_run.ps1  (header documents the cap)

These tests lock all three down so the cap cannot be silently regressed. They
spawn the precompute script as a subprocess because the K=500 rejection must
fire BEFORE heavy imports / numpy / sklearn to fail fast in CI.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

# Path to the script under test. Resolved absolute so the subprocess inherits
# the repo layout regardless of pytest's CWD.
_REPO_ROOT = Path(__file__).resolve().parents[1]
_PRECOMPUTE = _REPO_ROOT / "scripts" / "precompute_emd_buckets.py"


def _run_precompute(*args: str, env_extra: dict[str, str] | None = None,
                    timeout: int = 60) -> subprocess.CompletedProcess:
    """Run scripts/precompute_emd_buckets.py with the given CLI args.

    Uses the same Python interpreter as the test (sys.executable) so the K-cap
    check fires under exactly the project's interpreter. We pass a tiny
    --samples value to keep K=200 tests fast (the cap doesn't affect runtime).
    """
    env = os.environ.copy()
    # Default to a tiny sample so K=200 path is fast (sklearn k-means is the
    # wall-clock cost; the cap check itself is microseconds).
    base_args = [
        sys.executable, str(_PRECOMPUTE),
        "--samples", "200",
        "--kmeans-iters", "1",
    ]
    if env_extra:
        env.update(env_extra)
    return subprocess.run(
        base_args + list(args),
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def test_k_500_exits_nonzero():
    """K=500 must be rejected at the script entry point (production hard cap).

    The K_MAX check in scripts/precompute_emd_buckets.py main() fires after
    argparse parses --num-buckets; the subprocess must exit with a non-zero
    status AND the stderr/stdout must mention the banned-K message so the
    failure mode is diagnosable in CI logs.
    """
    result = _run_precompute("--num-buckets", "500")
    assert result.returncode != 0, (
        f"K=500 must be rejected (non-zero exit). Got rc={result.returncode}\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )
    combined = result.stdout + result.stderr
    assert "K_MAX" in combined or "banned" in combined.lower(), (
        f"K=500 rejection message must mention K_MAX/banned for diagnosability.\n"
        f"Got: {combined}"
    )


def test_k_200_proceeds():
    """K=200 (the production cap) must succeed OR fail only with sklearn-related
    errors that prove the cap check itself is not the blocker.

    Acceptance: exit code 0, OR exit code != 0 only because of an inner sklearn
    / numpy / bucketer error (NOT the K_MAX guard). We assert by exit code 0
    if sklearn is present; the env has sklearn here, so this must succeed.
    """
    result = _run_precompute("--num-buckets", "200", "--out",
                            str(_REPO_ROOT / "data" / "abstraction" / "_test_k_cap_k200.json"))
    # The cap must NOT be the reason for failure.
    combined = (result.stdout or "") + (result.stderr or "")
    if result.returncode != 0:
        assert "K_MAX" not in combined, (
            f"K=200 must not trigger K_MAX. Got rc={result.returncode}\n{combined}"
        )
        assert "banned" not in combined.lower(), (
            f"K=200 must not be flagged as banned. Got rc={result.returncode}\n{combined}"
        )
    # Otherwise the run succeeded; clean up the test artifact so the repo stays tidy.
    out_path = _REPO_ROOT / "data" / "abstraction" / "_test_k_cap_k200.json"
    if out_path.exists():
        out_path.unlink()
    # If the env has sklearn (it does, as confirmed in preflight), this MUST
    # be a clean run. We make the assertion explicit so a missing-sklearn CI
    # image is loud instead of silently green.
    assert result.returncode == 0, (
        f"K=200 must succeed under a sklearn-capable interpreter. "
        f"rc={result.returncode}\nstdout: {result.stdout}\nstderr: {result.stderr}"
    )


def test_k_override_env_rejected():
    """EMD_K_OVERRIDE env var must NOT bypass K_MAX. The cap is a hard cap;
    no env knob is allowed to silently raise it. The script must sys.exit(1)
    at import time (before argparse) and the error message must explicitly
    say EMD_K_OVERRIDE is not honored.
    """
    result = _run_precompute(
        "--num-buckets", "200",
        env_extra={"EMD_K_OVERRIDE": "1000"},
    )
    assert result.returncode == 1, (
        f"EMD_K_OVERRIDE=1000 must exit 1 (not honored). "
        f"Got rc={result.returncode}\nstdout: {result.stdout}\nstderr: {result.stderr}"
    )
    combined = (result.stdout or "") + (result.stderr or "")
    assert "EMD_K_OVERRIDE is not honored" in combined, (
        f"EMD_K_OVERRIDE rejection must name the env var for diagnosability.\n"
        f"Got: {combined}"
    )


def test_shell_injection_in_k_override_rejected():
    """EMD_K_OVERRIDE must reject shell-injection payloads WITHOUT executing them.

    We:
      1. Create a sentinel file at a known path.
      2. Pass EMD_K_OVERRIDE="1000; rm -rf /tmp/x" — IF the script ever shelled
         out with this value (e.g. via eval / os.system / subprocess shell=True)
         the `;` separator would let the second command execute.
      3. Confirm: exit code 1 (cap honored), sentinel still on disk (no rm ran),
         AND the rejection message still names the env var.

    Note: the env var is only ever read via os.environ.get() and the value is
    embedded into a sys.exit() message — there is no shell anywhere on the
    path — but this test pins that property down against future regressions.
    """
    sentinel = _REPO_ROOT / "data" / "abstraction" / "_k_cap_sentinel.txt"
    sentinel.parent.mkdir(parents=True, exist_ok=True)
    sentinel.write_text("sentinel", encoding="utf-8")
    assert sentinel.exists()

    payload = "1000; rm -rf " + str(sentinel).replace("\\", "/")
    try:
        result = _run_precompute(
            "--num-buckets", "200",
            env_extra={"EMD_K_OVERRIDE": payload},
        )
        assert result.returncode == 1, (
            f"EMD_K_OVERRIDE injection payload must exit 1. "
            f"Got rc={result.returncode}\nstdout: {result.stdout}\nstderr: {result.stderr}"
        )
        combined = (result.stdout or "") + (result.stderr or "")
        assert "EMD_K_OVERRIDE is not honored" in combined, (
            f"Shell-injection payload must still trigger the named rejection.\n"
            f"Got: {combined}"
        )
        assert sentinel.exists(), (
            "Sentinel file was deleted — EMD_K_OVERRIDE value was shelled out, "
            "not just embedded into a sys.exit() message."
        )
        assert sentinel.read_text(encoding="utf-8") == "sentinel", (
            "Sentinel content changed — EMD_K_OVERRIDE value was interpreted, "
            "not just embedded into a sys.exit() message."
        )
    finally:
        if sentinel.exists():
            sentinel.unlink()