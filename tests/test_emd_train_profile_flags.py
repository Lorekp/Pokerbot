"""Static guards for Step 3 native trainer profiling flag propagation.

These tests deliberately do not launch EC2 or run training. They lock down the control-plane path:
PowerShell babysitter params -> launcher CLI args -> cloud-init env vars consumed by
scripts/emd_pipeline_onbox.sh.
"""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BABYSITTER = ROOT / "scripts" / "emd_pipeline_babysitter_v2.ps1"
LAUNCHER = ROOT / "scripts" / "launch_emd_pipeline_spot.py"
ONBOX = ROOT / "scripts" / "emd_pipeline_onbox.sh"


def test_babysitter_exposes_default_safe_profile_flags() -> None:
    src = BABYSITTER.read_text(encoding="utf-8")
    assert "[int]$TrainProfileEvery = 0" in src
    assert "[switch]$TrainFastMerge" in src
    assert '"--train-profile-every", $TrainProfileEvery' in src
    assert 'if ($TrainFastMerge) { $argsList += "--train-fast-merge" }' in src


def test_launcher_accepts_and_exports_profile_flags() -> None:
    src = LAUNCHER.read_text(encoding="utf-8")
    assert 'p.add_argument("--train-profile-every", type=int, default=0' in src
    assert 'p.add_argument("--train-fast-merge", action="store_true"' in src
    assert 'p.error("--train-profile-every must be >= 0")' in src
    assert 'export TRAIN_PROFILE_EVERY="{int(train_profile_every)}"' in src
    assert 'export TRAIN_FAST_MERGE="{1 if train_fast_merge else 0}"' in src
    assert "train_profile_every=args.train_profile_every" in src
    assert "train_fast_merge=args.train_fast_merge" in src


def test_onbox_consumes_exported_profile_flags() -> None:
    src = ONBOX.read_text(encoding="utf-8")
    assert 'os.environ.get("TRAIN_PROFILE_EVERY", "0")' in src
    assert 'os.environ.get("TRAIN_FAST_MERGE", "0")' in src
    assert "TRAIN_PROFILE_JSON" in src
    assert "TRAIN_FAST_MERGE" in src and "NlheCFRTrainer" in src
