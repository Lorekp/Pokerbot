#!/usr/bin/env python3
"""Train AoF push/fold policy via symmetric self-play + Nash teacher."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch

from aof_rl.checkpoint import load_policy_from_checkpoint
from aof_rl.policy import AoFPolicyNet
from aof_rl.self_play import AoFSelfPlayConfig, train_self_play

_PROFILES: dict[str, dict] = {
    "train": {
        "teacher_samples": 300,
        "payoff_samples": 1200,
        "use_fast_teacher": True,
        "teacher_margin_bb": 0.4,
        "teacher_min_samples": 200,
    },
    "eval": {
        "teacher_samples": 2000,
        "payoff_samples": 4000,
        "use_fast_teacher": False,
        "teacher_margin_bb": 0.0,
        "teacher_min_samples": 200,
    },
    "full": {
        "teacher_samples": 2000,
        "payoff_samples": 4000,
        "use_fast_teacher": False,
        "teacher_margin_bb": 0.0,
        "teacher_min_samples": 200,
    },
    "smoke": {
        "teacher_samples": 80,
        "payoff_samples": 200,
        "use_fast_teacher": True,
        "teacher_margin_bb": 0.4,
        "teacher_min_samples": 50,
    },
}


def _resolve_device(name: str, *, require_cuda: bool) -> str:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if require_cuda and name.startswith("cuda") and not torch.cuda.is_available():
        cuda_build = getattr(torch.version, "cuda", None)
        raise SystemExit(
            "CUDA requested but not available. GPU driver is present but this "
            f"PyTorch build has no CUDA (torch {torch.__version__}, cuda={cuda_build}).\n"
            "Install a CUDA wheel, e.g.:\n"
            "  py -3.13 -m pip install torch --index-url https://download.pytorch.org/whl/cu126\n"
            "Then run with: py -3.13 scripts/train_aof_self_play.py --device cuda ..."
        )
    if name.startswith("cuda") and torch.cuda.is_available():
        return "cuda"
    if name == "cuda":
        return "cpu"
    return name


def main() -> None:
    p = argparse.ArgumentParser(description="AoF RL self-play (Nash teacher + PPO)")
    p.add_argument("--updates", type=int, default=200)
    p.add_argument("--hands-per-update", type=int, default=128)
    p.add_argument("--teacher-samples", type=int, default=None)
    p.add_argument("--payoff-samples", type=int, default=None)
    p.add_argument("--lambda-teacher", type=float, default=0.3)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument(
        "--device",
        default="cuda",
        help="cuda (default), cpu, or auto",
    )
    p.add_argument(
        "--allow-cpu",
        action="store_true",
        help="Use CPU if CUDA is unavailable (default: require CUDA)",
    )
    p.add_argument("--no-compile", action="store_true")
    p.add_argument(
        "--profile",
        choices=tuple(_PROFILES.keys()),
        default="train",
        help="Preset sample counts and fast teacher (default: train)",
    )
    p.add_argument("--checkpoint-dir", default="checkpoints/aof_rl")
    p.add_argument("--checkpoint", type=str, default=None, help="Resume from checkpoint")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--league-size", type=int, default=0)
    p.add_argument("--stack-bb", type=float, default=8.0)
    p.add_argument(
        "--minutes",
        type=float,
        default=None,
        help="Stop after this many minutes (overrides --updates when set)",
    )
    p.add_argument(
        "--parallel",
        action="store_true",
        help="Collect rollouts in parallel worker processes",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Parallel rollout workers (default: cpu_count-2, capped at 6)",
    )
    p.add_argument("--obs-version", type=int, choices=(1, 2), default=1)
    p.add_argument("--teacher-version", type=int, choices=(1, 2), default=1)
    p.add_argument(
        "--out",
        type=str,
        default=None,
        help="Write final checkpoint to this path (also saves under checkpoint-dir).",
    )
    args = p.parse_args()
    if args.profile == "smoke" and args.updates == 200:
        args.updates = 8

    prof = _PROFILES[args.profile]
    require_cuda = not args.allow_cpu
    device = _resolve_device(args.device, require_cuda=require_cuda)
    if device == "cpu":
        os.environ.setdefault("NUMBA_NUM_THREADS", str(max(1, (os.cpu_count() or 4) - 1)))

    model = None
    if args.checkpoint:
        model, ckpt_meta = load_policy_from_checkpoint(
            args.checkpoint,
            device,
            obs_version=args.obs_version,
            teacher_version=args.teacher_version,
        )
        print(
            f"Resume checkpoint obs_version={ckpt_meta.get('obs_version')} "
            f"teacher_version={ckpt_meta.get('teacher_version')} "
            f"obs_dim={ckpt_meta.get('obs_dim')}"
        )

    total_updates = 999_999 if args.minutes else args.updates
    teacher_samples = (
        args.teacher_samples
        if args.teacher_samples is not None
        else prof["teacher_samples"]
    )
    payoff_samples = (
        args.payoff_samples
        if args.payoff_samples is not None
        else prof["payoff_samples"]
    )
    use_compile = device == "cuda" and not args.no_compile
    if use_compile:
        try:
            import triton  # noqa: F401
        except ImportError:
            use_compile = False
            print("triton not installed; torch.compile off")

    rollout_workers = 0
    if args.parallel:
        ncpu = os.cpu_count() or 4
        rollout_workers = args.workers if args.workers is not None else max(1, ncpu - 2)
        rollout_workers = min(rollout_workers, 6)

    ckpt_out = Path(args.out) if args.out else None
    cfg = AoFSelfPlayConfig(
        hands_per_update=args.hands_per_update,
        total_updates=total_updates,
        checkpoint_path=ckpt_out,
        lr=args.lr,
        device=device,
        checkpoint_dir=Path(args.checkpoint_dir),
        seed=args.seed,
        teacher_samples=teacher_samples,
        payoff_samples=payoff_samples,
        lambda_teacher=args.lambda_teacher,
        league_size=args.league_size,
        stack_bb=args.stack_bb,
        max_wall_seconds=args.minutes * 60.0 if args.minutes else None,
        use_fast_teacher=prof["use_fast_teacher"],
        teacher_margin_bb=prof["teacher_margin_bb"],
        teacher_min_samples=prof["teacher_min_samples"],
        torch_compile=use_compile,
        retain_teacher_cache=(args.profile == "train"),
        rollout_workers=rollout_workers,
        obs_version=args.obs_version,
        teacher_version=args.teacher_version,
    )
    if device == "cuda":
        torch.set_float32_matmul_precision("high")
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print(
        f"Profile={args.profile} device={device} teacher_samples={teacher_samples} "
        f"payoff_samples={payoff_samples} fast_teacher={cfg.use_fast_teacher} "
        f"compile={cfg.torch_compile} parallel_workers={rollout_workers} "
        f"obs_version={cfg.obs_version} teacher_version={cfg.teacher_version}"
    )
    train_self_play(model, cfg)
    saved = ckpt_out or (cfg.checkpoint_dir / "policy_selfplay.pt")
    print(f"Saved checkpoint: {saved}")


if __name__ == "__main__":
    main()
