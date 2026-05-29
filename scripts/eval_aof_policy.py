#!/usr/bin/env python3
"""Evaluate AoF policy vs Nash-only baseline."""

from __future__ import annotations

import argparse
import random
import sys
from collections import defaultdict
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch

from aof import NashTeacher
from aof_equity import warmup
from aof_ranges import active_seats
from aof_rl.checkpoint import load_policy_from_checkpoint
from aof_rl.encode import encode_obs
from aof_rl.env import ACTION_ALL_IN, ACTION_FOLD, AoFHandState, reset_hand, step_hand
from aof_rl.scripted_opponent import ScriptedProfile, eval_hero_modes
from aof_rl.teacher import NashTeacherCache

_EVAL_PROFILES: dict[str, dict[str, int | bool]] = {
    "default": {"teacher_samples": 2000, "payoff_samples": 4000, "use_fast_teacher": False},
    "fast": {"teacher_samples": 200, "payoff_samples": 400, "use_fast_teacher": True},
}


def _resolve_device(name: str, *, allow_cpu: bool) -> str:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name.startswith("cuda") and not torch.cuda.is_available():
        if not allow_cpu:
            cuda_build = getattr(torch.version, "cuda", None)
            raise SystemExit(
                "CUDA requested but not available. "
                f"(torch {torch.__version__}, cuda={cuda_build}). "
                "Pass --allow-cpu or install a CUDA PyTorch wheel."
            )
        return "cpu"
    if name.startswith("cuda") and torch.cuda.is_available():
        return "cuda"
    return name


def _nash_policy(state: AoFHandState, seat: str, cache: NashTeacherCache, samples: int, seed: int) -> int:
    teacher = cache.get(state, seat, samples=samples, seed=seed)
    return ACTION_ALL_IN if teacher.action == "ALL IN" else ACTION_FOLD


def _trained_policy(
    model,
    device: torch.device,
    state: AoFHandState,
    seat: str,
    cache: NashTeacherCache,
    samples: int,
    seed: int,
    *,
    ablate_teacher: bool = False,
    obs_version: int = 1,
) -> int:
    tr = cache.get_with_path(state, seat, samples=samples, seed=seed)
    teacher = tr.teacher
    if ablate_teacher:
        teacher = NashTeacher(
            action="FOLD",
            ev_primary=0.0,
            ev_alt=0.0,
            edge_bb=0.0,
            p_fold=0.0,
            facing=state.facing_shove(),
        )
        tr_path, prov = "ablated", "unknown"
    else:
        tr_path, prov = tr.path, tr.provenance
    obs = encode_obs(
        state,
        seat,
        teacher,
        version=obs_version,
        teacher_path=tr_path,
        provenance=prov,
    )
    obs_t = torch.from_numpy(obs).unsqueeze(0).to(device)
    with torch.no_grad():
        idx, _ = model.act(obs_t, deterministic=True)
    return int(idx)


def run_eval(
    *,
    mode: str,
    hands: int,
    seed: int,
    table_sizes: tuple[int, ...],
    stack_bb: float,
    teacher_samples: int,
    payoff_samples: int,
    checkpoint: str | None,
    device: str,
    ablate_teacher: bool,
    obs_version: int,
    teacher_version: int,
    use_fast_teacher: bool = False,
) -> dict:
    warmup()
    rng = random.Random(seed)
    cache = NashTeacherCache(
        teacher_version=teacher_version, use_fast=use_fast_teacher
    )
    dev = torch.device(device)
    model = None
    ckpt_meta: dict = {}
    if mode in ("policy", "scripted"):
        if not checkpoint:
            raise ValueError("--checkpoint required for policy/scripted mode")
        model, ckpt_meta = load_policy_from_checkpoint(
            checkpoint,
            dev,
            obs_version=obs_version,
            teacher_version=teacher_version,
        )

    if mode == "scripted":
        assert model is not None
        profile = ScriptedProfile(call_prob=0.55, name="scripted_loose")
        stats = eval_hero_modes(
            model,
            dev,
            profile,
            hands=hands,
            seed=seed,
            teacher_samples=teacher_samples,
            table_size=table_sizes[0] if len(table_sizes) == 1 else 4,
        )
        return {
            "bb_per_100": 100.0 * stats["policy_bb_per_hand"],
            "teacher_bb_per_100": 100.0 * stats["teacher_bb_per_hand"],
            "hands": hands,
            "by_table_size": {},
            "checkpoint_meta": ckpt_meta,
        }

    totals: dict[int, list[float]] = defaultdict(list)
    hero_totals: list[float] = []

    for h in range(hands):
        table_size = rng.choice(table_sizes)
        hero = rng.choice(active_seats(table_size))
        state = reset_hand(rng, table_size=table_size, stack_bb=stack_bb)
        cache.clear()
        guard = 0
        while not state.terminal:
            guard += 1
            seat = state.action_seat
            if seat is None:
                break
            t_seed = seed + h * 1000 + guard
            if mode == "nash":
                action = _nash_policy(state, seat, cache, teacher_samples, t_seed)
            elif mode == "policy":
                assert model is not None
                action = _trained_policy(
                    model,
                    dev,
                    state,
                    seat,
                    cache,
                    teacher_samples,
                    t_seed,
                    ablate_teacher=ablate_teacher,
                    obs_version=obs_version,
                )
            else:
                raise ValueError(mode)
            state = step_hand(
                state,
                action,
                payoff_samples=payoff_samples,
                payoff_seed=t_seed,
            )

        hero_totals.append(state.payoffs.get(hero, 0.0))
        totals[table_size].append(state.payoffs.get(hero, 0.0))

    bb100 = 100.0 * sum(hero_totals) / max(len(hero_totals), 1)
    by_size = {
        str(ts): 100.0 * sum(v) / max(len(v), 1) for ts, v in totals.items()
    }
    return {
        "bb_per_100": bb100,
        "hands": hands,
        "by_table_size": by_size,
        "checkpoint_meta": ckpt_meta,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate AoF RL policy")
    p.add_argument(
        "--mode",
        choices=("nash", "policy", "policy-no-teacher", "scripted"),
        default="policy",
    )
    p.add_argument("--hands", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument(
        "--profile",
        choices=tuple(_EVAL_PROFILES.keys()),
        default="default",
        help="Sample budget preset (fast=validation smoke)",
    )
    p.add_argument("--teacher-samples", type=int, default=None)
    p.add_argument("--payoff-samples", type=int, default=None)
    p.add_argument("--stack-bb", type=float, default=8.0)
    p.add_argument("--obs-version", type=int, choices=(1, 2), default=1)
    p.add_argument("--teacher-version", type=int, choices=(1, 2), default=1)
    p.add_argument("-o", "--out", type=str, default=None, help="Write JSON eval report")
    p.add_argument(
        "--allow-cpu",
        action="store_true",
        help="Fall back to CPU when CUDA is unavailable (default: require CUDA).",
    )
    args = p.parse_args()
    prof = _EVAL_PROFILES[args.profile]
    teacher_samples = (
        args.teacher_samples
        if args.teacher_samples is not None
        else int(prof["teacher_samples"])
    )
    payoff_samples = (
        args.payoff_samples
        if args.payoff_samples is not None
        else int(prof["payoff_samples"])
    )
    use_fast_teacher = bool(prof["use_fast_teacher"])
    device = _resolve_device(args.device, allow_cpu=args.allow_cpu)
    if device == "cuda":
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")

    mode = "policy" if args.mode == "policy-no-teacher" else args.mode
    ablate = args.mode == "policy-no-teacher"
    ckpt = args.checkpoint or "checkpoints/aof_rl/policy_selfplay.pt"
    needs_ckpt = mode in ("policy", "scripted")

    stats = run_eval(
        mode=mode,
        hands=args.hands,
        seed=args.seed,
        table_sizes=(2, 3, 4) if mode != "scripted" else (4,),
        stack_bb=args.stack_bb,
        teacher_samples=teacher_samples,
        payoff_samples=payoff_samples,
        checkpoint=ckpt if needs_ckpt else None,
        device=device,
        ablate_teacher=ablate,
        obs_version=args.obs_version,
        teacher_version=args.teacher_version,
        use_fast_teacher=use_fast_teacher,
    )
    meta = stats.get("checkpoint_meta") or {}
    print(f"Mode: {args.mode}")
    print(
        f"obs_version={args.obs_version} teacher_version={args.teacher_version} "
        f"(checkpoint obs={meta.get('obs_version')} teacher={meta.get('teacher_version')} "
        f"dim={meta.get('obs_dim')})"
    )
    print(f"bb/100 (hero): {stats['bb_per_100']:.2f} over {stats['hands']} hands")
    if "teacher_bb_per_100" in stats:
        print(f"bb/100 (teacher-following): {stats['teacher_bb_per_100']:.2f}")
    for k, v in stats["by_table_size"].items():
        print(f"  table {k}: {v:.2f} bb/100")

    if args.out:
        import json
        from datetime import datetime, timezone

        payload = {
            "schema_version": 1,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "mode": args.mode,
            "obs_version": args.obs_version,
            "teacher_version": args.teacher_version,
            "profile": args.profile,
            "teacher_samples": teacher_samples,
            "payoff_samples": payoff_samples,
            "seed": args.seed,
            "hands": stats["hands"],
            "bb_per_100": stats["bb_per_100"],
            "by_table_size": stats.get("by_table_size", {}),
            "checkpoint_meta": meta,
        }
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
