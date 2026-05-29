#!/usr/bin/env python3
"""Build consolidated optimization research pack (max 20 files)."""

from __future__ import annotations

import argparse
import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = Path.home() / "Downloads" / "research-pack" / "optimization"
MAX_FILES = 20


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _code_block(rel_path: str) -> str:
    path = ROOT / rel_path
    if not path.is_file():
        return f"## `{rel_path}`\n\n**MISSING**\n"
    ext = path.suffix.lstrip(".")
    lang = "rust" if ext == "rs" else ext if ext in ("py", "yaml", "json", "md") else ""
    body = _read(path)
    return (
        f"## `{rel_path}` ({len(body.splitlines())} lines)\n\n"
        f"```{lang}\n{body}\n```\n"
    )


def _strip_per_update(md: str) -> str:
    return re.sub(
        r"\n## Per-update breakdown\n.*?(?=\n## |\Z)",
        "\n\n*(Per-update breakdown omitted in consolidated pack.)*\n",
        md,
        flags=re.DOTALL,
    )


def _profile_section(title: str, rel_path: str) -> str:
    path = ROOT / rel_path
    if not path.is_file():
        return f"# {title}\n\n**MISSING:** `{rel_path}`\n"
    if path.suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        slim = {
            k: data[k]
            for k in (
                "total_wall_seconds",
                "updates",
                "hands",
                "transitions",
                "sec_per_update",
                "spots_attempted",
                "labels_written",
                "sec_per_spot",
                "metadata",
                "counters",
                "buckets",
                "cprofile_top",
            )
            if k in data
        }
        if "per_update" in data:
            slim["per_update_count"] = len(data["per_update"])
            slim["per_update_sample"] = data["per_update"][:5]
        if "per_spot" in data:
            slim["per_spot"] = data["per_spot"]
        body = json.dumps(slim, indent=2)
        return f"# {title}\n\nArtifact: `{rel_path}`\n\n```json\n{body}\n```\n"
    body = _strip_per_update(_read(path))
    return f"# {title}\n\nArtifact: `{rel_path}`\n\n{body}\n"


def _merge_sections(title: str, parts: list[str]) -> str:
    return f"# {title}\n\n" + "\n\n---\n\n".join(parts)


def _cprofile_section(title: str, rel_path: str, *, max_lines: int = 150) -> str:
    path = ROOT / rel_path
    if not path.is_file():
        return f"## {title}\n\n**MISSING:** `{rel_path}`\n"
    lines = _read(path).splitlines()
    body = "\n".join(lines[:max_lines])
    if len(lines) > max_lines:
        body += f"\n\n... truncated {len(lines) - max_lines} lines ..."
    return f"## {title}\n\nArtifact: `{rel_path}`\n\n```\n{body}\n```\n"


CODE_ROLLOUT = [
    "aof_rl/league_trainer.py",
    "aof_rl/rollout.py",
    "aof_rl/batched_rollout.py",
]
CODE_TEACHER_POLICY = [
    "aof_rl/teacher.py",
    "aof_rl/fast_teacher.py",
    "aof_rl/policy_v3.py",
    "aof_rl/ppo_v3.py",
    "aof_rl/teacher_lut/tabulated.py",
]
CODE_ENV_PROF = [
    "aof_rl/env.py",
    "aof_rl/profiling/timer.py",
    "aof_rl/profiling/league_profiler.py",
    "scripts/profile_aof_league_gpu.py",
]
CODE_SOLVER_PY = [
    "scripts/build_solver_labels.py",
    "scripts/profile_solver_labels.py",
    "pokerbot/solver/postflop_solver_client.py",
    "pokerbot/solver/spot_enumerator.py",
    "pokerbot/encoding/postflop_solver_encoder.py",
]
CONFIGS = [
    "configs/aof_v3_ppo_league_profile.yaml",
    "configs/aof_v3_ppo_league.yaml",
    "config/solver_labels_smoke.yaml",
    "config/solver_labels_prod600.yaml",
]
PROFILE_AOF = [
    ("AoF bottlenecks doc", "docs/aof-training-bottlenecks.md"),
    ("AoF profile HU-only (30 upd)", "artifacts/profile/league_gpu_20260528T013815Z.md"),
    ("AoF profile pre-LUT full curriculum", "artifacts/profile/league_gpu_20260528T015849Z.md"),
    ("AoF profile 10 upd + cProfile", "artifacts/profile/league_gpu_20260528T144632Z.md"),
    ("AoF profile LUT+ batched 60 upd", "artifacts/profile/league_gpu_20260528T150728Z.md"),
    ("AoF profile LUT+ batched 220 upd (latest)", "artifacts/profile/league_gpu_20260528T152628Z.md"),
]
PROFILE_SOLVER = [
    ("Solver label profile 1 spot (clean 95s)", "artifacts/profile/solver_labels_20260528T152407Z.md"),
]
PROFILE_JSON = [
    ("AoF latest JSON summary", "logs/aof_rl_profile_latest.json"),
    ("Solver latest JSON summary", "logs/solver_label_profile_latest.json"),
]
CPROFILE = [
    ("AoF cProfile update 0", "artifacts/profile/cprofile_update0_20260528T144632Z.txt"),
    ("Solver cProfile spot 0", "artifacts/profile/cprofile_spot0_20260528T152407Z.txt"),
]


def main() -> None:
    p = argparse.ArgumentParser(description="Build consolidated optimization research pack")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = p.parse_args()
    out: Path = args.out

    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    files: dict[str, str] = {}

    files["README.md"] = "\n".join(
        [
            "# Optimization research pack (consolidated)",
            "",
            f"Generated: {stamp}",
            "",
            "Max **20 files** — full code + profiling for an optimizing research agent.",
            "",
            "## Read order",
            "",
            "1. [00-priorities-and-summary.md](00-priorities-and-summary.md)",
            "2. [01-profile-results-aof.md](01-profile-results-aof.md)",
            "3. [02-profile-results-solver.md](02-profile-results-solver.md)",
            "4. [03-profile-json-summaries.md](03-profile-json-summaries.md)",
            "5. [04-cprofile-dumps.md](04-cprofile-dumps.md)",
            "6. Code: `05`–`09` · Configs: [10-configs.md](10-configs.md)",
            "",
            "| Track | Best sec/unit | Top bottleneck |",
            "|-------|---------------|----------------|",
            "| AoF pre-LUT | 10.86 s/update | hero_teacher ~70% |",
            "| AoF LUT+batched 220 upd | **0.40 s/update** | facing_two ~28%, villain ~23%, payoff ~21% |",
            "| Solver 1 spot | **95.5 s/spot** | Rust CLI subprocess ~99% |",
            "",
            "Regenerate: `python scripts/build_optimization_research_pack.py`",
            "",
        ]
    )

    files["00-priorities-and-summary.md"] = "\n".join(
        [
            "# Priorities and optimization targets",
            "",
            "## AoF PPO league (GPU) — current targets",
            "",
            "1. `teacher.path.static_profile_facing_two_conditional` (~28% wall)",
            "2. `rollout.villain_act` (~23%)",
            "3. `rollout.step_payoff` (~21%)",
            "4. `update.ppo.train` (~13%)",
            "",
            "Shipped: teacher LUT, batched rollout, single hero forward.",
            "",
            "## Postflop-solver labels (CPU)",
            "",
            "- ~95s/spot HU sb_vs_bb (12-thread, uncontended)",
            "- 99%+ in Rust CLI; parallel workers not implemented",
            "",
        ]
    )

    files["01-profile-results-aof.md"] = _merge_sections(
        "AoF profile results",
        [_profile_section(t, r) for t, r in PROFILE_AOF],
    )

    files["02-profile-results-solver.md"] = _merge_sections(
        "Solver profile results",
        [_profile_section(t, r) for t, r in PROFILE_SOLVER],
    )

    files["03-profile-json-summaries.md"] = _merge_sections(
        "Profile JSON summaries",
        [_profile_section(t, r) for t, r in PROFILE_JSON],
    )

    files["04-cprofile-dumps.md"] = _merge_sections(
        "cProfile dumps",
        [_cprofile_section(t, r) for t, r in CPROFILE],
    )

    files["05-code-aof-rollout-trainer.md"] = _merge_sections(
        "AoF rollout and trainer",
        [_code_block(r) for r in CODE_ROLLOUT],
    )

    files["06-code-aof-teacher-policy-ppo.md"] = _merge_sections(
        "AoF teacher, policy, PPO",
        [_code_block(r) for r in CODE_TEACHER_POLICY],
    )

    files["07-code-aof-env-profiling.md"] = _merge_sections(
        "AoF env and profiling",
        [_code_block(r) for r in CODE_ENV_PROF],
    )

    files["08-code-solver-python.md"] = _merge_sections(
        "Solver pipeline (Python)",
        [_code_block(r) for r in CODE_SOLVER_PY],
    )

    files["09-code-solver-rust.md"] = _code_block("external/postflop_solver_cli/src/main.rs")
    files["09-code-solver-rust.md"] = "# Rust postflop_solver_cli\n\n" + files["09-code-solver-rust.md"]

    files["10-configs.md"] = _merge_sections(
        "Training and solver configs",
        [_code_block(r) for r in CONFIGS],
    )

    if len(files) > MAX_FILES - 1:
        raise SystemExit(f"Pack would exceed {MAX_FILES} files ({len(files)} md + manifest)")

    for name, content in sorted(files.items()):
        (out / name).write_text(content, encoding="utf-8")

    manifest = {
        "generated_at": stamp,
        "output_dir": str(out),
        "max_files": MAX_FILES,
        "files": sorted(files.keys()) + ["MANIFEST.json"],
        "total_files": len(files) + 1,
    }
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
