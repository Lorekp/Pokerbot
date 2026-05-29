#!/usr/bin/env python3
"""Smoke-test production teacher LUT load (config hash + schema)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from aof_rl.config import load_train_config
from aof_rl.teacher_lut.schema import TeacherLutSchema
from aof_rl.teacher_lut.tabulated import TabulatedTeacher


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--config",
        default="configs/aof_v3_ppo_league.yaml",
        help="Train config with use_teacher_lut and teacher_lut_path",
    )
    args = p.parse_args()
    cfg = load_train_config(args.config)
    tcfg = cfg.train
    if not tcfg.use_teacher_lut:
        print("FAIL: use_teacher_lut is false in config")
        sys.exit(1)

    lut_path = Path(tcfg.teacher_lut_path)
    if not lut_path.is_file():
        print(f"FAIL: LUT file missing: {lut_path}")
        sys.exit(1)

    schema_path = lut_path.with_suffix(".schema.json")
    schema = TeacherLutSchema.load_json(schema_path)
    lut = TabulatedTeacher(
        lut_path,
        teacher_version=cfg.teacher_version,
        use_fast=tcfg.use_fast_teacher,
        teacher_samples=tcfg.teacher_samples,
        teacher_margin_bb=tcfg.teacher_margin_bb,
        teacher_min_samples=tcfg.teacher_min_samples,
        stack_bb=tcfg.stack_bb,
        repo_root=_ROOT,
    )
    print(
        f"OK: loaded {lut_path} "
        f"(n_ctx={schema.n_ctx}, n_hand={schema.n_hand}, "
        f"hash={schema.teacher_config_hash[:12]}…)"
    )


if __name__ == "__main__":
    main()
