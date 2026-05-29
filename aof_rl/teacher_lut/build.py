"""Offline teacher LUT builder (combo-mean EVs per ctx × 169)."""

from __future__ import annotations

import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np

from aof import static_profile_teacher_call, static_profile_teacher_open
from aof_parse import CallSpot, OpenSpot
from aof_ranges import active_seats
from aof_rl.encode import RANKS, SUITS, _hand_label
from aof_rl.teacher_lut.context import TeacherCtxRow, load_ctx_rows_jsonl
from aof_rl.teacher_lut.schema import (
    LUT_SCHEMA_VERSION,
    TeacherLutSchema,
    V1_PATH_SET,
    compute_teacher_config_hash,
    range_file_hashes,
)
from pokerbot.solver.hu_preflop_dcfr import _build_169_buckets

N_HAND = 169
VALUE_KEYS = ("q_fold_bb", "q_shove_bb", "p_fold", "edge_bb")


def _all_combos_for_label(label: str) -> list[tuple[str, str]]:
    """Enumerate combos for a 169-class label."""
    out: list[tuple[str, str]] = []
    if len(label) == 2 and label[0] == label[1]:
        r = label[0]
        for i, s1 in enumerate(SUITS):
            for s2 in SUITS[i + 1 :]:
                out.append((f"{r}{s1}", f"{r}{s2}"))
        return out
    hi, lo, suf = label[0], label[1], label[2]
    suited = suf == "s"
    if suited:
        for s in SUITS:
            out.append((f"{hi}{s}", f"{lo}{s}"))
    else:
        for s1 in SUITS:
            for s2 in SUITS:
                if s1 != s2:
                    out.append((f"{hi}{s1}", f"{lo}{s2}"))
    return out


def _row_to_spot(
    row: TeacherCtxRow, hero_cards: tuple[str, str], *, stack_bb: float
) -> OpenSpot | CallSpot:
    seats = active_seats(row.table_size)
    hero = seats[row.hero_pos]
    label = _hand_label(hero_cards)
    if row.spot_kind == "open":
        return OpenSpot(
            hand_label=label,
            hero_cards=hero_cards,
            position=hero,
            stack_bb=stack_bb,
            table_size=row.table_size,
        )
    shover = seats[row.shover_a] if row.shover_a < len(seats) else seats[0]
    behind = seats[row.hero_pos + 1 : row.hero_pos + 1 + row.players_behind]
    return CallSpot(
        hand_label=label,
        hero_cards=hero_cards,
        position=hero,
        shover=shover,
        stack_bb=stack_bb,
        table_size=row.table_size,
        players_behind=row.players_behind,
        behind_seats=tuple(behind),
        second_shover=None,
    )


def _teacher_q_for_spot(
    spot: OpenSpot | CallSpot,
    *,
    samples: int,
    seed: int,
    use_fast: bool,
    teacher_margin_bb: float,
    teacher_min_samples: int,
) -> tuple[float, float, float | None, float]:
    if isinstance(spot, OpenSpot):
        if use_fast:
            from aof_rl.fast_teacher import nash_teacher_open_fast

            t = nash_teacher_open_fast(
                spot,
                samples=samples,
                seed=seed,
                margin_bb=teacher_margin_bb,
                min_samples=teacher_min_samples,
            )
        else:
            t = static_profile_teacher_open(spot, samples=samples, seed=seed)
        facing = False
    else:
        if use_fast:
            from aof_rl.fast_teacher import nash_teacher_call_fast

            t = nash_teacher_call_fast(
                spot,
                samples=samples,
                seed=seed,
                margin_bb=teacher_margin_bb,
                min_samples=teacher_min_samples,
            )
        else:
            t = static_profile_teacher_call(spot, samples=samples, seed=seed)
        facing = True

    if t.action == "ALL IN":
        q_shove, q_fold = float(t.ev_primary), float(t.ev_alt)
    else:
        q_fold, q_shove = float(t.ev_primary), float(t.ev_alt)
    edge = abs(q_shove - q_fold)
    return q_fold, q_shove, t.p_fold, edge


def _build_row_hand_slice(
    row: TeacherCtxRow,
    hand_idx: int,
    label: str,
    *,
    stack_bb: float,
    samples: int,
    seed: int,
    use_fast: bool,
    teacher_margin_bb: float,
    teacher_min_samples: int,
) -> tuple[int, float, float, float, float]:
    combos = _all_combos_for_label(label)
    qf_sum = qs_sum = 0.0
    pf_vals: list[float] = []
    n = 0
    for ci, cards in enumerate(combos):
        spot = _row_to_spot(row, cards, stack_bb=stack_bb)
        qf, qs, pf, edge = _teacher_q_for_spot(
            spot,
            samples=samples,
            seed=seed + hand_idx * 1009 + ci * 17,
            use_fast=use_fast,
            teacher_margin_bb=teacher_margin_bb,
            teacher_min_samples=teacher_min_samples,
        )
        qf_sum += qf
        qs_sum += qs
        if pf is not None:
            pf_vals.append(pf)
        n += 1
    if n == 0:
        return hand_idx, 0.0, 0.0, np.nan, 0.0
    p_fold = float(np.mean(pf_vals)) if pf_vals else np.nan
    return hand_idx, qf_sum / n, qs_sum / n, p_fold, abs(qs_sum / n - qf_sum / n)


def build_lut_arrays(
    rows: list[TeacherCtxRow],
    *,
    stack_bb: float = 8.0,
    samples: int = 300,
    seed: int = 42,
    use_fast: bool = False,
    teacher_margin_bb: float = 0.4,
    teacher_min_samples: int = 200,
    max_workers: int = 1,
) -> dict[str, np.ndarray]:
    labels, _ = _build_169_buckets()
    n_ctx = len(rows)
    q_fold = np.zeros((n_ctx, N_HAND), dtype=np.float32)
    q_shove = np.zeros((n_ctx, N_HAND), dtype=np.float32)
    p_fold = np.full((n_ctx, N_HAND), np.nan, dtype=np.float32)
    edge_bb = np.zeros((n_ctx, N_HAND), dtype=np.float32)

    for ctx_i, row in enumerate(rows):
        if max_workers <= 1:
            for hi, lab in enumerate(labels):
                _, qf, qs, pf, ed = _build_row_hand_slice(
                    row,
                    hi,
                    lab,
                    stack_bb=stack_bb,
                    samples=samples,
                    seed=seed + ctx_i * 5003,
                    use_fast=use_fast,
                    teacher_margin_bb=teacher_margin_bb,
                    teacher_min_samples=teacher_min_samples,
                )
                q_fold[ctx_i, hi] = qf
                q_shove[ctx_i, hi] = qs
                p_fold[ctx_i, hi] = pf
                edge_bb[ctx_i, hi] = ed
        else:
            with ProcessPoolExecutor(max_workers=max_workers) as ex:
                futs = [
                    ex.submit(
                        _build_row_hand_slice,
                        row,
                        hi,
                        lab,
                        stack_bb=stack_bb,
                        samples=samples,
                        seed=seed + ctx_i * 5003,
                        use_fast=use_fast,
                        teacher_margin_bb=teacher_margin_bb,
                        teacher_min_samples=teacher_min_samples,
                    )
                    for hi, lab in enumerate(labels)
                ]
                for fut in as_completed(futs):
                    hi, qf, qs, pf, ed = fut.result()
                    q_fold[ctx_i, hi] = qf
                    q_shove[ctx_i, hi] = qs
                    p_fold[ctx_i, hi] = pf
                    edge_bb[ctx_i, hi] = ed

    import json as _json

    return {
        "q_fold_bb": q_fold,
        "q_shove_bb": q_shove,
        "p_fold": p_fold,
        "edge_bb": edge_bb,
        "ctx_rows_json": np.array(
            [_json.dumps(r.to_dict()) for r in rows], dtype=object
        ),
        "hand_labels": np.array(labels),
    }


def _git_commit() -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return out or None
    except Exception:
        return None


def write_teacher_lut(
    path: Path,
    rows: list[TeacherCtxRow],
    arrays: dict[str, np.ndarray],
    *,
    teacher_version: int,
    use_fast: bool,
    teacher_samples: int,
    teacher_margin_bb: float,
    teacher_min_samples: int,
    stack_bb: float,
) -> TeacherLutSchema:
    config_hash = compute_teacher_config_hash(
        teacher_version=teacher_version,
        use_fast=use_fast,
        teacher_samples=teacher_samples,
        teacher_margin_bb=teacher_margin_bb,
        teacher_min_samples=teacher_min_samples,
        stack_bb=stack_bb,
    )
    schema = TeacherLutSchema(
        lut_schema_version=LUT_SCHEMA_VERSION,
        teacher_config_hash=config_hash,
        teacher_version=teacher_version,
        use_fast=use_fast,
        teacher_samples=teacher_samples,
        teacher_margin_bb=teacher_margin_bb,
        teacher_min_samples=teacher_min_samples,
        stack_bb=stack_bb,
        v1_path_set=V1_PATH_SET,
        n_ctx=len(rows),
        build_git_commit=_git_commit(),
        range_file_hashes=range_file_hashes(),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    schema_path = path.with_suffix(".schema.json")
    schema.save_json(schema_path)
    return schema


def build_tiny_fixture_lut(
    path: Path,
    *,
    teacher_version: int = 2,
    stack_bb: float = 8.0,
    samples: int = 80,
    seed: int = 0,
) -> TeacherLutSchema:
    """Small LUT for CI: first open row + first call_clean, 169 hands."""
    from aof_rl.teacher_lut.context import enumerate_reachable_ctx_rows

    rows = enumerate_reachable_ctx_rows(stack_bb=stack_bb, teacher_version=teacher_version)
    tiny_rows = [r for r in rows if r.spot_kind in ("open", "call_clean")][:2]
    if len(tiny_rows) < 2:
        tiny_rows = rows[:2]
    arrays = build_lut_arrays(
        tiny_rows,
        stack_bb=stack_bb,
        samples=samples,
        seed=seed,
        use_fast=False,
        max_workers=1,
    )
    return write_teacher_lut(
        path,
        tiny_rows,
        arrays,
        teacher_version=teacher_version,
        use_fast=False,
        teacher_samples=samples,
        teacher_margin_bb=0.4,
        teacher_min_samples=50,
        stack_bb=stack_bb,
    )


def build_from_ctx_rows_file(
    ctx_rows_path: Path,
    out_path: Path,
    *,
    teacher_version: int = 2,
    stack_bb: float = 8.0,
    samples: int = 300,
    seed: int = 42,
    use_fast: bool = False,
    teacher_margin_bb: float = 0.4,
    teacher_min_samples: int = 200,
    max_workers: int = 4,
) -> TeacherLutSchema:
    rows = load_ctx_rows_jsonl(ctx_rows_path)
    arrays = build_lut_arrays(
        rows,
        stack_bb=stack_bb,
        samples=samples,
        seed=seed,
        use_fast=use_fast,
        teacher_margin_bb=teacher_margin_bb,
        teacher_min_samples=teacher_min_samples,
        max_workers=max_workers,
    )
    return write_teacher_lut(
        out_path,
        rows,
        arrays,
        teacher_version=teacher_version,
        use_fast=use_fast,
        teacher_samples=samples,
        teacher_margin_bb=teacher_margin_bb,
        teacher_min_samples=teacher_min_samples,
        stack_bb=stack_bb,
    )
