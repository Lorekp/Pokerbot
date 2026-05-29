"""Teacher LUT context keys (census-validated reachable rows)."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from aof_ranges import _depth_key, active_seats

SpotKindLut = Literal["open", "call_clean", "call_behind"]
SEAT_NONE = 255
N_HAND = 169

V1_SPOT_KINDS: tuple[SpotKindLut, ...] = ("open", "call_clean", "call_behind")


@dataclass(frozen=True)
class TeacherCtxRow:
    table_size: int
    hero_pos: int
    spot_kind: SpotKindLut
    n_shovers_before: int
    shover_a: int
    shover_b: int
    players_behind: int
    stack_bucket: int
    teacher_version: int

    def pack_key(self) -> tuple:
        return (
            self.table_size,
            self.hero_pos,
            self.spot_kind,
            self.n_shovers_before,
            self.shover_a,
            self.shover_b,
            self.players_behind,
            self.stack_bucket,
            self.teacher_version,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> TeacherCtxRow:
        return cls(
            table_size=int(d["table_size"]),
            hero_pos=int(d["hero_pos"]),
            spot_kind=d["spot_kind"],  # type: ignore[arg-type]
            n_shovers_before=int(d["n_shovers_before"]),
            shover_a=int(d["shover_a"]),
            shover_b=int(d["shover_b"]),
            players_behind=int(d["players_behind"]),
            stack_bucket=int(d["stack_bucket"]),
            teacher_version=int(d["teacher_version"]),
        )


def _seat_index(seats: tuple[str, ...], label: str) -> int:
    return seats.index(label)


def enumerate_reachable_ctx_rows(
    *,
    stack_bb: float = 8.0,
    teacher_version: int = 2,
    table_sizes: tuple[int, ...] = (2, 3, 4),
) -> list[TeacherCtxRow]:
    """
    Constrained enumeration of contexts that can occur under v2 teacher paths.

    Excludes facing_two/three and invalid call_behind with players_behind=0.
    """
    depth = _depth_key(stack_bb)
    rows: list[TeacherCtxRow] = []
    seen: set[tuple] = set()

    def add(row: TeacherCtxRow) -> None:
        k = row.pack_key()
        if k not in seen:
            seen.add(k)
            rows.append(row)

    for table_size in table_sizes:
        seats = active_seats(table_size)
        n = len(seats)
        for hero_i in range(n):
            hero = seats[hero_i]
            prior = seats[:hero_i]
            behind_all = seats[hero_i + 1 :]

            add(
                TeacherCtxRow(
                    table_size=table_size,
                    hero_pos=hero_i,
                    spot_kind="open",
                    n_shovers_before=0,
                    shover_a=SEAT_NONE,
                    shover_b=SEAT_NONE,
                    players_behind=0,
                    stack_bucket=depth,
                    teacher_version=teacher_version,
                )
            )

            if not prior:
                continue

            for si, shover in enumerate(prior):
                sh_idx = _seat_index(seats, shover)
                max_behind = len(behind_all)
                for n_behind in range(0, max_behind + 1):
                    if n_behind == 0:
                        add(
                            TeacherCtxRow(
                                table_size=table_size,
                                hero_pos=hero_i,
                                spot_kind="call_clean",
                                n_shovers_before=1,
                                shover_a=sh_idx,
                                shover_b=SEAT_NONE,
                                players_behind=0,
                                stack_bucket=depth,
                                teacher_version=teacher_version,
                            )
                        )
                    else:
                        add(
                            TeacherCtxRow(
                                table_size=table_size,
                                hero_pos=hero_i,
                                spot_kind="call_behind",
                                n_shovers_before=1,
                                shover_a=sh_idx,
                                shover_b=SEAT_NONE,
                                players_behind=n_behind,
                                stack_bucket=depth,
                                teacher_version=teacher_version,
                            )
                        )

    rows.sort(key=lambda r: r.pack_key())
    return rows


def ctx_row_to_teacher_path(row: TeacherCtxRow) -> str:
    if row.spot_kind == "open":
        return "static_profile_open"
    if row.spot_kind == "call_clean":
        return "static_profile_clean_call"
    return "static_profile_call_behind"


def save_ctx_rows_jsonl(rows: list[TeacherCtxRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row.to_dict(), sort_keys=True) + "\n")


def load_ctx_rows_jsonl(path: Path) -> list[TeacherCtxRow]:
    rows: list[TeacherCtxRow] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(TeacherCtxRow.from_dict(json.loads(line)))
    return rows


def build_ctx_index(rows: list[TeacherCtxRow]) -> dict[tuple, int]:
    return {row.pack_key(): i for i, row in enumerate(rows)}


def pack_ctx_from_state(
    *,
    table_size: int,
    hero: str,
    spot_kind: SpotKindLut,
    prior_ins: tuple[str, ...],
    players_behind: int,
    stack_bb: float,
    teacher_version: int,
) -> TeacherCtxRow:
    seats = active_seats(table_size)
    hero_i = seats.index(hero)
    depth = _depth_key(stack_bb)
    sh_a = SEAT_NONE
    sh_b = SEAT_NONE
    n_prior = len(prior_ins)
    if n_prior >= 1:
        sh_a = seats.index(prior_ins[0])
    if n_prior >= 2:
        sh_b = seats.index(prior_ins[1])
    return TeacherCtxRow(
        table_size=table_size,
        hero_pos=hero_i,
        spot_kind=spot_kind,
        n_shovers_before=n_prior,
        shover_a=sh_a,
        shover_b=sh_b,
        players_behind=players_behind,
        stack_bucket=depth,
        teacher_version=teacher_version,
    )
