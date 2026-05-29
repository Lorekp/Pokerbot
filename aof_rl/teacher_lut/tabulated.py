"""Runtime tabulated teacher lookup (LUT before NashTeacherCache)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from aof import AdvisorTeacher, _teacher_from_evs
from aof_rl.env import AoFHandState
from aof_rl.spot_diagnostics import classify_decision_spot
from aof_rl.teacher import TeacherResult
from aof_rl.teacher_lut.context import (
    TeacherCtxRow,
    build_ctx_index,
    pack_ctx_from_state,
)
from aof_rl.teacher_lut.schema import (
    LUT_SCHEMA_VERSION,
    TeacherLutConfigMismatch,
    TeacherLutSchema,
    verify_config_hash,
)

V1_COVERED_PATHS = frozenset(
    {
        "static_profile_open",
        "static_profile_clean_call",
        "static_profile_call_behind",
    }
)

_SPOT_TO_KIND = {
    "open": "open",
    "call_clean": "call_clean",
    "call_behind": "call_behind",
}


class TabulatedTeacher:
    def __init__(
        self,
        lut_path: str | Path,
        *,
        teacher_version: int,
        use_fast: bool,
        teacher_samples: int,
        teacher_margin_bb: float,
        teacher_min_samples: int,
        stack_bb: float,
        repo_root: Path | None = None,
    ) -> None:
        self.lut_path = Path(lut_path)
        schema_path = self.lut_path.with_suffix(".schema.json")
        if not schema_path.is_file():
            raise FileNotFoundError(f"Teacher LUT schema missing: {schema_path}")
        if not self.lut_path.is_file():
            raise FileNotFoundError(f"Teacher LUT missing: {self.lut_path}")

        self.schema = TeacherLutSchema.load_json(schema_path)
        if self.schema.lut_schema_version != LUT_SCHEMA_VERSION:
            raise TeacherLutConfigMismatch(
                f"Unsupported lut_schema_version {self.schema.lut_schema_version}"
            )

        verify_config_hash(
            self.schema,
            teacher_version=teacher_version,
            use_fast=use_fast,
            teacher_samples=teacher_samples,
            teacher_margin_bb=teacher_margin_bb,
            teacher_min_samples=teacher_min_samples,
            stack_bb=stack_bb,
            repo_root=repo_root,
        )

        data = np.load(self.lut_path, allow_pickle=True)
        self.q_fold = np.asarray(data["q_fold_bb"], dtype=np.float32)
        self.q_shove = np.asarray(data["q_shove_bb"], dtype=np.float32)
        self.p_fold = np.asarray(data["p_fold"], dtype=np.float32)
        self.edge_bb = np.asarray(data["edge_bb"], dtype=np.float32)

        rows_raw = data["ctx_rows_json"]
        self.rows: list[TeacherCtxRow] = []
        for r in rows_raw:
            if isinstance(r, bytes):
                r = r.decode()
            if isinstance(r, str):
                self.rows.append(TeacherCtxRow.from_dict(json.loads(r)))
            else:
                self.rows.append(TeacherCtxRow.from_dict(dict(r)))
        self.hand_labels: tuple[str, ...] = tuple(
            str(x) for x in data["hand_labels"].tolist()
        )
        self._index = build_ctx_index(self.rows)
        self.covered_paths = frozenset(self.schema.v1_path_set)
        self._miss_logged = False

    def lookup(
        self,
        state: AoFHandState,
        hero: str,
        *,
        path: str,
    ) -> TeacherResult:
        from aof_rl.spot_diagnostics import all_ins_before_hero

        kind = classify_decision_spot(state, hero)
        spot_kind = _SPOT_TO_KIND.get(kind)
        if spot_kind is None:
            raise KeyError(f"LUT does not cover spot kind {kind}")

        prior = all_ins_before_hero(state, hero)
        behind_n = len(state.players_behind(hero)) if state.facing_shove() else 0
        row = pack_ctx_from_state(
            table_size=state.table_size,
            hero=hero,
            spot_kind=spot_kind,  # type: ignore[arg-type]
            prior_ins=prior,
            players_behind=behind_n,
            stack_bb=state.stack_bb,
            teacher_version=self.schema.teacher_version,
        )
        ctx_idx = self._index.get(row.pack_key(), -1)
        if ctx_idx < 0:
            if not self._miss_logged:
                import logging

                logging.getLogger(__name__).warning(
                    "Teacher LUT ctx miss (fallback): %s", row.pack_key()
                )
                self._miss_logged = True
            raise KeyError(f"ctx not in LUT: {row.pack_key()}")

        from aof_rl.encode import hand_label_for

        label = hand_label_for(state, hero)
        hi = self.hand_labels.index(label)
        qf = float(self.q_fold[ctx_idx, hi])
        qs = float(self.q_shove[ctx_idx, hi])
        pf = float(self.p_fold[ctx_idx, hi])
        p_fold = None if np.isnan(pf) else pf
        facing = spot_kind != "open"
        if facing:
            teacher = _teacher_from_evs(qs, qf, p_fold=None, facing=True)
        else:
            teacher = _teacher_from_evs(qs, qf, p_fold=p_fold, facing=False)
        prov = "teacher_lut"
        return TeacherResult(teacher=teacher, path=path, provenance=prov)  # type: ignore[arg-type]

    def try_lookup(
        self,
        state: AoFHandState,
        hero: str,
        *,
        path: str,
    ) -> TeacherResult | None:
        try:
            return self.lookup(state, hero, path=path)
        except KeyError:
            return None
