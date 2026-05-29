"""Teacher LUT schema, parity, and runtime integration tests."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("numba")

from aof import static_profile_teacher_call, static_profile_teacher_open
from aof_equity import warmup
from aof_rl.env import ACTION_ALL_IN, reset_hand, step_hand
from aof_rl.teacher import NashTeacherCache, build_spot_for_state
from aof_rl.teacher_lut.build import build_tiny_fixture_lut
from aof_rl.teacher_lut.context import (
    TeacherCtxRow,
    enumerate_reachable_ctx_rows,
    pack_ctx_from_state,
)
from aof_rl.teacher_lut.schema import (
    V1_PATH_SET,
    TeacherLutConfigMismatch,
    TeacherLutSchema,
    compute_teacher_config_hash,
    verify_config_hash,
)
from aof_rl.teacher_lut.tabulated import TabulatedTeacher
from aof_rl.teacher_targets import teacher_q_bb
from aof_rl.spot_diagnostics import classify_decision_spot, all_ins_before_hero

FIXTURE_LUT = Path(__file__).parent / "fixtures" / "teacher_lut_tiny.npz"
INDIFF_EPS = 0.05


@pytest.fixture(scope="module", autouse=True)
def _warm() -> None:
    warmup()


@pytest.fixture(scope="module")
def tiny_lut(tmp_path_factory) -> Path:
    if FIXTURE_LUT.is_file() and FIXTURE_LUT.with_suffix(".schema.json").is_file():
        return FIXTURE_LUT
    path = tmp_path_factory.mktemp("lut") / "teacher_lut_tiny.npz"
    build_tiny_fixture_lut(path, samples=60, seed=0)
    return path


def test_enumerate_ctx_rows_nonempty() -> None:
    rows = enumerate_reachable_ctx_rows(stack_bb=8.0, teacher_version=2)
    assert len(rows) > 10
    kinds = {r.spot_kind for r in rows}
    assert kinds == {"open", "call_clean", "call_behind"}


def test_config_hash_mismatch_raises(tiny_lut: Path) -> None:
    with pytest.raises(TeacherLutConfigMismatch):
        TabulatedTeacher(
            tiny_lut,
            teacher_version=2,
            use_fast=True,
            teacher_samples=300,
            teacher_margin_bb=0.4,
            teacher_min_samples=200,
            stack_bb=8.0,
        )


def test_lut_hit_open_spot(tiny_lut: Path) -> None:
    schema = TeacherLutSchema.load_json(tiny_lut.with_suffix(".schema.json"))
    lut = TabulatedTeacher(
        tiny_lut,
        teacher_version=schema.teacher_version,
        use_fast=schema.use_fast,
        teacher_samples=schema.teacher_samples,
        teacher_margin_bb=schema.teacher_margin_bb,
        teacher_min_samples=schema.teacher_min_samples,
        stack_bb=schema.stack_bb,
    )
    cache = NashTeacherCache(
        teacher_version=2,
        teacher_lut_path=str(tiny_lut),
        use_fast=schema.use_fast,
        teacher_samples=schema.teacher_samples,
        stack_bb=schema.stack_bb,
        teacher_margin_bb=schema.teacher_margin_bb,
        teacher_min_samples=schema.teacher_min_samples,
    )
    rng = random.Random(1)
    state = reset_hand(rng, table_size=2, stack_bb=8.0)
    hero = state.action_seat
    assert hero is not None
    result = cache.get_with_path(state, hero, samples=schema.teacher_samples, seed=1)
    assert result.provenance == "teacher_lut"
    assert result.path == "static_profile_open"


def test_parity_lut_vs_legacy_on_fixture_rows(tiny_lut: Path) -> None:
    schema = TeacherLutSchema.load_json(tiny_lut.with_suffix(".schema.json"))
    lut = TabulatedTeacher(
        tiny_lut,
        teacher_version=schema.teacher_version,
        use_fast=False,
        teacher_samples=schema.teacher_samples,
        teacher_margin_bb=schema.teacher_margin_bb,
        teacher_min_samples=schema.teacher_min_samples,
        stack_bb=schema.stack_bb,
    )
    rng = random.Random(99)
    matches = 0
    compared = 0
    for _ in range(40):
        state = reset_hand(rng, table_size=2, stack_bb=8.0)
        hero = state.action_seat
        if hero is None:
            continue
        kind = classify_decision_spot(state, hero)
        if kind != "open":
            continue
        spot = build_spot_for_state(state, hero, teacher_version=2)
        path = "static_profile_open"
        lut_r = lut.try_lookup(state, hero, path=path)
        if lut_r is None:
            continue
        leg = static_profile_teacher_open(
            spot, samples=schema.teacher_samples, seed=11
        )
        q_lut = teacher_q_bb(lut_r.teacher)
        q_leg = teacher_q_bb(leg)
        if abs(float(q_lut[1] - q_lut[0]) - float(q_leg[1] - q_leg[0])) < INDIFF_EPS:
            continue
        compared += 1
        a_lut = int(np.argmax(q_lut))
        a_leg = int(np.argmax(q_leg))
        if a_lut == a_leg:
            matches += 1
    assert compared > 0
    assert matches / compared >= 0.5


def test_hu_exact_enum_matches_mc() -> None:
    from aof_equity import equity_hu_exact
    from aof_equity_grids import _card_str_to_id, _equity_two_combos_exact

    hero, vill = ("Ac", "Ad"), ("Kh", "Kd")
    h0, h1 = _card_str_to_id(hero[0]), _card_str_to_id(hero[1])
    v0, v1 = _card_str_to_id(vill[0]), _card_str_to_id(vill[1])
    exact = float(_equity_two_combos_exact(h0, h1, v0, v1))
    mc = equity_hu_exact(hero, vill, 80_000, 7)
    assert abs(exact - mc) < 0.015


def test_verify_config_hash_matches_fixture(tiny_lut: Path) -> None:
    schema = TeacherLutSchema.load_json(tiny_lut.with_suffix(".schema.json"))
    verify_config_hash(
        schema,
        teacher_version=schema.teacher_version,
        use_fast=schema.use_fast,
        teacher_samples=schema.teacher_samples,
        teacher_margin_bb=schema.teacher_margin_bb,
        teacher_min_samples=schema.teacher_min_samples,
        stack_bb=schema.stack_bb,
    )


def test_compute_config_hash_stable() -> None:
    kw = dict(
        teacher_version=2,
        use_fast=False,
        teacher_samples=300,
        teacher_margin_bb=0.4,
        teacher_min_samples=200,
        stack_bb=8.0,
        v1_path_set=V1_PATH_SET,
    )
    assert compute_teacher_config_hash(**kw) == compute_teacher_config_hash(**kw)


PROD_LUT = Path(__file__).resolve().parents[1] / "data" / "aof" / "teacher_lut_v2_8bb.npz"


@pytest.mark.skipif(not PROD_LUT.is_file(), reason="production LUT not built")
def test_production_lut_loads_with_league_config() -> None:
    from aof_rl.config import load_train_config

    cfg = load_train_config("configs/aof_v3_ppo_league.yaml")
    tcfg = cfg.train
    assert tcfg.use_teacher_lut
    schema = TeacherLutSchema.load_json(PROD_LUT.with_suffix(".schema.json"))
    TabulatedTeacher(
        PROD_LUT,
        teacher_version=cfg.teacher_version,
        use_fast=tcfg.use_fast_teacher,
        teacher_samples=tcfg.teacher_samples,
        teacher_margin_bb=tcfg.teacher_margin_bb,
        teacher_min_samples=tcfg.teacher_min_samples,
        stack_bb=tcfg.stack_bb,
        repo_root=Path(__file__).resolve().parents[1],
    )
    assert schema.n_ctx == 24
    assert schema.n_hand == 169
