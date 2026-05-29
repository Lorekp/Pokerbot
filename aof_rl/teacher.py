"""Static-profile teacher features with per-rollout cache (v1 + v2)."""



from __future__ import annotations



from collections import OrderedDict

from dataclasses import dataclass



from aof import (

    AdvisorTeacher,

    NashTeacher,

    nash_teacher_call,

    nash_teacher_open,

    static_profile_teacher_call,

    static_profile_teacher_open,

)

from aof_parse import CallSpot, OpenSpot

from aof_ranges import lookup_facing_two_ranges

from aof_rl.encode import hand_label_for

from aof_rl.env import AoFHandState

from aof_rl.fast_teacher import nash_teacher_call_fast, nash_teacher_open_fast

from aof_rl.spot_diagnostics import (

    TeacherPath,

    all_ins_before_hero,

    classify_decision_spot,

    teacher_path_for_spot,

)





@dataclass(frozen=True)

class TeacherResult:

    teacher: AdvisorTeacher

    path: TeacherPath

    provenance: str = "unknown"





def _facing_two_teacher_path(

    first: str,

    second: str,

    stack_bb: float,

    table_size: int,

) -> TeacherPath:

    ft = lookup_facing_two_ranges(first, second, stack_bb, table_size=table_size)

    if ft.second_source == "conditional":

        return "static_profile_facing_two_conditional"

    return "static_profile_facing_two_fallback"





def resolve_teacher_path(

    state: AoFHandState,

    hero: str,

    spot: OpenSpot | CallSpot,

    *,

    teacher_version: int,

) -> TeacherPath:

    kind = classify_decision_spot(state, hero)

    if teacher_version < 2:

        return teacher_path_for_spot(kind, teacher_version=teacher_version)



    if kind == "open":

        return "static_profile_open"

    if kind == "call_clean":

        return "static_profile_clean_call"

    if kind == "call_behind":

        return "static_profile_call_behind"

    if kind == "facing_three":
        return "static_profile_facing_three_missing"

    if kind == "facing_two":
        if isinstance(spot, CallSpot) and spot.second_shover:
            return _facing_two_teacher_path(
                spot.shover,
                spot.second_shover,
                spot.stack_bb,
                spot.table_size,
            )
        return "static_profile_unsupported"

    return "unknown"





def build_spot_for_state(

    state: AoFHandState,

    hero: str,

    *,

    teacher_version: int = 1,

) -> OpenSpot | CallSpot:

    label = hand_label_for(state, hero)

    cards = state.holes[hero]

    prior = all_ins_before_hero(state, hero)



    if len(prior) >= 3:
        return CallSpot(
            hand_label=label,
            hero_cards=cards,
            position=hero,
            shover=prior[0],
            stack_bb=state.stack_bb,
            table_size=state.table_size,
            players_behind=0,
            behind_seats=(),
            second_shover=prior[1],
            third_shover=prior[2],
        )

    if len(prior) >= 2 and teacher_version >= 2:
        first, second = prior[0], prior[1]
        behind = state.players_behind(hero)
        return CallSpot(
            hand_label=label,
            hero_cards=cards,
            position=hero,
            shover=first,
            stack_bb=state.stack_bb,
            table_size=state.table_size,
            players_behind=len(behind),
            behind_seats=behind,
            second_shover=second,
        )



    if state.facing_shove() and state.shover:

        behind = state.players_behind(hero)

        return CallSpot(

            hand_label=label,

            hero_cards=cards,

            position=hero,

            shover=state.shover,

            stack_bb=state.stack_bb,

            table_size=state.table_size,

            players_behind=len(behind),

            behind_seats=behind,

            second_shover=None,

        )



    return OpenSpot(

        hand_label=label,

        hero_cards=cards,

        position=hero,

        stack_bb=state.stack_bb,

        table_size=state.table_size,

    )





class NashTeacherCache:

    def __init__(

        self,

        *,

        use_fast: bool = False,

        teacher_margin_bb: float = 0.4,

        teacher_min_samples: int = 200,

        max_entries: int = 8192,

        teacher_version: int = 1,

        teacher_lut_path: str | None = None,

        teacher_samples: int = 300,

        stack_bb: float = 8.0,

    ) -> None:

        self._cache: OrderedDict[tuple, TeacherResult] = OrderedDict()

        self._max_entries = max_entries

        self.use_fast = use_fast

        self.teacher_margin_bb = teacher_margin_bb

        self.teacher_min_samples = teacher_min_samples

        self.teacher_version = teacher_version

        self.teacher_samples = teacher_samples

        self.stack_bb = stack_bb

        self._lut = None

        if teacher_lut_path:

            from aof_rl.teacher_lut.tabulated import TabulatedTeacher

            self._lut = TabulatedTeacher(

                teacher_lut_path,

                teacher_version=teacher_version,

                use_fast=use_fast,

                teacher_samples=teacher_samples,

                teacher_margin_bb=teacher_margin_bb,

                teacher_min_samples=teacher_min_samples,

                stack_bb=stack_bb,

            )



    def clear(self) -> None:

        self._cache.clear()



    def _put(self, key: tuple, result: TeacherResult) -> None:

        if key in self._cache:

            self._cache.move_to_end(key)

        self._cache[key] = result

        while len(self._cache) > self._max_entries:

            self._cache.popitem(last=False)



    def get_with_path(

        self,

        state: AoFHandState,

        hero: str,

        *,

        samples: int,

        seed: int,

    ) -> TeacherResult:

        import time as _time

        from aof_rl.profiling.league_profiler import get_profiler

        prof = get_profiler()
        t0 = _time.perf_counter() if prof is not None else None

        spot = build_spot_for_state(state, hero, teacher_version=self.teacher_version)

        path = resolve_teacher_path(

            state, hero, spot, teacher_version=self.teacher_version

        )

        if self._lut is not None and path in self._lut.covered_paths:
            lut_result = self._lut.try_lookup(state, hero, path=path)
            if lut_result is not None:
                if prof is not None and t0 is not None:
                    prof.add("teacher.path.lut_hit", _time.perf_counter() - t0)
                    prof.inc_counter("teacher.path.calls.lut_hit")
                return lut_result
            if prof is not None:
                prof.inc_counter("teacher.path.calls.lut_ctx_miss")

        prior = all_ins_before_hero(state, hero)

        behind_key = state.players_behind(hero) if state.facing_shove() else ()

        second = prior[1] if len(prior) >= 2 else None
        third = prior[2] if len(prior) >= 3 else None

        key = (
            state.hand_id,
            hero,
            state.shover,
            second,
            third,
            len(behind_key),

            behind_key,

            int(state.stack_bb * 10),

            samples // 100,

            self.use_fast,

            self.teacher_version,

        )

        if key in self._cache:
            if prof is not None:
                with prof.measure("teacher.cache_hit"):
                    self._cache.move_to_end(key)
                    return self._cache[key]
            self._cache.move_to_end(key)
            return self._cache[key]

        if isinstance(spot, CallSpot):
            if path == "static_profile_facing_three_missing":
                from aof_multiway import multiway_call_terms

                terms = multiway_call_terms(
                    spot.position,
                    spot.stack_bb,
                    villain_positions=(
                        spot.shover,
                        spot.second_shover or "",
                        spot.third_shover or "",
                    ),
                    table_size=spot.table_size,
                )
                teacher = AdvisorTeacher(
                    action="FOLD",
                    ev_primary=terms.ev_fold,
                    ev_alt=terms.ev_fold,
                    edge_bb=0.0,
                    p_fold=None,
                    facing=True,
                )
                prov = "static_profile"
            elif self.use_fast:
                teacher = nash_teacher_call_fast(

                    spot,

                    samples=samples,

                    seed=seed,

                    margin_bb=self.teacher_margin_bb,

                    min_samples=self.teacher_min_samples,

                )

            else:

                teacher = static_profile_teacher_call(

                    spot, samples=samples, seed=seed

                )

            if spot.second_shover and path == "static_profile_unsupported":

                path = _facing_two_teacher_path(

                    spot.shover,

                    spot.second_shover,

                    spot.stack_bb,

                    spot.table_size,

                )

            prov = "conditional_call_chart" if "conditional" in path else "static_profile"

        else:

            if self.use_fast:

                teacher = nash_teacher_open_fast(

                    spot,

                    samples=samples,

                    seed=seed,

                    margin_bb=self.teacher_margin_bb,

                    min_samples=self.teacher_min_samples,

                )

            else:

                teacher = static_profile_teacher_open(

                    spot, samples=samples, seed=seed

                )

            prov = "static_profile"



        result = TeacherResult(teacher=teacher, path=path, provenance=prov)

        self._put(key, result)

        if prof is not None and t0 is not None:
            prof.add(f"teacher.path.{path}", _time.perf_counter() - t0)
            prof.inc_counter(f"teacher.path.calls.{path}")

        return result



    def get(

        self,

        state: AoFHandState,

        hero: str,

        *,

        samples: int,

        seed: int,

    ) -> NashTeacher:

        return self.get_with_path(state, hero, samples=samples, seed=seed).teacher


