"""RL spot-type instrumentation (Priority 6A+)."""



from __future__ import annotations



import json

from collections import Counter

from dataclasses import dataclass, field

from typing import Literal



from aof_rl.env import AoFHandState



SpotKind = Literal[
    "open",
    "call_clean",
    "call_behind",
    "facing_two",
    "facing_three",
    "terminal_3way",
    "terminal_4way",
    "other",
]



TeacherPath = Literal[

    "static_profile_open",

    "static_profile_call",

    "static_profile_clean_call",

    "static_profile_call_behind",

    "static_profile_facing_two",

    "static_profile_facing_two_conditional",

    "static_profile_facing_two_fallback",

    "static_profile_facing_two_supported",

    "static_profile_facing_two_missing",
    "static_profile_facing_three_missing",
    "static_profile_facing_three_supported",
    "static_profile_facing_three_fallback",
    "static_profile_unsupported",

    "unknown",

]





@dataclass

class SpotDiagnostics:

    counts: Counter[str] = field(default_factory=Counter)

    teacher_paths: Counter[str] = field(default_factory=Counter)



    def record(self, kind: SpotKind, teacher_path: TeacherPath) -> None:

        self.counts[kind] += 1

        self.teacher_paths[teacher_path] += 1



    def to_dict(self) -> dict:

        return {

            "spot_counts": dict(self.counts),

            "teacher_paths": dict(self.teacher_paths),

        }



    def to_json(self) -> str:

        return json.dumps(self.to_dict(), indent=2, sort_keys=True)





def all_ins_before_hero(state: AoFHandState, hero: str) -> tuple[str, ...]:

    """All-in seats acting before hero (preflop order)."""

    if hero not in state.seats:

        return ()

    idx = state.seats.index(hero)

    prior = state.seats[:idx]

    return tuple(

        s

        for s in prior

        if s in state.all_in and s not in state.folded

    )





def classify_decision_spot(state: AoFHandState, hero: str) -> SpotKind:

    if state.terminal:

        alive = [s for s in state.seats if s not in state.folded]

        if len(alive) == 3:

            return "terminal_3way"

        if len(alive) == 4:

            return "terminal_4way"

        return "other"

    prior = all_ins_before_hero(state, hero)

    if len(prior) >= 3:
        return "facing_three"

    if len(prior) >= 2:
        return "facing_two"

    if state.facing_shove():

        behind = state.players_behind(hero)

        if behind:

            return "call_behind"

        return "call_clean"

    return "open"





def teacher_path_for_spot(

    kind: SpotKind,

    *,

    teacher_version: int = 1,

) -> TeacherPath:

    """Default teacher path label for a spot kind (v1 = legacy missing facing-two)."""

    if kind == "open":

        return "static_profile_open"

    if kind == "call_clean":

        return "static_profile_clean_call"

    if kind == "call_behind":

        return "static_profile_call_behind"

    if kind == "facing_two":
        if teacher_version >= 2:
            return "static_profile_facing_two_supported"
        return "static_profile_facing_two_missing"

    if kind == "facing_three":
        return "static_profile_facing_three_missing"

    return "unknown"





def record_decision(

    diag: SpotDiagnostics,

    state: AoFHandState,

    hero: str,

    *,

    teacher_path: TeacherPath | None = None,

    teacher_version: int = 1,

) -> tuple[SpotKind, TeacherPath]:

    kind = classify_decision_spot(state, hero)

    path = teacher_path or teacher_path_for_spot(kind, teacher_version=teacher_version)

    diag.record(kind, path)

    return kind, path


