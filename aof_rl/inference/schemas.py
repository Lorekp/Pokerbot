"""Pydantic request/response models for AoF inference API."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from aof_rl.game_state import game_state_from_dict


class SeatStateIn(BaseModel):
    seat_id: int
    role: Literal["first_to_act", "button", "small_blind", "big_blind"]
    is_hero: bool
    state: Literal[
        "pending",
        "folded",
        "all_in",
        "acted_fold",
        "acted_all_in",
    ]
    committed_bb: float = 0.0
    stack_remaining_bb: float = 0.0
    blind_posted_bb: float = 0.0
    shover_rank: int = 0
    eligible_pot_mask: int = 0


class ActionHistoryEntry(BaseModel):
    seat_id: int
    action: Literal["FOLD", "ALL_IN", "ALL IN", "ALLIN"]


class AoFGameStateIn(BaseModel):
    variant: Literal["NLH_AOF"] = "NLH_AOF"
    table_size: int = Field(ge=2, le=4)
    hero_seat_id: int = Field(ge=0)
    hero_cards: list[int | str]
    action_history: list[ActionHistoryEntry] = Field(default_factory=list)
    seats: list[SeatStateIn]
    small_blind_bb: float = 0.5
    big_blind_bb: float = 1.0
    effective_stack_bb: float = Field(gt=0)
    flat_rake_bb: float = 0.0
    pct_rake: float = 0.0
    rake_cap_bb: float = 0.0
    equal_stacks: bool = True
    side_pots_present: bool = False

    @field_validator("hero_cards")
    @classmethod
    def _hero_cards_len(cls, v: list) -> list:
        if len(v) != 2:
            raise ValueError("hero_cards must have length 2")
        return v

    def to_game_state(self):
        return game_state_from_dict(self.model_dump(mode="json"))


class ActOptions(BaseModel):
    include_teacher: bool = False
    stochastic: bool = False
    teacher_samples: int = Field(default=200, ge=50, le=8000)
    teacher_seed: int | None = None


class ActRequest(BaseModel):
    game_state: AoFGameStateIn
    options: ActOptions = Field(default_factory=ActOptions)


class ActBatchRequest(BaseModel):
    items: list[ActRequest]
    fail_fast: bool = False


class OodDiagnosticsOut(BaseModel):
    drift_score: float
    teacher_disagreement: float
    uncertainty_flag: bool
    ensemble_disagreement: float | None = None
    ood_alert: bool


class TeacherComparisonOut(BaseModel):
    action: Literal["FOLD", "ALL_IN"]
    q_bb: list[float]
    edge_bb: float
    provenance: str
    approximation_tier: str
    action_probs: list[float] | None = None


class PolicyActResponse(BaseModel):
    chosen_action: Literal["FOLD", "ALL_IN"]
    action_logits: list[float]
    q_bb: list[float]
    uncertainty_bb: float
    state_value_bb: float
    action_probs: list[float]
    legal_mask: list[float]
    obs_version: int
    model_id: str
    ood: OodDiagnosticsOut
    teacher: TeacherComparisonOut | None = None


class BatchItemResult(BaseModel):
    ok: bool
    response: PolicyActResponse | None = None
    error: str | None = None
    error_code: str | None = None


class ActBatchResponse(BaseModel):
    results: list[BatchItemResult]


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    model_id: str
    obs_version: int
    device: str
    compile_enabled: bool
    amp_enabled: bool
    manifest_verified: bool
    encoder_type: str
    git_commit: str | None = None


class ReadyResponse(BaseModel):
    ready: bool


class ErrorDetail(BaseModel):
    error: str
    error_code: str
    detail: Any | None = None
