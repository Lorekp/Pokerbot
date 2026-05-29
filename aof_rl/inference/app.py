"""FastAPI application for AoF policy inference."""

from __future__ import annotations

import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from aof_rl.game_state import AoFEncodeError
from aof_rl.inference.engine import InferenceEngine, InferenceResult, ServerConfig
from aof_rl.inference.logging_config import configure_inference_logging, log_request
from aof_rl.inference.schemas import (
    ActBatchRequest,
    ActBatchResponse,
    ActRequest,
    BatchItemResult,
    ErrorDetail,
    HealthResponse,
    OodDiagnosticsOut,
    PolicyActResponse,
    ReadyResponse,
    TeacherComparisonOut,
)
from aof_rl.policy_v3 import EncoderConfig

logger = logging.getLogger("aof_rl.inference.app")


def _result_to_response(r: InferenceResult) -> PolicyActResponse:
    teacher = None
    if r.teacher is not None:
        teacher = TeacherComparisonOut(
            action=r.teacher["action"],
            q_bb=r.teacher["q_bb"],
            edge_bb=r.teacher["edge_bb"],
            provenance=r.teacher["provenance"],
            approximation_tier=r.teacher["approximation_tier"],
            action_probs=r.teacher.get("action_probs"),
        )
    return PolicyActResponse(
        chosen_action=r.chosen_action,  # type: ignore[arg-type]
        action_logits=r.action_logits,
        q_bb=r.q_bb,
        uncertainty_bb=r.uncertainty_bb,
        state_value_bb=r.state_value_bb,
        action_probs=r.action_probs,
        legal_mask=r.legal_mask,
        obs_version=r.obs_version,
        model_id=r.model_id,
        ood=OodDiagnosticsOut(
            drift_score=r.ood.drift_score,
            teacher_disagreement=r.ood.teacher_disagreement,
            uncertainty_flag=r.ood.uncertainty_flag,
            ensemble_disagreement=r.ood.ensemble_disagreement,
            ood_alert=r.ood.ood_alert,
        ),
        teacher=teacher,
    )


def create_app(engine: InferenceEngine | None = None, config: ServerConfig | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        configure_inference_logging()
        nonlocal engine
        if engine is None:
            if config is None:
                raise RuntimeError("engine or config required")
            engine = InferenceEngine(config)
            engine.load()
        app.state.engine = engine
        yield

    app = FastAPI(title="AoF Inference", version="1.0.0", lifespan=lifespan)

    @app.middleware("http")
    async def request_context(request: Request, call_next):
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        request.state.request_id = request_id
        start = time.perf_counter()
        response: Response = await call_next(request)
        latency_ms = (time.perf_counter() - start) * 1000.0
        response.headers["X-Request-ID"] = request_id
        eng: InferenceEngine | None = getattr(app.state, "engine", None)
        log_request(
            logger,
            request_id=request_id,
            path=request.url.path,
            latency_ms=latency_ms,
            status_code=response.status_code,
            model_id=eng.model_id if eng else None,
        )
        return response

    @app.exception_handler(AoFEncodeError)
    async def encode_error_handler(request: Request, exc: AoFEncodeError):
        code = "unsupported_side_pots" if "side pot" in str(exc).lower() else "encode_error"
        return JSONResponse(
            status_code=422,
            content=ErrorDetail(error=str(exc), error_code=code).model_dump(),
        )

    @app.exception_handler(RequestValidationError)
    async def validation_handler(request: Request, exc: RequestValidationError):
        return JSONResponse(
            status_code=422,
            content=ErrorDetail(
                error="validation failed",
                error_code="validation_error",
                detail=exc.errors(),
            ).model_dump(),
        )

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        eng: InferenceEngine = app.state.engine
        enc = EncoderConfig()
        if eng.meta.get("encoder"):
            from aof_rl.policy_v3 import encoder_config_from_dict

            enc = encoder_config_from_dict(eng.meta["encoder"])
        status = "ok" if eng.ready else "degraded"
        return HealthResponse(
            status=status,
            model_id=eng.model_id,
            obs_version=3,
            device=str(eng.device),
            compile_enabled=eng.compile_enabled,
            amp_enabled=eng.amp_enabled,
            manifest_verified=eng.manifest_verified,
            encoder_type=enc.encoder_type,
            git_commit=eng.meta.get("git_commit"),
        )

    @app.get("/ready", response_model=ReadyResponse)
    def ready() -> ReadyResponse:
        eng: InferenceEngine = app.state.engine
        if not eng.ready:
            raise HTTPException(status_code=503, detail="not ready")
        return ReadyResponse(ready=True)

    @app.post("/v1/act", response_model=PolicyActResponse)
    def act(req: ActRequest) -> PolicyActResponse:
        eng: InferenceEngine = app.state.engine
        if not eng.ready:
            raise HTTPException(status_code=503, detail="not ready")
        game = req.game_state.to_game_state()
        result = eng.predict_one(
            game,
            include_teacher=req.options.include_teacher,
            stochastic=req.options.stochastic,
            teacher_samples=req.options.teacher_samples,
            teacher_seed=req.options.teacher_seed or 0,
        )
        return _result_to_response(result)

    @app.post("/v1/act/batch", response_model=ActBatchResponse)
    def act_batch(req: ActBatchRequest) -> ActBatchResponse:
        eng: InferenceEngine = app.state.engine
        if not eng.ready:
            raise HTTPException(status_code=503, detail="not ready")

        valid_games: list[Any] = []
        valid_indices: list[int] = []
        results: list[BatchItemResult] = [
            BatchItemResult(ok=False, error="pending", error_code="pending")
            for _ in req.items
        ]

        for i, item in enumerate(req.items):
            try:
                game = item.game_state.to_game_state()
                eng.validate_side_pots(game)
                valid_games.append(game)
                valid_indices.append(i)
            except AoFEncodeError as exc:
                code = (
                    "unsupported_side_pots"
                    if "side pot" in str(exc).lower()
                    else "encode_error"
                )
                results[i] = BatchItemResult(
                    ok=False,
                    error=str(exc),
                    error_code=code,
                )
                if req.fail_fast:
                    return ActBatchResponse(results=results)

        if valid_games:
            opts = req.items[valid_indices[0]].options
            try:
                batch_results = eng.predict_batch(
                    valid_games,
                    include_teacher=opts.include_teacher,
                    stochastic=opts.stochastic,
                    teacher_samples=opts.teacher_samples,
                    teacher_seed=opts.teacher_seed or 0,
                )
            except AoFEncodeError as exc:
                if req.fail_fast:
                    raise HTTPException(status_code=422, detail=str(exc)) from exc
                for idx in valid_indices:
                    results[idx] = BatchItemResult(
                        ok=False,
                        error=str(exc),
                        error_code="encode_error",
                    )
                return ActBatchResponse(results=results)

            for idx, res in zip(valid_indices, batch_results):
                results[idx] = BatchItemResult(
                    ok=True,
                    response=_result_to_response(res),
                )

        return ActBatchResponse(results=results)

    return app
