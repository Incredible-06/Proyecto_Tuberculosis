"""Rutas de calidad (HU07), inferencia congelada (HU09-HU14), versiones (HU22) y observaciones (HU15)."""

from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import FileResponse
from starlette.concurrency import run_in_threadpool

from .clinical import (
    ModelVersionNotConfiguredError,
    PredictionAlreadyExistsError,
    PredictionNotFoundError,
    create_observation,
    get_explainability_artifact,
    get_prediction,
    list_model_versions,
    list_observations,
    persist_analysis_result,
    prepare_analysis_context,
    record_analysis_failure,
    validate_evaluation_quality,
)
from .config import Settings
from .database import connect_database
from .dependencies import require_any_permission, require_permission
from .evaluations import EvaluationNotFoundError, InvalidEvaluationStateError, RadiographNotFoundError
from .inference import BundleIntegrityError, FrozenPhase5Engine, InferenceExecutionError
from .permissions import (
    MODEL_VERSIONS_MANAGE,
    OBSERVATIONS_READ,
    OBSERVATIONS_WRITE,
    PREDICTIONS_READ,
    RADIOGRAPHS_UPLOAD,
    permissions_for_role,
)
from .schemas import (
    CreateObservationRequest,
    EvaluationPublic,
    ModelVersionPublic,
    ObservationPublic,
    PredictionPublic,
    QualityCheckPublic,
)
from .evaluations import get_evaluation


clinical_router = APIRouter(prefix="/api/v1")


def _get_engine(request: Request) -> FrozenPhase5Engine:
    engine = getattr(request.app.state, "inference_engine", None)
    if engine is not None:
        return engine
    settings: Settings = request.app.state.settings
    engine = FrozenPhase5Engine(
        settings.frozen_model_dir,
        device=settings.inference_device,
        low_confidence_threshold=settings.low_confidence_threshold,
    )
    request.app.state.inference_engine = engine
    return engine


@clinical_router.get("/model-versions", response_model=list[ModelVersionPublic])
def list_model_versions_route(
    request: Request,
    current_user: dict[str, str | bool] = Depends(
        require_any_permission(MODEL_VERSIONS_MANAGE, PREDICTIONS_READ)
    ),
) -> list[ModelVersionPublic]:
    settings: Settings = request.app.state.settings
    connection = connect_database(settings.database_path)
    try:
        versions = list_model_versions(connection)
    finally:
        connection.close()
    return [ModelVersionPublic.model_validate(item) for item in versions]


@clinical_router.post(
    "/evaluations/{evaluation_id}/radiograph/validate",
    response_model=QualityCheckPublic,
)
def validate_radiograph_quality_route(
    evaluation_id: str,
    request: Request,
    current_user: dict[str, str | bool] = Depends(require_permission(RADIOGRAPHS_UPLOAD)),
) -> QualityCheckPublic:
    settings: Settings = request.app.state.settings
    connection = connect_database(settings.database_path)
    try:
        result = validate_evaluation_quality(
            connection,
            evaluation_id=evaluation_id,
            actor_user_id=str(current_user["id"]),
            settings=settings,
        )
    except EvaluationNotFoundError as error:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="La evaluación solicitada no existe.",
        ) from error
    except RadiographNotFoundError as error:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="La evaluación no tiene una radiografía cargada.",
        ) from error
    except InvalidEvaluationStateError as error:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(error),
        ) from error
    finally:
        connection.close()
    return QualityCheckPublic.model_validate(result)


@clinical_router.post(
    "/evaluations/{evaluation_id}/analyze",
    response_model=EvaluationPublic,
)
async def analyze_evaluation_route(
    evaluation_id: str,
    request: Request,
    current_user: dict[str, str | bool] = Depends(
        require_any_permission(RADIOGRAPHS_UPLOAD, OBSERVATIONS_WRITE)
    ),
) -> EvaluationPublic:
    settings: Settings = request.app.state.settings
    actor_user_id = str(current_user["id"])

    connection = connect_database(settings.database_path)
    try:
        context = prepare_analysis_context(
            connection,
            evaluation_id=evaluation_id,
            settings=settings,
        )
    except EvaluationNotFoundError as error:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="La evaluación solicitada no existe.",
        ) from error
    except RadiographNotFoundError as error:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="El archivo de la radiografía no está disponible.",
        ) from error
    except PredictionAlreadyExistsError as error:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="La evaluación ya fue analizada.",
        ) from error
    except (InvalidEvaluationStateError, ModelVersionNotConfiguredError) as error:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(error),
        ) from error
    finally:
        connection.close()

    engine = _get_engine(request)
    try:
        output = await run_in_threadpool(engine.analyze_image, context["image_path"])
    except (BundleIntegrityError, InferenceExecutionError, OSError, RuntimeError) as error:
        fail_conn = connect_database(settings.database_path)
        try:
            record_analysis_failure(
                fail_conn,
                evaluation_id=evaluation_id,
                actor_user_id=actor_user_id,
                reason=str(error),
            )
        finally:
            fail_conn.close()
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"No se pudo completar el análisis del modelo congelado: {error}",
        ) from error

    save_conn = connect_database(settings.database_path)
    try:
        persist_analysis_result(
            save_conn,
            evaluation_id=evaluation_id,
            radiograph_id=str(context["radiograph_id"]),
            model_version_id=str(context["model_version_id"]),
            output=output,
            actor_user_id=actor_user_id,
            settings=settings,
        )
        can_read_preds = PREDICTIONS_READ in permissions_for_role(str(current_user["role"]))
        evaluation = get_evaluation(
            save_conn,
            evaluation_id,
            include_prediction_summary=can_read_preds,
        )
    except PredictionAlreadyExistsError as error:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="La evaluación ya fue analizada.",
        ) from error
    finally:
        save_conn.close()

    return EvaluationPublic.model_validate(evaluation)


@clinical_router.get(
    "/evaluations/{evaluation_id}/prediction",
    response_model=PredictionPublic,
)
def get_prediction_route(
    evaluation_id: str,
    request: Request,
    current_user: dict[str, str | bool] = Depends(require_permission(PREDICTIONS_READ)),
) -> PredictionPublic:
    settings: Settings = request.app.state.settings
    connection = connect_database(settings.database_path)
    try:
        prediction = get_prediction(connection, evaluation_id)
    except EvaluationNotFoundError as error:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="La evaluación solicitada no existe.",
        ) from error
    except PredictionNotFoundError as error:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="La evaluación aún no ha sido analizada por el modelo.",
        ) from error
    finally:
        connection.close()
    return PredictionPublic.model_validate(prediction)


@clinical_router.get("/evaluations/{evaluation_id}/prediction/explainability/{kind}")
def download_explainability_route(
    evaluation_id: str,
    kind: Literal["gradcam", "rollout"],
    request: Request,
    current_user: dict[str, str | bool] = Depends(require_permission(PREDICTIONS_READ)),
) -> FileResponse:
    settings: Settings = request.app.state.settings
    connection = connect_database(settings.database_path)
    try:
        target = get_explainability_artifact(
            connection,
            evaluation_id=evaluation_id,
            kind=kind,
            settings=settings,
        )
    except EvaluationNotFoundError as error:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="La evaluación solicitada no existe.",
        ) from error
    except PredictionNotFoundError as error:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="La evaluación aún no cuenta con mapas de explicabilidad.",
        ) from error
    finally:
        connection.close()

    if not target.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="El mapa de explicabilidad no está disponible.",
        )
    return FileResponse(
        target,
        media_type="image/png",
        headers={
            "Cache-Control": "private, no-store",
            "X-Content-Type-Options": "nosniff",
            "Content-Disposition": f'inline; filename="{target.name}"',
        },
    )


@clinical_router.get(
    "/evaluations/{evaluation_id}/observations",
    response_model=list[ObservationPublic],
)
def list_observations_route(
    evaluation_id: str,
    request: Request,
    current_user: dict[str, str | bool] = Depends(require_permission(OBSERVATIONS_READ)),
) -> list[ObservationPublic]:
    settings: Settings = request.app.state.settings
    connection = connect_database(settings.database_path)
    try:
        items = list_observations(connection, evaluation_id)
    except EvaluationNotFoundError as error:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="La evaluación solicitada no existe.",
        ) from error
    finally:
        connection.close()
    return [ObservationPublic.model_validate(item) for item in items]


@clinical_router.post(
    "/evaluations/{evaluation_id}/observations",
    response_model=ObservationPublic,
    status_code=status.HTTP_201_CREATED,
)
def create_observation_route(
    evaluation_id: str,
    payload: CreateObservationRequest,
    request: Request,
    current_user: dict[str, str | bool] = Depends(require_permission(OBSERVATIONS_WRITE)),
) -> ObservationPublic:
    settings: Settings = request.app.state.settings
    connection = connect_database(settings.database_path)
    try:
        item = create_observation(
            connection,
            evaluation_id=evaluation_id,
            clinical_impression=payload.clinical_impression,
            agreement_status=payload.agreement_status.value,
            mark_reviewed=payload.mark_reviewed,
            actor_user_id=str(current_user["id"]),
        )
    except EvaluationNotFoundError as error:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="La evaluación solicitada no existe.",
        ) from error
    except InvalidEvaluationStateError as error:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(error),
        ) from error
    finally:
        connection.close()
    return ObservationPublic.model_validate(item)
