"""Rutas de evaluaciones y radiografías para HU05, HU06, HU07 y HU08."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, File, HTTPException, Query, Request, UploadFile, status
from fastapi.responses import FileResponse

from .config import Settings
from .database import connect_database
from .dependencies import require_permission
from .evaluations import (
    EvaluationNotFoundError,
    InvalidEvaluationStateError,
    RadiographAlreadyExistsError,
    RadiographNotFoundError,
    attach_radiograph,
    create_evaluation,
    get_evaluation,
    get_radiograph_record,
    list_evaluations,
    replace_radiograph,
    resolve_storage_path,
)
from .images import ImageValidationError, validate_radiograph
from .permissions import (
    EVALUATIONS_CREATE,
    EVALUATIONS_READ,
    PREDICTIONS_READ,
    RADIOGRAPHS_READ,
    RADIOGRAPHS_UPLOAD,
    permissions_for_role,
)
from .schemas import (
    ConcordanceState,
    CreateEvaluationRequest,
    EvaluationPublic,
    EvaluationStatus,
    RadiographPublic,
)


evaluation_router = APIRouter(prefix="/api/v1")


def _can_read_predictions(user: dict[str, str | bool]) -> bool:
    return PREDICTIONS_READ in permissions_for_role(str(user["role"]))


@evaluation_router.post(
    "/evaluations",
    response_model=EvaluationPublic,
    status_code=status.HTTP_201_CREATED,
)
def create_evaluation_route(
    payload: CreateEvaluationRequest,
    request: Request,
    current_user: dict[str, str | bool] = Depends(require_permission(EVALUATIONS_CREATE)),
) -> EvaluationPublic:
    settings: Settings = request.app.state.settings
    connection = connect_database(settings.database_path)
    try:
        evaluation = create_evaluation(
            connection,
            notes=payload.notes,
            actor_user_id=str(current_user["id"]),
        )
    finally:
        connection.close()
    return EvaluationPublic.model_validate(evaluation)


@evaluation_router.get("/evaluations", response_model=list[EvaluationPublic])
def list_evaluations_route(
    request: Request,
    status_filter: Annotated[EvaluationStatus | None, Query(alias="status")] = None,
    search: Annotated[str | None, Query(max_length=64)] = None,
    review_required: Annotated[bool | None, Query()] = None,
    concordance_state: Annotated[ConcordanceState | None, Query()] = None,
    current_user: dict[str, str | bool] = Depends(require_permission(EVALUATIONS_READ)),
) -> list[EvaluationPublic]:
    settings: Settings = request.app.state.settings
    can_read_preds = _can_read_predictions(current_user)
    connection = connect_database(settings.database_path)
    try:
        evaluations = list_evaluations(
            connection,
            status=status_filter.value if status_filter else None,
            search=search,
            review_required=review_required if can_read_preds else None,
            concordance_state=concordance_state.value if (can_read_preds and concordance_state) else None,
            include_prediction_summary=can_read_preds,
        )
    finally:
        connection.close()
    return [EvaluationPublic.model_validate(evaluation) for evaluation in evaluations]


@evaluation_router.get("/evaluations/{evaluation_id}", response_model=EvaluationPublic)
def get_evaluation_route(
    evaluation_id: str,
    request: Request,
    current_user: dict[str, str | bool] = Depends(require_permission(EVALUATIONS_READ)),
) -> EvaluationPublic:
    settings: Settings = request.app.state.settings
    can_read_preds = _can_read_predictions(current_user)
    connection = connect_database(settings.database_path)
    try:
        evaluation = get_evaluation(
            connection,
            evaluation_id,
            include_prediction_summary=can_read_preds,
        )
    except EvaluationNotFoundError as error:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="La evaluación solicitada no existe.",
        ) from error
    finally:
        connection.close()
    return EvaluationPublic.model_validate(evaluation)


@evaluation_router.post(
    "/evaluations/{evaluation_id}/radiograph",
    response_model=RadiographPublic,
    status_code=status.HTTP_201_CREATED,
)
async def upload_radiograph_route(
    evaluation_id: str,
    request: Request,
    file: UploadFile = File(...),
    current_user: dict[str, str | bool] = Depends(require_permission(RADIOGRAPHS_UPLOAD)),
) -> RadiographPublic:
    settings: Settings = request.app.state.settings
    data = await file.read(settings.max_upload_bytes + 1)
    try:
        validated = validate_radiograph(
            filename=file.filename,
            content_type=file.content_type,
            data=data,
            max_bytes=settings.max_upload_bytes,
        )
    except ImageValidationError as error:
        raise HTTPException(status_code=error.status_code, detail=str(error)) from error

    connection = connect_database(settings.database_path)
    try:
        radiograph = attach_radiograph(
            connection,
            evaluation_id=evaluation_id,
            validated=validated,
            data=data,
            actor_user_id=str(current_user["id"]),
            settings=settings,
        )
    except EvaluationNotFoundError as error:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="La evaluación solicitada no existe.",
        ) from error
    except RadiographAlreadyExistsError as error:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="La evaluación ya tiene una radiografía cargada.",
        ) from error
    finally:
        connection.close()
    return RadiographPublic.model_validate(radiograph)


@evaluation_router.put(
    "/evaluations/{evaluation_id}/radiograph",
    response_model=RadiographPublic,
)
async def replace_radiograph_route(
    evaluation_id: str,
    request: Request,
    file: UploadFile = File(...),
    current_user: dict[str, str | bool] = Depends(require_permission(RADIOGRAPHS_UPLOAD)),
) -> RadiographPublic:
    settings: Settings = request.app.state.settings
    data = await file.read(settings.max_upload_bytes + 1)
    try:
        validated = validate_radiograph(
            filename=file.filename,
            content_type=file.content_type,
            data=data,
            max_bytes=settings.max_upload_bytes,
        )
    except ImageValidationError as error:
        raise HTTPException(status_code=error.status_code, detail=str(error)) from error

    connection = connect_database(settings.database_path)
    try:
        radiograph = replace_radiograph(
            connection,
            evaluation_id=evaluation_id,
            validated=validated,
            data=data,
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
            detail="La evaluación no tiene una radiografía cargada para reemplazar.",
        ) from error
    except InvalidEvaluationStateError as error:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(error),
        ) from error
    finally:
        connection.close()
    return RadiographPublic.model_validate(radiograph)


@evaluation_router.get("/evaluations/{evaluation_id}/radiograph")
def download_radiograph_route(
    evaluation_id: str,
    request: Request,
    current_user: dict[str, str | bool] = Depends(require_permission(RADIOGRAPHS_READ)),
) -> FileResponse:
    settings: Settings = request.app.state.settings
    connection = connect_database(settings.database_path)
    try:
        radiograph = get_radiograph_record(connection, evaluation_id)
    except RadiographNotFoundError as error:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="La evaluación no tiene una radiografía cargada.",
        ) from error
    finally:
        connection.close()

    target = resolve_storage_path(settings, str(radiograph["storage_path"]))
    if not target.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="El archivo de la radiografía no está disponible.",
        )
    stored_filename = Path(str(radiograph["storage_path"])).name
    return FileResponse(
        target,
        media_type=str(radiograph["content_type"]),
        headers={
            "Cache-Control": "private, no-store",
            "X-Content-Type-Options": "nosniff",
            "Content-Disposition": f'inline; filename="{stored_filename}"',
        },
    )
