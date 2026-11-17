"""Servicios de evaluaciones y radiografías para HU05, HU06, HU07 y HU08."""

from __future__ import annotations

import json
import secrets
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from .config import Settings
from .images import ValidatedImage
from .services import record_audit_event


class EvaluationNotFoundError(Exception):
    """La evaluación solicitada no existe."""


class RadiographNotFoundError(Exception):
    """La evaluación no tiene una radiografía asociada."""


class RadiographAlreadyExistsError(Exception):
    """La evaluación ya tiene una radiografía en este bloque."""


class InvalidEvaluationStateError(Exception):
    """El estado actual de la evaluación no permite la operación solicitada."""


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _timestamp(value: datetime) -> str:
    return value.isoformat()


def _evaluation_query() -> str:
    return """
        SELECT
            evaluations.id,
            evaluations.internal_code,
            evaluations.status,
            evaluations.notes,
            evaluations.created_at,
            evaluations.updated_at,
            evaluations.created_by_user_id,
            users.display_name AS created_by_display_name,
            radiographs.id AS radiograph_id,
            radiographs.storage_path,
            radiographs.content_type,
            radiographs.size_bytes,
            radiographs.width,
            radiographs.height,
            radiographs.sha256,
            radiographs.uploaded_at,
            qc.id AS qc_id,
            qc.passed AS qc_passed,
            qc.warnings_json AS qc_warnings_json,
            qc.failures_json AS qc_failures_json,
            qc.metrics_json AS qc_metrics_json,
            qc.checked_by_user_id AS qc_checked_by_user_id,
            qc.checked_at AS qc_checked_at,
            predictions.id AS prediction_id,
            predictions.concordance_state AS prediction_concordance_state,
            predictions.review_required AS prediction_review_required,
            (
                SELECT COUNT(*)
                FROM observations
                WHERE observations.evaluation_id = evaluations.id
            ) AS observation_count
        FROM evaluations
        LEFT JOIN users ON users.id = evaluations.created_by_user_id
        LEFT JOIN radiographs ON radiographs.evaluation_id = evaluations.id
        LEFT JOIN radiograph_quality_checks AS qc ON qc.evaluation_id = evaluations.id
        LEFT JOIN predictions ON predictions.evaluation_id = evaluations.id
    """


def _evaluation_public(
    row: sqlite3.Row,
    *,
    include_prediction_summary: bool = True,
) -> dict[str, object]:
    radiograph = None
    if row["radiograph_id"]:
        radiograph = {
            "id": row["radiograph_id"],
            "evaluation_id": row["id"],
            "filename": Path(row["storage_path"]).name,
            "content_type": row["content_type"],
            "size_bytes": int(row["size_bytes"]),
            "width": int(row["width"]),
            "height": int(row["height"]),
            "sha256": row["sha256"],
            "uploaded_at": row["uploaded_at"],
            "content_url": f"/api/v1/evaluations/{row['id']}/radiograph",
        }
    quality_check = None
    if row["qc_id"]:
        quality_check = {
            "id": row["qc_id"],
            "evaluation_id": row["id"],
            "radiograph_id": row["radiograph_id"],
            "passed": bool(row["qc_passed"]),
            "warnings": json.loads(row["qc_warnings_json"]),
            "failures": json.loads(row["qc_failures_json"]),
            "metrics": json.loads(row["qc_metrics_json"]),
            "checked_by_user_id": row["qc_checked_by_user_id"],
            "checked_at": row["qc_checked_at"],
        }
    has_prediction = bool(row["prediction_id"])
    return {
        "id": row["id"],
        "internal_code": row["internal_code"],
        "status": row["status"],
        "notes": row["notes"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
        "created_by_user_id": row["created_by_user_id"],
        "created_by_display_name": row["created_by_display_name"],
        "radiograph": radiograph,
        "quality_check": quality_check,
        "has_prediction": has_prediction,
        "concordance_state": (
            row["prediction_concordance_state"]
            if include_prediction_summary and has_prediction
            else None
        ),
        "review_required": (
            bool(row["prediction_review_required"])
            if include_prediction_summary and has_prediction
            else None
        ),
        "observation_count": int(row["observation_count"] or 0),
    }


def _generate_internal_code(connection: sqlite3.Connection) -> str:
    date_part = _utc_now().strftime("%Y%m%d")
    for _ in range(20):
        candidate = f"TB-{date_part}-{secrets.token_hex(3).upper()}"
        existing = connection.execute(
            "SELECT 1 FROM evaluations WHERE internal_code = ?",
            (candidate,),
        ).fetchone()
        if existing is None:
            return candidate
    raise RuntimeError("No se pudo generar un código interno único.")


def create_evaluation(
    connection: sqlite3.Connection,
    *,
    notes: str | None,
    actor_user_id: str,
) -> dict[str, object]:
    evaluation_id = str(uuid4())
    internal_code = _generate_internal_code(connection)
    now = _timestamp(_utc_now())
    with connection:
        connection.execute(
            """
            INSERT INTO evaluations(
                id, internal_code, status, notes, created_at, updated_at, created_by_user_id
            ) VALUES (?, ?, 'pendiente_imagen', ?, ?, ?, ?)
            """,
            (evaluation_id, internal_code, notes, now, now, actor_user_id),
        )
        record_audit_event(
            connection,
            "evaluation.created",
            actor_user_id=actor_user_id,
            subject_user_id=actor_user_id,
            metadata={"evaluation_id": evaluation_id, "internal_code": internal_code},
        )
    return get_evaluation(connection, evaluation_id)


def get_evaluation(
    connection: sqlite3.Connection,
    evaluation_id: str,
    *,
    include_prediction_summary: bool = True,
) -> dict[str, object]:
    row = connection.execute(
        _evaluation_query() + " WHERE evaluations.id = ?",
        (evaluation_id,),
    ).fetchone()
    if row is None:
        raise EvaluationNotFoundError
    return _evaluation_public(row, include_prediction_summary=include_prediction_summary)


def list_evaluations(
    connection: sqlite3.Connection,
    *,
    status: str | None = None,
    search: str | None = None,
    review_required: bool | None = None,
    concordance_state: str | None = None,
    include_prediction_summary: bool = True,
) -> list[dict[str, object]]:
    conditions: list[str] = []
    parameters: list[object] = []
    if status:
        conditions.append("evaluations.status = ?")
        parameters.append(status)
    normalized_search = (search or "").strip()
    if normalized_search:
        conditions.append("evaluations.internal_code LIKE ?")
        parameters.append(f"%{normalized_search}%")
    if include_prediction_summary and review_required is not None:
        conditions.append("predictions.review_required = ?")
        parameters.append(1 if review_required else 0)
    if include_prediction_summary and concordance_state:
        conditions.append("predictions.concordance_state = ?")
        parameters.append(concordance_state)
    query = _evaluation_query()
    if conditions:
        query += " WHERE " + " AND ".join(conditions)
    query += " ORDER BY evaluations.created_at DESC"
    rows = connection.execute(query, parameters).fetchall()
    return [
        _evaluation_public(row, include_prediction_summary=include_prediction_summary)
        for row in rows
    ]


def _relative_storage_path(evaluation_id: str, radiograph_id: str, extension: str) -> Path:
    return Path("radiographs") / evaluation_id / f"{radiograph_id}.{extension}"


def _safe_target(settings: Settings, relative_path: Path) -> Path:
    storage_root = settings.storage_root.resolve()
    target = (storage_root / relative_path).resolve()
    if storage_root != target and storage_root not in target.parents:
        raise RuntimeError("La ruta de almacenamiento quedó fuera del directorio permitido.")
    return target


def resolve_storage_path(settings: Settings, storage_path: str) -> Path:
    relative_path = Path(storage_path)
    if relative_path.is_absolute():
        raise RuntimeError("La ruta de almacenamiento debe ser relativa.")
    return _safe_target(settings, relative_path)


def _write_bytes_atomically(target: Path, data: bytes) -> None:
    temporary_path = target.with_name(target.name + ".tmp")
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary_path.write_bytes(data)
        temporary_path.replace(target)
    except OSError:
        temporary_path.unlink(missing_ok=True)
        raise


def _insert_radiograph_record(
    connection: sqlite3.Connection,
    *,
    radiograph_id: str,
    evaluation_id: str,
    relative_path: Path,
    validated: ValidatedImage,
    uploaded_at: str,
    actor_user_id: str,
    audit_event_type: str,
    set_loaded_status: bool,
) -> None:
    connection.execute(
        """
        INSERT INTO radiographs(
            id, evaluation_id, storage_path, content_type, size_bytes,
            width, height, sha256, uploaded_at, uploaded_by_user_id
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            radiograph_id,
            evaluation_id,
            relative_path.as_posix(),
            validated.content_type,
            validated.size_bytes,
            validated.width,
            validated.height,
            validated.sha256,
            uploaded_at,
            actor_user_id,
        ),
    )
    if set_loaded_status:
        connection.execute(
            "UPDATE evaluations SET status = 'imagen_cargada', updated_at = ? WHERE id = ?",
            (uploaded_at, evaluation_id),
        )
    else:
        connection.execute(
            "UPDATE evaluations SET updated_at = ? WHERE id = ?",
            (uploaded_at, evaluation_id),
        )
    record_audit_event(
        connection,
        audit_event_type,
        actor_user_id=actor_user_id,
        subject_user_id=actor_user_id,
        metadata={
            "evaluation_id": evaluation_id,
            "content_type": validated.content_type,
            "size_bytes": validated.size_bytes,
            "width": validated.width,
            "height": validated.height,
            "sha256": validated.sha256,
        },
    )


def attach_radiograph(
    connection: sqlite3.Connection,
    *,
    evaluation_id: str,
    validated: ValidatedImage,
    data: bytes,
    actor_user_id: str,
    settings: Settings,
) -> dict[str, object]:
    evaluation = connection.execute(
        "SELECT id FROM evaluations WHERE id = ?",
        (evaluation_id,),
    ).fetchone()
    if evaluation is None:
        raise EvaluationNotFoundError
    existing = connection.execute(
        "SELECT 1 FROM radiographs WHERE evaluation_id = ?",
        (evaluation_id,),
    ).fetchone()
    if existing is not None:
        raise RadiographAlreadyExistsError

    radiograph_id = str(uuid4())
    relative_path = _relative_storage_path(evaluation_id, radiograph_id, validated.extension)
    target = _safe_target(settings, relative_path)
    _write_bytes_atomically(target, data)

    uploaded_at = _timestamp(_utc_now())
    try:
        with connection:
            _insert_radiograph_record(
                connection,
                radiograph_id=radiograph_id,
                evaluation_id=evaluation_id,
                relative_path=relative_path,
                validated=validated,
                uploaded_at=uploaded_at,
                actor_user_id=actor_user_id,
                audit_event_type="radiograph.uploaded",
                set_loaded_status=True,
            )
    except sqlite3.IntegrityError as error:
        target.unlink(missing_ok=True)
        raise RadiographAlreadyExistsError from error
    except Exception:
        target.unlink(missing_ok=True)
        raise

    evaluation_public = get_evaluation(connection, evaluation_id)
    radiograph_public = evaluation_public["radiograph"]
    if radiograph_public is None:
        raise RuntimeError("No se pudo recuperar la radiografía recién guardada.")
    return radiograph_public


def replace_radiograph(
    connection: sqlite3.Connection,
    *,
    evaluation_id: str,
    validated: ValidatedImage,
    data: bytes,
    actor_user_id: str,
    settings: Settings,
) -> dict[str, object]:
    """Reemplaza la radiografía solo si la evaluación sigue en estado imagen_cargada (pre-análisis)."""
    evaluation = connection.execute(
        "SELECT id, status FROM evaluations WHERE id = ?",
        (evaluation_id,),
    ).fetchone()
    if evaluation is None:
        raise EvaluationNotFoundError
    existing = connection.execute(
        "SELECT id, storage_path FROM radiographs WHERE evaluation_id = ?",
        (evaluation_id,),
    ).fetchone()
    if existing is None:
        raise RadiographNotFoundError
    if evaluation["status"] != "imagen_cargada":
        raise InvalidEvaluationStateError(
            "Solo se puede reemplazar la radiografía antes de marcarla lista para análisis."
        )

    old_target = resolve_storage_path(settings, str(existing["storage_path"]))
    radiograph_id = str(uuid4())
    relative_path = _relative_storage_path(evaluation_id, radiograph_id, validated.extension)
    target = _safe_target(settings, relative_path)
    _write_bytes_atomically(target, data)

    uploaded_at = _timestamp(_utc_now())
    try:
        with connection:
            connection.execute(
                "DELETE FROM radiograph_quality_checks WHERE evaluation_id = ?",
                (evaluation_id,),
            )
            connection.execute(
                "DELETE FROM radiographs WHERE evaluation_id = ?",
                (evaluation_id,),
            )
            _insert_radiograph_record(
                connection,
                radiograph_id=radiograph_id,
                evaluation_id=evaluation_id,
                relative_path=relative_path,
                validated=validated,
                uploaded_at=uploaded_at,
                actor_user_id=actor_user_id,
                audit_event_type="radiograph.replaced",
                set_loaded_status=False,
            )
    except Exception:
        target.unlink(missing_ok=True)
        raise

    if old_target != target:
        old_target.unlink(missing_ok=True)

    evaluation_public = get_evaluation(connection, evaluation_id)
    radiograph_public = evaluation_public["radiograph"]
    if radiograph_public is None:
        raise RuntimeError("No se pudo recuperar la radiografía reemplazada.")
    return radiograph_public


def get_radiograph_record(
    connection: sqlite3.Connection,
    evaluation_id: str,
) -> sqlite3.Row:
    row = connection.execute(
        """
        SELECT id, evaluation_id, storage_path, content_type
        FROM radiographs
        WHERE evaluation_id = ?
        """,
        (evaluation_id,),
    ).fetchone()
    if row is None:
        raise RadiographNotFoundError
    return row
