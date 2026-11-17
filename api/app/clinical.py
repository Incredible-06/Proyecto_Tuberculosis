"""Servicios de versiones de modelo (HU22), calidad (HU07), inferencia (HU09-HU14) y observaciones (HU15)."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from .config import Settings
from .evaluations import (
    EvaluationNotFoundError,
    InvalidEvaluationStateError,
    RadiographNotFoundError,
    _write_bytes_atomically,
    resolve_storage_path,
)
from .images import evaluate_radiograph_quality
from .inference import FrozenPhase5Engine, verify_bundle_files
from .services import record_audit_event


class PredictionNotFoundError(Exception):
    """La evaluación aún no tiene una predicción registrada."""


class PredictionAlreadyExistsError(Exception):
    """La evaluación ya cuenta con una predicción registrada."""


class ModelVersionNotConfiguredError(RuntimeError):
    """No existe ninguna versión activa de modelo registrada."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _model_version_public(row: sqlite3.Row) -> dict[str, object]:
    return {
        "id": row["id"],
        "bundle_schema": row["bundle_schema"],
        "contract_schema": row["contract_schema"],
        "contract_sha256": row["contract_sha256"],
        "data_config_sha256": row["data_config_sha256"],
        "classifier_arch": row["classifier_arch"],
        "classifier_display_name": row["classifier_display_name"],
        "classifier_run_id": row["classifier_run_id"],
        "classifier_sha256": row["classifier_sha256"],
        "classifier_epoch": int(row["classifier_epoch"]),
        "classifier_threshold": float(row["classifier_threshold"]),
        "detector_arch": row["detector_arch"],
        "detector_run_id": row["detector_run_id"],
        "detector_sha256": row["detector_sha256"],
        "detector_threshold": float(row["detector_threshold"]),
        "concordance_rule": row["concordance_rule"],
        "image_size": int(row["image_size"]),
        "is_active": bool(row["is_active"]),
        "registered_at": row["registered_at"],
    }


def sync_frozen_model_version(
    connection: sqlite3.Connection,
    frozen_model_dir: Path,
) -> dict[str, object]:
    """Verifica metadatos del paquete congelado y lo registra de forma idempotente (HU22)."""
    spec = verify_bundle_files(frozen_model_dir, verify_weight_hashes=False)
    manifest = spec.manifest
    contract = spec.contract
    cls_cfg = contract["classifier"]
    det_cfg = contract["detector"]

    existing = connection.execute(
        """
        SELECT * FROM model_versions
        WHERE contract_sha256 = ? AND classifier_sha256 = ? AND detector_sha256 = ?
        """,
        (
            manifest["contract_sha256"],
            manifest["classifier_sha256"],
            manifest["detector_sha256"],
        ),
    ).fetchone()
    if existing is not None:
        if not bool(existing["is_active"]):
            with connection:
                connection.execute("UPDATE model_versions SET is_active = 0")
                connection.execute("UPDATE model_versions SET is_active = 1 WHERE id = ?", (existing["id"],))
            existing = connection.execute(
                "SELECT * FROM model_versions WHERE id = ?",
                (existing["id"],),
            ).fetchone()
        assert existing is not None
        return _model_version_public(existing)

    version_id = str(uuid4())
    now = _utc_now()
    with connection:
        connection.execute("UPDATE model_versions SET is_active = 0")
        connection.execute(
            """
            INSERT INTO model_versions(
                id, bundle_schema, contract_schema, contract_sha256, data_config_sha256,
                classifier_arch, classifier_display_name, classifier_run_id, classifier_sha256,
                classifier_epoch, classifier_threshold,
                detector_arch, detector_run_id, detector_sha256, detector_threshold,
                concordance_rule, image_size, is_active, registered_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?)
            """,
            (
                version_id,
                manifest["schema"],
                contract["schema"],
                manifest["contract_sha256"],
                manifest["data_config_sha256"],
                cls_cfg["architecture"],
                cls_cfg["display_name"],
                cls_cfg["run_id"],
                manifest["classifier_sha256"],
                int(cls_cfg["best_epoch"]),
                float(cls_cfg["operational_tb_threshold"]),
                det_cfg["architecture"],
                det_cfg["run_id"],
                manifest["detector_sha256"],
                float(det_cfg["confidence_threshold"]),
                contract["concordance_rule"]["name"],
                int(spec.data_config["image_size"]),
                now,
            ),
        )
    row = connection.execute(
        "SELECT * FROM model_versions WHERE id = ?",
        (version_id,),
    ).fetchone()
    assert row is not None
    return _model_version_public(row)


def list_model_versions(connection: sqlite3.Connection) -> list[dict[str, object]]:
    rows = connection.execute(
        "SELECT * FROM model_versions ORDER BY is_active DESC, registered_at DESC"
    ).fetchall()
    return [_model_version_public(row) for row in rows]


def _get_active_model_version(connection: sqlite3.Connection) -> sqlite3.Row:
    row = connection.execute(
        "SELECT * FROM model_versions WHERE is_active = 1 ORDER BY registered_at DESC LIMIT 1"
    ).fetchone()
    if row is None:
        raise ModelVersionNotConfiguredError("No hay una versión congelada activa registrada.")
    return row


def validate_evaluation_quality(
    connection: sqlite3.Connection,
    *,
    evaluation_id: str,
    actor_user_id: str,
    settings: Settings,
) -> dict[str, object]:
    """Ejecuta HU07 sobre la radiografía y promueve a lista_para_analisis si aprueba."""
    row = connection.execute(
        """
        SELECT evaluations.status, radiographs.id AS radiograph_id, radiographs.storage_path
        FROM evaluations
        LEFT JOIN radiographs ON radiographs.evaluation_id = evaluations.id
        WHERE evaluations.id = ?
        """,
        (evaluation_id,),
    ).fetchone()
    if row is None:
        raise EvaluationNotFoundError
    if not row["radiograph_id"]:
        raise RadiographNotFoundError
    if row["status"] in {"analizada", "revisada"}:
        raise InvalidEvaluationStateError(
            "La evaluación ya fue analizada y no requiere nueva validación de calidad."
        )

    image_path = resolve_storage_path(settings, str(row["storage_path"]))
    if not image_path.is_file():
        raise RadiographNotFoundError

    result = evaluate_radiograph_quality(
        image_path,
        min_dimension=settings.min_image_dimension,
        recommended_dimension=settings.recommended_image_dimension,
    )
    check_id = str(uuid4())
    checked_at = _utc_now()
    warnings_json = json.dumps(result.warnings, ensure_ascii=False)
    failures_json = json.dumps(result.failures, ensure_ascii=False)
    metrics_json = json.dumps(result.metrics, ensure_ascii=False)
    new_status = "lista_para_analisis" if result.passed else "imagen_cargada"

    with connection:
        connection.execute(
            "DELETE FROM radiograph_quality_checks WHERE evaluation_id = ?",
            (evaluation_id,),
        )
        connection.execute(
            """
            INSERT INTO radiograph_quality_checks(
                id, evaluation_id, radiograph_id, passed, warnings_json, failures_json,
                metrics_json, checked_by_user_id, checked_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                check_id,
                evaluation_id,
                row["radiograph_id"],
                1 if result.passed else 0,
                warnings_json,
                failures_json,
                metrics_json,
                actor_user_id,
                checked_at,
            ),
        )
        connection.execute(
            "UPDATE evaluations SET status = ?, updated_at = ? WHERE id = ?",
            (new_status, checked_at, evaluation_id),
        )
        record_audit_event(
            connection,
            "quality.validated",
            actor_user_id=actor_user_id,
            subject_user_id=actor_user_id,
            metadata={
                "evaluation_id": evaluation_id,
                "passed": result.passed,
                "warning_count": len(result.warnings),
                "failure_count": len(result.failures),
                "new_status": new_status,
            },
        )

    return {
        "id": check_id,
        "evaluation_id": evaluation_id,
        "radiograph_id": row["radiograph_id"],
        "passed": result.passed,
        "warnings": result.warnings,
        "failures": result.failures,
        "metrics": result.metrics,
        "checked_by_user_id": actor_user_id,
        "checked_at": checked_at,
    }


def _explainability_relative_path(evaluation_id: str, prediction_id: str, kind: str) -> Path:
    return Path("explainability") / evaluation_id / f"{prediction_id}_{kind}.png"


def prepare_analysis_context(
    connection: sqlite3.Connection,
    *,
    evaluation_id: str,
    settings: Settings,
) -> dict[str, object]:
    """Valida precondiciones antes de liberar la conexión y ejecutar inferencia fuera de SQLite."""
    row = connection.execute(
        """
        SELECT evaluations.id, evaluations.status, radiographs.id AS radiograph_id, radiographs.storage_path
        FROM evaluations
        LEFT JOIN radiographs ON radiographs.evaluation_id = evaluations.id
        WHERE evaluations.id = ?
        """,
        (evaluation_id,),
    ).fetchone()
    if row is None:
        raise EvaluationNotFoundError
    if not row["radiograph_id"]:
        raise InvalidEvaluationStateError("La evaluación no tiene radiografía cargada.")
    existing_pred = connection.execute(
        "SELECT 1 FROM predictions WHERE evaluation_id = ?",
        (evaluation_id,),
    ).fetchone()
    if existing_pred is not None or row["status"] in {"analizada", "revisada"}:
        raise PredictionAlreadyExistsError
    if row["status"] != "lista_para_analisis":
        raise InvalidEvaluationStateError(
            "Debe aprobar la validación de calidad (HU07) antes de ejecutar el análisis."
        )
    image_path = resolve_storage_path(settings, str(row["storage_path"]))
    if not image_path.is_file():
        raise RadiographNotFoundError
    model_version = _get_active_model_version(connection)
    return {
        "evaluation_id": evaluation_id,
        "radiograph_id": str(row["radiograph_id"]),
        "image_path": image_path,
        "model_version_id": str(model_version["id"]),
    }


def record_analysis_failure(
    connection: sqlite3.Connection,
    *,
    evaluation_id: str,
    actor_user_id: str,
    reason: str,
) -> None:
    with connection:
        record_audit_event(
            connection,
            "prediction.failed",
            actor_user_id=actor_user_id,
            subject_user_id=actor_user_id,
            metadata={"evaluation_id": evaluation_id, "reason": reason[:200]},
        )


def persist_analysis_result(
    connection: sqlite3.Connection,
    *,
    evaluation_id: str,
    radiograph_id: str,
    model_version_id: str,
    output: object,
    actor_user_id: str,
    settings: Settings,
) -> dict[str, object]:
    """Guarda artefactos PNG de explicabilidad y persiste predicción + detecciones atómicamente."""
    from .inference import InferenceOutput

    assert isinstance(output, InferenceOutput)
    prediction_id = str(uuid4())
    gradcam_rel = _explainability_relative_path(evaluation_id, prediction_id, "gradcam")
    rollout_rel = _explainability_relative_path(evaluation_id, prediction_id, "rollout")
    gradcam_abs = resolve_storage_path(settings, gradcam_rel.as_posix())
    rollout_abs = resolve_storage_path(settings, rollout_rel.as_posix())

    _write_bytes_atomically(gradcam_abs, output.gradcam_png)
    try:
        _write_bytes_atomically(rollout_abs, output.rollout_png)
    except Exception:
        gradcam_abs.unlink(missing_ok=True)
        raise

    created_at = _utc_now()
    try:
        with connection:
            current_eval = connection.execute(
                "SELECT status FROM evaluations WHERE id = ?",
                (evaluation_id,),
            ).fetchone()
            if current_eval is None:
                raise EvaluationNotFoundError
            if current_eval["status"] != "lista_para_analisis":
                raise PredictionAlreadyExistsError
            connection.execute(
                """
                INSERT INTO predictions(
                    id, evaluation_id, radiograph_id, model_version_id,
                    predicted_class, p_health, p_sick, p_tb, classifier_alert,
                    detector_box_count, detector_max_conf, detector_alert,
                    concordance_state, review_required, low_confidence,
                    low_confidence_reasons_json, inference_ms, explainability_verified,
                    gradcam_path, rollout_path, created_by_user_id, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    prediction_id,
                    evaluation_id,
                    radiograph_id,
                    model_version_id,
                    output.predicted_class,
                    output.p_health,
                    output.p_sick,
                    output.p_tb,
                    1 if output.classifier_alert else 0,
                    output.detector_box_count,
                    output.detector_max_conf,
                    1 if output.detector_alert else 0,
                    output.concordance_state,
                    1 if output.review_required else 0,
                    1 if output.low_confidence else 0,
                    json.dumps(output.low_confidence_reasons, ensure_ascii=False),
                    output.inference_ms,
                    1 if output.explainability_verified else 0,
                    gradcam_rel.as_posix(),
                    rollout_rel.as_posix(),
                    actor_user_id,
                    created_at,
                ),
            )
            for det in output.detections:
                lx1, ly1, lx2, ly2 = det["letterbox_xyxy"]
                ox1, oy1, ox2, oy2 = det["xyxy"]
                nx1, ny1, nx2, ny2 = det["normalized_xyxy"]
                connection.execute(
                    """
                    INSERT INTO detections(
                        id, prediction_id, box_index, label, confidence,
                        x1, y1, x2, y2, norm_x1, norm_y1, norm_x2, norm_y2,
                        letterbox_x1, letterbox_y1, letterbox_x2, letterbox_y2
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        str(uuid4()),
                        prediction_id,
                        int(det["box_index"]),
                        str(det["label"]),
                        float(det["confidence"]),
                        float(ox1),
                        float(oy1),
                        float(ox2),
                        float(oy2),
                        float(nx1),
                        float(ny1),
                        float(nx2),
                        float(ny2),
                        float(lx1),
                        float(ly1),
                        float(lx2),
                        float(ly2),
                    ),
                )
            connection.execute(
                "UPDATE evaluations SET status = 'analizada', updated_at = ? WHERE id = ?",
                (created_at, evaluation_id),
            )
            record_audit_event(
                connection,
                "prediction.created",
                actor_user_id=actor_user_id,
                subject_user_id=actor_user_id,
                metadata={
                    "evaluation_id": evaluation_id,
                    "prediction_id": prediction_id,
                    "model_version_id": model_version_id,
                    "concordance_state": output.concordance_state,
                    "inference_ms": output.inference_ms,
                },
            )
    except Exception:
        gradcam_abs.unlink(missing_ok=True)
        rollout_abs.unlink(missing_ok=True)
        raise

    return get_prediction(connection, evaluation_id)


def get_prediction(
    connection: sqlite3.Connection,
    evaluation_id: str,
) -> dict[str, object]:
    row = connection.execute(
        """
        SELECT predictions.*
        FROM predictions
        WHERE predictions.evaluation_id = ?
        """,
        (evaluation_id,),
    ).fetchone()
    if row is None:
        exists = connection.execute(
            "SELECT 1 FROM evaluations WHERE id = ?",
            (evaluation_id,),
        ).fetchone()
        if exists is None:
            raise EvaluationNotFoundError
        raise PredictionNotFoundError

    model_row = connection.execute(
        "SELECT * FROM model_versions WHERE id = ?",
        (row["model_version_id"],),
    ).fetchone()
    if model_row is None:
        raise ModelVersionNotConfiguredError("La versión del modelo asociada ya no existe.")

    det_rows = connection.execute(
        """
        SELECT * FROM detections
        WHERE prediction_id = ?
        ORDER BY box_index ASC
        """,
        (row["id"],),
    ).fetchall()
    detections = [
        {
            "id": det["id"],
            "box_index": int(det["box_index"]),
            "label": det["label"],
            "confidence": float(det["confidence"]),
            "xyxy": [float(det["x1"]), float(det["y1"]), float(det["x2"]), float(det["y2"])],
            "normalized_xyxy": [
                float(det["norm_x1"]),
                float(det["norm_y1"]),
                float(det["norm_x2"]),
                float(det["norm_y2"]),
            ],
        }
        for det in det_rows
    ]
    return {
        "id": row["id"],
        "evaluation_id": row["evaluation_id"],
        "radiograph_id": row["radiograph_id"],
        "predicted_class": row["predicted_class"],
        "p_health": float(row["p_health"]),
        "p_sick": float(row["p_sick"]),
        "p_tb": float(row["p_tb"]),
        "classifier_alert": bool(row["classifier_alert"]),
        "detector_box_count": int(row["detector_box_count"]),
        "detector_max_conf": float(row["detector_max_conf"]),
        "detector_alert": bool(row["detector_alert"]),
        "concordance_state": row["concordance_state"],
        "review_required": bool(row["review_required"]),
        "low_confidence": bool(row["low_confidence"]),
        "low_confidence_reasons": json.loads(row["low_confidence_reasons_json"]),
        "inference_ms": float(row["inference_ms"]),
        "explainability_verified": bool(row["explainability_verified"]),
        "gradcam_url": f"/api/v1/evaluations/{row['evaluation_id']}/prediction/explainability/gradcam",
        "rollout_url": f"/api/v1/evaluations/{row['evaluation_id']}/prediction/explainability/rollout",
        "detections": detections,
        "model_version": _model_version_public(model_row),
        "created_by_user_id": row["created_by_user_id"],
        "created_at": row["created_at"],
    }


def get_explainability_artifact(
    connection: sqlite3.Connection,
    *,
    evaluation_id: str,
    kind: str,
    settings: Settings,
) -> Path:
    if kind not in {"gradcam", "rollout"}:
        raise ValueError("Tipo de mapa inválido.")
    column = "gradcam_path" if kind == "gradcam" else "rollout_path"
    row = connection.execute(
        f"SELECT {column} AS artifact_path FROM predictions WHERE evaluation_id = ?",
        (evaluation_id,),
    ).fetchone()
    if row is None:
        exists = connection.execute("SELECT 1 FROM evaluations WHERE id = ?", (evaluation_id,)).fetchone()
        if exists is None:
            raise EvaluationNotFoundError
        raise PredictionNotFoundError
    return resolve_storage_path(settings, str(row["artifact_path"]))


def list_observations(
    connection: sqlite3.Connection,
    evaluation_id: str,
) -> list[dict[str, object]]:
    exists = connection.execute("SELECT 1 FROM evaluations WHERE id = ?", (evaluation_id,)).fetchone()
    if exists is None:
        raise EvaluationNotFoundError
    rows = connection.execute(
        """
        SELECT
            observations.id,
            observations.evaluation_id,
            observations.prediction_id,
            observations.author_user_id,
            users.display_name AS author_display_name,
            observations.clinical_impression,
            observations.agreement_status,
            observations.created_at
        FROM observations
        LEFT JOIN users ON users.id = observations.author_user_id
        WHERE observations.evaluation_id = ?
        ORDER BY observations.created_at DESC
        """,
        (evaluation_id,),
    ).fetchall()
    return [
        {
            "id": row["id"],
            "evaluation_id": row["evaluation_id"],
            "prediction_id": row["prediction_id"],
            "author_user_id": row["author_user_id"],
            "author_display_name": row["author_display_name"],
            "clinical_impression": row["clinical_impression"],
            "agreement_status": row["agreement_status"],
            "created_at": row["created_at"],
        }
        for row in rows
    ]


def create_observation(
    connection: sqlite3.Connection,
    *,
    evaluation_id: str,
    clinical_impression: str,
    agreement_status: str,
    mark_reviewed: bool,
    actor_user_id: str,
) -> dict[str, object]:
    eval_row = connection.execute(
        """
        SELECT evaluations.status, predictions.id AS prediction_id
        FROM evaluations
        LEFT JOIN predictions ON predictions.evaluation_id = evaluations.id
        WHERE evaluations.id = ?
        """,
        (evaluation_id,),
    ).fetchone()
    if eval_row is None:
        raise EvaluationNotFoundError
    if eval_row["status"] not in {"analizada", "revisada"}:
        raise InvalidEvaluationStateError(
            "Solo se pueden registrar observaciones clínicas después de ejecutar el análisis."
        )

    observation_id = str(uuid4())
    created_at = _utc_now()
    transition_to_reviewed = bool(mark_reviewed and eval_row["status"] == "analizada")
    with connection:
        connection.execute(
            """
            INSERT INTO observations(
                id, evaluation_id, prediction_id, author_user_id,
                clinical_impression, agreement_status, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                observation_id,
                evaluation_id,
                eval_row["prediction_id"],
                actor_user_id,
                clinical_impression,
                agreement_status,
                created_at,
            ),
        )
        if transition_to_reviewed:
            connection.execute(
                "UPDATE evaluations SET status = 'revisada', updated_at = ? WHERE id = ?",
                (created_at, evaluation_id),
            )
            record_audit_event(
                connection,
                "evaluation.reviewed",
                actor_user_id=actor_user_id,
                subject_user_id=actor_user_id,
                metadata={"evaluation_id": evaluation_id},
            )
        record_audit_event(
            connection,
            "observation.created",
            actor_user_id=actor_user_id,
            subject_user_id=actor_user_id,
            metadata={
                "evaluation_id": evaluation_id,
                "observation_id": observation_id,
                "agreement_status": agreement_status,
            },
        )

    created = connection.execute(
        """
        SELECT
            observations.id,
            observations.evaluation_id,
            observations.prediction_id,
            observations.author_user_id,
            users.display_name AS author_display_name,
            observations.clinical_impression,
            observations.agreement_status,
            observations.created_at
        FROM observations
        LEFT JOIN users ON users.id = observations.author_user_id
        WHERE observations.id = ?
        """,
        (observation_id,),
    ).fetchone()
    assert created is not None
    return {
        "id": created["id"],
        "evaluation_id": created["evaluation_id"],
        "prediction_id": created["prediction_id"],
        "author_user_id": created["author_user_id"],
        "author_display_name": created["author_display_name"],
        "clinical_impression": created["clinical_impression"],
        "agreement_status": created["agreement_status"],
        "created_at": created["created_at"],
    }
