from __future__ import annotations

import asyncio
import io
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi import HTTPException
from fastapi.responses import FileResponse
from PIL import Image
from pydantic import ValidationError

from app.clinical import sync_frozen_model_version
from app.clinical_routes import (
    analyze_evaluation_route,
    create_observation_route,
    download_explainability_route,
    get_prediction_route,
    list_model_versions_route,
    list_observations_route,
    validate_radiograph_quality_route,
)
from app.database import connect_database
from app.dependencies import require_any_permission, require_permission
from app.permissions import (
    MODEL_VERSIONS_MANAGE,
    OBSERVATIONS_READ,
    OBSERVATIONS_WRITE,
    PREDICTIONS_READ,
    RADIOGRAPHS_UPLOAD,
)
from app.evaluation_routes import (
    create_evaluation_route,
    get_evaluation_route,
    list_evaluations_route,
    replace_radiograph_route,
    upload_radiograph_route,
)
from app.inference import (
    InferenceOutput,
    LetterboxGeometry,
    compute_concordance,
    evaluate_low_confidence,
    project_letterbox_box,
)
from app.schemas import (
    ConcordanceState,
    CreateEvaluationRequest,
    CreateObservationRequest,
    EvaluationStatus,
    ObservationAgreement,
)

from .conftest import RequestStub


class UploadFileStub:
    def __init__(self, filename: str, content_type: str, data: bytes) -> None:
        self.filename = filename
        self.content_type = content_type
        self._data = data

    async def read(self, size: int = -1) -> bytes:
        if size is None or size < 0:
            return self._data
        return self._data[:size]


def _make_png_bytes(width: int, height: int, *, flat: bool = False) -> bytes:
    if flat:
        arr = np.full((height, width), 120, dtype=np.uint8)
    else:
        rng = np.random.default_rng(20261116)
        grad = np.linspace(20, 220, width, dtype=np.float32)[None, :]
        noise = rng.normal(0.0, 22.0, size=(height, width))
        arr = np.clip(grad + noise, 0, 255).astype(np.uint8)
    im = Image.fromarray(arr, mode="L").convert("RGB")
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return buf.getvalue()


class _FakeEngine:
    def __init__(self, output: InferenceOutput) -> None:
        self._output = output

    def analyze_image(self, image_path: Path) -> InferenceOutput:
        assert image_path.is_file()
        return self._output


def _fake_output(
    *,
    predicted_class: str = "TB",
    p_health: float = 0.05,
    p_sick: float = 0.15,
    p_tb: float = 0.80,
    concordance_state: str = "both_alert",
    boxes: list[dict[str, object]] | None = None,
) -> InferenceOutput:
    tiny_png = _make_png_bytes(64, 64)
    det_list = boxes if boxes is not None else [
        {
            "box_index": 0,
            "label": "tb_lesion",
            "confidence": 0.67,
            "xyxy": [120.0, 90.0, 230.0, 210.0],
            "normalized_xyxy": [0.234375, 0.175781, 0.449219, 0.410156],
            "letterbox_xyxy": [120.0, 90.0, 230.0, 210.0],
        }
    ]
    return InferenceOutput(
        predicted_class=predicted_class,
        p_health=p_health,
        p_sick=p_sick,
        p_tb=p_tb,
        classifier_alert=p_tb >= 0.30,
        detector_box_count=len(det_list),
        detector_max_conf=max((float(b["confidence"]) for b in det_list), default=0.0),
        detector_alert=any(float(b["confidence"]) >= 0.25 for b in det_list),
        concordance_state=concordance_state,
        review_required=concordance_state != "neither_alert",
        low_confidence=False,
        low_confidence_reasons=[],
        inference_ms=145.5,
        explainability_verified=True,
        detections=det_list,
        gradcam_png=tiny_png,
        rollout_png=tiny_png,
    )


def _create_ready_evaluation(request_stub: RequestStub, technologist, *, width: int = 512, height: int = 512) -> str:
    ev = create_evaluation_route(CreateEvaluationRequest(notes="Prueba flujo"), request_stub, technologist)
    png = _make_png_bytes(width, height)
    asyncio.run(
        upload_radiograph_route(
            ev.id,
            request_stub,
            UploadFileStub("rx.png", "image/png", png),
            technologist,
        )
    )
    qc = validate_radiograph_quality_route(ev.id, request_stub, technologist)
    assert qc.passed is True
    return ev.id


def test_concordance_rule_4_states_and_no_suppression():
    both = compute_concordance(0.45, [{"confidence": 0.30}])
    assert both == {
        "classifier_alert": True,
        "detector_alert": True,
        "review_required": True,
        "state": "both_alert",
    }

    cls_only = compute_concordance(0.30, [{"confidence": 0.24}])
    assert cls_only == {
        "classifier_alert": True,
        "detector_alert": False,
        "review_required": True,
        "state": "classifier_only_review",
    }

    det_only = compute_concordance(0.29, [{"confidence": 0.25}])
    assert det_only == {
        "classifier_alert": False,
        "detector_alert": True,
        "review_required": True,
        "state": "detector_only_review",
    }

    neither = compute_concordance(0.10, [])
    assert neither == {
        "classifier_alert": False,
        "detector_alert": False,
        "review_required": False,
        "state": "neither_alert",
    }


def test_low_confidence_and_letterbox_box_projection():
    low, reasons = evaluate_low_confidence(
        (0.40, 0.30, 0.30),
        concordance_state="classifier_only_review",
        low_confidence_threshold=0.70,
        operational_tb_threshold=0.30,
    )
    assert low is True
    assert len(reasons) == 3

    geom = LetterboxGeometry(
        orig_width=800,
        orig_height=600,
        target_size=512,
        resized_width=512,
        resized_height=384,
        pad_x=0,
        pad_y=64,
    )
    orig_xyxy, norm_xyxy = project_letterbox_box([128.0, 64.0, 256.0, 256.0], geom)
    assert orig_xyxy == [200.0, 0.0, 400.0, 300.0]
    assert norm_xyxy == [0.25, 0.0, 0.5, 0.5]


def test_frozen_model_version_is_seeded_and_permission_gated(
    request_stub, administrator, radiologist, technologist
):
    versions_admin = list_model_versions_route(request_stub, administrator)
    versions_doc = list_model_versions_route(request_stub, radiologist)
    assert len(versions_admin) == 1
    assert versions_admin[0].classifier_arch == "vit_b_16"
    assert versions_admin[0].detector_arch == "yolo11n"
    assert versions_admin[0].classifier_threshold == 0.3
    assert versions_admin[0].detector_threshold == 0.25
    assert versions_admin[0].is_active is True
    assert versions_doc[0].id == versions_admin[0].id


def test_hu07_quality_validation_rejects_small_or_flat_and_allows_replace(
    request_stub, technologist, settings
):
    ev = create_evaluation_route(CreateEvaluationRequest(), request_stub, technologist)

    small_png = _make_png_bytes(120, 120)
    asyncio.run(
        upload_radiograph_route(
            ev.id,
            request_stub,
            UploadFileStub("small.png", "image/png", small_png),
            technologist,
        )
    )
    qc_small = validate_radiograph_quality_route(ev.id, request_stub, technologist)
    assert qc_small.passed is False
    assert any("120×120" in msg for msg in qc_small.failures)

    # No puede analizarse sin aprobar HU07
    with pytest.raises(HTTPException) as blocked:
        asyncio.run(analyze_evaluation_route(ev.id, request_stub, technologist))
    assert blocked.value.status_code == 409

    # Reemplazar por una imagen plana 512x512 -> falla por falta de variación tonal
    flat_png = _make_png_bytes(512, 512, flat=True)
    asyncio.run(
        replace_radiograph_route(
            ev.id,
            request_stub,
            UploadFileStub("flat.png", "image/png", flat_png),
            technologist,
        )
    )
    qc_flat = validate_radiograph_quality_route(ev.id, request_stub, technologist)
    assert qc_flat.passed is False
    assert any("variación tonal" in msg for msg in qc_flat.failures)

    # Reemplazar por 300x300 válida con advertencia (< 512 px) -> aprueba y pasa a lista_para_analisis
    ok_png = _make_png_bytes(300, 300)
    asyncio.run(
        replace_radiograph_route(
            ev.id,
            request_stub,
            UploadFileStub("ok300.png", "image/png", ok_png),
            technologist,
        )
    )
    qc_ok = validate_radiograph_quality_route(ev.id, request_stub, technologist)
    assert qc_ok.passed is True
    assert len(qc_ok.warnings) >= 1

    ev_after = get_evaluation_route(ev.id, request_stub, technologist)
    assert ev_after.status == EvaluationStatus.LISTA_PARA_ANALISIS

    # Una vez lista_para_analisis ya no admite reemplazo
    with pytest.raises(HTTPException) as locked:
        asyncio.run(
            replace_radiograph_route(
                ev.id,
                request_stub,
                UploadFileStub("other.png", "image/png", ok_png),
                technologist,
            )
        )
    assert locked.value.status_code == 409


def test_end_to_end_clinical_flow_with_stub_engine_and_role_boundaries(
    request_stub, settings, technologist, radiologist, coordinator
):
    eval_id = _create_ready_evaluation(request_stub, technologist)
    request_stub.app.state.inference_engine = _FakeEngine(_fake_output())

    # Tecnólogo dispara análisis, pero no ve campos clínicos resumidos
    analyzed_by_tech = asyncio.run(analyze_evaluation_route(eval_id, request_stub, technologist))
    assert analyzed_by_tech.status == EvaluationStatus.ANALIZADA
    assert analyzed_by_tech.has_prediction is True
    assert analyzed_by_tech.concordance_state is None
    assert analyzed_by_tech.review_required is None

    # Médico lee predicción, cajas, versión del modelo y mapas Grad-CAM/Rollout
    pred = get_prediction_route(eval_id, request_stub, radiologist)
    assert pred.predicted_class.value == "TB"
    assert pred.concordance_state == ConcordanceState.BOTH_ALERT
    assert pred.detector_box_count == 1
    assert pred.model_version.classifier_arch == "vit_b_16"

    grad_resp = download_explainability_route(eval_id, "gradcam", request_stub, radiologist)
    roll_resp = download_explainability_route(eval_id, "rollout", request_stub, coordinator)
    assert isinstance(grad_resp, FileResponse)
    assert isinstance(roll_resp, FileResponse)
    assert grad_resp.headers["cache-control"] == "private, no-store"
    assert roll_resp.headers["x-content-type-options"] == "nosniff"

    # Búsqueda/triaje HU08 en servidor distingue médico vs tecnólogo
    clin_filtered = list_evaluations_route(
        request_stub,
        review_required=True,
        concordance_state=ConcordanceState.BOTH_ALERT,
        current_user=radiologist,
    )
    assert [item.id for item in clin_filtered] == [eval_id]

    tech_list = list_evaluations_route(request_stub, current_user=technologist)
    assert tech_list[0].concordance_state is None

    # Validación anti-PII en observaciones (HU15)
    with pytest.raises(ValidationError):
        CreateObservationRequest(
            clinical_impression="Contactar a paciente@example.com",
            agreement_status=ObservationAgreement.CONCORDANTE,
        )
    with pytest.raises(ValidationError):
        CreateObservationRequest(
            clinical_impression="DNI 12345678 sospecha",
            agreement_status=ObservationAgreement.CONCORDANTE,
        )

    # Médico registra observación y marca evaluada -> revisada
    obs = create_observation_route(
        eval_id,
        CreateObservationRequest(
            clinical_impression="Infiltrado apical compatible; solicitar baciloscopía.",
            agreement_status=ObservationAgreement.CONCORDANTE,
            mark_reviewed=True,
        ),
        request_stub,
        radiologist,
    )
    assert obs.author_display_name == str(radiologist["display_name"])

    reviewed = get_evaluation_route(eval_id, request_stub, coordinator)
    assert reviewed.status == EvaluationStatus.REVISADA
    assert reviewed.concordance_state == ConcordanceState.BOTH_ALERT
    assert reviewed.observation_count == 1

    obs_list = list_observations_route(eval_id, request_stub, coordinator)
    assert len(obs_list) == 1

    # Auditoría no almacena texto clínico de la observación
    conn = connect_database(settings.database_path)
    try:
        audit_rows = conn.execute(
            "SELECT event_type, metadata_json FROM audit_events ORDER BY occurred_at DESC"
        ).fetchall()
    finally:
        conn.close()
    joined_meta = " ".join(r["metadata_json"] for r in audit_rows)
    assert "baciloscopía" not in joined_meta
    assert any(r["event_type"] == "evaluation.reviewed" for r in audit_rows)
    assert any(r["event_type"] == "prediction.created" for r in audit_rows)


def test_real_frozen_phase5_engine_executes_end_to_end(
    request_stub, technologist, radiologist
):
    """Prueba real cargando ml/models/frozen_phase5/ (ViT-B/16 + YOLO11n + Grad-CAM + Rollout)."""
    eval_id = _create_ready_evaluation(request_stub, technologist, width=640, height=512)
    analyzed = asyncio.run(analyze_evaluation_route(eval_id, request_stub, radiologist))
    assert analyzed.status == EvaluationStatus.ANALIZADA
    assert analyzed.has_prediction is True

    pred = get_prediction_route(eval_id, request_stub, radiologist)
    assert pytest.approx(pred.p_health + pred.p_sick + pred.p_tb, abs=1e-4) == 1.0
    assert pred.explainability_verified is True
    assert pred.inference_ms > 0.0
    assert pred.model_version.classifier_sha256.startswith("2873fbcf")

    grad_resp = download_explainability_route(eval_id, "gradcam", request_stub, radiologist)
    roll_resp = download_explainability_route(eval_id, "rollout", request_stub, radiologist)
    assert Path(grad_resp.path).is_file()
    assert Path(roll_resp.path).is_file()


def test_new_clinical_permissions_enforce_role_matrix(
    administrator, technologist, program_doctor, radiologist, coordinator
):
    model_dep = require_any_permission(MODEL_VERSIONS_MANAGE, PREDICTIONS_READ)
    for allowed in (administrator, program_doctor, radiologist, coordinator):
        assert model_dep(allowed) is allowed
    with pytest.raises(HTTPException) as denied_tech_model:
        model_dep(technologist)
    assert denied_tech_model.value.status_code == 403

    analyze_dep = require_any_permission(RADIOGRAPHS_UPLOAD, OBSERVATIONS_WRITE)
    for allowed in (technologist, program_doctor, radiologist):
        assert analyze_dep(allowed) is allowed
    for forbidden in (administrator, coordinator):
        with pytest.raises(HTTPException) as denied_analyze:
            analyze_dep(forbidden)
        assert denied_analyze.value.status_code == 403

    pred_dep = require_permission(PREDICTIONS_READ)
    for forbidden in (administrator, technologist):
        with pytest.raises(HTTPException) as denied_pred:
            pred_dep(forbidden)
        assert denied_pred.value.status_code == 403

    obs_read_dep = require_permission(OBSERVATIONS_READ)
    for forbidden in (administrator, technologist):
        with pytest.raises(HTTPException) as denied_obs_read:
            obs_read_dep(forbidden)
        assert denied_obs_read.value.status_code == 403

    obs_write_dep = require_permission(OBSERVATIONS_WRITE)
    for forbidden in (administrator, technologist, coordinator):
        with pytest.raises(HTTPException) as denied_obs_write:
            obs_write_dep(forbidden)
        assert denied_obs_write.value.status_code == 403
