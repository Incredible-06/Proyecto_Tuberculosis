from __future__ import annotations

import re
from uuid import uuid4

import pytest
from fastapi import HTTPException
from pydantic import ValidationError

from app.database import connect_database
from app.dependencies import require_permission
from app.evaluation_routes import create_evaluation_route, get_evaluation_route, list_evaluations_route
from app.evaluations import (
    EvaluationNotFoundError,
    create_evaluation,
    get_evaluation,
    list_evaluations,
)
from app.schemas import CreateEvaluationRequest, EvaluationStatus, UserRole


CODE_PATTERN = re.compile(r"^TB-\d{8}-[0-9A-F]{6}$")


def test_technologist_creates_evaluation_with_generated_anonymized_code(
    request_stub, technologist, settings
):
    evaluation = create_evaluation_route(
        CreateEvaluationRequest(notes="Control interno del turno"),
        request_stub,
        technologist,
    )

    assert CODE_PATTERN.fullmatch(evaluation.internal_code)
    assert evaluation.status == EvaluationStatus.PENDIENTE_IMAGEN
    assert evaluation.notes == "Control interno del turno"
    assert evaluation.created_by_user_id == str(technologist["id"])
    assert evaluation.created_by_display_name == str(technologist["display_name"])
    assert evaluation.radiograph is None

    connection = connect_database(settings.database_path)
    try:
        event = connection.execute(
            """
            SELECT metadata_json FROM audit_events
            WHERE event_type = 'evaluation.created'
            ORDER BY occurred_at DESC
            """
        ).fetchone()
    finally:
        connection.close()
    assert event is not None
    assert evaluation.internal_code in event["metadata_json"]
    assert "Control interno del turno" not in event["metadata_json"]


def test_evaluation_creation_requires_the_create_permission(technologist):
    create_permission = require_permission("evaluations.create")
    assert create_permission(technologist) is technologist

    for role in (
        "administrador",
        "medico_programa_tb",
        "medico_radiologo",
        "coordinador_estrategia_tb",
    ):
        with pytest.raises(HTTPException) as denied:
            create_permission({"id": "user-x", "role": role})
        assert denied.value.status_code == 403


def test_notes_reject_personal_data_and_normalize_whitespace():
    assert CreateEvaluationRequest(notes="  control   interno ").notes == "control interno"
    assert CreateEvaluationRequest(notes="   ").notes is None

    with pytest.raises(ValidationError):
        CreateEvaluationRequest(notes="Escribir a persona@example.com")
    with pytest.raises(ValidationError):
        CreateEvaluationRequest(notes="Documento 12345678")
    with pytest.raises(ValidationError):
        CreateEvaluationRequest(notes="contacto @ recepcion")


def test_creation_rolls_back_if_audit_fails(settings, technologist, monkeypatch):
    def failing_audit(*args, **kwargs):
        raise RuntimeError("auditoría no disponible")

    monkeypatch.setattr("app.evaluations.record_audit_event", failing_audit)

    connection = connect_database(settings.database_path)
    try:
        with pytest.raises(RuntimeError):
            create_evaluation(connection, notes=None, actor_user_id=str(technologist["id"]))
        total = connection.execute("SELECT COUNT(*) AS total FROM evaluations").fetchone()["total"]
    finally:
        connection.close()
    assert total == 0


def test_evaluations_can_be_filtered_by_status_and_code(
    request_stub, technologist, settings
):
    connection = connect_database(settings.database_path)
    try:
        first = create_evaluation(
            connection,
            notes=None,
            actor_user_id=str(technologist["id"]),
        )
        second = create_evaluation(
            connection,
            notes=None,
            actor_user_id=str(technologist["id"]),
        )
        by_code = list_evaluations(connection, search=first["internal_code"])
        pending = list_evaluations(connection, status="pendiente_imagen")
    finally:
        connection.close()

    assert [item["internal_code"] for item in by_code] == [first["internal_code"]]
    assert {item["internal_code"] for item in pending} == {
        first["internal_code"],
        second["internal_code"],
    }

    listed = list_evaluations_route(
        request_stub,
        status_filter=EvaluationStatus.PENDIENTE_IMAGEN,
        search=second["internal_code"],
        current_user=technologist,
    )
    assert [item.internal_code for item in listed] == [second["internal_code"]]


def test_get_missing_evaluation_returns_404(request_stub, technologist):
    with pytest.raises(HTTPException) as missing:
        get_evaluation_route(str(uuid4()), request_stub, technologist)
    assert missing.value.status_code == 404


def test_get_evaluation_raises_service_error_when_missing(settings, application):
    connection = connect_database(settings.database_path)
    try:
        with pytest.raises(EvaluationNotFoundError):
            get_evaluation(connection, str(uuid4()))
    finally:
        connection.close()


def test_all_clinical_roles_can_read_evaluations(technologist, program_doctor, radiologist, coordinator):
    read_permission = require_permission("evaluations.read")
    for user in (technologist, program_doctor, radiologist, coordinator):
        assert read_permission(user) is user

    with pytest.raises(HTTPException) as denied:
        read_permission({"id": "admin", "role": UserRole.ADMINISTRADOR.value})
    assert denied.value.status_code == 403
