from __future__ import annotations

from http.cookies import SimpleCookie

import pytest
from fastapi import HTTPException, Response
from pydantic import ValidationError

from app.database import connect_database
from app.dependencies import require_active_user, require_permission
from app.main import list_roles, list_users, login, update_user
from app.permissions import permissions_for_role
from app.schemas import LoginRequest, UserRole, UserUpdateRequest

from .conftest import RequestStub, create_user_with_role


EXPECTED_MATRIX = {
    UserRole.ADMINISTRADOR: {
        "users.read",
        "users.manage",
        "roles.read",
        "settings.manage",
        "audit.read",
        "model_versions.manage",
    },
    UserRole.TECNOLOGO_RADIOLOGIA: {
        "evaluations.create",
        "evaluations.read",
        "radiographs.upload",
        "radiographs.read",
    },
    UserRole.MEDICO_PROGRAMA_TB: {
        "evaluations.read",
        "radiographs.read",
        "predictions.read",
        "observations.read",
        "observations.write",
        "reports.read",
        "reports.generate",
    },
    UserRole.MEDICO_RADIOLOGO: {
        "evaluations.read",
        "radiographs.read",
        "predictions.read",
        "observations.read",
        "observations.write",
        "reports.read",
        "reports.generate",
    },
    UserRole.COORDINADOR_ESTRATEGIA_TB: {
        "evaluations.read",
        "radiographs.read",
        "predictions.read",
        "observations.read",
        "reports.read",
        "reports.generate",
    },
}


def login_as(request_stub: RequestStub, username: str, password: str) -> None:
    response = Response()
    login(LoginRequest(username=username, password=password), request_stub, response)
    parsed_cookie = SimpleCookie()
    parsed_cookie.load(response.headers["set-cookie"])
    request_stub.cookies["tbx_session"] = parsed_cookie["tbx_session"].value


def test_permission_matrix_is_exact_for_the_five_roles():
    assert {role.value for role in EXPECTED_MATRIX} == {role.value for role in UserRole}
    for role, expected_permissions in EXPECTED_MATRIX.items():
        assert set(permissions_for_role(role)) == expected_permissions


def test_require_permission_enforces_clinical_and_administration_boundaries():
    create_evaluation = require_permission("evaluations.create")
    technologist = {"id": "user-1", "role": "tecnologo_radiologia"}
    assert create_evaluation(technologist) is technologist

    for role in (
        "administrador",
        "medico_programa_tb",
        "medico_radiologo",
        "coordinador_estrategia_tb",
    ):
        with pytest.raises(HTTPException) as denied:
            create_evaluation({"id": "user-2", "role": role})
        assert denied.value.status_code == 403

    read_evaluations = require_permission("evaluations.read")
    for role in (
        "tecnologo_radiologia",
        "medico_programa_tb",
        "medico_radiologo",
        "coordinador_estrategia_tb",
    ):
        user = {"id": "user-3", "role": role}
        assert read_evaluations(user) is user
    with pytest.raises(HTTPException) as admin_denied:
        read_evaluations({"id": "user-4", "role": "administrador"})
    assert admin_denied.value.status_code == 403


def test_role_catalog_lists_five_roles_and_their_permissions(administrator):
    catalog = list_roles(administrator)
    assert {item.value for item in catalog} == set(UserRole)
    for item in catalog:
        assert item.permissions == permissions_for_role(item.value)
        assert item.label


def test_administrator_updates_role_and_records_audit(
    request_stub, administrator, technologist, settings
):
    updated = update_user(
        str(technologist["id"]),
        UserUpdateRequest(role=UserRole.MEDICO_RADIOLOGO),
        request_stub,
        administrator,
    )

    assert updated.role == UserRole.MEDICO_RADIOLOGO
    assert "predictions.read" in updated.permissions
    assert "radiographs.upload" not in updated.permissions

    connection = connect_database(settings.database_path)
    try:
        event = connection.execute(
            """
            SELECT metadata_json FROM audit_events
            WHERE event_type = 'user.updated'
            ORDER BY occurred_at DESC
            """
        ).fetchone()
    finally:
        connection.close()
    assert event is not None
    assert '"role":"medico_radiologo"' in event["metadata_json"]
    assert "password" not in event["metadata_json"]


def test_last_active_administrator_cannot_be_demoted(
    request_stub, administrator, settings
):
    with pytest.raises(HTTPException) as denied:
        update_user(
            str(administrator["id"]),
            UserUpdateRequest(role=UserRole.MEDICO_PROGRAMA_TB),
            request_stub,
            administrator,
        )
    assert denied.value.status_code == 409
    assert "administradores activos" in denied.value.detail

    connection = connect_database(settings.database_path)
    try:
        row = connection.execute(
            "SELECT role FROM users WHERE id = ?",
            (str(administrator["id"]),),
        ).fetchone()
    finally:
        connection.close()
    assert row["role"] == UserRole.ADMINISTRADOR.value


def test_self_deactivation_is_rejected(request_stub, administrator):
    with pytest.raises(HTTPException) as denied:
        update_user(
            str(administrator["id"]),
            UserUpdateRequest(is_active=False),
            request_stub,
            administrator,
        )
    assert denied.value.status_code == 409
    assert "propia cuenta" in denied.value.detail


def test_deactivated_account_session_is_rejected(
    request_stub, administrator, technologist, settings
):
    technologist_stub = RequestStub(settings)
    login_as(
        technologist_stub,
        str(technologist["username"]),
        "Otra clave segura para pruebas",
    )
    assert require_active_user(technologist_stub, credentials=None)["role"] == (
        UserRole.TECNOLOGO_RADIOLOGIA.value
    )

    update_user(
        str(technologist["id"]),
        UserUpdateRequest(is_active=False),
        request_stub,
        administrator,
    )

    with pytest.raises(HTTPException) as denied:
        require_active_user(technologist_stub, credentials=None)
    assert denied.value.status_code == 401


def test_non_administrators_cannot_read_users_or_roles(technologist):
    with pytest.raises(HTTPException) as users_denied:
        require_permission("users.read")(technologist)
    assert users_denied.value.status_code == 403

    with pytest.raises(HTTPException) as roles_denied:
        require_permission("roles.read")(technologist)
    assert roles_denied.value.status_code == 403


def test_administrator_can_list_users(request_stub, administrator, technologist):
    users = list_users(request_stub, administrator)
    assert {user.username for user in users} >= {
        str(administrator["username"]),
        str(technologist["username"]),
    }
    assert all("password" not in user.model_dump() for user in users)


def test_update_unknown_user_returns_404(request_stub, administrator):
    with pytest.raises(HTTPException) as missing:
        update_user(
            "00000000-0000-0000-0000-000000000000",
            UserUpdateRequest(is_active=False),
            request_stub,
            administrator,
        )
    assert missing.value.status_code == 404


def test_update_requires_at_least_one_change():
    with pytest.raises(ValidationError):
        UserUpdateRequest()


def test_new_administrator_can_be_created_by_administrator(
    request_stub, administrator, settings
):
    second_admin = create_user_with_role(
        settings,
        str(administrator["id"]),
        username="admin.segundo",
        display_name="Administradora secundaria",
        role=UserRole.ADMINISTRADOR,
    )
    updated = update_user(
        str(second_admin["id"]),
        UserUpdateRequest(is_active=False),
        request_stub,
        administrator,
    )
    assert updated.is_active is False
