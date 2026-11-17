from __future__ import annotations

from http.cookies import SimpleCookie
from pathlib import Path

import pytest
from fastapi import HTTPException, Response

from app.database import connect_database
from app.dependencies import require_active_user, require_administrator
from app.main import create_user, current_identity, login, logout
from app.schemas import CreateUserRequest, LoginRequest
from app.security import verify_password

from .conftest import RequestStub, create_regular_user


def login_as_administrator(request_stub: RequestStub):
    response = Response()
    result = login(
        LoginRequest(username="admin.local", password="Una clave inicial segura"),
        request_stub,
        response,
    )
    parsed_cookie = SimpleCookie()
    parsed_cookie.load(response.headers["set-cookie"])
    request_stub.cookies["tbx_session"] = parsed_cookie["tbx_session"].value
    return result, response


def test_valid_login_sets_http_only_session_and_exposes_identity(request_stub, administrator):
    result, response = login_as_administrator(request_stub)

    assert result.user.username == "admin.local"
    assert "httponly" in response.headers["set-cookie"].lower()
    assert "tbx_session" not in result.model_dump()

    current_user = require_active_user(request_stub, credentials=None)
    identity = current_identity(current_user)
    assert identity.role == "administrador"


def test_invalid_credentials_do_not_reveal_whether_an_account_exists(request_stub, administrator):
    with pytest.raises(HTTPException) as wrong_password:
        login(
            LoginRequest(username="admin.local", password="Contraseña incorrecta"),
            request_stub,
            Response(),
        )
    with pytest.raises(HTTPException) as missing_account:
        login(
            LoginRequest(username="cuenta.inexistente", password="Contraseña incorrecta"),
            request_stub,
            Response(),
        )

    assert wrong_password.value.status_code == 401
    assert missing_account.value.status_code == 401
    assert wrong_password.value.detail == missing_account.value.detail == "Credenciales inválidas."


def test_five_failed_attempts_lock_the_account(request_stub, administrator):
    for attempt_number in range(4):
        with pytest.raises(HTTPException) as failed_login:
            login(
                LoginRequest(username="admin.local", password="Contraseña incorrecta"),
                request_stub,
                Response(),
            )
        assert failed_login.value.status_code == 401, attempt_number

    with pytest.raises(HTTPException) as fifth_attempt:
        login(
            LoginRequest(username="admin.local", password="Contraseña incorrecta"),
            request_stub,
            Response(),
        )
    assert fifth_attempt.value.status_code == 429

    with pytest.raises(HTTPException) as blocked_valid_login:
        login_as_administrator(request_stub)
    assert blocked_valid_login.value.status_code == 429


def test_anonymous_and_non_administrator_sessions_cannot_create_users(request_stub, administrator, settings):
    with pytest.raises(HTTPException) as anonymous_request:
        require_active_user(request_stub, credentials=None)
    assert anonymous_request.value.status_code == 401

    regular_user = create_regular_user(settings, str(administrator["id"]))
    with pytest.raises(HTTPException) as regular_request:
        require_administrator(regular_user)
    assert regular_request.value.status_code == 403


def test_administrator_creates_user_with_a_hash_and_audit_event(request_stub, administrator, settings):
    login_as_administrator(request_stub)
    password = "Una frase de paso segura"
    payload = CreateUserRequest(
        username="tecnico.ray",
        display_name="Tecnólogo de radiología",
        role="tecnologo_radiologia",
        password=password,
    )

    created_user = create_user(payload, request_stub, administrator)

    assert created_user.username == "tecnico.ray"
    assert "password" not in created_user.model_dump()

    connection = connect_database(settings.database_path)
    try:
        row = connection.execute(
            "SELECT password_hash FROM users WHERE username = ?",
            ("tecnico.ray",),
        ).fetchone()
        event = connection.execute(
            "SELECT event_type, metadata_json FROM audit_events WHERE event_type = 'user.created' ORDER BY occurred_at DESC"
        ).fetchone()
    finally:
        connection.close()
    assert row is not None
    assert row["password_hash"] != password
    assert row["password_hash"].startswith("scrypt$")
    assert verify_password(password, row["password_hash"])
    assert event is not None
    assert password not in event["metadata_json"]


def test_duplicate_usernames_are_case_insensitive_and_do_not_create_partial_rows(request_stub, administrator, settings):
    payload = CreateUserRequest(
        username="ADMIN.LOCAL",
        display_name="Cuenta duplicada",
        role="medico_radiologo",
        password="Una frase de paso segura",
    )

    with pytest.raises(HTTPException) as duplicate_user:
        create_user(payload, request_stub, administrator)
    assert duplicate_user.value.status_code == 409

    connection = connect_database(settings.database_path)
    try:
        total = connection.execute("SELECT COUNT(*) AS total FROM users").fetchone()["total"]
    finally:
        connection.close()
    assert total == 1


def test_logout_revokes_the_active_session(request_stub, administrator):
    _, login_response = login_as_administrator(request_stub)
    logout_response = Response()
    current_user = require_active_user(request_stub, credentials=None)

    result = logout(request_stub, logout_response, current_user, credentials=None)

    assert result.message == "Sesión cerrada."
    assert "tbx_session" in login_response.headers["set-cookie"]
    with pytest.raises(HTTPException) as inactive_session:
        require_active_user(request_stub, credentials=None)
    assert inactive_session.value.status_code == 401


def test_web_interface_contains_the_react_entry_point():
    web_root = Path(__file__).resolve().parents[2] / "web"
    index = (web_root / "index.html").read_text(encoding="utf-8")
    package = (web_root / "package.json").read_text(encoding="utf-8")
    assert 'id="root"' in index
    assert "/src/main.tsx" in index
    assert '"react"' in package
    assert '"vite"' in package
