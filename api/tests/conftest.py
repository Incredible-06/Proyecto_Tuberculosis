from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


API_ROOT = Path(__file__).resolve().parents[1]
if str(API_ROOT) not in sys.path:
    sys.path.insert(0, str(API_ROOT))

from app.config import Settings
from app.database import connect_database
from app.main import create_app
from app.schemas import UserRole
from app.services import create_user_account


class RequestStub:
    def __init__(self, settings: Settings, cookies: dict[str, str] | None = None) -> None:
        self.app = SimpleNamespace(state=SimpleNamespace(settings=settings))
        self.cookies = cookies or {}


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        database_path=tmp_path / "identity.sqlite3",
        session_secret="test-session-secret-with-sufficient-random-looking-length",
        environment="test",
        session_ttl_minutes=60,
        max_login_failures=5,
        lockout_minutes=15,
        cookie_secure=False,
        storage_root=tmp_path / "storage",
        max_upload_bytes=5 * 1024 * 1024,
    )


@pytest.fixture
def application(settings: Settings):
    return create_app(settings)


@pytest.fixture
def request_stub(settings: Settings, application) -> RequestStub:
    return RequestStub(settings)


@pytest.fixture
def administrator(settings: Settings, application) -> dict[str, str | bool]:
    connection = connect_database(settings.database_path)
    try:
        return create_user_account(
            connection,
            username="admin.local",
            display_name="Administradora inicial",
            role=UserRole.ADMINISTRADOR,
            password="Una clave inicial segura",
            actor_user_id=None,
        )
    finally:
        connection.close()


def create_regular_user(settings: Settings, actor_user_id: str) -> dict[str, object]:
    return create_user_with_role(
        settings,
        actor_user_id,
        username="medico.tb",
        display_name="Médico del programa TB",
        role=UserRole.MEDICO_PROGRAMA_TB,
    )


def create_user_with_role(
    settings: Settings,
    actor_user_id: str,
    *,
    username: str,
    display_name: str,
    role: UserRole,
    password: str = "Otra clave segura para pruebas",
) -> dict[str, object]:
    connection = connect_database(settings.database_path)
    try:
        return create_user_account(
            connection,
            username=username,
            display_name=display_name,
            role=role,
            password=password,
            actor_user_id=actor_user_id,
        )
    finally:
        connection.close()


@pytest.fixture
def technologist(settings: Settings, administrator) -> dict[str, object]:
    return create_user_with_role(
        settings,
        str(administrator["id"]),
        username="tecnologo.rayos",
        display_name="Tecnóloga de radiología",
        role=UserRole.TECNOLOGO_RADIOLOGIA,
    )


@pytest.fixture
def program_doctor(settings: Settings, administrator) -> dict[str, object]:
    return create_user_with_role(
        settings,
        str(administrator["id"]),
        username="medico.programa",
        display_name="Médico del programa TB",
        role=UserRole.MEDICO_PROGRAMA_TB,
    )


@pytest.fixture
def radiologist(settings: Settings, administrator) -> dict[str, object]:
    return create_user_with_role(
        settings,
        str(administrator["id"]),
        username="medico.radiologo",
        display_name="Médico radiólogo",
        role=UserRole.MEDICO_RADIOLOGO,
    )


@pytest.fixture
def coordinator(settings: Settings, administrator) -> dict[str, object]:
    return create_user_with_role(
        settings,
        str(administrator["id"]),
        username="coordinador.tb",
        display_name="Coordinador de la estrategia TB",
        role=UserRole.COORDINADOR_ESTRATEGIA_TB,
    )
