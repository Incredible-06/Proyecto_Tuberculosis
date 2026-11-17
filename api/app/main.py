"""Punto de entrada FastAPI para HU01 y HU02."""

from __future__ import annotations

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request, Response, status
from fastapi.responses import FileResponse
from fastapi.security import HTTPAuthorizationCredentials
from fastapi.staticfiles import StaticFiles

from .clinical import sync_frozen_model_version
from .clinical_routes import clinical_router
from .config import PROJECT_ROOT, Settings
from .database import connect_database, initialize_database
from .dependencies import bearer_security, require_active_user, require_permission, session_token_from_request
from .evaluation_routes import evaluation_router
from .inference import BundleIntegrityError, FrozenPhase5Engine
from .permissions import ROLES_READ, USERS_MANAGE, USERS_READ, role_catalog
from .schemas import (
    CreateUserRequest,
    LoginRequest,
    LoginResponse,
    MessageResponse,
    PublicUser,
    RoleInfo,
    UserUpdateRequest,
)
from .services import (
    AccountLockedError,
    DuplicateUsernameError,
    InvalidCredentialsError,
    LastAdministratorError,
    SelfDeactivationError,
    UserNotFoundError,
    authenticate_user,
    create_user_account,
    list_user_accounts,
    revoke_session,
    update_user_account,
)


api_router = APIRouter(prefix="/api/v1")


@api_router.get("/health")
def health(request: Request) -> dict[str, str]:
    settings: Settings = request.app.state.settings
    return {"status": "ok", "environment": settings.environment}


@api_router.post("/auth/login", response_model=LoginResponse)
def login(credentials: LoginRequest, request: Request, response: Response) -> LoginResponse:
    settings: Settings = request.app.state.settings
    connection = connect_database(settings.database_path)
    try:
        result = authenticate_user(
            connection,
            username=credentials.username,
            password=credentials.password.get_secret_value(),
            settings=settings,
        )
    except AccountLockedError as error:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Demasiados intentos. Intente nuevamente más tarde.",
        ) from error
    except InvalidCredentialsError as error:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Credenciales inválidas.",
        ) from error
    finally:
        connection.close()

    response.set_cookie(
        key=settings.session_cookie_name,
        value=result.token,
        max_age=settings.session_ttl_minutes * 60,
        httponly=True,
        secure=settings.cookie_secure,
        samesite="strict",
        path="/",
    )
    return LoginResponse(user=result.user, session_expires_at=result.expires_at)


@api_router.post("/auth/logout", response_model=MessageResponse)
def logout(
    request: Request,
    response: Response,
    current_user: dict[str, str | bool] = Depends(require_active_user),
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_security),
) -> MessageResponse:
    settings: Settings = request.app.state.settings
    token = session_token_from_request(request, credentials)
    if token is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Se requiere una sesión válida.")
    connection = connect_database(settings.database_path)
    try:
        revoke_session(
            connection,
            token=token,
            settings=settings,
            actor_user_id=str(current_user["id"]),
        )
    finally:
        connection.close()
    response.delete_cookie(
        key=settings.session_cookie_name,
        path="/",
        secure=settings.cookie_secure,
        httponly=True,
        samesite="strict",
    )
    return MessageResponse(message="Sesión cerrada.")


@api_router.get("/auth/me", response_model=PublicUser)
def current_identity(
    current_user: dict[str, str | bool] = Depends(require_active_user),
) -> PublicUser:
    return PublicUser.model_validate(current_user)


@api_router.post("/users", response_model=PublicUser, status_code=status.HTTP_201_CREATED)
def create_user(
    payload: CreateUserRequest,
    request: Request,
    current_user: dict[str, str | bool] = Depends(require_permission(USERS_MANAGE)),
) -> PublicUser:
    settings: Settings = request.app.state.settings
    connection = connect_database(settings.database_path)
    try:
        user = create_user_account(
            connection,
            username=payload.username,
            display_name=payload.display_name,
            role=payload.role,
            password=payload.password.get_secret_value(),
            actor_user_id=str(current_user["id"]),
        )
    except DuplicateUsernameError as error:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="El identificador de acceso ya está registrado.",
        ) from error
    finally:
        connection.close()
    return PublicUser.model_validate(user)


@api_router.get("/users", response_model=list[PublicUser])
def list_users(
    request: Request,
    current_user: dict[str, str | bool] = Depends(require_permission(USERS_READ)),
) -> list[PublicUser]:
    settings: Settings = request.app.state.settings
    connection = connect_database(settings.database_path)
    try:
        users = list_user_accounts(connection)
    finally:
        connection.close()
    return [PublicUser.model_validate(user) for user in users]


@api_router.patch("/users/{user_id}", response_model=PublicUser)
def update_user(
    user_id: str,
    payload: UserUpdateRequest,
    request: Request,
    current_user: dict[str, str | bool] = Depends(require_permission(USERS_MANAGE)),
) -> PublicUser:
    settings: Settings = request.app.state.settings
    connection = connect_database(settings.database_path)
    try:
        user = update_user_account(
            connection,
            user_id=user_id,
            role=payload.role,
            is_active=payload.is_active,
            actor_user_id=str(current_user["id"]),
        )
    except UserNotFoundError as error:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="La cuenta solicitada no existe.",
        ) from error
    except LastAdministratorError as error:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="La operación dejaría el sistema sin administradores activos.",
        ) from error
    except SelfDeactivationError as error:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="No puede desactivar su propia cuenta.",
        ) from error
    finally:
        connection.close()
    return PublicUser.model_validate(user)


@api_router.get("/roles", response_model=list[RoleInfo])
def list_roles(
    current_user: dict[str, str | bool] = Depends(require_permission(ROLES_READ)),
) -> list[RoleInfo]:
    return [RoleInfo.model_validate(role) for role in role_catalog()]


def create_app(settings: Settings | None = None) -> FastAPI:
    resolved_settings = settings or Settings.from_environment()
    initialize_database(resolved_settings)
    if resolved_settings.frozen_model_dir.is_dir():
        connection = connect_database(resolved_settings.database_path)
        try:
            sync_frozen_model_version(connection, resolved_settings.frozen_model_dir)
        except BundleIntegrityError:
            if resolved_settings.environment == "production":
                raise
        finally:
            connection.close()

    application = FastAPI(
        title="Proyecto Tuberculosis API",
        version="0.3.0",
        description="Prototipo académico de apoyo diagnóstico de tuberculosis (Fase 5 congelada).",
    )
    application.state.settings = resolved_settings
    application.state.inference_engine = FrozenPhase5Engine(
        resolved_settings.frozen_model_dir,
        device=resolved_settings.inference_device,
        low_confidence_threshold=resolved_settings.low_confidence_threshold,
    )
    application.include_router(api_router)
    application.include_router(evaluation_router)
    application.include_router(clinical_router)

    web_dist = PROJECT_ROOT / "web" / "dist"
    if web_dist.is_dir():
        assets_directory = web_dist / "assets"
        if assets_directory.is_dir():
            application.mount("/assets", StaticFiles(directory=assets_directory), name="assets")

        @application.get("/", include_in_schema=False)
        def serve_web_index() -> FileResponse:
            return FileResponse(web_dist / "index.html")

        @application.get("/{full_path:path}", include_in_schema=False)
        def serve_web_application(full_path: str) -> FileResponse:
            if full_path.startswith("api/"):
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND)
            candidate = (web_dist / full_path).resolve()
            if web_dist.resolve() in candidate.parents and candidate.is_file():
                return FileResponse(candidate)
            return FileResponse(web_dist / "index.html")
    else:
        @application.get("/", include_in_schema=False)
        def development_entry_point() -> dict[str, str]:
            return {
                "message": "API activa. Ejecute npm run dev dentro de web/ para la interfaz.",
                "docs": "/docs",
            }

    return application


app = create_app()
