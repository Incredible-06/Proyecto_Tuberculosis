"""Dependencias de autenticación para las rutas protegidas."""

from __future__ import annotations

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from .config import Settings
from .database import connect_database
from .permissions import permissions_for_role
from .schemas import UserRole
from .services import user_for_session_token


bearer_security = HTTPBearer(auto_error=False)


def session_token_from_request(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None,
) -> str | None:
    if credentials is not None:
        return credentials.credentials
    settings: Settings = request.app.state.settings
    return request.cookies.get(settings.session_cookie_name)


def require_active_user(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_security),
) -> dict[str, str | bool]:
    token = session_token_from_request(request, credentials)
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Se requiere una sesión válida.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    settings: Settings = request.app.state.settings
    connection = connect_database(settings.database_path)
    try:
        user = user_for_session_token(connection, token=token, settings=settings)
    finally:
        connection.close()
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Se requiere una sesión válida.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user


def require_permission(permission: str):
    return require_any_permission(permission)


def require_any_permission(*permissions: str):
    allowed = frozenset(permissions)

    def permission_dependency(
        user: dict[str, str | bool] = Depends(require_active_user),
    ) -> dict[str, str | bool]:
        if not (set(permissions_for_role(str(user["role"]))) & allowed):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="No tiene permiso para realizar esta operación.",
            )
        return user

    return permission_dependency


def require_administrator(
    user: dict[str, str | bool] = Depends(require_active_user),
) -> dict[str, str | bool]:
    if user["role"] != UserRole.ADMINISTRADOR.value:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Esta operación requiere una cuenta administradora.",
        )
    return user
