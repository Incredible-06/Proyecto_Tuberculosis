"""Reglas de negocio para cuentas, sesiones y auditoría."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from uuid import uuid4

from .config import Settings
from .permissions import permissions_for_role
from .schemas import UserRole
from .security import DUMMY_PASSWORD_HASH, create_session_token, hash_password, session_token_digest, verify_password


class InvalidCredentialsError(Exception):
    """Se usa para respuestas que no revelan el motivo del fallo."""


class AccountLockedError(Exception):
    """La cuenta alcanzó el máximo de intentos consecutivos."""


class DuplicateUsernameError(Exception):
    """El identificador ya existe sin distinguir mayúsculas."""


class UserNotFoundError(Exception):
    """La cuenta solicitada no existe."""


class LastAdministratorError(Exception):
    """La operación dejaría el sistema sin administradores activos."""


class SelfDeactivationError(Exception):
    """Una cuenta no puede desactivarse a sí misma."""


@dataclass(frozen=True)
class AuthenticationResult:
    user: dict[str, object]
    token: str
    expires_at: str


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _timestamp(value: datetime) -> str:
    return value.isoformat()


def _parse_timestamp(value: str | None) -> datetime | None:
    if not value:
        return None
    return datetime.fromisoformat(value)


def _public_user(row: sqlite3.Row) -> dict[str, object]:
    return {
        "id": row["id"],
        "username": row["username"],
        "display_name": row["display_name"],
        "role": row["role"],
        "is_active": bool(row["is_active"]),
        "created_at": row["created_at"],
        "permissions": permissions_for_role(str(row["role"])),
    }


def record_audit_event(
    connection: sqlite3.Connection,
    event_type: str,
    actor_user_id: str | None = None,
    subject_user_id: str | None = None,
    metadata: dict[str, object] | None = None,
) -> None:
    connection.execute(
        """
        INSERT INTO audit_events(id, actor_user_id, subject_user_id, event_type, metadata_json, occurred_at)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            str(uuid4()),
            actor_user_id,
            subject_user_id,
            event_type,
            json.dumps(metadata or {}, ensure_ascii=False, separators=(",", ":")),
            _timestamp(_utc_now()),
        ),
    )


def count_users(connection: sqlite3.Connection) -> int:
    return int(connection.execute("SELECT COUNT(*) AS total FROM users").fetchone()["total"])


def create_user_account(
    connection: sqlite3.Connection,
    *,
    username: str,
    display_name: str,
    role: UserRole,
    password: str,
    actor_user_id: str | None,
) -> dict[str, object]:
    created_at = _timestamp(_utc_now())
    user_id = str(uuid4())
    try:
        with connection:
            connection.execute(
                """
                INSERT INTO users(
                    id, username, display_name, role, password_hash, is_active,
                    failed_login_count, locked_until, created_at, created_by_user_id
                ) VALUES (?, ?, ?, ?, ?, 1, 0, NULL, ?, ?)
                """,
                (
                    user_id,
                    username,
                    display_name,
                    role.value,
                    hash_password(password),
                    created_at,
                    actor_user_id,
                ),
            )
            record_audit_event(
                connection,
                "user.created",
                actor_user_id=actor_user_id,
                subject_user_id=user_id,
            )
    except sqlite3.IntegrityError as error:
        if "users.username" in str(error):
            raise DuplicateUsernameError from error
        raise

    created_user = connection.execute(
        """
        SELECT id, username, display_name, role, is_active, created_at
        FROM users WHERE id = ?
        """,
        (user_id,),
    ).fetchone()
    if created_user is None:
        raise RuntimeError("No se pudo recuperar la cuenta recién creada.")
    return _public_user(created_user)


def _find_user_by_username(connection: sqlite3.Connection, username: str) -> sqlite3.Row | None:
    return connection.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()


def authenticate_user(
    connection: sqlite3.Connection,
    *,
    username: str,
    password: str,
    settings: Settings,
) -> AuthenticationResult:
    now = _utc_now()
    user = _find_user_by_username(connection, username)
    if user is None:
        verify_password(password, DUMMY_PASSWORD_HASH)
        with connection:
            record_audit_event(connection, "auth.login_failed")
        raise InvalidCredentialsError

    locked_until = _parse_timestamp(user["locked_until"])
    if locked_until and locked_until > now:
        with connection:
            record_audit_event(connection, "auth.login_blocked", subject_user_id=user["id"])
        raise AccountLockedError

    password_is_valid = verify_password(password, user["password_hash"])
    if not password_is_valid or not bool(user["is_active"]):
        previous_failures = 0 if locked_until else int(user["failed_login_count"])
        failure_count = previous_failures + 1
        new_locked_until = None
        event_type = "auth.login_failed"
        if failure_count >= settings.max_login_failures:
            new_locked_until = _timestamp(now + timedelta(minutes=settings.lockout_minutes))
            event_type = "auth.login_locked"
        with connection:
            connection.execute(
                """
                UPDATE users
                SET failed_login_count = ?, locked_until = ?
                WHERE id = ?
                """,
                (failure_count, new_locked_until, user["id"]),
            )
            record_audit_event(connection, event_type, subject_user_id=user["id"])
        if new_locked_until:
            raise AccountLockedError
        raise InvalidCredentialsError

    token = create_session_token()
    expires_at = _timestamp(now + timedelta(minutes=settings.session_ttl_minutes))
    with connection:
        connection.execute(
            "UPDATE users SET failed_login_count = 0, locked_until = NULL WHERE id = ?",
            (user["id"],),
        )
        connection.execute(
            """
            INSERT INTO auth_sessions(id, user_id, token_digest, issued_at, expires_at, revoked_at)
            VALUES (?, ?, ?, ?, ?, NULL)
            """,
            (
                str(uuid4()),
                user["id"],
                session_token_digest(token, settings.session_secret),
                _timestamp(now),
                expires_at,
            ),
        )
        record_audit_event(
            connection,
            "auth.login_succeeded",
            actor_user_id=user["id"],
            subject_user_id=user["id"],
        )
    return AuthenticationResult(user=_public_user(user), token=token, expires_at=expires_at)


def user_for_session_token(
    connection: sqlite3.Connection,
    *,
    token: str,
    settings: Settings,
) -> dict[str, object] | None:
    token_digest = session_token_digest(token, settings.session_secret)
    session_user = connection.execute(
        """
        SELECT users.id, users.username, users.display_name, users.role, users.is_active, users.created_at
        FROM auth_sessions
        JOIN users ON users.id = auth_sessions.user_id
        WHERE auth_sessions.token_digest = ?
          AND auth_sessions.revoked_at IS NULL
          AND auth_sessions.expires_at > ?
          AND users.is_active = 1
        """,
        (token_digest, _timestamp(_utc_now())),
    ).fetchone()
    return _public_user(session_user) if session_user else None


def revoke_session(
    connection: sqlite3.Connection,
    *,
    token: str,
    settings: Settings,
    actor_user_id: str,
) -> None:
    token_digest = session_token_digest(token, settings.session_secret)
    with connection:
        connection.execute(
            """
            UPDATE auth_sessions
            SET revoked_at = ?
            WHERE token_digest = ? AND revoked_at IS NULL
            """,
            (_timestamp(_utc_now()), token_digest),
        )
        record_audit_event(
            connection,
            "auth.logout",
            actor_user_id=actor_user_id,
            subject_user_id=actor_user_id,
        )


def list_user_accounts(connection: sqlite3.Connection) -> list[dict[str, object]]:
    rows = connection.execute(
        """
        SELECT id, username, display_name, role, is_active, created_at
        FROM users
        ORDER BY created_at ASC, username ASC
        """
    ).fetchall()
    return [_public_user(row) for row in rows]


def update_user_account(
    connection: sqlite3.Connection,
    *,
    user_id: str,
    role: UserRole | None,
    is_active: bool | None,
    actor_user_id: str,
) -> dict[str, object]:
    target = connection.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    if target is None:
        raise UserNotFoundError

    changes: dict[str, object] = {}
    if role is not None and role.value != target["role"]:
        changes["role"] = role.value
    if is_active is not None and bool(target["is_active"]) != is_active:
        changes["is_active"] = is_active

    if not changes:
        return _public_user(target)

    if changes.get("is_active") is False and user_id == actor_user_id:
        raise SelfDeactivationError

    current_role = str(target["role"])
    current_is_active = bool(target["is_active"])
    new_role = str(changes.get("role", current_role))
    new_is_active = bool(changes.get("is_active", current_is_active))
    if current_role == UserRole.ADMINISTRADOR.value and current_is_active:
        if not (new_role == UserRole.ADMINISTRADOR.value and new_is_active):
            other_active_administrators = connection.execute(
                """
                SELECT COUNT(*) AS total
                FROM users
                WHERE role = ? AND is_active = 1 AND id != ?
                """,
                (UserRole.ADMINISTRADOR.value, user_id),
            ).fetchone()["total"]
            if int(other_active_administrators) == 0:
                raise LastAdministratorError

    assignments = ", ".join(f"{field} = ?" for field in changes)
    parameters = [*changes.values(), user_id]
    with connection:
        connection.execute(
            f"UPDATE users SET {assignments} WHERE id = ?",
            parameters,
        )
        record_audit_event(
            connection,
            "user.updated",
            actor_user_id=actor_user_id,
            subject_user_id=user_id,
            metadata={"changed_fields": sorted(changes), **changes},
        )

    updated = connection.execute(
        """
        SELECT id, username, display_name, role, is_active, created_at
        FROM users WHERE id = ?
        """,
        (user_id,),
    ).fetchone()
    if updated is None:
        raise RuntimeError("No se pudo recuperar la cuenta actualizada.")
    return _public_user(updated)
