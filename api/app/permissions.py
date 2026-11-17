"""Matriz central de permisos por rol para HU03."""

from __future__ import annotations

from .schemas import UserRole


USERS_READ = "users.read"
USERS_MANAGE = "users.manage"
ROLES_READ = "roles.read"
EVALUATIONS_CREATE = "evaluations.create"
EVALUATIONS_READ = "evaluations.read"
RADIOGRAPHS_UPLOAD = "radiographs.upload"
RADIOGRAPHS_READ = "radiographs.read"
PREDICTIONS_READ = "predictions.read"
OBSERVATIONS_READ = "observations.read"
OBSERVATIONS_WRITE = "observations.write"
REPORTS_READ = "reports.read"
REPORTS_GENERATE = "reports.generate"
SETTINGS_MANAGE = "settings.manage"
AUDIT_READ = "audit.read"
MODEL_VERSIONS_MANAGE = "model_versions.manage"


ROLE_LABELS: dict[UserRole, str] = {
    UserRole.ADMINISTRADOR: "Administrador del sistema",
    UserRole.TECNOLOGO_RADIOLOGIA: "Tecnólogo médico en radiología",
    UserRole.MEDICO_PROGRAMA_TB: "Médico del programa de TB",
    UserRole.MEDICO_RADIOLOGO: "Médico radiólogo",
    UserRole.COORDINADOR_ESTRATEGIA_TB: "Coordinador de la Estrategia Sanitaria de TB",
}


ROLE_PERMISSIONS: dict[UserRole, frozenset[str]] = {
    UserRole.ADMINISTRADOR: frozenset(
        {
            USERS_READ,
            USERS_MANAGE,
            ROLES_READ,
            SETTINGS_MANAGE,
            AUDIT_READ,
            MODEL_VERSIONS_MANAGE,
        }
    ),
    UserRole.TECNOLOGO_RADIOLOGIA: frozenset(
        {
            EVALUATIONS_CREATE,
            EVALUATIONS_READ,
            RADIOGRAPHS_UPLOAD,
            RADIOGRAPHS_READ,
        }
    ),
    UserRole.MEDICO_PROGRAMA_TB: frozenset(
        {
            EVALUATIONS_READ,
            RADIOGRAPHS_READ,
            PREDICTIONS_READ,
            OBSERVATIONS_READ,
            OBSERVATIONS_WRITE,
            REPORTS_READ,
            REPORTS_GENERATE,
        }
    ),
    UserRole.MEDICO_RADIOLOGO: frozenset(
        {
            EVALUATIONS_READ,
            RADIOGRAPHS_READ,
            PREDICTIONS_READ,
            OBSERVATIONS_READ,
            OBSERVATIONS_WRITE,
            REPORTS_READ,
            REPORTS_GENERATE,
        }
    ),
    UserRole.COORDINADOR_ESTRATEGIA_TB: frozenset(
        {
            EVALUATIONS_READ,
            RADIOGRAPHS_READ,
            PREDICTIONS_READ,
            OBSERVATIONS_READ,
            REPORTS_READ,
            REPORTS_GENERATE,
        }
    ),
}


def permissions_for_role(role: str | UserRole) -> list[str]:
    resolved_role = role if isinstance(role, UserRole) else UserRole(role)
    return sorted(ROLE_PERMISSIONS[resolved_role])


def role_catalog() -> list[dict[str, object]]:
    return [
        {
            "value": role.value,
            "label": ROLE_LABELS[role],
            "permissions": permissions_for_role(role),
        }
        for role in UserRole
    ]
