"""Configuración de la API obtenida exclusivamente del entorno."""

from __future__ import annotations

import os
import secrets
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _read_boolean(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} debe ser true o false.")


def _read_positive_integer(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        parsed = int(value)
    except ValueError as error:
        raise ValueError(f"{name} debe ser un entero positivo.") from error
    if parsed <= 0:
        raise ValueError(f"{name} debe ser un entero positivo.")
    return parsed


def _read_positive_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        parsed = float(value)
    except ValueError as error:
        raise ValueError(f"{name} debe ser un número positivo.") from error
    if parsed <= 0.0 or parsed > 1.0:
        raise ValueError(f"{name} debe estar en (0, 1].")
    return parsed


@dataclass(frozen=True)
class Settings:
    database_path: Path
    session_secret: str
    environment: str = "development"
    session_ttl_minutes: int = 480
    max_login_failures: int = 5
    lockout_minutes: int = 15
    cookie_secure: bool = False
    session_cookie_name: str = "tbx_session"
    storage_root: Path = PROJECT_ROOT / ".local" / "runtime" / "api"
    max_upload_bytes: int = 10 * 1024 * 1024
    frozen_model_dir: Path = PROJECT_ROOT / "ml" / "models" / "frozen_phase5"
    inference_device: str = "auto"
    min_image_dimension: int = 224
    recommended_image_dimension: int = 512
    low_confidence_threshold: float = 0.70

    def __post_init__(self) -> None:
        if len(self.session_secret) < 32:
            raise ValueError("APP_SESSION_SECRET debe contener al menos 32 caracteres.")
        if self.session_ttl_minutes <= 0:
            raise ValueError("La sesión debe tener una duración positiva.")
        if self.max_login_failures <= 0:
            raise ValueError("El máximo de intentos debe ser positivo.")
        if self.lockout_minutes <= 0:
            raise ValueError("El bloqueo debe tener una duración positiva.")
        if self.max_upload_bytes <= 0:
            raise ValueError("El tamaño máximo de carga debe ser positivo.")
        if self.inference_device not in {"auto", "cpu", "cuda"}:
            raise ValueError("APP_INFERENCE_DEVICE debe ser auto, cpu o cuda.")
        if self.min_image_dimension <= 0 or self.recommended_image_dimension < self.min_image_dimension:
            raise ValueError("Dimensiones mínimas de imagen inválidas.")
        if not (0.0 < self.low_confidence_threshold <= 1.0):
            raise ValueError("APP_LOW_CONFIDENCE_THRESHOLD debe estar en (0, 1].")

    @property
    def radiographs_directory(self) -> Path:
        return self.storage_root / "radiographs"

    @property
    def explainability_directory(self) -> Path:
        return self.storage_root / "explainability"

    @classmethod
    def from_environment(cls) -> "Settings":
        environment = os.getenv("APP_ENV", "development").strip().lower()
        configured_database = os.getenv(
            "APP_DATABASE_PATH",
            str(PROJECT_ROOT / ".local" / "runtime" / "api" / "tuberculosis.sqlite3"),
        )
        database_path = Path(configured_database)
        if not database_path.is_absolute():
            database_path = PROJECT_ROOT / database_path

        configured_storage = os.getenv(
            "APP_STORAGE_ROOT",
            str(PROJECT_ROOT / ".local" / "runtime" / "api"),
        )
        storage_root = Path(configured_storage)
        if not storage_root.is_absolute():
            storage_root = PROJECT_ROOT / storage_root

        configured_model_dir = os.getenv(
            "APP_FROZEN_MODEL_DIR",
            str(PROJECT_ROOT / "ml" / "models" / "frozen_phase5"),
        )
        frozen_model_dir = Path(configured_model_dir)
        if not frozen_model_dir.is_absolute():
            frozen_model_dir = PROJECT_ROOT / frozen_model_dir

        session_secret = os.getenv("APP_SESSION_SECRET")
        if not session_secret:
            if environment == "production":
                raise ValueError("APP_SESSION_SECRET es obligatorio en producción.")
            session_secret = secrets.token_urlsafe(48)

        return cls(
            database_path=database_path,
            session_secret=session_secret,
            environment=environment,
            session_ttl_minutes=_read_positive_integer("APP_SESSION_TTL_MINUTES", 480),
            max_login_failures=_read_positive_integer("APP_MAX_LOGIN_FAILURES", 5),
            lockout_minutes=_read_positive_integer("APP_LOCKOUT_MINUTES", 15),
            cookie_secure=_read_boolean("APP_COOKIE_SECURE", environment == "production"),
            storage_root=storage_root,
            max_upload_bytes=_read_positive_integer("APP_MAX_UPLOAD_BYTES", 10 * 1024 * 1024),
            frozen_model_dir=frozen_model_dir,
            inference_device=os.getenv("APP_INFERENCE_DEVICE", "auto").strip().lower(),
            min_image_dimension=_read_positive_integer("APP_MIN_IMAGE_DIMENSION", 224),
            recommended_image_dimension=_read_positive_integer("APP_RECOMMENDED_IMAGE_DIMENSION", 512),
            low_confidence_threshold=_read_positive_float("APP_LOW_CONFIDENCE_THRESHOLD", 0.70),
        )
