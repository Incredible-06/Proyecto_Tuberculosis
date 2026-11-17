"""Esquemas públicos del contrato de autenticación y usuarios."""

from __future__ import annotations

import re
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, SecretStr, ValidationInfo, field_validator, model_validator


USERNAME_PATTERN = re.compile(r"^[A-Za-z0-9._-]{3,64}$")
EMAIL_PATTERN = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
LONG_DIGITS_PATTERN = re.compile(r"\b\d{8,}\b")


class UserRole(str, Enum):
    ADMINISTRADOR = "administrador"
    TECNOLOGO_RADIOLOGIA = "tecnologo_radiologia"
    MEDICO_PROGRAMA_TB = "medico_programa_tb"
    MEDICO_RADIOLOGO = "medico_radiologo"
    COORDINADOR_ESTRATEGIA_TB = "coordinador_estrategia_tb"


class ApiSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")


class LoginRequest(ApiSchema):
    username: str = Field(description="Identificador de acceso de la cuenta.")
    password: SecretStr = Field(description="Secreto de la cuenta.")

    @field_validator("username")
    @classmethod
    def normalize_username(cls, value: str) -> str:
        normalized = value.strip().casefold()
        if not USERNAME_PATTERN.fullmatch(normalized):
            raise ValueError("El identificador de acceso no tiene un formato válido.")
        return normalized

    @field_validator("password")
    @classmethod
    def validate_login_password(cls, value: SecretStr) -> SecretStr:
        password = value.get_secret_value()
        if not password or len(password) > 128:
            raise ValueError("La contraseña no tiene un formato válido.")
        return value


class CreateUserRequest(LoginRequest):
    display_name: str = Field(description="Nombre visible del integrante autorizado.")
    role: UserRole

    @field_validator("display_name")
    @classmethod
    def normalize_display_name(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized or len(normalized) > 100:
            raise ValueError("El nombre visible debe contener entre 1 y 100 caracteres.")
        return normalized

    @field_validator("password")
    @classmethod
    def enforce_password_policy(
        cls,
        value: SecretStr,
        validation_info: ValidationInfo,
    ) -> SecretStr:
        password = value.get_secret_value()
        username = str(validation_info.data.get("username", ""))
        if len(password) < 12 or len(password) > 128:
            raise ValueError("La contraseña debe contener entre 12 y 128 caracteres.")
        if not password.strip():
            raise ValueError("La contraseña no puede contener solo espacios.")
        if password.casefold() == username.casefold():
            raise ValueError("La contraseña no puede coincidir con el identificador.")
        return value


class PublicUser(ApiSchema):
    id: str
    username: str
    display_name: str
    role: UserRole
    is_active: bool
    created_at: str
    permissions: list[str] = Field(default_factory=list)


class LoginResponse(ApiSchema):
    user: PublicUser
    session_expires_at: str


class MessageResponse(ApiSchema):
    message: str


class UserUpdateRequest(ApiSchema):
    role: UserRole | None = None
    is_active: bool | None = None

    @model_validator(mode="after")
    def require_at_least_one_change(self) -> "UserUpdateRequest":
        if self.role is None and self.is_active is None:
            raise ValueError("Debe indicar al menos un cambio de rol o de estado.")
        return self


class RoleInfo(ApiSchema):
    value: UserRole
    label: str
    permissions: list[str]


def _sanitize_clinical_text(
    value: str | None,
    *,
    field_label: str,
    allow_empty: bool,
) -> str | None:
    if value is None:
        if allow_empty:
            return None
        raise ValueError(f"{field_label} es obligatorio.")
    normalized = " ".join(value.split())
    if not normalized:
        if allow_empty:
            return None
        raise ValueError(f"{field_label} no puede estar vacío.")
    if EMAIL_PATTERN.search(normalized) or "@" in normalized:
        raise ValueError(f"{field_label} no puede contener correos ni datos personales.")
    if LONG_DIGITS_PATTERN.search(normalized):
        raise ValueError(f"{field_label} no puede contener documentos ni datos personales.")
    return normalized


class EvaluationStatus(str, Enum):
    PENDIENTE_IMAGEN = "pendiente_imagen"
    IMAGEN_CARGADA = "imagen_cargada"
    LISTA_PARA_ANALISIS = "lista_para_analisis"
    ANALIZADA = "analizada"
    REVISADA = "revisada"


class PredictionClass(str, Enum):
    HEALTH = "health"
    SICK = "sick"
    TB = "TB"


class ConcordanceState(str, Enum):
    BOTH_ALERT = "both_alert"
    CLASSIFIER_ONLY_REVIEW = "classifier_only_review"
    DETECTOR_ONLY_REVIEW = "detector_only_review"
    NEITHER_ALERT = "neither_alert"


class ObservationAgreement(str, Enum):
    CONCORDANTE = "concordante"
    DISCREPANTE = "discrepante"
    REQUIERE_CONFIRMACION = "requiere_confirmacion"
    SIN_HALLAZGOS_RELEVANTES = "sin_hallazgos_relevantes"


class CreateEvaluationRequest(ApiSchema):
    notes: str | None = Field(default=None, max_length=200)

    @field_validator("notes")
    @classmethod
    def reject_personal_data(cls, value: str | None) -> str | None:
        return _sanitize_clinical_text(value, field_label="La nota", allow_empty=True)


class RadiographPublic(ApiSchema):
    id: str
    evaluation_id: str
    filename: str
    content_type: str
    size_bytes: int
    width: int
    height: int
    sha256: str
    uploaded_at: str
    content_url: str


class QualityMetricsPublic(ApiSchema):
    width: int
    height: int
    min_dimension: int
    aspect_ratio: float
    intensity_std: float
    dynamic_range_p99_p01: float
    clip_low_ratio: float
    clip_high_ratio: float


class QualityCheckPublic(ApiSchema):
    id: str
    evaluation_id: str
    radiograph_id: str
    passed: bool
    warnings: list[str] = Field(default_factory=list)
    failures: list[str] = Field(default_factory=list)
    metrics: QualityMetricsPublic
    checked_by_user_id: str | None = None
    checked_at: str


class ModelVersionPublic(ApiSchema):
    id: str
    bundle_schema: str
    contract_schema: str
    contract_sha256: str
    data_config_sha256: str
    classifier_arch: str
    classifier_display_name: str
    classifier_run_id: str
    classifier_sha256: str
    classifier_epoch: int
    classifier_threshold: float
    detector_arch: str
    detector_run_id: str
    detector_sha256: str
    detector_threshold: float
    concordance_rule: str
    image_size: int
    is_active: bool
    registered_at: str


class DetectionPublic(ApiSchema):
    id: str
    box_index: int
    label: str
    confidence: float
    xyxy: list[float]
    normalized_xyxy: list[float]


class PredictionPublic(ApiSchema):
    id: str
    evaluation_id: str
    radiograph_id: str
    predicted_class: PredictionClass
    p_health: float
    p_sick: float
    p_tb: float
    classifier_alert: bool
    detector_box_count: int
    detector_max_conf: float
    detector_alert: bool
    concordance_state: ConcordanceState
    review_required: bool
    low_confidence: bool
    low_confidence_reasons: list[str] = Field(default_factory=list)
    inference_ms: float
    explainability_verified: bool
    gradcam_url: str
    rollout_url: str
    detections: list[DetectionPublic] = Field(default_factory=list)
    model_version: ModelVersionPublic
    created_by_user_id: str | None = None
    created_at: str


class CreateObservationRequest(ApiSchema):
    clinical_impression: str = Field(min_length=3, max_length=1000)
    agreement_status: ObservationAgreement
    mark_reviewed: bool = True

    @field_validator("clinical_impression")
    @classmethod
    def reject_personal_data(cls, value: str) -> str:
        sanitized = _sanitize_clinical_text(
            value,
            field_label="La observación clínica",
            allow_empty=False,
        )
        assert sanitized is not None
        return sanitized


class ObservationPublic(ApiSchema):
    id: str
    evaluation_id: str
    prediction_id: str | None = None
    author_user_id: str | None = None
    author_display_name: str | None = None
    clinical_impression: str
    agreement_status: ObservationAgreement
    created_at: str


class EvaluationPublic(ApiSchema):
    id: str
    internal_code: str
    status: EvaluationStatus
    notes: str | None = None
    created_at: str
    updated_at: str
    created_by_user_id: str | None = None
    created_by_display_name: str | None = None
    radiograph: RadiographPublic | None = None
    quality_check: QualityCheckPublic | None = None
    has_prediction: bool = False
    concordance_state: ConcordanceState | None = None
    review_required: bool | None = None
    observation_count: int = 0
