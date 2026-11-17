"""Validación de radiografías PNG/JPG sin depender de bibliotecas externas."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path


PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
JPEG_SIGNATURE = b"\xff\xd8\xff"
ALLOWED_EXTENSIONS = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}
CONTENT_TYPE_ALIASES = {
    "image/png": "image/png",
    "image/jpeg": "image/jpeg",
    "image/jpg": "image/jpeg",
    "image/pjpeg": "image/jpeg",
}
JPEG_SOF_MARKERS = {
    0xC0,
    0xC1,
    0xC2,
    0xC3,
    0xC5,
    0xC6,
    0xC7,
    0xC9,
    0xCA,
    0xCB,
    0xCD,
    0xCE,
    0xCF,
}


class ImageValidationError(Exception):
    def __init__(self, message: str, status_code: int = 422) -> None:
        super().__init__(message)
        self.status_code = status_code


class UnsupportedImageTypeError(ImageValidationError):
    def __init__(self, message: str) -> None:
        super().__init__(message, status_code=415)


class ImageTooLargeError(ImageValidationError):
    def __init__(self, message: str) -> None:
        super().__init__(message, status_code=413)


class CorruptImageError(ImageValidationError):
    def __init__(self, message: str) -> None:
        super().__init__(message, status_code=422)


@dataclass(frozen=True)
class ValidatedImage:
    extension: str
    content_type: str
    width: int
    height: int
    size_bytes: int
    sha256: str


def _is_dicom(data: bytes) -> bool:
    return (len(data) >= 132 and data[128:132] == b"DICM") or data.startswith(b"DICM")


def _png_dimensions(data: bytes) -> tuple[int, int]:
    if len(data) < 24 or not data.startswith(PNG_SIGNATURE) or data[12:16] != b"IHDR":
        raise CorruptImageError("El archivo PNG está incompleto o dañado.")
    width = int.from_bytes(data[16:20], "big")
    height = int.from_bytes(data[20:24], "big")
    if width <= 0 or height <= 0:
        raise CorruptImageError("El PNG no declara dimensiones válidas.")
    return width, height


def _jpeg_dimensions(data: bytes) -> tuple[int, int]:
    if len(data) < 4 or not data.startswith(b"\xff\xd8"):
        raise CorruptImageError("El archivo JPEG está incompleto o dañado.")
    index = 2
    while index + 4 <= len(data):
        if data[index] != 0xFF:
            index += 1
            continue
        marker = data[index + 1]
        if marker == 0xFF:
            index += 1
            continue
        if marker in (0xD8, 0xD9, 0x01) or 0xD0 <= marker <= 0xD7:
            index += 2
            continue
        if marker == 0xDA:
            break
        segment_length = int.from_bytes(data[index + 2 : index + 4], "big")
        if segment_length < 2 or index + 2 + segment_length > len(data):
            raise CorruptImageError("El archivo JPEG está truncado.")
        if marker in JPEG_SOF_MARKERS:
            if segment_length < 7 or index + 9 > len(data):
                raise CorruptImageError("El JPEG no declara dimensiones válidas.")
            height = int.from_bytes(data[index + 5 : index + 7], "big")
            width = int.from_bytes(data[index + 7 : index + 9], "big")
            if width <= 0 or height <= 0:
                raise CorruptImageError("El JPEG no declara dimensiones válidas.")
            return width, height
        index += 2 + segment_length
    raise CorruptImageError("No se pudieron leer las dimensiones del JPEG.")


def validate_radiograph(
    *,
    filename: str | None,
    content_type: str | None,
    data: bytes,
    max_bytes: int,
) -> ValidatedImage:
    if not data:
        raise CorruptImageError("El archivo está vacío.")
    if len(data) > max_bytes:
        if max_bytes >= 1024 * 1024:
            limit = f"{max_bytes // (1024 * 1024)} MB"
        else:
            limit = f"{max(1, max_bytes // 1024)} KB"
        raise ImageTooLargeError(f"La imagen supera el límite de {limit}.")
    if _is_dicom(data):
        raise UnsupportedImageTypeError("No se admite DICOM. Cargue una imagen PNG o JPG.")

    extension = Path(filename or "").suffix.lower()
    declared_type = (content_type or "").split(";")[0].strip().lower()
    normalized_type = CONTENT_TYPE_ALIASES.get(declared_type)
    expected_type = ALLOWED_EXTENSIONS.get(extension)
    if expected_type is None:
        raise UnsupportedImageTypeError("Solo se admiten archivos PNG o JPG.")
    if normalized_type is None:
        raise UnsupportedImageTypeError("El tipo declarado no es una imagen PNG o JPG.")
    if normalized_type != expected_type:
        raise UnsupportedImageTypeError("El tipo declarado no coincide con la extensión del archivo.")

    if data.startswith(PNG_SIGNATURE):
        if expected_type != "image/png":
            raise UnsupportedImageTypeError("La firma del archivo no coincide con su extensión.")
        width, height = _png_dimensions(data)
    elif data.startswith(b"\xff\xd8"):
        if expected_type != "image/jpeg":
            raise UnsupportedImageTypeError("La firma del archivo no coincide con su extensión.")
        width, height = _jpeg_dimensions(data)
    else:
        raise UnsupportedImageTypeError("El contenido no corresponde a una imagen PNG o JPG.")

    return ValidatedImage(
        extension="png" if expected_type == "image/png" else "jpg",
        content_type=expected_type,
        width=width,
        height=height,
        size_bytes=len(data),
        sha256=hashlib.sha256(data).hexdigest(),
    )


@dataclass(frozen=True)
class RadiographQualityResult:
    passed: bool
    failures: list[str]
    warnings: list[str]
    metrics: dict[str, int | float]


def evaluate_radiograph_quality(
    image_path: Path,
    *,
    min_dimension: int = 224,
    recommended_dimension: int = 512,
) -> RadiographQualityResult:
    """Evalúa decodificación, resolución mínima y legibilidad básica (HU07)."""
    import numpy as np
    from PIL import Image, UnidentifiedImageError

    failures: list[str] = []
    warnings: list[str] = []

    try:
        with Image.open(image_path) as image:
            image.load()
            width, height = image.size
            gray = np.asarray(image.convert("L"), dtype=np.float32)
    except (OSError, UnidentifiedImageError, ValueError):
        return RadiographQualityResult(
            passed=False,
            failures=["No se pudo decodificar el contenido de la radiografía. Cargue otro PNG o JPG válido."],
            warnings=[],
            metrics={
                "width": 0,
                "height": 0,
                "min_dimension": 0,
                "aspect_ratio": 0.0,
                "intensity_std": 0.0,
                "dynamic_range_p99_p01": 0.0,
                "clip_low_ratio": 0.0,
                "clip_high_ratio": 0.0,
            },
        )

    min_dim = min(width, height)
    aspect_ratio = float(max(width, height) / max(1, min_dim))
    intensity_std = float(gray.std())
    p01, p99 = np.percentile(gray, [1.0, 99.0])
    dynamic_range = float(p99 - p01)
    clip_low = float((gray <= 5.0).mean())
    clip_high = float((gray >= 250.0).mean())

    if min_dim < min_dimension:
        failures.append(
            f"Resolución insuficiente ({width}×{height} px). El mínimo requerido es {min_dimension}×{min_dimension} px."
        )
    if intensity_std < 2.0:
        failures.append("La imagen carece de variación tonal suficiente (imagen plana o en blanco).")

    if min_dim < recommended_dimension:
        warnings.append(
            f"Resolución menor a {recommended_dimension}×{recommended_dimension} px; se aplicará letterbox a {recommended_dimension} px."
        )
    if aspect_ratio > 2.5:
        warnings.append(
            f"Relación de aspecto inusual ({aspect_ratio:.2f}:1); verifique que el campo pulmonar esté completo."
        )
    if dynamic_range < 25.0 and intensity_std >= 2.0:
        warnings.append("Bajo rango dinámico de contraste; revise la legibilidad visual.")
    if clip_low > 0.85 or clip_high > 0.85:
        warnings.append("Alta proporción de píxeles saturados en negro o blanco.")

    return RadiographQualityResult(
        passed=not failures,
        failures=failures,
        warnings=warnings,
        metrics={
            "width": int(width),
            "height": int(height),
            "min_dimension": int(min_dim),
            "aspect_ratio": round(aspect_ratio, 4),
            "intensity_std": round(intensity_std, 4),
            "dynamic_range_p99_p01": round(dynamic_range, 4),
            "clip_low_ratio": round(clip_low, 4),
            "clip_high_ratio": round(clip_high, 4),
        },
    )
