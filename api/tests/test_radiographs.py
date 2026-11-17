from __future__ import annotations

import asyncio
import base64
from dataclasses import replace
from uuid import uuid4

import pytest
from fastapi import HTTPException
from fastapi.responses import FileResponse

from app.database import connect_database
from app.dependencies import require_permission
from app.evaluation_routes import (
    create_evaluation_route,
    download_radiograph_route,
    upload_radiograph_route,
)
from app.schemas import CreateEvaluationRequest

from .conftest import RequestStub


PNG_BYTES = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAFgwJ/"
    "lVn5WQAAAABJRU5ErkJggg=="
)
JPEG_BYTES = base64.b64decode(
    "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAgGBgcGBQgHBwcJCQgKDBQNDAsLDBkSEw8UHRofHh0a"
    "HBwgJC4nICIsIxwcKDcpLDAxNDQ0Hyc5PTgyPC4zNDL/wAALCAABAAEBAREA/8QAFAABAAAAAAAA"
    "AAAAAAAAAAAACf/EABQQAQAAAAAAAAAAAAAAAAAAAAD/2gAIAQEAAD8AKp//2Q=="
)


class UploadFileStub:
    def __init__(self, filename: str, content_type: str, data: bytes) -> None:
        self.filename = filename
        self.content_type = content_type
        self._data = data

    async def read(self, size: int = -1) -> bytes:
        if size is None or size < 0:
            return self._data
        return self._data[:size]


def create_evaluation_for(request_stub: RequestStub, technologist) -> dict[str, object]:
    evaluation = create_evaluation_route(
        CreateEvaluationRequest(),
        request_stub,
        technologist,
    )
    return evaluation.model_dump()


def upload(
    request_stub: RequestStub,
    user,
    evaluation_id: str,
    *,
    filename: str,
    content_type: str,
    data: bytes,
):
    return asyncio.run(
        upload_radiograph_route(
            evaluation_id,
            request_stub,
            UploadFileStub(filename, content_type, data),
            user,
        )
    )


def test_png_upload_stores_metadata_without_original_filename(
    request_stub, technologist, settings
):
    evaluation = create_evaluation_for(request_stub, technologist)
    radiograph = upload(
        request_stub,
        technologist,
        str(evaluation["id"]),
        filename="PACIENTE_JUAN.png",
        content_type="image/png",
        data=PNG_BYTES,
    )

    assert radiograph.content_type == "image/png"
    assert radiograph.width == 1
    assert radiograph.height == 1
    assert radiograph.filename.endswith(".png")
    assert radiograph.content_url == f"/api/v1/evaluations/{evaluation['id']}/radiograph"

    connection = connect_database(settings.database_path)
    try:
        row = connection.execute(
            "SELECT * FROM radiographs WHERE evaluation_id = ?",
            (str(evaluation["id"]),),
        ).fetchone()
        status = connection.execute(
            "SELECT status FROM evaluations WHERE id = ?",
            (str(evaluation["id"]),),
        ).fetchone()["status"]
        audit = connection.execute(
            """
            SELECT metadata_json FROM audit_events
            WHERE event_type = 'radiograph.uploaded'
            ORDER BY occurred_at DESC
            """
        ).fetchone()
    finally:
        connection.close()

    assert row is not None
    assert row["size_bytes"] == len(PNG_BYTES)
    assert row["sha256"] == radiograph.sha256
    assert "PACIENTE" not in row["storage_path"]
    assert "PACIENTE" not in audit["metadata_json"]
    assert status == "imagen_cargada"
    stored_path = settings.storage_root / row["storage_path"]
    assert stored_path.is_file()


def test_jpeg_upload_is_accepted(request_stub, technologist):
    evaluation = create_evaluation_for(request_stub, technologist)
    radiograph = upload(
        request_stub,
        technologist,
        str(evaluation["id"]),
        filename="radiografia.jpg",
        content_type="image/jpeg",
        data=JPEG_BYTES,
    )
    assert radiograph.content_type == "image/jpeg"
    assert radiograph.filename.endswith(".jpg")


def test_upload_rejects_text_mismatched_and_dicom_files(request_stub, technologist):
    evaluation = create_evaluation_for(request_stub, technologist)
    evaluation_id = str(evaluation["id"])

    with pytest.raises(HTTPException) as text_denied:
        upload(
            request_stub,
            technologist,
            evaluation_id,
            filename="nota.png",
            content_type="text/plain",
            data=b"no es una imagen",
        )
    assert text_denied.value.status_code == 415

    with pytest.raises(HTTPException) as mismatch_denied:
        upload(
            request_stub,
            technologist,
            evaluation_id,
            filename="radiografia.jpg",
            content_type="image/jpeg",
            data=PNG_BYTES,
        )
    assert mismatch_denied.value.status_code == 415

    dicom_bytes = b"\x00" * 128 + b"DICM" + b"\x00" * 32
    with pytest.raises(HTTPException) as dicom_denied:
        upload(
            request_stub,
            technologist,
            evaluation_id,
            filename="estudio.dcm",
            content_type="application/dicom",
            data=dicom_bytes,
        )
    assert dicom_denied.value.status_code == 415


def test_upload_rejects_truncated_and_oversized_images(
    request_stub, technologist, settings
):
    evaluation = create_evaluation_for(request_stub, technologist)
    evaluation_id = str(evaluation["id"])

    with pytest.raises(HTTPException) as truncated_denied:
        upload(
            request_stub,
            technologist,
            evaluation_id,
            filename="truncada.png",
            content_type="image/png",
            data=PNG_BYTES[:10],
        )
    assert truncated_denied.value.status_code == 422

    small_settings = replace(settings, max_upload_bytes=100)
    small_request = RequestStub(small_settings)
    with pytest.raises(HTTPException) as oversized_denied:
        upload(
            small_request,
            technologist,
            evaluation_id,
            filename="grande.png",
            content_type="image/png",
            data=PNG_BYTES + b"x" * 100,
        )
    assert oversized_denied.value.status_code == 413
    assert "KB" in oversized_denied.value.detail


def test_second_upload_is_rejected_without_duplicate_rows(
    request_stub, technologist, settings
):
    evaluation = create_evaluation_for(request_stub, technologist)
    evaluation_id = str(evaluation["id"])
    upload(
        request_stub,
        technologist,
        evaluation_id,
        filename="primera.png",
        content_type="image/png",
        data=PNG_BYTES,
    )

    with pytest.raises(HTTPException) as duplicate:
        upload(
            request_stub,
            technologist,
            evaluation_id,
            filename="segunda.png",
            content_type="image/png",
            data=PNG_BYTES,
        )
    assert duplicate.value.status_code == 409

    connection = connect_database(settings.database_path)
    try:
        total = connection.execute(
            "SELECT COUNT(*) AS total FROM radiographs WHERE evaluation_id = ?",
            (evaluation_id,),
        ).fetchone()["total"]
    finally:
        connection.close()
    assert total == 1


def test_upload_requires_an_existing_evaluation_and_permission(request_stub, technologist):
    with pytest.raises(HTTPException) as missing:
        upload(
            request_stub,
            technologist,
            str(uuid4()),
            filename="radiografia.png",
            content_type="image/png",
            data=PNG_BYTES,
        )
    assert missing.value.status_code == 404

    with pytest.raises(HTTPException) as denied:
        require_permission("radiographs.upload")(
            {"id": "admin", "role": "administrador"}
        )
    assert denied.value.status_code == 403


def test_download_is_authenticated_and_returns_the_stored_file(
    request_stub, technologist, radiologist
):
    evaluation = create_evaluation_for(request_stub, technologist)
    radiograph = upload(
        request_stub,
        technologist,
        str(evaluation["id"]),
        filename="radiografia.png",
        content_type="image/png",
        data=PNG_BYTES,
    )

    response = download_radiograph_route(
        str(evaluation["id"]),
        request_stub,
        radiologist,
    )
    assert isinstance(response, FileResponse)
    assert response.media_type == "image/png"
    assert response.headers["cache-control"] == "private, no-store"
    assert response.headers["x-content-type-options"] == "nosniff"
    assert radiograph.filename in response.headers["content-disposition"]

    with pytest.raises(HTTPException) as denied:
        require_permission("radiographs.read")(
            {"id": "admin", "role": "administrador"}
        )
    assert denied.value.status_code == 403


def test_download_without_radiograph_returns_404(request_stub, radiologist):
    evaluation = create_evaluation_for(request_stub, radiologist)
    with pytest.raises(HTTPException) as missing:
        download_radiograph_route(
            str(evaluation["id"]),
            request_stub,
            radiologist,
        )
    assert missing.value.status_code == 404
