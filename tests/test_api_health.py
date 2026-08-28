"""Health endpoint."""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.ai.faiss.index import FaceIndex
from app.core.face_processing.gallery import Gallery


def test_health_reports_degraded_until_anyone_is_enrolled(client: TestClient) -> None:
    """A running camera with an empty gallery recognises nobody -- that is not 'ok'."""
    body = client.get("/api/v1/health").json()
    assert body["status"] == "degraded"
    assert body["camera"] == "stopped"
    assert body["service"] == "face-recognition-service"
    assert body["index_size"] == 0


def test_health_is_ok_once_the_gallery_is_built(client: TestClient, container) -> None:
    container.gallery_handle.swap(
        Gallery(index=FaceIndex(dim=512), recognizer=None, employees=3, photos=3)
    )
    body = client.get("/api/v1/health").json()
    assert body["status"] == "ok"


def test_health_no_longer_pings_a_local_database(client: TestClient) -> None:
    """There is no local database to be up or down any more."""
    assert client.get("/api/v1/health").json()["database"] == "n/a"
