"""Recognition log / unknown face read endpoint tests."""

from __future__ import annotations

from datetime import UTC, datetime

from fastapi.testclient import TestClient

from app.database.session import sync_session
from app.repositories.recognition_log_repo import RecognitionLogRepository
from app.repositories.unknown_face_repo import UnknownFaceRepository
from tests.conftest import use_fake_camera

NOW = datetime.now(UTC)


def test_recognition_logs_returns_rows(client: TestClient) -> None:
    with sync_session() as session:
        RecognitionLogRepository().add_entry(
            session,
            employee_code="EMP1",
            camera_id="cam-test",
            timestamp=NOW,
            confidence=0.91,
            reported=True,
            attendance_response="published",
        )

    resp = client.get("/api/v1/recognition/logs")
    assert resp.status_code == 200
    body = resp.json()
    assert body["total"] == 1
    row = body["items"][0]
    assert row["employee_code"] == "EMP1"
    assert row["reported"] is True


def test_recognition_logs_filter_by_employee(client: TestClient) -> None:
    with sync_session() as session:
        repo = RecognitionLogRepository()
        repo.add_entry(
            session,
            employee_code="EMP1",
            camera_id="cam-test",
            timestamp=NOW,
            confidence=0.9,
            reported=False,
            attendance_response="duplicate_suppressed",
        )
        repo.add_entry(
            session,
            employee_code="EMP2",
            camera_id="cam-test",
            timestamp=NOW,
            confidence=0.9,
            reported=False,
        )

    resp = client.get("/api/v1/recognition/logs", params={"employee_code": "EMP2"})
    body = resp.json()
    assert body["total"] == 1
    assert body["items"][0]["employee_code"] == "EMP2"


def test_unknown_faces_returns_rows(client: TestClient) -> None:
    with sync_session() as session:
        UnknownFaceRepository().add_entry(
            session,
            camera_id="cam-test",
            timestamp=NOW,
            snapshot_path="/s/unknown.jpg",
            confidence_of_best_nonmatch=0.42,
        )

    resp = client.get("/api/v1/unknown-faces")
    assert resp.status_code == 200
    body = resp.json()
    assert body["total"] == 1
    assert body["items"][0]["snapshot_path"] == "/s/unknown.jpg"


def test_rate_limiting_on_control_endpoint(
    client: TestClient, container, auth_headers, monkeypatch
) -> None:
    import app.core.security as security_module
    from app.core.security import RateLimiter

    use_fake_camera(container)
    monkeypatch.setattr(security_module, "_rate_limiter", RateLimiter(2, 60))

    assert (
        client.post("/api/v1/camera/start", headers=auth_headers).status_code == 200
    )
    # Second request is processed (even though the handler may 409 for "already running").
    assert (
        client.post("/api/v1/camera/start", headers=auth_headers).status_code == 409
    )
    # Third request exceeds the limit.
    assert (
        client.post("/api/v1/camera/start", headers=auth_headers).status_code == 429
    )
