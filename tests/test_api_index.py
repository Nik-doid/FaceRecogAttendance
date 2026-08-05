"""Face index control endpoint tests."""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.database.session import sync_session
from app.repositories.audit_repo import AuditLogRepository


def test_index_status_idle(client: TestClient) -> None:
    resp = client.get("/api/v1/index/status")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "idle"
    assert body["size"] == 0


def test_rebuild_requires_auth(client: TestClient) -> None:
    resp = client.post("/api/v1/index/rebuild")
    assert resp.status_code == 401


def test_rebuild_starts_and_audits(
    client: TestClient, container, auth_headers
) -> None:
    resp = client.post("/api/v1/index/rebuild", headers=auth_headers)
    assert resp.status_code == 200
    assert "background" in resp.json()["message"]

    with sync_session() as session:
        audit = AuditLogRepository().list_recent(session)
    assert any(entry.action == "index.rebuild" for entry in audit)


def test_rebuild_conflict_when_already_building(
    client: TestClient, container, auth_headers
) -> None:
    container.index_service._building = True  # noqa: SLF001
    resp = client.post("/api/v1/index/rebuild", headers=auth_headers)
    assert resp.status_code == 409
