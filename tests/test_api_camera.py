"""Camera runner control endpoints.

These are the surface for starting, stopping and watching the always-on runner, so
they are also how the whole flow gets exercised by hand.
"""

from __future__ import annotations

from fastapi.testclient import TestClient


def test_status_is_readable_without_a_token(client: TestClient) -> None:
    """Read-only previews of a running camera need no credentials."""
    body = client.get("/api/v1/camera/status").json()
    assert body["camera_id"] == "cam-test"
    assert body["status"] == "stopped"
    assert body["gallery"]["ready"] is False


def test_controls_require_a_token(client: TestClient) -> None:
    """Silently stopping attendance capture is not a read-only operation."""
    assert client.post("/api/v1/camera/start").status_code == 401
    assert client.post("/api/v1/camera/stop").status_code == 401


def test_start_stop_flow(client: TestClient, container, auth_headers) -> None:
    start = client.post("/api/v1/camera/start", headers=auth_headers)
    assert start.status_code == 200
    assert start.json()["status"] == "running"
    assert client.get("/api/v1/camera/status").json()["status"] == "running"
    assert container.camera_runner.running

    # A second start must be refused rather than quietly spawning a second thread.
    assert client.post("/api/v1/camera/start", headers=auth_headers).status_code == 409

    stop = client.post("/api/v1/camera/stop", headers=auth_headers)
    assert stop.status_code == 200
    assert stop.json()["status"] == "stopped"
    assert container.camera_runner.running is False


def test_frame_is_unavailable_until_the_camera_produces_one(client: TestClient) -> None:
    assert client.get("/api/v1/camera/frame").status_code == 503


def test_frame_serves_whatever_the_runner_published(client: TestClient, container) -> None:
    container.frame_hub.publish_frame(b"\xff\xd8jpeg-bytes")
    response = client.get("/api/v1/camera/frame")
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"
    assert response.content == b"\xff\xd8jpeg-bytes"
