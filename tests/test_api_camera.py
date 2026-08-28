"""Camera control endpoint tests."""

from __future__ import annotations

from fastapi.testclient import TestClient

from tests.conftest import use_fake_camera


def test_camera_status_stopped(client: TestClient) -> None:
    resp = client.get("/api/v1/camera/status")
    assert resp.status_code == 200
    assert resp.json()["camera_id"] == "cam-test"
    assert resp.json()["status"] == "stopped"


def test_camera_controls_require_auth(client: TestClient) -> None:
    assert client.post("/api/v1/camera/start").status_code == 401
    assert client.post("/api/v1/camera/stop").status_code == 401


def test_camera_start_stop_flow(client: TestClient, container, auth_headers) -> None:
    use_fake_camera(container)

    start = client.post("/api/v1/camera/start", headers=auth_headers)
    assert start.status_code == 200
    assert start.json()["status"] == "running"

    status = client.get("/api/v1/camera/status")
    assert status.json()["status"] == "running"

    # A second start must be refused while already running.
    conflict = client.post("/api/v1/camera/start", headers=auth_headers)
    assert conflict.status_code == 409

    stop = client.post("/api/v1/camera/stop", headers=auth_headers)
    assert stop.status_code == 200
    assert stop.json()["status"] == "stopped"


def test_camera_controls_start_and_stop_the_runner(
    client: TestClient, container, auth_headers
) -> None:
    """Start and stop must be reachable, so the runner can be tested and halted."""
    started = client.post("/api/v1/camera/start", headers=auth_headers)
    assert started.status_code == 200
    assert started.json()["status"] == "running"
    assert container.camera_runner.running

    # A second start is a conflict, not a silent second thread.
    assert client.post("/api/v1/camera/start", headers=auth_headers).status_code == 409

    stopped = client.post("/api/v1/camera/stop", headers=auth_headers)
    assert stopped.status_code == 200
    assert stopped.json()["status"] == "stopped"
    assert container.camera_runner.running is False


def test_camera_controls_require_a_token(client: TestClient) -> None:
    """Stopping attendance capture is not a read-only operation."""
    assert client.post("/api/v1/camera/start").status_code == 401
    assert client.post("/api/v1/camera/stop").status_code == 401
