"""Shared fixtures.

No database and no model loading by default: the Container builds models lazily, so a
test that never touches recognition never pays for it. The camera runner and the
attendance consumer both stay stopped unless a test starts them.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from app.api.deps import set_container
from app.config.settings import Settings
from app.config.settings import settings as env_settings
from app.container import Container
from app.core.security import create_access_token
from app.main import create_app
from app.runtime import Models, load_models


@pytest.fixture(scope="session")
def models() -> Models:
    """One SCRFD + one ArcFace for the whole test run; loading them costs seconds."""
    return load_models(env_settings)


@pytest.fixture
def test_settings(tmp_path: object) -> Settings:
    return Settings(
        _env_file=None,
        app_env="development",
        log_level="DEBUG",
        camera_id="cam-test",
        camera_autostart=False,
        attendance_broker="null",
        attendance_consumer_enabled=False,
        employee_photos_source=[str(tmp_path)],
        storage_path=tmp_path,  # type: ignore[arg-type]
        jwt_secret_key="test-secret-that-is-long-enough-1234567890",
    )


@pytest.fixture
def container(test_settings: Settings) -> Container:
    return Container(test_settings)


@pytest.fixture
def client(container: Container) -> Iterator[TestClient]:
    # The container is injected, so the lifespan skips enrolling photos and opening
    # a camera behind the test.
    with TestClient(create_app(container)) as test_client:
        yield test_client
    set_container(None)  # type: ignore[arg-type]


@pytest.fixture
def auth_headers() -> dict[str, str]:
    """create_access_token signs with the process-wide settings, not the fixture."""
    return {"Authorization": f"Bearer {create_access_token('tester')}"}
